"""Call logging: one header per process, then one record per public API call.

Off unless `DS_LOG` names a target:

    unset                   zero overhead -- `log_call` returns the function
    "1" / "on" / "stderr"   write to stderr
    "file" / "log"          ./deep_select_<date>_<time>.log
    a directory             <dir>/deep_select_<date>_<time>.log
    anything else           that path, used verbatim

Every public entry point in `interface.py` is decorated, so a call records the
shapes and dtypes it was handed and the ones it returned -- which is what makes
a shape mismatch on the caller's side readable after the fact, rather than only
as a contract rejection.  Records are metadata only: a tensor is reported as
shape/dtype/device and its values are never read, so turning this on cannot
change what a call computes.

A call into one arm (`topk` -> `topk_torch`) is one record, not two: the
innermost public entry is the one whose arguments the caller wrote, and the
outer one already reports the `backend` it was asked for.

**Enabling this synchronizes the device around every call**, which is the only
way the elapsed time means anything; a `DS_LOG` run is therefore not a
benchmark run.  (`get_stride_requirement` is not decorated: it takes no
tensors and is `lru_cache`d, so a call record would report cache hits.)
"""

from __future__ import annotations

import functools
import inspect
import os
import subprocess
import sys
import threading
import time
from datetime import datetime
from typing import Optional

import torch

_ENV = "DS_LOG"

# Reported in the header when set: the knobs that shape what a run does.
_KEY_ENV = ("DS_TOPK_BACKEND", "DS_RESULTS_DIR", "CUDA_VISIBLE_DEVICES",
            "MACA_PATH", "MACA_HOME")

# The outermost public call owns the record; see the module docstring.
_LOCAL = threading.local()


def _default_name() -> str:
    return f"deep_select_{datetime.now():%Y%m%d_%H%M%S}.log"


def _target() -> Optional[str]:
    """Resolve `DS_LOG` to `"stderr"`, a file path, or None when it is unset."""
    raw = os.environ.get(_ENV)
    if not raw:
        return None
    low = raw.lower()
    if low in ("1", "on", "true", "stderr"):
        return "stderr"
    if low in ("file", "disk", "log"):
        return _default_name()
    if raw.endswith(("/", os.sep)) or os.path.isdir(raw):
        return os.path.join(raw, _default_name())
    return raw


# ── value formatting ────────────────────────────────────────────────────────

def _format_short(val) -> str:
    """A bounded, metadata-only rendering of one value.

    Tensors are shape/dtype/device and **never** their contents: reading a
    tensor to format it could synchronize a stream or fault on a device the
    caller pinned, and a shape dump does not need the data.
    """
    if isinstance(val, torch.Tensor):
        return f"Tensor{list(val.shape)} {val.dtype} ({val.device})"
    if isinstance(val, (tuple, list)):
        inner = ", ".join(_format_short(v) for v in val[:4])
        inner += ", ..." if len(val) > 4 else ""
        return f"({inner})" if isinstance(val, tuple) else f"[{inner}]"
    if isinstance(val, dict):
        return f"dict(len={len(val)})"
    if isinstance(val, str):
        return repr(val if len(val) <= 50 else val[:50] + "...")
    text = repr(val)
    return text if len(text) <= 80 else text[:80] + "..."


def _format_call(fn, args, kwargs, outcome: str) -> str:
    """The call as the caller wrote it, then what it produced.

    Only the arguments actually passed are shown -- a default is not restated
    on every line, and what it does to the result is already in the shapes.
    """
    try:
        bound = inspect.signature(fn).bind(*args, **kwargs).arguments
    except (TypeError, ValueError):
        # A call this signature cannot describe is still worth a record.
        bound = {}
    parts = [f"{name}={_format_short(value)}" for name, value in bound.items()]
    return f"{fn.__name__}({', '.join(parts)}) -> {outcome}"


# ── header (once per process) ───────────────────────────────────────────────

_HEADER_WRITTEN = False


def _device() -> str:
    try:
        from ._arch import native_family, native_sm_count
        name = torch.cuda.get_device_name() if torch.cuda.is_available() else "no device"
        return f"xcore{native_family()} ({name}, {native_sm_count()} SMs)"
    except Exception:
        return "unknown"


def _git() -> str:
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = subprocess.run(["git", "-C", root, "rev-parse", "--short", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _enabled_env() -> str:
    items = sorted(f"{k}={v}" for k, v in os.environ.items() if k.startswith("DS_"))
    items += [f"{k}={os.environ[k]}" for k in _KEY_ENV if os.environ.get(k)]
    return ", ".join(items) if items else "(none)"


def _write_header(fp) -> None:
    global _HEADER_WRITTEN
    if _HEADER_WRITTEN:
        return
    _HEADER_WRITTEN = True
    from .__version__ import __version__
    fp.write("# ============== deep_select run log ==============\n")
    fp.write(f"# version : {__version__} (git {_git()})\n")
    fp.write(f"# device  : {_device()}\n")
    fp.write(f"# env     : {_enabled_env()}\n")
    fp.write(f"# started : {datetime.now().isoformat(timespec='seconds')}\n")
    fp.write("# ================================================\n")


def _stamp() -> str:
    return datetime.now().strftime("%H:%M:%S.%f")[:-3]


def _write(target: str, line: str) -> None:
    if target == "stderr":
        _write_header(sys.stderr)
        print(line, file=sys.stderr)
        return
    try:
        with open(target, "a") as fp:
            _write_header(fp)
            fp.write(line + "\n")
    except OSError as exc:
        print(f"[ds_log] cannot write {target}: {exc}", file=sys.stderr)


# ── the two entry points ────────────────────────────────────────────────────

def log(msg: str, **fields) -> None:
    """Record a line from any code point, not only from a decorated call.

    For the decisions a call record cannot show -- which backend a bare call
    resolved to, say.  A no-op when `DS_LOG` is unset.
    """
    target = _target()
    if target is None:
        return
    line = f"[{_stamp()}] {msg}"
    for key, value in fields.items():
        line += f" {key}={_format_short(value)}"
    _write(target, line)


def _record(fn, args, kwargs, start: float, outcome: str) -> None:
    # Re-read rather than captured at decoration: a target that went away
    # mid-process silences the log instead of raising on every call.
    target = _target()
    if target is None:
        return
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _write(target, f"[{_stamp()}] {_format_call(fn, args, kwargs, outcome)}"
                   f" [{elapsed_ms:.3f} ms]")


def log_call(fn=None):
    """Log a public API call's arguments and result when `DS_LOG` is set.

    Zero overhead when it is not: the function is returned unwrapped, so a
    disabled log costs one `os.environ` lookup at import.
    """

    def decorate(func):
        if _target() is None:
            return func

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            depth = getattr(_LOCAL, "depth", 0)
            _LOCAL.depth = depth + 1
            outer = depth == 0
            synced = outer and torch.cuda.is_available()
            try:
                if synced:
                    torch.cuda.synchronize()
                start = time.perf_counter()
                try:
                    ret = func(*args, **kwargs)
                except Exception as exc:
                    # A call the contract rejected is exactly the one whose
                    # shapes are worth having, so it is recorded too.
                    if outer:
                        _record(func, args, kwargs, start,
                                f"raised {type(exc).__name__}: {_format_short(str(exc))}")
                    raise
                if synced:
                    torch.cuda.synchronize()
                if outer:
                    _record(func, args, kwargs, start, _format_short(ret))
                return ret
            finally:
                _LOCAL.depth = depth

        return wrapper

    return decorate(fn) if fn is not None else decorate
