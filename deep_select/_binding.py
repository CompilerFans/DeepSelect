# 2026 - Modified for DeepSelect.  The tvm-ffi binding loader, modelled on the
# host repository's `deep_gemm/maca_binding.py` (mcDeepGemm, branch
# `dev_tvm_ffi`, commit 3a6e6ba3).
"""Load the torch-free kernel extensions through tvm-ffi.

The artifacts (`deep_select_xcore<N>*.so`) are built at the Apache TVM FFI ABI
and export `__tvm_ffi_*` symbols.  They have **no `PyInit`** and are not
importable python modules: they are loaded with `tvm_ffi.load_module`, and that
is deliberate.  The pybind11 artifacts this replaces linked libtorch/libc10 --
six DT_NEEDED entries -- which tied the extension to the host's torch build
(the `c10_cuda_check_implementation` trap) and to the cpython version its
extension suffix was stamped with.  Neither is true of these.

What a caller still needs is stated rather than implied:

* **torch, at the call site.**  Tensors cross as DLPack, so the extension does
  not link or import torch -- but a `torch.Tensor` is what has the `__dlpack__`
  protocol, and output buffers are allocated with `torch.empty`.  This module
  does not import torch itself; only `launching()` does, and only for the
  stream handoff below.
* **tvm_ffi, to load at all.**  One extra run-time dependency, in exchange for
  the two above.

The stream handoff is the subtle part and the reason `launching()` exists: the
C++ side reads its stream through `TVMFFIEnvGetStream`, which reports the null
handle unless something has installed torch's current stream.  Launching
without it lands every kernel on the legacy default stream, which does not
synchronize with torch's non-blocking side streams.  `tvm_ffi.use_torch_stream()`
is what installs it.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Any

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# One artifact per architecture, named for the architecture it serves, exactly
# as before the migration.  The loader resolves a backend by that name, so the
# FFI change is invisible from `backend="maca_c"`.
_LIBRARY_GLOB = "deep_select_{name}*.so"


class _Module:
    """A loaded extension with the two entries this tree exports.

    Functions are resolved through `get_function`, which *raises* rather than
    returning None when the name is absent (this build takes only
    `query_imports`), so the probe goes through `implements_function` first --
    a module loaded from a stale or foreign `.so` should say so, not leak an
    `AttributeError` out of the loader.
    """

    def __init__(self, handle: Any):
        self._handle = handle
        missing = [n for n in ("topk", "get_alignment_requirement")
                   if not handle.implements_function(n)]
        if missing:
            raise RuntimeError(
                f"the loaded extension does not export {', '.join(missing)}; "
                f"it is not a DeepSelect kernel artifact (a stale or foreign "
                f".so in deep_select/?)"
            )
        self._topk = handle.get_function("topk")
        self._alignment = handle.get_function("get_alignment_requirement")

    def topk(self, *args):
        return self._topk(*args)

    def get_alignment_requirement(self):
        return self._alignment()


@functools.lru_cache(maxsize=None)
def load(name: str) -> Any:
    """The loaded tvm-ffi module for architecture extension ``name``.

    Cached: the module holds a device binary, and a process that never calls
    `topk` on a kernel should not load one.
    """
    import tvm_ffi

    hits = sorted(glob.glob(os.path.join(_PACKAGE_DIR, _LIBRARY_GLOB.format(name=name))))
    if not hits:
        raise RuntimeError(
            f"backend {name!r} has not been built; build it with "
            f"CUCC_TARGETS={name} (setup.py builds one extension per "
            f"architecture, and only for the architectures it is asked for)"
        )
    # Newest build wins on stale-cache ties -- the same rule the host
    # repository's loader uses.
    return _Module(tvm_ffi.load_module(hits[-1]))


def launching():
    """Context manager pinning the FFI env stream to torch's current stream.

    Every kernel-launching call must run inside this.  Without it
    `TVMFFIEnvGetStream` reports the null handle and the launch lands on the
    legacy default stream, which does not synchronize with torch's non-blocking
    side streams -- a producer on `torch.cuda.Stream()` followed by an
    unwrapped call would race.  Entering costs a thread-local write.
    """
    import tvm_ffi

    return tvm_ffi.use_torch_stream()
