# 2026 - Modified for DeepSelect.  The tvm-ffi binding loader, modelled on the
# host repository's `deep_gemm/maca_binding.py`.
"""Load the torch-free kernel extension through tvm-ffi.

The artifact (`deep_select_maca*.so`) is built at the Apache TVM FFI ABI,
export `__tvm_ffi_*` symbols, have **no `PyInit`** and are not importable python
modules -- they are loaded with `tvm_ffi.load_module`.  Deliberate: the pybind11
artifacts this replaces linked libtorch/libc10, which tied the extension to the
host's torch build (the `c10_cuda_check_implementation` trap) and to a cpython
tag.

A caller still needs **torch at the call site** (a `torch.Tensor` is what has
the `__dlpack__` protocol; this module imports torch only in `launching()`) and
**tvm_ffi to load the artifact at all**.

The stream handoff is what `launching()` is for: the C++ side reads its stream
through `TVMFFIEnvGetStream`, which reports the null handle -- the legacy
default stream, which does not synchronize with torch's non-blocking side
streams -- unless something installed torch's current stream.
"""

from __future__ import annotations

import functools
import glob
import os
from typing import Any

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))

# One artifact, carrying every architecture the build was asked for.  The
# loader resolves a backend by that name, which is what keeps the FFI change
# invisible from `backend="maca_c"`.
_LIBRARY_GLOB = "{name}*.so"


class _Module:
    """A loaded extension with the two entries this tree exports.

    `get_function` *raises* rather than returning None when the name is absent,
    so the probe goes through `implements_function` first -- a module loaded
    from a stale or foreign `.so` should say so, not leak an `AttributeError`.
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


def _preload_tvm_ffi() -> None:
    """Load `libtvm_ffi.so` from the installed package before the extension.

    The extension carries `DT_NEEDED: libtvm_ffi.so`, and that library ships
    inside the `tvm_ffi` *python package* -- a path only the running process
    knows.  Letting the loader find it through `DT_RPATH` would bake the build
    machine's `site-packages` into the artifact, so it is loaded here instead:
    an object already in the global scope satisfies a later `DT_NEEDED` by
    soname, wherever it was found.  Measured: the shipped library's `SONAME` is
    exactly `libtvm_ffi.so`, the same string the extension records.

    Called once per process (`load` is cached), and only when a kernel is
    actually about to be loaded -- importing `deep_select` must not.
    """
    import ctypes

    import tvm_ffi

    root = os.path.dirname(os.path.abspath(tvm_ffi.__file__))
    for sub in ("lib", os.path.join("lib64")):
        hits = sorted(glob.glob(os.path.join(root, sub, "libtvm_ffi.so*")))
        if hits:
            # RTLD_GLOBAL: a private (RTLD_LOCAL) load does not enter the
            # global scope and would not satisfy the extension's DT_NEEDED.
            ctypes.CDLL(hits[0], mode=ctypes.RTLD_GLOBAL)
            return
    raise RuntimeError(
        f"tvm_ffi at {root} has no lib/ or lib64/ with libtvm_ffi.so in it; "
        f"the extension links it and cannot be loaded without it"
    )


@functools.lru_cache(maxsize=None)
def load(name: str) -> Any:
    """The loaded tvm-ffi module named ``name``.

    Cached: the module holds a device binary, and a process that never calls
    `topk` should not load one.
    """
    import tvm_ffi

    _preload_tvm_ffi()
    hits = sorted(glob.glob(os.path.join(_PACKAGE_DIR, _LIBRARY_GLOB.format(name=name))))
    if not hits:
        raise RuntimeError(
            f"{name} has not been built; build it with ./build.sh (setup.py "
            f"produces one extension, deep_select_maca, carrying an image for "
            f"each architecture in CUCC_TARGETS)"
        )
    # Newest build wins on stale-cache ties -- the host repository's rule.
    return _Module(tvm_ffi.load_module(hits[-1]))


def launching():
    """Context manager pinning the FFI env stream to torch's current stream.

    Every kernel-launching call must run inside this: without it
    `TVMFFIEnvGetStream` reports the null handle, the launch lands on the
    legacy default stream, and a producer on `torch.cuda.Stream()` followed by
    an unwrapped call races.  Entering costs a thread-local write.
    """
    import tvm_ffi

    return tvm_ffi.use_torch_stream()
