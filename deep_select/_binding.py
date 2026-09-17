"""Load the torch-free kernel extension through tvm-ffi.

The artifact (`deep_select_maca_xcore<N>.so`, one per family) is built at the
Apache TVM FFI ABI and exports `__tvm_ffi_*` symbols with **no `PyInit`**, so it
is not an importable python module -- it is loaded with `tvm_ffi.load_module`.

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
from typing import Any, Optional

_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


def _library_pattern(name: str, suffix: str) -> str:
    """The glob for ``name``'s artifact on a device of family ``suffix``.

    One artifact **per family** -- `deep_select_maca_xcore1000.so` and so on.
    The name is `deep_select_maca` (the C++ namespace) plus the family the
    artifact was compiled for; `family_suffix()` supplies the suffix and says
    why the *device's* family, not the build's target list, is what names the
    file.  The trailing `*` tolerates a build suffix (a stale
    `.cpython-310-...` from before `no_python_abi_suffix`).

    An empty suffix means "this process could not tell what family the device
    is", and it must not be allowed to match a family's artifact: the naive
    `deep_select_maca*.so` would load the C500 image on a part whose family
    could not be read, which is precisely the guess the loader exists to
    refuse.  So the no-suffix case matches only an artifact that carries no
    family either -- the unsuffixed `deep_select_maca.so` an ad-hoc build
    (`/tmp/dsv3/mkvar.sh`) produces, which makes no per-family claim at all.
    """
    return f"{name}{suffix}*.so" if suffix else f"{name}.so"


class _Module:
    """A loaded extension with the two entries it exports.

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


def device_family_suffix() -> str:
    """`_xcore<N>` for **this** device's xcore family, or `""` when there is none.

    **The device names the artifact, not the build.**  Each artifact's *host*
    half holds its family's constants (`csrc/structs.h`'s `ARCH_SM_COUNT` and
    `ARCH_SMEM_PER_AP_BYTES`), so loading the wrong one is not a slower kernel,
    it is a kernel sized for another machine.  The build's `CUCC_TARGETS` cannot
    answer which one is in front of us -- that is the same defect
    `perf_snapshot.provenance()` had once -- and neither can a driver query for
    the family, because MACA reports the family as an sm spelling rather than a
    name.

    The sm spellings are `deep_gemm/utils/arch_config.py`'s, which is the
    repository's one table for this: `xcore1000` is sm80, `xcore1500` sm86,
    `xcore1600` sm87/88/89 (the C600U-class part renumbers across SDK
    generations, which is why the family and not the number is the anchor).
    Anything outside those -- H200, a CPU-only host -- gets `""`, which names
    the unsuffixed artifact and nothing else; see `_library_pattern` for why an
    unknown family must not fall through to *some* family's image.

    Not cached: this is the raw device query, and `family_suffix` below is the
    cached view of it.  The two are separate because a process whose current
    device changes has to be able to re-ask -- `family_suffix` memoizes the
    *answer*, and memoizing the *question* as well would make the cache a
    device pin that nothing asked for.
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return ""
        major, minor = torch.cuda.get_device_capability()
    except Exception:
        return ""
    sm = major * 10 + minor
    if sm == 80:
        return "_xcore1000"
    if sm == 86:
        return "_xcore1500"
    if 87 <= sm < 90:
        return "_xcore1600"
    return ""


@functools.lru_cache(maxsize=None)
def family_suffix() -> str:
    """The cached `device_family_suffix()` -- a device does not change family
    within a process, so the driver query happens once."""
    return device_family_suffix()


@functools.lru_cache(maxsize=None)
def extension_path(name: str) -> Optional[str]:
    """The file ``load(name)`` will load, or None if there is none.

    **Public because a measurement has to hash the artifact the process will
    actually load.**  `run_bench.sh` and `perf_snapshot.py` receipt this file's
    md5 as the "which binary did I measure" record; a receipt taken from a
    path guessed outside the loader is not a receipt.  It matters most for the
    *installed* package, where `_PACKAGE_DIR` is `site-packages/deep_select/`
    -- the deployed wheel carries its `.so` beside this file, so this answers
    for a package exactly as it does for a checkout.

    One rule, one place: `load()` calls this, so the two cannot drift into
    naming different files.  Newest wins, so a stale artifact cannot shadow a
    fresh one.
    """
    hits = sorted(glob.glob(os.path.join(
        _PACKAGE_DIR, _library_pattern(name, family_suffix()))))
    return hits[-1] if hits else None


@functools.lru_cache(maxsize=None)
def load(name: str) -> Any:
    """The loaded tvm-ffi module named ``name``.

    Cached: the module holds a device binary, and a process that never calls
    `topk` should not load one.

    **The `import tvm_ffi` above is load-bearing beyond `load_module`.**  The
    extension carries `DT_NEEDED: libtvm_ffi.so`, which ships inside the
    `tvm_ffi` package, and importing that package maps the library into the
    process -- which is how the loader satisfies the `DT_NEEDED` by soname.  An
    `-Wl,-rpath` is not an alternative: it would bake the *build* machine's
    `site-packages` into the artifact.
    """
    import tvm_ffi

    path = extension_path(name)
    if path is None:
        raise RuntimeError(
            f"{name}{family_suffix()} has not been built; build it with "
            f"./develop.sh (setup.py produces one extension per family in "
            f"CUCC_TARGETS, and this process's device needs the "
            f"{family_suffix() or 'default'} one)"
        )
    return _Module(tvm_ffi.load_module(path))


def launching():
    """Context manager pinning the FFI env stream to torch's current stream.

    Every kernel-launching call must run inside this: without it
    `TVMFFIEnvGetStream` reports the null handle, the launch lands on the
    legacy default stream, and a producer on `torch.cuda.Stream()` followed by
    an unwrapped call races.  Entering costs a thread-local write.
    """
    import tvm_ffi

    return tvm_ffi.use_torch_stream()
