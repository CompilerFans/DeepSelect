"""The MACA backend, under the name the interface imports.

The official wheel's backends are importable extension modules
(`deep_select_cuda`, `deep_select_npu`) and `interface._import_backend()` picks
one by platform.  This tree has one platform, so the module at that name is
this one -- and the artifact behind it is a tvm-ffi module, not an importable
extension, so this module is the importable surface and `_binding` resolves
the per-family `.so` behind it.

`deep_select_cuda` re-exports this module: a framework that asks for the
official CUDA backend by name finds the MACA backend here, with no second
implementation to keep in step.
"""
from . import _binding

_KERNEL_NAME = "deep_select_maca"


def get_alignment_requirement():
    """The kernel's stride/alignment requirement, in bytes (two ints)."""
    return _binding.load(_KERNEL_NAME).get_alignment_requirement()


def topk(*args):
    """Launch the kernel with the argument tuple `interface.topk` builds.

    The launch runs inside `_binding.launching()`: that pins the FFI env
    stream to torch's current stream, which the C++ side reads through
    `TVMFFIEnvGetStream`.
    """
    with _binding.launching():
        _binding.load(_KERNEL_NAME).topk(*args)
