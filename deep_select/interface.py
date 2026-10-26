import functools
import os
import sys
import torch

from typing import Optional, Tuple

from . import _binding
from ._backends import topk_torch
from ._log import log, log_call


# The backend names.  `maca_c` is the kernel this device has; `torch` is a
# reference implementation of the contract.
_BACKENDS = ("maca_c", "torch")


_UNKNOWN_BACKEND_REPORTED = False


def _report_unknown_backend(value: str) -> None:
    """Say once, on stderr, that a non-empty `DS_TOPK_BACKEND` named nothing.

    Reported rather than raised: one bad environment variable must not break
    every call in the process.  Reported at all because the default is the
    kernel now -- a caller who exported a misspelling of `torch` would
    otherwise be handed the fast path without a word, and that is the
    direction in which a defect hides.  Printed rather than warned so a
    caller's `-W error` cannot turn it into the exception this avoids.
    """
    global _UNKNOWN_BACKEND_REPORTED
    if _UNKNOWN_BACKEND_REPORTED:
        return
    _UNKNOWN_BACKEND_REPORTED = True
    print(f"deep_select: DS_TOPK_BACKEND={value!r} names no backend "
          f"({', '.join(_BACKENDS)}); using maca_c", file=sys.stderr)


def _default_backend() -> str:
    """`DS_TOPK_BACKEND` if it names a backend, else `"maca_c"`.

    The kernel is the default because the operator this tree ports has no
    backend selector at all: upstream's call sites (DeepInfer's indexer and
    sampler) call `topk(...)` and get that platform's kernel, so a caller that
    names nothing must get the kernel here too.  `"torch"` stays available for
    differential testing; it is asked for by name.

    The value is compared after `strip().lower()`, because an environment
    variable arrives from a shell and `" torch"` or `"TORCH"` is a spelling of
    the same name, not a request for something else.  A value that still names
    no backend is ignored -- never raised -- and reported once.
    """
    v = os.environ.get("DS_TOPK_BACKEND", "").strip().lower()
    if v in _BACKENDS:
        return v
    if v:
        _report_unknown_backend(v)
    return "maca_c"


# The extension the build produces.
_KERNEL_NAME = "deep_select_maca"


@functools.lru_cache(maxsize=1)
def _backend_for():
    """The loaded tvm-ffi extension implementing `maca_c`."""
    return _binding.load(_KERNEL_NAME)


# `structs.h`'s INPUT_/OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT, for when no kernel
# is built.
_ALIGNMENT_REQUIREMENT_BYTES = (1024, 32)


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    try:
        # The FFI entry returns a python list, not a C++ pair; normalize it.
        pair = _backend_for().get_alignment_requirement()
        return (int(pair[0]), int(pair[1]))
    except RuntimeError:
        return _ALIGNMENT_REQUIREMENT_BYTES


def get_alignment_requirement() -> Tuple[int, int]:
    """DeepInfer's spelling of :func:`get_stride_requirement`."""
    return get_stride_requirement()


@log_call
def topk(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    begin: Optional[torch.Tensor] = None,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    hint: Optional[torch.Tensor] = None,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
    backend: Optional[str] = None,
    check_nan: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Arguments:
        input: (b, vocab_size), dtype=torch.bfloat16/torch.float. stride(0) must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and stride(1) must be 1.
        topk: int. Select topk elements for each row.
        sorted: bool. Whether to return sorted **output_val**. fp32 on the
                upstream contract; the MACA kernel also supports bfloat16.
        begin(optional): (b,), dtype=int32. CURRENTLY NOT SUPPORTED. The left(inclusive) range for input row, default is 0.
        end(optional): (b,), dtype=int32. The right(exclusive) range for input row, default is vocab_size. The stride of this tensor must be 1.
                       Note when end[i] <= topk, valid elements will be gathered at the beginning of values and indices returned. The rest of `values` will be filled with `value_oob_fill_value`, while the rest of `indices` will be filled with `idx_oob_fill_value` (won't be plused by `output_idx_offset`).
                       `end` <= `vocab_size` must be held
        indices_type: torch.dtype. The output indices dtype, only support torch.int32 and torch.int64.
        sorted_index: bool. Whether to return sorted **output_idx**.
        hint(optional): CURRENTLY NOT SUPPORTED
        output_idx(optional): (b, topk), dtype=indices_type. A contiguous tensor to store output.
        output_idx_offset(optional): (b,), dtype=int32. If provided, all output_idx (`idx_oob_fill_value` not included) will += output_idx_offset.
        idx_oob_fill_value: int. See comments above when end[i]-begin[i]<topk.
        return_value: bool. If False, only return indices without values to accelerate the kernel. The return value is still a Tuple, but the first element will be None.
        abort_when_nan_found: bool. When a NaN is found, if True, aborts the whole kernel; if False, writes 0x3F3F3F3F to the corresponding output_idx[batch_idx][0] and exits.
                Exception: when the row's length <= topk, it is skipped. Inert when `check_nan` is False, since then nothing is ever found.
        backend: str. Which implementation to run: `"maca_c"` (the default),
                this device's MACA kernel, which is what the operator this tree
                ports runs and the fast one; or `"torch"`, a reference
                implementation built from torch ops that runs on any device and
                dtype, and is the one to ask for when checking the kernel.
                `DS_TOPK_BACKEND` overrides the default for a whole process.
        check_nan: bool. Whether each row is scanned for a NaN at all.  With
                False no scan runs, `abort_when_nan_found` never fires, and a
                row holding a NaN is ranked like any other value -- the
                caller is asserting its input is NaN-free, and the operator
                takes it at its word.  It is not a faster way to get
                `abort_when_nan_found=False`.  A row whose visible length is
                <= topk is never scanned either way: its whole window is the
                answer, so a NaN in it is ranked like any other value.
                On every production route the scan is folded into a pass that
                already reads the row, so turning it off saves only the fused
                predicate's marginal cost -- measured at ~3% of the fp32
                split's stage 1 (`docs/C500-radix-perf-ledger.zh.md` §10.7).
                The default is the contract: check, and answer as
                `abort_when_nan_found` says.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    # `None` means "take the process default" (see `_default_backend`), read
    # per call so a process that sets the variable late still gets it.
    if backend is None:
        backend = _default_backend()
        # The call record shows `backend=None`; this is the backend it became.
        log("backend resolved", backend=backend)
    # Checked before anything is read off `input`, so a mistyped backend name
    # is reported as such rather than as whatever a None tensor does next.
    if backend not in _BACKENDS:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Expected one of "
            f"{', '.join(_BACKENDS)} -- 'maca_c' runs the kernel this device "
            f"has, 'torch' the reference implementation"
        )

    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, device: torch.device, dtype: torch.dtype):
        """
        Return a tensor with shape (dim0, dim1), and stride (X, 1), where X is a multiple of 32B
        """
        output_stride_requirement_bytes = get_stride_requirement()[1]
        output_stride_requirement = output_stride_requirement_bytes // dtype.itemsize
        assert output_stride_requirement > 0
        dim1_rounded = (dim1+output_stride_requirement-1) // output_stride_requirement * output_stride_requirement
        out = torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)
        # The slice is a no-op view when the row is already aligned (the common
        # case: topk * itemsize is a 32B multiple for every k this operator
        # serves), and view creation is not free (~2-3 us on this torch).
        if dim1_rounded == dim1:
            return out
        return out[:, :dim1]
    
    output_val = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=input.dtype) if return_value else None
    if output_idx is None:
        output_idx = get_empty_and_aligned_tensor(N, topk, device=input.device, dtype=indices_type)
    else:
        assert output_idx.dtype == indices_type

    assert begin is None, "`begin` is not supported now"
    assert hint is None, "`hint` is not supported now"

    if backend == "torch":
        return topk_torch(
            input, topk, sorted=sorted, end=end, indices_type=indices_type,
            sorted_index=sorted_index, output_idx=output_idx,
            output_idx_offset=output_idx_offset,
            idx_oob_fill_value=idx_oob_fill_value,
            value_oob_fill_value=value_oob_fill_value,
            return_value=return_value,
            abort_when_nan_found=abort_when_nan_found,
            check_nan=check_nan,
        )
    else:
        # The 14 positional arguments the tvm-ffi entry takes, all DLTensor or
        # plain scalar.  `begin` and `hint` are rejected above, so neither
        # crosses the boundary, and `output_val` is None when `return_value` is
        # False (`Optional[TensorView>`).  `kernels=[...]` is not part of this
        # operator's contract, so nothing here reads it.
        #
        # The SM count is the odd one out: the only argument derived from the
        # *device* rather than the problem.  It stays an argument even now that
        # the artifact carries its family's constants as compile-time values
        # (`csrc/structs.h`), because it is the one number that has to describe
        # the machine in front of the call and not the machine the image was
        # built for -- the grid-sizing rules size a grid to fill *this* part,
        # and `_binding.family_suffix()` guarantees only that the image matches
        # its family, not that every call is on the device that picked it.
        # Read here rather than cached: `get_device_properties` is a struct
        # copy from the driver's cache, measured at 1.7 us against a kernel
        # this operator runs in tens to thousands of us.  The kernel refuses a
        # zero itself rather than defaulting it -- a zero sizes every grid to
        # nothing, which is an empty answer, not a crash.
        #
        # This tuple's order is the C++ entry's parameter list, not this
        # signature's: `check_nan` is appended last here as a keyword, and
        # arrives next to `abort_when_nan_found` there.  `sm_count` stays last
        # on both sides.
        backend_args = (
            input,
            topk,
            end,
            sorted, sorted_index,
            output_val, output_idx,
            output_idx_offset,
            idx_oob_fill_value,
            value_oob_fill_value,
            return_value,
            abort_when_nan_found,
            check_nan,
            int(torch.cuda.get_device_properties(
                torch.cuda.current_device()).multi_processor_count),
        )
        # `maca_c` means this device's kernel: the extension is resolved here,
        # not in the signature, which keeps `torch` usable with no MACA device.
        # `launching()` installs torch's current stream for the call -- the C++
        # side reads it through TVMFFIEnvGetStream and would otherwise see the
        # null handle and launch on the legacy default stream.
        # `DEEP_SELECT_NO_STREAM_GUARD` is the escape hatch for bisecting that.
        if os.environ.get("DEEP_SELECT_NO_STREAM_GUARD"):
            _backend_for().topk(*backend_args)
        else:
            with _binding.launching():
                _backend_for().topk(*backend_args)
        return output_val, output_idx
