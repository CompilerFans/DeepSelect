import functools
import torch

from typing import Optional, Tuple

from . import _binding
from ._backends import topk_torch
from ._log import log_call


_BACKENDS = ("maca_c", "torch")

_KERNEL_NAME = "deep_select_maca"


# `structs.h`'s requirement, for when no kernel is built.
_ALIGNMENT_REQUIREMENT_BYTES = (1024, 32)


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    try:
        # The FFI entry returns a python list, not a C++ pair; normalize it.
        pair = _binding.load(_KERNEL_NAME).get_alignment_requirement()
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
    backend: str = "maca_c",
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
        backend: str. `"maca_c"` (the default) runs this device's MACA kernel;
                `"torch"` runs a reference implementation built from torch ops,
                which needs no MACA device.
        check_nan: bool. Whether each row is scanned for a NaN at all.  False
                is the caller asserting its input is NaN-free: no scan runs, a
                NaN is ranked like any other value, and `abort_when_nan_found`
                never fires.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    # Checked before anything is read off `input`, so a typo is reported as
    # such rather than as whatever a None tensor does next.
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
        # The slice is a no-op view when the row is already aligned.
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
        # The 14 positional arguments the tvm-ffi entry takes.  `begin` and
        # `hint` are rejected above and `output_val` is None when
        # `return_value` is False.  The order is the C++ entry's, not this
        # signature's: `check_nan` arrives next to `abort_when_nan_found`
        # there, and `sm_count` -- the one argument that describes the device
        # rather than the problem, so the grid is sized for the part in front
        # of the call -- stays last on both sides.
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
        # `launching()` installs torch's current stream, which the C++ side
        # reads through TVMFFIEnvGetStream; without it the null handle means
        # the legacy default stream.
        with _binding.launching():
            _binding.load(_KERNEL_NAME).topk(*backend_args)
        return output_val, output_idx
