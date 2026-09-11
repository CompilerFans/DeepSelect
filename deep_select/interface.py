import functools
import importlib
import torch

from typing import Optional, Tuple

from . import deep_select_cuda as _backend


# Backends that implement the same `topk` / `get_alignment_requirement` pair.
# `maca_c` is the MACA-native kernel; `upstream` is the ported upstream one,
# which is built one shared object per architecture and selected here by the
# architecture torch reports for the current device.
_BACKENDS = ("maca_c", "upstream")


@functools.lru_cache(maxsize=None)
def _backend_for(name: str):
    """Resolve a backend name to the module that implements it.

    The upstream kernel is gated at compile time on the shared memory capacity
    of the architecture it targets -- a config that cannot fit one SM is a
    compile error -- so exactly one build per architecture exists and the one
    to load is a property of the current device.  The mapping from the device's
    CUDA-compatible capability to that build lives in `._arch`, together with
    the table setup.py used to name the builds.
    """
    if name == "maca_c":
        return _backend
    from ._arch import native_target

    target = native_target()
    try:
        return importlib.import_module(f".deep_select_upstream_{target}", __package__)
    except ImportError as exc:
        raise RuntimeError(
            f"backend {name!r} has no build for {target} (this device); "
            f"build with CUCC_TARGETS containing {target}"
        ) from exc


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    return _backend.get_alignment_requirement()


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
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """
    Arguments:
        input: (b, vocab_size), dtype=torch.bfloat16/torch.float. stride(0) must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and stride(1) must be 1.
        topk: int. Select topk elements for each row.
        sorted: bool. Whether to return sorted **output_val**. Only supports fp32.
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
                The NaN check itself is always enabled. Exception: when the row's length <= topk, it is skipped.
        backend: str. Implementation to run. `maca_c` (default) is the MACA-native
                kernel; `upstream` is the ported upstream kernel (one build per
                architecture, selected by the current device); `torch` is a
                reference implementation of the same contract built on torch
                ops.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    N = input.shape[0]

    def get_empty_and_aligned_tensor(dim0: int, dim1: int, device: torch.device, dtype: torch.dtype):
        """
        Return a tensor with shape (dim0, dim1), and stride (X, 1), where X is a multiple of 32B
        """
        output_stride_requirement_bytes = get_stride_requirement()[1]
        output_stride_requirement = output_stride_requirement_bytes // dtype.itemsize
        assert output_stride_requirement > 0
        dim1_rounded = (dim1+output_stride_requirement-1) // output_stride_requirement * output_stride_requirement
        return torch.empty((dim0, dim1_rounded), device=device, dtype=dtype)[:, :dim1]
    
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
        )
    elif backend in _BACKENDS:
        backend_args = (
            input,
            topk,
            begin, end,
            sorted, sorted_index,
            output_val, output_idx,
            output_idx_offset,
            idx_oob_fill_value,
            value_oob_fill_value,
            return_value,
            abort_when_nan_found,
        )
        _backend_for(backend).topk(*backend_args)
        return output_val, output_idx
    else:
        raise ValueError(
            f"Unsupported backend: {backend!r}. "
            f"Expected: {', '.join(_BACKENDS + ('torch',))}"
        )


def topk_torch(
    input: torch.Tensor,
    topk: int,
    sorted: bool = False,
    end: Optional[torch.Tensor] = None,
    indices_type: torch.dtype = torch.int64,
    sorted_index: bool = False,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Reference `torch` implementation of the `topk` contract.

    Same contract as `topk`, expressed with torch ops only, so it runs on any
    device and any dtype.  Rows longer than `topk` are handled by masking the
    out-of-window tail to `-inf` and taking one `torch.topk` over the padded
    row, which reproduces the per-row window without a loop over rows.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > 4096:
        raise ValueError(f"topk must be <= 4096, got {topk}")
    if sorted and input.dtype != torch.float32:
        raise ValueError("`sorted` is only supported for float32 input")

    n_rows, vocab_size = input.shape
    device = input.device

    # `end` is an exclusive per-row upper bound.  The kernel never NaN-checks
    # a row whose visible length is <= topk; the check itself is always on.
    if end is not None:
        lengths = end.to(torch.int64).clamp(min=0, max=vocab_size)
    else:
        lengths = torch.full((n_rows,), vocab_size, dtype=torch.int64, device=device)

    nan_rows = torch.zeros(n_rows, dtype=torch.bool, device=device)
    checked = lengths > topk
    if bool(checked.any().item()):
        cols = torch.arange(vocab_size, device=device)
        visible = torch.isnan(input.float()) & (cols.unsqueeze(0) < lengths.unsqueeze(1))
        nan_rows = visible.any(dim=1)
    if bool(nan_rows.any().item()):
        if abort_when_nan_found:
            raise RuntimeError("NaN detected in the input")
        if output_idx is None:
            output_idx = torch.empty((n_rows, topk), dtype=indices_type, device=device)
        output_idx[nan_rows, 0] = 0x3F3F3F3F
        values = None
        if return_value:
            values = torch.full((n_rows, topk), value_oob_fill_value,
                                dtype=input.dtype, device=device)
        return values, output_idx

    # Rows shorter than topk select their whole visible prefix and are padded
    # with the fill values; masking the hidden tail to -inf makes a single
    # torch.topk reproduce that.
    work = input
    if bool((lengths < vocab_size).any().item()):
        cols = torch.arange(vocab_size, device=device)
        work = input.masked_fill(cols.unsqueeze(0) >= lengths.unsqueeze(1),
                                 float("-inf"))

    k_eff = min(topk, vocab_size)
    values, indices = torch.topk(work, k_eff, dim=1, sorted=bool(sorted))

    # Padding picked up by torch.topk on short rows becomes the fill value.
    valid = torch.arange(k_eff, device=device).unsqueeze(0) < lengths.unsqueeze(1)
    out_idx = torch.full((n_rows, topk), idx_oob_fill_value,
                         dtype=indices_type, device=device)
    idx_sel = indices.to(indices_type)
    if output_idx_offset is not None:
        idx_sel = idx_sel + output_idx_offset.to(indices_type).unsqueeze(1)
    out_idx[:, :k_eff] = torch.where(valid, idx_sel,
                                     torch.full_like(idx_sel, idx_oob_fill_value))

    out_val = None
    if return_value:
        out_val = torch.full((n_rows, topk), value_oob_fill_value,
                             dtype=input.dtype, device=device)
        out_val[:, :k_eff] = torch.where(valid, values.to(input.dtype),
                                         out_val[:, :k_eff])

    if sorted_index:
        # torch.gather requires an int64 index regardless of the output dtype.
        order = torch.argsort(out_idx.to(torch.int64), dim=1, stable=True)
        out_idx = torch.gather(out_idx, 1, order)
        if return_value:
            out_val = torch.gather(out_val, 1, order)

    if output_idx is not None:
        output_idx.copy_(out_idx)
        out_idx = output_idx

    return out_val, out_idx
