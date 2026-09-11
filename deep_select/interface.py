import functools
import importlib
import torch

from typing import Optional, Tuple

from ._arch import FAMILY_OF_TARGET, native_target


# A backend name is an architecture name: `xcore1000`, `xcore1500`,
# `xcore1600`.  Each names the kernel built for that architecture -- setup.py
# builds one extension per architecture, `deep_select.deep_select_xcore<N>`,
# holding the kernel whose capacity fits it -- so `backend=` selects a kernel
# by architecture and nothing else, and `topk` with no `backend` runs the
# kernel this device has.  `torch` is not a kernel: it is a reference
# implementation of the same contract, and runs anywhere.
_BACKEND_NAMES = tuple(f"xcore{family}"
                       for family in sorted(set(FAMILY_OF_TARGET.values())))


@functools.lru_cache(maxsize=None)
def _backend_for(name: str):
    """The extension module that implements `name`.

    Imported on demand and cached: the module holds a device binary, and a
    process that never calls `topk` on a kernel should not load one.
    """
    assert name in _BACKEND_NAMES, name
    try:
        return importlib.import_module(f".deep_select_{name}", __package__)
    except ImportError as exc:
        raise RuntimeError(
            f"backend {name!r} has not been built; build it with "
            f"CUCC_TARGETS={name} (setup.py builds one extension per "
            f"architecture, and only for the architectures it is asked for)"
        ) from exc


# `structs.h`'s INPUT_STRIDE_ALIGNMENT_REQUIREMENT and OUTPUT_STRIDE_ALIGNMENT_
# REQUIREMENT.  A built kernel exports them through `get_alignment_requirement()`
# and that is where the value comes from whenever one is available; these are
# the same numbers for when none is, so that `backend="torch"` works on a
# machine with no kernel built at all.  A fallback to a constant is sound here
# because the alignment is the operator's contract -- identical for every
# kernel -- rather than any one build's property.
_ALIGNMENT_REQUIREMENT_BYTES = (1024, 32)


@functools.lru_cache(maxsize=1)
def get_stride_requirement() -> Tuple[int, int]:
    """
    Returns the stride requirement for input / output tensors, in bytes
    """
    try:
        return _backend_for(native_target()).get_alignment_requirement()
    except RuntimeError:
        return _ALIGNMENT_REQUIREMENT_BYTES


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
                The NaN check itself is always enabled. Exception: when the row's length <= topk, it is skipped.
        backend: str. Which implementation to run, or None (the default) for
                the kernel this device has.  A backend name is an architecture
                name -- `xcore1000`, `xcore1500`, `xcore1600` -- and each names
                the kernel built for that architecture: `xcore1000` is the
                MACA-native kernel, `xcore1600` is the ported one, and which of
                them a given device runs is a property of the device, not a
                choice.  `torch` is not a kernel; it is a reference
                implementation of the same contract, built on torch ops, and
                runs on any device.

    Return:
        output_val: (b, topk), dtype=input.dtype.
        output_idx: (b, topk), dtype=indices_type.
                    The output tensors may not be contiguous, when topk * sizeof(input.dtype or indices_dtype) is not a multiple of 32 Bytes
    """

    # Checked before anything is read off `input`: a caller who mistyped a
    # backend name should hear about that, not about whatever the None they
    # passed in place of a tensor does to the next line.
    if backend is not None and backend != "torch" and backend not in _BACKEND_NAMES:
        raise ValueError(
            f"Unsupported backend: {backend!r}. Expected one of "
            f"{', '.join(_BACKEND_NAMES)}, 'torch', or None for the kernel this "
            f"device has"
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
    else:
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
        # No `backend` means this device's kernel; resolving it here rather
        # than in the signature keeps `torch` usable on a machine with no MACA
        # device at all.
        _backend_for(backend if backend is not None else native_target()).topk(*backend_args)
        return output_val, output_idx


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
    # The same contract rejections the kernel path enforces (upstream:
    # csrc/api.cpp).  They are part of the operator's contract, not of any one
    # implementation -- a caller that passes a strided view must get an error
    # rather than a silently different answer just because it chose this
    # backend.
    #
    # Note what is *not* here: upstream restricts `sorted` to float32, and the
    # kernel path keeps that restriction, but this repository's kernel
    # implements it for bfloat16 too (see `csrc/maca_topk.cu`).  The reference
    # describes the operator this repository ships, so it follows the wider
    # contract; `torch.topk` orders bfloat16 natively, so nothing extra is
    # needed for it.
    if sorted and not return_value:
        raise ValueError("`return_value` must be enabled when `sorted` is True")
    if sorted and sorted_index:
        raise ValueError("`sorted` and `sorted_index` cannot be used at the same time")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D, got {input.dim()} dimensions")
    if input.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input dtype must be float32 or bfloat16, got {input.dtype}")
    if input.stride(1) != 1:
        raise ValueError("input.stride(1) must be 1")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError(f"indices_type must be int32 or int64, got {indices_type}")
    if output_idx is not None:
        if output_idx.dtype != indices_type:
            raise ValueError(f"output_idx dtype must be {indices_type}, got {output_idx.dtype}")
        if output_idx.stride(1) != 1:
            raise ValueError("output_idx.stride(1) must be 1")
        if (output_idx.size(0) < input.shape[0]
                or output_idx.size(1) < topk):
            raise ValueError(
                f"output_idx must be at least (batch_size, topk) = "
                f"({input.shape[0]}, {topk}), got {tuple(output_idx.shape)}")

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
