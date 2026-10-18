"""The two reference backends: `torch` and `deep_gemm`.

Neither is the kernel -- `maca_c` is the extension `_binding` loads, and
`interface.py` owns the contract and dispatches to all three.  What lives here
is what the contract is answered *in terms of*: `topk_torch` in torch ops, which
runs on any device and dtype, and `topk_deep_gemm` through the soft-dependency
package's selector.

`deep_gemm` is imported on first use (`_deep_gemm`), never at module scope, so
this module loads -- and the package imports -- with that package absent.
"""

import functools
import torch

from typing import Optional, Tuple

from ._log import log_call


# `deep_gemm`'s `kMaxTopK` (`csrc/kernels/fp32_topk.cu`).  Its selector refuses
# more, so this backend refuses first, naming the gap rather than the kernel's
# assert.
_DEEP_GEMM_MAX_TOPK = 2048


@functools.lru_cache(maxsize=1)
def _deep_gemm():
    """The `deep_gemm` package, imported on first use.

    A soft dependency -- this repository is standalone -- so it is imported
    only when `backend="deep_gemm"` is asked for.
    """
    import deep_gemm

    return deep_gemm


@functools.lru_cache(maxsize=1)
def deep_gemm_available() -> bool:
    """Whether `backend="deep_gemm"` can run in this process.

    True when the `deep_gemm` package imports *and* carries the entry that
    backend calls.  Asked of the package, not of a location: which checkout
    `import deep_gemm` lands on is not this repository's business, only
    whether the call will work.

    The harnesses ask this to decide whether to run the cells only that
    backend can answer; a caller can ask it before naming the backend.
    """
    try:
        return callable(getattr(_deep_gemm(), "fp32_indexer_topk_selector", None))
    except Exception:
        return False


@log_call
def topk_deep_gemm(
    input: torch.Tensor,
    topk: int,
    end: Optional[torch.Tensor] = None,
    sorted_value: bool = False,
    sorted_index: bool = False,
    indices_type: torch.dtype = torch.int64,
    output_idx: Optional[torch.Tensor] = None,
    output_idx_offset: Optional[torch.Tensor] = None,
    idx_oob_fill_value: int = 2147483647,
    value_oob_fill_value: float = float("-inf"),
    return_value: bool = True,
    abort_when_nan_found: bool = True,
    check_nan: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """`topk` served by the `deep_gemm` package's indexer selector
    (`deep_gemm.fp32_indexer_topk_selector`).

    It implements a subset of this contract, and what it cannot serve raises
    `UnsupportedByBackend` rather than answering something narrower:

        * float32 scores only (no bfloat16);
        * `topk <= 2048`;
        * an unordered selection (`sorted_value` / `sorted_index` need the
          MACA kernel).

    `end`, `output_idx`, `output_idx_offset`, the out-of-band fills,
    `return_value` and the NaN contract are implemented here, on top of it.

    Known limitation: a row holding more than 4096 values in one threshold
    bucket can select an arbitrary subset of them, varying from run to run.
    `maca_c` does not have this limitation.
    """
    # The contract's exception is defined in `interface.py`, which imports this
    # module for both reference backends -- so this one name travels the other
    # way and cannot be imported at module scope.
    from .interface import UnsupportedByBackend

    try:
        module = _deep_gemm()
    except ImportError as exc:
        raise RuntimeError(
            "backend 'deep_gemm' needs the mcDeepGEMM package importable "
            "(install it, or put its repository root on PYTHONPATH); this "
            "repository runs standalone too, and that is the only case where "
            "the import is missing"
        ) from exc

    # The operator's own contract rejections, as `topk_torch` enforces them:
    # they are properties of the contract, not of an implementation, so a
    # caller must hear the same thing whichever backend it picked.
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D, got {input.dim()} dimensions")
    if input.stride(1) != 1:
        raise ValueError("input.stride(1) must be 1")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError(
            f"indices_type must be int32 or int64, got {indices_type}")
    if input.dtype != torch.float32:
        raise UnsupportedByBackend(
            f"backend 'deep_gemm' ranks float32 scores, and was given "
            f"{input.dtype}; 'maca_c' and 'torch' take bfloat16 as well")
    if topk > _DEEP_GEMM_MAX_TOPK:
        raise UnsupportedByBackend(
            f"backend 'deep_gemm' selects at most {_DEEP_GEMM_MAX_TOPK} per "
            f"row and was asked for {topk}; 'maca_c' goes to 4096")
    if sorted_value or sorted_index:
        raise UnsupportedByBackend(
            "backend 'deep_gemm' returns an unordered selection; "
            "`sorted_value`/`sorted_index` need backend 'maca_c'")

    n_rows, vocab_size = input.shape
    device = input.device
    if input.stride(0) != vocab_size:
        # The `deep_gemm` kernel walks a row as `scores + row * n_cols`, i.e. it
        # assumes tightly packed rows; a padded row stride would quietly rank
        # the wrong elements.  Padding is legal in this contract, so normalize
        # instead of refusing.
        input = input.contiguous()
    lengths = (torch.full((n_rows,), vocab_size, dtype=torch.int64,
                          device=device)
               if end is None
               else end.to(torch.int64).clamp(min=0, max=vocab_size))

    # The NaN contract, checked as `topk_torch` checks it: on unless the caller
    # turned it off, and skipped for a row whose window is not longer than
    # `topk`.  The `any()` probe is itself a device sync on a per-row predicate,
    # so it goes inside the flag rather than being saved by it.
    if check_nan and bool((lengths > topk).any().item()):
        cols = torch.arange(vocab_size, device=device)
        nan_rows = (torch.isnan(input)
                    & (cols.unsqueeze(0) < lengths.unsqueeze(1))).any(dim=1)
        if bool(nan_rows.any().item()):
            if abort_when_nan_found:
                raise RuntimeError("NaN detected in the input")
            if output_idx is None:
                output_idx = torch.empty((n_rows, topk), dtype=indices_type,
                                         device=device)
            output_idx[nan_rows, 0] = 0x3F3F3F3F
            values = None
            if return_value:
                values = torch.full((n_rows, topk), value_oob_fill_value,
                                    dtype=input.dtype, device=device)
            return values, output_idx

    selected = module.fp32_indexer_topk_selector(
        input, lengths.to(torch.int32), topk, return_val=False,
        backend="maca_c")["indices"]
    valid = selected >= 0

    # `-1` marks the padding; everything else is already the column index.
    # (`.to` because the selector answers in int32 whatever `indices_type` is.)
    columns = selected.to(torch.int64)
    if output_idx_offset is not None:
        columns = columns + output_idx_offset.to(torch.int64).unsqueeze(1)
    out_idx = torch.where(valid, columns,
                          torch.full_like(columns, idx_oob_fill_value)
                          ).to(indices_type)

    out_val = None
    if return_value:
        gathered = input.gather(1, selected.clamp_min(0).to(torch.int64))
        out_val = torch.where(valid, gathered,
                              torch.full_like(gathered, value_oob_fill_value))

    if output_idx is not None:
        output_idx.copy_(out_idx)
        out_idx = output_idx
    return out_val, out_idx


@log_call
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
    check_nan: bool = True,
) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
    """Reference `torch` implementation of the `topk` contract.

    Same contract as `topk`, in torch ops only, so it runs on any device and
    any dtype.  Rows longer than `topk` are handled by masking the
    out-of-window tail to `-inf` and taking one `torch.topk` over the padded
    row, which reproduces the per-row window without a loop over rows.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if topk > 4096:
        raise ValueError(f"topk must be <= 4096, got {topk}")
    # The same contract rejections the kernel path enforces
    # (`csrc/maca_kernels/common/maca_topk.cu`, which is what every family builds) -- the
    # operator's contract, not an implementation's, so a strided view must be an
    # error whichever backend was chosen.  `sorted` is *not* float32-only here:
    # that kernel orders bfloat16 too.
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
    # a row whose visible length is <= topk; the check is on unless
    # `check_nan` says otherwise.
    if end is not None:
        lengths = end.to(torch.int64).clamp(min=0, max=vocab_size)
    else:
        lengths = torch.full((n_rows,), vocab_size, dtype=torch.int64, device=device)

    nan_rows = torch.zeros(n_rows, dtype=torch.bool, device=device)
    checked = (lengths > topk) if check_nan else torch.zeros_like(lengths, dtype=torch.bool)
    if bool(checked.any().item()):
        cols = torch.arange(vocab_size, device=device)
        visible = torch.isnan(input.float()) & (cols.unsqueeze(0) < lengths.unsqueeze(1))
        nan_rows = visible.any(dim=1)
    if abort_when_nan_found and bool(nan_rows.any().item()):
        raise RuntimeError("NaN detected in the input")
    # A NaN row is answered in its first index slot and left otherwise
    # unspecified -- that row only.  Returning for the whole batch would leave
    # every clean row of it defined by whatever the buffer happened to hold.

    # Rows shorter than topk select their visible prefix and pad the rest;
    # masking the hidden tail to -inf makes one torch.topk reproduce that.
    work = input
    if bool((lengths < vocab_size).any().item()):
        cols = torch.arange(vocab_size, device=device)
        work = input.masked_fill(cols.unsqueeze(0) >= lengths.unsqueeze(1),
                                 float("-inf"))

    k_eff = min(topk, vocab_size)
    values, indices = torch.topk(work, k_eff, dim=1, sorted=bool(sorted))

    # The window mask is `-inf`, which is also a value a row can hold: a hidden
    # slot then ties with the visible ones and `torch.topk`'s tie order is
    # unspecified, so it can hand back an index outside the window.
    #
    # Such a row is re-picked from its window alone.  Rewriting the indices to
    # the window's first `k_eff` *positions* instead orders values against
    # positions, so the row reports a value it does not hold at the index it
    # reports.  Only `min(k_eff, length)` entries are answerable at all; the
    # rest of the row stays padding, masked into the fills below.
    in_window = indices < lengths.unsqueeze(1)
    if bool((~in_window).any().item()):
        for row in (~in_window).any(dim=1).nonzero().flatten().tolist():
            length = int(lengths[row])
            picked = min(k_eff, length)
            if picked == 0:
                continue
            row_values, row_indices = torch.topk(
                input[row, :length], picked, sorted=bool(sorted)
            )
            values[row, :picked] = row_values
            indices[row, :picked] = row_indices

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
        # `torch.gather` needs an int64 index whatever the output dtype.
        #
        # Order the *selected* slots only, sending padding to the end: a selected
        # slot holds `index + output_idx_offset` while padding holds the raw
        # fill, so sorting the row as it stands orders two currencies and can
        # put padding in the slots that must hold real in-window indices.
        sel = (torch.arange(topk, device=device).unsqueeze(0)
               < lengths.unsqueeze(1))
        key = torch.where(sel, out_idx.to(torch.int64),
                          torch.iinfo(torch.int64).max)
        order = torch.argsort(key, dim=1, stable=True)
        out_idx = torch.gather(out_idx, 1, order)
        if return_value:
            out_val = torch.gather(out_val, 1, order)

    if output_idx is not None:
        output_idx.copy_(out_idx)
        out_idx = output_idx

    # Stamped on the tensor that goes back to the caller, whichever it is.
    if bool(nan_rows.any().item()):
        out_idx[nan_rows, 0] = 0x3F3F3F3F

    return out_val, out_idx
