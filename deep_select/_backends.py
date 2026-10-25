"""The reference backend: `torch` ops, the contract's slow answer.

It is not the kernel -- `maca_c` is the extension `_binding` loads, and
`interface.py` owns the contract and dispatches to both.  What lives here is
what the contract is answered *in terms of*: `topk_torch`, built from torch ops,
which runs on any device and dtype, including where no kernel is built.
"""

import torch

from typing import Optional, Tuple

from ._log import log_call


def _check_common_contract(input: torch.Tensor, topk: int,
                           indices_type: torch.dtype) -> None:
    """The rejections the contract owes the caller, word for word.

    These four are properties of the operator rather than of an implementation:
    the kernel enforces the same four in `csrc/maca_kernels/common/maca_topk.cu`,
    and a caller must hear the same thing whichever backend it picked.
    """
    if topk <= 0:
        raise ValueError(f"topk must be positive, got {topk}")
    if input.dim() != 2:
        raise ValueError(f"input must be 2-D, got {input.dim()} dimensions")
    if input.stride(1) != 1:
        raise ValueError("input.stride(1) must be 1")
    if indices_type not in (torch.int32, torch.int64):
        raise ValueError(
            f"indices_type must be int32 or int64, got {indices_type}")


def _row_lengths(end: Optional[torch.Tensor], n_rows: int, vocab_size: int,
                 device: torch.device) -> torch.Tensor:
    """The per-row visible length `end` names, or a full row each.

    `end` is an exclusive upper bound, clamped into the row.  One body for
    both reference backends: they spelled this identically, and a difference
    here would be a difference in what "the row" means.
    """
    if end is None:
        return torch.full((n_rows,), vocab_size, dtype=torch.int64,
                          device=device)
    return end.to(torch.int64).clamp(min=0, max=vocab_size)


def _nan_rows(input: torch.Tensor, lengths: torch.Tensor,
              topk: int) -> Optional[torch.Tensor]:
    """Per-row NaN flags for the rows the contract scans, or `None`.

    Only a row whose visible length is longer than `topk` is scanned: one
    that short is answered by returning its whole window, so a NaN in it is
    ranked like any other value and never raises the flag.  That is the
    kernel's rule -- `contract.cuh`'s `length <= topk` shortcut returns before
    its scan -- and both reference backends keep it.  They used to scan every
    row whose window held a NaN as soon as one long row in the batch turned
    the scan on, so a batch mixing the two could abort where the kernel
    answered.

    `None` means nothing was scanned (no row is long enough), which is
    distinct from flags that are all False.  The `any()` probes are device
    syncs, so the caller runs this only when `check_nan` is on.
    """
    checked = lengths > topk
    if not bool(checked.any().item()):
        return None
    cols = torch.arange(input.shape[1], device=input.device)
    visible = (torch.isnan(input.float())
               & (cols.unsqueeze(0) < lengths.unsqueeze(1))
               & checked.unsqueeze(1))
    return visible.any(dim=1)


def _abort_on_nan_rows(nan_rows: Optional[torch.Tensor]) -> None:
    """The abort arm of the NaN contract, for either reference backend.

    One definition of the raise, so the two cannot answer the same input
    differently.  It fires only for the rows `_nan_rows` scanned -- a short
    row's NaN is ranked like any other value.
    """
    if nan_rows is not None and bool(nan_rows.any().item()):
        raise RuntimeError("NaN detected in the input")


def _stamp_nan_rows(out_idx: torch.Tensor,
                    nan_rows: Optional[torch.Tensor]) -> None:
    """The sentinel `abort_when_nan_found=False` leaves, for the reference.

    Slot 0 of the NaN row only, on the tensor that goes back to the caller
    whichever buffer that is; the rest of the row stays undefined, so a NaN row
    must be excluded from any value comparison.  The kernel writes the same
    constant to the same slot -- `dispose_nan_row` in `contract.cuh`; this is
    the reference's copy of that contract, not a second contract.
    """
    if nan_rows is not None and bool(nan_rows.any().item()):
        out_idx[nan_rows, 0] = 0x3F3F3F3F


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
    # The contract rejections both backends owe the caller, then this one's
    # own.  They are the kernel's too (`csrc/maca_kernels/common/maca_topk.cu`,
    # which is what every family builds) -- the operator's contract, not an
    # implementation's, so a strided view must be an error whichever backend
    # was chosen.  `sorted` is *not* float32-only here: the kernel orders
    # bfloat16 too.
    _check_common_contract(input, topk, indices_type)
    if topk > 4096:
        raise ValueError(f"topk must be <= 4096, got {topk}")
    if sorted and not return_value:
        raise ValueError("`return_value` must be enabled when `sorted` is True")
    if sorted and sorted_index:
        raise ValueError("`sorted` and `sorted_index` cannot be used at the same time")
    if input.dtype not in (torch.float32, torch.bfloat16):
        raise ValueError(f"input dtype must be float32 or bfloat16, got {input.dtype}")
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

    # `end` is an exclusive per-row upper bound; a row whose visible length is
    # <= topk is never NaN-checked (see `_nan_rows`).  The check is on unless
    # `check_nan` says otherwise.
    lengths = _row_lengths(end, n_rows, vocab_size, device)
    nan_rows = _nan_rows(input, lengths, topk) if check_nan else None
    if abort_when_nan_found:
        _abort_on_nan_rows(nan_rows)
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
    _stamp_nan_rows(out_idx, nan_rows)

    return out_val, out_idx
