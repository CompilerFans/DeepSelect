"""Ties in the merge, now that the merge ranks with the coarse12 row.

The merge's input is not a row of data -- it is each chunk's top-`topk`, so its
values are the row's extreme tail and they crowd.  Routing it through
`radix_topk_row_bf16_b` puts it on that row's arena, and CLAUDE.md records an
*unconfirmed* defect in that row's overflow path (`overflow_emit_member` may
write slot 0 twice when the threshold bin does not fit the arena).  Crowding is
what reaches the overflow path, so this drives it deliberately: every candidate
equal (the whole row is one value), and a row built from few distinct values,
which is the shape where the threshold bin is widest.

The contract checked is `tests/test.py`'s own and the operator's: indices
unique and in range, `value_i == input[index_i]`, and the definitional
`min(selected) >= max(unselected)` over the visible window.  A duplicate or a
mis-placed slot shows up as one of those failing, not as a different ranking --
ties leave the *set* unspecified, so the contract is the gate, never an
elementwise comparison against another run.

    CUDA_VISIBLE_DEVICES=2 python3 probe_merge_ties.py
"""
import os
import sys

import torch

TREE = os.environ.get("DS_TREE", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "tests"))
torch.set_default_device("cuda")
import lib                                                          # noqa: E402
import deep_select                                                  # noqa: E402

B, V, K = 6, 262144, 512            # `chunked_bf16_applies`: 6 <= 262144/21000
CHUNKS = 16
CHUNK = (V + CHUNKS - 1) // CHUNKS // 8 * 8

CASES = {
    # name: (how to fill, how many distinct values)
    "all-equal": ("constant", 1),
    "two-values": ("constant", 2),
    "mostly-equal": ("constant", 3),
}


def fill(kind, distinct, rows, vocab):
    """A row whose candidates all land in a handful of bins."""
    x = torch.zeros(rows, vocab, dtype=torch.bfloat16)
    if distinct == 1:
        x.fill_(1.0)
    else:
        # `distinct` values spread over the top of the bf16 range, so every
        # chunk's top-`topk` is drawn from the same few of them.
        for i in range(distinct):
            x[:, i::vocab] = 1.0 + i * 0.5
        x[:, :distinct] = torch.arange(distinct, dtype=torch.bfloat16) * 0.25
    return x


failures = []
for name, (kind, distinct) in CASES.items():
    p = lib.TestParam(B, V, K, False, False, True, torch.bfloat16,
                      torch.int32, num_runs=0)
    p.seed = 20261010
    t = lib.generate_testcase(p)
    t.input.copy_(fill(kind, distinct, B, V))
    got = deep_select.topk(
        t.input, p.topk, sorted=False, begin=None, end=None,
        indices_type=p.out_idx_dtype, sorted_index=False, hint=None,
        output_idx=None, output_idx_offset=t.output_idx_offset,
        idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,
        return_value=True, abort_when_nan_found=False, backend="maca_c")
    val, idx = (got if isinstance(got, (tuple, list)) else (None, got))
    idx = idx.to("cpu")
    val = val.to("cpu").float()
    x = t.input.to("cpu")
    n_rows, k = idx.shape
    problems = []
    for r in range(n_rows):
        row_idx = idx[r].to(torch.int64)
        if row_idx.min() < 0 or row_idx.max() >= V:
            problems.append(f"row{r}: index out of range")
            continue
        if len(set(row_idx.tolist())) != k:
            problems.append(f"row{r}: duplicate index")
        if not torch.equal(val[r], x[r][row_idx].float()):
            problems.append(f"row{r}: value != input[index]")
        sel = x[r][row_idx].min().item()
        # The row is not `end`-windowed here, so every unselected column counts.
        mask = torch.ones(V, dtype=torch.bool, device=x.device)
        mask[row_idx] = False
        if mask.any() and sel < x[r][mask].max().item():
            problems.append(
                f"row{r}: min(selected)={sel} < max(unselected)="
                f"{x[r][mask].max().item()}")
    ok = not problems
    print(f"{name:14} distinct={distinct}  rows={n_rows} k={k}  "
          f"{'contract holds' if ok else '; '.join(problems[:3])}")
    if not ok:
        failures.append(name)

print(f"\n{len(CASES) - len(failures)}/{len(CASES)} tie shapes answered the contract")
sys.exit(1 if failures else 0)
