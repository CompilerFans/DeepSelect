#!/usr/bin/env python3
"""Tail-coverage probe for the block-histogram walk's adjacent-4 tiling.

The official grid is powers of two, where `nblk % 4 == 0` and the tail loop
never runs.  This walks lengths that leave 0/1/2/3 blocks over (`nblk % 4`) at
both the block-hist arm (length >= 256*topk) and the dense arm, and checks the
contract -- `min(selected) >= max(unselected)` -- plus index range, uniqueness
and value/index pairing.

Run on the pod:
  CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python /tmp/tailprobe.py
"""
import sys

import torch

import deep_select

torch.set_default_device("cuda")

V = 8192
B = 6


def check(x, end, k, sorted_index=False):
    vals, idx = deep_select.topk(
        x, k, end=end, indices_type=torch.int32, return_value=True,
        sorted_index=sorted_index, backend="maca_c")
    idx_l = idx.long()
    for r in range(x.shape[0]):
        L = int(end[r].item())
        row = x[r, :L]
        sel = idx_l[r]
        assert int(sel.min()) >= 0 and int(sel.max()) < L, "index range"
        assert sel.unique().numel() == k, "unique"
        got = row[sel]
        assert torch.equal(got, vals[r]), "value/index pairing"
        mask = torch.zeros(V, dtype=torch.bool, device=x.device)
        mask[sel] = True
        unsel = row[~mask[:L]]
        if unsel.numel():
            assert float(got.min()) >= float(unsel.max()), (
                f"contract: min(sel)={float(got.min())} < "
                f"max(unsel)={float(unsel.max())}")
        if sorted_index:
            # test.py:100 -- `sorted_index` is index-ascending, not value order.
            assert bool((sel[:-1] <= sel[1:]).all()), "sorted_index order"
    return idx


def main():
    torch.manual_seed(1234)
    x = torch.randn(B, V, dtype=torch.float32).to(torch.bfloat16).contiguous()
    n = 0
    # block-hist arm: length >= 256 * k.  k=8 -> length >= 2048.
    for base in (2048, 4096, 6144):
        for over in (0, 1, 2, 3):
            L = base + 64 * over          # nblk = base/64 + over, tail = over
            end = torch.full((B,), L, dtype=torch.int32)
            for k in (8, 16):
                check(x, end, k)
                n += 1
    # dense arm, same remainders; here length < 256*k
    for base in (512, 1024):
        for over in (0, 1, 2, 3):
            L = base + 64 * over
            end = torch.full((B,), L, dtype=torch.int32)
            for k in (8,):
                check(x, end, k)
                n += 1
    # sorted_index on a few of the same shapes
    for L in (2112, 2176, 2240, 2048):
        end = torch.full((B,), L, dtype=torch.int32)
        check(x, end, 8, sorted_index=True)
        n += 1
    # a NaN-free row that is short enough to skip the check, and one tiny row
    end = torch.full((B,), 64, dtype=torch.int32)
    check(x, end, 8)
    n += 1
    print(f"TAILPROBE OK {n} calls")


if __name__ == "__main__":
    sys.exit(main())
