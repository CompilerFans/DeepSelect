#!/usr/bin/env python3
"""How often does the row kernel's exact-fit exit fire, and what does it cost?

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe_exact_fit.py

`radix_topk_row_bf16_b` walks the row twice.  It walks it a **third** time, in
two places, and one of them is a **scalar** load loop:

* `:1363-1368` -- the exact-fit exit, taken when `remain_topk` reaches 0 after
  the narrow step.  `for (idx = tx; idx < length; idx += BLOCK_SIZE)` reading
  `__ldg(input + idx)` two bytes at a time, with a shared atomic per emitted
  element.
* `:1434-1459` -- the overflow exit, when the threshold bin is wider than the
  4,096-slot arena.  This one is vectorized.

The scalar one is the expensive one: measured on this part at the production
geometry, a scalar full-row walk runs at **31% of the read wall** against the
vector walk's **90%** (docs/experiments/20261009_c500_dsa_decode/README.md).

`remain_topk` reaches 0 exactly when the count of elements strictly above the
12-bit threshold equals `topk` -- the coarse complement fills the window on its
own.  That is arithmetic on the row, so it can be counted here rather than
inferred: for each row, take `T` = the bin where the descending running count
crosses `topk`, and test `count(key >> 4 > T) == topk`.

The same file counts the overflow condition, `count(key >> 4 == T) > 4096`,
because it is the other way to buy a third walk and it is a property of the
same `T`.
"""

import argparse
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import lib                                    # noqa: E402
import test as official                       # noqa: E402

COARSE12_SHIFT = 4
ARENA = 4096


def key_rows(x, r0, r1):
    """bf16 rows [r0, r1) -> order-preserving 16-bit keys, as int32.

    Transcribed from `radix_core.cuh:173`.  Chunked because the grid's largest
    row set is 8.6 GB of bf16 and the upcast is 2x that per operator.
    """
    bits = x[r0:r1].contiguous().view(torch.int16).to(torch.int32)
    sign = (bits >> 15) & 1
    mask = 0x8000 | ((0 - sign) & 0xFFFF)
    return (bits ^ mask) & 0xFFFF


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", type=int, default=1,
                    help="how many grid cells to check (each is one bf16 grid row)")
    args = ap.parse_args()

    torch.set_default_device("cuda")

    cells = [(4096, 1048576, 512), (4096, 524288, 512), (4096, 131072, 512),
             (768, 262144, 512), (256, 1048576, 1024)]
    for b, v, topk in cells[:args.cells]:
        hit = [c for c in official.performance_cases()
               if (c.batch_size, c.vocab_size, c.topk) == (b, v, topk)
               and str(c.dtype) == "torch.bfloat16"]
        if not hit:
            print(f"{b}x{v} k{topk}: not in the grid")
            continue
        p = hit[0]
        p.seed = (b * 1_000_003 + v * 11 + topk) % 2**31
        x = lib.generate_testcase(p).input
        rows = x.shape[0]

        exact_fit = 0
        overflow = 0
        thr_hist = []
        above_hist = []
        CH = 16                      # rows per chunk; the upcast is 4x a chunk
        for r0 in range(0, rows, CH):
            r1 = min(r0 + CH, rows)
            k12 = key_rows(x, r0, r1) >> COARSE12_SHIFT          # (n, v) int32
            n = k12.shape[0]
            flat = (torch.arange(n, device=k12.device).view(-1, 1) * ARENA
                    + k12).reshape(-1)
            counts = torch.bincount(flat.to(torch.int64), minlength=n * ARENA).view(n, ARENA)
            counts = counts.float()
            # the bin where the descending count first reaches `topk`.
            # `cum[i, j]` counts the bins >= (ARENA-1-j), so the crossing bin is
            # the first j whose cumulative reaches `topk`.
            cum = counts.flip(1).cumsum(1)
            j = (cum < float(topk)).sum(1)
            t = ARENA - 1 - j
            thr = counts.gather(1, t.view(-1, 1)).view(-1)
            above = counts.sum(1) - counts.cumsum(1).gather(1, t.view(-1, 1)).view(-1)
            exact_fit += int((above == float(topk)).sum())
            overflow += int((thr > float(ARENA)).sum())
            thr_hist.append(thr)
            above_hist.append(above)
            del k12, flat, counts, cum, j, t, thr, above
            torch.cuda.empty_cache()

        thr = torch.cat(thr_hist)
        abv = torch.cat(above_hist)
        print(f"{b} rows x {v} k={topk}  ({rows} rows)")
        print(f"    exact-fit exit  (`count(>T) == topk`)      {exact_fit}/{rows} rows")
        print(f"    overflow exit   (`count(== T) > {ARENA}`)   {overflow}/{rows} rows")
        print(f"    threshold-bin population: min {int(thr.min())} "
              f"median {int(thr.median())} max {int(thr.max())}")
        print(f"    strictly above:           min {int(abv.min())} "
              f"median {int(abv.median())} max {int(abv.max())}  (topk = {topk})")
        print()


if __name__ == "__main__":
    main()
