#!/usr/bin/env python3
"""Does the threshold bin overflow the arena?  Computed from the data.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe_arena_fill.py

`probe_stage1_passes.py` asks the question by timing; this one asks it by
arithmetic, and the two are meant to be read together -- the timing says
whether the overflow walks cost anything, this says whether they fire at all.

Both rows being compared:

* `radix_topk_row_bf16_k` -- `stage1`'s row.  8-bit coarse (`key >> 8`, 256
  bins), arena = `kSmemInputSize` slots, x2 = 7028 with compact indices
  (`maca_topk.cu`'s `KTOPK_COMPACT_BF16_INDICES`).  Passes 3 and 4 of its
  two-pass walk run only when the threshold bin's population exceeds that.
* `radix_topk_row_bf16_b` -- the production row.  12-bit coarse
  (`key >> 4`, 4096 bins) aliased with `kCoarse12ArenaEntries` = 4096 slots.

The key encode is `radix_core.cuh:173` transcribed: positive keys xor 0x8000,
negative keys xor 0xffff, on the raw 16 bits.  The threshold bin is the one the
descending running count crosses `topk` at -- i.e. `count(bin > t) < topk <=
count(bin >= t)`, which is the same crossing the kernel's suffix-cumsum finds.

Reported per arm: the **worst threshold bin** over all rows (this is what
decides whether the rescan fires) and the **largest bin anywhere** (this is the
cliff -- the threshold lands on a bin by rank, not by size, so a large bin is
only a hazard, not a defect).
"""

import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import lib                                          # noqa: E402
import test as official                             # noqa: E402
from probe_stage1_passes import make_bituniform     # noqa: E402

COARSE12_SHIFT, COARSE12_CAP = 4, 4096
COARSE8_SHIFT, COARSE8_CAP = 8, 7028
CHUNKS = 16            # `wave_filled_chunks(16, 104)` -> 16 on a C500


def ordered_key(x):
    """bf16 -> order-preserving 16-bit key, transcribed from `radix_core.cuh:173`."""
    bits = x.contiguous().view(torch.int16).to(torch.int32) & 0xFFFF
    sign = bits >> 15
    mask = (0x8000 | ((0 - sign) & 0xFFFF)) & 0xFFFF
    return (bits ^ mask) & 0xFFFF


def threshold_and_max(seg, topk, shift):
    """(population of the threshold bin, population of the largest bin)."""
    counts = torch.bincount((seg >> shift).to(torch.int32)).float()
    cum = counts.flip(0).cumsum(0)
    i = int(torch.searchsorted(cum, float(topk)))
    return int(counts[len(counts) - 1 - i]), int(counts.max())


def report(label, key, topk, shift, cap, seg_len):
    worst_t, worst_max, over = 0, 0, 0
    rows = key.shape[0]
    for r in range(rows):
        t, m = threshold_and_max(key[r, :seg_len], topk, shift)
        worst_t, worst_max = max(worst_t, t), max(worst_max, m)
        if t > cap:
            over += 1
    width = "12-bit (key>>4)" if shift == COARSE12_SHIFT else "8-bit (key>>8)"
    print(f"  {label:<12} {width:<16} threshold={worst_t:>7}  "
          f"largest={worst_max:>7}  cap={cap:>5}  "
          f"-> {worst_t/cap:.2f}x cap, {over}/{rows} over")


def main():
    torch.set_default_device("cuda")

    for b, v, topk in [(64, 1048576, 512), (6, 1048576, 512)]:
        p = [c for c in official.performance_cases()
             if c.batch_size == b and c.vocab_size == v and c.topk == topk
             and str(c.dtype) == "torch.bfloat16"][0]
        p.seed = (b * 1_000_003 + v * 11 + topk) % 2**31
        crowded = ordered_key(lib.generate_testcase(p).input)
        uni = ordered_key(make_bituniform(b, v))

        print(f"b{b} V{v} k{topk}")
        if b >= 64:
            # Row kernel: one CTA walks one full row; coarse12 + arena 4096.
            report("crowded", crowded, topk, COARSE12_SHIFT, COARSE12_CAP, v)
            report("bituniform", uni, topk, COARSE12_SHIFT, COARSE12_CAP, v)
        else:
            # stage1: chunk = vocab/chunks elements; 8-bit coarse, arena 7028.
            csz = (((v + CHUNKS - 1) // CHUNKS) + 7) // 8 * 8
            report("crowded", crowded, topk, COARSE8_SHIFT, COARSE8_CAP, csz)
            report("bituniform", uni, topk, COARSE8_SHIFT, COARSE8_CAP, csz)
        print()


if __name__ == "__main__":
    main()
