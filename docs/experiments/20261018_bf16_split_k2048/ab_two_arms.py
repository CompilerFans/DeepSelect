#!/usr/bin/env python3
"""Targeted k2048-split A/B: pinned seed, contract check, event timing.

Run twice -- once with the baseline .so in deep_select/, once with the
k2048-gated build -- and diff the two outputs.  Each cell is generated once
(pinned seed), checked against the contract, then timed with the repo's
`kk.bench_by_cuda_events` (kernel time, not e2e).

Usage: CUDA_VISIBLE_DEVICES=0 PYTHONPATH=.:tests python3 /tmp/k2048_ab.py TAG
"""
import sys
import torch

sys.path.insert(0, ".")
sys.path.insert(0, "tests")

import lib
import deep_select
from kernelkit.bench import bench_by_cuda_events

tag = sys.argv[1] if len(sys.argv) > 1 else "run"

CELLS = [
    # newly split by the k2048 gate
    (1, 262144, 2048),
    (6, 262144, 2048),
    (1, 1048576, 2048),
    (1, 524288, 2048),
    # k512 controls (same shapes, split already on in both arms)
    (1, 262144, 512),
    (6, 262144, 512),
    (1, 1048576, 512),
    # k2048 controls (V below kChunkedMinVocab -> row path in both arms)
    (1, 131072, 2048),
]

torch.set_default_device("cuda")
print(f"# tag={tag}")
for b, v, k in CELLS:
    p = lib.TestParam(batch_size=b, vocab_size=v, topk=k,
                      sorted_value=False, sorted_index=False, return_value=False,
                      dtype=torch.bfloat16, out_idx_dtype=torch.int32, seed=1234)
    t = lib.generate_testcase(p)

    def run():
        return deep_select.topk(
            t.input, p.topk,
            sorted=p.sorted_value, begin=None, end=t.end,
            indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
            output_idx=None, output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value, abort_when_nan_found=False,
            backend="maca_c")

    val, idx = run()
    torch.cuda.synchronize()

    # Contract: unique indices in range, value pairing, min(selected) >= max(unselected).
    il = idx.long()
    uniq = bool((il.sort(dim=1).values.diff(dim=1) > 0).all().item())
    in_range = bool(((il >= 0) & (il < v)).all().item())
    pair = bool(torch.equal(t.input.gather(1, il), val)) if val is not None else True
    sel_min = t.input.gather(1, il).min(dim=1).values
    masked = t.input.scatter(1, il, float("-inf"))
    unsel_max = masked.max(dim=1).values
    contract = bool((sel_min >= unsel_max).all().item())

    sec = bench_by_cuda_events(run, num_warmups_each=3, num_runs_each=10)  # seconds
    us = sec * 1e6
    gbs = b * v * 2 / us / 1e3
    print(f"b{b}-v{v}-k{k}: {us:9.1f} us {gbs:8.1f} GB/s  "
          f"uniq={uniq} range={in_range} pair={pair} contract={contract}")
