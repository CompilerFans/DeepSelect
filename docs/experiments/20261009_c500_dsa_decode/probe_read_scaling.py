#!/usr/bin/env python3
"""What rate does this machine read at, at *these* sizes?

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe_read_scaling.py

`ledger` §4's CTA curve (4096 -> 1646.4 GB/s, 104 -> 921.3) was taken reading
4.295 GB.  DSA decode cells read 2 MB to 134 MB -- 32x to 2000x less -- and
every rate this campaign has measured on them lands near a third of the wall.
A rate taken on a long read does not say what a short one costs, so the curve
has to be re-taken at the size the campaign actually runs at.

No new kernel is needed for that.  `nan_scan_kernel` is a **pure one-pass read**
of the row (bit-pattern NaN test, one `__syncthreads_or`, one `atomicOr` per
CTA), it is on the production path, and its grid is `B * chunks` with
`chunks = 16` fixed by `wave_filled_chunks(16, sm_count)`.  Sweeping `b` at a
fixed `vocab_size` therefore sweeps bytes-read and CTA count together, in the
ratio the split actually uses -- 16 CTAs and 2.10 MB per row.

`lengths` is the full row, so the bytes are `b * vocab * 2` and nothing is
hidden by a window.  The linear fit's intercept is the part of the time that is
not streaming; its slope is the machine's marginal read rate.

This measures the *floor* the radix walks are failing to reach: whatever comes
out here, no row walk on the same data can beat it.
"""

import argparse
import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import lib                                    # noqa: E402
import test as official                       # noqa: E402
import kernelkit as kk                        # noqa: E402


def _load_perf_snapshot():
    spec = importlib.util.spec_from_file_location(
        "ds_perf_snapshot", REPO / "scripts" / "perf_snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf_snapshot = _load_perf_snapshot()

READ_WALL_GB_S = 1650.0      # C500, measured streaming read (`C500-to-parity-plan.zh.md` §7.1)

# Every `b` here is inside the landed gate (`b <= vocab_size / 20000` = 52 at
# V=1048576), so every cell takes the split and therefore launches
# `nan_scan_kernel` on `b * 16` CTAs.
BATCHES = [1, 2, 3, 6, 12, 26, 52]
VOCAB = 1048576
TOPK = 512


def measure(p, t, num_tests):
    call = lambda: perf_snapshot.call_topk(p, t, "maca_c")   # noqa: E731
    call()
    torch.cuda.synchronize()
    res = kk.bench(call, num_tests)
    spans = {}
    for name in res.get_kernel_names():
        total_s = sum(e - s for s, e in res.time_ranges[name])
        spans[name] = total_s / num_tests * 1e6       # `time_range/1e6` = seconds
    return spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tests", type=int, default=10)
    ap.add_argument("--device", type=int, default=None)
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    torch.set_default_device("cuda")
    print(f"# read wall = {READ_WALL_GB_S:.0f} GB/s (C500, measured streaming)")
    print(f"# device = {torch.cuda.get_device_name()}"
          f"  sm_count = {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"# V={VOCAB} k={TOPK} bf16; one row = {VOCAB*2/1e6:.1f} MB,"
          f" chunks = 16 -> nan_scan grid = b*16 CTAs")
    print()

    rows = []
    for b in BATCHES:
        p = official.TestParam(b, VOCAB, TOPK, False, False, False,
                               torch.bfloat16, torch.int32, num_runs=10)
        p.seed = (b * 1_000_003 + VOCAB * 11 + TOPK) % 2**31
        t = lib.generate_testcase(p)
        spans = measure(p, t, args.num_tests)
        ns = next((us for n, us in spans.items() if "nan_scan" in n.lower()), None)
        if ns is None:
            print(f"  b={b:<3} no nan_scan (did not take the split)")
            continue
        mb = b * VOCAB * 2 / 1e6
        gbs = mb * 1e3 / ns
        rows.append((b, b * 16, mb, ns, gbs))
        print(f"  b={b:<3} ctas={b*16:<5} bytes={mb:>7.2f} MB  "
              f"nan_scan={ns:>8.2f} us  {gbs:>7.1f} GB/s  "
              f"{100*gbs/READ_WALL_GB_S:>5.1f}% wall")

    if len(rows) >= 2:
        # Least squares on (MB, us).  The slope is the marginal rate -- what an
        # extra megabyte costs once the machine is running -- and the intercept
        # is everything that is paid once per launch/run.
        n = len(rows)
        sx = sum(r[2] for r in rows)
        sy = sum(r[3] for r in rows)
        sxx = sum(r[2] * r[2] for r in rows)
        sxy = sum(r[2] * r[3] for r in rows)
        den = n * sxx - sx * sx
        slope = (n * sxy - sx * sy) / den if den else float("nan")
        icept = (sy - slope * sx) / n
        marginal = 1e3 / slope if slope else float("nan")
        print()
        print(f"  fit: us = {icept:.2f} + {slope:.4f} * MB")
        print(f"       intercept {icept:.2f} us is paid per run, not per byte;")
        print(f"       slope   {slope:.4f} us/MB = {marginal:.1f} GB/s marginal"
              f" ({100*marginal/READ_WALL_GB_S:.1f}% wall)")
        print("  (marginal here is a 2-parameter fit over the b sweep; read it as")
        print("   the machine's steady-state rate at this size, not as a wall.)")


if __name__ == "__main__":
    main()
