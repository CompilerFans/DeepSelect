#!/usr/bin/env python3
"""What does the production launch actually contain, and what does each cost?

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe_row_kernels.py

`bf16_atomic_price.cu` (the extraction microbench) runs the production pass-1
loop, verbatim, at the production geometry and reports **96.5% of the read
wall**.  The campaign's phase decomposition attributes 44% of the row kernel to
that loop at **60% of the wall**.  A loop cannot be both, so one of the two
numbers is not measuring what it says.

This probe prints every kernel the production call launches, by name and by
span, so the decomposition's "radix kernel" total can be checked against the
launch it came from rather than assumed.  It names the route too: a cell served
by the split launches `nan_scan` + `chunk_stage1` + `stage2`, a cell served by
the row launches one kernel.

`kk.bench` reports per-kernel `time_ranges`; a name that appears with more
launches than one (a template instantiation launched `B` times is one name with
`B` ranges) is reported as launches, total, and mean, because "one name" and
"one launch" are not the same thing and the difference is a factor of the grid.
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

READ_WALL_GB_S = 1650.0

# The b=4096 bf16 cells, and the shape the extraction microbench matched:
# one row per CTA, grid = n_rows.  `--cells` overrides, so the README figures'
# own bars can be checked one by one.
CELLS = [(4096, 1048576, 512), (4096, 524288, 512), (4096, 131072, 512),
         (64, 1048576, 512)]

# The two README figures' cells: bf16 over batch 6/512/4096 x the six vocab
# sizes at topk=512, and fp32 over batch 6..4096 at vocab_size=129280.
BF16_FIG = [(b, v, 512, "torch.bfloat16")
            for b in (6, 512, 4096)
            for v in (16384, 65536, 131072, 262144, 524288, 1048576)]
FP32_FIG = [(b, 129280, 512, "torch.float32")
            for b in (6, 256, 512, 768, 4096)]


def short_name(name, tail=30):
    bare = name.strip()
    if bare.startswith("void "):
        bare = bare[5:]
    head, sep, args = bare.partition("<")
    head = head.rsplit("::", 1)[-1]
    if not sep:
        return head
    return f"{head}<...{args[-tail:]}" if len(args) > tail else f"{head}<{args}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tests", type=int, default=10)
    ap.add_argument("--device", type=int, default=None)
    ap.add_argument("--figures", action="store_true",
                    help="the cells the README's two figures draw")
    ap.add_argument("--dtype", default="torch.bfloat16",
                    help="with --cells, the dtype of every listed cell")
    ap.add_argument("--cells", default=None,
                    help="'b,v,k;b,v,k' -- overrides the built-in list")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    torch.set_default_device("cuda")
    print(f"# read wall = {READ_WALL_GB_S:.0f} GB/s (C500, measured streaming)")
    print(f"# device = {torch.cuda.get_device_name()}"
          f"  sm_count = {torch.cuda.get_device_properties(0).multi_processor_count}")
    print()

    if args.cells:
        cells = []
        for part in args.cells.split(";"):
            b, v, k = (int(x) for x in part.split(","))
            cells.append((b, v, k, args.dtype))
    elif args.figures:
        cells = BF16_FIG + FP32_FIG
    else:
        cells = [(b, v, k, "torch.bfloat16") for b, v, k in CELLS]

    for want in cells:
        hit = [c for c in official.performance_cases()
               if (c.batch_size, c.vocab_size, c.topk) == want[:3]
               and str(c.dtype) == want[3]]
        if not hit:
            # a figure cell the *perf grid* does not carry: build it by hand
            hit = [official.TestParam(want[0], want[1], want[2], False, False,
                                      False, eval(want[3]), torch.int32,
                                      num_runs=10)]
        if not hit:
            print(f"{want}: not in the grid"); continue
        p = hit[0]
        p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2**31
        t = lib.generate_testcase(p)

        call = lambda: perf_snapshot.call_topk(p, t, "maca_c")   # noqa: E731
        call()
        torch.cuda.synchronize()
        res = kk.bench(call, args.num_tests)

        b, v, k = p.batch_size, p.vocab_size, p.topk
        width = 4 if "float32" in str(p.dtype) else 2
        row_mb = b * v * width / 1e6
        print(f"=== {p.dtype} b{b} V{v} k{k}  ({row_mb:.1f} MB)")
        total = 0.0
        for name in sorted(res.get_kernel_names()):
            ranges = res.time_ranges[name]
            us = sum(e - s for s, e in ranges) / args.num_tests * 1e6
            total += us
            print(f"    {short_name(name):<52} launches={len(ranges):<4} "
                  f"{us:>10.2f} us  {us/len(ranges):>9.2f} us/launch  "
                  f"{(row_mb*1e3/us if us else 0):>7.1f} GB/s (1 pass)")
        print(f"    {'SUM of kernels':<52} {'':<13} {total:>10.2f} us")
        print()


if __name__ == "__main__":
    main()
