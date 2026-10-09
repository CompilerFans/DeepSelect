#!/usr/bin/env python3
"""Phase split over the DSA decode cells -- which kernel owns the time, and how
much of the wall clock no kernel owns at all.

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe.py

The cells are read out of `tests/test.py`'s `performance_cases()` rather than
restated here, so this probe cannot drift from the grid it is diagnosing.

Two clocks, and the columns say which is which:

* **per-kernel** -- `kk.bench`'s `BenchResult.time_ranges`, the same call the
  official harness makes (`tests/test.py:128`).  Printed once per matched name,
  never aggregated silently: this repo has been bitten by a substring filter
  that matched the wrong row while still printing a plausible number.
* **e2e** -- `scripts/perf_snapshot.py`'s `wall_clock_time`, events around the
  loop with the L2 flush *outside* the span.  `gap = e2e - sum(operator
  kernels)` is therefore launch, sync, and any kernel the name filter misses --
  it is the part of the operator the official clock cannot see.

Three filters, each for a reason this file can be checked against:

1. The 8 GB `zero_()` cold-L2 memset (`tests/kernelkit/bench.py:132`) runs
   inside `kk.bench`'s profiled range and is ~5.4 ms per run on this box.  It is
   not an operator kernel.  Excluded by name.
2. `nan_scan_kernel` does **not** contain "topk".  The chunked path launches it
   as a separate kernel (`maca_topk.cu:1720`) while the row path fuses the scan
   into pass 1, so a filter on "topk" alone silently drops it -- and silently
   changes which path looks expensive.  Both spellings count as operator.
3. Everything else that shows up is printed under `other`, so a kernel nobody
   anticipated is visible in the table rather than folded into `gap`.

`official_us` is the official clock's own answer, recomputed here to make the
gap between "what the harness records" and "what the operator costs" explicit.
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

import lib                                    # noqa: E402  (tests/lib.py)
import test as official                       # noqa: E402  (tests/test.py)
import kernelkit as kk                        # noqa: E402  (`kk.bench`, as tests/test.py)


def _load_perf_snapshot():
    """`scripts/perf_snapshot.py` as a module -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "ds_perf_snapshot", REPO / "scripts" / "perf_snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf_snapshot = _load_perf_snapshot()

# The C500 streaming-read wall, measured (`C500-to-parity-plan.zh.md` §7.1).
# Per-part, not a constant: a C600U is 1,545.
READ_WALL_GB_S = 1650.0

OPERATOR_MARKERS = ("topk", "nan_scan")


def is_operator_kernel(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in OPERATOR_MARKERS)


def short_name(name: str, tail: int = 34) -> str:
    """`void deep_select_maca::topk_kernel_radix<...>(RowParams)` ->
    `topk_kernel_radix<..., RowParams>`.

    The template arguments are kept: the block width is in them and it is what
    `radix_block_for` chose (`maca_topk.cu:642`).  Cut at the first `<`, not the
    last `::` -- the template arguments contain `::` too.
    """
    bare = name.strip()
    if bare.startswith("void "):
        bare = bare[5:]
    head, sep, args = bare.partition("<")
    head = head.rsplit("::", 1)[-1]
    if not sep:
        return head
    return f"{head}<...{args[-tail:]}" if len(args) > tail else f"{head}<{args}"


def route_of(names) -> str:
    """Which dataflow served the cell, from the kernel names alone.

    The two are told apart by names that exist at their launch sites and
    nowhere else: `topk_bf16_chunk_stage1_kernel*` is the split's
    (`radix_core.cuh:2238`), `topk_kernel_radix*` is the row kernel's
    (`maca_topk.cu:630`).
    """
    joined = " ".join(names).lower()
    if "chunk_stage1" in joined:
        return "split"
    if "radix" in joined:
        return "row"
    return "?"


def measure(p, num_tests: int, wall_iters: int):
    # `performance_cases()` leaves `seed=-1`, and `lib.generate_testcase` hands
    # it straight to `np.random.seed`, which refuses a negative.  The official
    # runner substitutes a global counter (`tests/test.py:169-171`); a counter
    # is the wrong thing here -- two A/B arms would then be timed on different
    # data.  The seed is pinned per cell instead, which is the same rule the
    # environment traps state for A/B ("pin the seed yourself").
    if p.seed == -1:
        p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2**31

    t = lib.generate_testcase(p)
    call = lambda: perf_snapshot.call_topk(p, t, "maca_c")   # noqa: E731

    # Warm up / check before timing, as the official runner does: a case that
    # selects wrong is a defect whatever it runs at.
    value, index = call()
    torch.cuda.synchronize()

    res = kk.bench(call, num_tests)
    # `_bench_kineto` stores `event.time_range.start / 1e6` (`bench.py:170`) and
    # torch's `time_range` is in microseconds, so these are **seconds**.  Read a
    # raw value before trusting a factor: the 8 GB cold-L2 memset in the same
    # trace must come out at ~5,360 us on this box (bench.py:132).
    spans = {}
    for name in res.get_kernel_names():
        total_s = sum(e - s for s, e in res.time_ranges[name])
        spans[name] = total_s / num_tests * 1e6          # seconds -> us

    wall_s = perf_snapshot.wall_clock_time(call, wall_iters)

    op_names = [n for n in res.get_kernel_names() if is_operator_kernel(n)]
    op_us = sum(spans[n] for n in op_names)
    tail_us = sum(us for n, us in spans.items() if not is_operator_kernel(n))
    total_bytes = official.topk_total_size(p, t, value, index)

    # The operator's own device span: first operator-kernel start to last
    # operator-kernel end, per run.  This is the e2e to compare `op_us` against
    # -- it is measured in `kk.bench`'s own cold-L2 regime, whereas
    # `wall_clock_time` does not flush between iterations and is a different
    # cache state.  The gap between them is what inter-kernel launch costs.
    span_us = res.get_e2e_time(op_names) * 1e6 if op_names else float("nan")

    # The official clock, recomputed from the same BenchResult: several "topk"
    # matches are the span over them, one is that kernel, none is None.
    official_names = [s for s in res.get_kernel_names() if "topk" in s.lower()]
    if len(official_names) == 1:
        official_us = spans[official_names[0]]
    elif official_names:
        official_us = res.get_e2e_time(official_names) * 1e6
    else:
        official_us = float("nan")

    return {
        "p": p, "spans": spans, "op_names": op_names,
        "op_us": op_us, "tail_us": tail_us, "official_us": official_us,
        "span_us": span_us, "gap_us": span_us - op_us,
        "wall_us": wall_s * 1e6, "bytes": total_bytes,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tests", type=int, default=10,
                    help="kk.bench num_tests; the grid's own num_runs is 10")
    ap.add_argument("--wall-iters", type=int, default=10)
    ap.add_argument("--device", type=int, default=None,
                    help="shorthand for CUDA_VISIBLE_DEVICES=N")
    ap.add_argument("--only", default=None,
                    help="comma-separated V values to restrict to")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    torch.set_default_device("cuda")

    cells = [c for c in official.performance_cases()
             if c.batch_size in (1, 64, 128) and str(c.dtype) == "torch.bfloat16"]
    if args.only:
        keep = {int(v) for v in args.only.split(",")}
        cells = [c for c in cells if c.vocab_size in keep]
    if not cells:
        raise SystemExit("probe.py: no DSA cells matched")

    print(f"# read wall = {READ_WALL_GB_S:.0f} GB/s (C500, measured streaming)")
    print(f"# cells = {len(cells)}  device = {torch.cuda.get_device_name()}"
          f"  sm_count = {torch.cuda.get_device_properties(0).multi_processor_count}")
    print()

    rows = []
    for p in cells:
        r = measure(p, args.num_tests, args.wall_iters)
        r["k"] = p.topk
        r["b"] = p.batch_size
        r["V"] = p.vocab_size
        rows.append(r)

        # Every matched name, printed before anything is summed -- a filter that
        # silently picks the wrong row still prints a plausible total.
        print(f"--- b{p.batch_size} V{p.vocab_size} k{p.topk}  "
              f"[{route_of(r['op_names'])}]")
        for n in sorted(r["spans"]):
            mark = "op " if n in r["op_names"] else "   "
            print(f"    {mark}{short_name(n):<44} {r['spans'][n]:>10.2f} us")
        print(f"    op_sum={r['op_us']:.2f}  span={r['span_us']:.2f}  "
              f"gap={r['gap_us']:.2f}  official={r['official_us']:.2f}  "
              f"wall={r['wall_us']:.2f}  non_op_in_bench={r['tail_us']:.2f} us")

    print()
    print("=" * 104)
    print(f"{'b':>5} {'V':>9} {'k':>5} {'route':>6} | "
          f"{'op_us':>9} {'span_us':>9} {'gap%':>6} | "
          f"{'span_GB/s':>10} {'wall%':>6} | {'wall_GB/s':>10} {'wall%':>6}")
    print("-" * 104)
    for r in rows:
        op, span, wall = r["op_us"], r["span_us"], r["wall_us"]
        span_bw = r["bytes"] / span / 1e3 if span else float("nan")
        wall_bw = r["bytes"] / wall / 1e3 if wall else float("nan")
        print(f"{r['b']:>5} {r['V']:>9} {r['k']:>5} {route_of(r['op_names']):>6} | "
              f"{op:>9.2f} {span:>9.2f} {100*r['gap_us']/span:>5.1f}% | "
              f"{span_bw:>10.1f} {100*span_bw/READ_WALL_GB_S:>5.1f}% | "
              f"{wall_bw:>10.1f} {100*wall_bw/READ_WALL_GB_S:>5.1f}%")
    print()
    print(f"# op_sum  = sum of names containing {OPERATOR_MARKERS} (case-insensitive).")
    print("# span_us = first operator-kernel start to last end, per run (kk.bench's")
    print("#           own cold-L2 regime).  gap% = (span - op_sum)/span.")
    print("# wall_us = wall_clock_time, which does NOT flush between iterations; it")
    print("#           is a different cache state and is here for reference only.")
    print("# official_us = the harness's own clock (topk-matching names only).")
    print(f"# GB/s is logical single-pass ({'bytes/(us*1e3)'}); wall = "
          f"{READ_WALL_GB_S:.0f} GB/s.")


if __name__ == "__main__":
    main()
