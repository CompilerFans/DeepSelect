#!/usr/bin/env python3
"""Clean input under the default abort settings must not abort, on every route.

`RowParams::nan_abort_flags` is never zeroed between calls: the row kernel
clears each row's entry as its CTA starts, so no call can read another call's
answer.  That is a claim about the *launch* -- `topk_kernel_radix` is one CTA
per row on every route -- and this gate is what makes it checkable rather than
asserted.  If a route ever answered rows without that kernel, its rows would
keep whatever the buffer held (a fresh `cudaMalloc`, or a previous call's
answer) and this gate would trap on a clean input.

    nan_flag_route_gate.py <tree> [n]

Two halves, because they cover different routes:

  * `n` cases drawn (seeded) from the correctness table's **non-NaN**
    distributions, run with `abort_when_nan_found=True` -- the default, and the
    arm neither suite takes (`tests/test.py:194` writes `False`).  These sweep
    dtypes, index widths, windows and both block widths.
  * the DSA decode shapes added to the perf grid, which are the ones that reach
    the split; `b=64 v=1048576 k=512` is inside `chunked_bf16_applies`.

A trap kills the process, so a run that reaches the summary at all is the pass
condition for the abort half; the results are additionally checked through the
suite's own `check_result`, so "did not trap" cannot be bought by answering
wrongly.

The last thing it prints is a **route receipt**: one benched call per shape
family, with the kernel names that ran.  Coverage by shape is only as good as
one's reading of the route gates, and this is what replaces that reading --
if the split did not run, the receipt says so.
"""
import random
import sys

import torch

TREE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 120
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")

import lib                                    # noqa: E402
import test as official                       # noqa: E402
import kernelkit as kk                        # noqa: E402
import deep_select                            # noqa: E402

assert deep_select.__file__.startswith(TREE), deep_select.__file__


def run(p, t):
    """The suite's own call, with the abort left at its default."""
    return deep_select.topk(
        t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
        indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
        output_idx=None, output_idx_offset=t.output_idx_offset,
        idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,
        return_value=p.return_value, backend="maca_c")


cases = official.correctness_cases_()
clean = [c for c in cases
         if not getattr(c.input_distrib, "allow_nan", False)
         and c.check_correctness]
rng = random.Random(20261009)
drawn = rng.sample(clean, N)
# The table's own seeds are the driver counter's and run past 2**32, which
# `torch.manual_seed` refuses; the suite reseeds every case the same way.
for p in drawn:
    p.seed = rng.randrange(0, 2 ** 31)
print(f"table: {len(cases)} cases, {len(clean)} of them NaN-free -> drew {len(drawn)}")

# The split's own shapes, which the table above may not reach.
dsa = [official.TestParam(b, v, k, False, False, False, torch.bfloat16,
                          torch.int32, num_runs=0)
       for k in (512, 1024)
       for b in (64, 128)
       for v in (262144, 1048576)]
for p in dsa:
    p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2 ** 31
print(f"plus {len(dsa)} DSA split shapes")

ok = bad = skipped = 0
for i, p in enumerate(drawn + dsa):
    try:
        t = lib.generate_testcase(p)
        val, idx = run(p, t)
        good = official.check_result(p, t, val, idx) if p.check_correctness else True
    except torch.cuda.OutOfMemoryError:
        print(f"[{i + 1}] SKIP out of memory", flush=True)
        skipped += 1
        torch.cuda.empty_cache()
        continue
    except Exception as exc:                  # noqa: BLE001 -- report, keep going
        print(f"[{i + 1}] CRASH {type(exc).__name__}: {exc}", flush=True)
        bad += 1
        continue
    print(f"[{i + 1}] {'pass' if good else 'CHECK_FAIL'}  "
          f"b={p.batch_size} v={p.vocab_size} k={p.topk} "
          f"{str(p.dtype).split('.')[-1]}", flush=True)
    ok += good
    bad += not good

print(f"abort-default route gate: {ok} pass, {bad} fail, {skipped} skipped "
      f"(out of memory), of {len(drawn) + len(dsa)}")

# ── route receipt ───────────────────────────────────────────────────────────
# One shape per route family, benched so the kernel names come from the
# profiler rather than from a reading of the gates.
receipts = [
    ("row (bf16, wide block)", 4096, 16384, 512, torch.bfloat16, torch.int32),
    ("row (bf16, long row)",      64, 1048576, 512, torch.bfloat16, torch.int32),
    ("split (bf16)",               6, 1048576, 512, torch.bfloat16, torch.int32),
    ("coarse12 / dgchunks (fp32)", 4096, 131072, 2048, torch.float32, torch.int32),
]
print("\nroute receipt (kernels by name, abort left at its default):")
for label, b, v, k, dt, it in receipts:
    p = official.TestParam(b, v, k, False, False, False, dt, it, num_runs=3)
    p.seed = (b * 1_000_003 + v * 11 + k) % 2 ** 31
    t = lib.generate_testcase(p)
    res = kk.bench(lambda: run(p, t), 3)
    names = sorted(n for n in res.get_kernel_names()
                   if "elementwise" not in n and "Fill" not in n)
    print(f"  b={b:5d} v={v:8d} k={k:5d} {label}")
    for n in names:
        print(f"      {n[:70]}")
    if not names:
        print("      (no operator kernel matched -- check the filter)")

sys.exit(1 if bad else 0)
