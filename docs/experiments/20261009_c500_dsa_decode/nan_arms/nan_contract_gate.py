#!/usr/bin/env python3
"""The NaN contract, on the table's own NaN cases, through its own checks.

The correctness table carries NaN-bearing rows (`allow_nan` distributions,
`tests/test.py:267`), and `check_result` gates them ("NaN guard": a row whose
window holds a NaN must come back with `0x3F3F3F3F` in slot 0).  A 200-case
sample draws whatever it draws; this draws only the NaN distributions, so a
change to the NaN path is measured on the cases that exercise it.

    nan_contract_gate.py <tree> [n]      # n defaults to every NaN case

Runs `abort_when_nan_found=False` like the suite does -- the abort arm kills
the process and is checked separately (see nan_abort_probe.py).
"""
import random
import sys

import torch

TREE = sys.argv[1]
N = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")

import test as official                      # noqa: E402
import deep_select                           # noqa: E402

assert deep_select.__file__.startswith(TREE), deep_select.__file__

cases = official.correctness_cases_()
nan_cases = [c for c in cases
             if type(c.input_distrib).__name__ == "UintDistributionWithHotspotAndSpecifiedPivot"
             and c.input_distrib.allow_nan]
print(f"table: {len(cases)} cases, {len(nan_cases)} of them carry NaN")
drawn = (random.Random(20261009).sample(nan_cases, N) if N else nan_cases)

ok = bad = skipped = 0
for i, c in enumerate(drawn):
    try:
        is_correct = official.run_testcase(c, backend="maca_c")
    except torch.cuda.OutOfMemoryError:      # a case that does not fit is the
        print(f"[{i + 1}/{len(drawn)}] SKIP out of memory", flush=True)
        skipped += 1                         # environment, not a defect
        torch.cuda.empty_cache()
        continue
    except Exception as exc:                 # noqa: BLE001 -- report, keep going
        print(f"[{i + 1}/{len(drawn)}] CRASH {type(exc).__name__}: {exc}", flush=True)
        bad += 1
        continue
    tag = "pass" if is_correct else "CHECK_FAIL"
    print(f"[{i + 1}/{len(drawn)}] {tag}  b={c.batch_size} v={c.vocab_size} "
          f"k={c.topk} dtype={str(c.dtype).split('.')[-1]} "
          f"end={bool(c.enable_end_position)} "
          f"sorted=({c.sorted_value},{c.sorted_index}) "
          f"rv={c.return_value}", flush=True)
    ok += is_correct
    bad += not is_correct
print(f"NaN contract gate: {ok} pass, {bad} fail, {skipped} skipped "
      f"(out of memory), of {len(drawn)}")
sys.exit(1 if bad else 0)
