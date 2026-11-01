"""A production indexer's call shapes, through the suite's own predicate.

The shapes are the fp32 `topk_coarse12` call a DSA indexer makes: logits
`[rows=8192, cols=max_regions]` with `topk=512`, where `max_regions` has three
values in the trace (`10240 / 9216 / 8192`, from the helper kernel's
`grid[1] = 640 / 576 / 512` x `BLOCK_R=16`).

Those shapes are outside this repo's benchmark grid -- the largest fp32 case
there is 4096 x 129280, a row many times longer with half the batch -- so "does
DeepSelect serve this shape, and at what rate" could only be answered by
extrapolation.  This case closes that gap: it runs the three shapes and asks
`check_result`, the suite's predicate, rather than a hand-written comparison.

The number this prints is one call's time against the operand size, next to a
turnaround target in microseconds.
"""
# This file tests the checkout, not the installed wheel; see `_casebase.py`.
import time

import torch

from _casebase import suite, build_case, call_topk

# (rows, max_regions, label) -- the three shapes the trace carries.
SHAPES = ((8192, 10240, 'qpos=81920  widest / judge'),
          (8192, 9216, 'qpos=73728'),
          (8192, 8192, 'qpos=65536'))
TOPK = 512
TARGET_US = 800.0

print(f'MC3-16686 Step5 indexer shapes: rows x max_regions, topk={TOPK}, fp32')
allok = True
for rows, cols, label in SHAPES:
    p, t = build_case(rows, cols, TOPK)
    call_topk(p, t)                       # warm: JIT and allocator
    torch.cuda.synchronize()

    # The operand size of the widest shape, for a reader comparing against the
    # pure-read roofline.
    operand_mb = rows * cols * 4 / 1e6

    reps, best = 20, float('inf')
    for _ in range(reps):
        start = time.perf_counter()
        val, idx = call_topk(p, t)
        torch.cuda.synchronize()
        best = min(best, time.perf_counter() - start)

    suite.check_call_contract(p, val, idx)
    ok = suite.check_result(p, t,
                            val.clone() if val is not None else None, idx.clone())
    allok &= ok
    verdict = 'PASS' if ok else 'FAIL'
    beat = 'meets' if best * 1e6 <= TARGET_US else 'misses'
    print(f'  {label:<28} {rows} x {cols:<6} operand {operand_mb:6.1f} MB -> '
          f'{best * 1e6:8.1f} us ({operand_mb / best / 1e3:6.0f} GB/s) '
          f'{beat} the {TARGET_US:.0f} us target | {verdict}')

print('all ok' if allok else 'SOME CASES FAILED')
raise SystemExit(0 if allok else 1)
