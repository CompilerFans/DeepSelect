#!/usr/bin/env python3
"""`abort_when_nan_found=True` must print a message and kill the process.

The suite never takes this arm -- `tests/test.py`'s call site writes
`abort_when_nan_found=False` (`test.py:194`) -- so the abort disposition has no
test anywhere in the tree.  It is checked here instead: row 1 carries a NaN,
row 0 is clean, and the process must die with the message on stderr and the
trapping kernel named.

    nan_abort_probe.py <tree>

Success is `rc != 0` plus, in the captured output, both the message and
`trapping: kernelName: ...nan_abort_kernel...`.  Falling through to the
"returned normally" branch prints what it returned, which is the failure: the
trap is a device-side abort, so a clean return means nothing raised it.
"""
import sys

import torch

TREE = sys.argv[1]
sys.path.insert(0, TREE)
torch.set_default_device("cuda")

import deep_select                            # noqa: E402

assert deep_select.__file__.startswith(TREE), deep_select.__file__

b, v, k = 2, 512, 8
x = torch.zeros((b, v), dtype=torch.bfloat16)
x[1, 100] = float("nan")                      # row 1 has a NaN, row 0 is clean
print("PROBE: calling abort_when_nan_found=True on a NaN row", flush=True)
out = deep_select.topk(x, k, abort_when_nan_found=True, backend="maca_c")
torch.cuda.synchronize()
print("PROBE FAILED: returned normally --", flush=True)
print("  ", [o.tolist() if o is not None else None for o in out], flush=True)
sys.exit(1)
