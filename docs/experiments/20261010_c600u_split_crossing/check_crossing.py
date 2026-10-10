"""Correctness of the cells the C600U split-crossing rule re-routes.

The moved cells now take the row kernel where they took the split; the
still-split and still-row cells must be unchanged.  Everything here goes
through the suite's own `run_testcase` -- `check_result` plus
`check_call_contract`, no copy.
"""
import sys

sys.path.insert(0, "/root/DeepSelect")
sys.path.insert(0, "/root/DeepSelect/tests")

import torch

torch.set_default_device("cuda")
import lib
import test as official

MOVED = [                      # the split's band on C600U: now the row kernel
    (8, 262144, 512), (12, 262144, 1024), (10, 524288, 512),
    (24, 524288, 1024), (16, 1048576, 512), (32, 1048576, 1024),
]
STILL_SPLIT = [                # inside the new boundary: unchanged
    (4, 262144, 512), (6, 524288, 1024), (10, 1048576, 512),
]
STILL_ROW = [                  # outside both rules: unchanged
    (16, 262144, 512), (64, 262144, 512), (48, 1048576, 512),
]

bad = 0
n = 0
for tag, cells in (("moved", MOVED), ("split", STILL_SPLIT), ("row", STILL_ROW)):
    for b, v, k in cells:
        n += 1
        p = lib.TestParam(b, v, k, False, False, False, torch.bfloat16,
                          torch.int32, num_runs=5)
        p.seed = (b * 1_000_003 + v * 11 + k) % 2 ** 31
        try:
            ok = official.run_testcase(p, "maca_c")
        except Exception as e:                      # noqa: BLE001
            ok = False
            print(f"  EXC {type(e).__name__}: {str(e)[:120]}")
        print(f"[{tag}] b{b} v{v} k{k}: {'PASS' if ok else 'FAIL'}", flush=True)
        if not ok:
            bad += 1
print(f"\n{n - bad}/{n} passed")
sys.exit(1 if bad else 0)
