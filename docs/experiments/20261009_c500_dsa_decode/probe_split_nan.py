"""The NaN contract on the split, now that stage 1 is the scan.

`nan_scan_kernel` is gone from the 16-bit split: stage 1 votes a row's flag while
it histograms the chunk, and the entry the contract half writes the sentinel
from is the same table.  That is a *window* claim -- the vote covers exactly the
bytes stage 1 walks -- so this puts one NaN per case at three positions (the
first element of the row, the middle of a middle chunk, and the last element of
the last chunk) and asks the operator's own answer for each: with
`check_nan=True` the row must carry `0x3F3F3F3F` in column 0, and with
`check_nan=False` it must not (a NaN is then ranked like any other key).

    CUDA_VISIBLE_DEVICES=2 python3 probe_split_nan.py
"""
import os
import sys

import torch

TREE = os.environ.get("DS_TREE", os.path.abspath(os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")))
sys.path.insert(0, TREE)
sys.path.insert(0, os.path.join(TREE, "tests"))
torch.set_default_device("cuda")
import lib                                                          # noqa: E402
import deep_select                                                  # noqa: E402

SENTINEL = 0x3F3F3F3F
B, V, K = 6, 262144, 512            # `chunked_bf16_applies`: 6 <= 262144/20000
CHUNKS = 16
CHUNK = (V + CHUNKS - 1) // CHUNKS // 8 * 8


def call(p, t, check_nan):
    return deep_select.topk(
        t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
        indices_type=p.out_idx_dtype, sorted_index=False, hint=None,
        output_idx=None, output_idx_offset=t.output_idx_offset,
        idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,
        return_value=True, abort_when_nan_found=False, check_nan=check_nan,
        backend="maca_c")


# One position per chunk kind: the row's first element (chunk 0), the middle of
# chunk 7, and the last element the last chunk covers (`length - 1`).
POSITIONS = [0, 7 * CHUNK + CHUNK // 2, V - 1]
failures = []
for pos in POSITIONS:
    for check_nan in (True, False):
        p = lib.TestParam(B, V, K, False, False, True, torch.bfloat16,
                          torch.int32, num_runs=0)
        p.seed = 20261010
        t = lib.generate_testcase(p)
        t.input[3, pos] = float("nan")
        got = call(p, t, check_nan)
        val, idx = (got if isinstance(got, (tuple, list)) else (None, got))
        row0 = int(idx[3, 0])
        flagged = row0 == SENTINEL
        want = check_nan
        ok = flagged == want
        # The other five rows hold no NaN and must be unaffected either way; the
        # NaN row's remaining slots are undefined, so only column 0 is read.
        others_ok = all(int(idx[r, 0]) != SENTINEL for r in range(B) if r != 3)
        print(f"pos={pos:>7} chunk={pos // CHUNK:>2} check_nan={int(check_nan)}  "
              f"row3 col0 = {row0:#010x}  flagged={flagged} want={want}  "
              f"others untouched={others_ok}")
        if not (ok and others_ok):
            failures.append((pos, check_nan))

print(f"\n{len(POSITIONS) * 2 - len(failures)}/{len(POSITIONS) * 2} "
      f"positions answered the contract")
sys.exit(1 if failures else 0)
