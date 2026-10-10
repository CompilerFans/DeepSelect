"""The 16-bit row's skip tables, against the contract and against themselves.

Two questions, two parts.

**Part 1 -- the contract**, on every official cell of the family the tables are
built for (bf16, `vocab_size >= kSkipMinVocab` -- the wider family, so the cells
the landed density half excludes are checked on the row walk here too): run the
case through `tests/test.py`'s own `run_testcase` with `num_runs=0`, so the
checks are the official ones (index range, uniqueness, `value == input[index]`,
`min(selected) >= max(unselected)`, the NaN guard, the orderings) and no timing
is taken.  A skipped block cannot hold a candidate -- the summary is that
block's max bin -- so the answer should be unchanged; this is where that has to
show rather than be argued.

**Part 2 -- the differential**, on `sorted_value` sequences.  The sets themselves
are not usable as a gate on bf16 (a row ties 7..44 elements at the k-th value, so
which of them is selected is unspecified -- see CLAUDE.md), but the *sequence of
the k largest values* is well defined whatever the tie-break, so the two arms
must print identical bytes there.  The knob is per-process
(`DEEP_SELECT_BF16_SKIP`), so the comparison is two runs of this script; the
driver prints a digest per case and `cmp`s them.

    DEEP_SELECT_BF16_SKIP=1 python check_skip.py --differential > on.txt
    DEEP_SELECT_BF16_SKIP=0 python check_skip.py --differential > off.txt
    cmp on.txt off.txt

**Part 3 -- the odd-length cells** (`--odd-ends`), and it is a different class
from part 1 rather than more of it.  The summary walk is block-granular, so a row
whose `length` is 8-aligned but not 64-aligned has a partial block that the
histogram's tail loop has to take; a tail that starts anywhere but the collect's
own `list_end` leaves those elements out of the count the stage is sized by, and
the collect's tail then finds them.  The perf grid cannot reach that: its bf16
cells either have a `vocab_size` that is not 8-aligned (scalar walk, no summary)
or one that is 64-aligned (no partial block).  The correctness table does reach
it, and only through `enable_end_position` cells, whose per-row lengths are what
makes `length % 64` land off zero.  This part runs a sample of those through the
official checks, which is where a lost or miscounted element shows up.
"""
import argparse
import hashlib
import os
import sys

import torch

TREE = os.environ.get(
    "DS_TREE", "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect")
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib                                                          # noqa: E402
import test as official                                             # noqa: E402
import deep_select                                                  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--differential", action="store_true")
parser.add_argument("--odd-ends", type=int, default=0, metavar="N",
                    help="part 3: this many odd-length cells from the "
                         "correctness table")
parser.add_argument("--cells", type=int, default=0,
                    help="cap part 1 at this many cells (0 = all)")
args = parser.parse_args()

B, V, K = 4096, 131072, 512
DIFF_SHAPES = [(B, V, K), (B, 262144, 512), (512, 131072, 1024)]

if args.odd_ends:
    # ── part 3: the rows whose length is 8-aligned but not 64-aligned ───────
    # One cell per distinct (batch, vocab, topk) that the tables are on for, so
    # the sample is a spread of shapes rather than repeats of one, and the
    # heaviest shapes first -- the failure this part exists for is a count that
    # goes wrong by a few elements, which a large row makes likelier rather
    # than rarer.
    seen, cells = set(), []
    for c in official.correctness_cases_():
        if (c.dtype != torch.bfloat16 or c.out_idx_dtype != torch.int32
                or not c.enable_end_position or c.vocab_size < 65536):
            continue
        key = (c.batch_size, c.vocab_size, c.topk)
        if key in seen:
            continue
        seen.add(key)
        cells.append(c)
    cells.sort(key=lambda c: -c.batch_size * c.vocab_size)
    cells = cells[:args.odd_ends]
    failed = []
    for p in cells:
        p.num_runs = 0
        p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2**31
        print(f"--- {p.batch_size}x{p.vocab_size} k={p.topk} "
              f"sv={p.sorted_value} si={p.sorted_index} rv={p.return_value} "
              f"distrib={type(p.input_distrib).__name__}")
        if not official.run_testcase(p, backend="maca_c"):
            failed.append(p)
    print(f"\n{len(cells) - len(failed)}/{len(cells)} odd-length cells "
          f"passed the official checks")
    sys.exit(1 if failed else 0)

if not args.differential:
    # ── part 1: the contract, on the cells the tables are on for ─────────────
    cells = [c for c in official.performance_cases()
             if str(c.dtype) == "torch.bfloat16" and c.vocab_size >= 65536]
    if args.cells:
        cells = cells[:args.cells]
    failed = []
    for p in cells:
        p.num_runs = 0                     # the checks, no timing
        p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2**31
        print(f"--- {p.batch_size}x{p.vocab_size} k={p.topk} "
              f"sv={p.sorted_value} si={p.sorted_index} rv={p.return_value}")
        if not official.run_testcase(p, backend="maca_c"):
            failed.append(p)
    print(f"\n{len(cells) - len(failed)}/{len(cells)} cells passed the official checks")
    sys.exit(1 if failed else 0)

# ── part 2: the value sequences, which the tie-break cannot move ─────────────
for (b, v, k) in DIFF_SHAPES:
    p = lib.TestParam(b, v, k, True, False, True, torch.bfloat16, torch.int32,
                      num_runs=0)
    p.seed = (b * 1_000_003 + v * 11 + k) % 2**31
    t = lib.generate_testcase(p)
    val, idx = deep_select.topk(t.input, k, sorted=True, begin=None, end=t.end,
                                indices_type=p.out_idx_dtype, sorted_index=False,
                                hint=None, output_idx=None,
                                output_idx_offset=t.output_idx_offset,
                                idx_oob_fill_value=p.idx_oob_fill_value,
                                value_oob_fill_value=p.value_oob_fill_value,
                                return_value=True, abort_when_nan_found=False,
                                backend="maca_c")
    digest = hashlib.sha256(val.view(torch.int16).cpu().numpy().tobytes()).hexdigest()
    # The contract still holds on this arm, and it is the half that would catch a
    # walk that lost elements (the sequence would be short or wrong).
    ok = official.check_result(p, t, val.clone(), idx.clone())
    print(f"{b}x{v} k={k}  value-sequence sha256 {digest[:32]}  contract {ok}")
    assert ok
