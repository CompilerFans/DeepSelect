"""The 16-bit row's skip tables, against the contract and against themselves.

Two questions, two parts.

**Part 1 -- the contract**, on every official cell the tables are switched on for
(bf16, `vocab_size >= kSkipMinVocab`): run the case through
`tests/test.py`'s own `run_testcase` with `num_runs=0`, so the checks are the
official ones (index range, uniqueness, `value == input[index]`,
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
"""
import argparse
import hashlib
import sys

import torch

TREE = "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect"
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib                                                          # noqa: E402
import test as official                                             # noqa: E402
import deep_select                                                  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("--differential", action="store_true")
parser.add_argument("--cells", type=int, default=0,
                    help="cap part 1 at this many cells (0 = all)")
args = parser.parse_args()

B, V, K = 4096, 131072, 512
DIFF_SHAPES = [(B, V, K), (B, 262144, 512), (512, 131072, 1024)]

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
