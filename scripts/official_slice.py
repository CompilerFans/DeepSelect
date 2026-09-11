"""Drive a slice of upstream DeepSelect's own test suite against this port.

`tests/test.py` builds its case table inside `__main__` and runs it in full --
105,138 correctness cases, hours of GPU time, much of it spent in the reference
`torch.topk` on tensors of several GB.  This driver reuses the official pieces
unchanged -- `lib.generate_testcase` for the input, the module-level
`run_testcase` from `tests/test.py` for the checks -- over a slice of that same
table:

    * the table is built by the loops in `tests/test.py` verbatim;
    * restricted to `batch_size * vocab_size <= 2**28`, which bounds the
      reference's cost without excluding any shape family;
    * uniformly sampled to `--sample` cases with a fixed seed (the official
      table itself uses an unseeded `random`, so it is not reproducible as
      written; seeding it is the only change to how the table is produced).

Nothing under `tests/` is modified: the table construction is copied into
`cases()` below, and the checks are the official ones, imported by path.

    PYTHONPATH=. python scripts/official_slice.py [--sample N] [--seed S]
"""

import argparse
import importlib.util
import os
import random
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, REPO)

import torch  # noqa: E402

torch.set_default_device("cuda")          # as `tests/test.py:167` does

import lib  # noqa: E402

spec = importlib.util.spec_from_file_location(
    "ds_official_test", os.path.join(REPO, "tests", "test.py"))
official = importlib.util.module_from_spec(spec)
spec.loader.exec_module(official)         # defines `run_testcase`, runs no table

from lib import (TestParam, NormalFloatDistribution,  # noqa: E402
                 UniformUIntDistribution,
                 UintDistributionWithHotspotAndSpecifiedPivot, get_fp_config)

ELEM_BUDGET = 2 ** 28


def cases():
    """`tests/test.py`'s correctness table, verbatim, minus the heavy shapes."""
    valid_sv_si_rv_combinations = [  # sv_si_rv: sorted_value, sorted_index, return_value
        (False, False, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, True, False),
    ]

    out = []
    for dtype in [torch.float, torch.bfloat16]:
        for out_idx_dtype in [torch.int32, torch.int64]:
            for sv, si, rv in valid_sv_si_rv_combinations:
                if sv and dtype == torch.bfloat16:
                    continue                      # sorted_value is fp32-only
                for b in [random.randint(1, 20), random.randint(1, 100), 121, 512, 4096]:
                    for vocab_size in [
                        1, random.randint(2, 500), 1602, 32768, 123245, 225467,
                        262144, 418673, 682965, 998123, 1048576, (1 << 23) - 1
                    ]:
                        if b == 4096 and vocab_size > 1048576:
                            continue              # To avoid OOM (upstream's reason)
                        for topk in [
                            1, random.randint(2, 500), 512, 1024, 1231, 2048,
                            2132, 2333, 4096
                        ]:
                            enable_end_position = topk % 2 == 1
                            for distrib in [
                                NormalFloatDistribution(post_proc_func)
                                for post_proc_func in [
                                    None,
                                    lambda x: x + (100 if enable_end_position else -100),
                                    lambda x: x * (100 if enable_end_position else 1 / 100),
                                    lambda x: x * (1000 if enable_end_position else 1 / 1000),
                                    lambda x: x * (10000 if enable_end_position else 1 / 10000),
                                ]
                            ] + [
                                UniformUIntDistribution(
                                    get_fp_config(dtype).negative_0_as_int if enable_end_position else get_fp_config(dtype).positive_0_as_int,
                                    get_fp_config(dtype).negative_inf_as_int if enable_end_position else get_fp_config(dtype).positive_inf_as_int
                                ),
                                UniformUIntDistribution(0x0, 0x1),
                                UniformUIntDistribution(0x0, 0x20),
                                UniformUIntDistribution(0x0, 0x1000),
                            ] + [
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).positive_nan, min(8, vocab_size))], True),
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).negative_nan, min(8, vocab_size))], True),
                            ]:
                                enable_output_idx_offset = b % 2 == 1
                                out.append(TestParam(
                                    b, vocab_size, topk, sv, si, rv, dtype,
                                    out_idx_dtype, enable_end_position,
                                    enable_output_idx_offset, num_runs=0,
                                    idx_oob_fill_value=-2000000 + vocab_size,
                                    input_distrib=distrib))
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200,
                        help="how many cases to draw from the table (default 200)")
    parser.add_argument("--seed", type=int, default=20260911)
    args = parser.parse_args()

    random.seed(args.seed)
    table = cases()
    light = [p for p in table if p.batch_size * p.vocab_size <= ELEM_BUDGET]
    sample = random.Random(args.seed).sample(light, min(args.sample, len(light)))
    print(f"official table: {len(table)} cases, {len(light)} within "
          f"{ELEM_BUDGET} elements; running {len(sample)} sampled with seed "
          f"{args.seed}", flush=True)

    passed, failed = 0, []
    started = time.time()
    for i, p in enumerate(sample, 1):
        torch.cuda.empty_cache()
        print(f"[{i:3d}/{len(sample)}]", flush=True)
        try:
            ok = official.run_testcase(p)
        except Exception as exc:                   # noqa: BLE001
            ok = False
            print(f"    raised {type(exc).__name__}: {exc}", flush=True)
        if ok:
            passed += 1
        else:
            failed.append(p)
            print(f"    FAILED: {p}", flush=True)

    print(f"\n{passed}/{len(sample)} passed, {len(failed)} failed, "
          f"{time.time() - started:.0f}s", flush=True)
    for p in failed[:20]:
        print(f"  failed: {p}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
