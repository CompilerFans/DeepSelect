"""Drive a slice of upstream DeepSelect's own test suite against this port.

`tests/test.py` builds its case table inside `__main__` and runs all of it
(105,138 correctness cases, hours of GPU time).  This driver reuses the official
pieces unchanged -- `lib.generate_testcase` for the input, its module-level
`run_testcase` for the checks -- over a slice: the table loops verbatim,
restricted to `batch_size * vocab_size <= 2**28`, then uniformly sampled to
`--sample` cases with a fixed seed (the official table uses an unseeded
`random`, so seeding it is the only change to how the table is produced).

Nothing under `tests/` is modified: the construction is copied into `cases()`
below and the checks are imported by path.  The one thing the official suite
cannot express is a backend choice -- it calls `deep_select.topk(...)` with no
`backend=` -- so `--backend` pins that call instead of editing the official file
(see `_bind_backend`).

    PYTHONPATH=. python scripts/official_slice.py [--sample N] [--seed S]
                                                 [--backend {maca_c,torch,deep_gemm}]
                                                 [--default-arm NAME]

`--backend` pins the call -- "is the kernel right?" -- and therefore says
nothing about the default.  `--default-arm` leaves the official call site
unmodified and changes only what a bare call resolves to (DS_TOPK_BACKEND): the
path a caller who names no backend actually takes.  Run both when the default
changes.
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

import deep_select  # noqa: E402
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


def _bind_backend(original, backend):
    """Pin every `deep_select.topk` the official suite makes to one backend.

    `tests/test.py` resolves the name at call time, so rebinding the module
    attribute is enough -- no official file changes.  Without `--backend` the
    library's own choice stands, exactly as the official suite runs it.
    """
    def bound(*a, **kwargs):
        kwargs.setdefault("backend", backend)
        return original(*a, **kwargs)
    return bound


def _set_default_backend(backend):
    """Point the *unpinned* call at a backend, through the library's own knob.

    The counterpart of `_bind_backend` and deliberately not the same mechanism:
    this sets `DS_TOPK_BACKEND` and leaves `deep_select.topk` alone, so the
    official call site is unmodified and what runs is the real default path.
    """
    os.environ["DS_TOPK_BACKEND"] = backend


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", type=int, default=200,
                        help="how many cases to draw from the table (default 200)")
    parser.add_argument("--seed", type=int, default=20260911)
    parser.add_argument("--backend", default=None,
                        choices=["maca_c", "torch", "deep_gemm"],
                        help="implementation under test; unset leaves the "
                             "library's own choice, which is what the official "
                             "suite runs with unmodified")
    parser.add_argument("--default-arm", default=None,
                        choices=["maca_c", "torch", "deep_gemm"],
                        help="leave the official call site unpinned and set "
                             "the *process default* instead "
                             "(DS_TOPK_BACKEND); this is the arm that tests "
                             "the default, as opposed to --backend, which "
                             "tests one implementation")
    parser.add_argument("--shard", default=None, metavar="I/N",
                        help="run every N-th case of the sample, offset I "
                             "(0-based): N processes together cover the whole "
                             "sample, which is how a full-table run is kept "
                             "down to one process's wall clock")
    args = parser.parse_args()

    if args.backend is not None and args.default_arm is not None:
        # Combining them would report one answer under the other's name:
        # `--backend` pins the call, so the process default is never consulted
        # and the `--default-arm` result would be a lie.
        raise SystemExit("official_slice.py: --backend pins the call and "
                         "--default-arm sets what an unpinned call resolves "
                         "to; pass one, not both")
    if args.backend is not None:
        deep_select.topk = _bind_backend(deep_select.topk, args.backend)
    elif args.default_arm is not None:
        _set_default_backend(args.default_arm)
        print(f"unpinned arm: DS_TOPK_BACKEND={args.default_arm} "
              f"(the official call site is unmodified)", flush=True)

    random.seed(args.seed)
    # The host repo's own correctness shapes come first and are not sampled:
    # they are the rows the `deep_gemm` backend has a kernel for, so without
    # them a `--backend deep_gemm` run compares nothing (same reason
    # `perf_snapshot.py` carries the host perf shapes).  Still element-budgeted.
    host = [p for p in official.host_selector_correctness_cases()
            if p.batch_size * p.vocab_size <= ELEM_BUDGET]
    table = host + cases()
    light = [p for p in table if p.batch_size * p.vocab_size <= ELEM_BUDGET]
    sample = host + random.Random(args.seed).sample(
        light, min(args.sample, len(light)))
    if args.shard is not None:
        index, count = (int(part) for part in args.shard.split("/"))
        sample = sample[index::count]
    print(f"official table: {len(table)} cases, {len(light)} within "
          f"{ELEM_BUDGET} elements; running {len(sample)} "
          f"({len(host)} host-selector + "
          f"{len(sample) - len(host)} sampled) with seed {args.seed}"
          + (f", shard {args.shard}" if args.shard else ""), flush=True)

    passed, failed, skipped = 0, [], 0
    started = time.time()
    for i, p in enumerate(sample, 1):
        torch.cuda.empty_cache()
        print(f"[{i:3d}/{len(sample)}]", flush=True)
        try:
            ok = official.run_testcase(p)
        except deep_select.UnsupportedByBackend as exc:
            # A backend narrower than the operator (deep_gemm answers fp32 only)
            # says so explicitly; that is a coverage gap, not a wrong answer, so
            # it does not fail the run.
            skipped += 1
            print(f"    unsupported: {exc}", flush=True)
            continue
        except Exception as exc:                   # noqa: BLE001
            ok = False
            print(f"    raised {type(exc).__name__}: {exc}", flush=True)
        if ok:
            passed += 1
        else:
            failed.append(p)
            print(f"    FAILED: {p}", flush=True)

    print(f"\n{passed}/{len(sample) - skipped} passed, {skipped} unsupported, "
          f"{len(failed)} failed, {time.time() - started:.0f}s", flush=True)
    for p in failed[:20]:
        print(f"  failed: {p}", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
