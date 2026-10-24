#!/usr/bin/env python3
"""The official correctness table, sampled under a size cap -- the gate a
delivery can finish in an hour, as distinct from the full-table gate.

    python tools/sample_gate.py --sample 6000 --seed 20261024
    CUDA_VISIBLE_DEVICES=1 python tools/sample_gate.py --sample 200 --max-elems 1048576

**Why a cap.**  `tests/test.py --sample N` draws uniformly from the whole
table, whose mean case is ~1e9 elements (~4 GB of fp32): the harness's checks
touch the input several times (a clone, a full-tensor mask, `isnan`), so a
uniform 8,000-case draw measures ~0.14 cases/s on a C600U -- sixteen hours,
which is not a gate anyone runs before shipping.  Cases above `--max-elems`
are therefore counted and reported (not silently dropped), and what remains is
still the whole table's *shape* coverage below that size: every dtype, index
dtype, ordering combination, `end`/offset/hotspot/fill axis, batch and topk.

**What it is not.**  Every case is `tests/test.py`'s own -- the table is
`correctness_cases_()`, the call and every assertion are `run_testcase`, so
this drives the official machinery rather than a copy of it (the rule the old
`scripts/official_slice.py` was deleted for breaking).  The full table remains
the pre-release gate (`tests/test.py --sample 1000000 -rf`, ~16 h recorded on
C500); this one is what each delivery candidate runs.

Data is deterministic: the table is built under `random.seed(--seed)` exactly
as `tests/test.py` builds it, the draw is `random.Random(--seed).sample`, and
each drawn case gets a fixed `p.seed` so the same command re-runs the same
data.  (The official loop leaves `seed=-1` cases to a process counter, which
makes a re-run different data for the same case -- fine for a sweep, wrong for
a release record.)

**Which package is under test** follows the rule every runner here keeps, and
that rule has one definition (`tools/provenance.py`): `$REPO` goes on
`sys.path` only when the checkout has a built extension, so a develop tree
tests itself and a tree with none tests the installed wheel -- the same
`deep_select` a caller would import.  The header line names the package and the
artifact md5, so the run is a record of *which binary* passed, not just of a
count.
"""
import argparse
import importlib.util
import os
import random
import sys
import time

import provenance

REPO = provenance.REPO


def load_official():
    """`tests/test.py` under its own name, so the stdlib `test` package cannot
    shadow it and nothing here re-implements its table or its checks."""
    provenance.use_checkout_if_built(with_tests=True)
    path = os.path.join(REPO, "tests", "test.py")
    spec = importlib.util.spec_from_file_location("ds_official_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample", type=int, default=6000,
                    help="cases to draw from the size-capped table")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-elems", type=int, default=1 << 26,
                    help="batch * vocab ceiling per case (default 2^26 = "
                         "67,108,864 elements = 256 MB of fp32)")
    ap.add_argument("--backend", default="maca_c")
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES shorthand")
    ap.add_argument("--run-to-finish", action="store_true",
                    help="keep going after the first wrong answer or crash "
                         "(the default stops, as `tests/test.py` does without "
                         "`-rf`)")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    # Before the import: the path decides which package answers.
    provenance.use_checkout_if_built(with_tests=True)
    import torch
    import deep_select
    official = load_official()
    torch.set_default_device("cuda")

    # The table, built exactly as `tests/test.py` builds it under the same seed.
    random.seed(args.seed)
    table = official.correctness_cases_()
    under = [p for p in table if p.batch_size * p.vocab_size <= args.max_elems]
    rng = random.Random(args.seed)
    drawn = rng.sample(under, min(args.sample, len(under)))
    for p in drawn:
        p.seed = rng.randrange(1, 1 << 31)

    print(f"table {len(table)} cases, {len(under)} at <= {args.max_elems} elements "
          f"({len(table) - len(under)} above the cap, not run); "
          f"drew {len(drawn)} under seed {args.seed}, backend {args.backend}")
    print(provenance.package_receipt(deep_select))
    print(f"device {torch.cuda.get_device_name(0)}")

    status = {"pass": [], "check_fail": [], "crash": [], "skip": [],
              "unsupported": []}
    started = time.time()
    stopped_early = None

    def context_alive():
        try:
            torch.cuda.synchronize()
            return True
        except Exception:  # noqa: BLE001 - any failure means no device
            return False

    for i, p in enumerate(drawn):
        print("================")
        print(f"[{i + 1:6d}/{len(drawn):6d}, {i / len(drawn) * 100:2.0f}%] ", end="")
        try:
            is_correct = official.run_testcase(p, backend=args.backend)
        except torch.cuda.OutOfMemoryError as exc:
            status["skip"].append((p, f"out of memory: {str(exc)[:120]}"))
            torch.cuda.empty_cache()
            print("    SKIPPED: out of memory", flush=True)
            continue
        except deep_select.UnsupportedByBackend as exc:
            status["unsupported"].append((p, str(exc)[:120]))
            print(f"    UNSUPPORTED: {str(exc)[:120]}", flush=True)
            continue
        except Exception as exc:  # noqa: BLE001 - a crash is a status
            status["crash"].append((p, f"{type(exc).__name__}: {str(exc)[:200]}"))
            print(f"    CRASHED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            if not context_alive():
                stopped_early = (f"the device stopped responding at case "
                                 f"{i + 1}/{len(drawn)}")
                break
            if not args.run_to_finish:
                stopped_early = "stopping at the first crash"
                break
            continue
        status["pass" if is_correct else "check_fail"].append((p, ""))
        if not is_correct:
            print("    SELECTED WRONG", flush=True)
            if not args.run_to_finish:
                stopped_early = "stopping at the first case that selected wrong"
                break

    elapsed = time.time() - started
    attempted = sum(len(v) for v in status.values())
    print(f"\n{'=' * 64}")
    for name in ("pass", "check_fail", "crash", "skip", "unsupported"):
        print(f"  {name:<12} {len(status[name]):>7}")
    if stopped_early:
        print(f"  (run stopped early: {stopped_early})")
    for name in ("check_fail", "crash", "skip", "unsupported"):
        for p, why in status[name]:
            print(f"  {name:<12} {p}" + (f"\n{'':<15}[{why}]" if why else ""))
    print(f"  {attempted} attempted in {elapsed:.0f} s "
          f"({attempted / elapsed:.2f} cases/s)")
    if status["check_fail"] or status["crash"]:
        print(f"\033[31m\033[1m{len(status['check_fail'])} case(s) selected "
              f"wrong, {len(status['crash'])} crashed\033[0m")
        return 1
    print(f"\033[32m\033[1mAll {len(status['pass'])} cases passed\033[0m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
