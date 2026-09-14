#!/usr/bin/env python3
"""Compare two `perf_snapshot.py` output directories, cell by cell.

Rows are matched on the *case*, not on their position: `(family, n_rows,
n_cols, seq_len, top_k, input_dtype, index_dtype)`.  A shape that appears in
one directory and not the other is reported as added/removed rather than
silently dropped -- a run that measured fewer cells must not read as "no
change".

Every arm is compared independently (`maca_c`, `torch`, `deep_gemm`), and an
empty field is "not measured", never "zero": `deep_gemm` refuses bf16 and
sorted cells, and `torch` is empty where no kernel name matched (see the
snapshot's own notes).  Comparing those as 0 would manufacture a 100% win.

Exit status: 0 when every arm is within tolerance, 1 when `maca_c` has at
least one regression.  (A `torch`/`deep_gemm` regression is reported but does
not fail the run -- those arms are not what this repository ships.)
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Tuple

KEY = ("family", "n_rows", "n_cols", "seq_len", "top_k", "input_dtype",
       "index_dtype")
ARMS = ("maca_c_us", "torch_us", "deep_gemm_us")


def load(directory: str) -> Dict[Tuple, Dict[str, str]]:
    """Every row of every `*_axes.csv` in the directory, keyed by case.

    Both files are read, not just the preferred one: a `--families` or
    `--full` run leaves one of them empty, and a directory whose official-axes
    file happens to be empty is not a directory with nothing to compare.  When
    the same case appears in both, the first file wins (they are the same
    measurement on the same binary; a duplicate key would be a bug in the
    snapshot, and silently taking the later one would hide it).
    """
    files = [f for f in ("deepselect_official_axes.csv",
                         "deepselect_deep_gemm_axes.csv")
             if os.path.exists(os.path.join(directory, f))]
    if not files:
        raise SystemExit(f"compare_snapshots: no *_axes.csv in {directory}")
    rows: Dict[Tuple, Dict[str, str]] = {}
    for f in files:
        for r in csv.DictReader(open(os.path.join(directory, f))):
            # `seq_len` was added after the first snapshots were written.  A
            # file without it means "the window is the whole row", which is
            # what the column defaults to -- so an old baseline still matches
            # rather than losing every row to a KeyError.
            if "seq_len" not in r:
                r["seq_len"] = r["n_cols"]
            key = tuple(r[k] for k in KEY)
            rows.setdefault(key, r)
    return rows


def label(k: Tuple) -> str:
    fam, n_rows, n_cols, seq_len, top_k, dtype, _idx = k
    s = f"{fam[:6]:<6} {dtype:<8} b{n_rows}-v{n_cols}"
    if seq_len != n_cols:
        s += f"/win{seq_len}"
    return f"{s}-k{top_k}"


def compare_one(base_us: str, cand_us: str, tol: float):
    if not base_us or not cand_us:
        return None
    b, c = float(base_us), float(cand_us)
    if b <= 0:
        return None
    delta = (c - b) / b
    verdict = "regressed" if delta > tol else ("improved" if delta < -tol else "noise")
    return b, c, delta, verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tol", type=float, default=0.03,
                    help="fraction beyond which a delta is called a change "
                         "(default 0.03; the measured per-round ratio spread "
                         "on this kernel is <= 1.4%%)")
    ap.add_argument("--top", type=int, default=25,
                    help="how many rows to print per arm, by |delta|")
    ap.add_argument("--chip", default="")
    args = ap.parse_args()

    base = load(args.base)
    cand = load(args.candidate)

    only_cand = sorted(set(cand) - set(base))
    only_base = sorted(set(base) - set(cand))
    shared = sorted(set(base) & set(cand))

    print(f"# base      {args.base}  ({len(base)} rows)")
    print(f"# candidate {args.candidate}  ({len(cand)} rows)")
    print(f"# matched   {len(shared)}   only-in-candidate {len(only_cand)}   "
          f"only-in-base {len(only_base)}   tolerance ±{args.tol * 100:.1f}%")
    if only_cand:
        print("\n# rows the candidate added (cannot be compared):")
        for k in only_cand:
            print(f"#   + {label(k)}")
    if only_base:
        print("\n# rows the baseline had and the candidate does not -- "
              "a coverage LOSS, not a speedup:")
        for k in only_base:
            print(f"#   - {label(k)}")

    failed = False
    for arm in ARMS:
        results: List[Tuple] = []
        skipped = 0
        for k in shared:
            got = compare_one(base[k].get(arm, ""), cand[k].get(arm, ""), args.tol)
            if got is None:
                skipped += 1
                continue
            results.append((k, *got))
        if not results:
            print(f"\n=== {arm}: no comparable rows ({skipped} unmeasured on "
                  f"one side) ===")
            continue
        n_reg = sum(1 for r in results if r[4] == "regressed")
        n_imp = sum(1 for r in results if r[4] == "improved")
        n_noise = len(results) - n_reg - n_imp
        tot_b = sum(r[1] for r in results)
        tot_c = sum(r[2] for r in results)
        print(f"\n=== {arm}: {n_reg} regressed / {n_imp} improved / "
              f"{n_noise} noise   ({skipped} not measured on one side) ===")
        print(f"    total {tot_b:.1f} -> {tot_c:.1f} us "
              f"({(tot_c - tot_b) / tot_b * 100:+.1f}%)")
        # Ranked by |delta| so the one real regression is not buried in noise.
        shown = sorted(results, key=lambda r: -abs(r[3]))[:args.top]
        changed = [r for r in shown if r[4] != "noise"]
        if not changed:
            print("    (no cell beyond tolerance)")
        for k, b, c, d, v in changed:
            print(f"    {v:<9} {label(k):<34} {b:9.1f} -> {c:9.1f} us  "
                  f"{d * 100:+7.1f}%")
        if n_reg and arm == "maca_c_us":
            failed = True

    if failed:
        print("\n# VERDICT: REGRESSED (maca_c has at least one cell beyond "
              f"±{args.tol * 100:.1f}%)")
        return 1
    print(f"\n# VERDICT: OK (maca_c within ±{args.tol * 100:.1f}% on every "
          f"matched cell)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
