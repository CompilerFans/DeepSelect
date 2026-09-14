#!/usr/bin/env python3
"""Compare two `perf_snapshot.py` runs, cell by cell.

Rows are matched on the *measured* cell, not on their position:
`(case_group, family, n_rows, n_cols, top_k, sorted_value, input_dtype,
index_dtype)`.  A cell that appears in one run and not the other is reported as
added/removed rather than silently dropped -- a run that measured fewer cells
must not read as "no change".

The comparison is per backend, and only for rows that are *measured on both
sides with the same status*.  An empty `time(us)` is "not measured", never
zero: `deep_gemm` is `unsupported` on every bf16 cell, and `torch` is untimed
where no kernel name matched, so comparing those against a number would
manufacture a win out of a refusal.  A `status` that changed between the two
runs is called out separately -- `pass` -> `fail` is a regression whatever the
clock says, and it is not a timing row.

The reference is this repository's own kernel: `relative_pct_vs_maca_c` is
defined there as 100%, so **>100% means that backend is faster than `maca_c`**
(`maca_c_us / that_us * 100`).  A `torch` row moving is informative; a `maca_c`
row moving is the regression the exit status reports.

What the exit status means: **1** when a `maca_c` cell moved beyond `--tol` or a
`maca_c` cell's `status` changed (either direction -- `unsupported` -> `pass` is
an improvement worth seeing, `pass` -> `fail` is a regression); **0** otherwise,
including when a `torch` or `deep_gemm` row moved.

Usage:
    tools/compare_snapshots.py <candidate_dir> --base <baseline_dir>
                               [--tol 0.03] [--top 25]

Exit status: 0 when every backend is within tolerance, 1 when `maca_c` has at
least one regression or one status change.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from typing import Dict, List, Optional, Tuple

KEY = ("case_group", "family", "n_rows", "n_cols", "top_k", "sorted_value",
       "input_dtype", "index_dtype")
# The column a run is compared on.  `relative_pct_vs_maca_c` is a derived
# number and `speedup_vs_torch` depends on another backend's row, so both are
# recomputed from the raw times rather than diffed.
REQUIRED = "time(us)"


def load(directory: str) -> Dict[Tuple, Dict[str, Dict[str, str]]]:
    """Every row of every CSV in the directory, keyed by (cell, backend).

    A CSV that predates the `case_group` column keys its rows by an empty group
    rather than failing: this tool's job is to compare two runs, and an older
    run is the most useful baseline there is.  Keys are read with `setdefault`,
    so an older file is matched on the columns it does have.
    """
    files = sorted(f for f in os.listdir(directory) if f.endswith(".csv"))
    if not files:
        raise SystemExit(f"compare_snapshots: no *.csv in {directory}")
    rows: Dict[Tuple, Dict[str, Dict[str, str]]] = {}
    for f in files:
        with open(os.path.join(directory, f)) as fh:
            for r in csv.DictReader(fh):
                for k in KEY:
                    r.setdefault(k, "")
                cell = tuple(r[k] for k in KEY)
                backend = r.get("backend", "")
                if backend not in rows.setdefault(cell, {}):
                    rows[cell][backend] = r
    return rows


def label(cell: Tuple) -> str:
    group, fam, n_rows, n_cols, top_k, _sv, dtype, _idx = cell
    return f"{group[:8]:<8} {fam[:6]:<6} {dtype:<8} b{n_rows}-v{n_cols}-k{top_k}"


def compare_one(base: Dict[str, str], cand: Dict[str, str], tol: float):
    """(base_us, cand_us, delta, verdict) -- or None when not comparable."""
    b_us = base.get(REQUIRED, "")
    c_us = cand.get(REQUIRED, "")
    if not b_us or not c_us:
        return None
    b, c = float(b_us), float(c_us)
    if b <= 0:
        return None
    delta = (c - b) / b
    verdict = ("regressed" if delta > tol
               else "improved" if delta < -tol else "noise")
    return b, c, delta, verdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("candidate")
    ap.add_argument("--base", required=True)
    ap.add_argument("--tol", type=float, default=0.03,
                    help="fraction beyond which a delta is called a change "
                         "(default 0.03; the official harness re-run, same "
                         "binary and same device, was measured at a median "
                         "0.25%% and a max 1.3%% per cell above 100us)")
    ap.add_argument("--top", type=int, default=25,
                    help="how many cells to print per backend, by |delta|")
    ap.add_argument("--chip", default="")
    args = ap.parse_args()

    base = load(args.base)
    cand = load(args.candidate)
    only_cand = sorted(set(cand) - set(base))
    only_base = sorted(set(base) - set(cand))
    shared = sorted(set(base) & set(cand))

    print(f"# base      {args.base}  ({len(base)} cells)")
    print(f"# candidate {args.candidate}  ({len(cand)} cells)")
    print(f"# matched   {len(shared)}   only-in-candidate {len(only_cand)}   "
          f"only-in-base {len(only_base)}   tolerance ±{args.tol * 100:.1f}%")
    if only_cand:
        print("\n# cells the candidate added (cannot be compared):")
        for k in only_cand:
            print(f"#   + {label(k)}")
    if only_base:
        print("\n# cells the baseline had and the candidate does not -- "
              "a coverage LOSS, not a speedup:")
        for k in only_base:
            print(f"#   - {label(k)}")

    failed = False
    for backend in ("maca_c", "torch", "deep_gemm"):
        results: List[Tuple] = []
        skipped: List[str] = []
        status_changes: List[Tuple] = []
        for cell in shared:
            b_row = base[cell].get(backend)
            c_row = cand[cell].get(backend)
            if b_row is None or c_row is None:
                skipped.append(f"{label(cell)} (absent on one side)")
                continue
            b_st, c_st = b_row.get("status", ""), c_row.get("status", "")
            if b_st != c_st:
                status_changes.append((cell, b_st, c_st))
                continue
            got = compare_one(b_row, c_row, args.tol)
            if got is None:
                skipped.append(f"{label(cell)} (not timed: {c_st or '?'})")
                continue
            results.append((cell, *got))

        print(f"\n=== {backend} ===")
        for cell, b_st, c_st in status_changes:
            print(f"    STATUS    {label(cell):<40} {b_st} -> {c_st}")
            if backend == "maca_c":
                failed = True
        if not results:
            print(f"    no comparable cells ({len(skipped)} skipped: "
                  f"{'; '.join(skipped[:3])}{' ...' if len(skipped) > 3 else ''})")
            continue
        n_reg = sum(1 for r in results if r[4] == "regressed")
        n_imp = sum(1 for r in results if r[4] == "improved")
        n_noise = len(results) - n_reg - n_imp
        tot_b = sum(r[1] for r in results)
        tot_c = sum(r[2] for r in results)
        print(f"    {n_reg} regressed / {n_imp} improved / {n_noise} noise   "
              f"({len(skipped)} not comparable)")
        print(f"    total {tot_b:.1f} -> {tot_c:.1f} us "
              f"({(tot_c - tot_b) / tot_b * 100:+.2f}%)")
        shown = [r for r in sorted(results, key=lambda r: -abs(r[3]))[:args.top]
                 if r[4] != "noise"]
        if not shown:
            print("    (no cell beyond tolerance)")
        for cell, b, c, d, v in shown:
            print(f"    {v:<9} {label(cell):<40} {b:10.1f} -> {c:10.1f} us  "
                  f"{d * 100:+7.1f}%")
        if n_reg and backend == "maca_c":
            failed = True

    if failed:
        print(f"\n# VERDICT: REGRESSED (maca_c beyond ±{args.tol * 100:.1f}% or a "
              f"status change)")
        return 1
    print(f"\n# VERDICT: OK (maca_c within ±{args.tol * 100:.1f}% and no status "
          f"change)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
