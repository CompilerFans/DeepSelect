#!/usr/bin/env python3
"""Paired A/B of two kernel artifacts, in one session.

`compare_snapshots.py` compares two *recorded* runs, and a recorded run is a
session: two snapshots taken hours apart on the same box differ by whatever the
box was doing in between.  Measured 2026-09-18 on a C500, comparing the 09-15
baseline against a 09-16 snapshot: `maca_c` median +0.53% and `torch` median
+1.2%, with `torch` regressing on 26 cells and improving on **none** -- the
control backend moving in one direction is the session, not the code.  A
per-cell tolerance cannot tell the two apart, so that comparator's verdict is
only as good as the two sessions were similar.

This tool removes the session from the comparison instead of correcting for it:
both artifacts run **in one session, alternating**, so clock, thermals and
whatever else drifted land on both arms.  Each arm gets its own process, so the
binary each one loaded is unambiguous (`lib` in the output is the md5 of the
file that process resolved), and the reported number is the median of the
per-round paired ratios.

It also records **which route served each cell** -- `c12` / `split` / `row`, by
kernel name.  Two arms can agree on every clock and still have made different
decisions, and a route that moved is a finding the timings alone would hide.

`--abort-when-nan-found` chooses the call: unset leaves the signature's default
(`True`), `no` is what every caller in this tree passes.  The child reports the
choice it *actually used* alongside the library it loaded, so the header and the
run cannot disagree.  Each run is reported twice -- the call's total, and the
same total with the abort follower kernel subtracted -- because "what the call
costs" and "what the selection kernels cost" are different questions.

Usage:
    tools/ab_snapshot.py --arm A=/path/to/deep_select --arm B=/path/to/deep_select \\
                         --snapshot perf_data/MetaX_C500/<run> [--rounds 3]
                         [--iters 10] [--device 0] [--out DIR]
                         [--abort-when-nan-found no]

Each `--arm` is a directory holding a `deep_select` package with its extension
beside it (a checkout, or a copy of an installed one).  The cell list comes from
the snapshot's CSV, restricted to one backend (`--backend`, default `maca_c`),
so the two arms are measured on exactly the cases that were recorded.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)

# One child per (arm, round).  The arm is chosen by `sys.path`, and the child
# asserts which package it got rather than trusting the path.
MEASURE = r'''
import sys, json, os, hashlib
sys.path.insert(0, "__ARM__")
sys.path.insert(0, "__TESTS__")
import torch, kernelkit as kk, deep_select
from deep_select._binding import extension_path

assert os.path.dirname(deep_select.__file__) == os.path.join("__ARM__", "deep_select"), \
    "arm __ARM__ loaded " + deep_select.__file__
_p = extension_path("deep_select_maca")
LIB = hashlib.md5(open(_p, "rb").read()).hexdigest()[:8]

IX = {"int32": torch.int32, "int64": torch.int64}
DT = {"bfloat16": torch.bfloat16, "float32": torch.float32}
CELLS = json.loads(sys.argv[1])
# One allocation width for every cell, so the shapes compared share a layout --
# but it has to cover the widest cell asked for.  It read a fixed `524288`
# until 2026-10-08, and `[:, :L]` *clamps*: a `4096 x 1048576 k=512` row was
# silently measured at `L = 524288` (half the elements, and 1.93x the time the
# same binary takes at the real width).  Ratios between two arms survived that
# only while both arms routed the same way at the truncated width -- and the
# width is exactly what the route predicates read.
WIDTH = max(524288, max(c[1] for c in CELLS))
out = []
for bs, L, k, dt, ix in CELLS:
    torch.manual_seed(0)
    s = torch.randn(bs, WIDTH, dtype=torch.float32, device="cuda:0")[:, :L]
    if dt == "bfloat16":
        s = s.to(torch.bfloat16)
    s = s.contiguous()

    def fn():
        return deep_select.topk(s, k, return_value=False, backend="maca_c",
                                indices_type=IX[ix]__ABORT__)
    try:
        r = kk.bench(fn, int(sys.argv[2]))
        # `get_kernel_time` is in seconds; the ledger's tables are microseconds.
        d = {n: r.get_kernel_time(n) * 1e6 for n in r.get_kernel_names()
             if "DeviceSynchronize" not in n}
        # The operator's own selection kernels, by name.  `coarse12` has to be
        # here: it is a whole route's worth of work and is named neither "stage"
        # nor "radix", so a filter without it scores the route that *replaces*
        # the split as if it had done nothing.
        #
        # `nan` is the widened half of the rule and it is counted **separately**
        # as well as in the sum, because the two readings answer different
        # questions: the sum is what the call costs, the split is what the
        # selection kernels cost.  `tests/test.py:129`'s own rule is the narrow
        # one (`"topk" in name.lower()`), which does not match
        # `nan_abort_kernel` at all.
        tot = sum(t for n, t in d.items()
                  if any(x in n for x in ("stage", "nan", "radix", "coarse12")))
        abort_us = sum(t for n, t in d.items() if "nan_abort" in n)
        route = ("c12" if any("coarse12" in n and t > 0 for n, t in d.items())
                 else "split" if any("stage" in n for n in d)
                 else "row")
        out.append([bs, L, k, dt, ix, tot, route, tot - abort_us, abort_us])
    except Exception as e:
        out.append([bs, L, k, dt, ix, None, "EXC " + str(e)[:40]])
print("R " + json.dumps({"lib": LIB, "so": _p, "abort": "__ABORT_LABEL__",
                         "rows": out}))
'''


def run_arm(arm_dir, cells, iters, device, abort=None):
    # `__ABORT__` is the call site's keyword; `__ABORT_LABEL__` is what the
    # child *reports having done*, and the two are substituted together so a
    # caller cannot print one and run the other.
    label = "default" if abort is None else str(bool(abort))
    code = (MEASURE.replace("__ARM__", arm_dir)
                   .replace("__TESTS__", os.path.join(REPO, "tests"))
                   .replace("__ABORT__",
                            "" if abort is None
                            else f", abort_when_nan_found={bool(abort)}")
                   .replace("__ABORT_LABEL__", label))
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(device))
    p = subprocess.run([sys.executable, "-c", code, json.dumps(cells), str(iters)],
                       cwd=arm_dir, env=env, capture_output=True, text=True)
    for line in p.stdout.splitlines():
        if line.startswith("R "):
            return json.loads(line[2:])
    sys.stderr.write(p.stdout[-2000:] + p.stderr[-2000:] + "\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR",
                    help="one arm per flag; the first is the reference (A)")
    ap.add_argument("--snapshot", required=True,
                    help="a run_bench.sh / perf_snapshot.py directory whose CSV "
                         "names the cells to measure")
    ap.add_argument("--backend", default="maca_c")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="0")
    ap.add_argument("--abort-when-nan-found", default=None,
                    choices=("yes", "no"),
                    help="pass `abort_when_nan_found` to the call.  Unset leaves "
                         "it off the call, which is the signature's default "
                         "(True) -- the path whose second kernel `nan_abort_kernel` "
                         "the widened filter then counts.  `no` is what every "
                         "caller in this tree passes and what `tests/test.py`'s "
                         "own name filter measures.")
    ap.add_argument("--out", default=None, help="write the paired detail as JSON")
    a = ap.parse_args()
    abort = {"yes": True, "no": False, None: None}[a.abort_when_nan_found]

    arms = []
    for spec in a.arm:
        name, _, path = spec.partition("=")
        if not path or not os.path.isdir(os.path.join(path, "deep_select")):
            sys.exit(f"ab_snapshot.py: --arm {spec!r} is not a directory holding "
                     f"a deep_select package")
        arms.append((name, os.path.abspath(path)))
    if len(arms) < 2:
        sys.exit("ab_snapshot.py: at least two --arm flags are needed")

    csv_path = os.path.join(a.snapshot, "deepselect_perf.csv")
    cells = [[int(r["n_rows"]), int(r["n_cols"]), int(r["top_k"]),
              r["input_dtype"], r["index_dtype"]]
             for r in csv.DictReader(open(csv_path)) if r["backend"] == a.backend]
    if not cells:
        sys.exit(f"ab_snapshot.py: {csv_path} has no {a.backend} rows")
    print(f"# {len(cells)} cells from {csv_path}")
    print(f"# {len(arms)} arms x {a.rounds} rounds x {a.iters} iters, "
          f"device {a.device}", flush=True)

    acc = collections.defaultdict(list)
    libs = collections.defaultdict(set)
    aborts = collections.defaultdict(set)
    for rd in range(a.rounds):
        for name, path in arms:
            r = run_arm(path, cells, a.iters, a.device, abort)
            if r is None:
                print(f"# round {rd} arm {name}: FAILED", flush=True)
                continue
            acc[name].append(r["rows"])
            libs[name].add(r["lib"])
            aborts[name].add(r["abort"])
            print(f"# round {rd} arm {name} lib={r['lib']} abort={r['abort']} "
                  f"{r['so']}", flush=True)

    print()
    for name, _ in arms:
        print(f"# arm {name}: libs={sorted(libs[name])} "
              f"abort_when_nan_found={sorted(aborts[name])}")
    print()

    ref = arms[0][0]
    detail = collections.defaultdict(dict)
    for name, _ in arms[1:]:
        n = min(len(acc[ref]), len(acc[name]))
        if not n:
            continue
        for i in range(n):
            for x, y in zip(acc[ref][i], acc[name][i]):
                if x[5] is None or y[5] is None:
                    continue
                key = tuple(x[:5])
                detail[name].setdefault(key, {"a": [], "b": [], "a_main": [],
                                              "b_main": [], "route_a": set(),
                                              "route_b": set()})
                d = detail[name][key]
                d["a"].append(x[5]); d["b"].append(y[5])
                # The follower-free subtotal, on the rows that carry it.
                if len(x) > 7 and len(y) > 7:
                    d["a_main"].append(x[7]); d["b_main"].append(y[7])
                d["route_a"].add(x[6]); d["route_b"].add(y[6])
        # Two readings of the same paired samples.  `tot` is what the call
        # costs; `main` drops the abort follower, which is the narrower rule
        # `tests/test.py:129` measures by.  A cell whose ratio differs between
        # them is a cell where the follower is a real part of the answer.
        for ka, kb, label in (("a", "b", "all operator kernels"),
                              ("a_main", "b_main",
                               "selection kernels only (no follower)")):
            ratios = collections.defaultdict(list)
            for key, d in detail[name].items():
                if not d[ka] or not d[kb]:
                    continue                      # no subtotal in this reading
                ratios[tuple(sorted(d["route_b"]))].append(
                    statistics.median(d[kb]) / statistics.median(d[ka]))
            print(f"=== {name} vs {ref} — {label} ===")
            everything = []
            for route, v in sorted(ratios.items()):
                everything += v
                print(f"  route {route[0]:6} n={len(v):3}  median "
                      f"{statistics.median(v):.4f}x  [{min(v):.3f}, {max(v):.3f}]")
            print(f"  {'ALL':6} n={len(everything):3}  median "
                  f"{statistics.median(everything):.4f}x  "
                  f"[{min(everything):.3f}, {max(everything):.3f}]")
            print()

    if a.out:
        os.makedirs(a.out, exist_ok=True)
        json.dump({name: {str(k): {"a": v["a"], "b": v["b"],
                                   "a_main": v["a_main"], "b_main": v["b_main"],
                                   "route_a": sorted(v["route_a"]),
                                   "route_b": sorted(v["route_b"])}
                          for k, v in by_shape.items()}
                   for name, by_shape in detail.items()},
                  open(os.path.join(a.out, "ab_paired.json"), "w"), indent=1)
        print(f"# wrote {os.path.join(a.out, 'ab_paired.json')}")


if __name__ == "__main__":
    main()
