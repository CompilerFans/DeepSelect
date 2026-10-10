"""Paired A/B over the cells the chunked split serves, one process per (arm, round).

The split's cells are the ones `chunked_bf16_applies` admits, and they are the
only cells a change to stage 1 can move -- so this runs those and nothing else,
alternating the arms inside one session so clock and thermals land on both.

    CUDA_VISIBLE_DEVICES=2 python3 probe_split_ab.py \
        --arm old=/tmp/ab_head --arm new=$PWD --rounds 3

Each child is one process, asserts which package it loaded, and prints the md5
of the `.so` it resolved, so an arm cannot silently measure the other's binary.

**Two readings per cell, because one of them cannot see the change.**  The
ledger's tables quote `tests/test.py`'s rule (`official.bench_topk`), which
takes the **span over kernels whose name contains `topk`** -- and the old arm
launches `nan_scan_kernel` *before* stage 1, so a fold that deletes the scan
moves nothing that rule counts while it does add the test to stage 1's walk.
The second reading is `ab_snapshot.py`'s "all operator kernels" (the sum over
names containing `stage` / `nan` / `radix` / `coarse12`), which is what the
call actually costs.  Both are printed per cell, with the matched names.
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

# The cells `chunked_bf16_applies` admits in the bf16 grid: `b <= vocab/20000`,
# `vocab >= 262144`, `topk in {512, 1024}`.  `b6 x 262144 k512` is the smallest
# and is also the one the split has always served; the `b=1` row is the DSA
# decode shape the campaign added to the grid.
CELLS = [(1, 262144, 512), (1, 262144, 1024), (1, 1048576, 512), (1, 1048576, 1024),
         (6, 262144, 512), (6, 262144, 1024), (6, 524288, 512), (6, 524288, 1024),
         (6, 1048576, 512), (6, 1048576, 1024)]

CHILD = r'''
import sys, json, os, hashlib
sys.path.insert(0, "__ARM__")
sys.path.insert(0, "__TESTS__")
import torch
torch.set_default_device("cuda")
import lib, test as official, kernelkit as kk, deep_select
from deep_select._binding import extension_path
assert os.path.dirname(deep_select.__file__) == "__ARM__/deep_select", deep_select.__file__
so = extension_path("deep_select_maca")
out = []
for B, V, K in json.loads(sys.argv[1]):
    p = lib.TestParam(B, V, K, False, False, False, torch.bfloat16, torch.int32,
                      num_runs=10)
    p.seed = (B * 1_000_003 + V * 11 + K) % 2 ** 31
    t = lib.generate_testcase(p)
    def call():
        return deep_select.topk(
            t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
            indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
            output_idx=None, output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value, abort_when_nan_found=False,
            backend="maca_c")
    call()
    # The ledger's rule, called rather than re-implemented: the span over
    # `"topk"`-named kernels.
    us, _ = official.bench_topk(call, p, t, None, None)
    # The operator's own cost: every selection kernel by name (`ab_snapshot.py`'s
    # widened rule, which is the one that counts `nan_scan_kernel`).
    res = kk.bench(call, p.num_runs)
    d = {n: res.get_kernel_time(n) * 1e6 for n in res.get_kernel_names()
         if "DeviceSynchronize" not in n}
    op = sum(x for n, x in d.items()
             if any(k in n for k in ("stage", "nan", "radix", "coarse12")))
    names = [n.split("(")[0].split("::")[-1][:34] for n in d
             if any(k in n for k in ("stage", "nan", "radix", "coarse12"))]
    route = ("c12" if any("coarse12" in n for n in d)
             else "split" if any("stage" in n for n in d) else "row")
    out.append([B, V, K, us * 1e6, op, route, names])
print("R " + json.dumps({"md5": hashlib.md5(open(so, "rb").read()).hexdigest()[:8],
                         "so": so, "rows": out}))
'''


def run(tree, cells, device):
    code = (CHILD.replace("__ARM__", os.path.abspath(tree))
                 .replace("__TESTS__", os.path.join(REPO, "tests")))
    p = subprocess.run([sys.executable, "-c", code, json.dumps(cells)],
                       cwd=tree, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device)))
    for line in p.stdout.splitlines():
        if line.startswith("R "):
            return json.loads(line[2:])
    sys.stderr.write(f"{tree} FAILED\n{p.stdout[-1200:]}{p.stderr[-1200:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="2")
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in a.arm]

    acc = {n: [] for n, _ in arms}
    receipt = {}
    for rd in range(a.rounds):
        order = arms if rd % 2 == 0 else arms[::-1]
        for name, tree in order:
            r = run(tree, CELLS, a.device)
            if r is None:
                return 1
            acc[name].append(r["rows"])
            receipt[name] = (r["md5"], r["so"])
            print(f"# round {rd} arm {name} md5={r['md5']}", flush=True)

    print("\n# cells = the split's own (b <= vocab/20000, vocab >= 262144):")
    for name in acc:
        print(f"#   {name:>4}: md5 {receipt[name][0]}  {receipt[name][1]}")
    for title, col in (("selection kernels only (tests/test.py rule)", 3),
                       ("all operator kernels (ab_snapshot rule)", 4)):
        print(f"\n# {title}:")
        hdr = f"{'B':>3} {'V':>8} {'k':>5} " + "".join(f"{n + ' us':>11}" for n, _ in arms)
        if len(arms) == 2:
            hdr += f"{'new/old':>9}"
        print(hdr)
        for i, (b, v, k) in enumerate(CELLS):
            med = {n: statistics.median(rows[i][col] for rows in acc[n])
                   for n, _ in arms}
            line = (f"{b:>3} {v:>8} {k:>5} "
                    + "".join(f"{med[n]:>11.1f}" for n, _ in arms))
            if len(arms) == 2:
                line += f"{med[arms[1][0]] / med[arms[0][0]]:>9.3f}"
            print(line)
    print("\n# kernels the widened reading summed, per cell (arm `%s`):"
          % arms[-1][0])
    for i, (b, v, k) in enumerate(CELLS):
        print(f"#   {b}x{v} k={k}: route={acc[arms[-1][0]][0][i][5]} "
              f"{acc[arms[-1][0]][0][i][6]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
