"""The split/row crossing, measured by forcing each route in its own arm tree.

`kChunkedVocabPerRow` admits the split while the rows the batch buys are worth
less than it pays -- so the constant has to be read off a curve, and neither
route is reachable at every batch from one tree: production admits the split
below `vocab / 20000` and the row kernel above it.  So each route is forced in
its own copy-and-point arm (the production tree is not edited):

    row arm   `chunked_bf16_applies` returns false   -> every batch is a row
    split arm `kChunkedVocabPerRow` widened to 2000  -> every batch is a split

    python3 probe_crossing.py --arm row=/tmp/gate_row --arm split=/tmp/gate_split

Timing is `tests/test.py`'s own rule (`official.bench_topk`).  Both routes fold
the NaN scan into their own walk as of `f7130c8`, so that rule's span covers
the operator's kernels on both sides and the two readings coincide here.
Each child prints the md5 of the `.so` it resolved and which kernels ran, so an
arm cannot silently measure the other route.
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

# Per row length, the batches around where the two curves are expected to meet.
# `vocab / 20000` -- the gate in force -- is 13 / 26 / 52 for these three.
CELLS = {
    262144:  [2, 4, 6, 8, 10, 12, 14, 16, 20, 24, 32],
    524288:  [6, 10, 14, 18, 22, 26, 30, 36, 44, 56],
    1048576: [12, 20, 28, 36, 44, 52, 60, 64],
}

CHILD = r'''
import sys, json, hashlib, os
sys.path.insert(0, "__ARM__")
sys.path.insert(0, "__TESTS__")
import torch
torch.set_default_device("cuda")
import lib, test as official, deep_select
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
    us, _ = official.bench_topk(call, p, t, None, None)
    out.append([B, V, K, us * 1e6])
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
    sys.stderr.write(f"{tree} FAILED\n{p.stdout[-1500:]}{p.stderr[-1500:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="2")
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in a.arm]
    cells = [(b, v, k) for v, bs in CELLS.items() for k in (512, 1024) for b in bs]

    acc = {n: [] for n, _ in arms}
    receipt = {}
    for rd in range(a.rounds):
        for name, tree in (arms if rd % 2 == 0 else arms[::-1]):
            r = run(tree, cells, a.device)
            if r is None:
                return 1
            acc[name].append(r["rows"])
            receipt[name] = (r["md5"], r["so"])
            print(f"# round {rd} arm {name} md5={r['md5']}", flush=True)

    for name in acc:
        print(f"#   {name:>5}: md5 {receipt[name][0]}  {receipt[name][1]}")
    cur_v = None
    hdr = f"\n{'V':>8} {'k':>5} {'b':>4} " + "".join(f"{n + ' us':>11}" for n, _ in arms)
    if len(arms) == 2:
        hdr += f"{'row/split':>10}"
    print(hdr)
    for i, (b, v, k) in enumerate(cells):
        if v != cur_v:
            cur_v = v
            print()
        med = {n: statistics.median(rows[i][3] for rows in acc[n]) for n, _ in arms}
        line = f"{v:>8} {k:>5} {b:>4} " + "".join(f"{med[n]:>11.1f}" for n, _ in arms)
        if len(arms) == 2:
            line += f"{med[arms[0][0]] / med[arms[1][0]]:>10.3f}"
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
