"""The split/row crossing on **C600U**, each route forced in its own arm tree.

`kChunkedVocabPerRow` (21000) is the one constant in `chunked_bf16_applies`
that is a *measured crossing* rather than a bound, and the crossing is a
property of when the row kernel alone fills the machine -- 208 CTA slots on a
104-AP C500, 56 on a 28-AP C600U.  It was fitted on C500.

    row arm   `chunked_bf16_applies` returns false
    split arm `kChunkedVocabPerRow` widened to 2000

Same shape and same reason as the C500 campaign's `probe_crossing.py`; the
cell list is widened downward because a smaller machine fills at a lower
batch.  Timing is `tests/test.py`'s own rule.
"""
import argparse
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = "/root/DeepSelect"
TESTS = os.path.join(REPO, "tests")

CELLS = {
    262144:  [2, 4, 6, 8, 10, 12, 16, 20, 24],
    524288:  [4, 6, 8, 10, 12, 16, 20, 24, 32],
    1048576: [6, 8, 10, 12, 16, 20, 24, 32, 40, 48],
}

CHILD = r'''
import sys, json, hashlib, os
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
    us, _ = official.bench_topk(call, p, t, None, None)
    res = kk.bench(call, p.num_runs)
    names = [n for n in res.get_kernel_names() if "DeviceSynchronize" not in n]
    route = ("split" if any("stage" in n for n in names) else "row")
    out.append([B, V, K, us * 1e6, route])
print("R " + json.dumps({"md5": hashlib.md5(open(so, "rb").read()).hexdigest()[:8],
                         "rows": out}))
'''


def run(tree, cells, device):
    code = (CHILD.replace("__ARM__", os.path.abspath(tree))
                 .replace("__TESTS__", TESTS))
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
    ap.add_argument("--device", default="0")
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
            receipt[name] = r["md5"]
            print(f"# round {rd} arm {name} md5={r['md5']}", flush=True)

    for name in acc:
        print(f"#   {name:>5}: md5 {receipt[name]}")

    def med(name, i):
        return statistics.median(rows[i][3] for rows in acc[name])

    print(f"\n{'V':>9} {'k':>5} {'b':>4} {'row us':>10} {'split us':>10} {'row/split':>10}  route")
    cur_v = None
    for i, (b, v, k) in enumerate(cells):
        if v != cur_v:
            cur_v = v
            print()
        r, s = med("row", i), med("split", i)
        got = acc["split"][0][i][4]
        print(f"{v:>9} {k:>5} {b:>4} {r:>10.1f} {s:>10.1f} {r / s:>10.3f}  {got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
