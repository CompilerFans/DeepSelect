"""The bf16 row's block width on **C600U**: 512-thread vs 1024-thread instantiation.

`radix_block_for` picks the 1024-thread row for `L >= 60000` (`needs_long_row_bf16`,
"the measured crossover is specific to the 16 KiB C500 row variant") and 512
everywhere else.  On C600U every skip-regime cell (v >= 65536) takes the 1024
arm, and those are exactly the cells whose C600U/C500 ratio (1.8-2.2) sits far
above the short-row cells' (1.3-1.5) -- so the crossover is the suspect.

Three arms, one session, alternating rounds, one process per (arm, round):

    parent  /root/DeepSelect            the shipped choice
    b512    radix_block_for -> 512      the whole bf16 grid on the narrow row
    b1024   radix_block_for -> 1024     the whole bf16 grid on the wide row

The child prints the launched row kernel's name, so the arm's instantiation is
a receipt in the output rather than an assumption about the patch.
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

CELLS = [
    # long rows -- the 1024 arm today; the skip regime where C600U loses most
    (4096, 1048576, 512), (4096, 524288, 512), (4096, 262144, 512),
    (4096, 131072, 1024), (4096, 65536, 512), (4096, 65536, 1024),
    (768, 1048576, 512), (512, 262144, 512), (256, 65536, 512),
    (64, 1048576, 512),
    # short rows -- the 512 arm today
    (4096, 16384, 512), (4096, 16384, 1024), (4096, 4096, 512),
    (768, 16384, 512), (4096, 1024, 512), (256, 1024, 512),
]

CHILD = r'''
import sys, json, hashlib, os
sys.path.insert(0, "__ARM__")
sys.path.insert(0, "__TESTS__")
import torch
torch.set_default_device("cuda")
import lib, test as official, kernelkit as kk, deep_select
from deep_select._binding import extension_path
assert os.path.dirname(deep_select.__file__) == os.path.join("__ARM__", "deep_select"), \
    deep_select.__file__
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
    row = [n for n in names if "topk_kernel_radix" in n]
    kern = (row[0] if row else (names[0] if len(names) == 1 else ""))
    out.append([B, V, K, us * 1e6, kern[:150]])
    del t
    torch.cuda.empty_cache()
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
    sys.stderr.write(f"{tree} FAILED\n{p.stdout[-2000:]}{p.stderr[-2000:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", action="append", required=True, metavar="NAME=DIR")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    arms = [tuple(x.split("=", 1)) for x in a.arm]
    cells = list(CELLS)

    acc = {n: [] for n, _ in arms}
    receipt = {}
    for rd in range(a.rounds):
        order = arms if rd % 2 == 0 else arms[::-1]
        for name, tree in order:
            r = run(tree, cells, a.device)
            if r is None:
                return 1
            acc[name].append(r["rows"])
            receipt[name] = r["md5"]
            print(f"# round {rd} arm {name} md5={r['md5']}", flush=True)

    for name in acc:
        print(f"#   {name:>7}: md5 {receipt[name]}")

    def med(name, i):
        return statistics.median(rows[i][3] for rows in acc[name])

    base = arms[0][0]
    print(f"\nbase arm = {base}")
    hdr = f"{'b':>5} {'V':>8} {'k':>5} | " + " ".join(f"{n:>10}" for n, _ in arms) \
          + " | " + " ".join(f"{n}/base".rjust(10) for n, _ in arms[1:])
    print(hdr)
    cur_v = None
    for i, (b, v, k) in enumerate(cells):
        if v != cur_v:
            cur_v = v
            print()
        vals = [med(n, i) for n, _ in arms]
        ratios = [f"{vals[j] / vals[0]:10.3f}" for j in range(1, len(arms))]
        print(f"{b:>5} {v:>8} {k:>5} | " + " ".join(f"{x:10.1f}" for x in vals)
              + " | " + " ".join(ratios))
    print("\nlaunched row kernel (arm, first cell):")
    for n, _ in arms:
        print(f"  {n:>7}: {acc[n][0][0][4]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
