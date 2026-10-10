"""The NaN check's two arms on **C600U**, at the call site (no rebuilds).

Two separate switches, measured separately, on both routes the grid uses:

  1. `check_nan` -- whether the row is scanned at all.  On the row path the
     scan is folded into pass 1 (`fold_nan`), on the bf16 split into stage 1's
     load, so its cost is inside the selection kernels' own time on C600U (the
     separate `nan_scan_kernel` belongs to the f32-chunks route, which the
     `sm_count == 104` gate closes here).
  2. `abort_when_nan_found` -- the disposition.  It is its own kernel
     (`nan_abort_kernel`, launched after), which the official `"topk" in name`
     rule does not match; this probe reports it as its own line, and the
     operator reading (all kernels) beside the official one.

Arms are call arguments, one child process per (arm, round), alternating:

    on      check_nan=True,  abort=False    what the official grid runs
    off     check_nan=False, abort=False    the scan's cost
    abort   check_nan=True,  abort=True     + the follower's launch

`abort` is skipped on a cell whose data holds a NaN (the kernel would trap);
the official distributions do not, and the probe asserts that rather than
trusting it.
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
    (4096, 1048576, 512), (4096, 65536, 512), (768, 1048576, 512),
    (4096, 16384, 512),
    (1, 1048576, 512), (6, 262144, 512),          # the split route
    (4096, 131072, 2048), (6, 129280, 512),       # fp32
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
CHECK, ABORT = json.loads(sys.argv[2])
out = []
for B, V, K, DT in json.loads(sys.argv[1]):
    dt = torch.bfloat16 if DT == "bfloat16" else torch.float32
    p = lib.TestParam(B, V, K, False, False, False, dt, torch.int32, num_runs=10)
    p.seed = (B * 1_000_003 + V * 11 + K) % 2 ** 31
    t = lib.generate_testcase(p)
    has_nan = bool(torch.isnan(t.input).any().item())
    if ABORT and has_nan:
        out.append([B, V, K, DT, None, None, None, None, "SKIP-NAN"])
        del t; torch.cuda.empty_cache(); continue
    def call():
        return deep_select.topk(
            t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
            indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
            output_idx=None, output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value, abort_when_nan_found=ABORT,
            check_nan=CHECK, backend="maca_c")
    call()
    us, _ = official.bench_topk(call, p, t, None, None)
    res = kk.bench(call, p.num_runs)
    # the operator's own kernels by name, the `tools/ab_snapshot.py` filter --
    # a bare "every name in the profile" walk hits runtime-API events
    # (`mcLaunchKernel`) whose run count is not a multiple of `num_tests`.
    d = {n: res.get_kernel_time(n) * 1e6 for n in res.get_kernel_names()
         if "DeviceSynchronize" not in n
         and any(x in n for x in ("stage", "nan", "radix", "coarse12", "topk"))}
    op = sum(d.values())
    nan = sum(t for n, t in d.items() if "nan" in n)
    route = ("split" if any("stage" in n for n in d) else "row")
    out.append([B, V, K, DT, us * 1e6, op, nan, route, has_nan])
    del t
    torch.cuda.empty_cache()
print("R " + json.dumps({"md5": hashlib.md5(open(so, "rb").read()).hexdigest()[:8],
                         "rows": out}))
'''


def run(arm_tree, cells, check, abort, device):
    code = (CHILD.replace("__ARM__", os.path.abspath(arm_tree))
                 .replace("__TESTS__", TESTS))
    p = subprocess.run([sys.executable, "-c", code, json.dumps(cells),
                        json.dumps([check, abort])],
                       cwd=arm_tree, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device)))
    for line in p.stdout.splitlines():
        if line.startswith("R "):
            return json.loads(line[2:])
    sys.stderr.write(f"{arm_tree} FAILED\n{p.stdout[-2000:]}{p.stderr[-2000:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=REPO)
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    arms = [("on", True, False), ("off", False, False), ("abort", True, True)]
    cells = [(b, v, k, "bfloat16" if k in (512, 1024) and v != 129280 else "float32")
             for b, v, k in CELLS]

    acc = {n: [] for n, _, _ in arms}
    receipt = None
    for rd in range(a.rounds):
        order = arms if rd % 2 == 0 else arms[::-1]
        for name, ck, ab in order:
            r = run(a.tree, cells, ck, ab, a.device)
            if r is None:
                return 1
            acc[name].append(r["rows"])
            receipt = r["md5"]
            print(f"# round {rd} arm {name} md5={receipt}", flush=True)
    print(f"# md5 {receipt}")

    def med(name, i, j):
        vs = [rows[i][j] for rows in acc[name] if rows[i][j] is not None]
        return statistics.median(vs) if vs else float("nan")

    print(f"\n{'b':>5} {'V':>8} {'k':>5} {'dtype':>8} {'route':>5} | "
          f"{'on us':>9} {'off us':>9} {'off/on':>7} | {'abort us':>9} {'abort/on':>8} | "
          f"{'nan kern':>8} (official rule = 'topk'-matching kernels only)")
    for i, (b, v, k, dt) in enumerate(cells):
        on = med("on", i, 4)
        off = med("off", i, 4)
        ab = med("abort", i, 4)
        nk = med("abort", i, 6)
        route = acc["on"][0][i][7]
        print(f"{b:>5} {v:>8} {k:>5} {dt:>8} {route:>5} | {on:9.1f} {off:9.1f} "
              f"{off/on:7.3f} | {ab:9.1f} {ab/on:8.3f} | {nk:8.2f}")
    print("\noperator reading (every kernel, official arm) vs official:")
    for i, (b, v, k, dt) in enumerate(cells):
        op = med("on", i, 5)
        on = med("on", i, 4)
        print(f"  {b:>5} {v:>8} {k:>5} {dt:>8}: official {on:9.1f}  operator {op:9.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
