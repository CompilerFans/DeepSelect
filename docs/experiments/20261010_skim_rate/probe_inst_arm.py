"""What does the *instantiation* cost, with the lever held out of the way?

Three arms, one cell list, one session:

  old    a tree from before the skip tables (the light instantiation)
  off    this tree with `DEEP_SELECT_BF16_SKIP=0` (the heavy instantiation, base
         dataflow -- the knob answers every call with the row walk, so this is
         the same two passes the old arm runs, compiled into the bigger body)
  on     this tree, default (heavy instantiation, list walk)

`old` vs `off` is the question this probe exists for and it is a single variable:
both walk the whole row twice and neither reads a summary, so the difference is
the 15 MT of registers the summary costs -- and with it, whether a 1024-thread
CTA still fits twice per AP (32 MT is the line).  `off` vs `on` is the lever.
Arm order and rounds alternate, so a clock drift lands on all three.

`--tree` may name the same directory for `off` and `on`; they differ only in the
environment the child is started with, which is why this is not `ab_snapshot.py`
(its arms are directories, and one process cannot hold both answers: the knob is
read once per process).

    CUDA_VISIBLE_DEVICES=2 python3 probe_inst_arm.py --tree <this tree> \
        [--old /tmp/ab_old] [--rounds 3] [--iters 10]
"""
import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))          # docs/experiments/<exp>
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))

CELLS = [(4096, 1048576, 512), (4096, 262144, 512), (768, 1048576, 512),
         (64, 1048576, 512), (4096, 16384, 512), (4096, 65536, 1024)]

# One child per (arm, round).  Same shape as `tools/ab_snapshot.py`'s child, and
# it asserts which package it loaded rather than trusting `sys.path`.
CHILD = r'''
import sys, json, os, hashlib
sys.path.insert(0, "__ARM__")
sys.path.insert(0, "__TESTS__")
import torch, kernelkit as kk, deep_select
from deep_select._binding import extension_path

assert os.path.dirname(deep_select.__file__) == os.path.join("__ARM__", "deep_select"), \
    "arm __ARM__ loaded " + deep_select.__file__
_p = extension_path("deep_select_maca")
LIB = hashlib.md5(open(_p, "rb").read()).hexdigest()[:8]
CELLS = json.loads(sys.argv[1]); WIDTH = max(c[1] for c in CELLS)
out = []
for bs, L, k in CELLS:
    torch.manual_seed(0)
    s = torch.randn(bs, WIDTH, dtype=torch.float32, device="cuda:0")[:, :L]
    s = s.to(torch.bfloat16).contiguous()
    r = kk.bench(lambda: deep_select.topk(s, k, return_value=False,
                                          backend="maca_c", indices_type=torch.int32,
                                          abort_when_nan_found=False),
                 int(sys.argv[2]))
    d = {n: r.get_kernel_time(n) * 1e6 for n in r.get_kernel_names()
         if "DeviceSynchronize" not in n}
    tot = sum(t for n, t in d.items()
              if any(x in n for x in ("stage", "nan", "radix", "coarse12")))
    out.append([bs, L, k, tot])
print("R " + json.dumps({"lib": LIB, "so": _p, "knob": os.environ.get("DEEP_SELECT_BF16_SKIP", "(unset)"),
                         "rows": out}))
'''


def run(tag, tree, env, cells, iters, device):
    code = (CHILD.replace("__ARM__", os.path.abspath(tree))
                 .replace("__TESTS__", os.path.join(REPO, "tests")))
    p = subprocess.run([sys.executable, "-c", code, json.dumps(cells), str(iters)],
                       cwd=tree, capture_output=True, text=True,
                       env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(device), **env))
    for line in p.stdout.splitlines():
        if line.startswith("R "):
            return json.loads(line[2:])
    sys.stderr.write(f"arm {tag} FAILED\n{p.stdout[-1500:]}{p.stderr[-1500:]}\n")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", default=REPO)
    ap.add_argument("--old", default="/tmp/ab_old")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--device", default="2")
    a = ap.parse_args()

    arms = [("old", a.old, {}), ("off", a.tree, {"DEEP_SELECT_BF16_SKIP": "0"}),
            ("on", a.tree, {})]
    acc = {name: [] for name, _, _ in arms}
    receipts = {}
    for rd in range(a.rounds):
        for name, tree, env in arms:
            r = run(name, tree, env, CELLS, a.iters, a.device)
            if r is None:
                return 1
            acc[name].append(r["rows"])
            receipts[name] = (r["lib"], r["knob"])
            print(f"# round {rd} arm {name} lib={r['lib']} knob={r['knob']}", flush=True)

    def med(name, i):
        return statistics.median(rows[i][3] for rows in acc[name])

    print(f"\n# cells {CELLS} ; rounds {a.rounds} ; iters {a.iters}")
    for name in acc:
        print(f"# {name:>3}: extension md5 {receipts[name][0]}, "
              f"DEEP_SELECT_BF16_SKIP={receipts[name][1]}")
    print(f"\n{'B':>5} {'V':>8} {'k':>5}  {'old us':>9} {'off us':>9} {'on us':>9}  "
          f"{'off/old':>8} {'on/off':>7} {'on/old':>7}")
    for i, (b, v, k) in enumerate(CELLS):
        o, f, n = med("old", i), med("off", i), med("on", i)
        print(f"{b:>5} {v:>8} {k:>5}  {o:>9.1f} {f:>9.1f} {n:>9.1f}  "
              f"{f / o:>8.3f} {n / f:>7.3f} {n / o:>7.3f}")
    print("\n# old->off is the instantiation (both walk the row twice, neither "
          "reads a summary);\n# off->on is the lever.")


if __name__ == "__main__":
    sys.exit(main())
