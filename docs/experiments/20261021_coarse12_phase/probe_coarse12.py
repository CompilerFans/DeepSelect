"""Read the coarse12 phase probe back through out_idx and print the phase table.

Run against a tree patched by `patch_coarse12_phase.py` (the probe record is
copied from the route's scratch columns into out_idx slots [k-16, k)), official
call shape only.

    PYTHONPATH=<probe-tree>:<probe-tree>/tests python3 probe_coarse12.py [cells...]

Columns of `out_idx[:, k-16:k]` (i.e. record offsets): 0..6 = clock64 deltas
from kernel start (start, pass 1, fold+narrow, collect, overflow-end,
arena-entry, end), 7 = arena round 0, 8 = path code, 9 = num_input[0],
10 = round at emit, 11 = requested_topk, 12 = threshold_bin_id, 13 = post-narrow
remaining topk, 14 = counter, 15 = arena rounds 1+2.

Phase shares are tick sums, so they need no clock rate.  Wall time per call is
printed beside them as the calibration: the coarse12 kernel is the dominant
term, and the ratio of the two says how much of the call it is.
"""
import statistics as st
import sys
import time

import torch

import deep_select
import lib

OOB = -2147483647
CELLS = [(4096, 131072, 2048), (4096, 129280, 512), (256, 131072, 2048),
         (16, 66551, 2048)]
if len(sys.argv) > 1:
    CELLS = [tuple(int(v) for v in a.split("x")) for a in sys.argv[1:]]

torch.set_default_device("cuda")

NAMES = ["start", "pass1", "narrow", "collect", "ovf-end", "arena-in", "end"]


def one(b, v, k, seed):
    p = lib.TestParam(batch_size=b, vocab_size=v, topk=k, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32, out_idx_dtype=torch.int32, seed=seed)
    case = lib.generate_testcase(p)
    x = case.input
    ts = []
    for _ in range(7):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        out = deep_select.topk(x, k, indices_type=torch.int32, return_value=False,
                               abort_when_nan_found=False, backend="maca_c")
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    wall = st.median(ts)

    rec = out[:, k - 16:k].to(torch.int64).cpu()
    total = rec[:, 6]                                    # end - start
    ok = total > 0
    rec, total = rec[ok], total[ok]
    shares = (rec[:, 0:7].sum(dim=0).double() / total.sum().double() * 100).tolist()

    codes = {}
    for c in rec[:, 8].tolist():
        codes[c] = codes.get(c, 0) + 1
    cand = rec[:, 9].tolist()
    tsum = total.sum().double()
    print(f"\nb{b} V{v} k{k}  rows={len(rec)}  wall {wall:8.3f} ms")
    print("  phases (% of coarse12 kernel):")
    for name, s in zip(NAMES, shares):
        print(f"    {name:<9} {s:6.2f}%")
    print(f"    arena r0  {100.0 * rec[:, 7].sum().item() / tsum.item():6.2f}%"
          f"   arena r1+2 {100.0 * rec[:, 15].sum().item() / tsum.item():6.2f}%")
    print(f"  path codes {codes}   num_input[0] min {min(cand)} "
          f"med {int(st.median(cand))} max {max(cand)}  cap 2048")
    print(f"  threshold_bin_id med {int(st.median(rec[:, 12].tolist()))}"
          f"   post-narrow topk med {int(st.median(rec[:, 13].tolist()))}")
    del x, out, case
    torch.cuda.empty_cache()


for i, (b, v, k) in enumerate(CELLS):
    one(b, v, k, 11 + i)
