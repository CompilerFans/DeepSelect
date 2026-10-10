"""What does the measured skim ratio predict on the official grid?

Inputs, all measured elsewhere and none of them fitted here:

  ratio(V)  the list+gather arm's `skim/full` from `run_sweep.sh` -- the
            skimmed collect walk against the full collect walk, same data, same
            occupancy, same L2 policy.
  phase(V)  the pass-1 / collect share of a bf16 row-kernel cell, from Phase
            9.3's in-kernel timestamps (`probe_phase_marks.py`).  Measured at
            k=512 on four cells; interpolated in log2(V) between them and held
            flat outside.

The model is one line per cell:

    t' = p1 * 1.016 + col * ratio(V) + (rest)

where the 1.016 is pass 1 paying one extra 2-byte store per 64 elements
(the summary), expressed against pass 1's own bytes.

Only k=512 cells are moved: `ratio(V)` was measured at k=512 (candidates/row is
a constant ~600, so a larger k means a denser threshold bin and a smaller win),
and no k=1024/2048 cell has been measured.  Leaving them at 1.0 makes the total
a lower bound over the k=512 subset, not the lever's ceiling.

    python predict_grid.py [perf_data/MetaX_C500/baseline/deepselect_perf.csv]
"""
import csv
import sys

CSV = sys.argv[1] if len(sys.argv) > 1 else \
    "perf_data/MetaX_C500/baseline/deepselect_perf.csv"

# measured (run_sweep.sh, device 2, 2026-10-10), the arm with the list in
# device memory -- the shape production takes.
RATIO = {1048576: 0.142, 524288: 0.223, 262144: 0.312, 131072: 0.508,
         65536: 0.705}
RATIO_MIN_V = 65536

# Phase 9.3, k=512, as fractions of a whole cell; (pass1, collect)
PHASE = [(131072, 0.418, 0.474), (524288, 0.440, 0.526),
         (1048576, 0.437, 0.544)]
SUMMARY_WRITE = 1.016


def phase(V):
    lo, hi = PHASE[0], PHASE[-1]
    if V <= lo[0]:
        return lo[1], lo[2]
    if V >= hi[0]:
        return hi[1], hi[2]
    for a, b in zip(PHASE, PHASE[1:]):
        if a[0] <= V <= b[0]:
            import math
            w = (math.log2(V) - math.log2(a[0])) / (math.log2(b[0]) - math.log2(a[0]))
            return a[1] + w * (b[1] - a[1]), a[2] + w * (b[2] - a[2])
    raise AssertionError


def ratio(V):
    ks = sorted(RATIO)
    if V <= ks[0]:
        return RATIO[ks[0]]
    if V >= ks[-1]:
        return RATIO[ks[-1]]
    for a, b in zip(ks, ks[1:]):
        if a <= V <= b:
            w = (V - a) / (b - a)
            return RATIO[a] + w * (RATIO[b] - RATIO[a])
    raise AssertionError


def k512(V):
    return V >= RATIO_MIN_V


rows = [r for r in csv.DictReader(open(CSV))
        if r["backend"] == "maca_c" and r["status"] == "pass"]
before = sum(float(r["time(us)"]) for r in rows)

print(f"{'cell':>28} {'now us':>10} {'p1':>5} {'col':>5} {'r':>5} {'pred us':>10} {'x':>5}")
after = 0.0
moved = 0
for r in sorted(rows, key=lambda r: -float(r["time(us)"])):
    t = float(r["time(us)"])
    V, k = int(r["n_cols"]), int(r["top_k"])
    if r["family"] == "lightning_indexer" and k == 512 and k512(V):
        p1, col = phase(V)
        r_ = ratio(V)
        t2 = t * (p1 * SUMMARY_WRITE + col * r_ + (1 - p1 - col))
        moved += 1
        print(f"{r['n_rows']:>10}x{V:<10}k{k:<4} {t:>10.1f} {p1:>5.3f} {col:>5.3f} "
              f"{r_:>5.3f} {t2:>10.1f} {t / t2:>5.2f}")
    else:
        t2 = t
    after += t2

print(f"\n{moved} of {len(rows)} cells moved "
      f"(k=512 bf16, V>={RATIO_MIN_V}); k=1024/2048 left at 1.00")
print(f"grid now {before/1000:.2f} ms -> predicted {after/1000:.2f} ms "
      f"({(1 - after/before)*100:+.1f}%)")
