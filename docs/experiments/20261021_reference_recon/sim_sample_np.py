"""Offline feasibility sim (numpy): sample a contiguous prefix, take its m-th
largest ordered key T, count the row's elements strictly above T.

The official input is `randn_like` (NormalFloatDistribution), so this is the
official distribution; no device is needed for a count.

For each shape: the tie group at the k-th value, the 12-bit threshold-bin
population (coarse12's overflow quantity), and the candidate-count distribution
for three sample sizes and three margins.  Candidate count must satisfy
k <= count <= CAP, else the route falls back to the two-pass coarse12.
"""
import numpy as np

CAP = 4096
FRACS = (8, 16, 32)
MARGINS = (1.25, 1.4, 1.6)
SHAPES = [(4096,131072,2048,256),(4096,129280,512,64),(4096,107520,2048,64),
          (256,131072,2048,64),(512,66551,2048,64),(256,129280,512,64)]

def okey(u32):                       # order-preserving key of a float bit pattern
    return (u32 ^ ((((u32 >> 15) & 0x00010001) * 0x7FFF) + 0x80008000)) & 0xFFFFFFFF

def to_bf16_bits(x):                 # round-to-nearest-even, keep the high half
    u = x.view(np.uint32).copy()
    r = (u + 0x8000 + ((u >> 16) & 1)) & 0xFFFF0000
    return r

def stats(v):
    v = np.asarray(v, dtype=np.float64)
    return v.mean(), v.std(), v.min(), v.max()

rng = np.random.default_rng(20261021)
for dtype in ("fp32", "bf16"):
    print(f"\n===== {dtype} =====")
    for (B, V, K, rows) in SHAPES:
        ties, bins, cnt = [], [], {f: {m: [] for m in MARGINS} for f in FRACS}
        for _ in range(rows):
            x = rng.standard_normal(V, dtype=np.float32)
            bits = x.view(np.uint32) if dtype == "fp32" else to_bf16_bits(x)
            key = okey(bits)
            kth = np.partition(key, V - K)[V - K]
            ties.append(int((key == kth).sum()))
            bins.append(int((key >> 20 == kth >> 20).sum()))
            for f in FRACS:
                s = V // f
                for mrg in MARGINS:
                    m = max(1, int(round(mrg * K * s / V)))
                    T = np.partition(key[:s], s - m)[s - m]
                    cnt[f][mrg].append(int((key > T).sum()))
        t_med, t_max = int(np.median(ties)), max(ties)
        b_med, b_max = int(np.median(bins)), max(bins)
        print(f"\n b{B} V{V} k{K}  ({rows} rows)")
        print(f"   ties@kth med {t_med} max {t_max} | 12-bit bin med {b_med} max {b_max}"
              f" | bin/k = {b_med/K:.2f}")
        for f in FRACS:
            for mrg in MARGINS:
                e, sd, lo, hi = stats(cnt[f][mrg])
                plo = np.mean(np.asarray(cnt[f][mrg]) < K) * 100
                phi = np.mean(np.asarray(cnt[f][mrg]) > CAP) * 100
                m = max(1, int(round(mrg * K * (V // f) / V)))
                print(f"   s=V/{f:<2d} m={m:5d} ({mrg:4.2f}k): E={e:7.1f} sd={sd:6.1f}"
                      f" sd/E={100*sd/e:4.1f}%  min {lo:6.0f} max {hi:6.0f}"
                      f"  P(<k)={plo:6.3f}% P(>cap)={phi:6.3f}%")
