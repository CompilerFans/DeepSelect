"""Per-cell table out of one `ab_snapshot.py` run.

Reads `<dir>/ab_paired.json` (`{"new": {"(B, V, k, dtype, idx)": {a, b, ...}}}`,
the medians are per-round so the table takes the median of each arm's rounds) and
prints the three families the landing receipt needs:

  carrier            bf16 + int32 + V >= 65536 -- the family the skip tables are
                     switched on for.  Split again by the gate's density half
                     (`V >= 128 * k`): the cells under it are excluded by the
                     landed gate and must read ~1.0 in a run of the landed tree.
  bf16+int32 short   V < 65536.  Same instantiation as the carrier (`kSkip` is
                     chosen by the index type), so it carries whatever the
                     heavier kernel body costs, but the host hands it a null
                     table: this is the family that says what that tax is.
  control            fp32 rows and bf16+int64-index rows.  Different
                     instantiation, so the only thing that can move them is the
                     box itself -- this is what the run's drift looks like, and
                     the number a citation of "no regression" has to carry.

`ratio` is median(new) / median(old), so > 1 means the new tree is slower.  Both
currencies are printed: us and the logical read bandwidth `B * V * 2 B / t`
against the measured 1,650 GB/s C500 wall (one pass of the row, which is the
currency `tests/test.py` uses).
"""
import argparse
import json
import statistics
import sys

WALL = 1650.0


def load(path):
    d = json.load(open(path))
    return d["new"] if "new" in d else d


def parse(key):
    b, v, k, dtype, idx = [x.strip().strip("'") for x in key.strip("()").split(",")]
    return int(b), int(v), int(k), dtype, idx


def med(xs):
    return statistics.median(xs)


def gbs(b, v, us):
    return b * v * 2 / us * 1e-3 if us else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--worst", type=int, default=12)
    args = ap.parse_args()

    rows = []
    for key, c in load(f"{args.dir}/ab_paired.json").items():
        b, v, k, dtype, idx = parse(key)
        a, nb = med(c["a"]), med(c["b"])
        rows.append(dict(b=b, v=v, k=k, dtype=dtype, idx=idx, a=a, nw=nb,
                         ratio=nb / a, route_a=set(c["route_a"]),
                         route_b=set(c["route_b"])))

    heavy = [r for r in rows if r["dtype"] == "bfloat16" and r["idx"] == "int32"]
    carrier = [r for r in heavy if r["v"] >= 65536]
    short = [r for r in heavy if r["v"] < 65536]
    control = [r for r in rows if r["dtype"] != "bfloat16" or r["idx"] != "int32"]
    gate_on = [r for r in carrier if r["v"] >= 128 * r["k"]]
    gate_off = [r for r in carrier if r["v"] < 128 * r["k"]]

    print(f"# {args.dir}: {len(rows)} cells, {len(carrier)} carrier "
          f"({len(gate_on)} gate-on / {len(gate_off)} density-excluded), "
          f"{len(short)} bf16+int32 short, {len(control)} control\n")

    def family(name, rs):
        if not rs:
            return
        rs = sorted(rs, key=lambda r: -r["ratio"])
        rat = [r["ratio"] for r in rs]
        tot_a = sum(r["a"] for r in rs)
        tot_b = sum(r["nw"] for r in rs)
        print(f"== {name}: n={len(rs)}  median {med(rat):.4f}x  "
              f"[{min(rat):.3f}, {max(rat):.3f}]  "
              f"total {tot_a:.1f} -> {tot_b:.1f} ms "
              f"({(tot_b / tot_a - 1) * 100:+.2f}%)")
        print(f"   worst {args.worst}:")
        print(f"   {'B':>5} {'V':>8} {'k':>5}  {'old us':>9} {'new us':>9}  "
              f"{'ratio':>6}  {'old GB/s':>9} {'new GB/s':>9}  {'%wall':>6}  routes")
        for r in rs[:args.worst]:
            print(f"   {r['b']:>5} {r['v']:>8} {r['k']:>5}  {r['a']:>9.1f} "
                  f"{r['nw']:>9.1f}  {r['ratio']:>6.3f}  "
                  f"{gbs(r['b'], r['v'], r['a']):>9.1f} "
                  f"{gbs(r['b'], r['v'], r['nw']):>9.1f}  "
                  f"{gbs(r['b'], r['v'], r['nw']) / WALL * 100:>5.1f}%  "
                  f"{'/'.join(sorted(r['route_a']))}->{'/'.join(sorted(r['route_b']))}")
        best = sorted(rs, key=lambda r: r["ratio"])[:8]
        print(f"   best {len(best)}:")
        for r in best:
            print(f"   {r['b']:>5} {r['v']:>8} {r['k']:>5}  {r['a']:>9.1f} "
                  f"{r['nw']:>9.1f}  {r['ratio']:>6.3f}  "
                  f"{gbs(r['b'], r['v'], r['a']):>9.1f} "
                  f"{gbs(r['b'], r['v'], r['nw']):>9.1f}  "
                  f"{gbs(r['b'], r['v'], r['nw']) / WALL * 100:>5.1f}%  "
                  f"{'/'.join(sorted(r['route_a']))}->{'/'.join(sorted(r['route_b']))}")
        print()

    family("carrier, gate-on (skip tables live)", gate_on)
    family("carrier, density-excluded (gate off -> row walk)", gate_off)
    family("bf16+int32 short (V < 65536, table null)", short)
    family("control (fp32 / int64 instantiation)", control)
    family("whole grid", rows)


if __name__ == "__main__":
    sys.exit(main())
