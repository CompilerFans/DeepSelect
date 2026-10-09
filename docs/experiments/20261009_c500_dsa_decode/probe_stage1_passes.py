#!/usr/bin/env python3
"""Does the split's stage1 pay the 8-bit coarse level's overflow walk?

    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
        python3 docs/experiments/20261009_c500_dsa_decode/probe_stage1_passes.py

The question.  `probe.py` measured, on b64 V1048576 k512, `nan_scan_kernel` at
110.36 us and `stage1` at 459.67 us -- for the *same* 134.2 MB and the *same*
grid (`B * chunks` = 1024 CTAs).  A 4.2x per-byte gap at identical CTA count is
not CTA starvation, so it is one of: block shape, or read volume.

Read volume is where the source points.  `stage1` calls
`radix_topk_row_bf16_k`, the static-k row, which walks the input twice and
**twice more when the threshold's high-byte group overflows the arena**:

    pass 1  histogram                      radix_core.cuh:1520
    pass 2  collect (bin > threshold -> out, == -> arena)   :1544
    pass 3  overflow rescan: full row, keep (key>>8) == high_threshold_bin  :1620
    pass 4  overflow emit:   full row, same filter, write   :1635

`radix_topk_row_bf16_b` -- the production row, which got coarse12 -- resolves
12 coarse + 4 fine = the whole 16-bit bf16 key, so its refine never overflows
and it walks twice, always.  The source records this disease for the row that
was fixed (`radix_core.cuh:345`: "The 8-bit level made the threshold bin wider
than the arena on ordinary rows, so every one of them paid a third row walk").
`_k` is the caller that never got the fix.

The test.  Hold the cell and the kernel fixed; change only the DATA.  The
arena holds `kSmemInputSize` slots (3514; x2 = 7028 with compact indices) and a
chunk is `V/chunks` elements, so overflow is a statement about one high byte's
population:

* **crowded** -- the harness's own generator (the official grid's data), whose
  rows pile into a few high bytes.  This is the arm `probe.py` measured.
* **bituniform** -- uniform over the whole 16-bit space, so each of the 256
  high bytes holds ~1/256 of the chunk (256 of 65536 elements).  Overflow is
  arithmetically impossible.

If passes 3+4 are firing on crowded data, stage1's time must roughly halve on
bituniform -- with the kernel byte-identical, which no code change can be
confused with.

Reported in both currencies: us, and GB/s under *both* pass-count hypotheses,
so the reader sees which one makes the kernel's real rate land where the
same-machine `nan_scan_kernel` (1 pass, 1216 GB/s = 73.7% of wall) already is.
"""

import argparse
import copy
import importlib.util
import os
import sys
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "tests"))

import lib                                    # noqa: E402  (tests/lib.py)
import test as official                       # noqa: E402  (tests/test.py)
import kernelkit as kk                        # noqa: E402  (`kk.bench`, as tests/test.py)


def _load_perf_snapshot():
    """`scripts/perf_snapshot.py` as a module -- `scripts/` is not a package."""
    spec = importlib.util.spec_from_file_location(
        "ds_perf_snapshot", REPO / "scripts" / "perf_snapshot.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


perf_snapshot = _load_perf_snapshot()

# The C500 streaming-read wall, measured (`C500-to-parity-plan.zh.md` §7.1).
READ_WALL_GB_S = 1650.0

OPERATOR_MARKERS = ("topk", "nan_scan")


def is_operator_kernel(name: str) -> bool:
    low = name.lower()
    return any(m in low for m in OPERATOR_MARKERS)


def short_name(name: str, tail: int = 34) -> str:
    """Template args are kept -- the block width is in them.  Cut at the first
    `<`, not the last `::`; the template arguments contain `::` too."""
    bare = name.strip()
    if bare.startswith("void "):
        bare = bare[5:]
    head, sep, args = bare.partition("<")
    head = head.rsplit("::", 1)[-1]
    if not sep:
        return head
    return f"{head}<...{args[-tail:]}" if len(args) > tail else f"{head}<{args}"


def route_of(names) -> str:
    joined = " ".join(names).lower()
    if "chunk_stage1" in joined:
        return "split"
    if "radix" in joined:
        return "row"
    return "?"


def make_bituniform(b, v, device="cuda", seed=1234):
    """bf16 whose *bit patterns* are uniform, so every high byte gets ~1/256.

    The two high bytes `0x7F` / `0xFF` are where bf16 puts every NaN and Inf
    (sign + exponent-all-ones), and they are the one part of the space that
    cannot be ranked: the NaN contract would fire.  They are remapped to zero,
    which costs 2/256 of the uniformity and nothing else.
    """
    g = torch.Generator(device=device).manual_seed(seed)
    bits = torch.randint(0, 1 << 16, (b, v), dtype=torch.int32,
                         device=device, generator=g)
    hi = bits >> 8
    bad = (hi == 0x7F) | (hi == 0xFF)
    bits = torch.where(bad, torch.zeros_like(bits), bits)
    return bits.to(torch.int16).view(torch.bfloat16)


def with_input(t, x):
    """`t` with `.input` replaced -- `perf_snapshot.call_topk` reads only
    `t.input` / `t.end` / `t.output_idx_offset`, so this is the whole delta."""
    try:
        return t._replace(input=x)
    except AttributeError:
        c = copy.copy(t)
        c.input = x
        return c


def measure(p, t, num_tests):
    call = lambda: perf_snapshot.call_topk(p, t, "maca_c")   # noqa: E731
    call()
    torch.cuda.synchronize()

    res = kk.bench(call, num_tests)
    # `_bench_kineto` stores `time_range.start / 1e6` (`bench.py:170`) and
    # torch's `time_range` is microseconds, so these are **seconds**.
    spans = {}
    for name in res.get_kernel_names():
        total_s = sum(e - s for s, e in res.time_ranges[name])
        spans[name] = total_s / num_tests * 1e6

    op_names = [n for n in res.get_kernel_names() if is_operator_kernel(n)]
    return {
        "spans": spans, "op_names": op_names,
        "op_us": sum(spans[n] for n in op_names),
        "span_us": res.get_e2e_time(op_names) * 1e6 if op_names else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-tests", type=int, default=10)
    ap.add_argument("--device", type=int, default=None,
                    help="shorthand for CUDA_VISIBLE_DEVICES=N")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    torch.set_default_device("cuda")

    # A cell the new gate still routes to the split: `kChunkedVocabPerRow`
    # admits `vocab_size / 20000` rows, and b=6 is inside that for both vocab
    # values the bf16 grid carries at b=6.
    want = [(6, 1048576, 512), (6, 262144, 512), (1, 1048576, 512),
            # The control: b=64 no longer splits (the gate landed in df8fe88),
            # so this cell runs `radix_topk_row_bf16_b` -- the row that *has*
            # coarse12.  If `_b` is immune to the data and `_k` is not, the
            # data-dependence is a property of the 8-bit dataflow, not of the
            # input.
            (64, 1048576, 512)]
    cells = []
    for b, v, k in want:
        hit = [c for c in official.performance_cases()
               if c.batch_size == b and c.vocab_size == v and c.topk == k
               and str(c.dtype) == "torch.bfloat16"]
        if hit:
            cells.append(hit[0])
    if not cells:
        raise SystemExit("probe_stage1_passes.py: no matching cells in the grid")

    print(f"# read wall = {READ_WALL_GB_S:.0f} GB/s (C500, measured streaming)")
    print(f"# device = {torch.cuda.get_device_name()}"
          f"  sm_count = {torch.cuda.get_device_properties(0).multi_processor_count}")
    print(f"# arena: kSmemInputSize = 3514 slots (x2 = 7028 with compact indices);"
          f" a chunk is vocab/chunks elements")
    print()

    for p in cells:
        if p.seed == -1:
            p.seed = (p.batch_size * 1_000_003 + p.vocab_size * 11 + p.topk) % 2**31
        t = lib.generate_testcase(p)
        b, v, k = p.batch_size, p.vocab_size, p.topk
        rows = b
        read_mb = rows * v * 2 / 1e6          # one full-row walk, MB

        arms = [("crowded", t)]
        xu = make_bituniform(b, v)
        arms.append(("bituniform", with_input(t, xu)))

        print(f"=== b{b} V{v} k{k}   (one row walk = {read_mb:.1f} MB)")
        measured = {}
        for label, tt in arms:
            r = measure(p, tt, args.num_tests)
            measured[label] = r
            print(f"  [{label}]  route={route_of(r['op_names'])}  "
                  f"span={r['span_us']:.2f} us   op_sum={r['op_us']:.2f} us")
            for n in sorted(r["spans"]):
                mark = "op " if n in r["op_names"] else "   "
                print(f"      {mark}{short_name(n):<44} {r['spans'][n]:>10.2f} us")
        print()

        # Per operator kernel, both currencies, and the arm-to-arm ratio.  The
        # pass columns are the two hypotheses worth reading a rate against: a
        # two-pass radix walks the row twice; a one-pass scan once.
        print(f"  {'kernel':<34} {'crowded':>9} {'bituniform':>11} {'ratio':>7}"
              f"   {'crowded real GB/s (%wall)':>30}")
        for n in sorted(set(measured["crowded"]["op_names"])):
            a = measured["crowded"]["spans"].get(n, float("nan"))
            b_ = measured["bituniform"]["spans"].get(n, float("nan"))
            if a != a or b_ != b_:
                continue
            ratio = a / b_
            # `nan_scan` is one pass; the radix rows are two.  Report the rate
            # the kernel would actually run at, so the column is comparable
            # across kernels and against the 1,650 wall.
            passes = 1 if "nan_scan" in n.lower() else 2
            real = read_mb * passes * 1e3 / a
            print(f"  {short_name(n, 26):<34} {a:>9.2f} {b_:>11.2f} {ratio:>6.2f}x"
                  f"   {real:>9.1f} ({passes} pass) {100*real/READ_WALL_GB_S:>4.1f}%")
        print(f"  {'op_sum':<34} {measured['crowded']['op_us']:>9.2f} "
              f"{measured['bituniform']['op_us']:>11.2f} "
              f"{measured['crowded']['op_us']/measured['bituniform']['op_us']:>6.2f}x")
        print(f"  (one row walk = {read_mb:.1f} MB; ratio >1 means the crowded arm"
              f" is slower)")
        print()


if __name__ == "__main__":
    main()
