#!/usr/bin/env python3
"""One grid, three reference drivers, one table.

`run_all.sh` builds the drivers and calls this.  For every cell it runs each
impl's `main` binary and prints

    impl | bs | len | k | ms | GB/s | agrees

What makes the rows comparable, each of these deliberately:

* **One traffic formula, every impl, every cell**: ``n_rows * (len + k) * 4``
  bytes -- the scores read plus the int32 indices written.  It is the formula
  `tests/bench_dsa_topk.py` uses, so a GB/s here is comparable to a GB/s there.
  The drivers' own bandwidth lines are ignored: the three report three
  different quantities (one counts writes, one counts only the row prefixes,
  one knows the index buffer) and none of them is the table's.
* **One timing unit**: each driver's own reported per-iteration milliseconds.
  The three timers are not cross-calibrated, so `ms` is a per-impl signal.
* **`--` for anything an impl cannot serve**, never a number from a different
  shape.  The reason is printed on that impl's own row, not only in the footer.

## Why every driver runs, not one

A single run per cell would be half the work and much less than half the value.
The three drivers write their timings as `<label>: <value> ms` and the table
needs one `ms` column out of them; the cheapest way to get that is a per-impl
regex, and the cheapest way to get a per-impl regex right is to have all three
drivers' real output in front of you at once.  It also means a formatting change
in one driver shows up as one impl going to `--`, with the raw reason attached,
instead of as a table that quietly lost a column.

## What `agrees` means here

`agrees` is **the driver's own check**: each ref verifies its output against a
CPU `std::nth_element` top-k and prints PASS/FAIL.  This script reads that
verdict; it does not re-derive it.  The column is `ok`, `FAIL`, or `--`.

That check is necessary but not sufficient and the table says so rather than
implying more: a driver can be self-consistent and still disagree with the
framing the library ops use (a different prefix window, an off-by-one at the tie
boundary, an index transform where the selector contract has none).  The
`torch.topk` cross-check that catches those is a *separate* step -- `tests/`-side
`bench_dsa_topk.py`, which is reported in README.md and is where the three
libraries, not the three drivers, are actually compared.
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
IMPLS = ["deep_gemm", "ds", "mcoplib"]

GRID_BS = [6, 256, 4096]
GRID_LEN = [16384, 65536, 524288]
GRID_K = [2048]
EXTRA = [(524288, 512)]

CFG = {
    "deep_gemm": {
        "time_re": re.compile(r"time:\s*([0-9.]+)\s*ms/iter"),
        "verify_re": re.compile(r"verify:\s*(\d+)/(\d+)\s*rows match"),
        # kMaxTopK = 2048: it sizes `__shared__ int selected_indices[kMaxTopK]`
        # (fp32_topk.cu:935) and is fed params.top_k, so top_k > 2048 over-reads
        # shared memory.  The public torch wrapper does not guard it -- kMaxTopK
        # appears only in a compile-time static_assert at :422.  Reported by the
        # impl's owner; the grid never exceeds 2048, but the harness should not
        # invite a shape the kernel cannot serve.
        "max_k": 2048,
        "min_k": 1,
        "note": "fp32; one selector over the whole row",
    },
    "ds": {
        # This driver spaces its labels (`time : 0.1402 ms`), hence its own regex.
        "time_re": re.compile(r"time\s*:\s*([0-9.]+)\s*ms"),
        "verify_re": re.compile(r"verify:\s*(PASS|FAIL)"),
        "max_k": 4096,          # deep_select_maca::kMaxTopK
        "min_k": 1,
        "note": "fp32; row path below the chunk floor, split above it",
    },
    "mcoplib": {
        "time_re": re.compile(r"time[: ]+\s*([0-9.]+)\s*(ms|us)"),
        # `KNOWN` is this driver's own verdict for a shape its source is known
        # to get wrong (main.cu:107).  It is read as "not measurable", not as a
        # pass -- the driver itself declines to call it one.
        "verify_re": re.compile(r"\b(PASS|FAIL|KNOWN)\b"),
        "max_k": 512,           # a compile-time constant in the kernel
        "min_k": 512,
        "note": "fp32; top_k fixed at 512",
    },
}


def driver_dir(impl: str) -> str:
    return os.path.join(HERE, impl)


def find_main(impl: str) -> str | None:
    for cand in ("main", "build/main"):
        p = os.path.join(driver_dir(impl), cand)
        if os.path.isfile(p) and os.access(p, os.X_OK):
            return p
    p = os.path.join(os.environ.get("DSREF_BUILD_DIR", "/tmp/dsref_build"),
                     impl, "main")
    return p if os.path.isfile(p) else None


def probe_argv(impl: str) -> tuple[bool, str]:
    """Whether the driver takes a shape CLI, read off its own main.cu.

    Probed, not assumed.  This is the check that keeps a driver with a fixed
    case list from being credited with numbers for a shape it never ran -- the
    worst failure mode available to this script, because the row would look
    completely normal.
    """
    src = os.path.join(driver_dir(impl), "main.cu")
    if not os.path.isfile(src):
        return False, "no main.cu in the impl directory"
    try:
        with open(src, "r", errors="replace") as fh:
            text = fh.read()
    except OSError as exc:
        return False, f"cannot read main.cu: {exc}"
    if not re.search(r"int\s+main\s*\([^)]*argv", text):
        return False, ("int main() takes no arguments -- the driver runs its own "
                       "fixed case list, so no grid cell can be requested")
    return True, ""


def run_driver(impl, bs, length, k, iters, timeout):
    """-> (ms | None, self_ok | None, rc | None, reason)"""
    binary = find_main(impl)
    if binary is None:
        return None, None, None, "not built"

    ok, why = probe_argv(impl)
    if not ok:
        return None, None, None, why

    env = dict(os.environ)
    maca = env.get("MACA_PATH", env.get("MACA_HOME", "/opt/maca"))
    env["MACA_PATH"] = maca
    env["LD_LIBRARY_PATH"] = ":".join(
        [os.path.join(maca, p) for p in ("lib", "mxgpu_llvm/lib")]
        + [env.get("LD_LIBRARY_PATH", "")])

    cmd = [binary, str(bs), str(length), str(k), str(iters)]
    try:
        proc = subprocess.run(cmd, cwd=driver_dir(impl), env=env, timeout=timeout,
                              stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    except subprocess.TimeoutExpired:
        return None, None, None, f"timed out after {timeout}s"
    out = proc.stdout.decode("utf-8", "replace")

    cfg = CFG[impl]
    ms = None
    m = cfg["time_re"].search(out)
    if m:
        ms = float(m.group(1))
        if m.lastindex and m.lastindex >= 2 and m.group(2) == "us":
            ms /= 1e3

    v = cfg["verify_re"].search(out)
    self_ok = None
    if v:
        if impl == "deep_gemm":
            self_ok = v.group(1) == v.group(2)
        else:
            self_ok = v.group(1) == "PASS"

    if ms is None:
        tail = [ln.strip() for ln in out.splitlines() if ln.strip()][-2:]
        return (None, self_ok, proc.returncode,
                ("no time in output: " + " | ".join(tail)) if tail else "no output")
    # A shape whose own verifier says the answer is wrong gets no bandwidth
    # number: the timing is real, but a GB/s on a wrong answer invites a
    # comparison that should not be made.
    if self_ok is False:
        return None, False, proc.returncode, "own verifier reports the output is wrong for this shape"
    return ms, self_ok, proc.returncode, ""


def cell(impl, bs, length, k, args, notes):
    if not (CFG[impl]["min_k"] <= k <= CFG[impl]["max_k"]):
        return (f"{impl:<10} {bs:>6} {length:>7} {k:>5} | {'--':>10} {'--':>9} | "
                f"{'--':<11}  k outside contract "
                f"[{CFG[impl]['min_k']}, {CFG[impl]['max_k']}]")
    if find_main(impl) is None:
        notes.append(f"{impl}: binary not built -- run build_all.sh")
        return (f"{impl:<10} {bs:>6} {length:>7} {k:>5} | {'--':>10} {'--':>9} | "
                f"{'--':<11}  not built")

    ms, self_ok, run_rc, reason = run_driver(impl, bs, length, k, args.iters, args.timeout)
    # A driver that exits non-zero is failing its own check by construction,
    # even if the verdict token could not be read -- that keeps a wrong result
    # from showing up as `no-verdict` and reading like a missing measurement.
    if self_ok is None and run_rc is not None and run_rc != 0:
        self_ok = False
    traffic = bs * (length + k) * 4
    gbs = None if ms is None else traffic / (ms * 1e-3) / 1e9

    if ms is not None and self_ok:
        agree, reason = "ok", ""
    elif self_ok is False:
        agree = "FAIL(self)"
    elif self_ok is None:
        agree = "no-verdict" if ms is not None else "--"
    else:
        agree = "ok"

    line = (f"{impl:<10} {bs:>6} {length:>7} {k:>5} | "
            f"{'--' if ms is None else f'{ms:.4f}':>10} "
            f"{'--' if gbs is None else f'{gbs:.1f}':>9} | {agree:<11}")
    # The reason a cell is empty goes on its own row: a bare `--` explained only
    # in the footer is how a gap gets mistaken for a miss.
    if reason:
        line += f"  {reason}"
        notes.append(f"{impl}: {reason}")
    return line


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--impls", default=",".join(IMPLS))
    ap.add_argument("--bs", default=",".join(str(x) for x in GRID_BS))
    ap.add_argument("--len", default=",".join(str(x) for x in GRID_LEN))
    ap.add_argument("--k", default=",".join(str(x) for x in GRID_K))
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=900)
    ap.add_argument("--dev", type=int, default=0)
    args = ap.parse_args()

    impls = [x.strip() for x in args.impls.split(",") if x.strip()]
    bs_axis = [int(x) for x in args.bs.split(",")]
    len_axis = [int(x) for x in args.len.split(",")]
    k_axis = [int(x) for x in args.k.split(",")]

    # The drivers are launched under this; the python side only supervises, but
    # pinning it keeps the report's device line from naming a second device.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.dev)

    print(f"# ref three-way grid | drivers under {HERE}")
    print(f"# device   : CUDA_VISIBLE_DEVICES={args.dev}")
    print(f"# traffic  : n_rows*(len+k)*4 B -- one formula, every impl, every cell")
    print(f"# ms       : each driver's own timer (per-impl signal, not cross-calibrated)")
    print(f"# agrees   : that driver's own CPU std::nth_element self-check.  The")
    print(f"#            torch.topk cross-check is a separate step -- see README.md")

    notes: list[str] = []
    for impl in impls:
        built = "built" if find_main(impl) else "NOT BUILT"
        ok, why = probe_argv(impl)
        print(f"# {impl:<9}: {built:<9} argv {'yes' if ok else 'NO':<3} "
              f"k [{CFG[impl]['min_k']}, {CFG[impl]['max_k']}] | {CFG[impl]['note']}")
        if not ok and built == "built":
            notes.append(f"{impl}: {why}")

    cells = [(bs, ln, k) for k in k_axis for bs in bs_axis for ln in len_axis]
    cells += [(bs, ln, k) for (ln, k) in EXTRA for bs in bs_axis]
    seen: set = set()
    cells = [c for c in cells if not (c in seen or seen.add(c))]

    head = (f"{'impl':<10} {'bs':>6} {'len':>7} {'k':>5} | {'ms':>10} {'GB/s':>9} "
            f"| {'agrees':<11}")
    print()
    print(head)
    print("-" * len(head))
    for (bs, length, k) in cells:
        for impl in impls:
            print(cell(impl, bs, length, k, args, notes))

    if notes:
        print()
        print("# notes")
        for n in dict.fromkeys(notes):
            print(f"#   {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
