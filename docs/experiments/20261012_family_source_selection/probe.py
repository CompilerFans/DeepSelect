#!/usr/bin/env python3
"""Which arms each xcore family lands on, read out of the tree rather than typed here.

    python3 docs/experiments/20261012_family_source_selection/probe.py

The record's central table says two things: every family is compiled from the
same kernel source, and the family-shaped *routing* is keyed on the device's
**AP count** rather than on the family the artifact was built for.  Both are
claims about the tree, so this probe reads them out of the tree --

  * `setup.py`                -> `SOURCES_BY_FAMILY` (which source each family builds)
  * `csrc/structs.h`          -> the per-family row (`ARCH_SM_COUNT`, `ARCH_SMEM_PER_AP_BYTES`)
  * `csrc/maca_kernels/xcore1000/maca_topk.cu`
                              -> the predicate constants and the AP-count set

-- and then evaluates each predicate over the AP counts a device can report.
Nothing here is a second copy of a production number: change a constant in the
source and this printout changes with it.  That is the whole point, because the
finding is that a *C600* satisfies a predicate that claims to name the C600U.

**What this probe cannot do, and does not pretend to**: it has no device and no
C600/C600U on this host, so it proves nothing about which arm is *faster* on
either.  It answers only "which branch does this code take", which is the
question the record needs settled; the measurement question is left open
explicitly there.
"""
import pathlib
import re
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
KERNEL = REPO / "csrc" / "maca_kernels" / "xcore1000" / "maca_topk.cu"
STRUCTS = REPO / "csrc" / "structs.h"
SETUP = REPO / "setup.py"

# AP counts a MACA device reports, taken from the family table itself plus the
# two the C600U ships at, which `is_c600u_ap_count` names.  The C500's 104 is
# `ARCH_SM_COUNT` for family 1000.
AP_COUNTS = (104, 28, 32)


def family_rows():
    """`{family: {"sm": N, "smem": N}}` from `csrc/structs.h`."""
    src = STRUCTS.read_text()
    rows = {}
    for m in re.finditer(
            r"#(?:if|elif) DEEP_SELECT_ARCH == (\d+)(.*?)(?=#elif|#else)", src, re.S):
        family, body = int(m.group(1)), m.group(2)
        sm = re.search(r"ARCH_SM_COUNT = (\d+)", body)
        smem = re.search(r"ARCH_SMEM_PER_AP_BYTES = ([\d\s*]+);", body)
        if sm and smem:
            rows[family] = {"sm": int(sm.group(1)),
                            "smem": eval(smem.group(1).strip())}
    return rows


def sources_by_family():
    """`{family: [source, ...]}` from `setup.py`'s table, plus its default."""
    src = SETUP.read_text()
    shared = re.search(r"SHARED_SOURCES = \[(.*?)\]", src, re.S)
    rows = re.search(r"SOURCES_BY_FAMILY = \{(.*?)\n\}", src, re.S)
    if shared is None or rows is None:
        sys.exit("setup.py has no SOURCES_BY_FAMILY table -- the layout fix is absent")
    shared = re.findall(r'"([^"]+)"', shared.group(1))
    table = {int(k): re.findall(r'"([^"]+)"', v) or shared
             for k, v in re.findall(r"(\d+):\s*(\[[^\]]*\]|\w+),", rows.group(1))}
    return shared, table


def kernel_consts():
    """The predicate constants, by name, out of `maca_topk.cu`."""
    src = KERNEL.read_text()
    out = {}
    for name in ("kC600UNarrowRowBatches", "kC600UChunkedVocabPerBatchSq",
                 "kF32Coarse12MeasuredSmCount", "kF32Coarse12C600UNarrowProduct",
                 "kF32Coarse12C600UWideProduct", "kChunkedVocabPerRow",
                 "kChunkedMaxBatches", "kChunkedMinVocab"):
        m = re.search(rf"constexpr uint(?:32|64)_t {name} = (\d+)", src)
        out[name] = int(m.group(1)) if m else None
    m = re.search(r"inline bool is_c600u_ap_count\(uint32_t (\w+)\) \{\s*"
                  r"return ([^;]+);", src)
    out["_c600u_param"] = m.group(1)
    out["c600u_set"] = {int(v) for v in re.findall(r"== (\d+)", m.group(2))}
    # The two block widths, from the header `radix_block_for` returns from.
    core = (KERNEL.parent / "radix_core.cuh").read_text()
    for name, key in (("kBlockSize", "block_narrow"),
                      ("kLongRowBlockSize", "block_wide")):
        m = re.search(rf"constexpr (?:int|uint32_t) {name}\s*=\s*(\d+)", core)
        out[key] = int(m.group(1)) if m else None
    # The coarse12 gate's *budget* half, whose request is
    # `max(route arena, the split it replaces)` -- the arena from the route's
    # own header, the split's from `kF32SmemInputSize` (its asserted value, so
    # this reads what the build refuses to let drift).
    out["dg12_smem"] = int(re.search(
        r"constexpr size_t kSmemBytes = (\d+) \* (\d+)",
        (KERNEL.parent / "dg_coarse12.cuh").read_text()).group(1)) * 1024
    out["f32_row_smem"] = 2 * int(re.search(
        r"kF32SmemInputSize == (\d+)", src).group(1)) * 4
    return out


def main():
    rows = family_rows()
    shared, table = sources_by_family()
    k = kernel_consts()

    print("── 1. which source each family builds (`setup.py`'s SOURCES_BY_FAMILY) ──")
    print(f"  default (a family with no row)      <- {', '.join(shared)}")
    for family in sorted(rows):
        got = table.get(family, shared)
        tag = "row" if family in table else "default"
        print(f"  family {family}  (ARCH_SM_COUNT={rows[family]['sm']:>3}, "
              f"{rows[family]['smem']:>6} B/AP)   [{tag:>7}] <- {', '.join(got)}")
    ported = REPO / "csrc" / "maca_kernels" / "xcore1600"
    pointed = [f for f, s in {**{f: table.get(f, shared) for f in rows}}.items()
               if any("maca_kernels/xcore1600" in p for p in s)]
    print(f"  rows naming csrc/maca_kernels/xcore1600/ (the ported tree): "
          f"{pointed if pointed else 'none'}")
    print(f"  that tree is on disk: {ported.is_dir()}")

    print()
    print("── 2. the AP-count set the family-shaped rules are keyed on ──")
    print(f"  is_c600u_ap_count({k['_c600u_param']}) returns true for "
          f"{sorted(k['c600u_set'])}")
    c600_sm = rows.get(1500, {}).get("sm")
    print(f"  ARCH_SM_COUNT for family 1500 (C600)            = {c600_sm}")
    print(f"  => a C600 satisfies it: {c600_sm in k['c600u_set']}   "
          f"<< the finding: three ladders fitted on a C600U are taken by a C600")

    needed = max(k["dg12_smem"], k["f32_row_smem"])
    print()
    short = [f for f in sorted(rows) if rows[f]["smem"] < needed]
    short_desc = (str(short) if short else
                  "none -- the budget half is inert for every family this "
                  "build ships")
    print(f"  coarse12's budget half asks for {needed} B/AP "
          f"= max(route arena {k['dg12_smem']}, the split it replaces "
          f"{k['f32_row_smem']})")
    print(f"  families whose budget is short of it: {short_desc}")

    print()
    print("── 3. the arm each (artifact family, device AP count) lands on ──")
    hdr = (f"  {'family':>6} {'APs':>4} | {'block':>6} | {'bf16 split':>10} | "
           f"{'coarse12':>10} | {'f32 chunks':>10} | {'f32 split':>9}")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for family in sorted(rows):
        for aps in sorted({rows[family]["sm"], *AP_COUNTS}):
            c6 = aps in k["c600u_set"]
            # radix_block_for: `long_row` is a shape question, held fixed here
            # to the case the C600U rule was fitted on (a long row); what this
            # column answers is which branch the *family* test sends it down.
            block = f"{k['block_narrow']:>6}" if c6 else f"{k['block_wide']:>6}"
            bf16 = "sqrt(V/B2)" if c6 else f"V/{k['kChunkedVocabPerRow']}"
            # The budget half first: it is `if constexpr`, so it is a property
            # of the *artifact* and cannot be talked out of by the device.
            if rows[family]["smem"] < needed:
                coarse = "off/budget"
            elif c6:
                coarse = "C6U lad"
            elif aps == k["kF32Coarse12MeasuredSmCount"]:
                coarse = "C5 lad"
            else:
                coarse = "off/meas"
            f32ch = "C5 only" if aps == k["kF32Coarse12MeasuredSmCount"] else "off"
            print(f"  {family:>6} {aps:>4} | {block} | {bf16:>10} | "
                  f"{coarse:>10} | {f32ch:>10} | {'gate free':>9}")

    print()
    print("── 4. what the two candidate keys would change ──")
    print(f"  today      : key = AP count in {sorted(k['c600u_set'])}   "
          f"-> a 28-AP C600 takes all three C600U ladders")
    print(f"  alternative: key = ARCH_FAMILY == 1600          "
          f"-> the same C600 takes the C500 band for the block width and the")
    print(f"               bf16 split, and is *denied* coarse12 / f32 chunks "
          f"(neither ladder names 28)")
    print()
    print("  Neither is measured on a C600.  Which is right is an open question;")
    print("  see the README's section 5.")


if __name__ == "__main__":
    main()
