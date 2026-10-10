"""Generate the RK_CUT arm of `radix_core.cuh` that the cost ladder measures.

`_b`'s per-invocation cost is spread across phases with no single dominant term,
so pricing it means pricing each phase.  This copies the real header and inserts
`if (RK_CUT <= N) return;` at five phase boundaries inside
`radix_topk_row_bf16_b`; compiling `probe_cut.cu` against a copy of that tree
with `-DRK_CUT=N` then gives the cumulative cost up to each boundary, and
`-DRK_CUT=99` is the whole function.

    python3 make_cut_arm.py /tmp/rkcut      # writes /tmp/rkcut/radix_core.cuh
    # compile probe_cut.cu with -I/tmp/rkcut -DRK_CUT={1..5,99}

It edits a COPY.  The production header is never written; the arm tree is
throwaway (see the campaign's no-residue rule).
"""
import io
import os
import shutil
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
SRC = os.path.join(REPO, "csrc", "maca_kernels", "xcore1000", "radix_core.cuh")

# (line number in the header, cut id) -- each names a `__syncthreads()` that
# closes a phase of `radix_topk_row_bf16_b`, and the script refuses to edit if
# the line is not one (the header moves; the anchors are checked, not assumed).
CUTS = [(1625, 5), (1611, 4), (1515, 3), (1458, 2), (1372, 1)]


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    dst = os.path.join(out_dir, "radix_core.cuh")
    shutil.copyfile(SRC, dst)
    lines = io.open(dst, encoding="utf-8").read().split("\n")
    for line_no, cut in CUTS:
        if not lines[line_no - 1].strip().startswith("__syncthreads"):
            sys.exit(f"line {line_no} is not a barrier: {lines[line_no-1]!r}")
        lines.insert(line_no, "        if (RK_CUT <= %d) return;   // PROBE-ONLY" % cut)
    lines.insert(37, "#ifndef RK_CUT\n#define RK_CUT 99\n#endif")
    io.open(dst, "w", encoding="utf-8").write("\n".join(lines))
    print(f"wrote {dst} (cuts after {[c[0] for c in CUTS]})")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/rkcut")
