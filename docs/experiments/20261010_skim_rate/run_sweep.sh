#!/usr/bin/env bash
# The §9.6-#3 measurement: what does a skimmed pass 2 actually run at?
#
# `skim_walk` (built by build.sh) reads a dumped dataset and times, per
# invocation: the full collect walk (no skip), the summary-only walk, and one
# skimmed arm.  Every number in README.md's table comes from this script --
# re-run it to re-take the measurement rather than trusting the table.
#
#   ./run_sweep.sh <data-dir> <binary>      # e.g. /tmp/skim_data /tmp/skim_build/skim_walk
#
# Arms, all BLOCK=1024 unless the line says otherwise:
#   unroll=1  plain skim -- the test sits in the walk
#   unroll=4  plain skim, 4 lines in flight per thread group
#   unroll=8  list+gather: pass A compacts kept blocks into a list in DEVICE
#             memory, pass B gathers those lines
#   unroll=9  the same, list in SHARED memory (capped at 2048 entries)
#   keep_every=N  rewrites the summary to a synthetic 1/N density (0 = keep all,
#             the no-skip control).  Destructive to the loaded summary, so those
#             lines come last for each dataset.
set -u
DATA=${1:-/tmp/skim_data}
BIN=${2:-/tmp/skim_build/skim_walk}
IT=10

run() {  # run <prefix> <rows> <V> <k> <BLOCK> [keep_every] [unroll]
    echo "--- $1 rows=$2 V=$3 k=$4 BLOCK=$5 keep_every=${6:-1} unroll=${7:-1}"
    "$BIN" "$DATA/$1" "$2" "$3" "$4" "$5" "$IT" "${6:-1}" "${7:-1}"
}

for spec in "v1048576 1024 1048576" "v524288 2048 524288" \
            "v262144 1024 262144" "v131072 4096 131072" \
            "v65536 4096 65536"; do
    set -- $spec
    run "$1" "$2" "$3" 512 1024 1 1
    run "$1" "$2" "$3" 512 1024 1 4
    run "$1" "$2" "$3" 512 1024 1 9
    run "$1" "$2" "$3" 512 1024 1 8
    run "$1" "$2" "$3" 512 512  1 9
done
# Synthetic densities (destructive): the plain-skim arm against a known kept
# fraction, plus the keep-all control.
run v1048576 1024 1048576 512 1024 4   1
run v1048576 1024 1048576 512 1024 16  1
run v1048576 1024 1048576 512 1024 64  1
run v1048576 1024 1048576 512 1024 0   1
run v262144  1024 262144  512 1024 0   1
