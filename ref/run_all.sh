#!/usr/bin/env bash
#
# Build the three reference drivers, run them on one shape grid, print one table.
#
#   ./run_all.sh                    the whole grid
#   ./run_all.sh --bs 6,256         forward options to the table driver
#   ./run_all.sh --dev 1            which device
#   ./run_all.sh --no-build         skip the build step
#
# Grid: bs in {6, 256, 4096} x len in {16384, 65536, 524288} x k = 2048,
#       plus len = 524288, k = 512.
#
# Every row's bandwidth is n_rows*(len+k)*4 bytes over that driver's own
# reported time -- one formula for every impl and every cell.  A cell an impl
# cannot serve prints `--` with the reason on the row; it never prints a number
# from a different shape.
#
# The device must be free.  Default is 0; pass --dev N to move it.

set -uo pipefail

HERE=$(cd "$(realpath "$(dirname "$0")")" && pwd)
BUILD=1
PY_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-build) BUILD=0; shift ;;
        -h|--help)  sed -n '2,18p' "$0"; exit 0 ;;
        *)          PY_ARGS+=("$1"); shift ;;
    esac
done

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export CUDA_HOME="${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$CUDA_HOME}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

if [[ $BUILD -eq 1 ]]; then
    echo "=== build ==="
    "$HERE/build_all.sh"
    rc=$?
    if [[ $rc -ne 0 ]]; then
        echo
        echo "run_all.sh: build_all.sh failed (rc=$rc); running the table anyway so"
        echo "            the impls that DID build get measured and the ones that"
        echo "            did not are reported per-cell rather than dropped."
        echo
    fi
fi

exec python3 "$HERE/crosscheck.py" "${PY_ARGS[@]+"${PY_ARGS[@]}"}"
