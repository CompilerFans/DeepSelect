#!/usr/bin/env bash
#
# Build and run wave64_probe on the device in front of you.
#
#     scripts/run_probe.sh                  # native target
#     scripts/run_probe.sh xcore1600        # or name one
#     scripts/run_probe.sh xcore1600 1      # ... on device 1
#
# Uses cu-bridge's cucc, the same device compiler a MACA torch extension build
# goes through, so the semantics reported are the ones the kernels under audit
# were actually built with.  Compiling straight through mxcc would report a
# different (and irrelevant) answer.
set -euo pipefail

# The target is `mxcc`'s own `-offload-arch` spelling (`native` included) and
# is passed through untouched -- `native` resolves to the xcore1000 family on
# this toolchain, measured byte-identical to `-offload-arch=xcore1000` on a
# C500.  No table here: naming a target is the compiler's vocabulary, not ours.
target="${1:-native}"
device="${2:-}"

script_dir=$(cd "$(dirname "$0")" && pwd)
maca_root="${MACA_HOME:-${MACA_PATH:-/opt/maca}}"

export MACA_PATH="$maca_root"
export CUDA_PATH="$maca_root/tools/cu-bridge"
export CUCC_PATH="$maca_root/tools/cu-bridge"

out=$(mktemp -d)/wave64_probe
echo "run_probe.sh: compiling for $target with $CUDA_PATH/bin/cucc"
"$CUDA_PATH/bin/cucc" "$script_dir/wave64_probe.cu" -o "$out" \
    --offload-arch="$target" -O3 -std=c++17 -w

if [[ -n "$device" ]]; then
    echo "run_probe.sh: running on device $device"
    CUDA_VISIBLE_DEVICES="$device" "$out"
else
    "$out"
fi
