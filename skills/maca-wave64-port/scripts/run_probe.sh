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

target="${1:-native}"
device="${2:-}"

script_dir=$(cd "$(dirname "$0")" && pwd)
maca_root="${MACA_HOME:-${MACA_PATH:-/opt/maca}}"

if [[ "$target" == "native" ]]; then
    # Ask the runtime, the same way deep_select._arch.native_target() does.
    sm=$(python - <<'PY' 2>/dev/null | tail -1
import torch
c = torch.cuda.get_device_capability()
print(c[0] * 10 + c[1])
PY
)
    case "$sm" in
        80)  target=xcore1000 ;;
        86)  target=xcore1500 ;;
        87|88|89) target=xcore1600 ;;
        *)   echo "run_probe.sh: device reports sm$sm, which is not a known MACA family;" >&2
             echo "              pass the target explicitly (xcore1000/xcore1500/xcore1600)" >&2
             exit 1 ;;
    esac
    echo "run_probe.sh: native target resolved to $target"
fi

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
