#!/bin/bash
# Build the DeepSelect+DeepJIT integration probe (host side only; the device
# kernel is JIT-compiled at run time by DeepJIT itself).
set -euo pipefail
MACA_PATH=${MACA_PATH:-/opt/maca}
REPO=$(cd "$(dirname "$0")/.." && pwd)
CUCC=${CUCC:-$MACA_PATH/tools/cu-bridge/bin/cucc}
exec $CUCC -std=c++20 -O2 \
  -I"$REPO/third-party/DeepJIT/include" \
  -I"$REPO/third-party/fmt/include" \
  -I"$MACA_PATH/include" \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  "$REPO/tools/jit_probe/jit_probe.cc" -o /tmp/jit_probe
