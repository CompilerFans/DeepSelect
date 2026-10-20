#!/bin/bash
# Build the DeepSelect+DeepJIT integration probe (host side only; the device
# kernel is JIT-compiled at run time by DeepJIT itself).
set -euo pipefail
MACA_PATH=${MACA_PATH:-/opt/maca}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CUCC=${CUCC:-$MACA_PATH/tools/cu-bridge/bin/cucc}
TORCH_INC=${TORCH_INC:-/opt/conda/lib/python3.10/site-packages/torch/include}
exec $CUCC -std=c++20 -O2 -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA \
  -I"$REPO/third-party/DeepJIT/include" \
  -I"$REPO/third-party/fmt/include" \
  -I"$MACA_PATH/include" \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  -I"$TORCH_INC" \
  -I"$TORCH_INC/torch/csrc/api/include" \
  -I/opt/conda/include/python3.10 \
  "$REPO/tools/jit_probe/jit_probe.cc" -o /tmp/jit_probe \
  -L/opt/conda/lib -lpython3.10 \
  -L/opt/conda/lib/python3.10/site-packages/torch/lib -lc10 -lc10_cuda
