#!/bin/bash
# Build the DeepSelect+DeepJIT integration probe (host side only; the device
# kernel is JIT-compiled at run time by DeepJIT itself).
#
# Paths are asked of the environment rather than baked in: MACA_PATH, CUCC,
# PYTHON (the interpreter whose torch import is the one to build against), and
# the TORCH_*/PY_* overrides when the derivation is not wanted.  The probe used
# to name a conda layout outright, which made it unusable on any host whose
# Python is not at /opt/conda.
set -euo pipefail
MACA_PATH=${MACA_PATH:-/opt/maca}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CUCC=${CUCC:-$MACA_PATH/tools/cu-bridge/bin/cucc}
PYTHON=${PYTHON:-python}

TORCH_ROOT=${TORCH_ROOT:-$("$PYTHON" -c 'import torch, os; print(os.path.dirname(torch.__file__))')}
TORCH_INC=${TORCH_INC:-$TORCH_ROOT/include}
TORCH_LIB=${TORCH_LIB:-$TORCH_ROOT/lib}
PY_INC=${PY_INC:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')}
PY_LIB=${PY_LIB:-$("$PYTHON" -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')}
PY_LIBNAME=${PY_LIBNAME:-$("$PYTHON" -c 'import sys; print("python%d.%d" % sys.version_info[:2])')}

for required in "$REPO/tools/jit_probe/third-party/DeepJIT/include" \
                "$REPO/tools/jit_probe/third-party/fmt/include" \
                "$MACA_PATH/include" "$CUCC" "$TORCH_INC" "$PY_INC"; do
  if [[ ! -e "$required" ]]; then
    echo "build.sh: not found: $required" >&2
    echo "build.sh: set MACA_PATH/CUCC and a PYTHON whose torch import works," >&2
    echo "build.sh: or override TORCH_INC/PY_INC directly" >&2
    exit 1
  fi
done

exec "$CUCC" -std=c++20 -O2 -DTORCH_API_INCLUDE_EXTENSION_H -DUSE_MACA \
  -I"$REPO/tools/jit_probe/third-party/DeepJIT/include" \
  -I"$REPO/tools/jit_probe/third-party/fmt/include" \
  -I"$MACA_PATH/include" \
  -I"$MACA_PATH/tools/cu-bridge/include" \
  -I"$TORCH_INC" \
  -I"$TORCH_INC/torch/csrc/api/include" \
  -I"$PY_INC" \
  "$REPO/tools/jit_probe/jit_probe.cc" -o /tmp/jit_probe \
  -L"$PY_LIB" -l"$PY_LIBNAME" \
  -L"$TORCH_LIB" -lc10 -lc10_cuda \
  -Wl,-rpath,"$TORCH_LIB" -Wl,-rpath,"$MACA_PATH/lib"
