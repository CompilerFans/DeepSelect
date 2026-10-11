#!/usr/bin/env bash
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

# This machine's family, which `setup.py` resolves (`native` -> this host's
# family): an in-place build is for the device in front of it, and the other
# families' artifacts are not runnable here.  `build.sh` is the path whose
# default carries every family.
export CUCC_TARGETS="${CUCC_TARGETS:-native}"

rm -rf build dist
rm -rf ./*.egg-info
rm -f deep_select/deep_select_*.so

which python
echo "develop.sh: CUCC_TARGETS=$CUCC_TARGETS"

python setup.py build_ext --inplace

echo "develop.sh: in-place extensions:"
ls -1 deep_select/deep_select_maca*.so
cd "$original_dir"
