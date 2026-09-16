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

export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"

rm -rf build dist
rm -rf ./*.egg-info

which python
echo "build.sh: CUCC_TARGETS=$CUCC_TARGETS"

python setup.py bdist_wheel

if [[ -n "${BUILDROOT:-}" ]]; then
    dest="${BUILDROOT}/wheel"
    mkdir -p "${dest}"
    cp dist/*.whl "${dest}/"
    echo "build.sh: wheel also copied to ${dest}"
fi

echo "build.sh: done"
cd "$original_dir"
