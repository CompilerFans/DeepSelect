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
# `-W` drops torch's own "flash_attn is not installed" import warning (nothing
# here uses it); every other warning still prints.
python -W "ignore:Could not find flash_attn:UserWarning" -c 'import sys, torch
print(f"build.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'
echo "build.sh: CUCC_TARGETS=$CUCC_TARGETS"

# `CUCC_TARGETS` becomes one `-offload-arch` list: every target is an image of
# the same source in the one extension, so a single build serves every family.
python setup.py bdist_wheel

# After the wheel exists, so what is exported is a wheel that was built.  Same
# destination and same variable as the host repository's `build.sh`.
if [[ -n "${BUILDROOT:-}" ]]; then
    dest="${BUILDROOT}/wheel"
    mkdir -p "${dest}"
    cp dist/*.whl "${dest}/"
    echo "build.sh: wheel also copied to ${dest}"
fi

echo "build.sh: done"
cd "$original_dir"
