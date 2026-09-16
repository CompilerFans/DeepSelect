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

export CUCC_TARGETS="${CUCC_TARGETS:-native}"

rm -rf build dist
rm -rf ./*.egg-info

which python
which pip
echo "install.sh: CUCC_TARGETS=$CUCC_TARGETS"

python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall --no-deps

# From outside the repo on purpose: run from here and the check reports the
# repo's own package, passing even when the install did nothing.  `-W` drops
# torch's "flash_attn is not installed" warning (see build.sh).
( cd /tmp && python -W "ignore:Could not find flash_attn:UserWarning" -c '
import deep_select, os
print("install.sh: installed", deep_select.__version__)
print("install.sh: from     ", os.path.dirname(deep_select.__file__))
' )

echo "install.sh: done"
cd "$original_dir"
