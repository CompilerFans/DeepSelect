#!/usr/bin/env bash
#
# Build a wheel **for this device** and install it into the active
# environment.  Same build as `./build.sh`, narrowed to one image and followed
# by a `pip install`.
#
#     ./install.sh                            # this device
#     CUCC_TARGETS=xcore1600 ./install.sh     # another architecture
#
# Env:
#     CUCC_TARGETS  targets to compile; unset means **`native`**, `mxcc`'s
#                   spelling for the part in front of you.  Narrowing is the
#                   point of this script: what it installs is the wheel this
#                   machine runs, and a caller who needs the shippable
#                   all-family artifact wants `./build.sh`.
#     MACA_PATH     MACA toolkit root (default /opt/maca).  MACA_HOME is
#                   consulted when this is unset; this one wins if both are set.
#     MAX_JOBS      ninja's -j
#
# `pip install .` is not used: a PEP 517 install runs `setup.py` twice
# (metadata, then wheel), and `setup.py` stamps its version with
# `datetime.now()`, so the two runs can straddle a second boundary and pip
# rejects the result as misnamed (`Wheel has unexpected file name`).  Building
# first keeps `setup.py` to one invocation.
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# See build.sh: a stale `CUDA_HOME`/`CUDA_PATH`/`CUCC_PATH` in the caller's
# environment silently beats MACA_PATH.  Derive all three from it.
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

# One image by default, unlike `build.sh`: this installs onto a machine, so the
# other two images would be dead weight in that machine's site-packages.  An
# explicit `CUCC_TARGETS` still wins.
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
