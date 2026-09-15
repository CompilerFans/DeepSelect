#!/usr/bin/env bash
#
# Build a wheel and install it into the active environment.  Same build as
# `./build.sh`, producing a wheel instead of an in-place extension.
#
#     ./install.sh                            # every family
#     CUCC_TARGETS=xcore1600 ./install.sh     # for another architecture
#     CUCC_TARGETS=native    ./install.sh     # just this device
#
# Env:
#     CUCC_TARGETS  targets to compile; unset means
#                   `deep_select/_arch.py::DEFAULT_TARGETS`, one per family
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

# `CUCC_TARGETS` is deliberately not set; see the header.

rm -rf build dist
rm -rf ./*.egg-info

which python
which pip
echo "install.sh: CUCC_TARGETS=${CUCC_TARGETS:-<unset: setup.py takes _arch.DEFAULT_TARGETS>}"

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
