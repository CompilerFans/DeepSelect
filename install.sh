#!/usr/bin/env bash
#
# Build a wheel for this device and install it into the active environment.
#
# The host repository's `install.sh` shape: script_dir cd, same MACA_PATH /
# LD_LIBRARY_PATH setup, same `CUCC_TARGETS` (default `native`), `bdist_wheel`
# then `pip install --force-reinstall --no-deps`, restore the caller's
# directory.  One difference, recorded because it is not obvious from the
# reference:
#
#   * the wheel is built by a single `bdist_wheel` run and pip is handed the
#     finished file, rather than `pip install .`.  `setup.py` stamps the
#     version with `datetime.now()`, and a PEP 517 install runs `setup.py`
#     twice -- once for metadata, once for the wheel -- so the two runs can
#     straddle a second boundary and pip rejects the result as misnamed
#     (`Wheel has unexpected file name`).  Build isolation adds a second,
#     unrelated failure (torch is not in pip's isolated environment).  One run
#     cannot straddle itself, and `pip install .` is the split this avoids.
#
# It is the same build as `./build.sh` -- same variable, same default, one
# extension per architecture -- differing only in what it produces: a wheel,
# installed, instead of an in-place extension.
#
# Env: CUCC_TARGETS (default `xcore1000,xcore1500,xcore1600`), MACA_PATH
#      (default /opt/maca), MAX_JOBS (torch reads it for ninja's -j).
#
# Cross-arch is the same line as any other install:
#
#     CUCC_TARGETS=xcore1600 ./install.sh
#     CUCC_TARGETS=native    ./install.sh         # just this device
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# See build.sh: torch's `_find_cuda_home()` reads `CUDA_HOME`/`CUDA_PATH` ahead
# of its `${MACA_PATH}/tools/cu-bridge` fallback and cucc execs `gomxccbin` from
# `CUCC_PATH`, so a stale one in the caller's environment silently wins (on this
# box it made `MACA_PATH=/opt/maca-3.8.1 ./install.sh` die inside cucc).  Derive
# all three from MACA_PATH so they cannot disagree with the toolkit this script
# is building against.
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

# Same default as this tree's `build.sh` -- one target per family -- and *not*
# the host repository's `install.sh`, which pins `native`.  There the wheel is
# built for the active device; here it is the artifact a packaging step
# collects, so it carries every family this tree names.  See `build.sh` for why
# the family aliases in the host's list cannot be used.
export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"

rm -rf build dist
rm -rf ./*.egg-info

which python
which pip
echo "install.sh: CUCC_TARGETS=$CUCC_TARGETS"

python setup.py bdist_wheel
pip install dist/*.whl --force-reinstall --no-deps

# Confirm what the environment now resolves, from a directory that is not the
# repo -- otherwise the check reports the repo's own package and passes even
# when the install did nothing.
( cd /tmp && python -c '
import deep_select, os
print("install.sh: installed", deep_select.__version__)
print("install.sh: from     ", os.path.dirname(deep_select.__file__))
' )

echo "install.sh: done"
cd "$original_dir"
