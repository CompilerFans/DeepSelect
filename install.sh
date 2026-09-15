#!/usr/bin/env bash
#
# Build a wheel and install it into the active environment.  Same build as
# `./build.sh` (same variable, same default, one extension per architecture),
# producing a wheel instead of an in-place extension.  Two departures from the
# host repository's `install.sh`:
#
#   * `pip install .` is avoided: a PEP 517 install runs `setup.py` twice
#     (metadata, then wheel) and it stamps its version with `datetime.now()`, so
#     the runs straddle a second boundary and pip rejects the result as misnamed
#     (`Wheel has unexpected file name`).  Build isolation adds a second failure
#     (no torch in pip's isolated env).
#   * `CUCC_TARGETS` is not defaulted here either -- same reason as `build.sh`,
#     and the same single source for the default.
#
# Env: CUCC_TARGETS (default: `_arch.DEFAULT_TARGETS`), MACA_PATH
#      (default /opt/maca), MAX_JOBS (torch reads it for ninja's -j).
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
# repo's own package, passing even when the install did nothing.
( cd /tmp && python -c '
import deep_select, os
print("install.sh: installed", deep_select.__version__)
print("install.sh: from     ", os.path.dirname(deep_select.__file__))
' )

echo "install.sh: done"
cd "$original_dir"
