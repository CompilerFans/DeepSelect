#!/usr/bin/env bash
#
# Build the DeepSelect MACA kernels in place.
#
#     ./build.sh                              # every family
#     CUCC_TARGETS=xcore1000,xcore1600 ./build.sh
#     CUCC_TARGETS=native ./build.sh          # just this device
#
# Env:
#     CUCC_TARGETS  targets to compile.  Unset means
#                   `deep_select/_arch.py::DEFAULT_TARGETS`, one per family;
#                   `setup.py` applies that default, so the scripts do not
#                   repeat the literal.  An unrecognized target fails the build.
#     MACA_PATH     MACA toolkit root (default /opt/maca).  MACA_HOME is
#                   consulted when this is unset; this one wins if both are set.
#     MAX_JOBS      ninja's -j
#
# `build_ext --inplace` rather than `bdist_wheel`: a wheel run executes
# `setup.py` twice, and `setup.py` stamps its version with `datetime.now()`.
# `install.sh` does the wheel and carries that workaround.
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# torch's `_find_cuda_home()` reads `CUDA_HOME`/`CUDA_PATH` *before* its
# `${MACA_PATH}/tools/cu-bridge` fallback and cucc execs `gomxccbin` out of
# `CUCC_PATH`, so a stale one in the caller's shell silently beats MACA_PATH.
# Derive all three from it: they then cannot disagree.
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

rm -rf build dist
rm -rf ./*.egg-info
# The in-place `.so` is gitignored, and `build_ext --inplace` copies out of
# `build/lib` by timestamp: left in place, a stale one would measure the
# previous binary against the new source.
rm -f deep_select/deep_select_*.so

which python
# `-W` drops torch's own "flash_attn is not installed" import warning (nothing
# here uses it); every other warning still prints.
python -W "ignore:Could not find flash_attn:UserWarning" -c 'import sys, torch
print(f"build.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'
echo "build.sh: CUCC_TARGETS=${CUCC_TARGETS:-<unset: setup.py takes _arch.DEFAULT_TARGETS>}"

# `CUCC_TARGETS` becomes one `-offload-arch` list: every target is an image of
# the same source in the one extension, so a single build serves every family.
python setup.py build_ext --inplace

echo "build.sh: done"
cd "$original_dir"
