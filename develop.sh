#!/usr/bin/env bash
#
# Build the DeepSelect MACA kernels **in place**, for local development: no
# wheel, nothing installed, `CUCC_TARGETS` defaulting to this device alone.
#
#     ./develop.sh                            # just this device
#     CUCC_TARGETS=xcore1000,xcore1600 ./develop.sh
#     CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh
#
# Env:
#     CUCC_TARGETS  targets to compile.  Unset means **`native`** -- the one
#                   family this device reports -- because this script exists to
#                   iterate on a kernel, and `./build.sh` is where the
#                   all-family artifact comes from.  An unrecognized target
#                   fails the build.
#     MACA_PATH     MACA toolkit root (default /opt/maca).  MACA_HOME is
#                   consulted when this is unset; this one wins if both are set.
#     MAX_JOBS      ninja's -j
#
# **This is the script that writes `deep_select/deep_select_maca*.so`.**
# `_binding.py` loads that file out of the package directory, and `run_test.sh`
# / `run_bench.sh` read its md5 as the "which artifact did I measure" receipt --
# both of which want an artifact in the *checkout*, not in site-packages.  So
# the three scripts divide by output, and this is the only one that produces
# an in-place extension:
#
#     develop.sh   build_ext --inplace  ->  deep_select/<so>   (why: run the tests)
#     build.sh     bdist_wheel          ->  dist/<whl>, ${BUILDROOT}/wheel/
#     install.sh   bdist_wheel + pip    ->  site-packages
#
# `build_ext --inplace` rather than `bdist_wheel`: a wheel run executes
# `setup.py` twice and `setup.py` stamps its version with `datetime.now()`, so
# the two runs can straddle a second boundary -- `install.sh` carries that
# workaround for the wheel it has to produce.
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

# The one target default that differs from `build.sh`'s, and the whole reason
# this is a separate script: developing against one device should not cost
# three device compiles.  An explicit `CUCC_TARGETS` still wins, so the
# cross-family check is `CUCC_TARGETS=xcore1000,xcore1500,xcore1600
# ./develop.sh` and needs no other flag.
export CUCC_TARGETS="${CUCC_TARGETS:-native}"

rm -rf build dist
rm -rf ./*.egg-info
# The in-place `.so` is gitignored, and `build_ext --inplace` copies out of
# `build/lib` by timestamp: left in place, a stale one would measure the
# previous binary against the new source.  (`build.sh` deletes `build/`
# wholesale, so it never sees this.)
rm -f deep_select/deep_select_*.so

which python
# `-W` drops torch's own "flash_attn is not installed" import warning (nothing
# here uses it); every other warning still prints.
python -W "ignore:Could not find flash_attn:UserWarning" -c 'import sys, torch
print(f"develop.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'
echo "develop.sh: CUCC_TARGETS=$CUCC_TARGETS"

# `CUCC_TARGETS` becomes one `-offload-arch` list: every target is an image of
# the same source in the one extension, so a single build serves every family.
python setup.py build_ext --inplace

echo "develop.sh: in-place extension at $(ls deep_select/deep_select_maca*.so)"
cd "$original_dir"
