#!/usr/bin/env bash
#
# Build the DeepSelect wheel.  This is the script a packaging step runs: it
# produces an artifact for **every family this tree names** and installs
# nothing.
#
#     ./build.sh                              # the shippable wheel
#     CUCC_TARGETS=xcore1600 ./build.sh       # one architecture
#     BUILDROOT=/out ./build.sh               # also copy the wheel to /out/wheel/
#
# Env:
#     CUCC_TARGETS  targets to compile, in `mxcc`'s `-offload-arch` spelling.
#                   Unset means one per family (see below); an unrecognized
#                   target is rejected by mxcc.
#     MACA_PATH     MACA toolkit root (default /opt/maca).  MACA_HOME is
#                   consulted when this is unset; this one wins if both are set.
#     MAX_JOBS      ninja's -j
#     BUILDROOT     if set, the wheel is also copied to `${BUILDROOT}/wheel/` --
#                   the host repository's own destination for the same variable
#                   (`mcDeepGEMM/build.sh`), so one packaging step can collect
#                   both wheels by pointing a single BUILDROOT at both trees.
#
# **This script does not produce `deep_select/deep_select_maca*.so`.**  That is
# `./develop.sh`, and the three scripts divide by output rather than by
# audience: a local iteration wants an extension in the checkout
# (`_binding.py` loads it from the package directory, and the runners receipt
# its md5), while a packaging step wants a wheel and nothing else.
#
#     develop.sh   build_ext --inplace  ->  deep_select/<so>   (run the tests)
#     build.sh     bdist_wheel          ->  dist/<whl>, ${BUILDROOT}/wheel/
#     install.sh   bdist_wheel + pip    ->  site-packages
#
# `bdist_wheel` rather than `build_ext --inplace` for the same reason
# `install.sh` builds its wheel first and hands the file to pip: `setup.py`
# stamps its version with `datetime.now()`, so the two runs of a PEP 517
# install can straddle a second boundary and produce a rejected wheel name.
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

# One target per family, which is what a wheel has to carry: a wheel built for
# one board cannot be shipped to another.  The spellings go to `mxcc` verbatim
# -- `-offload-arch` is its own vocabulary, and it is the thing that knows what
# this toolchain accepts.
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
