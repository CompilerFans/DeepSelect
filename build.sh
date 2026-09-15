#!/usr/bin/env bash
#
# Build the DeepSelect MACA kernels in place.  The host repository's `build.sh`
# shape (script_dir cd, MACA_PATH/LD_LIBRARY_PATH, `CUCC_TARGETS`), plus:
#
#   * no `bdist_wheel` -- `setup.py` stamps its version with `datetime.now()` and
#     a wheel run executes it twice (metadata, then wheel), so the two can
#     straddle a second boundary and be rejected as misnamed (`Wheel has
#     unexpected file name`).  `build_ext --inplace` runs it once.
#   * the stale in-place `.so` is removed first -- `build_ext --inplace` copies
#     out of `build/lib` by timestamp and the in-place `.so` is gitignored, so a
#     stale one measures the previous binary against the new source.
#   * `CUCC_TARGETS` defaults to one target per family this tree names, not the
#     host list verbatim: this `setup.py` names the extension after the FAMILY,
#     so two targets in one family build the same extension twice (the second
#     silently overwriting the first) and `mxcc` rejects some aliases outright.
#
# Env: CUCC_TARGETS, MACA_PATH (default /opt/maca), MAX_JOBS (ninja's -j).
#
#     CUCC_TARGETS=xcore1000,xcore1600 ./build.sh
#     CUCC_TARGETS=native ./build.sh          # just this device
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# torch's `_find_cuda_home()` reads `CUDA_HOME`/`CUDA_PATH` *before* its
# `${MACA_PATH}/tools/cu-bridge` fallback and cucc execs `gomxccbin` out of
# `CUCC_PATH`, so a stale one in the caller's shell silently beats MACA_PATH.
# Derive all three from it: they then cannot disagree.
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

# One target per family (see the header).  `_arch.py::FAMILY_OF_TARGET` is the
# authority on which spellings exist; an unrecognized one fails the build.
export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"

rm -rf build dist
rm -rf ./*.egg-info
rm -f deep_select/deep_select_*.so

which python
python -c 'import sys, torch
print(f"build.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'
echo "build.sh: CUCC_TARGETS=$CUCC_TARGETS"

# The loop over targets lives in setup.py: one extension per architecture, each
# with its own `--offload-arch`.
python setup.py build_ext --inplace

echo "build.sh: done"
cd "$original_dir"
