#!/usr/bin/env bash
#
# Build the DeepSelect MACA kernels in place.
#
# The host repository's `build.sh` shape: script_dir cd, same MACA_PATH /
# LD_LIBRARY_PATH setup, same `CUCC_TARGETS` variable and meaning, restore the
# caller's directory.  Three deliberate differences:
#
#   * it does NOT run `bdist_wheel`.  `setup.py` stamps the version with
#     `datetime.now()`, and a wheel run executes `setup.py` twice -- once for
#     metadata, once for the wheel -- so the two can straddle a second boundary
#     and be rejected as misnamed (`Wheel has unexpected file name`), an
#     upstream bug this tree inherits.  `build_ext --inplace` runs `setup.py`
#     once.  Producing the wheel is `install.sh`'s job.
#   * it removes the stale in-place `.so` first.  `build_ext --inplace` copies
#     out of `build/lib` by comparing timestamps, and the in-place `.so` sits
#     under `deep_select/` (gitignored), so it is stale by default -- a stale
#     one means measuring the previous binary while reading the new source
#     (handover §4).  Never trust the timestamp.
#   * `CUCC_TARGETS` defaults to one target per architecture family this tree
#     names -- `xcore1000,xcore1500,xcore1600` -- matching the host repository's
#     default in spirit.  It cannot be that list verbatim: several of the host
#     default's entries are *aliases of a family already in the list*
#     (`xcore1008`->1000, `xcore1502`/`xcore1520`->1500, `xcore1610`/`xcore1620`
#     ->1600), and this `setup.py` names the extension after the FAMILY
#     (`deep_select_xcore<N>`), so two targets in one family build the same
#     extension twice -- the second silently overwriting the first -- while
#     `mxcc` rejects `xcore1610`/`xcore1620` outright (`invalid target ID`) and
#     the 3.5.3.17 toolchain rejects `xcore1008` too.  A caller who wants a
#     specific part's spelling passes it, one target at a time.
#
# Env: CUCC_TARGETS (default `xcore1000,xcore1500,xcore1600`), MACA_PATH
#      (default /opt/maca), MAX_JOBS (torch reads it for ninja's -j).
#
# Cross-arch is the same line as any other build:
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

# torch's `_find_cuda_home()` reads `CUDA_HOME`/`CUDA_PATH` *before* falling back
# to `${MACA_PATH}/tools/cu-bridge` (guess #4) and cucc's own `CUCC_PATH` is what
# it execs `gomxccbin` from, so a stale one left in the caller's shell silently
# selects a different toolkit than MACA_PATH names -- measured on this box, whose
# profile exports `CUDA_PATH`/`CUCC_PATH` for an SDK root the `/opt/maca` symlink
# no longer points at, making `MACA_PATH=/opt/maca-3.8.1 ./build.sh` fail inside
# cucc with `.../tools/cu-bridge/bin/gomxccbin: No such file or directory`.
# Derive all three from MACA_PATH instead: they then cannot disagree, and
# `MACA_PATH=... ./build.sh` means what it says.
export CUDA_PATH="$MACA_PATH/tools/cu-bridge"
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$MACA_PATH/tools/cu-bridge"

# One target per family this tree names -- the same three `xcore<N>` bases the
# host repository's default resolves to, minus its per-part aliases, which this
# build cannot take (see the note above): one extension per family, so a second
# target in the same family would overwrite the first, and the toolchain does
# not accept every alias spelling anyway.  `deep_select/_arch.py::FAMILY_OF_TARGET`
# is the authority on which spellings exist; an unrecognized one fails the build
# rather than being skipped.
export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"

rm -rf build dist
rm -rf ./*.egg-info
rm -f deep_select/deep_select_*.so

which python
python -c 'import sys, torch
print(f"build.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'
echo "build.sh: CUCC_TARGETS=$CUCC_TARGETS"

# setup.py runs one extension per architecture, each with its own
# `--offload-arch`, so the loop over targets lives there and not here.  A
# target it cannot resolve, or one the compiler rejects, fails the build:
# `set -e` above and `_arch.family_of_target` below are the whole check.
python setup.py build_ext --inplace

echo "build.sh: done"
cd "$original_dir"
