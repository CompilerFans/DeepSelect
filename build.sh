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

export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"

# The toolkit version, for the version string `setup.py` builds -- it lands in
# the wheel name, which is the only place a consumer can read which SDK
# generation the artifact was linked against.  A caller-set value wins, the
# same precedence the line above uses; otherwise it comes out of the toolkit's
# own `Version.txt`, first `Version:` line only.
#
# A value that cannot appear in a PEP 440 local version is dropped here rather
# than handed on: setuptools would refuse the whole build without naming the
# file it came from.  Dropping it leaves `setup.py` recording the absence --
# `maca0.0.0.0`, which says "none was given" instead of inventing one.
if [[ -z "${MACA_VERSION:-}" && -r "$MACA_PATH/Version.txt" ]]; then
    while IFS= read -r _line || [[ -n "$_line" ]]; do
        if [[ "$_line" == Version:* ]]; then
            MACA_VERSION="${_line#Version:}"
            MACA_VERSION="${MACA_VERSION//[[:space:]]/}"
            break
        fi
    done < "$MACA_PATH/Version.txt"
fi
if [[ "${MACA_VERSION:-}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    export MACA_VERSION
    echo "build.sh: MACA_VERSION=$MACA_VERSION"
else
    unset MACA_VERSION
    echo "build.sh: warning: no MACA_VERSION from the caller and none readable in" >&2
    echo "build.sh:          $MACA_PATH/Version.txt -- the wheel version will record that" >&2
fi

rm -rf build dist
rm -rf ./*.egg-info

which python
echo "build.sh: CUCC_TARGETS=$CUCC_TARGETS"

python setup.py bdist_wheel

if [[ -n "${BUILDROOT:-}" ]]; then
    dest="${BUILDROOT}/wheel"
    mkdir -p "${dest}"
    cp dist/*.whl "${dest}/"
    echo "build.sh: wheel also copied to ${dest}"
fi

echo "build.sh: done"
cd "$original_dir"
