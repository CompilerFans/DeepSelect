#!/usr/bin/env bash
#
# Compile the three extracted reference drivers, using the toolchain in
# /tmp/dsprobe/TOOLCHAIN.md and nothing else.
#
#   ./build_all.sh            build every impl found under this directory
#   ./build_all.sh deep_gemm  build one
#
# Each impl directory is expected to hold a driver named `main.cu` plus at
# least one other translation unit (the kernel TU).  Drivers are compiled into
# $OUT (default /tmp/dsref_build/<impl>/main) -- never into the impl directory,
# so this script stays read-only with respect to the sources it builds.
#
# Two link lines are tried, in this order:
#
#   1. the verbatim TOOLCHAIN.md whole-program line
#      cucc -O2 -std=c++20 --offload-arch=xcore1000 <tus> -o <main>
#   2. the same line plus -I"$MACA_PATH/include"
#
# The second is not cosmetic: mcoplib's driver uses __float2half_rn and does not
# include <cuda_fp16.h>, so it only resolves against MACA's own include tree.
# Which line an impl needed is printed, so the extra flag is never silent.

set -uo pipefail

HERE=$(cd "$(realpath "$(dirname "$0")")" && pwd)
OUT=${DSREF_BUILD_DIR:-/tmp/dsref_build}

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export CUDA_HOME="${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$CUDA_HOME}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

CUCC="$CUDA_HOME/bin/cucc"
[[ -x "$CUCC" ]] || { echo "build_all.sh: no cucc at $CUCC" >&2; exit 1; }

IMPLS=("$@")
if [[ ${#IMPLS[@]} -eq 0 ]]; then
    IMPLS=()
    for d in "$HERE"/*/; do
        [[ -f "${d}main.cu" ]] && IMPLS+=("$(basename "$d")")
    done
fi
[[ ${#IMPLS[@]} -gt 0 ]] || { echo "build_all.sh: no impl with a main.cu under $HERE" >&2; exit 1; }

mkdir -p "$OUT"

# Sets: ok, cmd_used ("plain" | "+maca-include"), err
build_one() {
    local impl="$1" dir="$HERE/$1"
    local -a tus=() extra=()
    local src

    ok=0; cmd_used=""; err=""

    if [[ ! -d "$dir" ]]; then err="no such directory"; return; fi
    if [[ ! -f "$dir/main.cu" ]]; then err="no main.cu"; return; fi

    tus=("$dir/main.cu")
    while IFS= read -r src; do
        tus+=("$src")
    done < <(find "$dir" -maxdepth 1 -name '*.cu' ! -name 'main.cu' | sort)

    if [[ ${#tus[@]} -lt 2 ]]; then err="no kernel TU beside main.cu"; return; fi

    mkdir -p "$OUT/$impl"
    # -I"$MACA_PATH/include" is added first when present, so the extra flag is
    # only ever additive to the documented line.
    local -a common=(-O2 -std=c++20 --offload-arch=xcore1000)
    [[ -d "$MACA_PATH/include" ]] && extra=(-I"$MACA_PATH/include")

    for variant in plain maca_include; do
        local -a cmd=( "$CUCC" "${common[@]}" )
        [[ "$variant" == "maca_include" ]] && cmd+=("${extra[@]}")
        cmd+=("${tus[@]}" -o "$OUT/$impl/main")
        if err=$("${cmd[@]}" 2>&1); then
            ok=1
            cmd_used=$( [[ "$variant" == "plain" ]] && echo "plain" || echo "+ -I\$MACA_PATH/include" )
            return
        fi
    done
}

echo "cucc      : $CUCC"
echo "build dir : $OUT"
echo
printf '%-12s %-6s %s\n' "impl" "built" "link line"
printf '%s\n' "-------------------------------------------------------------"

rc=0
failed=()
for impl in "${IMPLS[@]}"; do
    build_one "$impl"
    if [[ "$ok" == "1" ]]; then
        printf '%-12s %-6s %s\n' "$impl" "yes" "$cmd_used"
    else
        printf '%-12s %-6s %s\n' "$impl" "NO" "see compiler error below"
        failed+=("$impl")
        rc=1
    fi
done
printf '%s\n' "-------------------------------------------------------------"

for impl in "${failed[@]+"${failed[@]}"}"; do
    echo
    echo "=== $impl: compiler error ==="
    build_one "$impl"
    printf '%s\n' "$err" | head -25
done

echo
if [[ $rc -eq 0 ]]; then
    echo "all impls built."
else
    echo "SOME IMPLS DID NOT BUILD (rc=$rc)."
fi
exit $rc
