#!/usr/bin/env bash
#
# Compile and run probe.cu against kerutils' MACA platform arm.
#
#   ./run_probe.sh              xcore1000 (this host's family)
#   ARCH=xcore1600 ./run_probe.sh
#
# **The flags are read out of `setup.py`, not copied from it.**  The claim this
# probe makes is "the arm holds under the flags the build uses", and a second
# hand-maintained copy of that list is the way such a claim goes quietly false
# -- `setup.py`'s own history has a hand-copied include catalogue drifting from
# cucc's as the worked example.  `nvcc_args` is a list of string literals, so
# `ast.literal_eval` reads it exactly, and a flag added to the build is a flag
# the probe compiles with on the next run.
#
# What is *not* read from setup.py: `-offload-arch` and `-DDEEP_SELECT_ARCH`,
# which are per-Extension there and per-invocation here, and the four `-I`
# paths, which `include_dirs` builds as absolute paths from a different root.
# `run_probe.sh` prints every one of them so the difference is visible rather
# than assumed.
#
# The compiler is cu-bridge's `cucc`, the same one torch drives -- this build
# has no `bin/nvcc` and never calls mxcc directly.  `cucc` supplies the MACA
# include catalogue (which is how `<maca_bfloat16.h>` and `<cuda_runtime_api.h>`
# resolve at all) and then invokes `mxcc`.

set -uo pipefail

HERE=$(cd "$(realpath "$(dirname "$0")")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export CUDA_HOME="${CUDA_HOME:-$MACA_PATH/tools/cu-bridge}"
export CUCC_PATH="${CUCC_PATH:-$CUDA_HOME}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:${LD_LIBRARY_PATH:-}"

CUCC="$CUDA_HOME/bin/cucc"
[[ -x "$CUCC" ]] || { echo "run_probe.sh: no cucc at $CUCC" >&2; exit 1; }

ARCH=${ARCH:-xcore1000}
FAMILY=${ARCH#xcore}
OUT=${DSPROBE_OUT:-/tmp/dsref_build/kerutils_platform_arms}
mkdir -p "$OUT"

# setup.py's own nvcc_args.
mapfile -t BUILD_FLAGS < <(python3 - "$REPO/setup.py" <<'PY'
import ast, sys

tree = ast.parse(open(sys.argv[1]).read())
for node in ast.walk(tree):
    if isinstance(node, ast.Assign) and any(
        getattr(t, "id", None) == "nvcc_args" for t in node.targets
    ):
        for flag in ast.literal_eval(node.value):
            print(flag)
        break
else:
    sys.exit("setup.py: no `nvcc_args = [...]` assignment found")
PY
)
[[ ${#BUILD_FLAGS[@]} -gt 0 ]] || { echo "run_probe.sh: read no flags from setup.py" >&2; exit 1; }

PROBE_FLAGS=(
    "-offload-arch=$ARCH"
    "-DDEEP_SELECT_ARCH=$FAMILY"
    "-I$REPO/csrc"
    "-I$REPO/csrc/ffi"
    # the vendored library's own include root: `host/host.h` names
    # `"kerutils/common/common.h"` from here, not from beside itself
    "-I$REPO/csrc/3rdparty/kerutils/include"
    "-I$MACA_PATH/include"
)

echo "cucc        : $CUCC"
echo "arch        : $ARCH  (-DDEEP_SELECT_ARCH=$FAMILY)"
echo "extension   : $OUT/probe"
echo
echo "flags from setup.py's nvcc_args:"
printf '    %s\n' "${BUILD_FLAGS[@]}"
echo "flags this probe adds:"
printf '    %s\n' "${PROBE_FLAGS[@]}"
echo

# `BASH_XTRACE=1 ./run_probe.sh` also prints cucc's own expansion, which is
# where its `mxcc` invocation appears.
set -x
"$CUCC" "${BUILD_FLAGS[@]}" "${PROBE_FLAGS[@]}" "$HERE/probe.cu" -o "$OUT/probe"
rc=$?
set +x
[[ $rc -eq 0 ]] || { echo; echo "run_probe.sh: compile FAILED (rc=$rc)" >&2; exit $rc; }

echo
"$OUT/probe"
rc=$?
echo
[[ $rc -eq 0 ]] && echo "run_probe.sh: PASS" || echo "run_probe.sh: probe FAILED (rc=$rc)"
exit $rc
