#!/usr/bin/env bash
#
# Build the DeepSelect MACA kernels.
#
# Mirrors the host repository's `build.sh` in shape (script_dir cd, MACA_PATH /
# LD_LIBRARY_PATH, CUCC_TARGETS, restore the caller's directory), with one
# deliberate difference: it does NOT run `bdist_wheel`.
#
# `setup.py` stamps the version with `datetime.now()`, and a PEP 517 / wheel
# run executes `setup.py` twice -- once for metadata and once for the wheel.
# When the two runs straddle a second boundary pip rejects the result as
# misnamed (`Wheel has unexpected file name`), an upstream bug this tree
# inherits.  `build_ext --inplace` runs `setup.py` once and sidesteps it, so
# that is the working path and the one this script drives.  See README.md,
# "Installation" / "MACA (MetaX)".
#
# Usage:
#     ./build.sh                               # this device (CUCC_TARGETS=native)
#     ./build.sh --all                         # every architecture this tree names
#     ./build.sh --targets xcore1000,xcore1600
#     ./build.sh --clean                       # rm -rf build first (full rebuild)
#     ./build.sh --list                        # print resolved targets and exit
#
# Env:
#     CUCC_TARGETS   overrides the targets (same variable and meaning as the
#                    host repository's build.sh); an explicit value always wins
#     MACA_PATH      MACA toolkit root (default /opt/maca)
#     MAX_JOBS       forwarded to ninja as -j
#
set -euo pipefail

# ── project root ────────────────────────────────────────────────────────────
original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: build.sh [options]

  --targets LIST   architectures to build (default: $CUCC_TARGETS or native)
  --all            xcore1000,xcore1500,xcore1600 -- every family this tree names
  --clean          rm -rf build before building (full rebuild)
  --list           print the resolved targets and exit
  -h, --help       this message

Env: CUCC_TARGETS (default native), MACA_PATH (default /opt/maca), MAX_JOBS
EOF
}

# ── environment ─────────────────────────────────────────────────────────────
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# A config's shared-memory footprint is valid for exactly one architecture, and
# torch appends a second `--offload-arch` when this is set -- which would put a
# 128 KiB-sized config into a 64 KiB extension.  `setup.py` refuses to build
# with it set, so clear it here rather than making the caller find out.
if [[ -n "${TORCH_EXTENSION_ENABLE_XC1500_COMPILE:-}" ]]; then
    echo "build.sh: clearing TORCH_EXTENSION_ENABLE_XC1500_COMPILE -- it makes" >&2
    echo "          torch append an architecture to every source; name the" >&2
    echo "          architectures with CUCC_TARGETS instead" >&2
    unset TORCH_EXTENSION_ENABLE_XC1500_COMPILE
fi

# ── arguments ───────────────────────────────────────────────────────────────
do_clean=0
list_only=0
targets_spec="${CUCC_TARGETS:-native}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --targets)
            [[ $# -ge 2 ]] || { echo "build.sh: --targets needs a value" >&2; exit 2; }
            targets_spec="$2"; shift 2 ;;
        --targets=*) targets_spec="${1#*=}"; shift ;;
        --all)       targets_spec="xcore1000,xcore1500,xcore1600"; shift ;;
        --clean)     do_clean=1; shift ;;
        --list)      list_only=1; shift ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "build.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# Resolve through the repo's own `_arch`, so the script and the build cannot
# disagree about what a target means or which kernel tree it lands on.
resolved=$(CUCC_TARGETS="$targets_spec" python - <<'PY'
import os
from deep_select._arch import CAPACITY_BYTES, family_of_target, kernel_directory, resolve_targets

for target in resolve_targets(os.environ.get("CUCC_TARGETS")):
    family = family_of_target(target)
    print(f"{target} {family} {CAPACITY_BYTES[family] // 1024} "
          f"{kernel_directory(family)}")
PY
) || { echo "build.sh: could not resolve CUCC_TARGETS='$targets_spec'" >&2; exit 1; }

echo "build.sh: MACA_PATH=$MACA_PATH"
echo "build.sh: CUCC_TARGETS=$targets_spec"
while read -r target family kibs tree; do
    echo "build.sh:   $target -> xcore$family (${kibs} KiB/SM, csrc/$tree/)"
done <<<"$resolved"
if [[ "$list_only" == 1 ]]; then
    exit 0
fi

# ── clean ───────────────────────────────────────────────────────────────────
if [[ "$do_clean" == 1 ]]; then
    echo "build.sh: rm -rf build"
    rm -rf build
fi

# ── drop the stale in-place extensions ──────────────────────────────────────
# `build_ext --inplace` copies out of build/lib by comparing timestamps, and
# the in-place `.so` sits under `deep_select/` (gitignored), so it is stale by
# default.  A stale one means measuring the previous binary while reading the
# new source -- the handover note records exactly this trap.  Hence: remove
# before building, never trust the timestamp.
while read -r target _family _kibs _tree; do
    for so in deep_select/deep_select_${target}*.so \
              build/lib.*/deep_select/deep_select_${target}*.so; do
        if [[ -e "$so" ]]; then
            echo "build.sh: rm -f $so"
            rm -f "$so"
        fi
    done
done <<<"$resolved"

# ── build ───────────────────────────────────────────────────────────────────
# setup.py runs one extension per architecture, each with its own
# `--offload-arch`, so the loop over targets lives in setup.py and not here.
export CUCC_TARGETS="$targets_spec"

which python
python -c 'import sys, torch
print(f"build.sh: python {sys.version.split()[0]}, torch {torch.__version__}")'

if [[ -n "${MAX_JOBS:-}" ]]; then
    python setup.py build_ext --inplace -j "$MAX_JOBS"
else
    python setup.py build_ext --inplace
fi

# ── verify ──────────────────────────────────────────────────────────────────
# An architecture with no kernel is a hole in `backend="maca_c"` on that
# device, not a smaller build, so a build that produced no extension for a
# requested target is an error here rather than at the first `topk` call.
missing=0
while read -r target family _kibs _tree; do
    found=$(ls deep_select/deep_select_${target}*.so 2>/dev/null | head -1 || true)
    if [[ -n "$found" ]]; then
        echo "build.sh: OK  $target (xcore$family): $(basename "$found")"
    else
        echo "build.sh: ERROR no extension produced for $target" >&2
        missing=1
    fi
done <<<"$resolved"
[[ "$missing" == 0 ]] || exit 1

echo "build.sh: done"
cd "$original_dir"
