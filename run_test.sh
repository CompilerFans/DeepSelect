#!/usr/bin/env bash
#
# Run this tree's tests the way upstream runs them: `tests/test.py`.
#
#   --perf   tests/test.py --perf-only    the performance grid, one process
#   --test   tests/test.py                a seeded correctness sample
#   --all    both, perf first             (the default is --perf)
#
# The suite is upstream's own file, extended with `--backend` / `--seed` /
# `--sample`; its checks are untouched (`git diff upstream/main HEAD --
# tests/test.py` touches no assertion).  What this adds over typing it in is
# **recording**: which extension the run actually loaded, and what the box
# looked like while it ran.
#
# Usage:
#     ./run_test.sh --perf -nc --dtype bf16
#     ./run_test.sh --test --sample 2000
#     ./run_test.sh --all
#
# Every unknown flag reaches the suite verbatim, so upstream's own (`-nc`,
# `-rf`, `--dtype`) work without this script knowing them all.
#
# There is no exclusivity gate, on purpose: `pgrep` cannot see which device a
# process is pinned to, and `mx-smi` reports processes against the wrong device
# and reports none while a job runs.  Pick the device with
# `CUDA_VISIBLE_DEVICES`, run, and read the recorded md5 before comparing two
# runs -- the `.so` is gitignored, so "what did I measure" is not implied by the
# source.  A stale extension is reported loudly but does not stop the run: the
# md5 is recorded either way, so a later reader is never misled.
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection; applied to the suite.
#     DS_RESULTS_DIR        same as --results
#     MACA_PATH             MACA toolkit root (default /opt/maca; MACA_HOME is
#                           consulted when this is unset, this one wins if both).
#     DS_TOPK_BACKEND       what a call with no `backend=` runs (the library
#                           default is `torch`).  **It does not reach this
#                           suite**, whose call always carries `--backend`; use
#                           the flag, and reach for the variable when what you
#                           want to test is an *unpinned* caller.
#
set -euo pipefail
cd "$(realpath "$(dirname "$0")")"

usage() {
    cat >&2 <<'EOF'
Usage: run_test.sh [--perf | --test | --all] [options]

  --perf             tests/test.py --perf-only   (the official perf grid)
  --test             tests/test.py               (correctness sample)
  --all              both, perf first
  --results DIR      where the log + receipt go (default results/)
  --allow-build      build first (./develop.sh) if no extension is found
  -h, --help         this message

Everything else is forwarded to tests/test.py verbatim:
  --dtype bf16|fp32, -nc / --no-cooldown, -rf / --run-to-finish,
  --sample N, --seed S, --backend NAME

Env: CUDA_VISIBLE_DEVICES, DS_RESULTS_DIR, MACA_PATH (or MACA_HOME)
EOF
}

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

suite=""
results_dir="${DS_RESULTS_DIR:-results}"
allow_build=0
declare -a suite_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf)         suite="${suite:+$suite,}perf"; shift ;;
        --test)         suite="${suite:+$suite,}test"; shift ;;
        --all)          suite="perf,test"; shift ;;
        --results)      results_dir="${2:?run_test.sh: --results needs a value}"; shift 2 ;;
        --results=*)    results_dir="${1#*=}"; shift ;;
        --allow-build)  allow_build=1; shift ;;
        -h|--help)      usage; exit 0 ;;
        --)             shift; suite_args+=("$@"); break ;;
        *)              suite_args+=("$1"); shift ;;
    esac
done
suite="${suite:-perf}"

# The correctness suite's defaults are the deleted driver's: the seed it
# defaulted to (20260911), a 200-case draw rather than the whole 105,138-case
# table, and `-rf` so a case that goes wrong is recorded and the run still
# reaches its summary.  The seed is the part that is easy to drop and expensive
# to lose: without it `--sample 200` draws a *different* 200 every run, and
# "200/200" stops being a receipt anyone can re-run.  A caller's own `--seed` /
# `--sample` is appended after these and argparse keeps the last occurrence.
declare -a test_args=(--seed 20260911 --sample 200 -rf)

# ── the extension: which one ────────────────────────────────────────────────
# Asked of the **package that will answer** -- `_binding.extension_path` is the
# same call `_binding.load` makes, so the md5 below is the artifact the process
# actually loads.  A checkout with a built extension is tested as the checkout;
# without one, an installed `deep_select` answers, so a wheel `./install.sh`
# put in site-packages is testable with no build step.
#
# The cwd comes off `sys.path` first: `python -` starts with it there, and with
# the tree reachable the import lands on its `deep_select/`, which imports fine
# and then fails at call time (the extension loads lazily).
so=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import glob, os, sys
here = os.getcwd()
sys.path[:] = [p for p in sys.path if p not in ("", here)]
if glob.glob(os.path.join(here, "deep_select", "deep_select_maca*.so")):
    sys.path.insert(0, here)
try:
    from deep_select.interface import _KERNEL_NAME, _binding
    print(_binding.extension_path(_KERNEL_NAME) or "")
except Exception:
    print("")
PY
)

if [[ -z "$so" ]]; then
    echo "run_test.sh: no extension found, in the checkout or installed." >&2
    if [[ "$allow_build" == "1" ]]; then
        ./develop.sh
        so="$PWD/$(ls deep_select/deep_select_maca*.so | head -1)"
    else
        echo "             build one (./develop.sh) or install one (./install.sh)." >&2
        exit 1
    fi
fi

# `$REPO` goes on the suite's path only when it is the tree answering -- the two
# must agree, or the run measures one artifact and receipts another.
stage_pythonpath=""
if [[ "$so" == "$PWD"/deep_select/* ]]; then stage_pythonpath="."; fi

# A header newer than the extension usually means it is not what the sources
# describe: this tree compiles the whole row kernel from one header
# (`radix_core.cuh`).  Reported, not enforced -- the md5 is the record either
# way.  Only a checkout can answer it; a wheel has no `csrc/` to compare against,
# and claiming "not stale" there would be asserting a check that never ran.
stale_note="# STALE           n/a (installed package; no csrc/ to compare against)"
if [[ "$so" == "$PWD"/deep_select/* ]]; then
    stale_note="# STALE           no (extension is newer than every csrc source)"
    newest_src=$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1 || true)
    if [[ -n "$newest_src" ]]; then
        echo "run_test.sh: $newest_src is newer than $so; the extension may not" >&2
        echo "             match the sources -- the md5 below is what is measured." >&2
        stale_note="# STALE           yes ($newest_src newer than the extension)"
    fi
fi

md5=$(md5sum "$so" | cut -d' ' -f1)
stamp=$(date +%Y%m%d_%H%M%S)
receipt="${results_dir}/deepselect_run_${stamp}"
devices="${CUDA_VISIBLE_DEVICES:-<unset: whatever the host set>}"
mkdir -p "$results_dir"

# ── run ─────────────────────────────────────────────────────────────────────
run_one() {
    local name="$1"; shift
    local log="${receipt}_${name}.log" rc=0
    {
        echo "# run_test.sh     suite=${name}"
        echo "# timestamp_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# host            $(hostname)"
        echo "# extension       ${so}"
        echo "# extension_md5   ${md5}"
        echo "# cuda_visible_devices ${devices}"
        echo "${stale_note}"
        echo "# ---"
        # Forensics, not a gate; the fallbacks are so a box without mx-smi
        # cannot fail a test run.
        echo "# mx-smi"
        mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
        echo "# pgrep -af python"
        pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
            | sed 's/^/#   /' || echo "#   (none)"
        echo "# ---"
        echo "# argv            $*"
    } > "$log"
    echo "run_test.sh: [$name] -> ${log}"
    PYTHONPATH="${stage_pythonpath}" python "$@" 2>&1 | tee -a "$log" || rc=$?
    echo "# ---" >> "$log"
    echo "# exit            ${rc}" >> "$log"
    return "$rc"
}

rc=0
IFS=',' read -ra suites <<<"$suite"
for a in "${suites[@]}"; do
    case "$a" in
        perf) run_one perf tests/test.py --perf-only "${suite_args[@]+"${suite_args[@]}"}" || rc=1 ;;
        test) run_one test tests/test.py "${test_args[@]}" "${suite_args[@]+"${suite_args[@]}"}" || rc=1 ;;
        *)    echo "run_test.sh: unknown suite '$a'" >&2; exit 2 ;;
    esac
done

{
    echo "# run_test.sh: ${suite} finished rc=${rc}"
    echo "# extension_md5 ${md5}   (${so})"
    echo "# cuda_visible_devices ${devices}"
} >> "${receipt}.txt"
echo "run_test.sh: receipt: ${receipt}.txt"
exit "$rc"
