#!/usr/bin/env bash
#
# Run this tree's tests the way upstream runs them.
#
# Two suites, one file: `tests/test.py --perf-only` (the whole performance grid
# in ONE process, every case checked before it is timed) and `tests/test.py`
# (a seeded sample of the correctness table, run through the same
# `run_testcase`).  The correctness half used to be a separate driver
# (`scripts/official_slice.py`, deleted 2026-09-16) whose `--backend` /
# `--seed` / `--sample` now live in the official file; "upstream's own file,
# extended" is the accurate description, not "unmodified" -- the checks
# themselves are untouched, and that is checkable (`git diff upstream/main HEAD
# -- tests/test.py` touches no assertion).
#
# What this adds over typing those in is *recording*, not gating: whichever
# extension the run actually loaded, and what the box looked like while it ran.
#
# Usage:
#     ./run_test.sh --perf                    # the performance grid
#     ./run_test.sh --perf -nc                # ... without the per-case cooldowns
#     ./run_test.sh --perf --dtype bf16       # 90 of the 95 cases
#     ./run_test.sh --test                    # correctness sample (default 200)
#     ./run_test.sh --test --sample 2000
#     ./run_test.sh --all                     # perf then correctness
#
# Anything after the known options is forwarded to the suite verbatim, so
# upstream's own flags (`-nc`, `-rf`, `--dtype`) work without this script knowing
# them all -- the ones listed are only the ones it has to parse.
#
# ── Recording, and what this script deliberately does NOT do ────────────────
#
# Pick the device with `CUDA_VISIBLE_DEVICES`, run, record the context.  There
# is no exclusivity gate, on purpose:
#
#   * `pgrep -af python` cannot see which device a process is pinned to -- it
#     fires on a neighbour pinned elsewhere, and the box is shared.  A gate that
#     is wrong most of the time trains people to stop reading it, which is worse
#     than no gate.
#   * `mx-smi` is not a gate either: it reports processes against the wrong
#     device, and reports none while a job runs.
#
# The consequence for the CLI is that there is no `--force`/`--strict` to
# override a gate, and asking for one is an unknown argument rather than a
# silent success: a flag that *asserts* a check ran when there is no check is
# the failure mode this section is about, so it is not kept for compatibility.
#
# The md5 is recorded because this tree's `.so` is gitignored, so "what did I
# measure" is not implied by the source -- and it has already been the thing
# that made a confusing timing difference findable.  A stale extension is
# reported loudly but does not stop the run (the suite is often checking a source
# tree whose build you only want the *result* of, and refusing costs a round
# trip); the md5 is recorded either way, so a later reader is never misled.
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection; applied to the suite.  Default:
#                           unchanged (whatever the host set).
#     DS_RESULTS_DIR        same as --results
#     MACA_PATH             MACA toolkit root (default /opt/maca).  MACA_HOME is
#                           consulted when this is unset; this one wins if both are set.
#     DS_TOPK_BACKEND       which implementation a call with no `backend=` runs
#                           (the library default is `torch`, the reference).
#                           **It does not reach these suites any more**: both go
#                           through `tests/test.py`, whose call always carries
#                           `backend=` explicitly (see `--backend` under Options).
#                           Use `--backend` here; reach for this variable when
#                           what you want to test is an *unpinned* caller.
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: run_test.sh [--perf | --test | --all] [options] [-- <suite args>]

Suites (default --perf):
  --perf               tests/test.py --perf-only   (the official perf grid)
  --test               tests/test.py               (correctness sample)
  --all                both, perf first

Options:
  --dtype DTYPE        bf16 | fp32 (forwarded to the suite)
  -nc, --no-cooldown   forwarded to tests/test.py; skip the per-case sleeps
  --sample N           correctness suite: how many cases (default 200; the
                       table is 105,138, which is hours)
  --seed S             seed the table before it is drawn (default 20260911, so
                       `--sample 200` is the same 200 every run).  Pass it
                       after `--` to override.
  --backend NAME       correctness suite: maca_c (the default), torch, or
                       deep_gemm.  Passed to `deep_select.topk` at the call site,
                       so the default tests the kernel rather than the reference.
                       (DS_TOPK_BACKEND -- what an *unpinned* caller resolves to
                       -- does NOT reach this suite: the call here always carries
                       a backend.  To test the default, call the library.)
  --results DIR        where the log + receipt go (default results/)
  --allow-build        build first if the extension is missing or older than
                       the sources (default: run anyway, and say so in the log)
  -h, --help           this message

Anything after `--` goes to the suite verbatim.
There is no exclusivity gate: pick the device with CUDA_VISIBLE_DEVICES, and
read the recorded md5 + mx-smi snapshot before comparing two runs.

Env: CUDA_VISIBLE_DEVICES, DS_RESULTS_DIR, MACA_PATH (or MACA_HOME)
EOF
}

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

suite=""
results_dir="${DS_RESULTS_DIR:-results}"
allow_build=0
dtype=""
declare -a perf_args=()      # --dtype, -nc/-rf: tests/test.py knows these
# The correctness suite's defaults are the ones the deleted driver applied:
# the fixed seed it defaulted to (`official_slice.py:144`, default 20260911),
# 200 cases rather than the whole 105,138-case table, and `-rf` so a case that
# goes wrong is recorded and the run still reaches its summary.  The seed is
# the part that is easy to drop by accident and expensive to lose: without it
# `--sample 200` draws a **different** 200 every run, so "200/200" stops being
# a receipt anyone can re-run -- this tree's evidence is a command plus the
# number it printed, and an unseeded draw makes the two unrelated.
#
# It lives here and not in `tests/test.py`, whose `--seed` unset must stay "the
# table upstream builds", i.e. unseeded, for the same reason `--backend` unset
# is upstream's call site.  A caller's own `--sample` / `--backend` is appended
# after these and argparse keeps the last occurrence, so the caller wins;
# `--seed` is reached the same way, after `--`.
declare -a test_args=(--seed 20260911 --sample 200 -rf)

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf)             suite="${suite:+$suite,}perf"; shift ;;
        --test)             suite="${suite:+$suite,}test"; shift ;;
        --all)              suite="perf,test"; shift ;;
        --dtype)            [[ $# -ge 2 ]] || { echo "run_test.sh: --dtype needs a value" >&2; exit 2; }
                            dtype="$2"; perf_args+=("--dtype" "$2"); shift 2 ;;
        --dtype=*)          dtype="${1#*=}"; perf_args+=("$1"); shift ;;
        --sample|--backend)
                            [[ $# -ge 2 ]] || { echo "run_test.sh: $1 needs a value" >&2; exit 2; }
                            test_args+=("$1" "$2"); shift 2 ;;
        --sample=*|--backend=*) test_args+=("$1"); shift ;;
        --results)          [[ $# -ge 2 ]] || { echo "run_test.sh: --results needs a value" >&2; exit 2; }
                            results_dir="$2"; shift 2 ;;
        --results=*)        results_dir="${1#*=}"; shift ;;
        # Both suites are `tests/test.py` now, so these reach both; they are
        # kept in perf_args only, so one command line carries one copy.
        -nc|--no-cooldown|-rf|--run-to-finish) perf_args+=("$1"); shift ;;
        --allow-build)      allow_build=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift
                            # After `--`, split on what the flag is: the two
                            # suites do not share a CLI, and forwarding the union
                            # to both was a bug (`--dtype` with `--perf-only`
                            # asks the perf grid a question it cannot answer).
                            while [[ $# -gt 0 ]]; do
                                case "$1" in
                                    --dtype|--dtype=*) perf_args+=("$1"); shift ;;
                                    -nc|--no-cooldown|-rf|--run-to-finish) perf_args+=("$1"); shift ;;
                                    *) test_args+=("$1"); shift ;;
                                esac
                            done
                            break ;;
        *)                  echo "run_test.sh: unknown argument: $1" >&2
                            echo "  (pass suite-specific flags after --)" >&2
                            usage; exit 2 ;;
    esac
done
suite="${suite:-perf}"

# ── the extension: which one, and is it the one the sources describe ────────
# The **device's** family, not the build list's first entry: a tree built for
# several architectures ships one `.so` each, and only this tells you which the
# device in front of you loads.  (`resolve_targets(CUCC_TARGETS)[0]` would name
# whichever was built first -- right only while `CUCC_TARGETS` happens to lead
# with this device's family, which is how a receipt ends up naming the C500
# artifact of a C600U measurement.)
family=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from deep_select._arch import family_of_target, native_target
print(family_of_target(native_target()))
PY
) || { echo "run_test.sh: could not resolve the target architecture" >&2; exit 1; }

# One extension for every family, so this does not depend on the device -- but
# the *note* below still names this device's family, because whether the .so
# carries an image for it is a real question a one-image local build answers
# with a launch failure.
so=$(ls deep_select/deep_select_maca*.so 2>/dev/null | head -1 || true)
if [[ -z "$so" ]]; then
    echo "run_test.sh: no extension in deep_select/" >&2
    echo "             build it first:  ./develop.sh" >&2
    if [[ "$allow_build" == "1" ]]; then
        ./develop.sh
        so=$(ls deep_select/deep_select_maca*.so 2>/dev/null | head -1 || true)
    else
        exit 1
    fi
fi

# A header newer than the .so usually means the extension is not what the
# sources describe -- this tree compiles the whole row kernel from one header
# (`radix_core.cuh`), so it is the common case.  Reported, not enforced: the md5
# below is the record either way.
stale_note="# STALE           no (extension is newer than every csrc source)"
newest_src=$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1 || true)
if [[ -n "$newest_src" ]]; then
    echo "run_test.sh: $newest_src is newer than $so" >&2
    echo "             the extension may not match the sources." >&2
    echo "             extension md5 below is what will actually be measured." >&2
    if [[ "$allow_build" == "1" ]]; then
        echo "run_test.sh: --allow-build: running ./develop.sh"
        ./develop.sh
        so=$(ls deep_select/deep_select_maca*.so 2>/dev/null | head -1 || true)
        stale_note="# STALE           rebuilt by --allow-build before this run"
    else
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
    local log="${receipt}_${name}.log"
    {
        echo "# run_test.sh     suite=${name}"
        echo "# timestamp_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
        echo "# host            $(hostname)"
        echo "# extension       ${so}"
        echo "# extension_md5   ${md5}"
        echo "# cuda_visible_devices ${devices}"
        echo "# dtype           ${dtype:-(all)}"
        echo "# argv            $*"
        echo "$stale_note"
        echo "# ---"
        # Forensics, not a gate; `|| true` because a box without mx-smi must not
        # fail a test run.
        echo "# mx-smi"
        mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
        echo "# pgrep -af python"
        pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
            | sed 's/^/#   /' || echo "#   (none)"
        echo "# ---"
    } > "$log"
    echo "run_test.sh: [$name] -> ${log}"
    set +e
    PYTHONPATH=. python "$@" 2>&1 | tee -a "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    echo "# ---" >> "$log"
    echo "# exit            ${rc}" >> "$log"
    return "$rc"
}

rc=0
IFS=',' read -ra suites <<<"$suite"
for a in "${suites[@]}"; do
    case "$a" in
        perf)
            run_one perf tests/test.py --perf-only "${perf_args[@]+"${perf_args[@]}"}" || rc=1
            ;;
        test)
            run_one test tests/test.py "${test_args[@]+"${test_args[@]}"}" || rc=1
            ;;
        *) echo "run_test.sh: unknown suite '$a'" >&2; exit 2 ;;
    esac
done

{
    echo "# run_test.sh: ${suite} finished rc=${rc}"
    echo "# extension_md5 ${md5}   (${so})"
    echo "# cuda_visible_devices ${devices}"
} >> "${receipt}.txt"
echo "run_test.sh: receipt: ${receipt}.txt"

cd "$original_dir"
exit "$rc"
