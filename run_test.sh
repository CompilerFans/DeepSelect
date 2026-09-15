#!/usr/bin/env bash
#
# Run this tree's tests the way upstream runs them.
#
# Two arms, both upstream's own, unmodified: `tests/test.py --perf-only` (the
# whole performance grid in ONE process, every case checked before it is timed)
# and `scripts/official_slice.py` (a seeded sample of the correctness table
# through upstream's own `run_testcase`).
#
# What this adds over typing those in is *recording*, not gating: whichever
# extension the run actually loaded, and what the box looked like while it ran.
#
# Usage:
#     ./run_test.sh --perf                    # the performance grid
#     ./run_test.sh --perf -nc                # ... without the per-case cooldowns
#     ./run_test.sh --perf --dtype bf16       # 90 of the 95 cases
#     ./run_test.sh --test                    # correctness sample (default 200)
#     ./run_test.sh --test --sample 1000000 --shard 0/4
#     ./run_test.sh --all                     # perf then correctness
#     ./run_test.sh --list                    # print the plan and exit
#
# Anything after the known options is forwarded to the arm verbatim, so
# upstream's own flags (`-nc`, `-rf`, `--dtype`) work without this script knowing
# them all -- the ones listed are only the ones it has to parse.
#
# ── Recording, and what this script deliberately does NOT do ────────────────
#
# Follows the host repository's model (`mcDeepGEMM/run_ci.sh`: "GPU selection:
# CUDA_VISIBLE_DEVICES applies to every stage").  Pick the device with the
# environment, run, record the context.  There is no exclusivity gate, on
# purpose:
#
#   * `pgrep -af python` cannot see which device a process is pinned to -- it
#     fires on a neighbour pinned elsewhere, and the box is shared.  A gate that
#     is wrong most of the time trains people to stop reading it, which is worse
#     than no gate.
#   * `mx-smi` is not a gate either; measured here lying in both directions.
#
# The consequence for the CLI is that there is no `--force`/`--strict` to
# override a gate, and asking for one is an unknown argument rather than a
# silent success: a flag that *asserts* a check ran when there is no check is
# the failure mode this section is about, so it is not kept for compatibility.
#
# The md5 is recorded because this tree's `.so` is gitignored, so "what did I
# measure" is not implied by the source -- and it has already been the thing
# that made a confusing timing difference findable.  A stale extension is
# reported loudly but does not stop the run (the arm is often checking a source
# tree whose build you only want the *result* of, and refusing costs a round
# trip); the md5 is recorded either way, so a later reader is never misled.
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection; applied to the arm.  Default:
#                           unchanged (whatever the host set).
#     DS_RESULTS_DIR        same as --results
#     MACA_PATH             MACA toolkit root (default /opt/maca)
#     DS_TOPK_BACKEND       which implementation a call with no `backend=` runs
#                           (the library default is `torch`, the reference),
#                           honoured by BOTH arms; `run_bench.sh` sets `maca_c`
#                           for its gate arms for the same reason.
#
set -euo pipefail

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: run_test.sh [--perf | --test | --all] [options] [-- <arm args>]

Arms (default --perf):
  --perf               tests/test.py --perf-only   (the official perf grid)
  --test               scripts/official_slice.py   (correctness sample)
  --all                both, perf first

Options:
  --dtype DTYPE        bf16 | fp32 (forwarded to the arm)
  -nc, --no-cooldown   forwarded to tests/test.py; skip the per-case sleeps
  --sample N           correctness arm: how many cases (default 200)
  --shard I/N          correctness arm: run shard I of N
  --backend NAME       correctness arm, PINNED call: maca_c | torch | deep_gemm.
                       Unset = the library's own default (torch), exercised
                       through the unmodified official call site.
  --default-arm NAME   correctness arm, UNPINNED call: set the process default
                       to NAME and leave `deep_select.topk(...)` as written.
                       "maca_c" here = "the default serves the kernel", which
                       --backend maca_c does not test (it pins the call).
  --results DIR        where the log + receipt go (default results/)
  --allow-build        build first if the extension is missing or older than
                       the sources (default: run anyway, and say so in the log)
  --list               print what would run and exit
  -h, --help           this message

Anything after `--` goes to the arm verbatim.
There is no exclusivity gate: pick the device with CUDA_VISIBLE_DEVICES, and
read the recorded md5 + mx-smi snapshot before comparing two runs.

Env: CUDA_VISIBLE_DEVICES, DS_RESULTS_DIR, MACA_PATH
EOF
}

export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

arm=""
results_dir="${DS_RESULTS_DIR:-results}"
allow_build=0
list_only=0
dtype=""
declare -a perf_args=()      # --dtype, -nc/-rf: tests/test.py knows these
declare -a test_args=()      # --sample, --shard, --backend: official_slice.py does

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf)             arm="${arm:+$arm,}perf"; shift ;;
        --test)             arm="${arm:+$arm,}test"; shift ;;
        --all)              arm="perf,test"; shift ;;
        --dtype)            [[ $# -ge 2 ]] || { echo "run_test.sh: --dtype needs a value" >&2; exit 2; }
                            dtype="$2"; perf_args+=("--dtype" "$2"); shift 2 ;;
        --dtype=*)          dtype="${1#*=}"; perf_args+=("$1"); shift ;;
        --sample|--shard|--backend|--default-arm)
                            [[ $# -ge 2 ]] || { echo "run_test.sh: $1 needs a value" >&2; exit 2; }
                            test_args+=("$1" "$2"); shift 2 ;;
        --sample=*|--shard=*|--backend=*|--default-arm=*) test_args+=("$1"); shift ;;
        --results)          [[ $# -ge 2 ]] || { echo "run_test.sh: --results needs a value" >&2; exit 2; }
                            results_dir="$2"; shift 2 ;;
        --results=*)        results_dir="${1#*=}"; shift ;;
        # tests/test.py knows these (they come from lib.stick_unit_test_args);
        # official_slice.py does not, so they must not reach the test arm.
        -nc|--no-cooldown|-rf|--run-to-finish) perf_args+=("$1"); shift ;;
        --allow-build)      allow_build=1; shift ;;
        --list)             list_only=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift
                            # After `--`, split on what the flag is: the two
                            # arms do not share a CLI, and forwarding the union
                            # to both was a bug (official_slice.py exits 2 on
                            # `--dtype`, which the default `--all` would hit).
                            while [[ $# -gt 0 ]]; do
                                case "$1" in
                                    --dtype|--dtype=*) perf_args+=("$1"); shift ;;
                                    -nc|--no-cooldown|-rf|--run-to-finish) perf_args+=("$1"); shift ;;
                                    *) test_args+=("$1"); shift ;;
                                esac
                            done
                            break ;;
        *)                  echo "run_test.sh: unknown argument: $1" >&2
                            echo "  (pass arm-specific flags after --)" >&2
                            usage; exit 2 ;;
    esac
done
arm="${arm:-perf}"

# ── the extension: which one, and is it the one the sources describe ────────
# The **device's** family, not the build list's first entry: a tree built for
# several architectures ships one `.so` each, and only this tells you which the
# device in front of you loads.  (`resolve_targets(CUCC_TARGETS)[0]` would name
# whichever was built first -- right only while `CUCC_TARGETS` happens to lead
# with this device's family, which is how a receipt ends up naming the C500
# artifact of a C600U measurement.)
family=$(python - <<'PY'
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
    echo "             build it first:  ./build.sh" >&2
    if [[ "$allow_build" == "1" ]]; then
        ./build.sh
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
        echo "run_test.sh: --allow-build: running ./build.sh"
        ./build.sh
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

if [[ "$list_only" == "1" ]]; then
    echo "run_test.sh: arm(s)      = $arm"
    echo "run_test.sh: extension   = $so"
    echo "run_test.sh: md5         = $md5"
    echo "run_test.sh: devices     = $devices"
    echo "run_test.sh: dtype       = ${dtype:-(all)}"
    echo "run_test.sh: perf args   = ${perf_args[*]:-(none)}"
    echo "run_test.sh: test args   = ${test_args[*]:-(none)}"
    echo "run_test.sh: receipt     = ${receipt}.txt"
    exit 0
fi

mkdir -p "$results_dir"

# ── run ─────────────────────────────────────────────────────────────────────
run_one() {
    local name="$1"; shift
    local log="${receipt}_${name}.log"
    {
        echo "# run_test.sh     arm=${name}"
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
IFS=',' read -ra arms <<<"$arm"
for a in "${arms[@]}"; do
    case "$a" in
        perf)
            run_one perf tests/test.py --perf-only "${perf_args[@]+"${perf_args[@]}"}" || rc=1
            ;;
        test)
            run_one test scripts/official_slice.py "${test_args[@]+"${test_args[@]}"}" || rc=1
            ;;
        *) echo "run_test.sh: unknown arm '$a'" >&2; exit 2 ;;
    esac
done

{
    echo "# run_test.sh: ${arm} finished rc=${rc}"
    echo "# extension_md5 ${md5}   (${so})"
    echo "# cuda_visible_devices ${devices}"
} >> "${receipt}.txt"
echo "run_test.sh: receipt: ${receipt}.txt"

cd "$original_dir"
exit "$rc"
