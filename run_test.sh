#!/usr/bin/env bash
#
# Run this tree's tests the way upstream runs them.
#
# Two arms, both upstream's own, unmodified:
#
#   perf   `tests/test.py --perf-only` -- the whole performance grid in ONE
#          process, exactly as upstream drives it: per case `kk.bench(fn, 10)`
#          (kineto kernel time, L2 flushed) with a `time.sleep(0.2)` cooldown
#          between cases.  Every case is checked before it is timed, so a case
#          that selects wrong is reported as a failure, not as a time.
#
#   test   `scripts/official_slice.py` -- a seeded uniform sample of the same
#          105,138-case correctness table driven through upstream's own
#          `run_testcase`.  (Upstream's full table is hours of GPU time; the
#          driver is this repo's addition, documented in README.)
#
# What this script adds over typing those in is *recording*, not gating:
# whichever extension the run actually loaded, and what the box looked like
# while it ran.  See "Recording" below for why that matters here specifically.
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
# upstream's own flags (`-nc`, `-rf`, `--dtype`) work without this script
# knowing them all -- the ones listed above are only the ones it has to parse.
#
# ── Recording, and what this script deliberately does NOT do ────────────────
#
# It follows the host repository's model (`mcDeepGEMM/run_ci.sh`: "GPU selection:
# CUDA_VISIBLE_DEVICES applies to every stage"; `run_bench.sh`: snapshot mx-smi
# into the log, `|| true`).  Pick the device with the environment, run, record
# the context.  There is no exclusivity gate, and that is on purpose:
#
#   * `pgrep -af python` is what README suggests to a human ("a timing run needs
#     the device to itself: `pgrep -f tests/test.py` first"), but it cannot see
#     which device a process was pinned to.  On this box it fires on a neighbour
#     pinned to device 0 even when the run is on device 3, and the box is
#     shared.  A gate that is wrong most of the time trains people to pass
#     --force without reading it, which is worse than no gate.
#   * `mx-smi` is not a gate either.  Measured here (2026-09-12, MACA 3.7.0.36):
#     `--show-process` printed "no process found" three times while a pytest job
#     was running on device 0, and `--show-all-process` labelled a process
#     holding 4 GB on device 3 as running on GPUs 0, 1 and 2.  It is recorded
#     for forensics, exactly as run_bench.sh records it, not trusted to decide.
#
# The measured cost of getting this wrong is real and is why the md5 matters: a
# 620 us cell in this tree read 619.6 us and 644-651 us in two runs whose
# extension md5 was identical (40a2c663b88a).  ~4% of that is the box, not the
# kernel, and the only way to tell the two apart afterwards is the recorded md5
# plus the recorded mx-smi.
#
# A stale extension is reported loudly but does not stop the run: the arm is
# often being used to check correctness of a source tree whose build you just
# want the *result* of, and refusing costs a round trip.  The md5 is recorded
# either way, so a later reader is never misled about what was measured.
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection; applied to the arm, as run_ci.sh
#                           does.  Default: unchanged (whatever the host set).
#     DS_RESULTS_DIR        same as --results
#     MACA_PATH             MACA toolkit root (default /opt/maca)
#
set -euo pipefail

# ── project root ────────────────────────────────────────────────────────────
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
  --backend NAME       correctness arm: maca_c (default) | torch | deep_gemm
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

# ── environment ─────────────────────────────────────────────────────────────
export MACA_PATH="${MACA_PATH:-/opt/maca}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# ── arguments ───────────────────────────────────────────────────────────────
arm=""
results_dir="${DS_RESULTS_DIR:-results}"
allow_build=0
list_only=0
dtype=""
declare -a arm_args=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --perf)             arm="${arm:+$arm,}perf"; shift ;;
        --test)             arm="${arm:+$arm,}test"; shift ;;
        --all)              arm="perf,test"; shift ;;
        --dtype)            [[ $# -ge 2 ]] || { echo "run_test.sh: --dtype needs a value" >&2; exit 2; }
                            dtype="$2"; arm_args+=("--dtype" "$2"); shift 2 ;;
        --dtype=*)          dtype="${1#*=}"; arm_args+=("$1"); shift ;;
        --sample|--shard|--backend)
                            [[ $# -ge 2 ]] || { echo "run_test.sh: $1 needs a value" >&2; exit 2; }
                            arm_args+=("$1" "$2"); shift 2 ;;
        --sample=*|--shard=*|--backend=*) arm_args+=("$1"); shift ;;
        --results)          [[ $# -ge 2 ]] || { echo "run_test.sh: --results needs a value" >&2; exit 2; }
                            results_dir="$2"; shift 2 ;;
        --results=*)        results_dir="${1#*=}"; shift ;;
        -nc|--no-cooldown|-rf|--run-to-finish) arm_args+=("$1"); shift ;;
        --allow-build)      allow_build=1; shift ;;
        --list)             list_only=1; shift ;;
        # Kept so a script or a habit written against the earlier revision does
        # not die on an unknown flag.  Both are no-ops now: there is no gate.
        --force|--strict)   shift ;;
        -h|--help)          usage; exit 0 ;;
        --)                 shift; arm_args+=("$@"); break ;;
        *)                  echo "run_test.sh: unknown argument: $1" >&2
                            echo "  (pass arm-specific flags after --)" >&2
                            usage; exit 2 ;;
    esac
done
arm="${arm:-perf}"

if [[ -n "${DS_ALLOW_BUSY:-}" ]]; then
    echo "run_test.sh: note: DS_ALLOW_BUSY is obsolete and ignored -- there is no" >&2
    echo "             busy check to override; see the header." >&2
fi

# ── the extension: which one, and is it the one the sources describe ────────
family=$(python - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from deep_select._arch import family_of_target, resolve_targets
targets = resolve_targets(os.environ.get("CUCC_TARGETS"))
print(family_of_target(targets[0]) if targets else "")
PY
) || { echo "run_test.sh: could not resolve the target architecture" >&2; exit 1; }

so=$(ls deep_select/deep_select_xcore${family}*.so 2>/dev/null | head -1 || true)
if [[ -z "$so" ]]; then
    echo "run_test.sh: no extension for xcore${family} in deep_select/" >&2
    echo "             build it first:  ./build.sh" >&2
    if [[ "$allow_build" == "1" ]]; then
        ./build.sh
        so=$(ls deep_select/deep_select_xcore${family}*.so 2>/dev/null | head -1 || true)
    else
        exit 1
    fi
fi

# A header newer than the .so usually means the extension is not what the
# sources describe -- and this tree compiles the whole row kernel from one
# header (`radix_core.cuh`), so it is the common case, not an exotic one.  It
# is reported, not enforced: the md5 below is the record either way.
stale_note="# STALE           no (extension is newer than every csrc source)"
newest_src=$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1 || true)
if [[ -n "$newest_src" ]]; then
    echo "run_test.sh: $newest_src is newer than $so" >&2
    echo "             the extension may not match the sources." >&2
    echo "             extension md5 below is what will actually be measured." >&2
    if [[ "$allow_build" == "1" ]]; then
        echo "run_test.sh: --allow-build: running ./build.sh"
        ./build.sh
        so=$(ls deep_select/deep_select_xcore${family}*.so 2>/dev/null | head -1 || true)
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
    echo "run_test.sh: arm args    = ${arm_args[*]:-(none)}"
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
        # Forensics, not a gate: the same thing run_bench.sh records.  `|| true`
        # because a box without mx-smi must not fail a test run.
        echo "# mx-smi"
        mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
        echo "# pgrep -af python"
        pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
            | sed 's/^/#   /' || echo "#   (none)"
        echo "# ---"
    } > "$log"
    echo "run_test.sh: [$name] -> ${log}"
    # `tee` so a failure is visible live and still recorded.
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
            # Upstream's own invocation, unchanged.
            run_one perf tests/test.py --perf-only "${arm_args[@]+"${arm_args[@]}"}" || rc=1
            ;;
        test)
            run_one test scripts/official_slice.py "${arm_args[@]+"${arm_args[@]}"}" || rc=1
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
