#!/usr/bin/env bash
#
# Run the whole performance grid, write it out as CSVs, and compare against the
# baseline.  The "how is this tree doing" entry point; `run_test.sh` is the
# correctness/gating one.
#
# ── The three arms ──────────────────────────────────────────────────────────
#
#   1. `perf_snapshot.py` -- writes the CSV: the OFFICIAL grid (`tests/test.py`'s
#      own `performance_cases()`, called not restated), plus the selector
#      grid, for the backends named in `--arms`.
#   2. `tests/test.py --perf-only` -- the same grid driven by upstream's own file
#      in one process.  Redundant with (1) on purpose: it is the gate.
#   3. `tests/test.py --perf-only --dtype fp32` -- the fp32 Sampler cells.
#
# Nothing here re-implements a measurement -- every arm is upstream's file, and
# `tests/test.py` can neither select a backend nor emit a CSV, which is why the
# snapshot is its own arm rather than a mode of it.
#
# ── Failure, and the baseline ───────────────────────────────────────────────
#
# The FIRST failing arm stops the run: a comparison against a baseline whose
# grid was not fully measured is not a comparison.  The verdict is written to
# `compare_result.txt` before the baseline moves.
#
# `--set-baseline` repoints when every ARM passed, not on the comparator's exit
# status; REGRESSED does not block it but is printed loudly, so repointing past
# a regression is a decision someone made.
#
# ── What this deliberately does NOT do ──────────────────────────────────────
#
# No exclusivity gate: nothing can see which device a process is pinned to, so
# pick one with `CUDA_VISIBLE_DEVICES`.  `mx-smi` is snapshotted for forensics,
# never trusted to decide.
#
# Usage:
#     ./run_bench.sh                          # snapshot + the official gate
#     ./run_bench.sh --set-baseline           # ... and repoint baseline at it
#     ./run_bench.sh --compare-only           # no measurement; latest vs baseline
#     ./run_bench.sh --arms maca_c,torch      # a subset of the three
#     ./run_bench.sh --list                   # print the plan and exit
#     ./run_bench.sh --baseline-dir 20260801  # repoint, do not measure
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection, applied to every arm.
#                           Default: unchanged.
#     DS_BENCH_DIR          output root (default perf_data)
#     DS_BENCH_TIMEOUT      per-arm timeout in seconds (default 5400)
#     MACA_PATH             MACA toolkit root (default /opt/maca).  MACA_HOME is
#                           consulted when this is unset; this one wins if both are set.
#
set -euo pipefail
# `set -e` with `[[ ]] && cmd` as a statement exits when the test is false, so
# every conditional in this script is spelled as an `if`.

original_dir=$(pwd)
script_dir=$(realpath "$(dirname "$0")")
cd "$script_dir"

usage() {
    cat >&2 <<'EOF'
Usage: run_bench.sh [options]

  --arms LIST         comma-separated subset of maca_c,torch,deep_gemm
  --device N          shorthand for CUDA_VISIBLE_DEVICES=N
  --set-baseline      repoint <chip>/baseline at this run (only if all arms pass)
  --baseline-dir DIR  repoint the baseline at DIR and exit without measuring
  --compare-only      compare the latest run against the baseline, measure nothing
  --no-deep-gemm-shapes  do not add the selector shapes to any arm (by default
                      they are added when the `deep_gemm` package can serve
                      them; that backend has no cell on the official bf16 grid,
                      so without them its column is 95 `unsupported` rows)
  --skip-gate         do not run tests/test.py --perf-only (the official grid)
  --results DIR       output root (default perf_data)
  --list              print what would run and exit
  -h, --help          this message

Env: CUDA_VISIBLE_DEVICES, DS_BENCH_DIR, DS_BENCH_TIMEOUT, MACA_PATH (or MACA_HOME)
EOF
}

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

# Whether the `deep_gemm` column can be measured at all is the *package's*
# answer -- does it import, and does it carry the entry `backend="deep_gemm"`
# calls -- and never a question about where its source lives.  Off when it
# cannot serve them, so the run does not spend 25 cells on rows that would all
# read `unsupported`.  `--no-deep-gemm-shapes` overrides it off.
deep_gemm_shapes=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from deep_select import deep_gemm_available
print(1 if deep_gemm_available() else 0)
PY
) || deep_gemm_shapes=0
arms="maca_c,torch,deep_gemm"
set_baseline=0
baseline_dir=""
compare_only=0
skip_gate=0
list_only=0
results_dir="${DS_BENCH_DIR:-perf_data}"
timeout_s="${DS_BENCH_TIMEOUT:-5400}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --no-deep-gemm-shapes)   deep_gemm_shapes=0; shift ;;
        --arms)             [[ $# -ge 2 ]] || { echo "run_bench.sh: --arms needs a value" >&2; exit 2; }
                            arms="$2"; shift 2 ;;
        --arms=*)           arms="${1#*=}"; shift ;;
        --device)           [[ $# -ge 2 ]] || { echo "run_bench.sh: --device needs a value" >&2; exit 2; }
                            export CUDA_VISIBLE_DEVICES="$2"; shift 2 ;;
        --device=*)         export CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
        --set-baseline)     set_baseline=1; shift ;;
        --baseline-dir)     [[ $# -ge 2 ]] || { echo "run_bench.sh: --baseline-dir needs a value" >&2; exit 2; }
                            baseline_dir="$2"; shift 2 ;;
        --baseline-dir=*)   baseline_dir="${1#*=}"; shift ;;
        --compare-only)     compare_only=1; shift ;;
        --skip-gate)        skip_gate=1; shift ;;
        --results)          [[ $# -ge 2 ]] || { echo "run_bench.sh: --results needs a value" >&2; exit 2; }
                            results_dir="$2"; shift 2 ;;
        --results=*)        results_dir="${1#*=}"; shift ;;
        --list)             list_only=1; shift ;;
        -h|--help)          usage; exit 0 ;;
        *)                  echo "run_bench.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# ── the extension under test ────────────────────────────────────────────────
# The **device's** family -- see the same block in `run_test.sh` for why the
# build list is the wrong question (it answers "what did I build first", not
# "what does this device load").
family=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from deep_select._arch import family_of_target, native_target
print(family_of_target(native_target()))
PY
) || { echo "run_bench.sh: could not resolve the target architecture" >&2; exit 1; }

# Named after the **device**, not the arch family: two boards can share one
# family, and the family is recorded per row (`chip`) anyway.  `perf_snapshot.py`
# derives the name from the same call, so the two cannot land in different dirs.
device_dir=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import torch
print((torch.cuda.get_device_name(0) or "").strip().replace(" ", "_"))
PY
) || { echo "run_bench.sh: could not read the device name from torch" >&2; exit 1; }
if [[ -z "$device_dir" ]]; then
    echo "run_bench.sh: torch reports no device name; naming the directory after" >&2
    echo "              the arch family instead (metax_xcore${family})" >&2
    device_dir="metax_xcore${family}"
fi

chip_dir="${results_dir}/${device_dir}"
so=$(ls deep_select/deep_select_maca*.so 2>/dev/null | head -1 || true)
if [[ -z "$so" ]]; then
    echo "run_bench.sh: no extension in deep_select/" >&2
    echo "              build it first:  ./build.sh" >&2
    exit 1
fi

# A header newer than the .so means the extension may not match the sources.
# Reported, not enforced -- the md5 is the record either way, as in run_test.sh.
stale_note="# STALE           no"
if [[ -n "$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1)" ]]; then
    newest_src=$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) | head -1)
    echo "run_bench.sh: $newest_src is newer than $so -- the extension may not" >&2
    echo "              match the sources; the md5 in the manifest is the record." >&2
    stale_note="# STALE           yes ($newest_src newer than the extension)"
fi
md5=$(md5sum "$so" | cut -d' ' -f1)
devices="${CUDA_VISIBLE_DEVICES:-<unset: whatever the host set>}"

latest_result_dir() {
    local candidate name latest=""
    for candidate in "${chip_dir}"/????????_??????*; do
        [[ -d "${candidate}" ]] || continue
        name=${candidate##*/}
        [[ "${name}" =~ ^[0-9]{8}_[0-9]{6}([_-].*)?$ ]] || continue
        [[ -f "${candidate}/manifest.json" ]] || continue
        if [[ -z "${latest}" || "${name}" > "${latest}" ]]; then latest=${name}; fi
    done
    printf '%s\n' "${latest}"
}

if [[ "$list_only" == "1" ]]; then
    echo "run_bench.sh: device dir  = ${device_dir}  (the device's own name;"
    echo "                              the arch family metax_xcore${family} is"
    echo "                              recorded per row as \`chip\`)"
    echo "run_bench.sh: extension   = ${so}"
    echo "run_bench.sh: md5         = ${md5}"
    echo "run_bench.sh: devices     = ${devices}"
    echo "run_bench.sh: arms        = ${arms}"
    echo "run_bench.sh: selector shapes = $([[ ${deep_gemm_shapes} -eq 1 ]] && echo yes || echo no)  (the deep_gemm backend's grid: 25 perf cells at top_k=2048 fp32, plus its correctness shapes on the gate; the only cells it can answer)"
    echo "run_bench.sh: grid        = $([[ ${deep_gemm_shapes} -eq 1 ]] && echo '95 official + 25 selector = 120 cells' || echo '95 official cells only')"
    echo "run_bench.sh: official gate = $([[ ${skip_gate} -eq 1 ]] && echo skipped || echo yes)"
    echo "run_bench.sh: out root    = ${chip_dir}/<YYYYmmdd_HHMMSS>"
    echo "run_bench.sh: baseline    = ${chip_dir}/baseline"
    echo "run_bench.sh: timeout     = ${timeout_s}s per arm"
    exit 0
fi

mkdir -p "$chip_dir"

# ── --baseline-dir: repoint and exit, measuring nothing ─────────────────────
# Moving the baseline back or forward to re-read an old verdict, without paying
# for a re-measure.
if [[ -n "$baseline_dir" ]]; then
    target="${baseline_dir}"
    if [[ ! -d "${chip_dir}/${target}" ]]; then target="${target##*/}"; fi
    if [[ ! -d "${chip_dir}/${target}" ]]; then
        echo "run_bench.sh: no such baseline directory: ${chip_dir}/${baseline_dir}" >&2
        exit 2
    fi
    prev=$(readlink "${chip_dir}/baseline" 2>/dev/null || echo "(none)")
    ln -sfn "${target}" "${chip_dir}/baseline"
    echo "run_bench.sh: baseline ${prev} -> ${target}"
    exit 0
fi

# ── --compare-only: compare the latest run against the baseline, measure nothing
if [[ ${compare_only} -eq 1 ]]; then
    latest=$(latest_result_dir)
    if [[ -z "${latest}" ]]; then
        echo "run_bench.sh: no dated result under ${chip_dir}" >&2; exit 2
    fi
    if [[ ! -d "${chip_dir}/baseline" ]]; then
        echo "run_bench.sh: no baseline at ${chip_dir}/baseline" >&2; exit 2
    fi
    if [[ "${latest}" == "$(readlink "${chip_dir}/baseline")" ]]; then
        echo "run_bench.sh: the latest run IS the baseline; nothing to compare"
        exit 0
    fi
    echo "run_bench.sh: comparing ${latest} against baseline $(readlink "${chip_dir}/baseline")"
    {
        echo "# compare: ${latest} vs baseline $(readlink "${chip_dir}/baseline")"
        python3 tools/compare_snapshots.py "${chip_dir}/${latest}" \
            --base "${chip_dir}/baseline" --device "${device_dir}"
    } 2>&1 | tee "${chip_dir}/${latest}/compare_result.txt" || exit $?
    exit 0
fi

stamp=$(date +%Y%m%d_%H%M%S)
out="${chip_dir}/${stamp}"
mkdir -p "$out"

{
    echo "# run_bench.sh    device_dir=${device_dir}  chip=metax_xcore${family}"
    echo "# timestamp_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host            $(hostname)"
    echo "# extension       ${so}"
    echo "# extension_md5   ${md5}"
    echo "# cuda_visible_devices ${devices}"
    echo "# arms            ${arms}"
    echo "${stale_note}"
    echo "# ---"
    echo "# mx-smi"
    mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
    echo "# pgrep -af python"
    pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
        | sed 's/^/#   /' || echo "#   (none)"
} > "${out}/run_header.txt"

run_arm() {
    local name="$1"; shift
    local log="${out}/${name}.log"
    echo "run_bench.sh: [${name}] -> ${log}"
    set +e
    PYTHONPATH=".:tests" timeout "${timeout_s}" "$@" 2>&1 | tee "$log"
    local rc=${PIPESTATUS[0]}
    set -e
    if [[ ${rc} -eq 124 ]]; then
        echo "run_bench.sh: [${name}] TIMED OUT after ${timeout_s}s" >&2
    elif [[ ${rc} -ne 0 ]]; then
        echo "run_bench.sh: [${name}] FAILED (rc=${rc})" >&2
    fi
    echo "# exit ${rc}" >> "$log"
    return "$rc"
}

bench_status=0
snapshot_failed=0

# arm 1 -- the snapshot.  It writes the CSVs, so its failure is reported as its
# own thing rather than "a bench failure" (it can fail on memory; the gate may
# still pass).
snap_args=(scripts/perf_snapshot.py --arms "${arms}" --out-dir "${results_dir}"
           --tag "${stamp}")
# Without the selector shapes the `deep_gemm` column compares nothing, so they
# ride in the SNAPSHOT whenever they are on (which is the default when the
# package can serve them).
if [[ ${deep_gemm_shapes} -eq 1 ]]; then snap_args+=(--deep-gemm-axes); fi
if ! run_arm snapshot python "${snap_args[@]}"; then
    snapshot_failed=1
    bench_status=1
fi

# arm 2 -- the official grid, upstream's own file.  The gate.
# `python tests/test.py`, not `./tests/test.py`: upstream's file is not
# executable, and a bare path there fails with rc=126 before it runs anything
# (which reads as a bench failure but is a launcher mistake).
gate_deep_gemm=()
if [[ ${deep_gemm_shapes} -eq 1 ]]; then gate_deep_gemm+=(--deep-gemm-shapes); fi
# `DS_TOPK_BACKEND=maca_c` pins the DEFAULT, not the call: `tests/test.py`'s call
# site is unmodified, so what runs is the path a caller who names no backend
# takes.  Without it the library default (`torch`, see `deep_select/interface.py`)
# serves these arms and the gate times the reference.  (`perf_snapshot.py` needs
# nothing -- it passes `backend=` per arm.)
#
# The `env` prefix is load-bearing for arm 3 too: its fp32 Sampler cells are where
# `deep_gemm` raises `UnsupportedByBackend`, so unpinned it would measure no
# kernel at all.
gate_env=(env DS_TOPK_BACKEND=maca_c)
if [[ ${skip_gate} -eq 0 ]]; then
    run_arm official "${gate_env[@]}" python tests/test.py --perf-only -nc "${gate_deep_gemm[@]}" || bench_status=1
fi

# arm 3 -- the official grid's fp32 arm: `tests/test.py` filters its own
# `performance_cases` (bf16), so this is the Sampler cells plus the selector
# shapes.
if [[ ${skip_gate} -eq 0 ]]; then
    run_arm official_fp32 "${gate_env[@]}" python tests/test.py --perf-only -nc --dtype fp32 "${gate_deep_gemm[@]}" || bench_status=1
fi

{
    echo "# run_bench.sh: finished rc=${bench_status}"
    echo "# snapshot_failed ${snapshot_failed}"
    echo "# extension_md5   ${md5}   (${so})"
    echo "# cuda_visible_devices ${devices}"
    echo "# out             ${out}"
} >> "${out}/run_header.txt"

# ── compare against the baseline (before --set-baseline moves it) ───────────
# Unconditional when a baseline exists: the comparator is a local script over
# two CSVs, it never fails the run, and against a missing baseline it is a
# no-op -- so there was nothing for a "do not compare" switch to gate.
if [[ ! -d "${chip_dir}/baseline" ]]; then
    echo "run_bench.sh: no baseline at ${chip_dir}/baseline, skipping comparison"
    if [[ ${set_baseline} -eq 0 ]]; then
        echo "run_bench.sh: run with --set-baseline to create one"
    fi
else
    base_name=$(readlink "${chip_dir}/baseline")
    if [[ "${base_name}" == "${stamp}" ]]; then
        echo "run_bench.sh: this run IS the baseline; nothing to compare"
    else
        echo ""
        echo "============================================================================"
        echo "  Performance comparison: ${stamp} vs baseline ${base_name}"
        echo "============================================================================"
        set +e
        {
            echo "# compare: ${stamp} vs baseline ${base_name}"
            python3 tools/compare_snapshots.py "${out}" \
                --base "${chip_dir}/baseline" --device "${device_dir}" --top 40
        } 2>&1 | tee "${out}/compare_result.txt"
        compare_rc=${PIPESTATUS[0]}
        set -e
        if [[ ${compare_rc} -ne 0 ]]; then
            echo ""
            echo "run_bench.sh: the comparator reports maca_c BEYOND tolerance or a"
            echo "              status change against the baseline (rc=${compare_rc})."
            echo "              Full report: ${out}/compare_result.txt"
            if [[ ${set_baseline} -eq 1 ]]; then
                echo "              --set-baseline was given, so the baseline WILL move"
                echo "              past this.  That is a decision: cite it in the commit."
            fi
        fi
    fi
fi

# ── set the baseline, after the comparison and only on a clean run ──────────
if [[ ${set_baseline} -eq 1 ]]; then
    if [[ ${bench_status} -eq 0 ]]; then
        ln -sfn "${stamp}" "${chip_dir}/baseline"
        echo "run_bench.sh: baseline set: ${chip_dir}/baseline -> ${stamp}"
    else
        echo "run_bench.sh: baseline NOT updated -- an arm failed (rc=${bench_status})" >&2
    fi
fi

echo ""
echo "run_bench.sh: done. results: ${out}"
echo "run_bench.sh: baseline: ${chip_dir}/baseline"
cd "$original_dir"
exit "${bench_status}"
