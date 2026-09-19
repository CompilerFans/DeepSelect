#!/usr/bin/env bash
#
# Measure the performance grid, write it out as CSVs, and compare against the
# baseline.  The "how is this tree doing" entry point; `run_test.sh` is the
# correctness/gating one.
#
# Stages, in order, all under `perf_data/<device>/<stamp>/`:
#
#   1. snapshot       `scripts/perf_snapshot.py` writes the CSV -- the official
#                     grid plus `--cases-file`, one row per (cell, backend).
#   2. official       `tests/test.py --perf-only -nc`, the same grid driven by
#                     upstream's own file.  Redundant with (1) on purpose: it is
#                     the gate.
#   3. official_fp32  the same, `--dtype fp32` -- the Sampler cells.
#
# Every stage is upstream's file; nothing here re-implements a measurement.
# `--skip-gate` drops 2 and 3, which is the only way to measure without gating.
#
# Usage:
#     ./run_bench.sh                          # snapshot + the official gate
#     ./run_bench.sh --set-baseline           # ... and repoint baseline at it
#     ./run_bench.sh --compare-only           # no measurement; latest vs baseline
#     ./run_bench.sh --baseline-dir 20260801  # repoint, measure nothing
#     ./run_bench.sh --backends maca_c,torch  # a subset of the three
#
# Failure, and the baseline: a failing stage is reported and the run continues
# to the summary, but it makes the run ineligible for `--set-baseline` -- a
# baseline whose grid was not fully measured is not a baseline.  The verdict is
# written to `compare_result.txt` before the baseline moves.  `--set-baseline`
# repoints on the stages' status, not on the comparator's exit status --
# REGRESSED does not block it but is printed loudly, so repointing past a
# regression is a decision someone made, and one to cite in the commit.
#
# No exclusivity gate, for the same reason as `run_test.sh`: nothing can see
# which device a process is pinned to, so pick one with `CUDA_VISIBLE_DEVICES`.
# `mx-smi` is snapshotted for forensics, never trusted to decide.  **A contended
# run is not a record**: if the header shows another torch process, do not pass
# `--set-baseline` -- every backend moves against a bit-identical `.so`, and
# `--compare-only` will show the reference backend drifting too.
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection, applied to every stage.
#     DS_BENCH_DIR          output root (default perf_data)
#     DS_BENCH_TIMEOUT      per-stage timeout in seconds (default 5400)
#     MACA_PATH             MACA toolkit root (default /opt/maca; MACA_HOME is
#                           consulted when this is unset, this one wins if both).
#
set -euo pipefail
cd "$(realpath "$(dirname "$0")")"

usage() {
    cat >&2 <<'EOF'
Usage: run_bench.sh [options]

  --backends LIST     comma-separated subset of maca_c,torch,deep_gemm
  --device N          shorthand for CUDA_VISIBLE_DEVICES=N
  --set-baseline      repoint <chip>/baseline at this run (only if all stages pass)
  --baseline-dir DIR  repoint the baseline at DIR and exit without measuring
  --compare-only      compare the latest run against the baseline, measure nothing
  --skip-gate         do not run tests/test.py --perf-only (the official grid)
  --results DIR       output root (default perf_data)
  -h, --help          this message

Env: CUDA_VISIBLE_DEVICES, DS_BENCH_DIR, DS_BENCH_TIMEOUT, MACA_PATH (or MACA_HOME)
EOF
}

export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$MACA_PATH/ompi/lib:${LD_LIBRARY_PATH:-}"

backends="maca_c,torch,deep_gemm"
set_baseline=0
baseline_dir=""
compare_only=0
skip_gate=0
results_dir="${DS_BENCH_DIR:-perf_data}"
timeout_s="${DS_BENCH_TIMEOUT:-5400}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends)       backends="${2:?run_bench.sh: --backends needs a value}"; shift 2 ;;
        --backends=*)     backends="${1#*=}"; shift ;;
        --device)         export CUDA_VISIBLE_DEVICES="${2:?run_bench.sh: --device needs a value}"; shift 2 ;;
        --device=*)       export CUDA_VISIBLE_DEVICES="${1#*=}"; shift ;;
        --results)        results_dir="${2:?run_bench.sh: --results needs a value}"; shift 2 ;;
        --results=*)      results_dir="${1#*=}"; shift ;;
        --baseline-dir)   baseline_dir="${2:?run_bench.sh: --baseline-dir needs a value}"; shift 2 ;;
        --baseline-dir=*) baseline_dir="${1#*=}"; shift ;;
        --set-baseline)   set_baseline=1; shift ;;
        --compare-only)   compare_only=1; shift ;;
        --skip-gate)      skip_gate=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "run_bench.sh: unknown argument: $1" >&2; usage; exit 2 ;;
    esac
done

# ── the extension, and the device the record belongs to ─────────────────────
# Same resolution as `run_test.sh`, for the same reason: the package that will
# answer names its own artifact, so an installed wheel is measured as itself and
# the md5 in the manifest is the one the process loads.  See that file for why
# the cwd is taken off `sys.path` first.
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
    echo "run_bench.sh: no extension found, in the checkout or installed." >&2
    echo "              build one (./develop.sh) or install one (./install.sh)." >&2
    exit 1
fi

# `$REPO` goes on the stages' path only when it is the tree answering; `tests`
# is always there (the grid lives in the repo, not in the wheel).
stage_pythonpath="tests"
if [[ "$so" == "$PWD"/deep_select/* ]]; then
    stage_pythonpath=".:tests"
else
    echo "run_bench.sh: measuring the installed package's extension:" >&2
    echo "              $so" >&2
fi

# A header newer than the extension means it may not match the sources.
# Reported, not enforced -- the md5 in the manifest is the record either way.
# Only a checkout can answer it; a wheel has no `csrc/` to compare against.
stale_note="# STALE           n/a (installed package; no csrc/ to compare against)"
if [[ "$so" == "$PWD"/deep_select/* ]]; then
    stale_note="# STALE           no"
    newest_src=$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1 || true)
    if [[ -n "$newest_src" ]]; then
        echo "run_bench.sh: $newest_src is newer than $so -- the extension may not" >&2
        echo "              match the sources; the md5 in the manifest is the record." >&2
        stale_note="# STALE           yes ($newest_src newer than the extension)"
    fi
fi

# The record is named after the **device**, not the arch family: two boards can
# share one family, and the family is recorded per row (`chip`) anyway.
# `perf_snapshot.py` derives the name from the same call, so the two cannot
# land in different directories.
device_dir=$(python -W "ignore:Could not find flash_attn:UserWarning" -c \
    'import torch; print((torch.cuda.get_device_name(0) or "unknown_device").strip().replace(" ", "_"))' \
    ) || { echo "run_bench.sh: could not read the device name from torch" >&2; exit 1; }

md5=$(md5sum "$so" | cut -d' ' -f1)
devices="${CUDA_VISIBLE_DEVICES:-<unset: whatever the host set>}"
chip_dir="${results_dir}/${device_dir}"
mkdir -p "$chip_dir"

# The latest dated run that `run_bench.sh` wrote: a `manifest.json` alone is not
# enough -- a directory written by `perf_snapshot.py` directly has no
# `run_header.txt` and is invisible here on purpose (it is not a gate record).
latest_result_dir() {
    local candidate name latest=""
    for candidate in "${chip_dir}"/????????_??????*; do
        [[ -d "${candidate}" && -f "${candidate}/manifest.json" && -f "${candidate}/run_header.txt" ]] || continue
        name=${candidate##*/}
        if [[ -z "${latest}" || "${name}" > "${latest}" ]]; then latest=${name}; fi
    done
    printf '%s\n' "${latest}"
}

# ── --baseline-dir: repoint and exit, measuring nothing ─────────────────────
# For re-reading an old verdict without paying for a re-measure.
if [[ -n "$baseline_dir" ]]; then
    target="${baseline_dir##*/}"
    if [[ ! -d "${chip_dir}/${target}" ]]; then
        echo "run_bench.sh: no such baseline directory: ${chip_dir}/${baseline_dir}" >&2
        exit 2
    fi
    prev=$(readlink "${chip_dir}/baseline" 2>/dev/null || echo "(none)")
    ln -sfn "${target}" "${chip_dir}/baseline"
    echo "run_bench.sh: baseline ${prev} -> ${target}"
    exit 0
fi

# ── --compare-only: latest vs baseline, measuring nothing ───────────────────
if [[ ${compare_only} -eq 1 ]]; then
    latest=$(latest_result_dir)
    if [[ -z "${latest}" ]]; then
        echo "run_bench.sh: no run_bench.sh record under ${chip_dir}" >&2; exit 2
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
        python3 tools/compare_snapshots.py "${chip_dir}/${latest}" --base "${chip_dir}/baseline"
    } 2>&1 | tee "${chip_dir}/${latest}/compare_result.txt"
    exit ${PIPESTATUS[0]}
fi

# ── measure ─────────────────────────────────────────────────────────────────
stamp=$(date +%Y%m%d_%H%M%S)
out="${chip_dir}/${stamp}"
mkdir -p "$out"

# The receipts, taken before anything runs: a run whose header names another
# torch process is not a measurement, whatever the util column said.
run_header() {
    echo "# run_bench.sh    device_dir=${device_dir}"
    echo "# timestamp_utc   $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "# host            $(hostname)"
    echo "# extension       ${so}"
    echo "# extension_md5   ${md5}"
    echo "# cuda_visible_devices ${devices}"
    echo "# backends        ${backends}"
    echo "${stale_note}"
    echo "# ---"
    echo "# mx-smi"
    mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
    echo "# pgrep -af python"
    pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
        | sed 's/^/#   /' || echo "#   (none)"
    echo "$@"
}
run_header > "${out}/run_header.txt"

run_stage() {
    local name="$1"; shift
    local log="${out}/${name}.log" rc=0
    echo "run_bench.sh: [${name}] -> ${log}"
    PYTHONPATH="${stage_pythonpath}" timeout "${timeout_s}" "$@" 2>&1 | tee "$log" || rc=$?
    if [[ ${rc} -eq 124 ]]; then
        echo "run_bench.sh: [${name}] TIMED OUT after ${timeout_s}s" >&2
    elif [[ ${rc} -ne 0 ]]; then
        echo "run_bench.sh: [${name}] FAILED (rc=${rc})" >&2
    fi
    echo "# exit ${rc}" >> "$log"
    return "$rc"
}

bench_status=0
# `python tests/test.py`, not `./tests/test.py`: upstream's file is not
# executable, and a bare path fails with rc=126 before it runs anything (which
# reads as a bench failure but is a launcher mistake).
#
# The snapshot's failure does not skip the gate: it is reported as its own thing
# (it can fail on memory) while the gate may still pass.  Any failure makes the
# run ineligible for `--set-baseline`, which is what the sticky status is for.
run_stage snapshot      python scripts/perf_snapshot.py --backends "${backends}" --out-dir "${out}" || bench_status=1
if [[ ${skip_gate} -eq 0 ]]; then
    # ── the correctness gate, which the two stages below are not ────────────
    # `--perf-only` filters `testcases` to `num_runs > 0`, and every one of
    # those is a `NormalFloatDistribution` cell -- `randn_like`, no NaN.  The
    # correctness table (which *does* carry the NaN cases, via
    # `UintDistributionWithHotspotAndSpecifiedPivot(..., True)`) has
    # `num_runs=0` and is never reached.  So the two stages below are a *perf*
    # gate wearing a test's name, and the two defects they let through
    # (the chunks arm serving a NaN row a normal top-k, and a captured graph
    # reading a scratch buffer a later call had moved) were both invisible to
    # them for exactly that reason.
    #
    # A sample rather than the whole 105,140-case table: it is the same
    # predicate on the same generator, and a fixed seed makes the draw a
    # receipt someone else can re-run.  `-rf` so one bad case does not hide
    # the rest.
    #
    # **The harness's exit status is all-or-nothing and this box cannot satisfy
    # it.**  `tests/test.py` returns 1 for *any* non-pass bucket, and the
    # 400-case draw always contains a few `b=4096 V~1e6` fp32 cells -- 15 GiB
    # per tensor, three alive at once -- which a 63.6 GiB shared device cannot
    # serve.  They land in `crash` rather than `skip` because the allocation
    # that fails is the *operator's* `cudaMalloc` (`maca_topk.cu`), so
    # `torch.cuda.OutOfMemoryError` is never raised for the harness to classify
    # it.  So `run_stage correctness ... || true`, and the verdict is read off
    # the harness's own summary block: **no `check_fail`** (a wrong selection --
    # the only thing this stage exists to catch) and **no crash whose detail
    # line is not an out-of-memory**.  The `^={20,}$` rule is the summary's own
    # separator; the per-case one is 16 characters wide.
    run_stage correctness python tests/test.py --seed 20260911 --sample 400 -rf || true
    csum=$(awk '/^={20,}$/{buf=""} {buf = buf $0 ORS} END{printf "%s", buf}' \
           "${out}/correctness.log") || true
    cf=$(printf '%s' "${csum}" | awk '/^  check_fail /{print $2; exit}') || true
    cr=$(printf '%s' "${csum}" | awk '/^  crash /{print $2; exit}') || true
    sk=$(printf '%s' "${csum}" | awk '/^  skip /{print $2; exit}') || true
    # `|| true` on every one of these: `set -o pipefail` is on, and a `grep`
    # that filters *everything* out exits 1 -- which is the healthy case here
    # ("no crash detail that is not an OOM"), and would otherwise take the
    # whole script down with `set -e` at exactly the moment it should pass.
    non_oom=$(printf '%s' "${csum}" | grep -A1 "^  crash  *TestParam" \
              | grep -vE "^  crash  *TestParam|^--" | grep -v "out of memory" \
              | head -3) || true
    if [[ "${cf:-?}" != "0" ]]; then
        echo "run_bench.sh: [correctness] FAILED -- ${cf:-?} case(s) selected wrong" >&2
        printf '%s' "${csum}" | grep -A1 "^  check_fail  *TestParam" | head -6 \
            | sed 's/^/              /' >&2 || true
        bench_status=1
    elif [[ -n "${non_oom}" ]]; then
        echo "run_bench.sh: [correctness] FAILED -- a crash that is not an OOM:" >&2
        echo "${non_oom}" | sed 's/^/              /' >&2
        bench_status=1
    else
        echo "run_bench.sh: [correctness] OK -- ${cr:-?} crashed and ${sk:-?} skipped, all out of memory; 0 selected wrong"
    fi
    # The arms' own cases, which pin the shapes the drawn table only samples:
    # `chunks_arm_official.py` covers the `b <= 2` band and the NaN contract on
    # it, `graph_capture.py` covers capture/replay and the shape-change probes.
    # Both print the extension they loaded; `PYTHONPATH` here is the same one
    # the other stages get, so it is the checkout's artifact.
    run_stage cases_chunks  python tests/cases/chunks_arm_official.py || bench_status=1
    run_stage cases_graph   python tests/cases/graph_capture.py || bench_status=1
    run_stage official      python tests/test.py --perf-only -nc || bench_status=1
    run_stage official_fp32 python tests/test.py --perf-only -nc --dtype fp32 || bench_status=1
fi

{
    echo "# ---"
    echo "# run_bench.sh: finished rc=${bench_status}"
    echo "# out             ${out}"
} >> "${out}/run_header.txt"

# ── compare against the baseline, before --set-baseline moves it ────────────
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
        compare_rc=0
        {
            echo "# compare: ${stamp} vs baseline ${base_name}"
            python3 tools/compare_snapshots.py "${out}" --base "${chip_dir}/baseline"
        } 2>&1 | tee "${out}/compare_result.txt" || compare_rc=$?
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
        echo "run_bench.sh: baseline NOT updated -- a stage failed (rc=${bench_status})" >&2
    fi
fi

echo ""
echo "run_bench.sh: done. results: ${out}"
echo "run_bench.sh: baseline: ${chip_dir}/baseline"
exit "${bench_status}"
