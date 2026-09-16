#!/usr/bin/env bash
#
# Run the whole performance grid, write it out as CSVs, and compare against the
# baseline.  The "how is this tree doing" entry point; `run_test.sh` is the
# correctness/gating one.
#
# ── The three stages ──────────────────────────────────────────────────────────
#
#   1. `perf_snapshot.py` -- writes the CSV: the OFFICIAL grid (`tests/test.py`'s
#      own `performance_cases()`, called not restated), plus the selector
#      grid, for the backends named in `--backends`.
#   2. `tests/test.py --perf-only` -- the same grid driven by upstream's own file
#      in one process.  Redundant with (1) on purpose: it is the gate.
#   3. `tests/test.py --perf-only --dtype fp32` -- the fp32 Sampler cells.
#
# Nothing here re-implements a measurement -- every stage is upstream's file, and
# `tests/test.py` can neither select a backend nor emit a CSV, which is why the
# snapshot is its own stage rather than a mode of it.
#
# ── Failure, and the baseline ───────────────────────────────────────────────
#
# The FIRST failing stage stops the run: a comparison against a baseline whose
# grid was not fully measured is not a comparison.  The verdict is written to
# `compare_result.txt` before the baseline moves.
#
# `--set-baseline` repoints when every stage passed, not on the comparator's exit
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
#     ./run_bench.sh --backends maca_c,torch  # a subset of the three
#     ./run_bench.sh --baseline-dir 20260801  # repoint, do not measure
#
# Env:
#     CUDA_VISIBLE_DEVICES  device selection, applied to every stage.
#                           Default: unchanged.
#     DS_BENCH_DIR          output root (default perf_data)
#     DS_BENCH_TIMEOUT      per-stage timeout in seconds (default 5400)
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

# Whether the `deep_gemm` column can be measured at all is the *package's*
# answer -- does it import, and does it carry the entry `backend="deep_gemm"`
# calls -- and never a question about where its source lives.  Asked here for
# the read-out below only: every runner asks it for itself, which is why there
# is no switch to forward (an override would have to reach all three, and one
# that does not is an override a caller believes they made).
deep_gemm_shapes=$(python -W "ignore:Could not find flash_attn:UserWarning" - <<'PY'
import os, sys
sys.path.insert(0, os.getcwd())
from deep_select import deep_gemm_available
print(1 if deep_gemm_available() else 0)
PY
) || deep_gemm_shapes=0
backends="maca_c,torch,deep_gemm"
set_baseline=0
baseline_dir=""
compare_only=0
skip_gate=0
results_dir="${DS_BENCH_DIR:-perf_data}"
timeout_s="${DS_BENCH_TIMEOUT:-5400}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --backends)         [[ $# -ge 2 ]] || { echo "run_bench.sh: --backends needs a value" >&2; exit 2; }
                            backends="$2"; shift 2 ;;
        --backends=*)       backends="${1#*=}"; shift ;;
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

# The extension under test, asked of the package that will answer.  This is
# `_binding.extension_path`, the same call `_binding.load` makes, so the md5
# below is the artifact the process actually loads -- for the checkout's
# package or for an installed wheel alike (a deployed wheel carries its `.so`
# beside `_binding.py`).
#
# The checkout goes on the path **only when it has a built extension**.  `python -`
# starts with the cwd on `sys.path`, so that has to be taken off explicitly --
# with the tree reachable the import lands on its `deep_select/`, which imports
# fine and then fails at call time (the extension loads lazily), and an
# installed wheel could never answer.
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
    echo "run_bench.sh: no extension found; looked in the checkout first, then in" >&2
    echo "              the installed deep_select.  Build one (./develop.sh) or" >&2
    echo "              install one (./install.sh)." >&2
    exit 1
fi

# The same axis, for the stages.  `$REPO` goes on the path only when it is the
# tree answering -- the two must agree, or the run measures one artifact and
# receipts another.  `tests` is always there: the grid lives in the repo and is
# not in the wheel.
if [[ "$so" == "$PWD"/deep_select/* ]]; then
    stage_pythonpath=".:tests"
else
    stage_pythonpath="tests"
    echo "run_bench.sh: measuring the installed package's extension:" >&2
    echo "              $so" >&2
fi

# A header newer than the .so means the extension may not match the sources.
# Reported, not enforced -- the md5 is the record either way, as in run_test.sh.
# Only a checkout's extension can answer this: `csrc/` describes what a local
# build would produce, not what an installed wheel was built from.
stale_note="# STALE           no"
if [[ "$so" == "$PWD"/deep_select/* ]] && [[ -n "$(find csrc -newer "$so" \( -name '*.cu' -o -name '*.cuh' \) 2>/dev/null | head -1)" ]]; then
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
            --base "${chip_dir}/baseline"
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
    echo "# backends        ${backends}"
    echo "${stale_note}"
    echo "# ---"
    echo "# mx-smi"
    mx-smi 2>/dev/null || echo "#   (mx-smi unavailable)"
    echo "# pgrep -af python"
    pgrep -af python 2>/dev/null | grep -vE "socks5|networkd|unattended-upgrade" \
        | sed 's/^/#   /' || echo "#   (none)"
} > "${out}/run_header.txt"

run_stage() {
    local name="$1"; shift
    local log="${out}/${name}.log"
    echo "run_bench.sh: [${name}] -> ${log}"
    set +e
    PYTHONPATH="${stage_pythonpath}" timeout "${timeout_s}" "$@" 2>&1 | tee "$log"
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

# stage 1 -- the snapshot.  It writes the CSVs, so its failure is reported as its
# own thing rather than "a bench failure" (it can fail on memory; the gate may
# still pass).
snap_args=(scripts/perf_snapshot.py --backends "${backends}"
           --out-dir "${out}")
if ! run_stage snapshot python "${snap_args[@]}"; then
    snapshot_failed=1
    bench_status=1
fi

# stage 2 -- the official grid, upstream's own file.  The gate.
# `python tests/test.py`, not `./tests/test.py`: upstream's file is not
# executable, and a bare path there fails with rc=126 before it runs anything
# (which reads as a bench failure but is a launcher mistake).
# No env prefix, and that is a change: these stages used to be launched as
# `env DS_TOPK_BACKEND=maca_c` so that the *library* default would not serve the
# gate with the `torch` reference.  `tests/test.py` now passes `backend=maca_c`
# itself (`run_testcase`'s default), which means it never consults the process
# default -- measured: with `DS_TOPK_BACKEND=torch` in the environment the call
# still arrives as `backend=maca_c`.  So the env var was left changing nothing,
# which is the knob this repo deletes rather than keeps.
if [[ ${skip_gate} -eq 0 ]]; then
    run_stage official python tests/test.py --perf-only -nc || bench_status=1
fi

# stage 3 -- the official grid's fp32 run: `tests/test.py` filters its own
# `performance_cases` (bf16), so this is the Sampler cells plus the selector
# shapes.
if [[ ${skip_gate} -eq 0 ]]; then
    run_stage official_fp32 python tests/test.py --perf-only -nc --dtype fp32 || bench_status=1
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
                --base "${chip_dir}/baseline"
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
        echo "run_bench.sh: baseline NOT updated -- a stage failed (rc=${bench_status})" >&2
    fi
fi

echo ""
echo "run_bench.sh: done. results: ${out}"
echo "run_bench.sh: baseline: ${chip_dir}/baseline"
cd "$original_dir"
exit "${bench_status}"
