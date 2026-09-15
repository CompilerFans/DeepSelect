#!/usr/bin/env python3
"""Run the official performance grid for every backend and write it as a CSV.
A *thin* recorder: the cases, the data, the checks and the timings are all
`tests/test.py`'s own, called rather than re-implemented.  The one axis added is
the one the official harness cannot express: **which backend answered**, one row
per (cell, backend), each with a `status` of `pass` / `fail` / `unsupported`.
A cell a backend cannot serve states its reason there, and is never dropped.
Backends: `maca_c` -- this repository's kernel, the production backend (the DEFAULT
is `torch`, so it is asked for by name); `torch` -- the official reference, a
bare `torch.topk`, NOT `backend="torch"`, which pads, masks and converts around
the same call and measures something else entirely; `deep_gemm` -- that
package's `fp32_indexer_topk_selector`, float32 only, hence `unsupported` on the
whole bf16 official grid.  The fp32 grid it answers rides along whenever
`deep_select.deep_gemm_available()` says that backend can run at all.
Cases: the official grid (`tests/test.py::performance_cases()`) plus
`--cases-file`, a JSON list of `lib.TestParam` fields, e.g.
`{"batch_size": 6, "vocab_size": 32768, "topk": 1024}`.  Those three are the
required keys; the rest default to the Lightning Indexer's configuration, and
`dtype` / `out_idx_dtype` take bf16 / fp32 / int32 / int64 -- an unknown name is
an error, not a default.

Output, following mcDeepGEMM's `deep_gemm/tests/perf_data/` layout:
`perf_data/<device>/<YYYYmmdd_HHMMSS>/` (`deepselect_perf.csv`, `manifest.json`).
`<device>` is the **device name torch reports** (`MetaX C500` -> `MetaX_C500`),
never the arch family: the folder answers "which board did I measure on", and two
boards of one family (both xcore1600) share an ISA but not a clock, a wall or an
SM count.  The family is recorded per row (`chip`), never inferred from it.
Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \\
        python3 scripts/perf_snapshot.py
    ... --backends a,b            # subset of the backends (default: all three)
    ... --cases-file extra.json   # + your own cases
    ... --out-dir DIR             # the record's own directory
    ./run_bench.sh                # the orchestrator: this + the official gate
"""
from __future__ import annotations
import argparse
import csv
import datetime as _dt
import glob
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "tests"))
sys.path.insert(0, REPO)
import torch  # noqa: E402
import kernelkit as kk  # noqa: E402
import lib  # noqa: E402
import test as official  # noqa: E402
from deep_select import _arch  # noqa: E402
PYBIN = sys.executable
BACKENDS = ("maca_c", "torch", "deep_gemm")
# Short spellings for `--cases-file`.  Only the pair this operator serves.
DTYPES = {"bf16": "bfloat16", "bfloat16": "bfloat16",
          "fp32": "float32", "float32": "float32"}
IDX_DTYPES = {"int32": "int32", "int64": "int64"}
COUNTER = kk.Counter()
class Unsupported(Exception):
    """The backend refused the case; a status, never a failure."""
def call_topk(p, t, backend: str):
    """The operator call `tests/test.py::run_testcase` makes, with a backend."""
    from deep_select import topk
    try:
        return topk(
            t.input,
            p.topk,
            sorted=p.sorted_value,
            begin=None,
            end=t.end,
            indices_type=p.out_idx_dtype,
            sorted_index=p.sorted_index,
            hint=None,
            output_idx=None,
            output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value,
            abort_when_nan_found=False,
            backend=backend,
        )
    except Exception as exc:
        if type(exc).__name__ == "UnsupportedByBackend":
            raise Unsupported(str(exc).strip()) from None
        raise
def time_operator(p, t, backend: str) -> Optional[float]:
    """`kk.bench` over the operator, with `tests/test.py`'s matching rule."""
    usage, _ = official.bench_topk(
        lambda: call_topk(p, t, backend), p, t, None, None)
    return usage
def measure(p, t, backend: str) -> Dict[str, Any]:
    """One (cell, backend) result: status first, then the numbers."""
    row: Dict[str, Any] = {"status": "pass", "error_type": "", "error_message": ""}
    if backend == "torch":
        # Nothing checks the reference, here as in `tests/test.py`: a speed
        # baseline, not a correctness check.
        if t.end is not None or t.output_idx_offset is not None or p.vocab_size < p.topk:
            # `run_testcase`'s own eligibility guard, carried as the reason so an
            # absent number is not read as a missing measurement.  A bare
            # `torch.topk` with `k > x.shape[1]` raises, hence the guard.
            row["error_message"] = ("not applicable: the reference backend needs "
                                    "vocab_size >= topk, no window, no offset "
                                    "(tests/test.py's own guard)")
            return row
        try:
            row["time(us)"] = _us(official.bench_torch_reference(p, t))
        except Exception as exc:
            row["status"], row["error_type"] = "fail", type(exc).__name__
            row["error_message"] = str(exc)[:200]
        return row
    try:
        value, index = call_topk(p, t, backend)
        torch.cuda.synchronize()
        official.check_call_contract(p, value, index)
    except Unsupported as exc:
        row["status"], row["error_type"] = "unsupported", "UnsupportedByBackend"
        row["error_message"] = str(exc)[:200]
        return row
    except Exception as exc:
        row["status"], row["error_type"] = "fail", type(exc).__name__
        row["error_message"] = str(exc)[:200]
        return row
    try:
        if not official.check_result(p, t, value, index):
            row["status"], row["error_type"] = "fail", "CheckFailed"
            row["error_message"] = "check_result() returned false"
    except Exception as exc:
        row["status"], row["error_type"] = "fail", type(exc).__name__
        row["error_message"] = str(exc)[:200]
    row["Byte(MB)"] = round(official.topk_total_size(p, t, value, index) / 1e6, 3)
    del value, index
    # Still timed when it can be: "selects wrong" and "slow" are different
    # defects; dropping the time answers neither.
    try:
        row["time(us)"] = _us(time_operator(p, t, backend))
    except Exception as exc:
        row["error_message"] = (row["error_message"] + "; " if row["error_message"]
                                else "") + f"{type(exc).__name__}: {str(exc)[:120]}"
    return row
def _us(seconds: Optional[float]) -> Any:
    """Seconds -> microseconds, or "" when the backend did not apply / was not timed.
    Empty rather than 0: a 0 us cell reads as an implausible win.
    """
    return round(seconds * 1e6, 3) if seconds else ""
# ── cases ──
def deep_gemm_cases() -> List[Any]:
    """The `deep_gemm` selector perf shapes, from `tests/test.py`.

    The table lives there (`DEEP_GEMM_SELECTOR_PERF_SHAPES`), not here: the official
    grid drives the *same* shapes, and a second copy drifts from the gate meant
    to check it.  `note` marks the rows whose `seq_len`
    is narrower than `n_cols` -- windows there, whole-row rankings here, so the
    two are not comparable at the same shape.
    """
    return [(_deep_gemm_note(b, v, seq),
             lib.TestParam(b, v, official.DEEP_GEMM_SELECTOR_PERF_TOPK, False, False,
                           False, torch.float32, torch.int32, num_runs=10))
            for b, v, seq in official.DEEP_GEMM_SELECTOR_PERF_SHAPES]


def _deep_gemm_note(b: int, v: int, seq: int) -> str:
    if seq == v:
        return ""
    return (f"sglang-bs{b}-seq{seq}: that grid declares a window; this "
            f"ranks the whole row")


# `lib.TestParam`'s four non-shape fields have no defaults, so spell them out.
PARAM_DEFAULTS = {"sorted_value": False, "sorted_index": False,
                  "return_value": False, "dtype": "bf16", "out_idx_dtype": "int32"}


def cases_from_file(path: str) -> List[Any]:
    """`--cases-file`: a JSON list of `lib.TestParam` fields.

    `batch_size` / `vocab_size` / `topk` are required, everything else comes from
    `PARAM_DEFAULTS`.  An unknown `dtype` / `out_idx_dtype` is an error: a typo
    must not become a different case.
    """
    with open(path) as f:
        specs = json.load(f)
    if not isinstance(specs, list):
        raise ValueError(f"{path}: expected a JSON list of case objects")
    out = []
    for i, spec in enumerate(specs):
        missing = {"batch_size", "vocab_size", "topk"} - set(spec)
        if missing:
            raise ValueError(f"{path}[{i}]: missing {sorted(missing)}")
        kw = dict(PARAM_DEFAULTS)
        kw.update(spec)
        for field, table in (("dtype", DTYPES), ("out_idx_dtype", IDX_DTYPES)):
            if field in kw:
                name = kw[field]
                if name not in table:
                    raise ValueError(
                        f"{path}[{i}]: {field} {name!r} is not one of "
                        f"{', '.join(table)}")
                kw[field] = getattr(torch, table[name])
        note = kw.pop("note", "")
        out.append((note, lib.TestParam(**kw)))
    return out


# ── provenance ──

def _run(cmd: List[str], cwd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""
def device_dir_name() -> str:
    """`perf_data/`'s directory, named after the **device** (see the module
    docstring): `MetaX C500` -> `MetaX_C500`.

    Falls back to the arch spelling only when torch reports no device name at
    all (a driver quirk, not a normal case).
    """
    name = (torch.cuda.get_device_name(0) or "").strip()
    if not name:
        return f"metax_{_arch.native_target()}"
    return name.replace(" ", "_")


def _deep_gemm_package() -> Dict[str, str]:
    """What the `deep_gemm` backend is, recorded from the *package* it imports.

    A package question, not a location one: `backend="deep_gemm"` is
    `import deep_gemm`, so what decides the numbers is the module that import
    lands on and its version -- not which checkout holds its source, which
    this repository has no business knowing.  Empty strings mean the package
    is not importable, which is a true answer and the one to record.
    """
    try:
        import deep_gemm
        return {
            "deep_gemm_package": os.path.dirname(os.path.abspath(deep_gemm.__file__)),
            "deep_gemm_version": str(getattr(deep_gemm, "__version__", "")),
        }
    except Exception:
        return {"deep_gemm_package": "", "deep_gemm_version": ""}


def provenance(sm_count: int) -> Dict[str, Any]:
    here = REPO
    # The extension the *device* loads -- `_binding.load()`, the same artifact
    # `run_bench.sh` resolves its md5 through.  There is one name now (the
    # build produces a single fat extension), but this stays in terms of the
    # loader's own constant so the two cannot drift into naming different
    # files: a record whose md5 is not the loaded artifact's is not a record.
    sos = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(here, "deep_select", "deep_select_maca*.so")))
    md5 = ""
    if sos:
        md5 = hashlib.md5(open(os.path.join(here, "deep_select", sos[0]),
                               "rb").read()).hexdigest()
    return {
        **_deep_gemm_package(),
        # Arch family per row (`metax_xcore<N>`), from the device, not the
        # directory name: the column to filter on for same-ISA rows.
        "chip": f"metax_{target}",
        "device_dir": device_dir_name(),
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": sm_count,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "deep_select_git_commit": _run(["git", "log", "-1", "--format=%H"], here),
        "deep_select_git_dirty": bool(_run(["git", "status", "--porcelain"], here)),
        "extension_so": sos[0] if sos else "",
        "extension_md5": md5,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
# ── writing ──
# `relative_pct_vs_maca_c` is maca_c = 100%, so >100% means that backend is
# faster than this repository's own kernel; repeated per row so a row stands alone.
COLUMNS = ["chip", "device_name", "sm_count", "git_commit", "extension_md5",
           "case_source", "family", "n_rows", "n_cols", "top_k",
           "sorted_value", "return_value", "input_dtype", "index_dtype",
           "num_runs", "backend", "status", "error_type", "error_message",
           "time(us)", "throughput(TB/s)", "bandwidth(GB/s)", "Byte(MB)",
           "relative_pct_vs_maca_c", "note"]


def rows_for(p, source: str, note: str, got: Dict[str, Dict[str, Any]],
             prov: Dict[str, Any]) -> List[Dict[str, Any]]:
    # The operator's own traffic (input read + outputs written), the figure
    # `tests/test.py:138` prints.  From the shape, not one backend's buffers, so
    # every row of a cell shares it and bandwidths compare across backends.
    nbytes = (p.batch_size * p.vocab_size * p.dtype.itemsize
              + p.batch_size * p.topk
              * (p.dtype.itemsize * int(p.return_value)
                 + p.out_idx_dtype.itemsize))
    ref = got.get("maca_c", {}).get("time(us)")
    out = []
    for backend in BACKENDS:
        g = got.get(backend)
        if g is None:
            continue
        row = {c: "" for c in COLUMNS}
        row.update({
            "chip": prov["chip"], "device_name": prov["device_name"],
            "sm_count": prov["sm_count"],
            "git_commit": prov["deep_select_git_commit"],
            "extension_md5": prov["extension_md5"],
            "case_source": source,
            "family": "sampler" if p.sorted_value else "lightning_indexer",
            "n_rows": p.batch_size, "n_cols": p.vocab_size, "top_k": p.topk,
            "sorted_value": int(p.sorted_value),
            "return_value": int(p.return_value),
            "input_dtype": str(p.dtype).replace("torch.", ""),
            "index_dtype": str(p.out_idx_dtype).replace("torch.", ""),
            "num_runs": p.num_runs,
            "backend": backend,
            "status": g["status"],
            "error_type": g.get("error_type", ""),
            "error_message": g.get("error_message", ""),
            "note": note,
        })
        for k in ("time(us)", "Byte(MB)"):
            if g.get(k):
                row[k] = g[k]
        us = g.get("time(us)")
        if us:
            row["throughput(TB/s)"] = round(nbytes / (us * 1e-6) / 1e12, 6)
            row["bandwidth(GB/s)"] = round(nbytes / (us * 1e-6) / 1e9, 3)
            row["relative_pct_vs_maca_c"] = round(ref / us * 100, 2) if ref else ""
        out.append(row)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="extra cases (--cases-file) are a JSON list of lib.TestParam "
               "fields; only batch_size, vocab_size and topk are required, and "
               "dtype/out_idx_dtype take " + "/".join(sorted(set(DTYPES))) + ".")
    ap.add_argument("--backends", default=",".join(BACKENDS),
                    help="comma-separated subset of the backends")
    ap.add_argument("--cases-file", default="",
                    help="JSON list of extra cases (see the epilog)")
    ap.add_argument("--out-dir", default="",
                    help="the directory to write this snapshot into.  "
                         "Default: <repo>/perf_data/<device>/<YYYYmmdd_HHMMSS> "
                         "(see the module docstring)")
    args = ap.parse_args()
    backends = [a for a in args.backends.split(",") if a]
    for a in backends:
        if a not in BACKENDS:
            raise SystemExit(f"unknown backend {a!r}; expected one of {', '.join(BACKENDS)}")
    cases = [("official", "", p) for p in official.performance_cases()]
    # Whether the `deep_gemm` column can be measured is the package's answer,
    # not a switch: no flag here, and no `--no-` on the harness above it.  A
    # switch would have to be forwarded to every runner that probes for itself,
    # and an override that is not forwarded is an override a caller believes
    # they made (measured: `run_bench.sh --no-deep-gemm-shapes` reported 95
    # cells and ran 120).
    from deep_select import deep_gemm_available
    if deep_gemm_available():
        cases += [("deep_gemm_axes", note, p) for note, p in deep_gemm_cases()]
    if args.cases_file:
        cases += [("extra", note, p) for note, p in cases_from_file(args.cases_file)]
    torch.set_default_device("cuda")
    import deep_select  # noqa: E402  (after set_default_device)
    target = _arch.native_target()
    sm_count = _arch.native_sm_count()
    prov = provenance(sm_count)
    # `--out-dir` is the directory itself, not a root to hang a name under: a
    # caller that names one has already decided where this record goes.
    out = args.out_dir or os.path.join(
        REPO, "perf_data", prov["device_dir"],
        _dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
    run_id = os.path.basename(out.rstrip(os.sep))
    os.makedirs(out, exist_ok=True)
    started = _dt.datetime.now().astimezone()
    rows: List[Dict[str, Any]] = []
    print(f"chip {prov['chip']}  device {prov['device_name']}  sm {sm_count}  "
          f"backends {backends}  cases {len(cases)}", flush=True)
    print(f"{'case':<46}{'backend':<10}{'status':<12}{'us':>12}{'GB/s':>10}",
          flush=True)
    for i, (source, note, p) in enumerate(cases):
        if p.seed == -1:
            p.seed = COUNTER.next()
        t0 = time.time()
        try:
            t = lib.generate_testcase(p)
        except Exception as exc:                # OOM guard, mirrors test.py:194
            print(f"  generate_testcase failed for {p}: {exc}", flush=True)
            break
        got = {}
        for backend in backends:
            got[backend] = measure(p, t, backend)
            g = got[backend]
            label = (f"{source[:5]}/{str(p.dtype).replace('torch.','')[:4]} "
                     f"b{p.batch_size}-v{p.vocab_size}-k{p.topk}")
            print(f"{label:<46}{backend:<10}{g['status']:<12}"
                  f"{g.get('time(us)', '')!s:>12}"
                  f"{(g.get('bandwidth(GB/s)') or '')!s:>10}"
                  + (f"  [{g['error_message'][:34]}]" if g.get("error_message") else ""),
                  flush=True)
        rows.extend(rows_for(p, source, note, got, prov))
        del t, got
        torch.cuda.empty_cache()
        if (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{len(cases)} cases ({time.time() - t0:.1f}s)",
                  flush=True)
    csv_path = os.path.join(out, "deepselect_perf.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)
    counts: Dict[Any, int] = {}
    for r in rows:
        counts[(r["backend"], r["status"])] = counts.get((r["backend"], r["status"]), 0) + 1
    manifest = dict(prov)
    manifest.update({
        "run_id": run_id,
        "output_dir": out,
        "command": " ".join([PYBIN] + sys.argv),
        "started_at_utc": started.astimezone(_dt.timezone.utc).isoformat(),
        "finished_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "csv_files": [os.path.basename(csv_path)],
        "backends": backends,
        "cases": len(cases),
        "rows": len(rows),
        "status_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "csv_format_version": 1,
        # The selector cells are not an axis any more: whether they are added is
        # `deep_select.deep_gemm_available()`'s answer, and `cases` below records
        # how many there turned out to be.  Naming a flag here would be naming
        # one that no longer exists.
        "case_source": "tests/test.py::performance_cases() (+ the selector grid "
                       "when the deep_gemm package can serve it, + --cases-file)",
        "measurement": ("tests/test.py's own: one 'topk'-matching kernel's time, "
                        "else the e2e span over the matching kernels; p.num_runs "
                        "reps, L2 flushed (kk.bench). Correctness: "
                        "tests/test.py::check_result / check_call_contract, "
                        "applied to every backend except `torch`, whose column is a "
                        "bare torchtopk the official harness does not check "
                        "either."),
    })
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {out}/deepselect_perf.csv  ({len(rows)} rows, "
          f"{len(cases)} cases)", flush=True)
    for k, v in sorted(counts.items()):
        print(f"  {k[0]:<10} {k[1]:<12} {v}", flush=True)
    return 0
if __name__ == "__main__":
    raise SystemExit(main())
