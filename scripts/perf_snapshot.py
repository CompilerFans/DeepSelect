#!/usr/bin/env python3
"""Run the official performance grid for every backend and write it as a CSV.
This is a *thin* recorder: the cases, the data, the checks and the timings are
all `tests/test.py`'s own, called rather than re-implemented.  Per cell the body
is `run_testcase`'s, line for line:
    t = lib.generate_testcase(p)
    value, index = deep_select.topk(...)         # backend=<arm>
    check_result(p, t, value, index)
    kk.bench(...) -> one matching kernel's time, else the span
and the reference arm is literally `torch.topk(t.input, p.topk, dim=1,
sorted=p.sorted_value)` under the same eligibility guard the official runner
uses, timed with the same rule.  The only thing added is the axis the official
harness cannot express: **which backend answered**, one row per (cell, backend),
with a `status` of `pass` / `fail` / `unsupported`.
Three backends:
  maca_c     this repository's kernel (the DEFAULT is `torch` now; this arm
             names what it wants explicitly, as the recorder always has)
  torch      the official reference -- a bare `torch.topk`, as `tests/test.py`
             times it (NOT `backend="torch"`, which pads, masks and converts
             around the same call and measures something else entirely)
  deep_gemm  the host repository's `fp32_indexer_topk_selector`, through
             `backend="deep_gemm"`.  It ranks float32 only, so it is
             `unsupported` on the whole bf16 grid -- which is why
             `--deep-gemm-axes` adds the host repo's own fp32 selector grid:
             without it the `deep_gemm` column has no number in it at all.
Cases: the official grid by default (`tests/test.py::performance_cases()`), plus
extra ones from `--cases-file`, a JSON list of `lib.TestParam` fields:
    [{"batch_size": 6, "vocab_size": 32768, "topk": 1024},
     {"batch_size": 256, "vocab_size": 131072, "topk": 2048,
      "dtype": "fp32", "num_runs": 20}]
Only `batch_size`, `vocab_size` and `topk` are required; the rest default to the
Lightning Indexer's configuration (sorted and return_value off, bf16, int32).
`dtype` / `out_idx_dtype` take the short names bf16, fp32, int32, int64; an
unknown one is an error rather than a default.

Output, following the host repository's `deep_gemm/tests/perf_data/` layout:
    perf_data/<device>/<YYYYmmdd_HHMMSS>/deepselect_perf.csv
    perf_data/<device>/<YYYYmmdd_HHMMSS>/manifest.json

`<device>` is the **device name torch reports** (`MetaX C500` -> `MetaX_C500`),
not the arch family: the folder answers "which board did I measure on", and the
part identity is what makes two records comparable.  Two boards of one family
(`MetaX C600` and `MetaX C600-U`, both xcore1600) share an ISA but not a clock,
a wall or an SM count, so a family-named folder would silently stack them.  The
arch family is still recorded per row (`chip`) and never inferred back out of
the folder name.  A stale empty directory left by the arch spelling
(`perf_data/metax_xcore1000/`) is skipped by the baseline scan below, so an
untouched one is inert rather than misleading.
Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \\
        python3 scripts/perf_snapshot.py
    ... --deep-gemm-axes          # + the host repo's fp32 selector grid
    ... --cases-file extra.json   # + your own cases
    ... --dry-run                 # print what would run, measure nothing
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
ARMS = ("maca_c", "torch", "deep_gemm")
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
        # The reference is checked by nothing, here as in `tests/test.py`: it is
        # a speed baseline, and the contract's own arm is `maca_c`.
        if t.end is not None or t.output_idx_offset is not None or p.vocab_size < p.topk:
            # `run_testcase`'s own eligibility guard, carried as the reason so
            # the gap is stated rather than looking like a missing measurement.
            # A bare `torch.topk(x, k)` with `k > x.shape[1]` raises (measured:
            # "selected index k out of range"), which is why the guard exists --
            # the windowed operator answers such a row with its whole prefix.
            row["error_message"] = ("not applicable: the reference arm needs "
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
    # A failing case is still timed when it can be: which of "selects wrong" and
    # "slow" it is matters, and hiding the measurement answers neither.
    try:
        row["time(us)"] = _us(time_operator(p, t, backend))
    except Exception as exc:
        row["error_message"] = (row["error_message"] + "; " if row["error_message"]
                                else "") + f"{type(exc).__name__}: {str(exc)[:120]}"
    return row
def _us(seconds: Optional[float]) -> Any:
    """Seconds -> microseconds, or "" when the arm did not apply / was not timed.
    Empty rather than 0: a 0 us cell reads as an implausible win, and the
    reference arm is legitimately absent where the guard excludes it.
    """
    return round(seconds * 1e6, 3) if seconds else ""
# ── cases ───────────────────────────────────────────────────────────────────
def deep_gemm_cases() -> List[Any]:
    """The host repo's selector perf shapes, from `tests/test.py`.

    The table lives in `tests/test.py` (`HOST_SELECTOR_PERF_SHAPES`), not here,
    because the official perf grid drives the *same* shapes via
    `--host-shapes`: a second copy is a copy that can drift from the gate, and
    the whole point of these rows is that the `deep_gemm` backend has a cell on
    them (every official cell is bf16 and `unsupported`).

    `note` marks the rows whose host `seq_len` is narrower than `n_cols`: those
    are windows in the host grid and whole-row rankings here, so the two are not
    numerically comparable at the same shape.
    """
    return [(_deep_gemm_note(b, v, seq),
             lib.TestParam(b, v, official.HOST_SELECTOR_PERF_TOPK, False, False,
                           False, torch.float32, torch.int32, num_runs=10))
            for b, v, seq in official.HOST_SELECTOR_PERF_SHAPES]


def _deep_gemm_note(b: int, v: int, seq: int) -> str:
    if seq == v:
        return ""
    return (f"sglang-bs{b}-seq{seq}: the host grid declares a window; this "
            f"ranks the whole row")


# `lib.TestParam` has four required fields beyond the shape; a `--cases-file`
# entry that omits them gets the Lightning Indexer's own configuration, which is
# also what the official grid uses.  Spelled out rather than relying on
# `TestParam`'s defaults because those fields have none.
PARAM_DEFAULTS = {"sorted_value": False, "sorted_index": False,
                  "return_value": False, "dtype": "bf16", "out_idx_dtype": "int32"}


def cases_from_file(path: str) -> List[Any]:
    """`--cases-file`: a JSON list of `lib.TestParam` fields.

    `batch_size`, `vocab_size` and `topk` are required.  The four configuration
    fields `TestParam` requires (`sorted_value`, `sorted_index`, `return_value`,
    `out_idx_dtype`) and the `dtype` default to `PARAM_DEFAULTS` -- the Lightning
    Indexer's configuration, which is what the official grid uses.  `dtype` /
    `out_idx_dtype` take the short names above and an unknown one is an error: a
    typo should not become a different case.
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


# ── provenance ──────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""
def device_dir_name() -> str:
    """`perf_data/`'s directory, named after the **device**, not the arch.

    `MetaX C500` -> `MetaX_C500`: the folder this run's record lives in should
    answer "which board was measured", because that is what makes two records
    comparable.  The arch family is not enough -- two parts of one family can
    differ in clocks, wall and SM count while sharing an ISA -- so it is
    recorded per row (`chip`) rather than used to name the directory.

    Falls back to the arch spelling only when torch reports no device name at
    all (a driver quirk, not a normal case).
    """
    name = (torch.cuda.get_device_name(0) or "").strip()
    if not name:
        return f"metax_{_arch.native_target()}"
    return name.replace(" ", "_")


def provenance(sm_count: int) -> Dict[str, Any]:
    here = REPO
    host = os.environ.get("DEEP_GEMM_REPO", "/home/compiler_gfx/tilelang/mcDeepGEMM")
    # The extension the *device* loads, which is the one every number below came
    # from -- `_binding.load(native_target())`, the same name `run_bench.sh`
    # resolves its md5 through.  Deliberately not "the `.so` in `deep_select/`":
    # a tree with several architectures built (the default `CUCC_TARGETS`, one
    # extension per family) has three of them, and taking the first names the
    # C500 artifact in a C600U record.  The manifest is the record of which
    # artifact was measured, so naming the wrong one is worse than naming none.
    target = _arch.native_target()
    sos = sorted(os.path.basename(p) for p in glob.glob(
        os.path.join(here, "deep_select", f"deep_select_{target}*.so")))
    md5 = ""
    if sos:
        md5 = hashlib.md5(open(os.path.join(here, "deep_select", sos[0]),
                               "rb").read()).hexdigest()
    dg = _run(["git", "log", "-1", "--format=%H%n%cd", "--date=iso"], host) \
        if os.path.isdir(os.path.join(host, ".git")) else ""
    dg_lines = dg.splitlines()
    return {
        # The arch family, per row: `metax_xcore<N>`, derived from the device
        # rather than from the directory name (the directory is the device's).
        # This is the column a reader filters on to find same-ISA records.
        "chip": f"metax_{target}",
        # The `perf_data/` directory this run belongs in -- the device's name.
        "device_dir": device_dir_name(),
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": sm_count,
        "torch": torch.__version__,
        "python": platform.python_version(),
        "deep_select_git_commit": _run(["git", "log", "-1", "--format=%H"], here),
        "deep_select_git_dirty": bool(_run(["git", "status", "--porcelain"], here)),
        "extension_so": sos[0] if sos else "",
        "extension_md5": md5,
        "deep_gemm_repo": host,
        "deep_gemm_git": dg_lines[0] if dg_lines else "",
        "deep_gemm_commit_date": dg_lines[1] if len(dg_lines) > 1 else "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
# ── writing ─────────────────────────────────────────────────────────────────
# The cell's shape and configuration, then the backend and its status, then the
# measurement.  `relative_pct_vs_maca_c` is maca_c = 100%, so >100% means that
# backend is faster than this repository's own kernel; it is repeated on every
# row of a cell so a row reads on its own.
COLUMNS = ["chip", "device_name", "sm_count", "git_commit", "extension_md5",
           "case_source", "family", "n_rows", "n_cols", "top_k",
           "sorted_value", "return_value", "input_dtype", "index_dtype",
           "num_runs", "backend", "status", "error_type", "error_message",
           "time(us)", "throughput(TB/s)", "bandwidth(GB/s)", "Byte(MB)",
           "relative_pct_vs_maca_c", "note"]


def rows_for(p, source: str, note: str, got: Dict[str, Dict[str, Any]],
             prov: Dict[str, Any]) -> List[Dict[str, Any]]:
    # The operator's own traffic, the same figure `tests/test.py:138` prints:
    # the input read plus the outputs written.  Computed from the shape rather
    # than from one backend's buffers, so every row of a cell shares it and the
    # bandwidths are comparable across backends (including `torch`, which
    # allocates nothing through this operator).
    nbytes = (p.batch_size * p.vocab_size * p.dtype.itemsize
              + p.batch_size * p.topk
              * (p.dtype.itemsize * int(p.return_value)
                 + p.out_idx_dtype.itemsize))
    ref = got.get("maca_c", {}).get("time(us)")
    out = []
    for arm in ARMS:
        g = got.get(arm)
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
            "backend": arm,
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
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of the backends")
    ap.add_argument("--deep-gemm-axes", action="store_true",
                    help="also run the host repo's fp32 selector grid (the only "
                         "cells the deep_gemm backend can answer)")
    ap.add_argument("--cases-file", default="",
                    help="JSON list of extra cases (see the epilog)")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "perf_data"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan, measure nothing")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown backend {a!r}; expected one of {', '.join(ARMS)}")
    cases = [("official", "", p) for p in official.performance_cases()]
    if args.deep_gemm_axes:
        cases += [("deep_gemm_axes", note, p) for note, p in deep_gemm_cases()]
    if args.cases_file:
        cases += [("extra", note, p) for note, p in cases_from_file(args.cases_file)]
    torch.set_default_device("cuda")
    import deep_select  # noqa: E402  (after set_default_device)
    target = _arch.native_target()
    sm_count = _arch.SM_COUNT[_arch.FAMILY_OF_TARGET[target]]
    if args.dry_run:
        print(f"chip {target}  dir {device_dir_name()}  backends {arms}")
        by = {}
        for source, _note, p in cases:
            by[(source, str(p.dtype), str(p.out_idx_dtype))] = \
                by.get((source, str(p.dtype), str(p.out_idx_dtype)), 0) + 1
        for k, v in sorted(by.items()):
            print(f"  {k[0]:<16} {k[1]:<18} idx {k[2]:<12} {v:>4} cases")
        print(f"  {'TOTAL':<16} {len(cases)} cases x {len(arms)} backends "
              f"= {len(cases) * len(arms)} rows")
        return 0
    stamp = args.tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    prov = provenance(sm_count)
    device_dir = prov["device_dir"]
    out = os.path.join(args.out_dir, device_dir, stamp)
    os.makedirs(out, exist_ok=True)
    started = _dt.datetime.now().astimezone()
    rows: List[Dict[str, Any]] = []
    print(f"chip {prov['chip']}  device {prov['device_name']}  sm {sm_count}  "
          f"backends {arms}  cases {len(cases)}", flush=True)
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
        for arm in arms:
            got[arm] = measure(p, t, arm)
            g = got[arm]
            label = (f"{source[:5]}/{str(p.dtype).replace('torch.','')[:4]} "
                     f"b{p.batch_size}-v{p.vocab_size}-k{p.topk}")
            print(f"{label:<46}{arm:<10}{g['status']:<12}"
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
        "run_id": stamp,
        "output_dir": out,
        "command": " ".join([PYBIN] + sys.argv),
        "started_at_utc": started.astimezone(_dt.timezone.utc).isoformat(),
        "finished_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "csv_files": [os.path.basename(csv_path)],
        "backends": arms,
        "cases": len(cases),
        "rows": len(rows),
        "status_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(counts.items())},
        "csv_format_version": 1,
        "case_source": "tests/test.py::performance_cases() (+ --deep-gemm-axes, "
                       "+ --cases-file)",
        "measurement": ("tests/test.py's own: one 'topk'-matching kernel's time, "
                        "else the e2e span over the matching kernels; p.num_runs "
                        "reps, L2 flushed (kk.bench). Correctness: "
                        "tests/test.py::check_result / check_call_contract, "
                        "applied to every backend except `torch`, whose arm is a "
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
