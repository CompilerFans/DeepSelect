#!/usr/bin/env python3
"""Record the official performance axes for every backend, as a CSV snapshot.

This runs the *official* axes (``tests/test.py``'s own `performance_cases`,
`test.py:225-242`) -- same `lib.TestParam`, same `lib.generate_testcase`, same
`kk.bench` and the same single-kernel-name-else-e2e timing rule the official
harness uses -- so the numbers are the operator's own cells and not a
re-derivation of them.  It does not re-run the correctness table; it *does*
check the contract of every arm before it is timed, and records the result, so
a case that selects wrong is visible in the file rather than silently averaged.

Three arms per eligible cell:

  maca_c     this repository's kernel (the default backend)
  torch      the reference arm, measured the way `tests/test.py` measures it
             (`torch.topk`), so the comparison is against the official one
  deep_gemm  the host repository's `fp32_indexer_topk_selector`, reached
             lazily through `deep_select.topk(backend="deep_gemm")`

`deep_gemm` implements a strict subset (float32 only, `topk <= 2048`,
unordered).  Where it cannot serve a cell the row is still written, with an
empty `deep_gemm_us` and the reason in `note` -- a cell that silently vanished
from the table would read as "covered".

Output layout, following the host repository's `deep_gemm/tests/perf_data/`
convention:

    perf_data/<chip>/<YYYYmmdd_HHMMSS>/manifest.json
    perf_data/<chip>/<YYYYmmdd_HHMMSS>/deepselect_official_axes.csv

and, when `--include-deep-gemm-axes` is given, a second CSV over the host
repository's own test-file axes for the same operator
(`deep_gemm/tests/test_indexer_topk_selector.py`), so the two files can be
compared by `(n_rows, n_cols, top_k)` without either one being re-derived here.

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \\
        python3 scripts/perf_snapshot.py
    ... --arms maca_c,torch            # skip the deep_gemm arm entirely
    ... --out-dir /tmp/x --tag trial1  # elsewhere, and named
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
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

from deep_select import _arch  # noqa: E402

PYBIN = sys.executable
NONCE = os.environ.get("PERF_SNAPSHOT_NONCE", "0")

# ── what a cell is ──────────────────────────────────────────────────────────

@dataclasses.dataclass
class Cell:
    n_rows: int
    n_cols: int
    top_k: int
    sorted_value: bool
    return_value: bool
    dtype: torch.dtype
    index_dtype: torch.dtype
    family: str          # "lightning_indexer" | "sampler"

    @property
    def key(self):
        return (self.n_rows, self.n_cols, self.top_k, self.sorted_value,
                self.return_value, str(self.dtype))


def official_axes_cells() -> List[Cell]:
    """`tests/test.py`'s `performance_cases`, verbatim in shape.

    Two deliberate departures, both recorded per row:
      * `dtype=torch.float` for the Lightning Indexer rows as well.  The
        official grid is bf16 there, which `backend="deep_gemm"` refuses
        outright; running the axis in fp32 is what makes the cell comparable
        across all three arms.  `perf_snapshot_fp32_widened` says so per row.
      * the Sampler rows keep the official `sorted_value=True`, which
        `deep_gemm` also refuses.  Widening those would change the operator
        (the sort is part of the case), so they stay official and simply have
        no `deep_gemm` arm.
    """
    cells = []
    for topk in (512, 1024):
        for b in (6, 256, 512, 768, 4096):
            for seqlen in (256, 1024, 4096, 16384, 65536, 131072, 262144,
                           524288, 1048576):
                cells.append(Cell(b, seqlen, topk, False, False, torch.float32,
                                  torch.int32, "lightning_indexer"))
    for b in (6, 256, 512, 768, 4096):
        cells.append(Cell(b, 129280, 512, True, True, torch.float32,
                          torch.int64, "sampler"))
    return cells


def deep_gemm_axes_cells() -> List[Cell]:
    """The host repo's `test_indexer_topk_selector.py` grid, read from it.

    Imported rather than transcribed: a second copy of another repository's
    grid is a copy that goes stale.  The rows it yields are `(n_rows, n_cols,
    top_k)`; the dtype is fp32 by that file's own definition.
    """
    host = os.environ.get("DEEP_GEMM_REPO",
                          "/home/compiler_gfx/tilelang/mcDeepGEMM")
    path = os.path.join(host, "deep_gemm", "tests",
                        "test_indexer_topk_selector.py")
    if not os.path.exists(path):
        return []
    src = open(path).read()
    cells: List[Cell] = []
    seen = set()
    import re
    # `(n_rows, n_cols, top_k)` triples as they appear in that file's grids.
    for m in re.finditer(r"\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\)", src):
        r, c, k = (int(x) for x in m.groups())
        if c < 2 or k < 1 or k > 2048 or (r, c, k) in seen:
            continue
        seen.add((r, c, k))
        cells.append(Cell(r, c, k, False, True, torch.float32, torch.int32,
                          "deep_gemm_axes"))
    return cells


# ── measurement ─────────────────────────────────────────────────────────────

def _timed(fn, reps: int) -> Optional[float]:
    """The official timing rule: one matching kernel's time, else the span.

    Returns None when nothing matched.  `tests/test.py` treats that as "time 0"
    and skips the print (`test.py:157-160`); here it is an empty CSV field,
    because a small cell's `torch.topk` can lower to kernels none of whose
    names contain "topk", and "0 us" would read as an implausibly good number.
    """
    res = kk.bench(fn, reps)
    names = [s for s in res.get_kernel_names() if "topk" in s]
    if not names:
        return None
    if len(names) == 1:
        return res.get_kernel_time(names[0])
    return res.get_e2e_time(names)


def contract_ok(x: torch.Tensor, k: int, idx: torch.Tensor,
                end: Optional[torch.Tensor] = None) -> Optional[bool]:
    """`tests/test.py`'s own checks, as a predicate.

    Two corrections against the naive form, both of which the official harness
    makes and a hand-written check does not:

      * **A row selects `min(vocab, topk)` entries, not `topk`.**  When the row
        is shorter than `topk` the tail slots are out-of-band fill and neither
        their indices nor their values mean anything (`test.py:70-75`).  Callers
        pass `end` so the visible length is known; otherwise every short cell
        reports a spurious failure.
      * **A NaN row's slot 0 is the guard value** and the row is excluded from
        the value checks (`test.py:77-82`).

    Returns None for "not checkable" (relative to the other arms' *sets*, which
    is what `sets_match_*` records) rather than a misleading False.
    """
    n_rows, n_cols = x.shape
    i64 = idx.to(torch.int64)
    visible = (torch.full((n_rows,), n_cols, dtype=torch.int64, device=x.device)
               if end is None else end.to(torch.int64))
    counts = torch.clamp(visible, max=k)
    selected = (torch.arange(k, device=x.device).unsqueeze(0)
                < counts.unsqueeze(1))
    if not bool(selected.any()):
        return None

    nan_rows = x.isnan().any(dim=1) if bool(x.isnan().any()) else None
    if nan_rows is not None:
        guarded = (i64[:, 0] == 0x3F3F3F3F) | (visible.clamp(max=k) <= k)
        if not bool(guarded[nan_rows].all()):
            return False
        selected = selected & ~nan_rows.unsqueeze(1)

    in_range = (i64 >= 0) & (i64 < visible.unsqueeze(1))
    if not bool((in_range | ~selected).all()):
        return False
    masked = torch.where(selected, i64, torch.iinfo(torch.int64).max)
    srt = masked.sort(dim=1).values
    if bool(((srt[:, 1:] == srt[:, :-1]) & selected[:, 1:]).any()):
        return False

    safe = torch.where(selected & in_range, i64, 0)
    gathered = x.gather(1, safe)
    rest = x.clone()
    rest.scatter_(1, safe, float("-inf"))
    # The visible window only: `row_wise_masked_fill_` is what the official
    # harness uses to push the columns past `end` out of the comparison.
    tail = torch.arange(n_cols, device=x.device).unsqueeze(0) >= visible.unsqueeze(1)
    rest = rest.masked_fill(tail, float("-inf"))
    sel_min = gathered.masked_fill(~selected, float("inf")).amin(dim=1)
    return bool((sel_min >= rest.amax(dim=1)).all())


def measure(cell: Cell, arms: List[str]) -> Dict[str, Any]:
    from lib import TestParam
    row: Dict[str, Any] = {}
    p = TestParam(batch_size=cell.n_rows, vocab_size=cell.n_cols,
                  topk=cell.top_k, sorted_value=cell.sorted_value,
                  sorted_index=False, return_value=cell.return_value,
                  dtype=cell.dtype, out_idx_dtype=cell.index_dtype,
                  seed=(cell.n_rows * 1000003 + cell.n_cols * 101 + cell.top_k
                        + int(NONCE)),
                  num_runs=10)
    try:
        t = lib.generate_testcase(p)
    except Exception as e:                      # OOM guard, mirrors test.py:194
        row["note"] = f"generate_testcase failed: {type(e).__name__}"
        return row
    x = t.input

    def call(backend: str):
        return __import__("deep_select").topk(
            x, cell.top_k, sorted=cell.sorted_value, end=t.end,
            indices_type=cell.index_dtype, sorted_index=False,
            output_idx=None, output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=cell.return_value, abort_when_nan_found=False,
            backend=backend)

    notes: List[str] = []
    idx_ref: Optional[torch.Tensor] = None

    for arm in arms:
        try:
            _, idx = call(arm)
            torch.cuda.synchronize()
        except Exception as e:
            row[f"{arm}_us"] = ""
            row[f"{arm}_ok"] = ""
            notes.append(f"{arm}: {type(e).__name__}: {str(e)[:90]}")
            continue
        ok = contract_ok(x, cell.top_k, idx, t.end)
        us = _timed(lambda: call(arm), p.num_runs)
        row[f"{arm}_us"] = round(us * 1e6, 2) if us is not None else ""
        if us is None:
            notes.append(f"{arm}: no kernel name contains \"topk\" (the filter "
                         f"is case-sensitive, so torch's `gatherTopK_opt` does "
                         f"not match either) -- not timed, as in tests/test.py")
        row[f"{arm}_ok"] = "" if ok is None else int(ok)
        if arm == "maca_c":
            row["sets_match_maca"] = ""
            idx_ref = idx.sort(dim=1).values
        elif idx_ref is not None:
            row[f"sets_match_{arm}"] = int(bool(torch.equal(
                idx.sort(dim=1).values, idx_ref)))
        else:
            row[f"sets_match_{arm}"] = ""

    row["note"] = "; ".join(notes)
    del t, x
    torch.cuda.empty_cache()
    return row


# ── provenance ──────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance(extra: Dict[str, Any]) -> Dict[str, Any]:
    here = REPO
    host = os.environ.get("DEEP_GEMM_REPO",
                          "/home/compiler_gfx/tilelang/mcDeepGEMM")
    so = [f for f in os.listdir(os.path.join(here, "deep_select"))
          if f.endswith(".so")]
    import hashlib
    md5 = ""
    if so:
        md5 = hashlib.md5(open(os.path.join(here, "deep_select", so[0]),
                               "rb").read()).hexdigest()
    dg_commit = _run(["git", "log", "-1", "--format=%H%n%cd", "--date=iso"],
                     host) if os.path.isdir(os.path.join(host, ".git")) else ""
    return {
        "crate": "deep_select",
        "chip": extra["chip"],
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": extra["sm_count"],
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "deep_select_git_commit": _run(["git", "log", "-1", "--format=%H"], here),
        "deep_select_git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], here),
        "deep_select_git_dirty": bool(_run(["git", "status", "--porcelain"], here)),
        "extension_so": so[0] if so else "",
        "extension_md5": md5,
        "deep_gemm_repo": host,
        "deep_gemm_git": dg_commit.splitlines()[0] if dg_commit else "",
        "deep_gemm_commit_date": (dg_commit.splitlines()[1]
                                  if len(dg_commit.splitlines()) > 1 else ""),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "git_commit": _run(["git", "log", "-1", "--format=%H"], here),
        "git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], here),
    }


# ── writing ─────────────────────────────────────────────────────────────────

COLUMNS = ["chip", "device_name", "sm_count", "git_commit", "extension_md5",
           "family", "n_rows", "n_cols", "top_k", "sorted_value",
           "return_value", "input_dtype", "index_dtype", "torch_us",
           "maca_c_us", "deep_gemm_us", "speedup_vs_torch",
           "speedup_vs_deep_gemm", "maca_c_ok", "torch_ok", "deep_gemm_ok",
           "sets_match_torch", "sets_match_deep_gemm",
           "logical_gbps_maca_c", "note"]


def to_row(cell: Cell, got: Dict[str, Any], prov: Dict[str, Any]) -> Dict[str, Any]:
    # Logical GB/s is the read the operator must do: every element of every row
    # once, at the input's width.  The official harness prints the same
    # quantity (`test.py:138`).
    nbytes = cell.n_rows * cell.n_cols * torch.tensor([], dtype=cell.dtype).element_size()
    got = dict(got)
    mc = got.get("maca_c_us")
    got["logical_gbps_maca_c"] = (round(nbytes / (mc * 1e-6) / 1e9, 1)
                                  if isinstance(mc, float) and mc else "")
    tg = got.get("torch_us")
    dg = got.get("deep_gemm_us")
    got["speedup_vs_torch"] = (round(tg / mc, 3)
                               if isinstance(mc, float) and mc and isinstance(tg, float) and tg else "")
    got["speedup_vs_deep_gemm"] = (round(dg / mc, 3)
                                   if isinstance(mc, float) and mc and isinstance(dg, float) and dg else "")
    out = {c: "" for c in COLUMNS}
    out.update({k: v for k, v in prov.items() if k in COLUMNS})
    out.update({"family": cell.family, "n_rows": cell.n_rows,
                "n_cols": cell.n_cols, "top_k": cell.top_k,
                "sorted_value": int(cell.sorted_value),
                "return_value": int(cell.return_value),
                "input_dtype": str(cell.dtype).replace("torch.", ""),
                "index_dtype": str(cell.index_dtype).replace("torch.", "")})
    for k, v in got.items():
        if k in out:
            out[k] = v
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="maca_c,torch,deep_gemm")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "perf_data"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--include-deep-gemm-axes", action="store_true",
                    help="also snapshot the host repo's own topk-selector grid")
    ap.add_argument("--families", default="lightning_indexer,sampler")
    ap.add_argument("--max-elements", type=float, default=2 ** 28,
                    help="skip a cell whose n_rows*n_cols exceeds this")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]

    torch.set_default_device("cuda")
    import deep_select  # noqa: E402  (after set_default_device)

    target = _arch.native_target()
    family_num = _arch.FAMILY_OF_TARGET[target]
    chip = f"metax_{target}"
    sm_count = _arch.SM_COUNT[family_num]

    stamp = args.tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, chip, stamp)
    os.makedirs(out, exist_ok=True)

    wants = set(args.families.split(","))
    cells = [c for c in official_axes_cells() if c.family in wants]
    cells = [c for c in cells if c.n_rows * c.n_cols <= args.max_elements]

    prov = provenance({"chip": chip, "sm_count": sm_count})
    started = _dt.datetime.now().astimezone()
    rows: List[Dict[str, Any]] = []
    print(f"chip {chip}  device {torch.cuda.get_device_name(0)}  "
          f"sm {sm_count}  arms {arms}  cells {len(cells)}", flush=True)
    print(f"{'cell':<34}{'torch':>10}{'maca_c':>10}{'deep_gemm':>11}  ok", flush=True)
    for i, c in enumerate(cells):
        t0 = time.time()
        got = measure(c, arms)
        rows.append(to_row(c, got, prov))
        cell_s = f"{c.family[:6]} b{c.n_rows}-v{c.n_cols}-k{c.top_k}"
        print(f"{cell_s:<34}{got.get('torch_us', ''):>10}"
              f"{got.get('maca_c_us', ''):>10}{got.get('deep_gemm_us', ''):>11}"
              f"  {'ok' if got.get('maca_c_ok') else ('FAIL' if got.get('maca_c_ok') == 0 else '-')}"
              f"  ({time.time() - t0:.1f}s)"
              + (f"  [{got['note'][:60]}]" if got.get("note") else ""), flush=True)

    csv_path = os.path.join(out, "deepselect_official_axes.csv")
    write_csv(csv_path, rows)

    files = [os.path.basename(csv_path)]
    if args.include_deep_gemm_axes:
        dg_cells = deep_gemm_axes_cells()
        dg_rows = [to_row(c, measure(c, ["maca_c", "torch"]), prov)
                   for c in dg_cells]
        p2 = os.path.join(out, "deepselect_deep_gemm_axes.csv")
        write_csv(p2, dg_rows)
        files.append(os.path.basename(p2))
        print(f"deep_gemm-axes rows: {len(dg_rows)}", flush=True)

    manifest = dict(prov)
    manifest.update({
        "run_id": stamp,
        "output_dir": out,
        "command": " ".join([PYBIN] + sys.argv),
        "started_at_utc": started.astimezone(_dt.timezone.utc).isoformat(),
        "finished_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "csv_files": files,
        "arms": arms,
        "cells_requested": len(cells),
        "cells_written": len(rows),
        "csv_format_version": 1,
        "measurement": ("kernel time from a single 'topk'-matching kernel, "
                        "else the e2e span over matching kernels (tests/test.py:139-144); "
                        "10 reps, L2 flushed between reps (kk.bench default)"),
    })
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {out}/  ({', '.join(files)}, manifest.json)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
