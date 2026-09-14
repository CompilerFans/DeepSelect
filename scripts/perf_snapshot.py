#!/usr/bin/env python3
"""Record the official performance grid, per backend, as a CSV.

"Official" is load-bearing here, and it is why this file is thin:

  * the cases are `tests/test.py`'s own `performance_cases()` -- the 95-cell grid
    (Lightning Indexer bf16 at 2 topk x 5 batch x 9 sequence lengths, plus the
    fp32 Sampler at 5 batch x 129280), with their own dtypes, index types,
    `NormalFloatDistribution` data and `num_runs=10`;
  * the data is built with `lib.generate_testcase` and seeded through the same
    `kk.Counter` the official run uses;
  * every arm is checked with `test.check_result` / `test.check_call_contract`
    and timed with `test.bench_topk` -- the harness's own assertions and the
    harness's own timing rule (one matching kernel's time, else the span over
    the matching kernels), so nothing here is a second opinion about either.

The one thing this adds is an axis: which backend answered.  Each cell produces
one row per backend, each with a `status` of `pass`, `fail` or `unsupported`.
A backend that cannot serve a cell *says so* rather than being dropped -- the
`deep_gemm` backend ranks float32 only (`topk <= 2048`, unordered), so it is
`unsupported` on all 90 bf16 cells, and a table that omitted those rows would
read as "the two backends agree everywhere" when only 5 cells were compared.

What is recorded per row: the shape, the backend, the status, the time, the
throughput and bandwidth the official run prints, and the relative percentage
against this repository's own kernel (`maca_c` = 100%).

Usage:
    CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \\
        python3 scripts/perf_snapshot.py
    ... --arms maca_c,torch        # a subset of the backends
    ... --out-dir /tmp/x --tag t1  # elsewhere, and named
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import datetime as _dt
import hashlib
import itertools
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

# ── the case table ──────────────────────────────────────────────────────────
#
# A case is a *shape plus a configuration*: which dtype the input is, which
# dtype the indices come back as, whether the output is sorted, how many times
# the case is timed.  They are declarative (`CaseSpec`) so a new case is a data
# edit, not a code edit -- `expand()` takes named axes and produces the
# cartesian product, and a group of cases is just a list of specs.
#
# Three ways to add one, in increasing order of ceremony:
#
#   1. add axes to an existing `expand(...)` call in `DEFAULT_CASE_GROUPS`;
#   2. add a new group to `DEFAULT_CASE_GROUPS` and name it in `--groups`;
#   3. pass `--cases-file extra.json` -- no code change at all.
#
# The `official` group is the one exception, and it is deliberate: it is
# *produced* by `tests/test.py::performance_cases()` rather than restated here,
# so it cannot drift from the gate.  Every other group says `official_cell=0`
# in the CSV.

DTYPES = {
    "bf16": "bfloat16",
    "bfloat16": "bfloat16",
    "fp32": "float32",
    "float32": "float32",
}
IDX_DTYPES = {"int32": "int32", "int64": "int64"}

# The input distributions `lib` provides that can be named from a table.  The
# two `UintDistributionWithHotspotAndSpecifiedPivot` (NaN-hotspot) cases in the
# official correctness table need arguments a table cannot carry, so they are
# not here; nothing in a performance grid uses them.
DISTRIBUTIONS = ("normal_float", "uint_uniform", "uniform_01")


@dataclasses.dataclass
class CaseSpec:
    """One measured case: a shape plus everything the harness needs to build it.

    Field names match `lib.TestParam` so a spec reads as the case it produces.
    """
    batch_size: int
    vocab_size: int
    topk: int
    dtype: str = "bf16"
    out_idx_dtype: str = "int32"
    sorted_value: bool = False
    sorted_index: bool = False
    return_value: bool = False
    enable_end_position: bool = False
    enable_output_idx_offset: bool = False
    input_distrib: str = "normal_float"
    num_runs: int = 10
    # `-1` takes the next seed from the same process-global `kk.Counter` the
    # official run uses, so two snapshots see fresh data exactly as two official
    # runs do.  Any other value pins the case, which is what an A/B needs (the
    # environment traps: a timing comparison must pin the seed yourself).
    seed: int = -1
    note: str = ""

    @property
    def family(self) -> str:
        return "sampler" if self.sorted_value else "lightning_indexer"

    def __post_init__(self) -> None:
        """Reject a spec that names a dtype or distribution this file cannot build.

        Done at construction rather than at `param()` so a bad axis is caught by
        `--dry-run` and by whoever wrote the table, not 40 minutes into a run.
        An unknown name is an error rather than a default: a case silently
        measured at the wrong width is not a measurement.
        """
        for field, table, what in (("dtype", DTYPES, "dtype"),
                                   ("out_idx_dtype", IDX_DTYPES, "index dtype"),
                                   ("input_distrib", DISTRIBUTIONS, "distribution")):
            value = getattr(self, field)
            if value not in table:
                raise ValueError(
                    f"{what} {value!r} is not one of {', '.join(table)} "
                    f"(add it to scripts/perf_snapshot.py::{field.upper()})")

    def param(self, seed: Optional[int] = None):
        """The `lib.TestParam` for this spec."""
        kw: Dict[str, Any] = {}
        if self.input_distrib == "uint_uniform":
            kw["input_distrib"] = lib.UniformUIntDistribution(0x0, 0xFFFFFFFF)
        return lib.TestParam(
            self.batch_size, self.vocab_size, self.topk,
            self.sorted_value, self.sorted_index, self.return_value,
            getattr(torch, DTYPES[self.dtype]), getattr(torch, IDX_DTYPES[self.out_idx_dtype]),
            self.enable_end_position, self.enable_output_idx_offset,
            num_runs=self.num_runs, seed=self.seed if seed is None else seed, **kw)

    @staticmethod
    def from_param(p) -> "CaseSpec":
        """The spec of an official `lib.TestParam` (for the `official` group)."""
        dtype = str(p.dtype).replace("torch.", "")
        idx = str(p.out_idx_dtype).replace("torch.", "")
        return CaseSpec(p.batch_size, p.vocab_size, p.topk,
                        "fp32" if dtype == "float32" else dtype, idx,
                        p.sorted_value, p.sorted_index, p.return_value,
                        p.enable_end_position, p.enable_output_idx_offset,
                        num_runs=p.num_runs, seed=p.seed)


def expand(**axes) -> List[CaseSpec]:
    """Cartesian product of named axes -> specs.  Unknown axis names raise.

    Scalars are broadcast, lists are iterated, so
    `expand(batch_size=[6, 256], vocab_size=4096, topk=[512, 1024])` is two
    batches x one vocab x two topk = four cases.
    """
    fields = {f.name for f in dataclasses.fields(CaseSpec)}
    bad = set(axes) - fields
    if bad:
        raise ValueError(f"unknown case axis {sorted(bad)}; "
                         f"expected any of {sorted(fields)}")
    names = sorted(axes)
    values = [[(n, axes[n])] if not isinstance(axes[n], (list, tuple))
              else [(n, v) for v in axes[n]] for n in names]
    out: List[CaseSpec] = []
    for combo in itertools.product(*values):
        kwargs = dict(combo)
        missing = {"batch_size", "vocab_size", "topk"} - set(kwargs)
        if missing:
            raise ValueError(f"case axes {sorted(missing)} are required "
                             f"(a case with no shape is not a case)")
        out.append(CaseSpec(**kwargs))
    return out


def _official_group() -> List[CaseSpec]:
    """`tests/test.py::performance_cases()` -- the 95-cell gate grid.

    Produced by the harness rather than restated, so the snapshot's `official`
    rows are the same cases the official run measures, by construction.
    """
    return [CaseSpec.from_param(p) for p in official.performance_cases()]


def _deep_gemm_grid() -> List[CaseSpec]:
    """The host repository's `SELECTOR_PERF_SHAPES`, `top_k=2048`, fp32.

    Transcribed from `deep_gemm/tests/test_indexer_topk_selector.py:78-123`
    (three named families: test-topk, sglang, dsa).  Kept as a group because it
    is the only grid here shaped for a *different* implementation than this
    repository's -- the comparison it exists for is `maca_c` vs `deep_gemm`.

    Its data is `torch.randn` in the host repo and `NormalFloatDistribution`
    here (the harness's own generator), so a row in this group and a row in
    `official` are not comparable even at the same shape.
    """
    out: List[CaseSpec] = []
    out += expand(batch_size=[1, 16, 132, 512], vocab_size=66551, topk=2048,
                  dtype="fp32", note="test-topk")
    for b in (1, 132, 256, 4096):
        for seq in (2048, 4096, 16384, 65536):
            out += [CaseSpec(b, 131072, 2048, "fp32",
                             note=f"sglang-bs{b}-seq{seq}; seq_len is a column "
                                  f"in the host grid -- this adapter ranks the "
                                  f"whole row")]
    out += expand(batch_size=[1, 16, 132, 256, 4096], vocab_size=107520,
                  topk=2048, dtype="fp32", note="dsa")
    return out


# name -> callable, so a table that is expensive to build is only built when asked
DEFAULT_CASE_GROUPS = {
    "official": _official_group,
    "deep_gemm_grid": _deep_gemm_grid,
}

# A group a caller supplies at run time.  Same shape of data as `expand`'s
# keywords, plus the group name:
#
#   {"fp8_probe": {"batch_size": [1, 256], "vocab_size": [32768, 131072],
#                  "topk": [1024], "dtype": ["fp32"], "num_runs": [20]}}
#
# Scalars and lists are both accepted, exactly as in `expand`.
def load_case_file(path: str) -> Dict[str, List[CaseSpec]]:
    with open(path) as f:
        raw = json.load(f)
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected an object of group -> axes")
    return {name: expand(**axes) for name, axes in raw.items()}


# Measured streaming-read wall per family, in GB/s: what a pure `uint4`
# grid-stride read of the same volume achieves on a quiet part.  Recorded as a
# *column* so `bandwidth_pct_of_wall` is reproducible from the CSV rather than
# from a number in someone's notes.  C500: the ledger's 1,487 GB/s streaming
# read (1,344 mixed).  C600U: 1,545 GB/s, from a purpose-written read kernel --
# a torch reduction on the same device measures 274 GB/s and is not the wall.
READ_WALL_GBPS = {
    1000: 1487.0,
    1500: None,      # not measured on this repository
    1600: 1545.0,
}

# `tests/test.py:138`'s own label for the input+output traffic is what the
# official printout calls "TB/s"; the same number as GB/s is recorded beside it
# because a bandwidth reader wants the unit they can compare to a wall.
COUNTER = kk.Counter()


class Unsupported(Exception):
    """The backend refused the case; carried as a status, never as a failure."""


def call_topk(p, t, backend: str):
    """Exactly the call `test.run_testcase` makes, with a backend named.

    Kept byte-for-byte in step with `run_testcase`'s argument list on purpose:
    an arm is only comparable if it is asked the same question.
    """
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


def measure_arm(p, t, backend: str) -> Dict[str, Any]:
    """One `(case, backend)` row: status first, then the numbers.

    A `fail` row still carries its time when one could be taken -- a case that
    selects wrong is a defect whatever it runs at, and hiding the measurement
    would hide which of the two problems it is.
    """
    row: Dict[str, Any] = {"status": "pass", "error_type": "", "error_message": ""}
    try:
        value, index = call_topk(p, t, backend)
        torch.cuda.synchronize()
        official.check_call_contract(p, value, index)
    except Unsupported as exc:
        row["status"] = "unsupported"
        row["error_type"] = "UnsupportedByBackend"
        row["error_message"] = str(exc)[:200]
        return row
    except Exception as exc:
        row["status"] = "fail"
        row["error_type"] = type(exc).__name__
        row["error_message"] = str(exc)[:200]
        return row

    if p.check_correctness:
        try:
            ok = official.check_result(p, t, value.clone() if value is not None else None,
                                       index.clone())
        except Exception as exc:
            row["status"] = "fail"
            row["error_type"] = type(exc).__name__
            row["error_message"] = str(exc)[:200]
            return row
        if not ok:
            row["status"] = "fail"
            row["error_type"] = "CheckFailed"
            row["error_message"] = "check_result() returned False"
    del value, index

    if p.num_runs > 0:
        # `bench_topk` takes the call's outputs because the traffic figure
        # counts the output buffers too, so one run is done to size them (the
        # timing run itself re-executes `fn`, as `tests/test.py` does).
        value, index = call_topk(p, t, backend)
        torch.cuda.synchronize()
        fn = lambda: call_topk(p, t, backend)          # noqa: E731
        usage, total_size = official.bench_topk(fn, p, t, value, index)
        del value, index
        row["Byte(MB)"] = round(total_size / 1e6, 3)
        if usage:
            row["time(us)"] = round(usage * 1e6, 3)
            row["throughput(TB/s)"] = round(total_size / usage / 1e12, 6)
            row["bandwidth(GB/s)"] = round(total_size / usage / 1e9, 3)
            row["logical_read_bandwidth(GB/s)"] = round(
                p.batch_size * p.vocab_size * t.input.element_size() / usage / 1e9, 3)
        else:
            # No kernel name contains "topk": `tests/test.py` skips the print for
            # exactly this reason (a small cell's torch.topk lowers to
            # `gatherTopK_opt`).  Not timed, and said so rather than recorded as 0.
            row["error_message"] = (row["error_message"] + "; " if row["error_message"] else "") \
                + "not timed: no kernel name contains \"topk\""
    return row


# ── provenance ──────────────────────────────────────────────────────────────

def _run(cmd: List[str], cwd: str) -> str:
    try:
        return subprocess.run(cmd, cwd=cwd, capture_output=True, text=True,
                              timeout=20).stdout.strip()
    except Exception:
        return ""


def provenance(chip: str, sm_count: int) -> Dict[str, Any]:
    here = REPO
    host = os.environ.get("DEEP_GEMM_REPO", "/home/compiler_gfx/tilelang/mcDeepGEMM")
    sos = sorted(f for f in os.listdir(os.path.join(here, "deep_select"))
                 if f.endswith(".so"))
    md5 = ""
    if sos:
        md5 = hashlib.md5(open(os.path.join(here, "deep_select", sos[0]),
                               "rb").read()).hexdigest()
    dg_commit = _run(["git", "log", "-1", "--format=%H%n%cd", "--date=iso"],
                     host) if os.path.isdir(os.path.join(host, ".git")) else ""
    dg_lines = dg_commit.splitlines()
    return {
        "chip": chip,
        "device_name": torch.cuda.get_device_name(0),
        "sm_count": sm_count,
        "device_count": torch.cuda.device_count(),
        "torch": torch.__version__,
        "python": platform.python_version(),
        "deep_select_git_commit": _run(["git", "log", "-1", "--format=%H"], here),
        "deep_select_git_branch": _run(["git", "rev-parse", "--abbrev-ref", "HEAD"], here),
        "deep_select_git_dirty": bool(_run(["git", "status", "--porcelain"], here)),
        "extension_so": sos[0] if sos else "",
        "extension_md5": md5,
        "deep_gemm_repo": host,
        "deep_gemm_git": dg_lines[0] if dg_lines else "",
        "deep_gemm_commit_date": dg_lines[1] if len(dg_lines) > 1 else "",
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


# ── writing ─────────────────────────────────────────────────────────────────

# The cells' own shape and configuration, then the backend's identity and
# status, then the measurement.  `relative_pct_vs_maca_c` is defined against
# `maca_c` = 100%, so **>100% means that backend is faster than this
# repository's own kernel**; it is repeated on every row of a cell so a reader
# never has to find the other row to interpret this one.
#
# `bandwidth_pct_of_wall` is *not* a fraction of the kernel's roof: it is
# `logical_read_bandwidth(GB/s)` -- the input read alone, no output buffers --
# over the measured streaming-read wall.  The official grid sits at a median
# 19% of the C500 wall while the same kernel's pass 1 measures 94.9-97.9% of it
# on a full-length row, because most of these cells are small and the grid is
# weighted by batch rather than bytes.  Read it with the cell's batch and row
# length, never as headroom.
COLUMNS = ["chip", "device_name", "sm_count", "git_commit", "extension_md5",
           "case_group", "official_cell", "family", "n_rows", "n_cols", "top_k",
           "sorted_value", "return_value", "input_dtype", "index_dtype",
           "num_runs", "backend", "status", "error_type", "error_message",
           "time(us)", "throughput(TB/s)", "bandwidth(GB/s)",
           "logical_read_bandwidth(GB/s)", "Byte(MB)",
           "speedup_vs_torch", "relative_pct_vs_maca_c",
           "bandwidth_pct_of_wall", "note"]


def to_rows(spec: CaseSpec, p, got: Dict[str, Dict[str, Any]],
            prov: Dict[str, Any], group: str) -> List[Dict[str, Any]]:
    """One row per backend, with the case's shared columns repeated on each.

    Repetition is deliberate: `relative_pct_vs_maca_c` is only readable if the
    row that defines 100% is in the same file, and a per-row `backend` column
    means the whole CSV is one table rather than a table per backend.
    """
    nbytes = spec.batch_size * spec.vocab_size * p.dtype.itemsize
    family_num = _arch.FAMILY_OF_TARGET[_arch.native_target()]
    wall = READ_WALL_GBPS.get(family_num)
    ref_us = got.get("maca_c", {}).get("time(us)")
    ref_torch = got.get("torch", {}).get("time(us)")

    out: List[Dict[str, Any]] = []
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
            "case_group": group, "official_cell": int(group == "official"),
            "family": spec.family, "n_rows": spec.batch_size,
            "n_cols": spec.vocab_size, "top_k": spec.topk,
            "sorted_value": int(spec.sorted_value),
            "return_value": int(spec.return_value),
            "input_dtype": spec.dtype, "index_dtype": spec.out_idx_dtype,
            "num_runs": spec.num_runs,
            "backend": arm,
            "status": g["status"],
            "error_type": g.get("error_type", ""),
            "error_message": g.get("error_message", ""),
            "note": spec.note,
        })
        for k in ("time(us)", "throughput(TB/s)", "bandwidth(GB/s)",
                  "logical_read_bandwidth(GB/s)", "Byte(MB)"):
            if k in g:
                row[k] = g[k]
        us = g.get("time(us)")
        if us:
            # `maca_c` is the reference: its speed is 100%, and another
            # backend's percentage is how fast it is relative to it.
            row["relative_pct_vs_maca_c"] = (
                round(ref_us / us * 100, 2) if ref_us else "")
            row["speedup_vs_torch"] = (round(ref_torch / us, 3) if ref_torch else "")
            if wall:
                row["bandwidth_pct_of_wall"] = round(
                    nbytes / (us * 1e-6) / 1e9 / wall * 100, 2)
        out.append(row)
    return out


def write_csv(path: str, rows: List[Dict[str, Any]]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def resolve_groups(names: List[str], case_file: str) -> Dict[str, List[CaseSpec]]:
    """Name -> specs, for the groups asked for.

    A name is looked up in the built-in table first, then in `--cases-file`, so
    a run-time group can shadow a built-in one (that is how a one-off grid is
    measured without editing this file) and the CSV's `case_group` column names
    which of the two produced a row.
    """
    external = load_case_file(case_file) if case_file else {}
    out: Dict[str, List[CaseSpec]] = {}
    for name in names:
        if name in external:
            out[name] = external[name]
        elif name in DEFAULT_CASE_GROUPS:
            out[name] = DEFAULT_CASE_GROUPS[name]()
        else:
            raise SystemExit(
                f"unknown case group {name!r}; built-in groups are "
                f"{', '.join(DEFAULT_CASE_GROUPS)}, or pass --cases-file")
        if not out[name]:
            raise SystemExit(f"case group {name!r} is empty")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
case groups (--groups):
  official         tests/test.py::performance_cases() -- the 95-cell gate grid
                   (default).  Produced by the harness, so it cannot drift.
  deep_gemm_grid   the host repo's SELECTOR_PERF_SHAPES, top_k=2048, fp32 --
                   the grid that exists for the maca_c-vs-deep_gemm comparison.
                   Its data (NormalFloatDistribution) differs from the host
                   repo's own (torch.randn), so its rows are not comparable to
                   the official ones even at the same shape.

adding cases:
  --cases-file FILE   a JSON object of group -> axes, e.g.
                        {"fp8_probe": {"batch_size": [1, 256],
                                       "vocab_size": [32768, 131072],
                                       "topk": [1024], "dtype": ["fp32"],
                                       "num_runs": [20]}}
                      Every `CaseSpec` field is an axis; a scalar is
                      broadcast, a list is iterated.  An unknown axis or dtype
                      is an error, not a default.

backends: """ + ", ".join(ARMS))
    ap.add_argument("--arms", default=",".join(ARMS),
                    help="comma-separated subset of the backends")
    ap.add_argument("--groups", default="official",
                    help="comma-separated case groups (default: official)")
    ap.add_argument("--cases-file", default="",
                    help="JSON file of extra case groups (see the epilog)")
    ap.add_argument("--out-dir", default=os.path.join(REPO, "perf_data"))
    ap.add_argument("--tag", default="")
    ap.add_argument("--dry-run", action="store_true",
                    help="expand the groups, print the plan, measure nothing")
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if a]
    for a in arms:
        if a not in ARMS:
            raise SystemExit(f"unknown backend {a!r}; expected one of {', '.join(ARMS)}")
    groups = resolve_groups([g for g in args.groups.split(",") if g], args.cases_file)

    torch.set_default_device("cuda")
    import deep_select  # noqa: E402  (after set_default_device)

    target = _arch.native_target()
    family_num = _arch.FAMILY_OF_TARGET[target]
    chip = f"metax_{target}"
    sm_count = _arch.SM_COUNT[family_num]

    if args.dry_run:
        print(f"chip {chip}  backends {arms}")
        total = 0
        for name, specs in groups.items():
            print(f"  group {name:<16} {len(specs):>5} cases "
                  f"x {len(arms)} backends = {len(specs) * len(arms)} rows")
            by_dtype: Dict[Any, int] = {}
            for s in specs:
                by_dtype[(s.dtype, s.out_idx_dtype)] = by_dtype.get((s.dtype, s.out_idx_dtype), 0) + 1
            for k, v in sorted(by_dtype.items()):
                print(f"      {k[0]:<10} idx {k[1]:<6} {v}")
            total += len(specs) * len(arms)
        print(f"  total rows: {total}")
        return 0

    stamp = args.tag or _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, chip, stamp)
    os.makedirs(out, exist_ok=True)

    # A spec's `seed` of -1 takes the next seed from the same process-global
    # `kk.Counter` the official run uses, so two snapshots see fresh data exactly
    # as two official runs do; any other value pins the case.
    cases: List[Any] = []
    for name, specs in groups.items():
        for spec in specs:
            p = spec.param()
            if spec.seed == -1:
                p.seed = COUNTER.next()
            cases.append((name, spec, p))

    n_cases = len(cases)
    prov = provenance(chip, sm_count)
    started = _dt.datetime.now().astimezone()
    rows: List[Dict[str, Any]] = []
    print(f"chip {chip}  device {torch.cuda.get_device_name(0)}  sm {sm_count}  "
          f"backends {arms}  cases {n_cases}  rows {n_cases * len(arms)}", flush=True)
    print(f"{'case':<44}{'backend':<10}{'status':<13}{'us':>12}{'GB/s':>10}", flush=True)

    for i, (group, spec, p) in enumerate(cases):
        t0 = time.time()
        try:
            t = lib.generate_testcase(p)
        except Exception as exc:                # OOM guard, mirrors test.py:194
            print(f"  generate_testcase failed for {spec}: {exc}", flush=True)
            break
        got: Dict[str, Dict[str, Any]] = {}
        for arm in arms:
            got[arm] = measure_arm(p, t, arm)
            g = got[arm]
            label = (f"{group[:5]}/{spec.family[:6]} {spec.dtype:<5}"
                     f" b{spec.batch_size}-v{spec.vocab_size}-k{spec.topk}")
            print(f"{label:<44}{arm:<10}{g['status']:<13}"
                  f"{g.get('time(us)', '')!s:>12}{g.get('bandwidth(GB/s)', '')!s:>10}"
                  + (f"  [{g['error_message'][:38]}]" if g.get("error_message") else ""),
                  flush=True)
        rows.extend(to_rows(spec, p, got, prov, group))
        del t, got
        torch.cuda.empty_cache()
        if i == 0 or (i + 1) % 10 == 0:
            print(f"  ... {i + 1}/{n_cases} cases ({time.time() - t0:.1f}s)", flush=True)

    csv_path = os.path.join(out, "deepselect_perf.csv")
    write_csv(csv_path, rows)

    n_by: Dict[Any, int] = {}
    for r in rows:
        n_by[(r["backend"], r["status"])] = n_by.get((r["backend"], r["status"]), 0) + 1
    manifest = dict(prov)
    manifest.update({
        "run_id": stamp,
        "output_dir": out,
        "command": " ".join([PYBIN] + sys.argv),
        "started_at_utc": started.astimezone(_dt.timezone.utc).isoformat(),
        "finished_at_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(),
        "csv_files": [os.path.basename(csv_path)],
        "backends": arms,
        "case_groups": {name: len(specs) for name, specs in groups.items()},
        "cases": n_cases,
        "rows": len(rows),
        "status_counts": {f"{k[0]}/{k[1]}": v for k, v in sorted(n_by.items())},
        "read_wall_gbps": READ_WALL_GBPS.get(family_num),
        "csv_format_version": 1,
        "case_source": ("tests/test.py::performance_cases() for the `official` "
                        "group; scripts/perf_snapshot.py's own tables otherwise"),
        "measurement": ("tests/test.py's own rule: one 'topk'-matching kernel's "
                        "time, else the e2e span over the matching kernels; "
                        "num_runs per case (10 for the official grid), L2 "
                        "flushed between reps (kk.bench default). "
                        "Correctness: tests/test.py::check_result / "
                        "check_call_contract, applied to every backend."),
    })
    with open(os.path.join(out, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    print(f"\nwrote {out}/deepselect_perf.csv  ({len(rows)} rows, {n_cases} cases)",
          flush=True)
    for k, v in sorted(n_by.items()):
        print(f"  {k[0]:<10} {k[1]:<12} {v}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
