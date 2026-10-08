#!/usr/bin/env python3
"""Render the README's performance figures from a `perf_snapshot.py` recording.

Two figures, both drawn from one `deepselect_perf.csv` so the picture and the
record it came from cannot disagree:

* `perf_bf16_maca.png` -- Lightning Indexer: bfloat16, `topk = 512`, one subplot
  per batch size, effective throughput (TB/s) over the column count, `maca_c`
  and `torch.topk` on a shared axis.
* `perf_fp32_maca.png` -- Sampling: float32, `vocab_size = 129280`, `topk = 512`,
  speedup against `torch.topk` over batch size.

The CSV is written by `scripts/perf_snapshot.py`; the timings in it are
`tests/test.py`'s own, one row per (cell, backend).  Only `maca_c` and `torch`
appear on the figures -- `deep_gemm` answers none of these cells (float32 only,
so every bf16 row is `unsupported`, and so are the sampler cells).

The `_maca` suffix is deliberate: `assets/perf_bf16.png` and
`assets/perf_fp32.png` are upstream's own CUDA figures and are displayed as
such, so a generator that wrote those names would silently replace a picture
the README labels with another.  These two are the MACA measurement, from the
recording the run names.

Usage:
    scripts/plot_perf.py                            # newest recording, both figures
    scripts/plot_perf.py perf_data/MetaX_C500/20260919_160228
    scripts/plot_perf.py <dir-or-csv> --figure bf16 --assets-dir /tmp/figs
"""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PERF_ROOT = REPO / "perf_data"
DEFAULT_ASSETS = REPO / "assets"

COLOR_MACA = "#2563eb"
COLOR_TORCH = "#f97316"


def newest_snapshot(perf_root: Path) -> Path:
    """The `deepselect_perf.csv` of the newest `<device>/<stamp>/` recording.

    The stamp is `YYYYmmdd_HHMMSS`, so lexicographic order is chronological and
    the path needs no stat to sort.
    """
    found = sorted(perf_root.glob("*/*/deepselect_perf.csv"))
    if not found:
        raise FileNotFoundError(
            f"no snapshot under {perf_root} -- record one with "
            f"`scripts/perf_snapshot.py` first"
        )
    return found[-1]


def load_rows(source: Path):
    path = source / "deepselect_perf.csv" if source.is_dir() else source
    if not path.is_file():
        raise FileNotFoundError(f"no deepselect_perf.csv at {path}")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"no rows found in {path}")
    required = {
        "family", "n_rows", "n_cols", "top_k", "input_dtype", "backend",
        "time(us)", "throughput(TB/s)",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
    return path, rows


def _us(row: str):
    """A row's `time(us)` as a float, or None when the backend was not timed.

    An empty clock is "not measured", never zero: the two must not be conflated
    into a point at the origin.
    """
    value = (row.get("time(us)") or "").strip()
    return float(value) if value else None


def _select(rows, **want):
    out = []
    for row in rows:
        if all(str(row.get(key, "")) == str(val) for key, val in want.items()):
            out.append(row)
    return out


def device_label(row) -> str:
    """The device's name for a figure title -- `device_name`, else `chip`.

    Two answers to one question live in the CSV: `device_name` is the part's
    own string ("MetaX C500"), `chip` is the directory-safe spelling of it
    ("MetaX_C500", `perf_snapshot.device_dir_name`).  A figure names the board
    a reader can recognize, so it takes the name; `chip` is the fallback for a
    recording old enough to predate the column, not the preferred label.
    """
    return row.get("device_name") or row.get("chip", "")


def plot_bf16(rows, source: Path, output: Path) -> bool:
    cells = _select(rows, family="lightning_indexer", input_dtype="bfloat16",
                    top_k="512")
    if not cells:
        print("bf16: no lightning_indexer/bfloat16/topk=512 cells -- skipped")
        return False

    def series(backend):
        return {
            (int(r["n_rows"]), int(r["n_cols"])): float(r["throughput(TB/s)"])
            for r in cells
            if r["backend"] == backend and r["throughput(TB/s)"]
        }

    maca, torch_ref = series("maca_c"), series("torch")
    batches = sorted({batch for batch, _ in maca})
    vocabs = sorted({vocab for _, vocab in maca})
    chip = device_label(cells[0])

    fig, axes = plt.subplots(
        1, len(batches), figsize=(3.4 * len(batches) + 1.4, 4.4), dpi=160,
        sharey=True, squeeze=False,
    )
    for ax, batch in zip(axes[0], batches):
        m = sorted((v, maca[(batch, v)]) for v in vocabs if (batch, v) in maca)
        t = sorted((v, torch_ref[(batch, v)]) for v in vocabs
                   if (batch, v) in torch_ref)
        if m:
            ax.plot([v for v, _ in m], [y for _, y in m], marker="o",
                    markersize=4, linewidth=1.6, color=COLOR_MACA,
                    label="DeepSelect")
        if t:
            ax.plot([v for v, _ in t], [y for _, y in t], marker="s",
                    markersize=3.5, linewidth=1.4, color=COLOR_TORCH,
                    label="torch.topk")
        ax.set_xscale("log", base=2)
        ax.set_xticks(vocabs)
        ax.set_xticklabels([f"{v / 1000:.3g}" for v in vocabs],
                           rotation=45, ha="right", fontsize=7)
        ax.set_title(f"batch = {batch}", fontsize=9)
        ax.set_xlabel("columns (thousands)", fontsize=8)
        ax.grid(True, which="both", linestyle=":", linewidth=0.6, alpha=0.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.set_ylim(bottom=0)
    axes[0][0].set_ylabel("Throughput (TB/s)", fontsize=9)
    axes[0][0].legend(frameon=True, fontsize=8, loc="upper left")
    title = "Lightning Indexer (bfloat16, topk=512)"
    fig.suptitle(f"{title} -- {chip}" if chip else title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    peak = max(maca.values())
    print(f"bf16: {len(maca)} maca_c cells over {len(batches)} batch sizes, "
          f"peak {peak:.3f} TB/s -> {output}")
    return True


def plot_fp32(rows, source: Path, output: Path) -> bool:
    cells = _select(rows, family="sampler", input_dtype="float32",
                    n_cols="129280", top_k="512")
    if not cells:
        print("fp32: no sampler/float32/vocab=129280/topk=512 cells -- skipped")
        return False

    clocks = {}
    for row in cells:
        us = _us(row)
        if us is not None:
            clocks.setdefault(int(row["n_rows"]), {})[row["backend"]] = us
    points = sorted((b, c["torch"] / c["maca_c"]) for b, c in clocks.items()
                    if "maca_c" in c and "torch" in c)
    if not points:
        print("fp32: no cell where both maca_c and torch were timed -- skipped")
        return False
    chip = device_label(cells[0])

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)
    ax.plot([b for b, _ in points], [s for _, s in points], marker="o",
            markersize=6, linewidth=1.8, color=COLOR_MACA,
            label="DeepSelect vs torch.topk")
    ax.axhline(1.0, color=COLOR_TORCH, linestyle="--", linewidth=1.5,
               label="torch.topk (parity)")
    for b, s in points:
        ax.annotate(f"{s:.2f}x", (b, s), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    ax.set_xlabel("Batches")
    ax.set_ylabel("Speedup vs torch.topk")
    ax.set_title(f"Sampling (float32, vocab_size=129280, topk=512)"
                 + (f" -- {chip}" if chip else ""))
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.55)
    ax.legend(frameon=True)
    ax.margins(x=0.06)
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    print(f"fp32: {len(points)} sampler cells, "
          f"speedup {min(s for _, s in points):.2f}-"
          f"{max(s for _, s in points):.2f}x -> {output}")
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", nargs="?", type=Path,
                        help="a snapshot directory or a deepselect_perf.csv "
                             "(default: the newest recording under perf_data/)")
    parser.add_argument("--figure", choices=("both", "bf16", "fp32"),
                        default="both", help="which figure(s) to render")
    parser.add_argument("--assets-dir", type=Path, default=DEFAULT_ASSETS,
                        help="where the PNGs go (default: assets/)")
    args = parser.parse_args()

    source = args.input or newest_snapshot(PERF_ROOT)
    path, rows = load_rows(source)
    print(f"reading {path}")

    wrote = []
    if args.figure in ("both", "bf16"):
        if plot_bf16(rows, source, args.assets_dir / "perf_bf16_maca.png"):
            wrote.append("perf_bf16_maca.png")
    if args.figure in ("both", "fp32"):
        if plot_fp32(rows, source, args.assets_dir / "perf_fp32_maca.png"):
            wrote.append("perf_fp32_maca.png")
    if not wrote:
        raise SystemExit("nothing rendered -- the recording has no cells for "
                         "the requested figure(s)")


if __name__ == "__main__":
    main()
