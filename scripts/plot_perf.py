#!/usr/bin/env python3
"""Render the README's performance figures from a `perf_snapshot.py` recording.

Two figures, both drawn from one `deepselect_perf.csv` so the picture and the
record it came from cannot disagree.  Both are grouped bar charts of effective
bandwidth, `maca_c` (DeepSelect) beside `torch.topk` in each group, laid out as
`assets/perf_bf16_cuda.png` and `assets/perf_fp32_cuda.png` are:

* `perf_bf16_maca_<device>.png` -- Lightning Indexer: bf16, `topk = 512`, one
  panel per batch size (6 / 512 / 4096), grouped bars over vocab size (16K /
  64K / 128K / 256K / 512K / 1M), one shared y-axis.
* `perf_fp32_maca_<device>.png` -- Sampling: fp32, `vocab_size = 129280`,
  `topk = 512`, grouped bars over batch size (6 / 256 / 512 / 768 / 4096).

`<device>` is the recording's own board (`device_dir_name`'s spelling, `MetaX
C500` -> `MetaX_C500`), and it is in the *filename* for the same reason it is in
the title and in `perf_data/`'s directory: `perf_data/` holds one recording per
device, and a name that stops at the platform gives two boards one output path,
so rendering the second silently replaces the first's figure.  The title has
always named the board; the file did not.  Named figures now sit side by side --
`perf_bf16_maca_MetaX_C500.png`, `perf_bf16_maca_MetaX_C600-U.png` -- and the
README picks the one it displays.

The upstream figures are in TB/s over a fixed 0-7 axis; these are in GB/s over a
range fitted to the data, because these parts' read wall is a fraction of an
H200's and a shared 0-7 TB/s axis would leave every bar a sliver.  What is
matched is the rest of the style: the two series' colours, the boxed legend, the
per-panel `batch = N` titles, the dotted horizontal grid, the `16K`-style tick
labels and the `-- {device}` suffix each title carries.

The CSV is written by `scripts/perf_snapshot.py`; the timings in it are
`tests/test.py`'s own, one row per (cell, backend).  Only `maca_c` and `torch`
appear on the figures -- `deep_gemm` answers none of these cells (float32 only,
so every bf16 row is `unsupported`, and so are the sampler cells).

The `_maca` suffix is upstream's own convention, which names a figure for the
part it was measured on (`perf_bf16_cuda.png`, `perf_bf16_ascend.png`); these
are the MACA member of that set, and the README shows them.  A generator that
wrote the bare `perf_bf16.png` would land outside the convention and on a name
that has meant the CUDA figure.

Usage:
    scripts/plot_perf.py                            # newest recording, both figures
    scripts/plot_perf.py perf_data/MetaX_C500/20260919_160228
    scripts/plot_perf.py <dir-or-csv> --figure bf16 --assets-dir /tmp/figs
"""

import argparse
import csv
import math
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
PERF_ROOT = REPO / "perf_data"
DEFAULT_ASSETS = REPO / "assets"

# Upstream's own series colours, sampled from `assets/perf_bf16_cuda.png`.
COLOR_MACA = "#66ccfe"
COLOR_TORCH = "#ed0000"

# A recording directory's name, `YYYYmmdd_HHMMSS`.
STAMP = re.compile(r"\d{8}_\d{6}")

# The upstream figure's grid: three batch sizes, six vocab sizes, `topk = 512`.
BF16_BATCHES = (6, 512, 4096)
BF16_VOCABS = (16384, 65536, 131072, 262144, 524288, 1048576)

BAR_WIDTH = 0.38
# How much air the y-axis leaves above the tallest bar drawn.
HEADROOM = 1.05
# Upstream tags each figure with the platform it was measured on -- `(CUDA)`,
# `(Ascend)`.  Ours carries the board too, since `perf_data/` holds one recording
# per device and "MACA" alone does not say which part.
PLATFORM = "MACA"


def newest_snapshot(perf_root: Path) -> Path:
    """The `deepselect_perf.csv` of the newest `<device>/<stamp>/` recording.

    The stamp is `YYYYmmdd_HHMMSS`, so lexicographic order is chronological and
    the path needs no stat to sort.  Two things are excluded deliberately:

    * `perf_data/<device>/baseline` is a *symlink* to a recording, and globbing
      `*/*` would let that name sort last under any device and pick a recording
      from the wrong board with no sign that it did.
    * The device directory's name.  Sorting the whole path would order the
      *boards* (`MetaX_C500` before `MetaX_C600-U`) and only then the stamps, so
      a week-old recording on the later-named board would beat this morning's on
      the earlier-named one.  The stamp is the timestamp; it is what is compared.
    """
    stamps = [p for p in perf_root.glob("*/*/deepselect_perf.csv")
              if STAMP.fullmatch(p.parent.name)]
    if not stamps:
        raise FileNotFoundError(
            f"no snapshot under {perf_root} -- record one with "
            f"`scripts/perf_snapshot.py` first"
        )
    return max(stamps, key=lambda p: (p.parent.name, str(p)))


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
        "bandwidth(GB/s)",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")
    return path, rows


def _select(rows, **want):
    out = []
    for row in rows:
        if all(str(row.get(key, "")) == str(val) for key, val in want.items()):
            out.append(row)
    return out


def _bandwidth(row):
    """A row's `bandwidth(GB/s)` as a float, or None when it was not measured.

    An empty cell is "not timed", never zero: the two must not be conflated
    into a bar of zero height.
    """
    value = (row.get("bandwidth(GB/s)") or "").strip()
    return float(value) if value else None


def device_label(row) -> str:
    """The device's name for a figure title -- `device_name`, else `chip`.

    Two answers to one question live in the CSV: `device_name` is the part's
    own string ("MetaX C500"), `chip` is the directory-safe spelling of it
    ("MetaX_C500", `perf_snapshot.device_dir_name`).  A figure names the board
    a reader can recognize, so it takes the name; `chip` is the fallback for a
    recording old enough to predate the column, not the preferred label.
    """
    return row.get("device_name") or row.get("chip", "")


def device_slug(row) -> str:
    """The same board as a filename component: `MetaX C500` -> `MetaX_C500`.

    `perf_snapshot.device_name`'s own rule -- spaces to underscores -- so a
    figure, the recording it came from and `perf_data/`'s directory all spell
    the board the same way.  Empty for a recording that names no device; the
    caller then falls back to the bare platform name.
    """
    return device_label(row).strip().replace(" ", "_")


def _vocab_label(n: int) -> str:
    """A column count as a short tick label: 16384 -> `16K`, 1048576 -> `1M`."""
    for div, suffix in ((1 << 20, "M"), (1 << 10, "K")):
        if n >= div and n % div == 0:
            return f"{n // div}{suffix}"
    return str(n)


def _bandwidth_by(cells, backend):
    """`(n_rows, n_cols) -> GB/s` for one backend's timed cells."""
    out = {}
    for row in cells:
        if row["backend"] != backend:
            continue
        value = _bandwidth(row)
        if value is not None:
            out[(int(row["n_rows"]), int(row["n_cols"]))] = value
    return out


def _axis_top(panels, headroom=HEADROOM):
    """A round y-limit that clears the tallest bar of every panel.

    It has to be computed from all panels at once and set explicitly: the
    panels share a y-axis, and `set_ylim` resolves that shared axis on the
    spot, so setting a limit from inside the panel loop would freeze the range
    at whatever the first panel happened to contain -- clipping every later
    panel's taller bars at the top of the axes.
    """
    values = [v for maca, torch_ref in panels
              for v in list(maca) + list(torch_ref) if v is not None]
    if not values:
        return 1.0
    target = max(values) * headroom
    step = 10 ** math.floor(math.log10(target))
    for mult in (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if step * mult >= target:
            return step * mult
    return step * 10


def _grouped_bars(ax, positions, maca, torch_ref):
    """One `maca_c`/`torch` bar pair per position, over matching values lists."""
    ax.bar([p - BAR_WIDTH / 2 for p, v in zip(positions, maca) if v is not None],
           [v for v in maca if v is not None], BAR_WIDTH,
           color=COLOR_MACA, label="DeepSelect")
    ax.bar([p + BAR_WIDTH / 2 for p, v in zip(positions, torch_ref) if v is not None],
           [v for v in torch_ref if v is not None], BAR_WIDTH,
           color=COLOR_TORCH, label="torch.topk")


def _finish_axes(ax, ticks, labels, title, xlabel, top, ylabel=None):
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.grid(True, axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=8)
    ax.set_ylim(0, top)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=9)


def plot_bf16(rows, output: Path) -> bool:
    cells = _select(rows, family="lightning_indexer", input_dtype="bfloat16",
                    top_k="512")
    if not cells:
        print("bf16: no lightning_indexer/bfloat16/topk=512 cells -- skipped")
        return False

    maca, torch_ref = _bandwidth_by(cells, "maca_c"), _bandwidth_by(cells, "torch")
    batches = [b for b in BF16_BATCHES if any((b, v) in maca for v in BF16_VOCABS)]
    vocabs = [v for v in BF16_VOCABS if any((b, v) in maca for b in batches)]
    if not batches or not vocabs:
        print("bf16: the requested grid is not in this recording -- skipped")
        return False
    chip = device_label(cells[0])

    panels = [([maca.get((b, v)) for v in vocabs],
               [torch_ref.get((b, v)) for v in vocabs]) for b in batches]
    top = _axis_top(panels)

    fig, axes = plt.subplots(1, len(batches), figsize=(14.0, 4.15), dpi=160,
                             sharey=True, squeeze=False)
    x = list(range(len(vocabs)))
    for ax, batch, (maca_bars, torch_bars) in zip(axes[0], batches, panels):
        _grouped_bars(ax, x, maca_bars, torch_bars)
        _finish_axes(ax, x, [_vocab_label(v) for v in vocabs],
                     f"batch = {batch}", "vocab_size", top)
    axes[0][0].set_ylabel("Effective bandwidth (GB/s)", fontsize=9)
    axes[0][0].legend(frameon=True, fontsize=8, loc="upper left")
    title = "bf16, topk = 512: DeepSelect vs torch.topk"
    fig.suptitle(f"{title} ({PLATFORM}, {chip})" if chip
                 else f"{title} ({PLATFORM})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    peak = max(v for maca_bars, torch_bars in panels
               for v in maca_bars + torch_bars if v is not None)
    print(f"bf16: {len(batches)} batch sizes x {len(vocabs)} vocab sizes, "
          f"peak {peak:.1f} GB/s on a 0-{top:g} axis -> {output}")
    return True


def plot_fp32(rows, output: Path) -> bool:
    cells = _select(rows, family="sampler", input_dtype="float32",
                    n_cols="129280", top_k="512")
    if not cells:
        print("fp32: no sampler/float32/vocab=129280/topk=512 cells -- skipped")
        return False

    maca, torch_ref = _bandwidth_by(cells, "maca_c"), _bandwidth_by(cells, "torch")
    batches = sorted({b for b, _ in maca} & {b for b, _ in torch_ref})
    if not batches:
        print("fp32: no cell where both maca_c and torch were timed -- skipped")
        return False
    chip = device_label(cells[0])

    maca_bars = [maca.get((b, 129280)) for b in batches]
    torch_bars = [torch_ref.get((b, 129280)) for b in batches]
    top = _axis_top([(maca_bars, torch_bars)])

    fig, ax = plt.subplots(figsize=(6.75, 4.3), dpi=160)
    _grouped_bars(ax, list(range(len(batches))), maca_bars, torch_bars)
    title = "fp32, vocab_size = 129280, topk = 512: DeepSelect vs torch.topk"
    _finish_axes(ax, list(range(len(batches))), [str(b) for b in batches],
                 f"{title} ({PLATFORM}, {chip})" if chip
                 else f"{title} ({PLATFORM})", "batch_size", top,
                 ylabel="Effective bandwidth (GB/s)")
    ax.legend(frameon=True, fontsize=8, loc="upper left")
    fig.tight_layout()

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight")
    plt.close(fig)
    peak = max(v for v in maca_bars + torch_bars if v is not None)
    print(f"fp32: {len(batches)} batch sizes, peak {peak:.1f} GB/s on a "
          f"0-{top:g} axis -> {output}")
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

    # One recording is one board, so the device is a property of the file and
    # not of the figure: it goes in the name once, here, rather than in each
    # `plot_*` -- which take the path they write and stay unaware a device
    # exists.  A recording that names no device keeps the bare platform name.
    slug = device_slug(rows[0])
    stem = f"_maca_{slug}" if slug else "_maca"

    wrote = []
    if args.figure in ("both", "bf16"):
        name = f"perf_bf16{stem}.png"
        if plot_bf16(rows, args.assets_dir / name):
            wrote.append(name)
    if args.figure in ("both", "fp32"):
        name = f"perf_fp32{stem}.png"
        if plot_fp32(rows, args.assets_dir / name):
            wrote.append(name)
    if not wrote:
        raise SystemExit("nothing rendered -- the recording has no cells for "
                         "the requested figure(s)")


if __name__ == "__main__":
    main()
