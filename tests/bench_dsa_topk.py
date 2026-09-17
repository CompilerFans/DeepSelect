#!/usr/bin/env python3
"""DSA Indexer TopK -- full-grid benchmark, DeepSelect as the baseline.

One table, three column groups:

  * ``ds``  -- ``deep_select.topk(backend="maca_c")``.  **The baseline.**
  * ``dg``  -- ``deep_gemm.fp32_indexer_topk_selector``, the csrc's own routing
               (``maca_policy=None``).  ``--backends`` can pin the policy
               instead of taking the csrc's choice.
  * ``mcoplib`` -- ``torch.ops.sgl_kernel.fast_topk_transform_fused``.

Each group prints **time (ms) and bandwidth (GB/s)**, then one ratio:

    ratio = t_ds / t_peer

``1.00`` is parity; the ratio is a plain number, direction documented here and
nowhere else in the output.

Comparability rules, which are what make the ratio mean something:

* **Indices only** -- every backend is asked for values off
  (``return_value=False`` / ``return_val=False``), so all of them write ``topk``
  int32 per row.  A backend that refuses a shape reports ``--``, never a time
  from a different workload.
* **One traffic formula** for every row and every backend:
  ``n_rows * (length + topk) * 4`` bytes (read the scores, write int32 indices).
  Bandwidth is that divided by the measured time.
* **One timer** -- ``kernelkit.bench``, the rule ``tests/test.py`` uses: exactly
  one kernel whose name contains "topk" is that kernel's time, several is the
  span over them.
* **``length <= topk``** takes the contract's "select everything" shortcut and
  is marked ``deg``; no ranking happens there.

Usage:
    python3 tests/bench_dsa_topk.py                       # the full grid
    python3 tests/bench_dsa_topk.py --quick               # 6 cells, for iterating
    python3 tests/bench_dsa_topk.py --backends ds,dg,mcoplib
    python3 tests/bench_dsa_topk.py --dg-policy coarse12  # pin deep_gemm's kernel
    python3 tests/bench_dsa_topk.py --bs 6,256 --len 16384,65536 --topk 2048
"""
from __future__ import annotations

import argparse
import os
import sys

import torch

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO not in sys.path:
    sys.path.insert(0, _REPO)

import kernelkit as kk                                            # noqa: E402
import deep_select                                                # noqa: E402

# Row stride.  Rows are 1024-byte aligned because deep_select requires
# `stride(0) * itemsize % 1024 == 0`; 524288 * 4 is a multiple of 1024, so one
# allocation covers every length on the axis.
SEQ = 524288

# deep_gemm's own selector axis -- its policy thresholds move at n_rows
# 32/128/1024 and n_cols 2049/65536, so these straddle all of them.  129280 is
# the Sampler's vocab.
GRID_BS = [6, 32, 128, 256, 1024, 4096]
GRID_LEN = [1024, 2048, 4096, 16384, 65536, 262144, 524288, 129280]
GRID_TOPK = [512, 2048]

QUICK_BS = [6, 256, 4096]
QUICK_LEN = [16384, 65536]
QUICK_TOPK = [2048]


def _load_mcoplib():
    """mcoplib's SGLang fused TopK op, or None when that tree is not built here.

    `mcoplib._C` is a *different* extension (vLLM), so importing `sgl_kernel`
    alone is not enough -- the op itself is probed, not assumed.
    """
    try:
        import mcoplib.sgl_kernel  # noqa: F401
        return torch.ops.sgl_kernel.fast_topk_transform_fused
    except Exception:                                             # noqa: BLE001
        return None


def timed(fn, iters):
    """Kernel time in seconds, by `tests/test.py`'s matching rule."""
    result = kk.bench(fn, iters)
    names = [s for s in result.get_kernel_names() if "topk" in s.lower()]
    if len(names) == 1:
        return result.get_kernel_time(names[0])
    if names:
        return result.get_e2e_time(names)
    return result.get_e2e_time(result.get_kernel_names())


def agrees(our, ref, score, length, topk, pad=-1):
    """Tie-tolerant set comparison: equal sets, or the differing slots carry
    equal values (a tied rank may be broken either way)."""
    a = torch.sort(our[:, :topk], dim=-1).values.cpu()
    b = torch.sort(ref[:, :topk], dim=-1).values.cpu()
    if torch.equal(a, b):
        return True
    for i in range(a.shape[0]):
        sa = set(int(v) for v in a[i].tolist()) - {pad}
        sb = set(int(v) for v in b[i].tolist()) - {pad}
        more, less = sa - sb, sb - sa
        if more or less:
            mv = sorted(score[i, idx].item() for idx in more if 0 <= idx < length)
            lv = sorted(score[i, idx].item() for idx in less if 0 <= idx < length)
            if mv != lv:
                return False
    return True


class Cell:
    """One (bs, len, topk) point: the inputs, the reference, and the runners."""

    def __init__(self, bs, length, topk, dev, mcp_op, dg_policy):
        self.bs, self.length, self.topk = bs, length, topk
        self.score = torch.randn(bs, SEQ, dtype=torch.float32, device=dev)
        self.lengths = torch.full((bs,), length, dtype=torch.int32, device=dev)
        self.traffic = bs * (length + topk) * 4
        self.ref = torch.topk(self.score[:, :length], topk, dim=-1,
                              sorted=False).indices.to(torch.int32)
        self._mcp_op = mcp_op
        self._dg_policy = dg_policy
        self._dg_mod = None
        self._xform = None

    def _dg(self):
        if self._dg_mod is None:
            import deep_gemm
            self._dg_mod = deep_gemm
        return self._dg_mod

    def _page_args(self):
        """Identity page table, so the fused transform is the identity on the
        selected columns and its answer is directly comparable to a selector's."""
        if self._xform is None:
            pt = torch.arange(SEQ, dtype=torch.int32, device=self.score.device)
            self._xform = (pt.unsqueeze(0).expand(self.bs, -1).contiguous(),
                           torch.arange(self.bs + 1, dtype=torch.int32,
                                        device=self.score.device),
                           torch.empty((self.bs, self.topk), dtype=torch.int32,
                                       device=self.score.device))
        return self._xform

    # -- the three runners; each returns the indices tensor it produced ------
    def run_ds(self):
        return deep_select.topk(self.score[:, :self.length], self.topk,
                                return_value=False, backend="maca_c")[1]

    def run_dg(self):
        kwargs = {} if self._dg_policy is None else {"maca_policy": self._dg_policy}
        return self._dg() .fp32_indexer_topk_selector(
            self.score[:, :self.length].contiguous(), self.lengths, self.topk,
            return_val=False, backend="maca_c", **kwargs)["indices"]

    def run_mcoplib(self):
        pt, cu, dst = self._page_args()
        self._mcp_op.default(self.score, self.lengths, dst, pt, cu, None)
        return dst

    def runner(self, name):
        return {"ds": self.run_ds, "dg": self.run_dg,
                "mcoplib": self.run_mcoplib}[name]

    def free(self):
        self._xform = None
        del self.score, self.ref
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--iters", type=int, default=30,
                    help="kernelkit bench rounds per measurement (default 30)")
    ap.add_argument("--dev", type=int, default=0)
    ap.add_argument("--bs", default=None, help="comma-separated batch axis")
    ap.add_argument("--len", default=None, help="comma-separated length axis")
    ap.add_argument("--topk", default=None, help="comma-separated topk axis")
    ap.add_argument("--backends", default="ds,dg,mcoplib",
                    help="comma-separated subset of ds,dg,mcoplib")
    ap.add_argument("--dg-policy", default=None,
                    choices=[None, "auto", "single", "chunks", "coarse12"],
                    help="pin deep_gemm's kernel instead of taking its own routing")
    ap.add_argument("--quick", action="store_true", help="a 6-cell grid")
    args = ap.parse_args()

    torch.manual_seed(42)
    dev = f"cuda:{args.dev}"

    bs_axis = [int(x) for x in args.bs.split(",")] if args.bs else (
        QUICK_BS if args.quick else GRID_BS)
    len_axis = [int(x) for x in args.len.split(",")] if args.len else (
        QUICK_LEN if args.quick else GRID_LEN)
    topk_axis = [int(x) for x in args.topk.split(",")] if args.topk else (
        QUICK_TOPK if args.quick else GRID_TOPK)

    mcp_op = _load_mcoplib()
    have_dg = deep_select.deep_gemm_available()

    wanted, dropped = [], {}
    for b in (x.strip() for x in args.backends.split(",") if x.strip()):
        if b == "mcoplib" and mcp_op is None:
            dropped[b] = "mcoplib op not built/importable here"
        elif b == "dg" and not have_dg:
            dropped[b] = "deep_gemm unavailable"
        elif b not in ("ds", "dg", "mcoplib"):
            dropped[b] = "unknown backend"
        else:
            wanted.append(b)
    if "ds" not in wanted:
        wanted.insert(0, "ds")            # the baseline is always measured
    peers = [b for b in wanted if b != "ds"]
    if not peers:
        print("no peer to compare against", file=sys.stderr)
        return 1

    dg_label = "dg" if args.dg_policy in (None, "auto") else f"dg:{args.dg_policy}"
    print(f"# device    : {torch.cuda.get_device_name(args.dev)}")
    print(f"# baseline  : ds = deep_select maca_c")
    print(f"# ratio     : x = t_ds / t_peer   (1.00 = parity)")
    for b, why in dropped.items():
        print(f"# dropped   : {b} -- {why}")
    print(f"# traffic   : n_rows*(length+topk)*4 B -- one formula, every backend, "
          f"indices only")
    print(f"# timer     : kernelkit.bench, 'topk'-matching kernel time "
          f"({args.iters} rounds)")
    print(f"# grid      : bs={bs_axis} len={len_axis} k={topk_axis}"
          f"   {'deg = length<=topk, the contract shortcut' if any(l <= k for k in topk_axis for l in len_axis) else ''}\n")

    # Header: a (ms, GB/s, x) triple per peer, (ms, GB/s) for the baseline.
    head = f"{'bs':>6} {'len':>7} {'k':>5} | {'ds ms':>9} {'ds GB/s':>8}"
    for b in peers:
        lbl = dg_label if b == "dg" else b
        head += f" | {lbl + ' ms':>12} {lbl + ' GB/s':>11} {lbl + ' x':>8}"
    print(head)
    print("-" * len(head))

    for topk in topk_axis:
        for bs in bs_axis:
            for length in len_axis:
                if length < topk:
                    continue
                cell = Cell(bs, length, topk, dev, mcp_op, args.dg_policy)
                res = {}
                for b in ["ds"] + peers:
                    fn = cell.runner(b)
                    try:
                        out = fn()
                        res[b] = (timed(fn, args.iters), agrees(
                            out, cell.ref, cell.score, length, topk))
                    except Exception:                          # noqa: BLE001
                        res[b] = (float("nan"), None)

                def gbs(t):
                    return cell.traffic / t / 1e9 if t == t else float("nan")

                t_ds = res["ds"][0]
                line = (f"{bs:>6} {length:>7} {topk:>5} | "
                        f"{t_ds * 1e3:>9.4f} {gbs(t_ds):>8.1f}")
                for b in peers:
                    t, ok = res[b]
                    if t != t:                     # refused this shape
                        line += f" | {'--':>12} {'--':>11} {'--':>8}"
                        continue
                    x = t_ds / t if t > 0 else float("nan")
                    line += f" | {t * 1e3:>12.4f} {gbs(t):>11.1f} {x:>8.3f}"
                if length <= topk:
                    line += "  deg"
                bad = [b for b in ["ds"] + peers if res[b][1] is False]
                if bad:
                    line += "  !wrong:" + ",".join(bad)
                print(line)
                cell.free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
