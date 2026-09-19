"""Check the chunks arm with the suite's **own** predicate.

`tests/test.py:check_result` is the official correctness assertion, split out
of `run_testcase` so a caller can check another backend against it rather than
a copy of it.  This is that caller: it drives the same shapes the arm serves
(`b <= 2`, fp32, `V >= 2048`) plus the two paths that must *decline* to the row
path, and asks the official predicate rather than a hand-written comparison --
a hand-written one is how a harness artifact gets mistaken for a kernel bug
(twice, in this session: a `[:, :V]` slice that made a strided view, and a
`-1`-padded reference whose sort put the padding first).
"""
# **This file measures the checkout, not the installed wheel.**  It puts the
# *repository* on `sys.path` (below), which is the only way a `deep_select`
# import can resolve to a tree that also carries `kernelkit` and `lib` -- the
# two modules this case is built on.  That is deliberate, and it is also the
# trap: a bare `python tests/cases/foo.py` run from the repository root gets
# `.` on `sys.path` from the interpreter, and with the package importable from
# site-packages the name can resolve there instead.  Run these with
# `PYTHONPATH=.` and read the `artifact=` line every one of them prints, or
# `run_bench.sh`, which resolves the extension the same way and records its
# md5 in the run header.
import os, sys, torch
torch.set_default_device("cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernelkit as kk, deep_select, lib, test as suite
from lib import (TestParam, NormalFloatDistribution, UniformUIntDistribution,
                 UintDistributionWithHotspotAndSpecifiedPivot, get_fp_config)

def case(tag, b, V, k, distrib=None, end=False, si=True, sv=False, rv=False,
         idx_dtype=torch.int32, seed=3):
    p = TestParam(batch_size=b, vocab_size=V, topk=k, sorted_value=sv,
                  sorted_index=si, return_value=rv, dtype=torch.float32,
                  out_idx_dtype=idx_dtype, enable_end_position=end,
                  input_distrib=distrib or NormalFloatDistribution(), seed=seed,
                  idx_oob_fill_value=-2000000 + V)
    kk.set_random_seed(seed)
    t = lib.generate_testcase(p)
    t.input = t.input.to("cuda")
    if t.end is not None: t.end = t.end.to("cuda")
    val, idx = deep_select.topk(t.input, k, sorted=sv, begin=None, end=t.end,
                                indices_type=idx_dtype, sorted_index=si, hint=None,
                                output_idx=None, output_idx_offset=None,
                                idx_oob_fill_value=p.idx_oob_fill_value,
                                value_oob_fill_value=p.value_oob_fill_value,
                                return_value=rv, abort_when_nan_found=False,
                                backend="maca_c")
    torch.cuda.synchronize()
    suite.check_call_contract(p, val, idx)
    ok = suite.check_result(p, t,
                            val.clone() if val is not None else None, idx.clone())
    print(f"  {tag:<38} b={b:<3} V={V:<7} k={k:<5} si={int(si)} rv={int(rv)} "
          f"end={int(end)} -> {'PASS' if ok else 'FAIL'}")
    return ok

allok = True
N = NormalFloatDistribution
# ── the band the arm serves ────────────────────────────────────────────────
for b in (1, 2):
    for V, k in ((2048, 512), (4096, 1024), (16384, 2048), (66551, 512),
                 (66551, 2048), (107520, 2048), (131072, 2048), (225467, 231)):
        allok &= case("band", b, V, k)
        allok &= case("band+end", b, V, k, end=True)
        allok &= case("band+si=False", b, V, k, si=False)
        allok &= case("band+rv", b, V, k, si=False, rv=True, idx_dtype=torch.int64)
# ── the declines ───────────────────────────────────────────────────────────
allok &= case("decline length==topk", 1, 2048, 2048)
allok &= case("decline length<topk", 1, 1024, 2048)
allok &= case("decline length<topk", 2, 512, 2048)
allok &= case("decline end<=topk", 2, 66551, 2048, end=True)
for V in (66551, 131072, 225467):
    allok &= case(f"decline wide bin V={V}", 1, V, 2048,
                  UintDistributionWithHotspotAndSpecifiedPivot(
                      None, 2048, [(0x3f800000, V // 2)], True))
allok &= case("decline wide bin pivot", 2, 66551, 2048,
              UintDistributionWithHotspotAndSpecifiedPivot(
                  0x3f800000, 2048, [(0x3f800000, 40000)], True))
# ── the NaN contract, on the band this arm actually answers ────────────────
# The arm answers a row *without ranking it* and hands the answer to the
# contract half as `preselected`, and that half **reads** a per-row flag
# instead of scanning (`maca_topk.cu:463`).  So the arm has to raise it, and
# for a while it did not: `dg_chunks.cuh` took `nan_flags` and did
# `(void)nan_flags;`, which served a NaN row a normal top-k with
# `abort_when_nan_found` -- the default -- inert.  These cases are what would
# have caught it.  They are here rather than in `tests/test.py`'s own table
# because the *default* routing reaches this arm only at `b <= 2` with
# `V >= 2048`, and the table's `b` values are drawn, not pinned.
#
# Both NaN signs, because the guard is `!= 0` over a flag raised by
# `is_nan_value`, and a sign-asymmetric predicate would pass one and not the
# other.  `V=2048, k=2048` is deliberately absent: there the window is no
# longer than `topk`, the shortcut arm answers before the NaN block runs, and
# the suite's own predicate exempts that row -- which is a different assertion
# and belongs to the shortcut, not here.
_NaN = UintDistributionWithHotspotAndSpecifiedPivot
for _sign in (get_fp_config(torch.float32).positive_nan,
              get_fp_config(torch.float32).negative_nan):
    for V, k in ((66551, 2048), (131072, 2048), (262144, 2048)):
        allok &= case(f"nan hotspot b=1 V={V}", 1, V, k,
                      _NaN(None, k, [(_sign, 8)], True))
        allok &= case(f"nan hotspot b=2 V={V}", 2, V, k,
                      _NaN(None, k, [(_sign, 8)], True))
# ── controls just outside the band ─────────────────────────────────────────
allok &= case("control b=4", 4, 66551, 2048)
allok &= case("control denormals", 1, 66551, 2048, distrib=UniformUIntDistribution(0, 0x1000))
print("ALL PASS" if allok else "SOME FAILED")
