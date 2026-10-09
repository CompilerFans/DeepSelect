#!/usr/bin/env python3
"""Time one NaN-flag arm on the headline cell, under both `check_nan` settings.

    python3 measure_nan.py <tree> <label>

Same 15 arguments the official call site passes, `check_nan` the only extra
keyword -- the same shape the earlier sweep used, so the two are comparable.
"""
import glob
import hashlib
import sys

import torch

TREE = sys.argv[1]
LABEL = sys.argv[2]
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")

import lib                                    # noqa: E402
import test as official                       # noqa: E402
import deep_select                            # noqa: E402

assert deep_select.__file__.startswith(TREE), deep_select.__file__
so = sorted(glob.glob(TREE + "/deep_select/deep_select_maca_xcore*.so"))[-1]
print(f"[{LABEL}] so={so.rsplit('/', 1)[-1]} "
      f"md5={hashlib.md5(open(so, 'rb').read()).hexdigest()[:12]}")

B, V, K = 4096, 524288, 512
p = [c for c in official.performance_cases()
     if (c.batch_size, c.vocab_size, c.topk) == (B, V, K)
     and str(c.dtype) == "torch.bfloat16"][0]
p.seed = (B * 1_000_003 + V * 11 + K) % 2 ** 31
t = lib.generate_testcase(p)


def call(flag):
    return deep_select.topk(
        t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
        indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
        output_idx=None, output_idx_offset=t.output_idx_offset,
        idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,
        return_value=p.return_value, abort_when_nan_found=False,
        backend="maca_c", check_nan=flag)


us, idx = {}, {}
for flag in (True, False):
    run = lambda flag=flag: call(flag)        # noqa: E731
    r = run()
    torch.cuda.synchronize()
    us[flag], size = official.bench_topk(run, p, t, None, None)
    ii = r[1] if isinstance(r, tuple) else r
    idx[flag] = ii.clone()
    print(f"    check_nan={str(flag):<5} {us[flag] * 1e6:>10.1f} us   "
          f"{size / us[flag] / 1e9:>7.1f} GB/s (logical, 1 pass)")

a, f = us[True], us[False]
print(f"    False/True = {f / a:.4f}   delta = {(a - f) * 1e6:+9.1f} us")

iT, iF = idx[True], idx[False]
sT, _ = iT.sort(1)
sF, _ = iF.sort(1)
rows_eq = int((sT == sF).all(1).sum())
uniq = (iT.shape[1] == torch.tensor([len(set(r.tolist())) for r in iT[:64]])
        ).all().item()
print(f"    elementwise identical = {bool((iT == iF).all())}   "
      f"sorted-set identical = {rows_eq}/{iT.shape[0]} rows   "
      f"in-range = {bool(iT.min() >= 0 and iT.max() < V)}   "
      f"unique-per-row(first 64) = {bool(uniq)}")
