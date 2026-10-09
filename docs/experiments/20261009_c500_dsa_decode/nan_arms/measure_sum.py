#!/usr/bin/env python3
"""Per-kernel breakdown of one arm on the headline cell.

`official.bench_topk`'s rule keeps only names containing "topk", and
`nan_scan_kernel` does not contain it -- so a separate-scan arm would be
credited with the row kernel alone.  This prints every operator kernel, their
sum, and the official number side by side.

    python3 measure_sum.py <tree> <label>
"""
import glob
import hashlib
import sys

import torch

TREE, LABEL = sys.argv[1], sys.argv[2]
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")

import lib                                    # noqa: E402
import test as official                       # noqa: E402
import kernelkit as kk                        # noqa: E402
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


def call():
    return deep_select.topk(
        t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
        indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
        output_idx=None, output_idx_offset=t.output_idx_offset,
        idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,
        return_value=p.return_value, abort_when_nan_found=False,
        backend="maca_c")


call()
torch.cuda.synchronize()
res = kk.bench(call, p.num_runs)
N = res.num_tests

op_total = 0.0
op_names = []
for name in sorted(res.get_kernel_names()):
    us = sum(e - s for s, e in res.time_ranges[name]) / N * 1e6
    if "elementwise" in name or "Fill" in name or "mcDeviceSync" in name:
        continue                              # harness fills, not the operator
    op_names.append(name)
    op_total += us
    bare = name.split("::")[-1].split("(")[0]
    print(f"    {bare[:64]:<64} {len(res.time_ranges[name]):>3} x  {us:>9.1f} us")

official_us, _ = official.bench_topk(call, p, t, None, None)
sum_label = "--- SUM of operator kernels"
off_label = '--- official rule (names containing "topk")'
print(f"    {sum_label:<64}      {op_total:>9.1f} us")
print(f"    {off_label:<64}      {official_us * 1e6:>9.1f} us")
