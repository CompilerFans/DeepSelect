#!/usr/bin/env python3
"""What the abort disposition costs a call that does NOT abort.

The abort path is the one shape the perf grid never takes (both suites pass
`abort_when_nan_found=False`), so its price is invisible in every number the
grid reports.  It is not zero: an `abort_on_nan` call gets the follower launched
and a clean input still pays it.  It is meant to be *small* -- the flag table is
written by the row kernel rather than zeroed on the stream (see
`RowParams::nan_abort_flags` in `maca_topk.cu`), which is what took this delta
from ~14 us to the follower alone.

Times the headline cell twice on one arm -- `abort_when_nan_found=False` and
`True` -- and prints every operator kernel by name (the follower is
`nan_abort_kernel`, which the "topk" substring filter would drop, so it is
printed with the rest rather than summed past).

    measure_abort_cost.py <tree> [label] [B V K]

The cell defaults to the headline `b4096 V524288 k512` bf16; the `B V K`
override is for the shapes where the constant part of the price is what matters
-- a launch is a launch whatever the row count, so the small-batch calls are
where it shows up as a percentage.
"""
import glob
import hashlib
import sys

import torch

TREE = sys.argv[1]
LABEL = sys.argv[2] if len(sys.argv) > 2 else TREE.rsplit("/", 1)[-1]
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
if len(sys.argv) > 5:
    B, V, K = (int(x) for x in sys.argv[3:6])
p = [c for c in official.performance_cases()
     if (c.batch_size, c.vocab_size, c.topk) == (B, V, K)
     and str(c.dtype) == "torch.bfloat16"]
if p:
    p = p[0]
else:                                         # off-grid shapes, same axes
    p = official.TestParam(B, V, K, False, False, False, torch.bfloat16,
                           torch.int32, num_runs=10)
p.seed = (B * 1_000_003 + V * 11 + K) % 2 ** 31
t = lib.generate_testcase(p)
print(f"  cell b={B} v={V} k={K} bf16")


def make_call(abort):
    def call():
        return deep_select.topk(
            t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
            indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
            output_idx=None, output_idx_offset=t.output_idx_offset,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value, abort_when_nan_found=abort,
            backend="maca_c")
    return call


def e2e_us(call, iters=30):
    """Wall time of the call itself, events around it -- the L2 flush the
    harness pays per iteration is not the operator's, so it stays outside."""
    for _ in range(5):
        call()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True); e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        call()
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3          # ms -> us


for abort in (False, True):
    call = make_call(abort)
    call()
    torch.cuda.synchronize()
    print(f"  abort_when_nan_found={abort}   e2e {e2e_us(call):9.1f} us (events, no L2 flush)")
    res = kk.bench(call, p.num_runs)
    N = res.num_tests
    print(f"  abort_when_nan_found={abort}")
    total = 0.0
    for name in sorted(res.get_kernel_names()):
        us = sum(e - s for s, e in res.time_ranges[name]) / N * 1e6
        if "elementwise" in name or "Fill" in name or "mcDeviceSync" in name:
            continue
        tag = ""
        if "nan_abort" in name:
            tag = "   <== the follower"
        elif "topk" in name.lower():
            tag = "   <== counted by the official rule"
        print(f"    {name[:60]:<60} {us:9.1f} us{tag}")
        total += us
    print(f"    {'--- SUM of operator kernels':<60} {total:9.1f} us")
