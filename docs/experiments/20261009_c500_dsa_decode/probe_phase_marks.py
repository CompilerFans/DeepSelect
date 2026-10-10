#!/usr/bin/env python3
"""Phase split of radix_topk_row_bf16_b, read out of the row kernel itself.

The probe (patched into radix_core.cuh on a throwaway tree) writes five
per-thread phase durations into `output[topk-8+tx]` for `tx < 5` on
`blockIdx.x < 8`.  Rows 0..7 therefore carry their own CTA's phase timeline.

    python3 phase_probe.py <tree> <label>
"""
import glob, hashlib, os, sys
import torch

TREE, LABEL = sys.argv[1], sys.argv[2]
CN = os.environ.get("DS_CN", "1") != "0"
sys.path.insert(0, TREE)
sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib, test as official, kernelkit as kk, deep_select          # noqa: E402

assert deep_select.__file__.startswith(TREE), deep_select.__file__
so = sorted(glob.glob(TREE + "/deep_select/deep_select_maca_xcore*.so"))[-1]
print(f"[{LABEL} check_nan={CN}] {so.rsplit('/',1)[-1]} md5={hashlib.md5(open(so,'rb').read()).hexdigest()[:12]}")

B, V, K = (int(os.environ.get("DS_B", 4096)), int(os.environ.get("DS_V", 524288)),
           int(os.environ.get("DS_K", 512)))
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
        check_nan=CN, backend="maca_c")


r = call()
torch.cuda.synchronize()
idx = r[1] if isinstance(r, (tuple, list)) else r
print(f"return type={type(r).__name__} idx={type(idx).__name__} "
      f"shape={tuple(getattr(idx, 'shape', ()))} dtype={getattr(idx,'dtype',None)}")

names = ["clear", "pass1", "threshold", "compact", "collect+emit"]
tail = idx[:8, p.topk - 8:p.topk].to("cpu")
print(f"cell={B}x{V}x{K}")
hdr = "  row  " + "".join(f"{n:>18}" for n in names) + f"{'sum':>14}"
print(hdr)
for row in range(8):
    v = [int(x) for x in tail[row]]
    ph = v[:5]
    print(f"  {row:>3}  " + "".join(f"{x:>9}" for x in ph)
          + f"  tot={sum(ph):>8}  p1={ph[1]*100.0/sum(ph):>5.1f}% col={ph[3]*100.0/sum(ph):>5.1f}%"
          + f"  col/p1={ph[3]/max(ph[1],1):>5.3f}")

res = kk.bench(call, p.num_runs)
op = sum(sum(e - s for s, e in res.time_ranges[n]) / res.num_tests * 1e6
         for n in res.get_kernel_names()
         if not any(x in n for x in ("elementwise", "Fill", "mcDeviceSync")))
official_us, _ = official.bench_topk(call, p, t, None, None)
print(f"\noperator kernel sum {op:.1f} us | official rule {official_us*1e6:.1f} us")
