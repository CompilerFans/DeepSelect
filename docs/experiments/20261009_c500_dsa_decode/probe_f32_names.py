"""Per-kernel breakdown of the fp32 b4096 cell with the names UNtruncated.

`probe_f32_split.py` prints `name.split('::')[-1][:56]`, which for
`topk_kernel_radix<...>(deep_select_maca::RowParams)` collapses to
"RowParams)" -- hiding whether the instantiation that ran is the PRE=0 (full
row path) or the PRE=1 (preselected merge) one.  This probe prints the whole
name and the launch count, and does not filter anything out.
"""
import sys, torch
TREE = "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect"
sys.path.insert(0, TREE); sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib, test as official, kernelkit as kk, deep_select

B, V, K = 4096, 129280, 512
p = [c for c in official.performance_cases() if (c.batch_size, c.vocab_size, c.topk) == (B, V, K)
     and str(c.dtype) == "torch.float32"][0]
p.seed = (B * 1_000_003 + V * 11 + K) % 2**31
t = lib.generate_testcase(p)
print(f"# end is None: {t.end is None}; enable_end_position={p.enable_end_position}")


def call():
    return deep_select.topk(t.input, p.topk, sorted=p.sorted_value, begin=None, end=t.end,
                            indices_type=p.out_idx_dtype, sorted_index=p.sorted_index, hint=None,
                            output_idx=None, output_idx_offset=t.output_idx_offset,
                            idx_oob_fill_value=p.idx_oob_fill_value,
                            value_oob_fill_value=p.value_oob_fill_value,
                            return_value=p.return_value,
                            abort_when_nan_found=False, backend="maca_c")


call(); torch.cuda.synchronize()
res = kk.bench(call, p.num_runs); N = res.num_tests
for name in sorted(res.get_kernel_names()):
    us = sum(e - s for s, e in res.time_ranges[name]) / N * 1e6
    print(f"   {len(res.time_ranges[name]):>3}x {us:>10.1f} us  {name}")
