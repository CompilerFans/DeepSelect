"""Which arm of the fp32 row kernel costs ~1000 us on `b4096 V129280 k512`?

The official cell asks for `sorted_value=True, return_value=True`, so its row
kernel is `topk_kernel_radix<float, long, 512, ...>` with SV/RV set.  Two
hypotheses predict ~1000 us and they need different fixes:

  H (ordered emit)  : the gather of 512 scattered values per row plus the
                      CTA-wide bitonic sort in `emit_ordered`.
  R (rerank)        : a large fraction of rows coming back from dg12 with a
                      `-1` slot and being re-ranked by the full row dataflow.

Variants, same cell and seed, only the three call flags moving -- none of them
is read by the dg12 route predicate, so the route stays fixed:

  (a) sv=True  si=False rv=True    the official cell
  (b) sv=False si=False rv=True    gather, no sort
  (c) sv=False si=False rv=False   no gather, no sort
  (f) sv=False si=True  rv=False   sort by index -- a sort with NO gather
  (g) sv=False si=True  rv=True    that sort plus the value gather

(`sv=True, rv=False` is not a callable arm -- the extension refuses it with
"`return_value` must be enabled when `sorted_value` is True".)

`emit_ordered`'s sort key is the *value* only under SV; under SI it is `src`
(the index), which it already has, so (f) isolates the 45-stage bitonic
network from both gathers and (g) - (f) is the value gather alone.
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
print(f"# official flags: sorted_value={p.sorted_value} sorted_index={p.sorted_index} "
      f"return_value={p.return_value}")


def make(sv, si, rv):
    def call():
        return deep_select.topk(t.input, p.topk, sorted=sv, begin=None, end=t.end,
                                indices_type=p.out_idx_dtype, sorted_index=si, hint=None,
                                output_idx=None, output_idx_offset=t.output_idx_offset,
                                idx_oob_fill_value=p.idx_oob_fill_value,
                                value_oob_fill_value=p.value_oob_fill_value,
                                return_value=rv, abort_when_nan_found=False, backend="maca_c")
    return call


for label, (sv, si, rv) in (("(a) sv=True  rv=True ", (True, False, True)),
                            ("(b) sv=False rv=True ", (False, False, True)),
                            ("(c) sv=False rv=False", (False, False, False)),
                            ("(f) si=True  rv=False", (False, True, False)),
                            ("(g) si=True  rv=True ", (False, True, True))):
    call = make(sv, si, rv)
    call(); torch.cuda.synchronize()
    res = kk.bench(call, p.num_runs); N = res.num_tests
    print(f"--- {label}")
    for name in sorted(res.get_kernel_names()):
        us = sum(e - s for s, e in res.time_ranges[name]) / N * 1e6
        if "Fill" in name or "Memset" in name or "Sync" in name or "Attribute" in name:
            continue
        print(f"    {len(res.time_ranges[name]):>3}x {us:>9.1f} us  {name[:110]}")
