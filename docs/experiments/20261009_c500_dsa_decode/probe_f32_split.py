import sys, torch
TREE="/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect"
sys.path.insert(0,TREE); sys.path.insert(0,TREE+"/tests")
torch.set_default_device("cuda")
import lib, test as official, kernelkit as kk, deep_select
B,V,K=4096,129280,512
p=[c for c in official.performance_cases() if (c.batch_size,c.vocab_size,c.topk)==(B,V,K)
   and str(c.dtype)=="torch.float32"][0]
p.seed=(B*1_000_003+V*11+K)%2**31
t=lib.generate_testcase(p)
def call():
    return deep_select.topk(t.input,p.topk,sorted=p.sorted_value,begin=None,end=t.end,
        indices_type=p.out_idx_dtype,sorted_index=p.sorted_index,hint=None,output_idx=None,
        output_idx_offset=t.output_idx_offset,idx_oob_fill_value=p.idx_oob_fill_value,
        value_oob_fill_value=p.value_oob_fill_value,return_value=p.return_value,
        abort_when_nan_found=False,backend="maca_c")
call(); torch.cuda.synchronize()
res=kk.bench(call,p.num_runs); N=res.num_tests; tot=0
for name in sorted(res.get_kernel_names()):
    us=sum(e-s for s,e in res.time_ranges[name])/N*1e6
    if "elementwise" in name or "Fill" in name or "mcDeviceSync" in name: continue
    tot+=us; print(f"   {name.split('::')[-1].split('(')[0][:56]:<56} {len(res.time_ranges[name]):>3}x {us:>9.1f} us")
us,_=official.bench_topk(call,p,t,None,None)
print(f"   {'SUM':<56}      {tot:>9.1f} us | official {us*1e6:>9.1f} us")
inp=B*V*4/1e6; print(f"   input {inp:.1f} MB, two passes = {2*inp:.1f} MB; wall 1650 GB/s -> {2*inp/1650*1e3:.0f} us")
