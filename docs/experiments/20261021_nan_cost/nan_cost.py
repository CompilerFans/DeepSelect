# In-situ marginal cost of the folded NaN scan on the row route, official shape.
# check_nan is a runtime bool in the shared template (contract.cuh:626), so one
# artifact serves both arms; alternate them to keep the clock on both.
import torch, time, statistics as st, deep_select

torch.set_default_device("cuda")
torch.manual_seed(7)

def run(x, k, cn, n=15):
    ts = []
    for _ in range(n):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        deep_select.topk(x, k, indices_type=torch.int32, return_value=False,
                         abort_when_nan_found=False, backend="maca_c",
                         check_nan=cn)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return st.median(ts)

for b, v, k in [(4096, 1048576, 512), (4096, 1048576, 1024),
                (4096, 262144, 512), (512, 1048576, 512),
                (4096, 131072, 512)]:
    x = torch.randn(b, v, dtype=torch.float32).to(torch.bfloat16)
    a = []; c = []
    for _ in range(3):
        a.append(run(x, k, True)); c.append(run(x, k, False))
    on, off = st.median(a), st.median(c)
    print(f"b{b:>4} v{v:>7} k{k:<5} nan-on {on:9.3f} ms  nan-off {off:9.3f} ms  "
          f"delta {100*(on-off)/off:6.2f}% of off")
    del x
    torch.cuda.empty_cache()
