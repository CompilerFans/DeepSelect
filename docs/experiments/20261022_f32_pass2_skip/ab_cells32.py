# A/B child: time the fp32 cells that reach the coarse12 route against one
# artifact.  argv[1] = repo root to import.  Prints the md5 of the .so that
# actually answered, so an arm cannot silently measure the other's binary.
import os, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import hashlib
import torch, kernelkit as kk, deep_select, lib
from deep_select._binding import extension_path

torch.set_default_device("cuda")
print("lib", hashlib.md5(open(extension_path("deep_select_maca"), "rb").read()).hexdigest()[:8])

# (b, V, k, sorted_value, sorted_index, return_value, idx dtype) -- the
# official cells whose row count dominates the fp32 grid, plus two controls.
CELLS = [
    (4096, 131072, 2048, False, False, False, torch.int32),   # selector flagship (x4 in the grid)
    (4096, 129280,  512, True,  False, True,  torch.int64),   # Sampler
    (4096, 107520, 2048, False, False, False, torch.int32),   # dsa
    (4096,  66551, 2048, False, False, False, torch.int32),   # test-topk
    ( 256, 131072, 2048, False, False, False, torch.int32),   # small batch
    (   6, 129280,  512, True,  False, True,  torch.int64),   # decode Sampler
]

for b, v, k, sv, si, rv, idx in CELLS:
    p = lib.TestParam(batch_size=b, vocab_size=v, topk=k, sorted_value=sv,
                      sorted_index=si, return_value=rv, dtype=torch.float32,
                      out_idx_dtype=idx, seed=0, num_runs=10)
    tc = lib.generate_testcase(p)
    x = tc.input

    def fn():
        return deep_select.topk(x, k, sorted=sv, sorted_index=si,
                                return_value=rv, backend="maca_c",
                                indices_type=idx, abort_when_nan_found=False)

    fn()
    torch.cuda.synchronize()
    r = kk.bench(fn, 10)
    tot = 0.0
    for n in r.get_kernel_names():
        if "topk" in n.lower():
            tot += r.get_kernel_time(n) * 1e6
    print("b%-5d V%-7d k%-4d %9.1f us" % (b, v, k, tot))
