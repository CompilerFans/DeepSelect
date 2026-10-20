# A/B driver: time the 7 official-shape bf16 cells against one tree (argv[1]),
# summing the kernel time of every kernel whose name contains "topk".
import os, sys
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import hashlib
import torch, kernelkit as kk, deep_select, lib
from deep_select._binding import extension_path

torch.set_default_device("cuda")
print("lib", hashlib.md5(open(extension_path("deep_select_maca"), "rb").read()).hexdigest()[:8])

CELLS = [
    (4096, 1048576, 1024),
    (4096, 1048576, 512),
    (4096, 524288, 1024),
    (4096, 262144, 1024),
    (4096, 131072, 1024),
    (4096, 65536, 1024),
    (512, 262144, 512),
]

for b, v, k in CELLS:
    p = lib.TestParam(batch_size=b, vocab_size=v, topk=k, sorted_value=False,
                      sorted_index=False, return_value=False, dtype=torch.bfloat16,
                      out_idx_dtype=torch.int32, seed=0, num_runs=10)
    tc = lib.generate_testcase(p)
    x = tc.input

    def fn():
        return deep_select.topk(x, k, return_value=False, backend="maca_c",
                                indices_type=torch.int32, abort_when_nan_found=False)

    fn()
    torch.cuda.synchronize()
    r = kk.bench(fn, 10)
    tot = 0.0
    for n in r.get_kernel_names():
        if "topk" in n.lower():
            tot += r.get_kernel_time(n) * 1e6
    print("b%-5d V%-7d k%-4d %9.1f us" % (b, v, k, tot))
