# Driver for filter_stage.cu: build the official fp32 case with tests/lib.py,
# compute each row's preset threshold (the true rank-m key, m = ceil(1.4k) for
# k >= 1024 else ceil(1.6k) -- the sim's margin rule), hand both to the probe.
#   python filter_stage_run.py /tmp/fs.so
# DS_TESTS points at the repo's tests/ directory (default is the pod checkout).
import os
import sys
import ctypes

sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import torch, lib

torch.set_default_device("cuda")

SHAPES = [(4096, 131072, 2048), (4096, 129280, 512), (4096, 107520, 2048)]

lib_ = ctypes.CDLL(sys.argv[1])
lib_.probe.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_int, ctypes.c_uint32, ctypes.c_int]

for B, V, K in SHAPES:
    p = lib.TestParam(batch_size=B, vocab_size=V, topk=K, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32, out_idx_dtype=torch.int32,
                      seed=0, num_runs=1)
    tc = lib.generate_testcase(p)
    x = tc.input.contiguous()

    m = (K * 14 + 9) // 10 if K >= 1024 else (K * 16 + 9) // 10
    kth = torch.topk(x, m, dim=1).values[:, -1].contiguous()
    bits = kth.view(torch.int32)
    key = torch.where(bits < 0, ~bits, bits | (-2147483648)).contiguous()
    outc = torch.empty(B, dtype=torch.int32)

    print(f"\n=== b{B} V{V} k{K}  m={m}  ({x.numel() * 4 / 1e9:.2f} GB) ===")
    lib_.probe(x.data_ptr(), key.data_ptr(), outc.data_ptr(), B, V, 5)
    # after the last mode (1024-thread mode 5) outc holds that mode's counts;
    # re-run mode order means counts come from mode 5 at THREADS 1024 -- the
    # count is mode-independent, so one read-back is a valid sanity check.
    c = outc.cpu()
    print(f"candidate counts: min {c.min().item()} med {int(c.median().item())} "
          f"max {c.max().item()}  (target ~{m}, cap 4096)")
    del x, kth, bits, key, outc, tc
    torch.cuda.empty_cache()
