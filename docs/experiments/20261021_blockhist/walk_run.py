# Probe driver for walk.cu / walk2.cu: build the case input with tests/lib.py and
# call the probe's `probe(ptr, rows, len, iters)`.
#   python walk_run.py /tmp/walk.so
#   python walk_run.py /tmp/walk2.so
# The run in docs/experiments/20261021_blockhist used exactly these two paths.
# DS_TESTS points at the repo's tests/ directory (default is the pod checkout).
import os
import sys
import ctypes

sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import torch, lib

torch.set_default_device("cuda")
V, B = 1048576, 1024
p = lib.TestParam(batch_size=B, vocab_size=V, topk=1024, sorted_value=False,
                  sorted_index=False, return_value=False, dtype=torch.bfloat16,
                  out_idx_dtype=torch.int32, seed=0, num_runs=1)
tc = lib.generate_testcase(p)
x = tc.input.contiguous()
print("input", tuple(x.shape), x.dtype, "%.2f GB" % (x.numel() * 2 / 1e9))

lib_ = ctypes.CDLL(sys.argv[1])
lib_.probe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32, ctypes.c_int]
lib_.probe(x.data_ptr(), B, V, 5)
