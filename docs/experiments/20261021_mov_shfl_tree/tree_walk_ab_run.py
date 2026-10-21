# Driver for tree_walk_ab.cu -- same pattern as 20261021_blockhist/walk_run.py:
# build the bf16 case with tests/lib.py, hand it to the probe.
#   python tree_walk_ab_run.py /tmp/ab.so
# DS_TESTS points at the repo's tests/ directory (default is the pod checkout).
# V=1M only: the row-paired arm covers whole sweeps of 8*rows blocks and has
# no tail loop; 16384 blocks divides evenly at BLOCK 512 and 1024.
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
lib_.probe.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_uint32,
                       ctypes.c_int]
lib_.probe(x.data_ptr(), B, V, 5)
