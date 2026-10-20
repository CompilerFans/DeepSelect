# Driver for the bf16 row kernel's clock64 phase probe (patch_row_phase.py).
# Reads the 9 slots the probe leaves at output[0, k-16:k] and prints the six
# phase deltas as percentages; `raw=` is the untouched slot list.
#   python row_phase.py <probe-tree>
# The three cases are the ones in this directory's README §0/§0's second table.
#
# Archived from the run that produced those tables.  Only the tree path and the
# tests path were parameterized at archive time: `arm` was the hardcoded
# "/root/ds_phase" and DS_TESTS defaults to the pod checkout.
import os
import sys

arm = sys.argv[1]
sys.path.insert(0, arm)
sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import torch, kernelkit as kk, deep_select, lib
from deep_select._binding import extension_path
import hashlib

torch.set_default_device("cuda")
print("lib", hashlib.md5(open(extension_path("deep_select_maca"), "rb").read()).hexdigest()[:8])

cases = [
    (torch.bfloat16, 4096, 1048576, 1024),
    (torch.bfloat16, 4096, 1048576, 512),
    (torch.bfloat16, 4096, 262144, 1024),
]
names = ["hist", "fold+narrow", "list", "collect", "ref-cumsum", "arena+emit"]

for dt, b, V, k in cases:
    p = lib.TestParam(batch_size=b, vocab_size=V, topk=k, sorted_value=False,
                      sorted_index=False, return_value=False, dtype=dt,
                      out_idx_dtype=torch.int32, seed=0, num_runs=10)
    tc = lib.generate_testcase(p)
    x = tc.input

    def fn():
        return deep_select.topk(x, k, return_value=False, backend="maca_c",
                                indices_type=torch.int32, abort_when_nan_found=False)

    out = fn()
    torch.cuda.synchronize()
    d = out[0, k - 16:k].tolist()
    r = kk.bench(fn, 10)
    tot = 0.0
    for n in r.get_kernel_names():
        if any(s in n for s in ("coarse12", "radix", "chunk", "stage", "nan")):
            tot += r.get_kernel_time(n) * 1e6
    code, end = d[8], d[7]
    pts = d[0:7]
    segs = []
    prev = pts[0]
    for i in range(1, 7):
        if 0 < pts[i] <= end and pts[i] >= prev:
            segs.append("%s %.0f(%.0f%%)" % (names[i - 1], pts[i] - prev, 100.0 * (pts[i] - prev) / max(end, 1)))
            prev = pts[i]
    print("b%-5d V%-7d k%-4d kernel %8.1fus code %d end %d" % (b, V, k, tot, code, end))
    print("    " + "  ".join(segs) + "   raw=%s" % pts)
