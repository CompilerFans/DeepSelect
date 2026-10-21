# Driver for the post-block-histogram clock64 phase probe
# (patch_row_phase_after.py).  Reads the nine slots the probe leaves at
# output[0, k-16:k]: eight marks + the path code.
#   python row_phase_after.py <probe-tree>
#
# Marks: 0 entry, 1 first walk done, 2 first fold+narrow done, 3 list done,
# 4 second element walk + its fold+narrow done, 5 collect done, 6 refine
# cumsum done, 7 end (set at whichever return the row takes).
# code: 1 exact-fit filter, 2 overflow rescan, 3 arena remain==0, 4 arena main.
#
# A mark the row's path never reaches reads 0 and its segment is skipped, so
# on code 1 the last printed segment under "last(path)" is the exact-fit
# filter, not the arena emit.
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
    (4096, 1048576, 1024),
    (4096, 1048576, 512),
    (4096, 262144, 1024),
    (4096, 131072, 1024),   # below the density gate: hist1b should read ~0
]
names = ["hist1a", "fold1", "list", "hist1b", "collect", "ref-cumsum", "last(path)"]

for b, V, k in cases:
    p = lib.TestParam(batch_size=b, vocab_size=V, topk=k, sorted_value=False,
                      sorted_index=False, return_value=False, dtype=torch.bfloat16,
                      out_idx_dtype=torch.int32, seed=0, num_runs=10)
    tc = lib.generate_testcase(p)
    x = tc.input

    def fn():
        return deep_select.topk(x, k, return_value=False, backend="maca_c",
                                indices_type=torch.int32, abort_when_nan_found=False)

    out = fn()
    torch.cuda.synchronize()
    # `return_value=False` still returns `(None, indices)` -- the tuple shape
    # is upstream's -- so the probe's slots are in the second element.
    d = out[1][0, k - 16:k].tolist()
    r = kk.bench(fn, 10)
    tot = 0.0
    for n in r.get_kernel_names():
        if any(s in n for s in ("coarse12", "radix", "chunk", "stage", "nan")):
            tot += r.get_kernel_time(n) * 1e6
    pts, code = d[0:8], d[8]
    end = pts[7]
    segs = []
    prev = pts[0]
    for i in range(1, 8):
        if 0 < pts[i] <= end and pts[i] >= prev:
            segs.append("%s %.0f(%.0f%%)" % (names[i - 1], pts[i] - prev,
                                             100.0 * (pts[i] - prev) / max(end, 1)))
            prev = pts[i]
    print("b%-5d V%-7d k%-4d kernel %8.1fus code %d end %d" % (b, V, k, tot, code, end))
    print("    " + "  ".join(segs) + "   raw=%s" % pts)
