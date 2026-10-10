"""One row, one number: the summary walk must not cost the histogram the row's
last partial block.

The row is 65,608 elements -- 8-aligned, so it takes the vector walk and builds a
summary, and not 64-aligned, so its last eight elements are a partial block that
no block summary covers.  Those eight are the row's largest, and `topk = 4` puts
all four answers inside them, so the histogram's tail loop is the only thing that
can count them: one that starts at `vec_len` instead of the collect's own
`list_end` leaves them out, the threshold lands in a bin four slots too low, and
the answer comes back as the four largest of the *counted* values.

Values are 1..16 at the end and 0 elsewhere, so the answer is written down rather
than computed -- the four largest are the last four indices, and `sorted_index`
prints indices ascending (the official check is `index[i] <= index[i+1]`), so
65604, 65605, 65606, 65607.  Integers that small are exact in bf16 and distinct,
so there is no tie for a legitimate tie-break to explain the difference away.

    CUDA_VISIBLE_DEVICES=2 python probe_odd_length.py                    # this tree
    CUDA_VISIBLE_DEVICES=2 DS_TREE=/tmp/armE python probe_odd_length.py  # control
"""
import os
import sys

import torch

TREE = os.environ.get(
    "DS_TREE", "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect")
sys.path.insert(0, TREE)
torch.set_default_device("cuda")
import deep_select                                                   # noqa: E402

V = L = 65608                     # V % 8 == 0, V % 64 == 8
K = 4
ROUND = deep_select.get_stride_requirement()[0] // 2   # elements per 1024 B

row = torch.zeros((1, (V + ROUND - 1) // ROUND * ROUND), dtype=torch.bfloat16)
row[0, V - 16:V] = torch.arange(1, 17, dtype=torch.float32).to(torch.bfloat16)
end = torch.tensor([L], dtype=torch.int32)

want = list(range(V - K, V))
_, idx = deep_select.topk(row[:, :V], K, sorted=False, sorted_index=True,
                          end=end, indices_type=torch.int32, return_value=False,
                          abort_when_nan_found=False, backend="maca_c")
got = idx[0].tolist()
print(f"want {want}")
print(f"got  {got}")
assert got == want, "the top four of an arange tail are its last four indices"
print("OK")
