"""How many 64-element blocks can pass 2 skip?

A block can be skipped iff no element in it has coarse bin >= the threshold
bin -- exactly the set pass 2 emits/stages today, so skipping is
answer-preserving by construction.  Measured on the official generator's
data (NormalFloatDistribution), the actual rows, a few k and V.
"""
import sys
import torch
TREE = "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect"
sys.path.insert(0, TREE); sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib, test as official                                  # noqa: E402

BLK = 64
for (V, K) in [(V, K) for V in (16384, 65536, 131072, 262144, 524288, 1048576)
                for K in (512, 1024)]:
    p = [c for c in official.performance_cases()
         if (c.batch_size, c.vocab_size, c.topk) == (4096, V, K)
         and str(c.dtype) == "torch.bfloat16"][0]
    p.seed = (4096 * 1_000_003 + V * 11 + K) % 2 ** 31
    t = lib.generate_testcase(p)
    x = t.input[:32]                                   # 32 rows is plenty
    b = x.view(torch.int16).to(torch.int32) & 0xFFFF   # raw bits, unsigned
    # ordered key (same flip as bf16_to_uint16): non-negative -> ^0x8000, else ^0xffff
    key = b ^ torch.where(b >= 0x8000, 0xFFFF, 0x8000)
    bin_ = key >> 4                                    # the 12-bit coarse level
    cand = (bin_ >= bin_.sort(dim=1, descending=True).values[:, K - 1:K])   # bin >= T
    per_row = []
    for r in range(cand.shape[0]):
        c = cand[r]
        nb = (V + BLK - 1) // BLK
        pad = torch.zeros(nb * BLK, dtype=torch.bool, device=c.device)
        pad[:V] = c
        per_row.append(pad.view(nb, BLK).any(dim=1).float().mean().item())
    m = torch.tensor(per_row)
    thr = bin_.sort(dim=1, descending=True).values[:, K - 1]
    n_cand = cand.sum(dim=1).float()
    print(f"V={V:>7} k={K:>4}  candidates/row {n_cand.mean():7.0f} "
          f"({n_cand.mean()/V*100:5.2f}% of row) | blocks that CANNOT be "
          f"skipped: mean {m.mean()*100:5.1f}%  worst row {m.max()*100:5.1f}%"
          f"  -> pass-2 traffic x{m.mean():.3f}")
