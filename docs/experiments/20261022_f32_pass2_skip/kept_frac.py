# Kept-block fraction at the kernel's own threshold: a block is kept when its
# max key's 12-bit bin is at or above the k-th value's bin -- exactly the test
# `topk_coarse12_row` compacts on.  Torch, no probe, so it answers "does the
# list arm get used, and at what density" for any cell on the grid.
#
# The encode is `primitives.cuh`'s `refine_bin` at 32 bits, on int64 so the
# xor cannot overflow: positives set the sign bit, negatives invert.
import torch, lib

torch.set_default_device("cuda")


def bin12(value):
    bits = value.view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    key = torch.where(bits & 0x80000000 != 0, ~bits & 0xFFFFFFFF, bits | 0x80000000)
    return key >> 20


CELLS = [(4096, 131072, 2048), (4096, 129280, 512), (4096, 107520, 2048),
         (4096, 66551, 2048), (256, 131072, 2048)]
for B, V, K in CELLS:
    p = lib.TestParam(batch_size=B, vocab_size=V, topk=K, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32, out_idx_dtype=torch.int32,
                      seed=0, num_runs=1)
    tc = lib.generate_testcase(p)
    x = tc.input.contiguous()
    del tc
    kth = torch.topk(x, K, dim=1).values[:, -1]
    thr = bin12(kth)[:, None]
    nblk = V // 16                      # whole blocks only, as the kernel walks them
    bb = x[:, :nblk * 16].view(B, nblk, 16).amax(dim=2)
    kept = (bin12(bb) >= thr).sum(1)
    frac = kept.float().mean().item() / nblk
    print("b%-5d V%-7d k%-4d kept %5.1f%%  (max row %5.1f%%, G=16 list %s)"
          % (B, V, K, 100 * frac, 100 * kept.max().item() / nblk,
             "used" if frac <= 0.75 else "NOT used -> dense"))
    del x, kth, thr, bb, kept
    torch.cuda.empty_cache()
