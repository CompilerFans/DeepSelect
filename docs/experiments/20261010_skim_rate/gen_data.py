"""Dump rows + the 64-element-block max-bin summary + the threshold bin T.

The walk under test is pass 2 of `radix_topk_row_bf16_b` with one test in front
of it: a 64-element block (128 B = one line, and exactly the 8 consecutive
lanes' `uint4` in the collect loop) is skipped iff no element in it has
`bin >= T`.  `bin = bf16_to_uint16(x) >> 4` (the 12-bit coarse level) and `T`
is the bin the kernel's own narrow lands on -- the smallest bin `b` with
`count(bin >= b) > k`, which is the (k+1)-th largest bin.  Pass 2 visits exactly
the elements with `bin > T` (emit) and `bin == T` (stage + fine histogram), so
skipping a block whose max bin is `< T` drops nothing it would have touched.

Data is the official generator's (`NormalFloatDistribution` = `randn_like`, the
perf grid's case), generated the way `lib.generate_testcase` does it: allocate
the row stride-aligned (1024 B / itemsize) and slice, so the rows are aligned.

    python gen_data.py <rows> <V> <k> <out_prefix>

Writes `<out_prefix>.data` (rows x V bf16), `.summary` (rows x V/64 uint16,
the per-block max bin), `.t` (rows x int32).
"""
import sys

import torch

TREE = "/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect"
sys.path.insert(0, TREE); sys.path.insert(0, TREE + "/tests")
torch.set_default_device("cuda")
import lib                                                          # noqa: E402

rows, V, K, out = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
torch.manual_seed((rows * 1_000_003 + V * 11 + K) % 2 ** 31)

ALIGN = 1024 // 2                       # bf16: the 1024-byte stride requirement
Vr = (V + ALIGN - 1) // ALIGN * ALIGN
x = torch.empty((rows, Vr), dtype=torch.bfloat16)[:, :V]
lib.NormalFloatDistribution().generate(x)
x = x.contiguous()

b = x.view(torch.int16).to(torch.int32) & 0xFFFF                     # raw bits
key = b ^ torch.where(b >= 0x8000, 0xFFFF, 0x8000)                   # ordered key
bins = key >> 4                                                      # 12-bit coarse
T = bins.kthvalue(V - K, dim=1).values.to(torch.int32)               # (k+1)-th largest
nb = V // 64
assert nb * 64 == V, "V must be a multiple of 64"
blkmax = bins.view(rows, nb, 64).amax(dim=2).to(torch.uint16)

open(out + ".data", "wb").write(x.view(torch.int16).cpu().numpy().tobytes())
open(out + ".summary", "wb").write(blkmax.cpu().numpy().tobytes())
open(out + ".t", "wb").write(T.cpu().numpy().tobytes())

keep = (blkmax.to(torch.int32) >= T[:, None])
print(f"{out}: rows={rows} V={V} k={K} blocks/row={nb} "
      f"T={T.min().item()}..{T.max().item()} blocks kept/row "
      f"= {keep.sum(dim=1).float().mean().item():.1f} "
      f"({keep.float().mean().item()*100:.2f}%)")
