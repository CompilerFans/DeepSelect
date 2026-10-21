# Why the sampled threshold cannot work for bf16 at V=1M: the tie mass at the
# k-th value.  Same code as the fp32 sim, one shape, 8 rows (1M x 8 = 32 MB).
import numpy as np
def okey(u): return (u ^ ((((u >> 15) & 0x00010001) * 0x7FFF) + 0x80008000)) & 0xFFFFFFFF
def bf16_bits(x):
    u = x.view(np.uint32).copy()
    return (u + 0x8000 + ((u >> 16) & 1)) & 0xFFFF0000
rng = np.random.default_rng(11)
for V,K in ((1048576,512),(1048576,1024),(262144,1024),(131072,512)):
    ties=[]; above=[]; over=[]
    for _ in range(8):
        key = okey(bf16_bits(rng.standard_normal(V, dtype=np.float32)))
        kth = np.partition(key, V-K)[V-K]
        ties.append(int((key==kth).sum()))
        s=V//8; m=max(1,int(round(1.4*K*s/V)))
        T=np.partition(key[:s], s-m)[s-m]
        above.append(int((key>T).sum())); over.append(int((key>=kth).sum()))
    print(f"bf16 V{V} k{K}: ties@kth med {int(np.median(ties))} max {max(ties)}"
          f" | count(>=kth) med {int(np.median(over))}"
          f" | sampled T (m={m}) -> count(>T) med {int(np.median(above))}  (need >= {K})")
