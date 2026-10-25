import sys
p = "/root/bisect/csrc/maca_kernels/common/radix_row_bf16.cuh"
s = open(p).read()
old = """    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t group = threadIdx.x >> 3;
#pragma unroll 2
    for (uint32_t j = group; j < n; j += BLOCK_SIZE / 8u) {
        const uint32_t base = list[j] << 6;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + base) + lane);
        const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; i++) body(base + lane * 8u + (uint32_t)i, h[i]);
    }
"""
new = """    const uint32_t nvec = n << 3;
    const uint32_t nfull = nvec & ~(4u * BLOCK_SIZE - 1u);
    uint32_t k = threadIdx.x;
    for (; k < nfull; k += 4u * BLOCK_SIZE) {
        uint4 v0, v1, v2, v3;
        uint32_t b0, b1, b2, b3;
        {
            const uint32_t j0 = k, j1 = k + BLOCK_SIZE, j2 = k + 2u * BLOCK_SIZE,
                           j3 = k + 3u * BLOCK_SIZE;
            b0 = list[j0 >> 3] << 6;
            b1 = list[j1 >> 3] << 6;
            b2 = list[j2 >> 3] << 6;
            b3 = list[j3 >> 3] << 6;
            const uint4* p0 = reinterpret_cast<const uint4*>(input + b0);
            const uint4* p1 = reinterpret_cast<const uint4*>(input + b1);
            const uint4* p2 = reinterpret_cast<const uint4*>(input + b2);
            const uint4* p3 = reinterpret_cast<const uint4*>(input + b3);
            v0 = __ldg(p0 + (j0 & 7u));
            v1 = __ldg(p1 + (j1 & 7u));
            v2 = __ldg(p2 + (j2 & 7u));
            v3 = __ldg(p3 + (j3 & 7u));
        }
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const maca_bfloat16* h0 = reinterpret_cast<const maca_bfloat16*>(&v0);
            const maca_bfloat16* h1 = reinterpret_cast<const maca_bfloat16*>(&v1);
            const maca_bfloat16* h2 = reinterpret_cast<const maca_bfloat16*>(&v2);
            const maca_bfloat16* h3 = reinterpret_cast<const maca_bfloat16*>(&v3);
            body(b0 + (k & 7u) * 8u + (uint32_t)i, h0[i]);
            body(b1 + ((k + BLOCK_SIZE) & 7u) * 8u + (uint32_t)i, h1[i]);
            body(b2 + ((k + 2u * BLOCK_SIZE) & 7u) * 8u + (uint32_t)i, h2[i]);
            body(b3 + ((k + 3u * BLOCK_SIZE) & 7u) * 8u + (uint32_t)i, h3[i]);
        }
    }
    for (; k < nvec; k += BLOCK_SIZE) {
        const uint32_t base = list[k >> 3] << 6;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + base) + (k & 7u));
        const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; i++) body(base + (k & 7u) * 8u + (uint32_t)i, h[i]);
    }
"""
assert s.count(old) == 1
open(p, "w").write(s.replace(old, new))
print("walk patched")
