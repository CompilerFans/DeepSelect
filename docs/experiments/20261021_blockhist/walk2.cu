// The real kernel's summary-walk shape: 8-lane groups cover one 64-element
// block, two blocks in flight per thread, group-max tree, block_max store.
//   MODE 0: per-element shared atomics (today)
//   MODE 1: one atomic per 64 elements (block-max histogram)
//   MODE 2: no atomic (ceiling)
#include <cstdio>
#include <cstdint>

template <bool kAtomic>
__device__ __forceinline__ uint32_t hist8(uint32_t *s_wide,
                                          const uint16_t *row, uint32_t idx) {
    uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + idx));
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
    uint32_t key = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp = w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        if (kAtomic) {
            atomicAdd(&s_wide[(kp >> 4) & 0x0FFFu], 1u);
            atomicAdd(&s_wide[kp >> 20], 1u);
        }
        const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
        const uint32_t m = lo > hi ? lo : hi;
        key = m > key ? m : key;
    }
    return key;
}

template <int MODE, int BLOCK>
__global__ __launch_bounds__(BLOCK) void walk_kernel(
    const uint16_t *__restrict__ in, uint32_t len, uint16_t *__restrict__ bm) {
    extern __shared__ uint32_t s_wide[];
    const uint32_t tx = threadIdx.x;
    const uint16_t *row = in + (uint64_t)blockIdx.x * len;
    uint16_t *block_max = bm + (uint64_t)blockIdx.x * (len / 64);
    for (uint32_t b = tx; b < 4096u; b += BLOCK) s_wide[b] = 0;
    __syncthreads();
    const uint32_t group = tx >> 3;
    const uint32_t lane = tx & 7u;
    const uint32_t step = BLOCK / 8u;
    uint32_t blk = group;
    #define TREE(k)                                                          \
        { uint32_t o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                \
              (int)(((tx + 1u) & 63u) << 2), (int)(k));                      \
          (k) = (k) > o_ ? (k) : o_;                                         \
          o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                         \
              (int)(((tx + 2u) & 63u) << 2), (int)(k));                      \
          (k) = (k) > o_ ? (k) : o_;                                         \
          o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                         \
              (int)(((tx + 4u) & 63u) << 2), (int)(k));                      \
          (k) = (k) > o_ ? (k) : o_; }
    for (; ((blk + step + 1u) << 6) <= len; blk += 2 * step) {
        uint32_t k0 = hist8<MODE == 0>(s_wide, row, (blk << 6) + lane * 8u);
        uint32_t k1 = hist8<MODE == 0>(s_wide, row, ((blk + step) << 6) + lane * 8u);
        TREE(k0)
        TREE(k1)
        if (lane == 0u) {
            block_max[blk] = (uint16_t)(k0 >> 4);
            block_max[blk + step] = (uint16_t)(k1 >> 4);
            if (MODE == 1) {
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    }
    for (; ((blk + 1u) << 6) <= len; blk += step) {
        uint32_t k0 = hist8<MODE == 0>(s_wide, row, (blk << 6) + lane * 8u);
        TREE(k0)
        if (lane == 0u) {
            block_max[blk] = (uint16_t)(k0 >> 4);
            if (MODE == 1) atomicAdd(&s_wide[k0 >> 4], 1u);
        }
    }
    __syncthreads();
    if (s_wide[0] == 0xDEADBEEFu) block_max[0] = 1;
}

template <int BLOCK>
static float bench_one(const void *p, uint16_t *bm, int rows, uint32_t len,
                       int mode, int iters, int smem) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < 2; r++) {
        if (mode == 0) walk_kernel<0, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
        else if (mode == 1) walk_kernel<1, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
        else walk_kernel<2, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < iters; r++) {
        if (mode == 0) walk_kernel<0, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
        else if (mode == 1) walk_kernel<1, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
        else walk_kernel<2, BLOCK><<<rows, BLOCK, smem>>>((const uint16_t *)p, len, bm);
    }
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

extern "C" void probe(const void *p, int rows, uint32_t len, int iters) {
    uint16_t *bm = nullptr;
    cudaMalloc(&bm, (size_t)rows * (len / 64) * sizeof(uint16_t));
    const int smem = 4096 * 4;
    const char *names[3] = {"elem-atomic (today)", "blockmax-atomic    ", "no-atomic (ceiling)"};
    for (int mode = 0; mode < 3; mode++) {
        float ms = bench_one<1024>(p, bm, rows, len, mode, iters, smem);
        double gbs = (double)rows * len * 2.0 / (ms * 1e6);
        printf("BLOCK 1024 %s %8.3f ms  %7.1f GB/s\n", names[mode], ms, gbs);
    }
    fflush(stdout);
    cudaFree(bm);
}
