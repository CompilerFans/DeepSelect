// Pass-1 walk microbenchmark: the real kernel's uint4 + key-transform + max
// body at the real grid shape, with the histogram atomic varied.
//   MODE 0: walk + thread-local max, no atomics, no group tree
//   MODE 1: per-element shared atomics (what the kernel does today)
//   MODE 2: group-max tree + one atomic per 64 elements (block-max histogram)
//   MODE 3: group-max tree, no atomic (the proposed pass-1 ceiling)
#include <cstdio>
#include <cstdint>

template <int MODE, int BLOCK>
__global__ __launch_bounds__(BLOCK) void walk_kernel(
    const uint16_t *__restrict__ in, uint32_t len) {
    extern __shared__ uint32_t s_wide[];
    const uint32_t tx = threadIdx.x;
    const uint16_t *row = in + (uint64_t)blockIdx.x * len;
    for (uint32_t b = tx; b < 4096u; b += BLOCK) s_wide[b] = 0;
    __syncthreads();
    uint32_t acc = 0;
    for (uint32_t idx = tx * 8u; idx < len; idx += BLOCK * 8u) {
        uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + idx));
        const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
        uint32_t key = 0;
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t kp = w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
            if (MODE == 1) {
                atomicAdd(&s_wide[(kp >> 4) & 0x0FFFu], 1u);
                atomicAdd(&s_wide[kp >> 20], 1u);
            }
            const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
            const uint32_t m = lo > hi ? lo : hi;
            key = m > key ? m : key;
        }
        if (MODE >= 2) {
            uint32_t o = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 1u) & 63u) << 2), (int)key);
            key = key > o ? key : o;
            o = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 2u) & 63u) << 2), (int)key);
            key = key > o ? key : o;
            o = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 4u) & 63u) << 2), (int)key);
            key = key > o ? key : o;
            if (MODE == 2 && (tx & 7u) == 0u) atomicAdd(&s_wide[key >> 4], 1u);
        }
        acc += key;
    }
    if (acc == 0x12345678u && tx == 0) s_wide[0] = acc;
}

static float bench_one(const void *p, int rows, uint32_t len, int mode, int block,
                       int iters, int smem) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    // warm
    for (int r = 0; r < 2; r++) {
        if (block == 1024) {
            if (mode == 0) walk_kernel<0, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else if (mode == 1) walk_kernel<1, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else if (mode == 2) walk_kernel<2, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else walk_kernel<3, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
        } else {
            if (mode == 0) walk_kernel<0, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else if (mode == 1) walk_kernel<1, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else if (mode == 2) walk_kernel<2, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else walk_kernel<3, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
        }
    }
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < iters; r++) {
        if (block == 1024) {
            if (mode == 0) walk_kernel<0, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else if (mode == 1) walk_kernel<1, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else if (mode == 2) walk_kernel<2, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
            else walk_kernel<3, 1024><<<rows, 1024, smem>>>((const uint16_t *)p, len);
        } else {
            if (mode == 0) walk_kernel<0, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else if (mode == 1) walk_kernel<1, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else if (mode == 2) walk_kernel<2, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
            else walk_kernel<3, 512><<<rows, 512, smem>>>((const uint16_t *)p, len);
        }
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
    const int smem = 4096 * 4;
    const char *names[4] = {"walk-only      ", "elem-atomic    ",
                            "blockmax-atomic", "blockmax-noatom"};
    for (int block : {1024, 512}) {
        for (int mode = 0; mode < 4; mode++) {
            float ms = bench_one(p, rows, len, mode, block, iters, smem);
            double gbs = (double)rows * len * 2.0 / (ms * 1e6);
            printf("BLOCK %4d %s %8.3f ms  %7.1f GB/s\n", block, names[mode], ms, gbs);
        }
    }
    fflush(stdout);
}
