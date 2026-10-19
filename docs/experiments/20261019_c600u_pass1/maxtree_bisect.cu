// Micro-bisect: does the extracted group_max8 (function form) compute the same
// block maxima as the original inline tree?  One kernel, 1024 threads, the
// summary walk's own geometry (8-lane groups over 64-element blocks), both
// forms side by side on the same registers.
//
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1600 maxtree_bisect.cu -o /tmp/mtb
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

constexpr uint32_t kCoarse12Shift = 4;

// the extracted function form (from the L2 edit)
__device__ __forceinline__ uint32_t group_max8(uint32_t key, uint32_t tx) {
    uint32_t other = (uint32_t)__builtin_mxc_bsm_bpermute(
        (int)(((tx + 1u) & 63u) << 2), (int)key);
    key = key > other ? key : other;
    other = (uint32_t)__builtin_mxc_bsm_bpermute(
        (int)(((tx + 2u) & 63u) << 2), (int)key);
    key = key > other ? key : other;
    other = (uint32_t)__builtin_mxc_bsm_bpermute(
        (int)(((tx + 4u) & 63u) << 2), (int)key);
    key = key > other ? key : other;
    return key;
}

__global__ void bisect(const uint16_t* in, uint32_t n_blocks, uint16_t* out_a,
                       uint16_t* out_b) {
    const uint32_t tx = threadIdx.x;
    const uint32_t group = tx >> 3;
    const uint32_t lane = tx & 7u;
    const uint32_t step = blockDim.x / 8;
    for (uint32_t blk = group; ((blk + 1u) << 6) <= (n_blocks << 6);
         blk += step) {
        const uint16_t* base = in + (blk << 6) + lane * 8u;
        // load 8 u16 = one uint4's worth
        uint4 v = *reinterpret_cast<const uint4*>(base);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
        uint32_t key = 0;
#pragma unroll
        for (int i = 0; i < 4; i++) {
            const uint32_t lo = w[i] & 0xFFFFu, hi = w[i] >> 16;
            const uint32_t m = lo > hi ? lo : hi;
            key = m > key ? m : key;
        }
        // form A: extracted function
        uint32_t keyA = group_max8(key, tx);
        // form B: the original inline spelling
        uint32_t keyB = key;
        {
            uint32_t other = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 1u) & 63u) << 2), (int)keyB);
            keyB = keyB > other ? keyB : other;
            other = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 2u) & 63u) << 2), (int)keyB);
            keyB = keyB > other ? keyB : other;
            other = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((tx + 4u) & 63u) << 2), (int)keyB);
            keyB = keyB > other ? keyB : other;
        }
        if (lane == 0u) {
            out_a[blk] = (uint16_t)(keyA >> kCoarse12Shift);
            out_b[blk] = (uint16_t)(keyB >> kCoarse12Shift);
        }
    }
}

int main() {
    const uint32_t n_blocks = 16384;   // 1M elements
    uint16_t *in, *oa, *ob;
    CK(cudaMalloc(&in, (size_t)n_blocks * 128));
    CK(cudaMalloc(&oa, n_blocks * 2)); CK(cudaMalloc(&ob, n_blocks * 2));
    // fill with pseudo-random u16
    uint16_t* h = (uint16_t*)malloc((size_t)n_blocks * 128);
    for (size_t i = 0; i < (size_t)n_blocks * 64; i++)
        h[i] = (uint16_t)(i * 2654435761u >> 13);
    CK(cudaMemcpy(in, h, (size_t)n_blocks * 128, cudaMemcpyHostToDevice));
    bisect<<<1, 1024>>>(in, n_blocks, oa, ob);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint16_t* a = (uint16_t*)malloc(n_blocks * 2);
    uint16_t* b = (uint16_t*)malloc(n_blocks * 2);
    CK(cudaMemcpy(a, oa, n_blocks * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b, ob, n_blocks * 2, cudaMemcpyDeviceToHost));
    uint32_t diff = 0;
    for (uint32_t i = 0; i < n_blocks; i++)
        if (a[i] != b[i]) {
            if (diff < 8) printf("blk %u: A=%u B=%u\n", i, a[i], b[i]);
            diff++;
        }
    printf("diffs: %u / %u\n", diff, n_blocks);
    return 0;
}
