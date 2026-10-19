// Pair-loop bisect: replicate the L2 summary walk EXACTLY (two batched loads,
// hist_add_bf16_wide_v twice, group_max8 twice) against the single-block
// reference, on the same data, one 1024-thread CTA.  Diff block_max.
//
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1500 pair_bisect.cu -o /tmp/pb
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

constexpr uint32_t kCoarse12Shift = 4;

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

// the max-tracking core of hist_add_bf16_wide_v (no atomics needed here)
__device__ __forceinline__ uint32_t keymax_of(uint4 v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t key = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t lo = w[i] & 0xFFFFu, hi = w[i] >> 16;
        const uint32_t m = lo > hi ? lo : hi;
        key = m > key ? m : key;
    }
    return key;
}

__global__ void bisect(const uint16_t* in, uint32_t nb, uint16_t* out_pair,
                       uint16_t* out_single) {
    const uint32_t tx = threadIdx.x;
    const uint32_t group = tx >> 3;
    const uint32_t lane = tx & 7u;
    const uint32_t step = blockDim.x / 8;
    // ── the L2 pair loop, verbatim shape ──
    uint32_t blk = group;
    for (; ((blk + step + 1u) << 6) <= (nb << 6); blk += 2 * step) {
        const uint4 v0 = *reinterpret_cast<const uint4*>(in + ((blk << 6) + lane * 8u));
        const uint4 v1 = *reinterpret_cast<const uint4*>(in + (((blk + step) << 6) + lane * 8u));
        uint32_t key0 = keymax_of(v0), key1 = keymax_of(v1);
        key0 = group_max8(key0, tx);
        key1 = group_max8(key1, tx);
        if (lane == 0u) {
            out_pair[blk] = (uint16_t)(key0 >> kCoarse12Shift);
            out_pair[blk + step] = (uint16_t)(key1 >> kCoarse12Shift);
        }
    }
    for (; ((blk + 1u) << 6) <= (nb << 6); blk += step) {
        const uint4 v = *reinterpret_cast<const uint4*>(in + ((blk << 6) + lane * 8u));
        uint32_t key = group_max8(keymax_of(v), tx);
        if (lane == 0u) out_pair[blk] = (uint16_t)(key >> kCoarse12Shift);
    }
    // ── reference: single-block loop ──
    for (uint32_t b = group; ((b + 1u) << 6) <= (nb << 6); b += step) {
        const uint4 v = *reinterpret_cast<const uint4*>(in + ((b << 6) + lane * 8u));
        uint32_t key = group_max8(keymax_of(v), tx);
        if (lane == 0u) out_single[b] = (uint16_t)(key >> kCoarse12Shift);
    }
}

int main() {
    const uint32_t nb = 8192;
    uint16_t *in, *op, *os;
    CK(cudaMalloc(&in, (size_t)nb * 128));
    CK(cudaMalloc(&op, nb * 2)); CK(cudaMalloc(&os, nb * 2));
    uint16_t* h = (uint16_t*)malloc((size_t)nb * 128);
    for (size_t i = 0; i < (size_t)nb * 64; i++)
        h[i] = (uint16_t)((i * 2654435761u) >> 13);
    CK(cudaMemcpy(in, h, (size_t)nb * 128, cudaMemcpyHostToDevice));
    bisect<<<1, 1024>>>(in, nb, op, os);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    uint16_t* a = (uint16_t*)malloc(nb * 2);
    uint16_t* b = (uint16_t*)malloc(nb * 2);
    CK(cudaMemcpy(a, op, nb * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(b, os, nb * 2, cudaMemcpyDeviceToHost));
    uint32_t diff = 0;
    for (uint32_t i = 0; i < nb; i++)
        if (a[i] != b[i]) {
            if (diff < 12) printf("blk %u: pair=%u single=%u\n", i, a[i], b[i]);
            diff++;
        }
    printf("diffs: %u / %u\n", diff, nb);
    return 0;
}
