// walk5.cu -- pick the shipping shape for the adjacent-pair summary walk.
//
// walk4 established: with the SAME tree/store/atomic counts, pairing a group's
// two blocks as (2g, 2g+1) instead of (g, g+step) is 22% faster (1049 vs 859
// GB/s), and B128's 23% win is that same locality effect, not the block size.
// This file asks how far it goes:
//   ADJ2P  the pair's two uint16 bins written as one uint32 (only legal because
//          the pair is adjacent and blk is even)
//   ADJ2S  ADJ2 with the store removed -- does the store stop costing?
//   ADJ4/ADJ4P  four adjacent blocks per group, 4 trees, 2 packed stores
//   ADJ2 at BLOCK 512 (the shipping kernel's other instantiation)
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define ROWS 4096u
#define VOCAB (1u << 20)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

__device__ __forceinline__ uint32_t keymax8(const uint4 v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t key = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
        const uint32_t m = lo > hi ? lo : hi;
        key = m > key ? m : key;
    }
    return key;
}

#define TREE3(key)                                                        \
    do {                                                                  \
        uint32_t o_ = (uint32_t)__builtin_mxc_bsm_bpermute(               \
            (int)(((tx + 1u) & 63u) << 2), (int)(key));                   \
        key = (key) > o_ ? (key) : o_;                                    \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                        \
            (int)(((tx + 2u) & 63u) << 2), (int)(key));                   \
        key = (key) > o_ ? (key) : o_;                                    \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                        \
            (int)(((tx + 4u) & 63u) << 2), (int)(key));                   \
        key = (key) > o_ ? (key) : o_;                                    \
    } while (0)

template <int MODE, uint32_t BLOCK>
__global__ void walk5(const uint4* __restrict__ in, uint16_t* __restrict__ bmax,
                      uint32_t* __restrict__ sink) {
    __shared__ uint32_t s_wide[4096];
    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 7u;
    const uint32_t group = tx >> 3;
    const uint32_t step = BLOCK / 8u;
    const uint4* row = in + (size_t)blockIdx.x * (VOCAB / 8u);

    for (uint32_t b = tx; b < 4096u; b += BLOCK) s_wide[b] = 0;
    __syncthreads();
    uint16_t* row_bmax = bmax + (size_t)blockIdx.x * (VOCAB / 64u);

    if constexpr (MODE == 0 || MODE == 1 || MODE == 2) {  // FULL2 / ADJ2 / ADJ2S
        const bool adj = MODE != 0;
        for (uint32_t blk = adj ? 2u * group : group;
             ((blk + (adj ? 2u : step) + (adj ? 0u : 1u)) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + (adj ? 1u : step)) << 6) >> 3)
                                   + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            TREE3(k0);
            TREE3(k1);
            if (lane == 0u) {
                if constexpr (MODE != 2) {
                    row_bmax[blk] = (uint16_t)(k0 >> 4);
                    row_bmax[blk + (adj ? 1u : step)] = (uint16_t)(k1 >> 4);
                }
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 3) {  // ADJ2P: the pair in one uint32
        for (uint32_t blk = 2u * group; ((blk + 2u) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + 1u) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            TREE3(k0);
            TREE3(k1);
            if (lane == 0u) {
                *reinterpret_cast<uint32_t*>(&row_bmax[blk]) =
                    (k0 >> 4) | ((k1 >> 4) << 16);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 4 || MODE == 5) {  // ADJ4 / ADJ4P
        for (uint32_t blk = 4u * group; ((blk + 4u) << 6) <= VOCAB;
             blk += 4u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + 1u) << 6) >> 3) + lane);
            const uint4 v2 = __ldg(row + (((blk + 2u) << 6) >> 3) + lane);
            const uint4 v3 = __ldg(row + (((blk + 3u) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            uint32_t k2 = keymax8(v2), k3 = keymax8(v3);
            TREE3(k0);
            TREE3(k1);
            TREE3(k2);
            TREE3(k3);
            if (lane == 0u) {
                if constexpr (MODE == 4) {
                    row_bmax[blk] = (uint16_t)(k0 >> 4);
                    row_bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                    row_bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                    row_bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                } else {
                    uint32_t* p = reinterpret_cast<uint32_t*>(&row_bmax[blk]);
                    p[0] = (k0 >> 4) | ((k1 >> 4) << 16);
                    p[1] = (k2 >> 4) | ((k3 >> 4) << 16);
                }
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
                atomicAdd(&s_wide[k2 >> 4], 1u);
                atomicAdd(&s_wide[k3 >> 4], 1u);
            }
        }
    }
}

static __global__ void fill(uint4* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t x = seed ^ (uint32_t)(i * 2654435761u);
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        uint4 v;
        v.x = x; x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.y = x;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.z = x;
        x ^= x << 13; x ^= x >> 17; x ^= x << 5; v.w = x;
        p[i] = v;
    }
}

template <int MODE, uint32_t BLOCK>
static void run(const char* name, const uint4* in, uint16_t* bmax,
                uint32_t* sink, int rounds) {
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &occ, walk5<MODE, BLOCK>, BLOCK, 0));
    walk5<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        walk5<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    const double gb = (double)ROWS * VOCAB * 2.0 / 1e9;
    printf("%-8s BLOCK %4u  occ %d  %8.3f ms  %7.1f GB/s\n",
           name, BLOCK, occ, best, gb / (best / 1000.0));
}

int main(int argc, char** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    const size_t n4 = (size_t)ROWS * VOCAB / 8u;
    uint4* in = nullptr;
    uint16_t* bmax = nullptr;
    uint32_t* sink = nullptr;
    CK(cudaMalloc(&in, n4 * sizeof(uint4)));
    CK(cudaMalloc(&bmax, (size_t)ROWS * VOCAB / 64u * sizeof(uint16_t)));
    CK(cudaMalloc(&sink, ROWS * sizeof(uint32_t)));
    fill<<<512, 256>>>(in, n4, 12345u);
    CK(cudaDeviceSynchronize());
    printf("rows %u vocab %u  input %.2f GB\n", ROWS, VOCAB,
           (double)n4 * 16.0 / 1e9);
    const int rounds = 3;
    run<0, 1024>("FULL2", in, bmax, sink, rounds);
    run<1, 1024>("ADJ2", in, bmax, sink, rounds);
    run<3, 1024>("ADJ2P", in, bmax, sink, rounds);
    run<2, 1024>("ADJ2S", in, bmax, sink, rounds);
    run<4, 1024>("ADJ4", in, bmax, sink, rounds);
    run<5, 1024>("ADJ4P", in, bmax, sink, rounds);
    run<1, 512>("ADJ2", in, bmax, sink, rounds);
    run<3, 512>("ADJ2P", in, bmax, sink, rounds);
    run<0, 512>("FULL2", in, bmax, sink, rounds);
    CK(cudaFree(in));
    CK(cudaFree(bmax));
    CK(cudaFree(sink));
    return 0;
}
