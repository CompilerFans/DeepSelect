// walk6.cu -- how far the adjacency trend goes, and what it does to the dense
// (per-element-atomic) walk that the below-gate cells run.
//   ADJ8   eight adjacent blocks per group (one full 1 KB per group iteration)
//   ADJ4S  ADJ4 without the block_max store -- remaining headroom
//   HIST_FULL2 / HIST_ADJ4  the kHist=true walk (8 shared atomics per uint4),
//          which is the below-gate path: does adjacency help or hurt there?
//   ADJ4 at BLOCK 512
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define ROWS 4096u
#define VOCAB (1u << 20)

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// kHist: per-element atomics (the dense walk).  Otherwise the block-max only.
__device__ __forceinline__ uint32_t keymax8(const uint4 v, uint32_t* s_wide) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t key = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        if (s_wide != nullptr) {
            atomicAdd(&s_wide[(kp >> 4) & 0x0FFFu], 1u);
            atomicAdd(&s_wide[kp >> 20], 1u);
        }
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
__global__ void walk6(const uint4* __restrict__ in, uint16_t* __restrict__ bmax,
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
    uint32_t* hist = (MODE == 4) ? s_wide : nullptr;

    if constexpr (MODE == 0 || MODE == 3 || MODE == 4) {  // ADJ4 / ADJ4S / ADJ4H
        for (uint32_t blk = 4u * group; ((blk + 4u) << 6) <= VOCAB;
             blk += 4u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + 1u) << 6) >> 3) + lane);
            const uint4 v2 = __ldg(row + (((blk + 2u) << 6) >> 3) + lane);
            const uint4 v3 = __ldg(row + (((blk + 3u) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0, hist), k1 = keymax8(v1, hist);
            uint32_t k2 = keymax8(v2, hist), k3 = keymax8(v3, hist);
            TREE3(k0);
            TREE3(k1);
            TREE3(k2);
            TREE3(k3);
            if (lane == 0u) {
                if constexpr (MODE != 3) {
                    row_bmax[blk] = (uint16_t)(k0 >> 4);
                    row_bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                    row_bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                    row_bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                }
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
                atomicAdd(&s_wide[k2 >> 4], 1u);
                atomicAdd(&s_wide[k3 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 1) {  // ADJ8
        for (uint32_t blk = 8u * group; ((blk + 8u) << 6) <= VOCAB;
             blk += 8u * step) {
            uint32_t k[8];
            #pragma unroll
            for (int j = 0; j < 8; j++)
                k[j] = keymax8(__ldg(row + (((blk + (uint32_t)j) << 6) >> 3)
                                     + lane), hist);
            #pragma unroll
            for (int j = 0; j < 8; j++) TREE3(k[j]);
            if (lane == 0u) {
                #pragma unroll
                for (int j = 0; j < 8; j++) {
                    row_bmax[blk + (uint32_t)j] = (uint16_t)(k[j] >> 4);
                    atomicAdd(&s_wide[k[j] >> 4], 1u);
                }
            }
        }
    } else if constexpr (MODE == 2) {  // HIST_FULL2: dense walk, step pairing
        for (uint32_t blk = group; ((blk + step + 1u) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + step) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0, s_wide), k1 = keymax8(v1, s_wide);
            TREE3(k0);
            TREE3(k1);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                row_bmax[blk + step] = (uint16_t)(k1 >> 4);
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
        &occ, walk6<MODE, BLOCK>, BLOCK, 0));
    walk6<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        walk6<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    const double gb = (double)ROWS * VOCAB * 2.0 / 1e9;
    printf("%-10s BLOCK %4u  occ %d  %8.3f ms  %7.1f GB/s\n",
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
    run<0, 1024>("ADJ4", in, bmax, sink, rounds);
    run<1, 1024>("ADJ8", in, bmax, sink, rounds);
    run<3, 1024>("ADJ4S", in, bmax, sink, rounds);
    run<2, 1024>("HIST_FULL2", in, bmax, sink, rounds);
    run<4, 1024>("HIST_ADJ4", in, bmax, sink, rounds);
    run<0, 512>("ADJ4", in, bmax, sink, rounds);
    CK(cudaFree(in));
    CK(cudaFree(bmax));
    CK(cudaFree(sink));
    return 0;
}
