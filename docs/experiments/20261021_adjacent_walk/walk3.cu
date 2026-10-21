// walk3.cu -- the bf16 block-max summary walk's cost structure, at the real
// shape and the real launch geometry (one CTA per row, BLOCK 1024, 4096 rows x
// 1M elements).  Standalone on purpose: an in-kernel ablation that perturbs the
// histogram moves the threshold and the collect writes past its slots
// (Memory Violation 0x4), so the ablations live here.
//
// Arms, each a template on the walk shape:
//   FULL2   the shipping shape: key transform, per-thread max of 8, 3-step
//           bpermute tree, lane-0 store + 1 atomic per 64-element block,
//           2 blocks in flight per group
//   FULL4   the same with 4 blocks in flight (the MLP lever)
//   ONE1    FULL2 with a 1-step tree instead of 3 (WRONG answer; bounds the
//           tree's cost at a third of it, same atomic rate)
//   RAW2    FULL2 without the ordered-key transform (WRONG answer; bounds the
//           transform)
//   READ2   loads + per-thread max only, store gated on an impossible value
//   READ4   the same with 4 blocks in flight -- the pure-read curve at this
//           thread count, which no earlier probe measured
//   B128    two adjacent uint4 per lane, one tree per 128-element block
//           (halves the tree, store and atomic rate per element)
//
// Every arm prints its occupancy, so "the walk is capped by X" is read against
// how many CTAs were actually resident.
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

__device__ __forceinline__ uint32_t rawmax8(const uint4 v) {
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

// The shipping tree, textually -- never extract this into a function.
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

#define TREE1(key)                                                        \
    do {                                                                  \
        uint32_t o_ = (uint32_t)__builtin_mxc_bsm_bpermute(               \
            (int)(((tx + 1u) & 63u) << 2), (int)(key));                   \
        key = (key) > o_ ? (key) : o_;                                    \
    } while (0)

template <int MODE, uint32_t BLOCK>
__global__ void walk3(const uint4* __restrict__ in, uint16_t* __restrict__ bmax,
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

    if constexpr (MODE == 0) {  // FULL2
        for (uint32_t blk = group; ((blk + step + 1u) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + step) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            TREE3(k0);
            TREE3(k1);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                row_bmax[blk + step] = (uint16_t)(k1 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 1) {  // FULL4
        for (uint32_t blk = group; ((blk + 3u * step + 1u) << 6) <= VOCAB;
             blk += 4u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + step) << 6) >> 3) + lane);
            const uint4 v2 = __ldg(row + (((blk + 2u * step) << 6) >> 3) + lane);
            const uint4 v3 = __ldg(row + (((blk + 3u * step) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            uint32_t k2 = keymax8(v2), k3 = keymax8(v3);
            TREE3(k0);
            TREE3(k1);
            TREE3(k2);
            TREE3(k3);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                row_bmax[blk + step] = (uint16_t)(k1 >> 4);
                row_bmax[blk + 2u * step] = (uint16_t)(k2 >> 4);
                row_bmax[blk + 3u * step] = (uint16_t)(k3 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
                atomicAdd(&s_wide[k2 >> 4], 1u);
                atomicAdd(&s_wide[k3 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 2) {  // ONE1: 1-step tree, same atomic rate
        for (uint32_t blk = group; ((blk + step + 1u) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + step) << 6) >> 3) + lane);
            uint32_t k0 = keymax8(v0), k1 = keymax8(v1);
            TREE1(k0);
            TREE1(k1);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                row_bmax[blk + step] = (uint16_t)(k1 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 3) {  // RAW2: no ordered-key transform
        for (uint32_t blk = group; ((blk + step + 1u) << 6) <= VOCAB;
             blk += 2u * step) {
            const uint4 v0 = __ldg(row + ((blk << 6) >> 3) + lane);
            const uint4 v1 = __ldg(row + (((blk + step) << 6) >> 3) + lane);
            uint32_t k0 = rawmax8(v0), k1 = rawmax8(v1);
            TREE3(k0);
            TREE3(k1);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                row_bmax[blk + step] = (uint16_t)(k1 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
            }
        }
    } else if constexpr (MODE == 4) {  // READ2
        uint32_t acc = 0;
        for (uint32_t blk = group; ((blk + step + 1u) << 6) <= VOCAB;
             blk += 2u * step) {
            acc ^= keymax8(__ldg(row + ((blk << 6) >> 3) + lane));
            acc ^= keymax8(__ldg(row + (((blk + step) << 6) >> 3) + lane));
        }
        if (acc == 0xFFFFFFFFu) sink[blockIdx.x] = acc;
    } else if constexpr (MODE == 5) {  // READ4
        uint32_t acc = 0;
        for (uint32_t blk = group; ((blk + 3u * step + 1u) << 6) <= VOCAB;
             blk += 4u * step) {
            acc ^= keymax8(__ldg(row + ((blk << 6) >> 3) + lane));
            acc ^= keymax8(__ldg(row + (((blk + step) << 6) >> 3) + lane));
            acc ^= keymax8(__ldg(row + (((blk + 2u * step) << 6) >> 3) + lane));
            acc ^= keymax8(__ldg(row + (((blk + 3u * step) << 6) >> 3) + lane));
        }
        if (acc == 0xFFFFFFFFu) sink[blockIdx.x] = acc;
    } else if constexpr (MODE == 6) {  // B128: one tree per 128 elements
        for (uint32_t blk = group; ((blk + 1u) << 7) <= VOCAB; blk += step) {
            const uint32_t base = (blk << 7) + lane * 16u;
            const uint4 v0 = __ldg(row + (base >> 3));
            const uint4 v1 = __ldg(row + (base >> 3) + 1u);
            const uint32_t m1 = keymax8(v1);
            uint32_t k0 = keymax8(v0);
            k0 = k0 > m1 ? k0 : m1;
            TREE3(k0);
            if (lane == 0u) {
                row_bmax[blk] = (uint16_t)(k0 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
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
        &occ, walk3<MODE, BLOCK>, BLOCK, 0));
    // warmup
    walk3<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        walk3<MODE, BLOCK><<<ROWS, BLOCK>>>(in, bmax, sink);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    const double gb = (double)ROWS * VOCAB * 2.0 / 1e9;
    printf("%-8s BLOCK %4u  occ %d CTA/SM  %8.3f ms  %7.1f GB/s\n",
           name, BLOCK, occ, best, gb / (best / 1000.0));
}

int main(int argc, char** argv) {
    int dev = 0;
    if (argc > 1) dev = atoi(argv[1]);
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
    run<4, 1024>("READ2", in, bmax, sink, rounds);
    run<5, 1024>("READ4", in, bmax, sink, rounds);
    run<0, 1024>("FULL2", in, bmax, sink, rounds);
    run<1, 1024>("FULL4", in, bmax, sink, rounds);
    run<2, 1024>("ONE1", in, bmax, sink, rounds);
    run<3, 1024>("RAW2", in, bmax, sink, rounds);
    run<6, 1024>("B128", in, bmax, sink, rounds);
    run<0, 512>("FULL2", in, bmax, sink, rounds);
    run<0, 256>("FULL2", in, bmax, sink, rounds);
    run<4, 512>("READ2", in, bmax, sink, rounds);
    CK(cudaFree(in));
    CK(cudaFree(bmax));
    CK(cudaFree(sink));
    return 0;
}
