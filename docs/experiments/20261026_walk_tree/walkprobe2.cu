// The summary walk's group-max tree, five shapes at the real geometry.
// One CTA per row (ADJ4, BLOCK 1024), 1120 rows x 1M elements bf16.
//
//   a0  read only (four uint4 per iteration, XOR-accumulated)
//   a1  + the order-key transform and the per-lane max, no tree, no store
//   a2  + the production tree (3 chained mov_shfl per block) + lane-0 stores
//   a3  a2 + the post-loop block count over `block_max`  <- the production body
//   a4  a2 with two blocks packed per shuffle word (12-bit bins, one SWAR max
//       per step instead of two selects, and one 4-byte store per block pair)
//   a5  a5's pairwise mapping: each lane covers half of one block, so the tree
//       is a single row_shl:1 shuffle per block
//   a6  a1 + the stores only (the lane's own max, not the block's) -- the
//       upper bound with the table traffic still in place
//
// a4 and a6 do not compute the true block max; they exist to price a shape,
// and nothing is decided from them alone.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define BLOCK 1024
#ifndef ROWS
#define ROWS 1120
#endif
#ifndef VEC
#define VEC (1u << 20)
#endif

// row_shl:d -- lane i reads lane i+d, within the 16-lane row.  Chained d=4,2,1
// lands each half-row's max on its first lane, the `(tx & 7) == 0` leader.
#define TREE3(key)                                                              \
    do {                                                                        \
        uint32_t o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x104, 0xf, 0xf, 1); \
        key = (key) > o_ ? (key) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x102, 0xf, 0xf, 1);  \
        key = (key) > o_ ? (key) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x101, 0xf, 0xf, 1);  \
        key = (key) > o_ ? (key) : o_;                                          \
    } while (0)

__device__ __forceinline__ uint32_t keymax(const uint4 &v) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
    uint32_t m = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
        const uint32_t t = lo > hi ? lo : hi;
        m = t > m ? t : m;
    }
    return m;
}

// Per-half unsigned max of two 16-bit lanes, valid when both halves are below
// 0x8000 -- true of the 12-bit bins it is used on.  `(a | 0x8000) - (b & 0x7fff)`
// sets a half's bit 15 exactly when that half of `a` is at or above `b`'s, and
// the low half cannot carry into the high one (its result is <= 0xffff).
__device__ __forceinline__ uint32_t vmaxu2_swar(uint32_t a, uint32_t b) {
    const uint32_t d = (a | 0x80008000u) - (b & 0x7FFF7FFFu);
    const uint32_t g = (d & 0x80008000u) >> 15;   // 1 per half where a >= b
    const uint32_t m = g * 0xFFFFu;               // 0xffff per half
    return (a & m) | (b & ~m);
}

__device__ __forceinline__ uint32_t raw(const uint4 &v) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
    return w[0] ^ w[1] ^ w[2] ^ w[3];
}

template <int STAGE>
__global__ __launch_bounds__(BLOCK) void walk(const uint4 *__restrict__ in,
                                              uint32_t vec_len, uint32_t *out,
                                              uint16_t *bmax, uint32_t *histsum) {
    __shared__ uint32_t s_wide[4096];
    __shared__ uint32_t s_acc;
    for (int i = threadIdx.x; i < 4096; i += BLOCK) s_wide[i] = 0;
    if (threadIdx.x == 0) s_acc = 0;
    __syncthreads();

    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 7u;
    uint32_t blk = 8u * (tx >> 4) + (((tx >> 3) & 1u) << 2);
    constexpr uint32_t step = BLOCK / 8;
    const uint4 *row = in + (size_t)blockIdx.x * (vec_len / 8);
    uint32_t acc = 0;

    for (; ((blk + 4u) << 6) <= vec_len; blk += 4u * step) {
        const uint32_t o0 = (blk << 6) + lane * 8u;
        const uint32_t o1 = ((blk + 1u) << 6) + lane * 8u;
        const uint32_t o2 = ((blk + 2u) << 6) + lane * 8u;
        const uint32_t o3 = ((blk + 3u) << 6) + lane * 8u;
        const uint4 v0 = __ldg(row + (o0 >> 3));
        const uint4 v1 = __ldg(row + (o1 >> 3));
        const uint4 v2 = __ldg(row + (o2 >> 3));
        const uint4 v3 = __ldg(row + (o3 >> 3));
        if constexpr (STAGE == 0) {
            acc ^= raw(v0) ^ raw(v1) ^ raw(v2) ^ raw(v3);
        } else {
            uint32_t k0 = keymax(v0), k1 = keymax(v1), k2 = keymax(v2),
                     k3 = keymax(v3);
            if constexpr (STAGE == 1) {
                acc ^= k0 ^ k1 ^ k2 ^ k3;
            } else if constexpr (STAGE == 2 || STAGE == 3) {
                TREE3(k0);
                TREE3(k1);
                TREE3(k2);
                TREE3(k3);
                if (lane == 0u) {
                    bmax[blk] = (uint16_t)(k0 >> 4);
                    bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                    bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                    bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                }
                acc ^= k0 ^ k1 ^ k2 ^ k3;
            } else if constexpr (STAGE == 4) {
                // two blocks per word: bins first, then one SWAR max per step
                uint32_t p01 = (k0 >> 4) | ((k1 >> 4) << 16);
                uint32_t p23 = (k2 >> 4) | ((k3 >> 4) << 16);
                uint32_t o = (uint32_t)__builtin_mxc_mov_shfl((int)p01, 0x104, 0xf, 0xf, 1);
                p01 = vmaxu2_swar(p01, o);
                o = (uint32_t)__builtin_mxc_mov_shfl((int)p01, 0x102, 0xf, 0xf, 1);
                p01 = vmaxu2_swar(p01, o);
                o = (uint32_t)__builtin_mxc_mov_shfl((int)p01, 0x101, 0xf, 0xf, 1);
                p01 = vmaxu2_swar(p01, o);
                o = (uint32_t)__builtin_mxc_mov_shfl((int)p23, 0x104, 0xf, 0xf, 1);
                p23 = vmaxu2_swar(p23, o);
                o = (uint32_t)__builtin_mxc_mov_shfl((int)p23, 0x102, 0xf, 0xf, 1);
                p23 = vmaxu2_swar(p23, o);
                o = (uint32_t)__builtin_mxc_mov_shfl((int)p23, 0x101, 0xf, 0xf, 1);
                p23 = vmaxu2_swar(p23, o);
                if (lane == 0u) {
                    *reinterpret_cast<uint32_t *>(&bmax[blk]) = p01;
                    *reinterpret_cast<uint32_t *>(&bmax[blk + 2u]) = p23;
                }
                acc ^= p01 ^ p23;
            } else if constexpr (STAGE == 6) {
                if (lane == 0u) {
                    bmax[blk] = (uint16_t)(k0 >> 4);
                    bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                    bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                    bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                }
                acc ^= k0 ^ k1 ^ k2 ^ k3;
            }
        }
    }
    if constexpr (STAGE == 3) {
        __syncthreads();
        const uint32_t nb = vec_len >> 6;
        for (uint32_t b = tx; b < nb; b += BLOCK) atomicAdd(&s_wide[bmax[b]], 1u);
    }
    if constexpr (STAGE == 3) {
        if (tx < 4096u && (tx & 63u) == 0u) atomicAdd(&s_acc, s_wide[tx]);
    }
    if (tx == 0u) out[blockIdx.x] = acc + s_acc;
    if constexpr (STAGE == 3) {
        if (tx == 1u) histsum[blockIdx.x] = s_wide[1];
    }
}

// a5: each lane covers half of one block (four uint4 = 32 elements), so the 8
// lanes of a group hold four blocks' partial maxima and one row_shl:1 shuffle
// finishes every one of them.  The block a lane reads is `blk + (lane >> 1)`,
// its half `lane & 1`; the group still covers four adjacent blocks, so the
// table traffic and the block indices are the same as the other arms'.
__global__ __launch_bounds__(BLOCK) void walk_pair(const uint4 *__restrict__ in,
                                                   uint32_t vec_len, uint32_t *out,
                                                   uint16_t *bmax, uint32_t *histsum) {
    __shared__ uint32_t s_wide[4096];
    __shared__ uint32_t s_acc;
    for (int i = threadIdx.x; i < 4096; i += BLOCK) s_wide[i] = 0;
    if (threadIdx.x == 0) s_acc = 0;
    __syncthreads();

    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 7u;
    uint32_t blk = 8u * (tx >> 4) + (((tx >> 3) & 1u) << 2);
    constexpr uint32_t step = BLOCK / 8;
    const uint4 *row = in + (size_t)blockIdx.x * (vec_len / 8);
    uint32_t acc = 0;

    for (; ((blk + 4u) << 6) <= vec_len; blk += 4u * step) {
        // uint4 index of this lane's first chunk: block (blk + lane/2), half
        // (lane & 1), i.e. eight chunks per block, four per half.
        const uint32_t base = ((blk + (lane >> 1)) << 3) + ((lane & 1u) << 2);
        const uint4 v0 = __ldg(row + base + 0u);
        const uint4 v1 = __ldg(row + base + 1u);
        const uint4 v2 = __ldg(row + base + 2u);
        const uint4 v3 = __ldg(row + base + 3u);
        uint32_t k = keymax(v0);
        const uint32_t k1 = keymax(v1), k2 = keymax(v2), k3 = keymax(v3);
        k = k1 > k ? k1 : k;
        k = k2 > k ? k2 : k;
        k = k3 > k ? k3 : k;
        const uint32_t o = (uint32_t)__builtin_mxc_mov_shfl((int)k, 0x101, 0xf, 0xf, 1);
        k = k > o ? k : o;
        if ((lane & 1u) == 0u) bmax[blk + (lane >> 1)] = (uint16_t)(k >> 4);
        acc ^= k;
    }
    if (tx == 0u) out[blockIdx.x] = acc + s_acc;
}

static float bench(void (*launch)(const uint4 *, uint32_t, uint32_t *, uint16_t *,
                                  uint32_t *),
                   const uint4 *in, uint32_t *out, uint16_t *bmax,
                   uint32_t *hist, int reps) {
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e30f;
    for (int i = 0; i < reps; i++) {
        cudaEventRecord(a);
        launch(in, VEC, out, bmax, hist);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    return best;
}

#define LAUNCHER(name, kern)                                                   \
    static void name(const uint4 *in, uint32_t v, uint32_t *o, uint16_t *bm,   \
                     uint32_t *h) {                                            \
        kern<<<ROWS, BLOCK>>>(in, v, o, bm, h);                                \
    }

LAUNCHER(l0, walk<0>)
LAUNCHER(l1, walk<1>)
LAUNCHER(l2, walk<2>)
LAUNCHER(l3, walk<3>)
LAUNCHER(l4, walk<4>)
LAUNCHER(l6, walk<6>)
LAUNCHER(l5, walk_pair)

int main() {
    const size_t elems = (size_t)ROWS * VEC;
    const size_t bytes = elems * 2;
    uint4 *in;
    uint32_t *out, *hist;
    uint16_t *bmax;
    cudaMalloc(&in, bytes);
    cudaMalloc(&out, ROWS * 4);
    cudaMalloc(&hist, ROWS * 4);
    cudaMalloc(&bmax, ROWS * (VEC / 64) * 2);
    cudaMemset(in, 0x3c, bytes);
    printf("rows %d  vec %u  bytes %.2f GB\n", ROWS, VEC, bytes / 1e9);
    struct { const char *n; void (*f)(const uint4 *, uint32_t, uint32_t *,
                                      uint16_t *, uint32_t *); } v[] = {
        {"a0 read", l0},          {"a1 key+max", l1},
        {"a2 tree+store", l2},    {"a3 +count loop", l3},
        {"a4 packed tree", l4},   {"a5 pair map", l5},
        {"a6 store only", l6}};
    for (auto &e : v) {
        float ms = bench(e.f, in, out, bmax, hist, 10);
        printf("%-16s %8.3f ms   %7.1f GB/s  %5.1f%% of 1545\n", e.n, ms,
               bytes / (ms * 1e6), bytes / (ms * 1e6) / 1545 * 100);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    return 0;
}
