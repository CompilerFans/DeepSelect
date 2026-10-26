// What the summary walk's `block_max` write costs, and to what: four arms over
// the same body, differing only in where the four 2-byte stores go.
//
//   b0  read only                                (the wall reference)
//   b1  key transform + per-lane max, no store   (the ALU reference)
//   b2  store lane 0's four maxima to `uint16_t*` (the production shape)
//   b3  the same, with the table pointer `__restrict__`
//   b4  the same, into shared memory
//   b5  b2 with the 3-step tree in front of the store (production walk)
//   b6  b4 with the tree (a shared table's full shape)
//
// One CTA per row (ADJ4, BLOCK 1024), 1120 rows x 1M elements bf16.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

#define BLOCK 1024
#ifndef ROWS
#define ROWS 1120
#endif
#ifndef VEC
#define VEC (1u << 20)
#endif

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

__device__ __forceinline__ uint32_t raw(const uint4 &v) {
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
    return w[0] ^ w[1] ^ w[2] ^ w[3];
}

// The row is read once per call and never reused, so a streaming load and a
// non-temporal store are the two cache hints that could matter: the table and
// the staged candidates want the L2 the streaming row would otherwise take.
template <bool STREAM>
__device__ __forceinline__ uint4 ldrow(const uint4 *p) {
    if constexpr (STREAM) return __ldcs(p);
    else return __ldg(p);
}

// STORE: -1 no read of the key at all (the pure-read reference), 0 keys
// without a store, 1 global plain, 2 global restrict, 3 shared,
// 4 global non-temporal (`__stcs`)
template <int STORE, bool TREE, bool STREAM = false>
__global__ __launch_bounds__(BLOCK) void walk(const uint4 *__restrict__ in,
                                              uint32_t vec_len, uint32_t *out,
                                              uint16_t *bmax, uint32_t *histsum) {
    __shared__ uint32_t s_wide[4096];
    for (int i = threadIdx.x; i < 4096; i += BLOCK) s_wide[i] = 0;
    __syncthreads();

    uint16_t *__restrict__ rbmax = bmax;
    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 7u;
    uint32_t blk = 8u * (tx >> 4) + (((tx >> 3) & 1u) << 2);
    constexpr uint32_t step = BLOCK / 8;
    const uint4 *row = in + (size_t)blockIdx.x * (vec_len / 8);
    uint32_t acc = 0;

    for (; ((blk + 4u) << 6) <= vec_len; blk += 4u * step) {
        const uint4 v0 = ldrow<STREAM>(row + (((blk << 6) + lane * 8u) >> 3));
        const uint4 v1 = ldrow<STREAM>(row + ((((blk + 1u) << 6) + lane * 8u) >> 3));
        const uint4 v2 = ldrow<STREAM>(row + ((((blk + 2u) << 6) + lane * 8u) >> 3));
        const uint4 v3 = ldrow<STREAM>(row + ((((blk + 3u) << 6) + lane * 8u) >> 3));
        if constexpr (STORE == -1) {
            acc ^= raw(v0) ^ raw(v1) ^ raw(v2) ^ raw(v3);
            continue;
        }
        uint32_t k0 = keymax(v0), k1 = keymax(v1), k2 = keymax(v2), k3 = keymax(v3);
        if constexpr (TREE) {
            TREE3(k0);
            TREE3(k1);
            TREE3(k2);
            TREE3(k3);
        }
        if constexpr (STORE > 0) {
            if (lane == 0u) {
                const uint16_t m0 = (uint16_t)(k0 >> 4), m1 = (uint16_t)(k1 >> 4);
                const uint16_t m2 = (uint16_t)(k2 >> 4), m3 = (uint16_t)(k3 >> 4);
                if constexpr (STORE == 1) {
                    bmax[blk] = m0; bmax[blk + 1u] = m1;
                    bmax[blk + 2u] = m2; bmax[blk + 3u] = m3;
                } else if constexpr (STORE == 2) {
                    rbmax[blk] = m0; rbmax[blk + 1u] = m1;
                    rbmax[blk + 2u] = m2; rbmax[blk + 3u] = m3;
                } else if constexpr (STORE == 4) {
                    __stcs(&bmax[blk], m0);
                    __stcs(&bmax[blk + 1u], m1);
                    __stcs(&bmax[blk + 2u], m2);
                    __stcs(&bmax[blk + 3u], m3);
                } else {
                    // shared: same instruction count and pattern, different
                    // address space (the bin is wrapped, the value is a proxy)
                    s_wide[(blk + lane) & 4095u] = m0;
                    s_wide[(blk + 1u) & 4095u] = m1;
                    s_wide[(blk + 2u) & 4095u] = m2;
                    s_wide[(blk + 3u) & 4095u] = m3;
                }
            }
        }
        acc ^= k0 ^ k1 ^ k2 ^ k3;
    }
    if (tx == 0u) out[blockIdx.x] = acc + s_wide[0];
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

#define LAUNCHER(name, ...)                                                    \
    static void name(const uint4 *in, uint32_t v, uint32_t *o, uint16_t *bm,   \
                     uint32_t *h) {                                            \
        walk<__VA_ARGS__><<<ROWS, BLOCK>>>(in, v, o, bm, h);                   \
    }

LAUNCHER(l0, -1, false)
LAUNCHER(l1, 0, false)
LAUNCHER(l2, 1, false)
LAUNCHER(l3, 2, false)
LAUNCHER(l4, 3, false)
LAUNCHER(l5, 1, true)
LAUNCHER(l6, 3, true)
LAUNCHER(l7, 0, false, true)
LAUNCHER(l8, 4, false)
LAUNCHER(l9, 4, true)
LAUNCHER(l10, 4, true, true)

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
        {"b0 read", l0},          {"b1 key+max", l1},
        {"b2 store glob", l2},    {"b3 store restr", l3},
        {"b4 store smem", l4},    {"b5 tree+glob", l5},
        {"b6 tree+smem", l6},     {"b7 key+max ldcs", l7},
        {"b8 store stcs", l8},    {"b9 tree+stcs", l9},
        {"b10 tree ld+st cs", l10}};
    for (auto &e : v) {
        float ms = bench(e.f, in, out, bmax, hist, 10);
        printf("%-16s %8.3f ms   %7.1f GB/s  %5.1f%% of 1545\n", e.n, ms,
               bytes / (ms * 1e6), bytes / (ms * 1e6) / 1545 * 100);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    return 0;
}
