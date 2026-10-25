// Phase split of the production summary walk (ADJ4, BLOCK=1024), one CTA per
// row, at the real shape.  Each stage adds exactly one piece of the real body,
// so the deltas are what each piece costs:
//
//   s0  read only (four uint4 per iteration, XOR-accumulated)
//   s1  + the order-key transform and the per-block max (integer only)
//   s2  + the 8-lane group-max tree and the block_max store (lane 0)
//   s3  + the four shared 12-bit histogram atomics (lane 0)   <- the production body
//   s4  s3 with the next iteration's four loads issued before this one is
//       consumed (eight uint4 in flight instead of four)
//
// Numbers are one read of the row (R * V * 2 bytes) over the kernel's own time.

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

#define GROUP_MAX8_TREE(key)                                          \
    do {                                                              \
        uint32_t o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x104, 0xf, 0xf, 1); \
        key = (key) > o_ ? (key) : o_;                                \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x102, 0xf, 0xf, 1);          \
        key = (key) > o_ ? (key) : o_;                                \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(key), 0x101, 0xf, 0xf, 1);          \
        key = (key) > o_ ? (key) : o_;                                \
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
            if constexpr (STAGE >= 2) {
                GROUP_MAX8_TREE(k0);
                GROUP_MAX8_TREE(k1);
                GROUP_MAX8_TREE(k2);
                GROUP_MAX8_TREE(k3);
                if (lane == 0u) {
                    bmax[blk] = (uint16_t)(k0 >> 4);
                    bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                    bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                    bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                }
            }
            if constexpr (STAGE >= 3) {
                if (lane == 0u) {
                    atomicAdd(&s_wide[k0 >> 4], 1u);
                    atomicAdd(&s_wide[k1 >> 4], 1u);
                    atomicAdd(&s_wide[k2 >> 4], 1u);
                    atomicAdd(&s_wide[k3 >> 4], 1u);
                }
            }
            acc ^= k0 ^ k1 ^ k2 ^ k3;
        }
    }
    if (STAGE >= 3 && tx < 4096u && (tx & 63u) == 0u)
        atomicAdd(&s_acc, s_wide[tx]);
    if (tx == 0u) out[blockIdx.x] = acc + s_acc;
    if (STAGE <= 2 && tx == 0u) out[blockIdx.x] = acc;
    if (STAGE >= 3) {
        // keep the histogram's contents live without a second pass
        if (tx == 1u) histsum[blockIdx.x] = s_wide[1];
    }
}

// s4: the same body, next iteration's loads issued before this one is consumed.
__global__ __launch_bounds__(BLOCK) void walk_pre(const uint4 *__restrict__ in,
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
    constexpr uint32_t stride = 4u * step;
    const int n_iter = (int)(((vec_len >> 6) - 4u - blk) / stride) + 1;
    const uint4 *row = in + (size_t)blockIdx.x * (vec_len / 8);

    uint4 c0, c1, c2, c3;
    {
        c0 = __ldg(row + (((blk << 6) + lane * 8u) >> 3));
        c1 = __ldg(row + ((((blk + 1u) << 6) + lane * 8u) >> 3));
        c2 = __ldg(row + ((((blk + 2u) << 6) + lane * 8u) >> 3));
        c3 = __ldg(row + ((((blk + 3u) << 6) + lane * 8u) >> 3));
    }
    uint32_t acc = 0;
    for (int it = 0; it < n_iter; it++) {
        const uint32_t nb = blk + stride;
        uint4 n0, n1, n2, n3;
        if (it + 1 < n_iter) {
            n0 = __ldg(row + (((nb << 6) + lane * 8u) >> 3));
            n1 = __ldg(row + ((((nb + 1u) << 6) + lane * 8u) >> 3));
            n2 = __ldg(row + ((((nb + 2u) << 6) + lane * 8u) >> 3));
            n3 = __ldg(row + ((((nb + 3u) << 6) + lane * 8u) >> 3));
        }
        uint32_t k0 = keymax(c0), k1 = keymax(c1), k2 = keymax(c2), k3 = keymax(c3);
        GROUP_MAX8_TREE(k0);
        GROUP_MAX8_TREE(k1);
        GROUP_MAX8_TREE(k2);
        GROUP_MAX8_TREE(k3);
        if (lane == 0u) {
            bmax[blk] = (uint16_t)(k0 >> 4);
            bmax[blk + 1u] = (uint16_t)(k1 >> 4);
            bmax[blk + 2u] = (uint16_t)(k2 >> 4);
            bmax[blk + 3u] = (uint16_t)(k3 >> 4);
            atomicAdd(&s_wide[k0 >> 4], 1u);
            atomicAdd(&s_wide[k1 >> 4], 1u);
            atomicAdd(&s_wide[k2 >> 4], 1u);
            atomicAdd(&s_wide[k3 >> 4], 1u);
        }
        acc ^= k0 ^ k1 ^ k2 ^ k3;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        blk = nb;
    }
    if (tx < 4096u && (tx & 63u) == 0u) atomicAdd(&s_acc, s_wide[tx]);
    if (tx == 0u) out[blockIdx.x] = acc + s_acc;
    if (tx == 1u) histsum[blockIdx.x] = s_wide[1];
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
LAUNCHER(l4, walk_pre)

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
        {"s0 read", l0}, {"s1 key+max", l1}, {"s2 +tree+store", l2},
        {"s3 +atomic", l3}, {"s4 +prefetch", l4}};
    for (auto &e : v) {
        float ms = bench(e.f, in, out, bmax, hist, 10);
        printf("%-16s %8.3f ms   %7.1f GB/s  %5.1f%% of 1545\n", e.n, ms,
               bytes / (ms * 1e6), bytes / (ms * 1e6) / 1545 * 100);
    }
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    return 0;
}
