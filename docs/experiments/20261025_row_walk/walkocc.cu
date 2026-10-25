// What the summary walk achieves, as a function of the occupancy it is given
// and the block size it is launched with.  The body is the production one
// (`summary_walk`'s ADJ4 arm): four adjacent blocks per group iteration, the
// three-shuffle group tree, the block_max store and the four shared histogram
// atomics from lane 0.
//
//   stage 0  read only
//   stage 3  the production body
//
// Each is run at BLOCK 1024 and 512, each at its natural occupancy and with
// enough dynamic shared memory requested to pin it to one CTA per SM, and the
// driver's own cudaOccupancyMaxActiveBlocksPerMultiprocessor is printed beside
// every row -- a hand division is what CLAUDE.md forbids here.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

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

template <int STAGE, int BS>
__global__ __launch_bounds__(BS) void walk(const uint4 *__restrict__ in,
                                           uint32_t vec_len, uint32_t *out,
                                           uint16_t *bmax, uint32_t *histsum) {
    __shared__ uint32_t s_wide[4096];
    __shared__ uint32_t s_acc;
    for (int i = threadIdx.x; i < 4096; i += BS) s_wide[i] = 0;
    if (threadIdx.x == 0) s_acc = 0;
    __syncthreads();

    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 7u;
    uint32_t blk = 8u * (tx >> 4) + (((tx >> 3) & 1u) << 2);
    constexpr uint32_t step = BS / 8;
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
                if constexpr (STAGE == 5 || STAGE == 7) {
                    if (lane == 0u) {
                        const uint2 p = make_uint2(
                            (uint32_t)(k0 >> 4) | ((uint32_t)(k1 >> 4) << 16),
                            (uint32_t)(k2 >> 4) | ((uint32_t)(k3 >> 4) << 16));
                        *reinterpret_cast<uint2 *>(bmax + blk) = p;
                    }
                } else if constexpr (STAGE != 8) {
                    if (lane == 0u) {
                        bmax[blk] = (uint16_t)(k0 >> 4);
                        bmax[blk + 1u] = (uint16_t)(k1 >> 4);
                        bmax[blk + 2u] = (uint16_t)(k2 >> 4);
                        bmax[blk + 3u] = (uint16_t)(k3 >> 4);
                    }
                }
            }
            if constexpr (STAGE == 3 || STAGE == 5) {
                if (lane == 0u) {
                    atomicAdd(&s_wide[k0 >> 4], 1u);
                    atomicAdd(&s_wide[k1 >> 4], 1u);
                    atomicAdd(&s_wide[k2 >> 4], 1u);
                    atomicAdd(&s_wide[k3 >> 4], 1u);
                }
            }
            if constexpr (STAGE == 6 || STAGE == 7) {
                const uint32_t mine = lane == 0u ? k0 : lane == 1u ? k1
                                     : lane == 2u ? k2 : lane == 3u ? k3 : 0u;
                if (lane < 4u) atomicAdd(&s_wide[mine >> 4], 1u);
            }
            acc ^= k0 ^ k1 ^ k2 ^ k3;
        }
    }
    if (STAGE != 0 && tx < 4096u && (tx & 63u) == 0u)
        atomicAdd(&s_acc, s_wide[tx]);
    if (tx == 0u) out[blockIdx.x] = acc + s_acc;
    if (tx == 0u) out[blockIdx.x] = acc;
    if (tx == 1u) histsum[blockIdx.x] = s_wide[1];
}

static const char *stage_name(int s) {
    switch (s) {
    case 0: return "s0 read";
    case 1: return "s1 key+max";
    case 2: return "s2 +tree+store";
    case 3: return "s3 +atom(lane0)";
    case 5: return "s5 packstore";
    case 6: return "s6 spreadatom";
    case 7: return "s7 pack+spread";
    default: return "s8 tree-nostore";
    }
}

template <int STAGE, int BS>
static void run(const uint4 *in, uint32_t *out, uint16_t *bmax, uint32_t *hist,
                size_t rec, bool pinned, const char *tag) {
    const size_t smem = pinned ? rec : 0;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, walk<STAGE, BS>, BS,
                                                  smem);
    float best = 1e30f;
    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(a);
        walk<STAGE, BS><<<ROWS, BS, smem>>>(in, VEC, out, bmax, hist);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    const double bytes = (double)ROWS * VEC * 2;
    printf("%-11s BLOCK %4d  occ %d  smem %6zu B   %8.3f ms  %7.1f GB/s  %5.1f%%\n",
           stage_name(STAGE), BS, occ, smem, best, bytes / (best * 1e6),
           bytes / (best * 1e6) / 1545 * 100);
    fflush(stdout);
    (void)tag;
}

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
    printf("rows %d  vec %u  bytes %.2f GB  (one read of the row)\n", ROWS, VEC,
           bytes / 1e9);

    // The pin: more than half of the AP's shared memory, so only one CTA fits.
    const size_t pin = cudaFuncSetAttribute(walk<3, 1024>,
                                            cudaFuncAttributeMaxDynamicSharedMemorySize,
                                            70000) == cudaSuccess ? 70000 : 0;
    cudaFuncSetAttribute(walk<3, 512>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<0, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<1, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<2, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<5, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<6, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<7, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<8, 1024>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<0, 512>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<5, 512>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);
    cudaFuncSetAttribute(walk<7, 512>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, 70000);

    run<0, 1024>(in, out, bmax, hist, pin, false, "");
    run<1, 1024>(in, out, bmax, hist, pin, false, "");
    run<2, 1024>(in, out, bmax, hist, pin, false, "");
    run<3, 1024>(in, out, bmax, hist, pin, false, "");
    run<5, 1024>(in, out, bmax, hist, pin, false, "");
    run<6, 1024>(in, out, bmax, hist, pin, false, "");
    run<7, 1024>(in, out, bmax, hist, pin, false, "");
    run<8, 1024>(in, out, bmax, hist, pin, false, "");
    printf("-- BLOCK 512 at its natural occupancy --\n");
    run<3, 512>(in, out, bmax, hist, pin, false, "");
    run<5, 512>(in, out, bmax, hist, pin, false, "");
    run<7, 512>(in, out, bmax, hist, pin, false, "");
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) printf("ERROR: %s\n", cudaGetErrorString(err));
    return 0;
}
