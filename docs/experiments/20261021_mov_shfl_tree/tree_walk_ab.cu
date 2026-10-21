// hist1a ADJ4 summary-walk A/B: current bsm_bpermute group tree vs the
// mov_shfl row-paired tree vs a no-tree ceiling. bf16, V=1M, one CTA/row.
//   cucc -O3 --offload-arch=xcore1600 -shared -fPIC tree_walk_ab.cu -o /tmp/ab.so
//   python tree_walk_ab_run.py /tmp/ab.so
//
// TREE arms (template int TREE):
//   0 = shipping form: 8-lane groups, 3 chained bsm_bpermute per block
//     (LDS-pipe, from radix_row_bf16.cuh's DS_GROUP_MAX8_TREE)
//   1 = row-paired: a 16-lane row covers 2x4 adjacent blocks; 3 chained
//     mov_shfl (Row Shift Right d=4,2,1, bc=1) reduce both halves at once --
//     lane 0 holds the low half's block max, lane 8 the high half's
//     (direction per the guide; the RATE does not depend on it).
//     1.5 shuffles per block, ALU pipe.
//   2 = no tree (local max only, lane 0 stores): the ceiling.
// NAN arms: with/without the production NaN predicate (2 ALU/word; the
// predicate here is a cost-equivalent stand-in, not the bit-exact vote).
//
// All arms write the block_max table and do the same 4 leader atomics per
// iteration, so the only moving part is the tree.
#include <cstdio>
#include <cstdint>

template <bool kNan>
__device__ __forceinline__ uint32_t key8(const uint16_t *row, uint32_t idx,
                                         uint32_t *nan_local) {
    uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + idx));
    const uint32_t *w = reinterpret_cast<const uint32_t *>(&v);
    uint32_t key = 0;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        if (kNan)
            *nan_local |= ((w[i] & 0x7F807F80u) == 0x7F807F80u) &
                          ((w[i] & 0x007F007Fu) != 0u);
        const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
        const uint32_t m = lo > hi ? lo : hi;
        key = m > key ? m : key;
    }
    return key;
}

#define TREE_BPERMUTE(k, tx)                                                \
    do {                                                                    \
        uint32_t o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                 \
            (int)(((tx + 1u) & 63u) << 2), (int)(k));                       \
        (k) = (k) > o_ ? (k) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                          \
            (int)(((tx + 2u) & 63u) << 2), (int)(k));                       \
        (k) = (k) > o_ ? (k) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                          \
            (int)(((tx + 4u) & 63u) << 2), (int)(k));                       \
        (k) = (k) > o_ ? (k) : o_;                                          \
    } while (0)

#define TREE_MOV_SHFL(k)                                                    \
    do {                                                                    \
        uint32_t o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(k), 0x114,     \
                                                       0xf, 0xf, 1);        \
        (k) = (k) > o_ ? (k) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(k), 0x112, 0xf, 0xf, 1);\
        (k) = (k) > o_ ? (k) : o_;                                          \
        o_ = (uint32_t)__builtin_mxc_mov_shfl((int)(k), 0x111, 0xf, 0xf, 1);\
        (k) = (k) > o_ ? (k) : o_;                                          \
    } while (0)

template <int TREE, bool kNan, int BLOCK>
__global__ __launch_bounds__(BLOCK) void ab_kernel(
    const uint16_t *__restrict__ in, uint32_t len, uint16_t *__restrict__ bm,
    uint32_t *__restrict__ guard) {
    extern __shared__ uint32_t s_wide[];
    const uint32_t tx = threadIdx.x;
    const uint16_t *row = in + (uint64_t)blockIdx.x * len;
    const uint32_t nblk = len >> 6;
    uint16_t *block_max = bm + (uint64_t)blockIdx.x * nblk;
    for (uint32_t b = tx; b < 4096u; b += BLOCK) s_wide[b] = 0u;
    __syncthreads();
    uint32_t nan_local = 0u;

    if (TREE == 1) {
        // row r owns blocks [8r, 8r+8), strides 8*rows; each half-row takes
        // its 4 adjacent blocks, one uint4 per lane per block.
        const uint32_t row16 = tx >> 4;
        const uint32_t half = (tx >> 3) & 1u;
        const uint32_t lane8 = tx & 7u;
        const uint32_t rows_per_cta = BLOCK / 16u;
        for (uint32_t base = 8u * row16; base + 8u <= nblk;
             base += 8u * rows_per_cta) {
            const uint32_t b0 = base + half * 4u;
            uint32_t k0 = key8<kNan>(row, (b0 << 6) + lane8 * 8u, &nan_local);
            uint32_t k1 = key8<kNan>(row, ((b0 + 1u) << 6) + lane8 * 8u,
                                     &nan_local);
            uint32_t k2 = key8<kNan>(row, ((b0 + 2u) << 6) + lane8 * 8u,
                                     &nan_local);
            uint32_t k3 = key8<kNan>(row, ((b0 + 3u) << 6) + lane8 * 8u,
                                     &nan_local);
            TREE_MOV_SHFL(k0);
            TREE_MOV_SHFL(k1);
            TREE_MOV_SHFL(k2);
            TREE_MOV_SHFL(k3);
            if (lane8 == 0u) {  // lanes 0 and 8 of the row
                block_max[b0] = (uint16_t)(k0 >> 4);
                block_max[b0 + 1u] = (uint16_t)(k1 >> 4);
                block_max[b0 + 2u] = (uint16_t)(k2 >> 4);
                block_max[b0 + 3u] = (uint16_t)(k3 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
                atomicAdd(&s_wide[k2 >> 4], 1u);
                atomicAdd(&s_wide[k3 >> 4], 1u);
            }
        }
    } else {
        // shipping mapping: group g owns [4g, 4g+4), strides 4*step.
        const uint32_t group = tx >> 3;
        const uint32_t lane = tx & 7u;
        const uint32_t step = BLOCK / 8u;
        for (uint32_t blk = 4u * group; ((blk + 4u) << 6) <= len;
             blk += 4u * step) {
            uint32_t k0 = key8<kNan>(row, (blk << 6) + lane * 8u, &nan_local);
            uint32_t k1 = key8<kNan>(row, ((blk + 1u) << 6) + lane * 8u,
                                     &nan_local);
            uint32_t k2 = key8<kNan>(row, ((blk + 2u) << 6) + lane * 8u,
                                     &nan_local);
            uint32_t k3 = key8<kNan>(row, ((blk + 3u) << 6) + lane * 8u,
                                     &nan_local);
            if (TREE == 0) {
                TREE_BPERMUTE(k0, tx);
                TREE_BPERMUTE(k1, tx);
                TREE_BPERMUTE(k2, tx);
                TREE_BPERMUTE(k3, tx);
            }
            if (lane == 0u) {
                block_max[blk] = (uint16_t)(k0 >> 4);
                block_max[blk + 1u] = (uint16_t)(k1 >> 4);
                block_max[blk + 2u] = (uint16_t)(k2 >> 4);
                block_max[blk + 3u] = (uint16_t)(k3 >> 4);
                atomicAdd(&s_wide[k0 >> 4], 1u);
                atomicAdd(&s_wide[k1 >> 4], 1u);
                atomicAdd(&s_wide[k2 >> 4], 1u);
                atomicAdd(&s_wide[k3 >> 4], 1u);
            }
        }
    }
    __syncthreads();
    if (nan_local == 0xFFFFFFFFu && tx == 0) guard[blockIdx.x] = nan_local;
    if (s_wide[0] == 0xDEADBEEFu && tx == 0) guard[blockIdx.x] = 1u;
}

template <int TREE, bool kNan, int BLOCK>
static float bench_one(const void *p, uint16_t *bm, uint32_t *g, int rows,
                       uint32_t len, int iters) {
    const int smem = 4096 * 4;
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < 2; r++)
        ab_kernel<TREE, kNan, BLOCK><<<rows, BLOCK, smem>>>(
            (const uint16_t *)p, len, bm, g);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < iters; r++)
        ab_kernel<TREE, kNan, BLOCK><<<rows, BLOCK, smem>>>(
            (const uint16_t *)p, len, bm, g);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

template <int BLOCK>
static void run_block(const void *p, uint16_t *bm, uint32_t *g, int rows,
                      uint32_t len, int iters) {
    struct { int tree; const char *name; } arms[3] = {
        {0, "bpermute tree (shipping)"},
        {1, "mov_shfl row-paired    "},
        {2, "no tree (ceiling)      "}};
    for (int nan = 1; nan >= 0; nan--) {
        for (int a = 0; a < 3; a++) {
            float ms;
            if (nan) {
                switch (arms[a].tree) {
                case 0: ms = bench_one<0, true, BLOCK>(p, bm, g, rows, len, iters); break;
                case 1: ms = bench_one<1, true, BLOCK>(p, bm, g, rows, len, iters); break;
                default: ms = bench_one<2, true, BLOCK>(p, bm, g, rows, len, iters); break;
                }
            } else {
                switch (arms[a].tree) {
                case 0: ms = bench_one<0, false, BLOCK>(p, bm, g, rows, len, iters); break;
                case 1: ms = bench_one<1, false, BLOCK>(p, bm, g, rows, len, iters); break;
                default: ms = bench_one<2, false, BLOCK>(p, bm, g, rows, len, iters); break;
                }
            }
            const double gbs = (double)rows * (double)len * 2.0 / (ms * 1e6);
            printf("BLOCK %4d  NaN=%d  %s  %8.3f ms  %7.1f GB/s\n", BLOCK, nan,
                   arms[a].name, ms, gbs);
        }
    }
    fflush(stdout);
}

extern "C" void probe(const void *p, int rows, uint32_t len, int iters) {
    const uint32_t nblk = len >> 6;
    uint16_t *bm = nullptr;
    uint32_t *g = nullptr;
    cudaMalloc(&bm, (size_t)rows * nblk * sizeof(uint16_t));
    cudaMalloc(&g, (size_t)rows * sizeof(uint32_t));
    run_block<1024>(p, bm, g, rows, len, iters);
    run_block<512>(p, bm, g, rows, len, iters);
    cudaFree(bm);
    cudaFree(g);
}
