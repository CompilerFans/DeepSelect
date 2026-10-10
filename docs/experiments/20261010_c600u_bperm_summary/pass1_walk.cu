// What does pass 1 cost per byte on this part, and where does it go?
//
// The row kernel is now ~pass 1 plus a little collect (the kept-block list has
// pass 2 walking ~4% of a long row), so pass 1's per-AP throughput is the
// kernel's ceiling.  This carves the production pass-1 loop out of
// `radix_core.cuh` (the summary branch of `radix_topk_row_bf16_b`, ~1393-1432)
// and prices its pieces by ablation:
//
//   full        production: SWAR key fold, 2 shared atomics per word, the
//               per-block max summary with its chained shuffles, NaN vote
//   noatomic    the same, histogram atomics removed
//   nosummary   the same, no block max (key kept alive via the NaN-style xor)
//   nosummary_noatomic
//   bare        the uint4 loads alone
//
// Build:  cucc -O2 -std=c++20 --offload-arch=xcore1600 pass1_walk.cu -o pw
// Run:    ./pw <rows> <V> <BLOCK> <iters> <arm>
// `rows` CTAs walk one row each, in one wave if rows == SM x CTAs-per-SM.
#include <cuda_runtime.h>
#include <maca_bfloat16.h>

#include <cstdint>
#include <vector>
#include <cstdio>
#include <cstdlib>

#define CHK(x)                                                                 \
    do {                                                                       \
        cudaError_t s_ = (x);                                                  \
        if (s_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "FAIL %s: %s\n", #x, cudaGetErrorString(s_)); \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

constexpr uint32_t kShift = 4;             // kCoarse12Shift
constexpr uint32_t kBins = 4096;           // kCoarse12Bins

__device__ __forceinline__ void walk_wide(uint32_t* __restrict__ s_wide,
                                          const maca_bfloat16* __restrict__ in,
                                          uint32_t idx, uint32_t* max_key,
                                          uint32_t* acc, bool do_atomic,
                                          bool do_summary,
                                          const uint4* pre = nullptr) {
    uint4 v = pre ? *pre : __ldg(reinterpret_cast<const uint4*>(in + idx));
    (void)in; (void)idx;
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t z = 0u;
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        if (do_atomic) {
            atomicAdd(&s_wide[(kp >> kShift) & 0x0FFFu], 1u);
            atomicAdd(&s_wide[kp >> (16 + kShift)], 1u);
        } else {
            z ^= kp;
        }
        if (do_summary) {
            const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
            const uint32_t m = lo > hi ? lo : hi;
            *max_key = m > *max_key ? m : *max_key;
        }
        z |= (((w[i] & 0x7FFF7FFFu) | 0x80008000u) - 0x7F817F81u) & 0x80008000u;
    }
    *acc ^= z;
}

// UNROLL = how many uint4 blocks one thread has in flight per iteration.
// The trip count is a runtime value in production, so the outer loop is not
// unrolled there; this axis asks what unrolling it would buy.
template <int ARM, int UNROLL = 1>
__global__ __launch_bounds__(1024) void p1_kernel(
    const maca_bfloat16* __restrict__ in, uint32_t len,
    uint16_t* __restrict__ block_max, uint32_t* __restrict__ sink) {
    constexpr bool kAtomic = (ARM == 0 || ARM == 2 || ARM == 5);
    constexpr bool kSummary = (ARM == 0 || ARM == 1 || ARM == 5);
    __shared__ uint32_t s_wide[kBins];
    const uint32_t tx = threadIdx.x;
    for (uint32_t b = tx; b < kBins; b += blockDim.x) s_wide[b] = 0;
    __syncthreads();

    const maca_bfloat16* row = in + (size_t)blockIdx.x * len;
    const uint32_t vec_len = len & ~7u;
    uint32_t acc = 0;
    uint16_t* my_max = block_max + (size_t)blockIdx.x * (len >> 6);

    const uint32_t group = tx >> 3;
    const uint32_t lane = tx & 7u;
    const uint32_t stride = blockDim.x / 8;
    for (uint32_t blk = group; ((blk + (UNROLL - 1) * stride + 1u) << 6) <= vec_len;
         blk += stride * UNROLL) {
        uint4 vs[UNROLL];
        #pragma unroll
        for (int u = 0; u < UNROLL; u++)
            vs[u] = __ldg(reinterpret_cast<const uint4*>(
                row + ((blk + u * stride) << 6) + lane * 8u));
        #pragma unroll
        for (int u = 0; u < UNROLL; u++) {
        const uint32_t idx = ((blk + u * stride) << 6) + lane * 8u;
        (void)idx;
        uint32_t key = 0;
        walk_wide(s_wide, row, idx, &key, &acc, kAtomic, kSummary, &vs[u]);
        if (kSummary) {
#ifdef __MACACC__
            if (ARM == 5) {
                // the swap candidate: the same chained tree through the
                // platform's own 64-lane permute (1 op + index math).  No mask:
                // only lane 0 is read, and contamination can only enter lanes
                // >= 5, which never feed lane 0's chain.
                uint32_t other = (uint32_t)__builtin_mxc_bsm_bpermute(
                    (int)(((tx + 1u) & 63u) << 2), (int)key);
                key = key > other ? key : other;
                other = (uint32_t)__builtin_mxc_bsm_bpermute(
                    (int)(((tx + 2u) & 63u) << 2), (int)key);
                key = key > other ? key : other;
                other = (uint32_t)__builtin_mxc_bsm_bpermute(
                    (int)(((tx + 4u) & 63u) << 2), (int)key);
                key = key > other ? key : other;
            } else {
                const unsigned long long gmask = 0xFFull << (tx & 0x38u);
                uint32_t other = (uint32_t)__shfl_down_sync(gmask, key, 1);
                key = key > other ? key : other;
                other = (uint32_t)__shfl_down_sync(gmask, key, 2);
                key = key > other ? key : other;
                other = (uint32_t)__shfl_down_sync(gmask, key, 4);
                key = key > other ? key : other;
            }
#else
            const unsigned gmask = 0xFFu << (tx & 0x18u);
            uint32_t other = (uint32_t)__shfl_down_sync(gmask, key, 1);
            key = key > other ? key : other;
            other = (uint32_t)__shfl_down_sync(gmask, key, 2);
            key = key > other ? key : other;
            other = (uint32_t)__shfl_down_sync(gmask, key, 4);
            key = key > other ? key : other;
#endif
            if (lane == 0u) my_max[blk + u * stride] = (uint16_t)(key >> kShift);
        }
        }
    }
    // The tail elements [vec_len, len) are production's own tail loop; keep one
    // read of them so the arm reads the same bytes.
    for (uint32_t idx = vec_len + tx; idx < len; idx += blockDim.x)
        acc ^= (uint32_t)__bfloat16_as_ushort(__ldg(row + idx));
    if (acc == 0x5A5A5A5Au) sink[blockIdx.x] = acc;
    __syncthreads();
    if (tx == 0 && s_wide[0] == 0xDEADBEEFu) sink[blockIdx.x] = s_wide[1];
}

int main(int argc, char** argv) {
    if (argc < 6) {
        std::fprintf(stderr, "usage: %s <rows> <V> <BLOCK> <iters> <arm 0..4>\n",
                     argv[0]);
        return 2;
    }
    const int rows = std::atoi(argv[1]);
    const uint32_t V = (uint32_t)std::atoi(argv[2]);
    const int block = std::atoi(argv[3]);
    const int iters = std::atoi(argv[4]);
    const int arm = std::atoi(argv[5]);
    if (block != 1024) { std::fprintf(stderr, "block is 1024\n"); return 2; }

    maca_bfloat16* in = nullptr;
    uint16_t* bmax = nullptr;
    uint32_t* sink = nullptr;
    const size_t bytes = (size_t)rows * V * sizeof(maca_bfloat16);
    CHK(cudaMalloc(&in, bytes));
    {
        // Normal-ish bf16 values, host-side: the production crowding shape
        // (many elements share the top bins) is what makes the histogram
        // atomic hot, and a constant fill is the pathological end of it.
        std::vector<maca_bfloat16> host((size_t)rows * V);
        uint32_t h_ = 0x9E3779B9u;
        auto rnd = [&]() {
            h_ ^= h_ << 13; h_ ^= h_ >> 17; h_ ^= h_ << 5;
            return (float)(h_ & 0xFFFFFFu) / (float)0x1000000u;
        };
        for (size_t i = 0; i < host.size(); i++) {
            const float t = (rnd() + rnd() + rnd()) * 1.3333333f - 2.0f;
            host[i] = __float2bfloat16(t);
        }
        CHK(cudaMemcpy(in, host.data(), bytes, cudaMemcpyHostToDevice));
    }
    CHK(cudaMalloc(&bmax, (size_t)rows * (V / 64) * sizeof(uint16_t)));
    CHK(cudaMalloc(&sink, rows * sizeof(uint32_t)));

    auto launch = [&](int a) {
        switch (a) {
        case 0: p1_kernel<0><<<rows, block>>>(in, V, bmax, sink); break;
        case 1: p1_kernel<1><<<rows, block>>>(in, V, bmax, sink); break;
        case 2: p1_kernel<2><<<rows, block>>>(in, V, bmax, sink); break;
        case 3: p1_kernel<3><<<rows, block>>>(in, V, bmax, sink); break;
        case 4: p1_kernel<4><<<rows, block>>>(in, V, bmax, sink); break;
        case 12: p1_kernel<4,2><<<rows, block>>>(in, V, bmax, sink); break;
        case 14: p1_kernel<4,4><<<rows, block>>>(in, V, bmax, sink); break;
        case 22: p1_kernel<0,2><<<rows, block>>>(in, V, bmax, sink); break;
        case 24: p1_kernel<0,4><<<rows, block>>>(in, V, bmax, sink); break;
        case 30: p1_kernel<5><<<rows, block>>>(in, V, bmax, sink); break;
        default: p1_kernel<4><<<rows, block>>>(in, V, bmax, sink); break;
        }
    };
    launch(arm);
    CHK(cudaDeviceSynchronize());

    cudaEvent_t t0, t1;
    CHK(cudaEventCreate(&t0));
    CHK(cudaEventCreate(&t1));
    CHK(cudaEventRecord(t0));
    for (int i = 0; i < iters; i++) launch(arm);
    CHK(cudaEventRecord(t1));
    CHK(cudaEventSynchronize(t1));
    float ms = 0;
    CHK(cudaEventElapsedTime(&ms, t0, t1));

    {
        // correctness receipt: the block_max table this arm produced, hashed
        std::vector<uint16_t> hb((size_t)rows * (V / 64));
        CHK(cudaMemcpy(hb.data(), bmax, hb.size() * sizeof(uint16_t),
                       cudaMemcpyDeviceToHost));
        uint64_t h = 1469598103934665603ull;
        for (uint16_t x : hb) { h ^= x; h *= 1099511628211ull; }
        std::printf("  bmax hash %016llx  first %u %u %u %u\n",
                    (unsigned long long)h, hb[0], hb[1], hb[2], hb[3]);
    }
    const double gb = (double)rows * V * 2.0 * iters / 1e9;
    std::printf("arm %d rows %d V %u iters %d: %.3f ms  %.1f GB/s  %.2f GB/s/AP\n",
                arm, rows, V, iters, ms, gb / (ms / 1e3), gb / (ms / 1e3) / 28.0);
    const char* names[] = {"full", "noatomic", "nosummary",
                           "nosummary_noatomic", "bare"};
    std::fprintf(stderr, "  (%s%s)\n", arm < 5 ? names[arm] : "arm",
                 arm == 12 ? " bare u2" : arm == 14 ? " bare u4"
                 : arm == 22 ? " full u2" : arm == 24 ? " full u4" : "");
    return 0;
}
