// How fast is a pass-2 walk that skips whole 128-byte lines?
//
//   ./skim_walk <prefix> <rows> <V> <k> <BLOCK> <iters> [keep_every] [unroll]
//
// `unroll` picks the arm: 1 (default) / 4 plain skim, 8 list+gather with the
// list in global memory, 9 list+gather with the list in shared memory.
//
// The three walks below are the production collect loop of
// `radix_topk_row_bf16_b` (csrc/maca_kernels/xcore1000/radix_core.cuh:1389-1420),
// carved: same `idx = tx*8; idx += BLOCK*8` uint4 walk, same `bin > T` emit
// through `output[atomicAdd(&s_counter,1)]`, same `bin == T` staging into the
// arena plus the 16-bin fine histogram.  What is left out is everything that
// does not touch the row: the histogram fold, the narrow, the refine and the
// emit that follow it.
//
//   walk_full     the loop as production runs it today
//   walk_skim     the same loop behind one summary load and a branch per
//                 8 elements: `if (summary[idx >> 6] < T) continue;`
//   walk_summary  the summary stream alone, no row loads at all -- the floor
//                 the skim walk cannot go below
//
// `summary` is one uint16 per 64 elements = the block's max 12-bit bin, which
// `gen_data.py` computes from the same rows.  A block whose max bin is `< T`
// holds no element pass 2 would visit, so skipping it is answer-preserving by
// construction; this driver checks that rather than assuming it (the kept-block
// count it prints is derived from the summary, and the emit count the kernel
// reports must not move between the two walks).
//
// `--synth <keep_every>` rewrites the summary to keep every n-th block and
// skip the rest, so the rate can be read as a function of density on one
// dataset instead of one density per dataset.  Kept blocks get `0xFFFF` -- the
// elements inside are still tested against their real bins, so a kept block
// costs what any kept block costs.
//
// Build (the ref/ drivers' line, ref/build_all.sh):
//   cucc -O2 -std=c++20 --offload-arch=xcore1000 skim_walk.cu -o skim_walk

#include <cuda_runtime.h>
#include <maca_bfloat16.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define DG_CHK(call)                                                           \
    do {                                                                       \
        cudaError_t s_ = (call);                                               \
        if (s_ != cudaSuccess) {                                               \
            std::fprintf(stderr, "FAIL %s: %s\n", #call,                       \
                         cudaGetErrorString(s_));                              \
            std::exit(1);                                                      \
        }                                                                      \
    } while (0)

// Carved verbatim from radix_core.cuh:173.
__device__ __forceinline__ uint16_t bf16_to_uint16(maca_bfloat16 x) {
    uint16_t bits = __bfloat16_as_ushort(x);
#ifdef __MACACC__
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint16_t mask = static_cast<uint16_t>(0x8000u | (0u - sign));
    return static_cast<uint16_t>(bits ^ mask);
#else
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                           : static_cast<uint16_t>(bits | 0x8000);
#endif
}

constexpr uint32_t kCoarse12Shift = 4;
constexpr uint32_t kArenaEntries = 4096;    // kCoarse12ArenaEntries (16 KB)
constexpr uint32_t kOutStride = 4096;       // room for k; the trailer sits past it

// kSkim = the one test this file is about.  Everything else is production's.
// kUnroll = how many walk steps one loop trip covers; 1 is what production's
// loop compiles to when the length is a runtime argument (the trip count is
// not known, so the loop is not unrolled and each thread has ONE summary load
// in flight).  The 2/4/8 arms exist to tell "the walk is latency-bound at one
// outstanding load per thread" apart from "a sparse walk is slow".
template <int BLOCK, bool kSkim, int kUnroll>
__global__ __launch_bounds__(BLOCK) void walk_kernel(
    const maca_bfloat16 *__restrict__ in, int32_t *__restrict__ out,
    const uint16_t *__restrict__ summary, const uint32_t *__restrict__ Ts,
    uint32_t V)
{
    extern __shared__ uint32_t smem[];
    uint32_t *const stage = smem;
    __shared__ uint32_t s_counter, s_num_input, s_histogram[16];
    const uint32_t tx = threadIdx.x;
    if (tx < 16u) s_histogram[tx] = 0;
    if (tx == 0u) { s_counter = 0; s_num_input = 0; }
    __syncthreads();

    const maca_bfloat16 *const row = in + (uint64_t)blockIdx.x * V;
    int32_t *const row_out = out + (uint64_t)blockIdx.x * kOutStride;
    const uint16_t *const row_sum =
        summary == nullptr ? nullptr : summary + (uint64_t)blockIdx.x * (V >> 6);
    const uint32_t vec_len = V / 8u * 8u;
    const uint32_t T = __ldg(Ts + blockIdx.x);

    auto step = [&](uint32_t idx) {
        if (kSkim && __ldg(row_sum + (idx >> 6)) < T) return;
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + idx));
        const maca_bfloat16 *const h = reinterpret_cast<const maca_bfloat16 *>(&v);
#pragma unroll
        for (int j = 0; j < 8; j++) {
            const uint16_t key = bf16_to_uint16(h[j]);
            const uint32_t bin = key >> kCoarse12Shift;
            if (bin > T) {
                row_out[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + j);
            } else if (bin == T) {
                const uint32_t pos = atomicAdd(&s_num_input, 1u);
                if (pos < kArenaEntries) stage[pos] = idx + j;
                atomicAdd(&s_histogram[key & 15u], 1u);
            }
        }
    };
    constexpr uint32_t kStep = BLOCK * 8u;
    for (uint32_t idx = tx * 8u; idx + (kUnroll - 1) * kStep < vec_len; idx += kUnroll * kStep) {
#pragma unroll
        for (int u = 0; u < kUnroll; u++) step(idx + u * kStep);
    }
    // The tail (fewer than kUnroll steps left) runs one step at a time.
    for (uint32_t idx = tx * 8u + (vec_len / (kUnroll * kStep)) * (kUnroll * kStep);
         idx < vec_len; idx += kStep)
        step(idx);
    __syncthreads();
    if (tx == 0u) row_out[kOutStride - 1] = static_cast<int32_t>(s_counter + s_num_input);
}

// List-then-gather: pass A scans the summary (V/64 entries) and compacts the
// kept blocks into a list in shared memory; pass B walks that list with the
// same 8-lanes-per-block layout as the production walk.  The point is MLP: the
// plain skim stalls on one 128 B line at a time (measured ~6 us/MB against the
// wall's 0.6), while a list lets several blocks be in flight at once.
//
// The list is capped at 2048 entries (8 KB); a row that keeps more blocks than
// that falls back to walking the whole row, which this arm counts rather than
// hides.
constexpr uint32_t kListCap = 2048;

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void gather_kernel(
    const maca_bfloat16 *__restrict__ in, int32_t *__restrict__ out,
    const uint16_t *__restrict__ summary, const uint32_t *__restrict__ Ts,
    uint32_t V, uint32_t *n_kept, uint32_t *n_overflow)
{
    extern __shared__ uint32_t smem[];
    uint32_t *const stage = smem;                 // kArenaEntries
    uint32_t *const list = smem + kArenaEntries;  // kListCap
    __shared__ uint32_t s_counter, s_num_input, s_histogram[16], s_nlist;
    const uint32_t tx = threadIdx.x;
    if (tx < 16u) s_histogram[tx] = 0;
    if (tx == 0u) { s_counter = 0; s_num_input = 0; s_nlist = 0; }
    __syncthreads();

    const maca_bfloat16 *const row = in + (uint64_t)blockIdx.x * V;
    int32_t *const row_out = out + (uint64_t)blockIdx.x * kOutStride;
    const uint16_t *const row_sum = summary + (uint64_t)blockIdx.x * (V >> 6);
    const uint32_t T = __ldg(Ts + blockIdx.x);
    const uint32_t nb = V >> 6;

    for (uint32_t i = tx; i < nb; i += BLOCK) {
        if (__ldg(row_sum + i) < T) continue;
        const uint32_t pos = atomicAdd(&s_nlist, 1u);
        if (pos < kListCap) list[pos] = i;
    }
    __syncthreads();
    uint32_t n = s_nlist;
    if (n > kListCap) { if (tx == 0u) atomicAdd(n_overflow, 1u); n = kListCap; }

    // 8 lanes per 128 B block, four blocks per lane-group in flight.
    const uint32_t lane = tx & 7u, grp = tx >> 3;
#pragma unroll 2
    for (uint32_t j = grp; j < n; j += BLOCK / 8u) {
        const uint32_t base = list[j] << 6;   // block -> element index
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + base) + lane);
        const maca_bfloat16 *const h = reinterpret_cast<const maca_bfloat16 *>(&v);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const uint16_t key = bf16_to_uint16(h[e]);
            const uint32_t bin = key >> kCoarse12Shift;
            const uint32_t idx = base + lane * 8u + (uint32_t)e;
            if (bin > T) {
                row_out[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == T) {
                const uint32_t pos = atomicAdd(&s_num_input, 1u);
                if (pos < kArenaEntries) stage[pos] = idx;
                atomicAdd(&s_histogram[key & 15u], 1u);
            }
        }
    }
    __syncthreads();
    if (tx == 0u) {
        row_out[kOutStride - 1] = static_cast<int32_t>(s_counter + s_num_input);
        atomicAdd(n_kept, n);
    }
}

// The same list-then-gather, but the list lives in global memory and only the
// counter is in shared.  The production row kernel has ~9 KB of shared-memory
// headroom at 2 CTA/SM and an occupancy it must not lose, and a smem list also
// needs a capacity the phase-B walk must not exceed; a global list has neither
// constraint.  This arm prices the trade rather than arguing it: the list is
// ~3 KB/row against a 2 MB row.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void gather_gmem_kernel(
    const maca_bfloat16 *__restrict__ in, int32_t *__restrict__ out,
    const uint16_t *__restrict__ summary, const uint32_t *__restrict__ Ts,
    uint32_t V, uint32_t *__restrict__ list, uint32_t *n_kept, uint32_t *n_overflow)
{
    extern __shared__ uint32_t smem[];
    uint32_t *const stage = smem;                 // kArenaEntries
    __shared__ uint32_t s_counter, s_num_input, s_histogram[16], s_nlist;
    const uint32_t tx = threadIdx.x;
    if (tx < 16u) s_histogram[tx] = 0;
    if (tx == 0u) { s_counter = 0; s_num_input = 0; s_nlist = 0; }
    __syncthreads();

    const maca_bfloat16 *const row = in + (uint64_t)blockIdx.x * V;
    int32_t *const row_out = out + (uint64_t)blockIdx.x * kOutStride;
    const uint16_t *const row_sum = summary + (uint64_t)blockIdx.x * (V >> 6);
    uint32_t *const row_list = list + (uint64_t)blockIdx.x * (V >> 6);
    const uint32_t T = __ldg(Ts + blockIdx.x);
    const uint32_t nb = V >> 6;

    for (uint32_t i = tx; i < nb; i += BLOCK) {
        if (__ldg(row_sum + i) < T) continue;
        const uint32_t pos = atomicAdd(&s_nlist, 1u);
        row_list[pos] = i;                     // nb entries, so never over
    }
    __syncthreads();
    const uint32_t n = s_nlist;
    if (n > nb) { if (tx == 0u) atomicAdd(n_overflow, 1u); }

    const uint32_t lane = tx & 7u, grp = tx >> 3;
#pragma unroll 2
    for (uint32_t j = grp; j < n; j += BLOCK / 8u) {
        const uint32_t base = row_list[j] << 6;
        const uint4 v = __ldg(reinterpret_cast<const uint4 *>(row + base) + lane);
        const maca_bfloat16 *const h = reinterpret_cast<const maca_bfloat16 *>(&v);
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const uint16_t key = bf16_to_uint16(h[e]);
            const uint32_t bin = key >> kCoarse12Shift;
            const uint32_t idx = base + lane * 8u + (uint32_t)e;
            if (bin > T) {
                row_out[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == T) {
                const uint32_t pos = atomicAdd(&s_num_input, 1u);
                if (pos < kArenaEntries) stage[pos] = idx;
                atomicAdd(&s_histogram[key & 15u], 1u);
            }
        }
    }
    __syncthreads();
    if (tx == 0u) {
        row_out[kOutStride - 1] = static_cast<int32_t>(s_counter + s_num_input);
        atomicAdd(n_kept, n);
    }
}

// The summary stream alone: no row loads, so this is the skim walk's floor.
template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void summary_kernel(
    const uint16_t *__restrict__ summary, const uint32_t *__restrict__ Ts,
    uint32_t V, int32_t *sink)
{
    const uint16_t *const row_sum = summary + (uint64_t)blockIdx.x * (V >> 6);
    const uint32_t T = __ldg(Ts + blockIdx.x);
    uint32_t acc = 0;
    for (uint32_t i = threadIdx.x; i < (V >> 6); i += BLOCK)
        acc += (__ldg(row_sum + i) >= T);
    if (acc == 0xFFFFFFFFu) sink[blockIdx.x] = static_cast<int32_t>(acc);
}

// 0xFFFF in every `keep_every`-th block, 0 elsewhere: a density knob that does
// not need a new dataset.  keep_every == 0 means "keep everything".
__global__ void synth_kernel(uint16_t *dst, uint32_t nb, uint32_t keep_every)
{
    const uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb) return;
    dst[i] = (keep_every == 0u || (i % keep_every) == 0u) ? 0xFFFFu : 0u;
}

struct Args {
    std::string prefix;
    uint32_t rows, V, k, block, iters;
    uint32_t keep_every = 1;   // 1 = the summary as generated; 0 = keep everything
    uint32_t unroll = 1;
    uint32_t *d_kept = nullptr, *d_over = nullptr;   // gather arm's counters
    uint32_t *d_list = nullptr;                      // the global-list arm's list
};

template <int BLOCK>
static double run(const Args &a, const maca_bfloat16 *d_in, int32_t *d_out,
                  const uint16_t *d_sum, const uint32_t *d_T, bool skim,
                  char *flush, size_t flush_bytes, int unroll)
{
    int32_t *d_sink;
    DG_CHK(cudaMalloc(&d_sink, a.rows * sizeof(int32_t)));
    DG_CHK(cudaFuncSetAttribute(walk_kernel<BLOCK, true, 1>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kArenaEntries * 4));
    DG_CHK(cudaFuncSetAttribute(walk_kernel<BLOCK, false, 1>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kArenaEntries * 4));
    DG_CHK(cudaFuncSetAttribute(walk_kernel<BLOCK, true, 4>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kArenaEntries * 4));
    DG_CHK(cudaFuncSetAttribute(walk_kernel<BLOCK, false, 4>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kArenaEntries * 4));
    DG_CHK(cudaFuncSetAttribute(gather_kernel<BLOCK>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                (kArenaEntries + kListCap) * 4));
    DG_CHK(cudaFuncSetAttribute(gather_gmem_kernel<BLOCK>,
                                cudaFuncAttributeMaxDynamicSharedMemorySize,
                                kArenaEntries * 4));
    cudaEvent_t t0, t1;
    DG_CHK(cudaEventCreate(&t0));
    DG_CHK(cudaEventCreate(&t1));
    double total = 0.0;
    for (uint32_t it = 0; it < a.iters; it++) {
        // Evict L2 between iterations, outside the timed span.
        DG_CHK(cudaMemsetAsync(flush, 0, flush_bytes));
        DG_CHK(cudaEventRecord(t0));
        if (skim && unroll == 9)
            gather_kernel<BLOCK><<<a.rows, BLOCK, (kArenaEntries + kListCap) * 4>>>(
                d_in, d_out, d_sum, d_T, a.V, a.d_kept, a.d_over);
        else if (skim && unroll == 8)
            gather_gmem_kernel<BLOCK><<<a.rows, BLOCK, kArenaEntries * 4>>>(
                d_in, d_out, d_sum, d_T, a.V, a.d_list, a.d_kept, a.d_over);
        else if (skim && unroll == 4)
            walk_kernel<BLOCK, true, 4><<<a.rows, BLOCK, kArenaEntries * 4>>>(
                d_in, d_out, d_sum, d_T, a.V);
        else if (skim)
            walk_kernel<BLOCK, true, 1><<<a.rows, BLOCK, kArenaEntries * 4>>>(
                d_in, d_out, d_sum, d_T, a.V);
        else
            walk_kernel<BLOCK, false, 1><<<a.rows, BLOCK, kArenaEntries * 4>>>(
                d_in, d_out, nullptr, d_T, a.V);
        DG_CHK(cudaEventRecord(t1));
        DG_CHK(cudaEventSynchronize(t1));
        float ms = 0.f;
        DG_CHK(cudaEventElapsedTime(&ms, t0, t1));
        total += ms;
    }
    cudaFree(d_sink);
    return total / a.iters * 1e3;   // us per iteration
}

static void *read_file(const std::string &path, size_t want)
{
    FILE *f = std::fopen(path.c_str(), "rb");
    if (!f) { std::fprintf(stderr, "FAIL: cannot open %s\n", path.c_str()); std::exit(1); }
    void *buf = std::malloc(want);
    if (std::fread(buf, 1, want, f) != want) {
        std::fprintf(stderr, "FAIL: %s is short of %zu bytes\n", path.c_str(), want);
        std::exit(1);
    }
    std::fclose(f);
    return buf;
}

int main(int argc, char **argv)
{
    Args a;
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <prefix> <rows> <V> <k> <BLOCK> <iters> "
                     "[keep_every] [unroll]\n", argv[0]);
        return 1;
    }
    a.prefix = argv[1];
    a.rows = atoi(argv[2]); a.V = atoi(argv[3]); a.k = atoi(argv[4]);
    a.block = atoi(argv[5]); a.iters = atoi(argv[6]);
    if (argc > 7) a.keep_every = atoi(argv[7]);
    if (argc > 8) a.unroll = atoi(argv[8]);
    if (a.block != 512 && a.block != 1024) {
        std::fprintf(stderr, "FAIL: BLOCK must be 512 or 1024\n"); return 1;
    }
    const uint32_t nb = a.V >> 6;
    const size_t data_bytes = (size_t)a.rows * a.V * 2;
    const size_t sum_bytes = (size_t)a.rows * nb * 2;

    std::vector<maca_bfloat16> h_data(data_bytes / 2);
    std::vector<uint16_t> h_sum(sum_bytes / 2);
    std::vector<int32_t> h_t(a.rows);
    std::memcpy(h_data.data(), read_file(a.prefix + ".data", data_bytes), data_bytes);
    std::memcpy(h_sum.data(), read_file(a.prefix + ".summary", sum_bytes), sum_bytes);
    std::memcpy(h_t.data(), read_file(a.prefix + ".t", a.rows * 4), a.rows * 4);

    maca_bfloat16 *d_in; uint16_t *d_sum; int32_t *d_out; char *d_flush;
    uint32_t *d_T;
    DG_CHK(cudaMalloc(&d_in, data_bytes));
    DG_CHK(cudaMalloc(&d_sum, sum_bytes));
    DG_CHK(cudaMalloc(&d_T, (size_t)a.rows * 4));
    DG_CHK(cudaMalloc(&d_out, (size_t)a.rows * kOutStride * 4));
    uint32_t *h_count;
    DG_CHK(cudaHostAlloc(&h_count, 2 * sizeof(uint32_t), cudaHostAllocDefault));
    DG_CHK(cudaMemset(h_count, 0, 2 * sizeof(uint32_t)));
    DG_CHK(cudaMalloc(&a.d_kept, 2 * sizeof(uint32_t)));
    DG_CHK(cudaMemcpy(a.d_kept, h_count, 2 * sizeof(uint32_t), cudaMemcpyHostToDevice));
    a.d_over = a.d_kept + 1;
    DG_CHK(cudaMalloc(&a.d_list, (size_t)a.rows * nb * 4));
    const size_t flush_bytes = 128u << 20;
    DG_CHK(cudaMalloc(&d_flush, flush_bytes));
    DG_CHK(cudaMemcpy(d_in, h_data.data(), data_bytes, cudaMemcpyHostToDevice));
    DG_CHK(cudaMemcpy(d_sum, h_sum.data(), sum_bytes, cudaMemcpyHostToDevice));
    DG_CHK(cudaMemcpy(d_T, h_t.data(), (size_t)a.rows * 4, cudaMemcpyHostToDevice));

    // T is per row (a handful of bins apart across the grid), read once per CTA.
    uint64_t kept = 0;
    for (uint32_t r = 0; r < a.rows; r++)
        for (uint32_t b = 0; b < nb; b++)
            kept += (h_sum[(size_t)r * nb + b] >= (uint32_t)h_t[r]);

    const double data_gb = (double)data_bytes / 1e9;
    double t_full = a.block == 1024
        ? run<1024>(a, d_in, d_out, d_sum, d_T, false, d_flush, flush_bytes, 1)
        : run<512>(a, d_in, d_out, d_sum, d_T, false, d_flush, flush_bytes, 1);

    if (a.keep_every != 1) {
        // Synthetic density: rewrite the whole summary in place.
        const uint32_t total = a.rows * nb;
        synth_kernel<<<(total + 255) / 256, 256>>>(d_sum, total, a.keep_every);
        DG_CHK(cudaDeviceSynchronize());
    }
    double t_sum = [&] {
        cudaEvent_t t0, t1; DG_CHK(cudaEventCreate(&t0)); DG_CHK(cudaEventCreate(&t1));
        int32_t *sink; DG_CHK(cudaMalloc(&sink, a.rows * 4));
        double tot = 0;
        for (uint32_t it = 0; it < a.iters; it++) {
            DG_CHK(cudaMemsetAsync(d_flush, 0, flush_bytes));
            DG_CHK(cudaEventRecord(t0));
            summary_kernel<1024><<<a.rows, 1024, 0>>>(d_sum, d_T, a.V, sink);
            DG_CHK(cudaEventRecord(t1)); DG_CHK(cudaEventSynchronize(t1));
            float ms = 0.f; DG_CHK(cudaEventElapsedTime(&ms, t0, t1)); tot += ms;
        }
        return tot / a.iters * 1e3;
    }();
    double t_skim = a.block == 1024
        ? run<1024>(a, d_in, d_out, d_sum, d_T, true, d_flush, flush_bytes, a.unroll)
        : run<512>(a, d_in, d_out, d_sum, d_T, true, d_flush, flush_bytes, a.unroll);

    const double keep_frac = (a.keep_every == 0 || a.keep_every > 1)
        ? (a.keep_every == 0 ? 1.0 : 1.0 / a.keep_every)
        : (double)kept / ((double)a.rows * nb);
    const double skim_gb = (double)sum_bytes / 1e9 + keep_frac * data_gb;
    std::printf("rows=%u V=%u k=%u BLOCK=%u  blocks kept %.2f%%  "
                "full %.1f us (%.0f GB/s of %.2f GB)  skim %.1f us "
                "(%.1f effective GB/s over %.3f GB)  summary %.1f us  "
                "skim/full %.3f\n",
                a.rows, a.V, a.k, a.block, keep_frac * 100.0, t_full,
                data_gb / t_full * 1e6, data_gb, t_skim,
                skim_gb / t_skim * 1e6, skim_gb, t_sum, t_skim / t_full);
    if (a.unroll == 9 || a.unroll == 8) {
        DG_CHK(cudaMemcpy(h_count, a.d_kept, 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));
        std::printf("    gather: %u kept-block reads over %u launches (%.1f/row), "
                    "%u rows over the %u-entry list cap\n", h_count[0], a.iters,
                    (double)h_count[0] / a.iters / a.rows, h_count[1], kListCap);
    }
    cudaFree(d_in); cudaFree(d_sum); cudaFree(d_out); cudaFree(d_flush);
    return 0;
}
