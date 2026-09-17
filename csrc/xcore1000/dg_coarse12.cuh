// deep_gemm's `topk_coarse12`, ported whole.
//
// Why this file exists
// -------------------
// `radix_topk_row_f32` (radix_core.cuh) and `topk_coarse12`
// (`mcDeepGEMM/csrc/kernels/fp32_topk.cu`, extracted at
// `ref/deep_gemm/xcore1000_fp32_topk.cu`) answer the same question with the
// same class of algorithm -- histogram the high bits, pick the bin holding
// rank `topk`, rank that bin -- and the ported one is **2.0-2.3x faster** on
// the shapes where deep_gemm routes to it (`ref/README.md`, and the ledger's
// §12 table).  Both are two row walks, so the walk count is not the
// difference; the differences that are measurable are:
//
//   * 12 coarse bits against 8, so the bin the refine ranks is 2-3x smaller.
//   * 640 threads against 512 -- and the block size is the one knob that
//     moved our own row kernel by 24% (`-DKBLOCK_SIZE=1024`).
//   * one 16 KB arena doing double duty (coarse histogram, then candidates)
//     against a 14 KB arena plus a ping-pong refine buffer.
//
// The port is deliberately an *extraction*, not a rewrite: the arithmetic,
// the round structure and the overflow handler below are deep_gemm's, so the
// two can be compared without asking which of them is better written.  What
// is dropped is the page-table transform (`Transform = true`), which needs
// the `page_table` / `cu_seqlens_row` / `q_positions` plumbing this
// repository does not synthesize -- the same subtraction `ref/` already made.
// What is added is the one thing deep_gemm keeps outside the kernel: this
// runs over `[0, length)` of the row, which on this path is the whole row,
// and it returns *positions in that window*, so the caller's contract half
// (`topk_kernel_radix`) can treat them exactly like the split's answer.
//
// The gate that keeps it off C600U is in `maca_topk.cu` (`f32_coarse12_applies`),
// and it has two halves.  The *budget* half -- does this part's shared memory
// hold the arena -- is a compile-time constant of the artifact's family
// (`ARCH_SMEM_PER_AP_BYTES`, `csrc/structs.h`), which is possible because
// `setup.py` builds one artifact per family.  The *tuning* half -- is the
// measured crossing the right one here -- stays a runtime test on `sm_count`,
// because a ladder read off one machine does not transfer by arithmetic.
// Neither half is an arch macro: `__MACA_ARCH__` is device-pass only, so a host
// `#ifdef` on it is dead in every image (see the memory note
// `maca-arch-macro-and-torch-free-builds`).  Nothing in this file reads the
// architecture, so no other target's code changes when it is enabled: the
// kernels are only ever launched from the gated route.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace rk {
namespace dg12 {

// ── deep_gemm's `detail` constants, verbatim ───────────────────────────────
//
// `kMaxTopK` is 2048 in the original and the arena below is sized from
// `kCoarse12SmemBytes`, so this port serves `topk <= 2048`.  That is also
// where deep_gemm's own dispatch stops consulting coarse12 for `top_k`, so
// the boundary is the original's rather than a new one.  Callers gate on it
// (`f32_coarse12_applies`), and this header asserts it rather than trusting
// the call site.
constexpr int kThreads = 640;
constexpr int kMaxTopK = 2048;
constexpr int kRadix = 256;
constexpr int kHistogramPadding = 1;
constexpr int kCoarseBits = 12;
constexpr int kCoarseBins = 1 << kCoarseBits;
constexpr int kFirstShift = 32 - kCoarseBits;              // 20
constexpr int kWideGroupSize = 1 << (kCoarseBits - 8);     // 16
constexpr int kRefineShift0 = kFirstShift - 8;             // 12
constexpr int kRefineShift1 = kFirstShift - 16;            // 4
constexpr int kFinalRadixBits = kRefineShift1;             // 4
constexpr int kFinalRadixMask = (1 << kFinalRadixBits) - 1;
constexpr size_t kSmemBytes = 16 * 1024;
constexpr int kCandidateCapacity =
    (int)(kSmemBytes / (2 * sizeof(int32_t)));             // 2048
static_assert(kCandidateCapacity >= kMaxTopK,
              "the coarse12 arena must hold a whole top-k answer");
static_assert(kSmemBytes >= 2 * (size_t)kCandidateCapacity * sizeof(int32_t),
              "the arena is aliased onto the coarse histogram, so it must fit "
              "both -- which is why the gate is a *budget* test: on a device "
              "that cannot afford this arena the two do not overlap");
static_assert((size_t)kCoarseBins * sizeof(int32_t) <= kSmemBytes,
              "the coarse histogram is written into the arena before the "
              "candidates are, so the arena must hold 4096 32-bit bins -- and "
              "it does, with nothing to spare (16 KB is exactly 4096 bins and "
              "exactly the two 2048-entry candidate halves)");

constexpr float kNegativeInfinity = -__builtin_huge_valf();

// The NaN predicate `maca_topk.cu`'s row contract uses, in this file's own
// namespace so the port stays self-contained: `radix_core.cuh` -- the only
// header this is included beside -- defines it inside `deep_select_maca`, and
// this file is included *before* that definition, so a call to the outer name
// here would not resolve.  Same expression, `v != v` spelled out in bits
// because the build enables `--use_fast_math`.
__device__ __forceinline__ bool is_nan_value(float v)
{
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

__device__ __forceinline__ uint32_t refine_bin(float value)
{
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// `warp_cumsum_histogram<kRadix>` at `kRadix == kFineBins == 256`, which is
// the **fine** arm of deep_gemm's helper -- not the 1024-bin one.
//
// Which arm matters, and getting it wrong is not a tuning difference: the
// coarse arm indexes `histogram[tid]` over *every* thread in the block, so at
// 640 threads it walks 383 words past a `[257]` array and rewrites `counter`,
// `threshold_bin_id` and the rest of the block's shared state on its way.  A
// first version of this file did exactly that (it read the helper's shape off
// the `kCoarseBins == 1024` instantiation), and the symptom was a threshold of
// 3073 -- a 12-bit bin id, which cannot exist -- followed by an illegal shared
// address.  The fine arm touches only lanes 0..63 for `Bins == 256`, and
// `static_assert`s below keep that arithmetic tied to `kRadix` rather than to
// the 64 it was written for.
//
// Shape: each of the 64 lanes owns `kItemsPerLane` consecutive bins, suffix-
// sums them in registers, then one warp shuffle pass gives each lane the total
// of the lanes above it.  The result is the same suffix sum over all 256 bins
// the original produces; the ordering of the three steps is the original's.
__device__ __forceinline__ void warp_cumsum_histogram_256(int (&histogram)[kRadix + kHistogramPadding])
{
    constexpr int kWarpSize = 64;
    constexpr uint64_t kWarpMask = 0xffffffffffffffffULL;
    constexpr int kItemsPerLane = kRadix / kWarpSize;
    static_assert(kRadix % kWarpSize == 0 && kRadix / kWarpSize <= kWarpSize);

    const int tid = threadIdx.x;
    if (tid < kWarpSize) {
        const int lane = tid;
        int values[kItemsPerLane];
#pragma unroll
        for (int item = 0; item < kItemsPerLane; ++item)
            values[item] = histogram[lane * kItemsPerLane + item];
#pragma unroll
        for (int item = kItemsPerLane - 2; item >= 0; --item)
            values[item] += values[item + 1];

        const int lane_total = values[0];
        int warp_suffix = lane_total;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int other = __shfl_down_sync(kWarpMask, warp_suffix, offset, kWarpSize);
            if (lane + offset < kWarpSize) warp_suffix += other;
        }
        const int lane_offset = warp_suffix - lane_total;

#pragma unroll
        for (int item = 0; item < kItemsPerLane; ++item)
            histogram[lane * kItemsPerLane + item] = values[item] + lane_offset;
    }
    __syncthreads();
}

// One row of `[0, length)`, ranking `requested_topk` of it.
//
// `output` receives *positions in this window*, which is the row's own columns
// on this path (the window always starts at column 0 here -- see the contract
// half in `maca_topk.cu`).  Ties are broken by whichever order the radix
// passes produce, exactly as in the original: the caller compares as
// multisets, not elementwise.
//
// Returns false when the shape is outside what the arena can serve, in which
// case nothing is written and the caller must take the row path.  deep_gemm
// asserts instead (its dispatch never picks coarse12 there); returning makes
// the bound checkable at the call site rather than a trap in the field.
__device__ __forceinline__ bool topk_coarse12_row(
    const float *__restrict__ input, int32_t *__restrict__ output,
    int length, int requested_topk, int32_t *__restrict__ nan_flag)
{
    const int tid = threadIdx.x;
    const unsigned int u_length = static_cast<unsigned int>(length);

    // The arena does double duty: the first pass writes a 4096-wide coarse
    // histogram into it, the second overwrites it with candidates.  The two
    // never overlap in time, which is why one buffer is enough and why the
    // coarse histogram is `kCoarseBins` words of a `kCandidateCapacity`-word
    // region.
    extern __shared__ int shared_arena[];
    int *wide_histogram = shared_arena;
    int *candidate_indices = shared_arena;

    __shared__ int histogram[kRadix + kHistogramPadding];
    __shared__ int counter;
    __shared__ int threshold_bin_id;
    __shared__ int threshold_exclusive_count;
    __shared__ int num_input[2];
    __shared__ int last_remain;

    // 16-byte alignment prefix/tail, as the original walks it.  Keeping this
    // identical matters: the whole point of the port is that the two
    // implementations differ in their *design*, not in how they read memory.
    const auto address = reinterpret_cast<uintptr_t>(input);
    const int prefix_unclamped =
        static_cast<int>((alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
    const int prefix =
        (address & (alignof(float4) - 1)) == 0 ? 0 : min(length, prefix_unclamped);
    const int vector_length = (length - prefix) / 4;
    const int tail = prefix + vector_length * 4;
    const float4 *vector_input = reinterpret_cast<const float4 *>(input + prefix);

    for (int bin = tid; bin < kCoarseBins; bin += kThreads) wide_histogram[bin] = 0;
    __syncthreads();

    // Pass 1, with the contract half's NaN scan folded into it: one predicate
    // per element on words this kernel is already loading, against a whole
    // separate pass over the row (`nan_scan_kernel` on the split, or the
    // contract half's own scan otherwise).  The same fusion on the split's pass
    // 1 measured ~3% of that pass (ledger §10.7), which is the cost this buys
    // back -- and without it this route cannot be entered under `check_nan` at
    // all: the contract half's preselected arm reads this row's flag instead of
    // scanning, and it has to be a slot someone wrote.
    //
    // `nan_flag` is this row's slot in the caller's per-row table, zeroed by
    // the caller; this only ever raises it.  It is raised before the answer is
    // written, so the contract half -- which runs after this kernel -- sees the
    // flag and takes its own NaN path (trap, or the `0x3F3F3F3F` guard) exactly
    // as it would have had it scanned the row itself.
    bool nan_found = false;
    for (int index = tid; index < prefix; index += kThreads) {
        const float value = __ldg(input + index);
        nan_found |= is_nan_value(value);
        atomicAdd(&wide_histogram[refine_bin(value) >> kFirstShift], 1);
    }
    for (int vec = tid; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        nan_found |= is_nan_value(values.x) || is_nan_value(values.y)
                     || is_nan_value(values.z) || is_nan_value(values.w);
        atomicAdd(&wide_histogram[refine_bin(values.x) >> kFirstShift], 1);
        atomicAdd(&wide_histogram[refine_bin(values.y) >> kFirstShift], 1);
        atomicAdd(&wide_histogram[refine_bin(values.z) >> kFirstShift], 1);
        atomicAdd(&wide_histogram[refine_bin(values.w) >> kFirstShift], 1);
    }
    for (int index = tail + tid; index < length; index += kThreads) {
        const float value = __ldg(input + index);
        nan_found |= is_nan_value(value);
        atomicAdd(&wide_histogram[refine_bin(value) >> kFirstShift], 1);
    }
    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {
        if (tid == 0) atomicOr(nan_flag, 1);
    }
    __syncthreads();

    // The 4096-bin histogram folds into a 256-bin one *in the same words*: bin
    // `b` of the coarse level sums the 16 sub-bins that share its high byte,
    // which is what makes the next step a plain 8-bit threshold.  8 bits is
    // also where the refine's own arithmetic lines up (`kRefineShift0 = 12`),
    // so no separate "wide bin" bookkeeping is needed past this point.
    if (tid < kRadix) {
        int count = 0;
#pragma unroll
        for (int sub_bin = 0; sub_bin < kWideGroupSize; ++sub_bin)
            count += wide_histogram[tid * kWideGroupSize + sub_bin];
        histogram[tid] = count;
    }
    else if (tid == kRadix) {
        histogram[tid] = 0;
    }
    __syncthreads();

    warp_cumsum_histogram_256(histogram);
    if (tid < kRadix && histogram[tid] > requested_topk
        && histogram[tid + 1] <= requested_topk) {
        threshold_bin_id = tid;
        threshold_exclusive_count = histogram[tid + 1];
    }
    __syncthreads();

    // Thread 0 narrows the 8-bit threshold to a 12-bit one: within the chosen
    // high byte, walk sub-bins from the top until the cumulative count would
    // pass the rank.  This is the step that turns "the bin holds 3000" into
    // "the bin holds 210" before a single candidate is touched.
    if (tid == 0) {
        const int high8_bin = threshold_bin_id;
        const int high8_exclusive = threshold_exclusive_count;
        const int remain = requested_topk - high8_exclusive;
        int sub_exclusive = 0;
        for (int sub_bin = kWideGroupSize - 1; sub_bin >= 0; --sub_bin) {
            const int count = wide_histogram[high8_bin * kWideGroupSize + sub_bin];
            if (sub_exclusive + count > remain) {
                threshold_bin_id = high8_bin * kWideGroupSize + sub_bin;
                threshold_exclusive_count = high8_exclusive + sub_exclusive;
                break;
            }
            sub_exclusive += count;
        }
        num_input[0] = 0;
        counter = 0;
    }
    __syncthreads();

    const int wide_threshold = threshold_bin_id;
    int topk = requested_topk - threshold_exclusive_count;
    if (topk == 0) {
        // Every slot is above the threshold, so this is a pure filter pass.
        for (unsigned int index = tid; index < u_length; index += kThreads) {
            if (static_cast<int>(refine_bin(__ldg(input + index)) >> kFirstShift) > wide_threshold) {
                const int position = atomicAdd(&counter, 1);
                output[position] = static_cast<int32_t>(index);
            }
        }
        __syncthreads();
        return true;
    }

    if (tid < kRadix + 1) histogram[tid] = 0;
    __syncthreads();

    // Pass 2.  Explicitly expanded rather than a lambda: deep_gemm measured
    // the capturing-lambda form 1.7-2.2% slower on large batches on MetaX and
    // left the note at the original site.  Keeping the macro keeps the port
    // byte-for-byte in what it generates.
#define DG12_COLLECT(value, index)                                                       \
    do {                                                                                 \
        const uint32_t key = refine_bin(value);                                          \
        const int bin = key >> kFirstShift;                                              \
        if (bin > wide_threshold) {                                                      \
            const int position = atomicAdd(&counter, 1);                                 \
            output[position] = static_cast<int32_t>(index);                              \
        }                                                                                \
        else if (bin == wide_threshold) {                                                \
            const int position = atomicAdd(&num_input[0], 1);                            \
            if (position < kCandidateCapacity) {                                         \
                candidate_indices[position] = static_cast<int>(index);                   \
                atomicAdd(&histogram[(key >> kRefineShift0) & 0xffu], 1);                \
            }                                                                            \
        }                                                                                \
    } while (0)

    for (int index = tid; index < prefix; index += kThreads)
        DG12_COLLECT(__ldg(input + index), index);
    for (int vec = tid; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        const int index = prefix + vec * 4;
        DG12_COLLECT(values.x, index);
        DG12_COLLECT(values.y, index + 1);
        DG12_COLLECT(values.z, index + 2);
        DG12_COLLECT(values.w, index + 3);
    }
    for (int index = tail + tid; index < length; index += kThreads)
        DG12_COLLECT(__ldg(input + index), index);
#undef DG12_COLLECT
    __syncthreads();

    // The overflow handler: when the 12-bit bin does not fit the arena, the
    // row is ranked again rather than truncated.  Three rounds of
    // histogram-then-narrow over whole-row re-reads, each round consuming 8,
    // 8 and 4 more key bits, so the prefix is exact after the third for any
    // input.  This is the branch that makes the algorithm correct on shapes
    // where the coarse bin is huge; the sizing question ("does it fire on the
    // shapes we care about") is measured, not assumed.
    if (num_input[0] > kCandidateCapacity) {
        int selected_prefix = wide_threshold;
        int prefix_bits = kCoarseBits;
#pragma unroll 3
        for (int round = 0; round < 3; ++round) {
            if (tid < kRadix + 1) histogram[tid] = 0;
            __syncthreads();

            const int radix_bits = round == 2 ? kFinalRadixBits : 8;
            const int offset = round == 0 ? kRefineShift0 : (round == 1 ? kRefineShift1 : 0);
            const int mask = (1 << radix_bits) - 1;
            for (unsigned int index = tid; index < u_length; index += kThreads) {
                const uint32_t key = refine_bin(__ldg(input + index));
                if (static_cast<int>(key >> (32 - prefix_bits)) != selected_prefix) continue;
                atomicAdd(&histogram[(key >> offset) & mask], 1);
            }
            __syncthreads();

            warp_cumsum_histogram_256(histogram);
            if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
                threshold_bin_id = tid;
                threshold_exclusive_count = histogram[tid + 1];
                last_remain = topk - histogram[tid + 1];
            }
            __syncthreads();

            const int threshold_bin = threshold_bin_id;
            topk -= threshold_exclusive_count;
            for (unsigned int index = tid; index < u_length; index += kThreads) {
                const uint32_t key = refine_bin(__ldg(input + index));
                if (static_cast<int>(key >> (32 - prefix_bits)) != selected_prefix) continue;
                const int bin = (key >> offset) & mask;
                if (bin > threshold_bin) {
                    const int position = atomicAdd(&counter, 1);
                    output[position] = static_cast<int32_t>(index);
                }
                else if (round == 2 && bin == threshold_bin) {
                    const int position = atomicAdd(&last_remain, -1);
                    if (position > 0) output[requested_topk - position] = static_cast<int32_t>(index);
                }
            }
            __syncthreads();
            if (topk == 0 || round == 2) return true;
            selected_prefix = (selected_prefix << radix_bits) | threshold_bin;
            prefix_bits += radix_bits;
        }
        return true;
    }

    // The arena path: three rounds over the *staged candidates only*, ping-
    // ponging between the two halves of the arena.  Because the coarse level
    // consumed 12 bits and the three rounds consume 8 + 8 + 4, the full 32-bit
    // key is resolved exactly at the end, so the tail slots fill from the last
    // byte's ties the same way the arena path above fills them.
#pragma unroll 3
    for (int round = 0; round < 3; ++round) {
        const int read = round & 1;
        const int current_offset = read * kCandidateCapacity;
        const int next_offset = (read ^ 1) * kCandidateCapacity;
        const int candidate_count = num_input[read];

        warp_cumsum_histogram_256(histogram);
        if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
            threshold_bin_id = tid;
            threshold_exclusive_count = histogram[tid + 1];
            num_input[read ^ 1] = 0;
            last_remain = topk - histogram[tid + 1];
        }
        __syncthreads();

        const int threshold_bin = threshold_bin_id;
        topk -= threshold_exclusive_count;
        const int offset = round == 0 ? kRefineShift0 : (round == 1 ? kRefineShift1 : 0);
        const int mask = round == 2 ? kFinalRadixMask : 0xff;
        if (topk == 0) {
            for (int i = tid; i < candidate_count; i += kThreads) {
                const int index = candidate_indices[current_offset + i];
                const int bin = (refine_bin(__ldg(input + index)) >> offset) & mask;
                if (bin > threshold_bin) {
                    const int position = atomicAdd(&counter, 1);
                    output[position] = static_cast<int32_t>(index);
                }
            }
            __syncthreads();
            return true;
        }

        if (tid < kRadix + 1) histogram[tid] = 0;
        __syncthreads();
        for (int i = tid; i < candidate_count; i += kThreads) {
            const int index = candidate_indices[current_offset + i];
            const uint32_t key = refine_bin(__ldg(input + index));
            const int bin = (key >> offset) & mask;
            if (bin > threshold_bin) {
                const int pos = atomicAdd(&counter, 1);
                output[pos] = static_cast<int32_t>(index);
            }
            else if (bin == threshold_bin) {
                if (round == 2) {
                    const int pos = atomicAdd(&last_remain, -1);
                    if (pos > 0) output[requested_topk - pos] = static_cast<int32_t>(index);
                }
                else {
                    const int position = atomicAdd(&num_input[read ^ 1], 1);
                    if (position < kCandidateCapacity) {
                        candidate_indices[next_offset + position] = index;
                        const int next_offset_bits = round == 0 ? kRefineShift1 : 0;
                        const int next_mask = round == 0 ? 0xff : kFinalRadixMask;
                        atomicAdd(&histogram[(key >> next_offset_bits) & next_mask], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
    return true;
}

// One CTA per row.  `__launch_bounds__` is 640 for the same reason the
// original chose it: the kernel's widest cooperative step is the 4096-bin
// histogram clear and the candidate walk, and 640 is where deep_gemm measured
// the balance between those and register pressure on this device.
__global__ __launch_bounds__(kThreads) void topk_coarse12_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    int32_t *__restrict__ out, int32_t *__restrict__ nan_flags, int topk, int B,
    int64_t stride, int default_length)
{
    const int row = blockIdx.x;
    if (row >= B) return;

    // `lengths` is the per-row table when the caller has one and `default_length`
    // is the scanned width when it does not.  **Not `stride`**: the row stride
    // is a property of the tensor, and a caller that hands over a column slice
    // of a wider matrix has a stride larger than the vocabulary it wants ranked.
    // Reading the stride as the length is invisible whenever the two agree (a
    // full-width tensor), which is why this was wrong for a while and why the
    // fallback is now an argument the caller has to supply.
    const int length =
        lengths == nullptr ? default_length : __ldg(lengths + row);
    const float *row_input = scores + (int64_t)row * stride;
    int32_t *row_out = out + (int64_t)row * topk;
    // One slot per row, zeroed by the caller.  Null means "the caller is not
    // scanning at all" (`check_nan` false), which is the only case where the
    // contract half does not read the table either.
    int32_t *row_nan = nan_flags == nullptr ? nullptr : nan_flags + row;

    if (length <= topk) {
        // Same padding contract the row kernel uses: the visible prefix is the
        // answer, the rest is -1.  The caller's contract half re-derives this
        // for itself, so this arm only has to leave the buffer well-formed --
        // and it does not scan, because the contract half does not scan a row
        // whose window is no longer than `topk` either.
        for (int i = threadIdx.x; i < topk; i += kThreads)
            row_out[i] = i < length ? i : -1;
        return;
    }

    const bool served = topk_coarse12_row(row_input, row_out, length, topk, row_nan);
    if (!served) {
        for (int i = threadIdx.x; i < topk; i += kThreads) row_out[i] = -1;
    }
}

inline cudaError_t launch_topk_coarse12(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream)
{
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_coarse12_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)kSmemBytes);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_coarse12_kernel<<<B, kThreads, kSmemBytes, stream>>>(
        scores, lengths, out, nan_flags, topk, B, stride, default_length);
    return cudaGetLastError();
}

}  // namespace dg12
}  // namespace rk
