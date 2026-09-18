// deep_gemm's `topk_chunks`, ported for the b <= 2 band.
//
// Why this file exists
// -------------------
// `radix_topk_row_f32` (radix_core.cuh) and `topk_coarse12`
// (`dg_coarse12.cuh`) both serve this operator, and neither is the right shape
// at one or two rows.  Measured, interleaved five rounds, device 2, `k = 2048`
// (ledger §12.16.6):
//
//     b     V        ours     deep_gemm    dg/ours
//     1     66551     90.8      53.0        0.584x
//     2     66551     93.4      54.1        0.580x
//     4     66551     93.8     360.8        3.844x   <- deep_gemm's own cliff
//     1    107520    174.8      60.6        0.347x
//     1    131072    135.0      62.7        0.464x
//
// So the gap is real, it is confined to `b <= 2`, and it is **not** the chunk
// count: sweeping `DEEP_SELECT_F32_CHUNKS` from 2 to 32 at `b1 V=66551` moves
// ours between 89.5 and 135.6 with the optimum at 5-8, and the auto rule
// already picks 8.  What deep_gemm has instead is a **structural** difference
// (`mcDeepGEMM/csrc/kernels/fp32_topk.cu`, `topk_chunks_compact_refine`): it
// walks the row a second time and, in that same walk, writes every element
// above the threshold bin **straight to the output** (the `guaranteed` half),
// staging only the threshold bin's own members for refinement.  The split
// instead materializes `num_chunks * topk` candidates and merges them, and at
// `k = 2048, V = 66551` the elements above the threshold bin are already
// almost the whole answer -- so the merge is the entire stage-2 cost (43.9 of
// the 78.3 us, ledger §12.16.6).
//
// What this port is, and what it is not
// -------------------------------------
// The arithmetic -- `coarse_bin`, `refine_bin`, the 1024-bin coarse level, the
// 256-bin fine level, the four refine bytes, the three rounds -- is deep_gemm's,
// verbatim.  What is dropped:
//
//   * the `Transform` page-table half, which needs the `page_table` /
//     `cu_seqlens_row` / `q_positions` plumbing this repository does not
//     synthesize (the same subtraction `ref/` and `dg_coarse12.cuh` make);
//   * the **multi-CTA chunking**.  deep_gemm splits a row across
//     `NChunks in [3, 6]` CTAs with a per-row workspace, an `arrival` counter
//     and a cross-CTA merge.  This port is one CTA per row, so it needs no
//     workspace, no arrival protocol and no per-row allocation -- and at
//     `b <= 2` the chunking buys at most 6 CTAs on a 104-AP part, which is
//     the one thing it cannot be buying.  The second pass over the row is
//     unchanged either way: deep_gemm reads each chunk twice too, so the two
//     have the same traffic.
//
// The `rescanned` branch is also dropped: deep_gemm re-walks the row when the
// threshold bin is wider than `kCandidateCapacity`, and this port **returns
// false** instead, which the caller turns into the per-row `-1` the contract
// half already understands -- that row is then re-ranked by the row path.  It
// changes coverage, not correctness, and the branch it removes only fires on a
// bin wider than 4096, which `randn` at these widths does not produce (the
// same measurement `kF32OverflowChunkLen` rests on, `maca_topk.cu`).

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace rk {
namespace dgchunks {

// ── deep_gemm's `detail` constants, verbatim ───────────────────────────────
constexpr int kThreads = 1024;
constexpr int kMaxTopK = 2048;
constexpr int kCoarseBins = 1024;
constexpr int kFineBins = 256;
constexpr int kCandidateCapacity = 4096;
constexpr int kHistogramPadding = 1;
constexpr size_t kSmemBytes = 2 * (size_t)kCandidateCapacity * sizeof(int32_t);
constexpr float kNegativeInfinity = -__builtin_huge_valf();

static_assert(kFineBins + kHistogramPadding <= kThreads,
              "the fine histogram is cleared and cumsum'd by one thread per "
              "bin, so the block has to cover it");
static_assert(kCoarseBins <= kThreads,
              "same, for the coarse histogram");
static_assert(kCandidateCapacity >= kMaxTopK,
              "the staged candidates must be able to hold a whole top-k "
              "answer -- which is what lets the `remaining == 0` arm write the "
              "boundary out of the staging buffer without a second walk");

// `dg_coarse12.cuh`'s predicate, in this file's own namespace: `radix_core.cuh`
// defines its copy inside `deep_select_maca` and this header is included before
// that, so the outer name would not resolve here.  Same expression, `v != v`
// spelled out in bits because the build enables `--use_fast_math`.
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

// deep_gemm's `coarse_bin`, verbatim: the top 10 bits of the **half**-encoded
// key.  Half, not float -- the coarse level only has to put the threshold in
// the right neighbourhood, and the refine below resolves within the bin on the
// exact float bits.  `coarse_bins_pair` is its float2 form; it is not used
// here because this port reads `float4`, so the four lanes go through
// `coarse_bin` one at a time exactly as `topk_chunks_coarse_hist` does for its
// vector tail.
__device__ __forceinline__ int coarse_bin(float value)
{
    const __half half_value = __float2half_rn(value);
    const uint16_t bits = __half_as_ushort(half_value);
    const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                          : static_cast<uint16_t>(bits | 0x8000u);
    return static_cast<int>(key >> 6);
}

// `warp_cumsum_histogram<Bins>` at `Bins == 256`, i.e. deep_gemm's **fine** arm
// -- not the 1024-bin one.  The distinction is not cosmetic and
// `dg_coarse12.cuh` already paid for learning it: the coarse arm indexes
// `histogram[tid]` over every thread in the block, so at 1024 threads it walks
// past a `[257]` array and rewrites the block's other shared state on the way.
// The fine arm touches only lanes 0..63 for `Bins == 256`.
//
// Shape: each of the 64 lanes owns `kItemsPerLane` consecutive bins,
// suffix-sums them in registers, then one warp shuffle pass gives each lane the
// total of the lanes above it.
__device__ __forceinline__ void warp_cumsum_histogram_256(
    int (&histogram)[kFineBins + kHistogramPadding])
{
    constexpr int kWarpSize = 64;
    constexpr uint64_t kWarpMask = 0xffffffffffffffffULL;
    constexpr int kItemsPerLane = kFineBins / kWarpSize;
    static_assert(kFineBins % kWarpSize == 0 && kFineBins / kWarpSize <= kWarpSize);

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

// The same suffix sum over the 1024-bin coarse level, run by the whole block:
// bins are strided across `kThreads` and the scan is a straight shared-memory
// two-level walk rather than a warp shuffle, because 1024 bins over 64 lanes
// would need 16 items per lane and the extra registers buy nothing here (this
// runs once per row).
__device__ __forceinline__ void block_cumsum_histogram_1024(
    int (&histogram)[kCoarseBins + kHistogramPadding])
{
    const int tid = threadIdx.x;
    if (tid == 0) histogram[kCoarseBins] = 0;
    __syncthreads();
    for (int offset = 1; offset < kCoarseBins; offset <<= 1) {
        const int value = tid < kCoarseBins - offset ? histogram[tid + offset] : 0;
        __syncthreads();
        if (tid < kCoarseBins - offset) histogram[tid] += value;
        __syncthreads();
    }
}

// One row of `[0, length)`, ranking `requested_topk` of it.
//
// `output` receives *positions in this window*, which on this path is the row's
// own columns (the window always starts at column 0 here), so the caller's
// contract half can treat them exactly like the split's or `dg12`'s answer.
// Ties are broken by whichever order the passes produce, exactly as in the
// original: the caller compares as multisets.
//
// Returns false when the row is outside what this kernel serves -- a window no
// longer than `topk`, or a threshold bin wider than `kCandidateCapacity`.  The
// caller turns that into the per-row `-1` sentinel and the row path re-ranks
// it, so a `false` is a coverage statement, never a wrong answer.
__device__ __forceinline__ bool topk_chunks_row(
    const float *__restrict__ input, int32_t *__restrict__ output,
    int length, int requested_topk, int32_t *__restrict__ nan_flag)
{
    const int tid = threadIdx.x;

    // deep_gemm splits the row across `NChunks in [3, 6]` CTAs and merges
    // through a per-row workspace; this port is one CTA per row, so the
    // candidate array is the only thing that needs the arena.
    extern __shared__ int candidate_indices[];
    __shared__ int histogram[kFineBins + kHistogramPadding];
    __shared__ int wide_histogram[kCoarseBins + kHistogramPadding];
    __shared__ int counter;
    __shared__ int num_input;
    __shared__ int threshold_bin_id;
    __shared__ int threshold_exclusive_count;
    __shared__ int last_remain;

    // 16-byte alignment prefix/tail, as the original walks it.
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

    // Pass 1: the 1024-bin coarse histogram, with the contract half's NaN scan
    // folded into it.  `dg_coarse12.cuh` does the same fold for the same
    // reason: without it this route cannot be entered under `check_nan` at all,
    // because the contract half's preselected arm reads this row's flag instead
    // of scanning, and the flag has to be a slot someone wrote.
    bool nan_found = false;
    for (int index = tid; index < prefix; index += kThreads) {
        const float value = __ldg(input + index);
        nan_found |= is_nan_value(value);
        atomicAdd(&wide_histogram[coarse_bin(value)], 1);
    }
    for (int vec = tid; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        nan_found |= is_nan_value(values.x) || is_nan_value(values.y)
                     || is_nan_value(values.z) || is_nan_value(values.w);
        atomicAdd(&wide_histogram[coarse_bin(values.x)], 1);
        atomicAdd(&wide_histogram[coarse_bin(values.y)], 1);
        atomicAdd(&wide_histogram[coarse_bin(values.z)], 1);
        atomicAdd(&wide_histogram[coarse_bin(values.w)], 1);
    }
    for (int index = tail + tid; index < length; index += kThreads) {
        const float value = __ldg(input + index);
        nan_found |= is_nan_value(value);
        atomicAdd(&wide_histogram[coarse_bin(value)], 1);
    }
    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {
        if (tid == 0) atomicOr(nan_flag, 1);
    }
    __syncthreads();

    // The threshold, by suffix sum over the coarse level: `wide_histogram[b]`
    // becomes the count of elements **at or above** bin `b`, which is what makes
    // `[b] >= topk && [b+1] < topk` the bin holding rank `topk`.  deep_gemm
    // scans the same array with a warp-level helper; this port scans it in
    // place with the whole block, which at 1024 bins over 1024 threads is the
    // same work without the shuffle rounds.
    block_cumsum_histogram_1024(wide_histogram);
    if (tid < kCoarseBins && wide_histogram[tid] >= requested_topk
        && wide_histogram[tid + 1] < requested_topk) {
        threshold_bin_id = tid;
        threshold_exclusive_count = wide_histogram[tid + 1];
    }
    if (tid == 0) {
        counter = 0;
        num_input = 0;
    }
    __syncthreads();

    const int wide_threshold = threshold_bin_id;
    int remaining = requested_topk - threshold_exclusive_count;
    if (remaining <= 0) {
        // Every slot is above the threshold bin, so pass 2's `guaranteed` half
        // already wrote exactly `requested_topk` indices and there is nothing
        // left to rank.  **No extra pass**: one would re-select the same
        // elements and write them a second time.
        return true;
    }

    // The candidates live in a **two-buffer** arena and every stage ping-pongs
    // between the halves, as deep_gemm's `staged_indices[read_buffer]` does.
    // Compacting in place would be a race -- one thread reads `pos` while
    // another overwrites it -- and the symptom is not a slow answer but a wrong
    // one.
    int *const staging_a = candidate_indices;
    int *const staging_b = candidate_indices + kCandidateCapacity;
    // Which half each stage reads and writes.  Pass 2 stages into `staging_a`,
    // so the first refine stage reads it and writes `staging_b`, and the two
    // swap from there.
    int *read_buf = staging_a;
    int *write_buf = staging_b;

    if (tid < kFineBins + kHistogramPadding) histogram[tid] = 0;
    if (tid == 0) num_input = 0;
    __syncthreads();

    // Pass 2.  **The whole point of this file**: the walk that stages the
    // threshold bin is the same walk that writes everything above it, so the
    // `guaranteed` half of the answer costs no merge at all.  deep_gemm left a
    // note at the original site that the capturing-lambda form measured
    // 1.7-2.2% slower on MetaX and the macro is kept for that reason.
    //
    // The histogram this fills is the **first** refine byte (bits 31..24), and
    // the stages below advance one byte at a time: byte 16, then 8, then 0.
    // Keeping that alignment is what makes the threshold and the bins it is
    // compared against live in the same byte -- get it wrong and every element
    // lands on the wrong side of a threshold that belongs to a different byte,
    // which reads as an answer full of one repeated index rather than as a
    // crash.
#define DGCH_COLLECT(value, index)                                                    \
    do {                                                                              \
        const uint32_t key = refine_bin(value);                                       \
        const int bin = coarse_bin(value);                                            \
        if (bin > wide_threshold) {                                                   \
            const int position = atomicAdd(&counter, 1);                              \
            output[position] = static_cast<int32_t>(index);                           \
        }                                                                             \
        else if (bin == wide_threshold) {                                             \
            const int position = atomicAdd(&num_input, 1);                            \
            if (position < kCandidateCapacity) {                                      \
                staging_a[position] = static_cast<int>(index);                        \
                atomicAdd(&histogram[key >> 24], 1);                                  \
            }                                                                         \
        }                                                                             \
    } while (0)

    for (int index = tid; index < prefix; index += kThreads)
        DGCH_COLLECT(__ldg(input + index), index);
    for (int vec = tid; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        const int index = prefix + vec * 4;
        DGCH_COLLECT(values.x, index);
        DGCH_COLLECT(values.y, index + 1);
        DGCH_COLLECT(values.z, index + 2);
        DGCH_COLLECT(values.w, index + 3);
    }
    for (int index = tail + tid; index < length; index += kThreads)
        DGCH_COLLECT(__ldg(input + index), index);
#undef DGCH_COLLECT
    __syncthreads();

    // The one branch deep_gemm has that this port does not: its `rescanned` arm
    // re-walks the row when the threshold bin is wider than the arena, which
    // costs it extra whole-row passes.  Here the row is simply declined, and
    // the caller's per-row `-1` sends it to the row path -- a coverage
    // statement rather than a wrong answer.  `num_input` is the true bin width,
    // known before any member was staged.
    if (num_input > kCandidateCapacity) return false;
    int count = num_input;

    // The first refine byte: the threshold comes out of the histogram pass 2
    // filled, and the members that match it advance to the next byte.
    warp_cumsum_histogram_256(histogram);
    if (tid < kFineBins && histogram[tid] >= remaining && histogram[tid + 1] < remaining) {
        threshold_bin_id = tid;
        last_remain = remaining - histogram[tid + 1];
    }
    __syncthreads();
    int fine_threshold = threshold_bin_id;
    remaining -= histogram[fine_threshold + 1];

    if (remaining == 0) {
        // This byte alone decides the boundary: every staged candidate above it
        // is part of the answer, and none below is.  `counter` is already the
        // `guaranteed` count, so they append directly behind it.
        for (int pos = tid; pos < count; pos += kThreads) {
            const int column = read_buf[pos];
            if (static_cast<int>(refine_bin(input[column]) >> 24) > fine_threshold) {
                const int out = atomicAdd(&counter, 1);
                output[out] = column;
            }
        }
        __syncthreads();
        return true;
    }

    // Advance to byte 16 and run the three refine rounds, which consume 8, 8 and
    // 4 more key bits -- so the prefix is exact after the third for any input.
    // deep_gemm peels the byte-16 stage out of the loop because that stage reads
    // the *whole* staged list rather than the previous stage's output; from
    // there on each round reads what the one before it wrote.
    if (tid < kFineBins + kHistogramPadding) histogram[tid] = 0;
    if (tid == 0) num_input = 0;
    __syncthreads();
    for (int pos = tid; pos < count; pos += kThreads) {
        const int column = read_buf[pos];
        const uint32_t key = refine_bin(input[column]);
        const int bin = static_cast<int>(key >> 24);
        if (bin > fine_threshold) {
            const int out = atomicAdd(&counter, 1);
            output[out] = column;
        }
        else if (bin == fine_threshold) {
            const int put = atomicAdd(&num_input, 1);
            if (put < kCandidateCapacity) {
                write_buf[put] = column;
                atomicAdd(&histogram[(key >> 16) & 0xffu], 1);
            }
        }
    }
    __syncthreads();
    { int *const t = read_buf; read_buf = write_buf; write_buf = t; }
    count = num_input;
    if (count == 0) return true;

    for (int round = 0; round < 3; ++round) {
        const int offset = 16 - round * 8;
        warp_cumsum_histogram_256(histogram);
        if (tid < kFineBins && histogram[tid] >= remaining
            && histogram[tid + 1] < remaining) {
            threshold_bin_id = tid;
            last_remain = remaining - histogram[tid + 1];
        }
        __syncthreads();
        fine_threshold = threshold_bin_id;
        remaining -= histogram[fine_threshold + 1];

        if (remaining == 0) {
            // Nothing left to narrow: the members still above this byte are the
            // rest of the answer.
            for (int pos = tid; pos < count; pos += kThreads) {
                const int column = read_buf[pos];
                if (static_cast<int>((refine_bin(input[column]) >> offset) & 0xffu)
                    > fine_threshold) {
                    const int out = atomicAdd(&counter, 1);
                    output[out] = column;
                }
            }
            __syncthreads();
            return true;
        }

        if (tid < kFineBins + kHistogramPadding) histogram[tid] = 0;
        if (tid == 0) num_input = 0;
        __syncthreads();
        for (int pos = tid; pos < count; pos += kThreads) {
            const int column = read_buf[pos];
            const uint32_t key = refine_bin(input[column]);
            const int bin = (key >> offset) & 0xffu;
            if (bin > fine_threshold) {
                const int out = atomicAdd(&counter, 1);
                output[out] = column;
            }
            else if (bin == fine_threshold) {
                if (round == 2) {
                    // The last byte has no further histogram to feed: the
                    // remaining slots are filled by whichever members arrive,
                    // which is a valid tie-break because every one of them
                    // carries the same value.
                    const int left = atomicAdd(&last_remain, -1);
                    if (left > 0) output[requested_topk - left] = column;
                }
                else {
                    const int put = atomicAdd(&num_input, 1);
                    if (put < kCandidateCapacity) {
                        write_buf[put] = column;
                        atomicAdd(&histogram[(key >> (offset - 8)) & 0xffu], 1);
                    }
                }
            }
        }
        __syncthreads();
        if (round == 2) break;
        { int *const t = read_buf; read_buf = write_buf; write_buf = t; }
        count = num_input;
        if (count == 0) return true;
    }
    return true;
}

__global__ void topk_chunks_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    int32_t *__restrict__ out, int32_t *__restrict__ nan_flags,
    int topk, int rows, int64_t stride, int default_length)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    const int32_t raw = lengths == nullptr ? default_length : __ldg(lengths + row);
    const int length = raw < 0 ? 0 : (raw > default_length ? default_length : raw);
    int32_t *const row_out = out + (int64_t)row * topk;
    int32_t *const row_nan = nan_flags == nullptr ? nullptr : nan_flags + row;
    const float *const row_input = scores + (int64_t)row * stride;

    if (length <= topk) {
        // The same padding contract the other arms use: the visible prefix is
        // the answer, the rest is -1, and the contract half re-derives this for
        // itself rather than trusting it.
        for (int i = threadIdx.x; i < topk; i += kThreads)
            row_out[i] = i < length ? i : -1;
        return;
    }

    if (!topk_chunks_row(row_input, row_out, length, topk, row_nan)) {
        // Declined: the per-row `-1` is what routes it to the row path.
        for (int i = threadIdx.x; i < topk; i += kThreads) row_out[i] = -1;
    }
}

inline cudaError_t launch_topk_chunks(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream)
{
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_chunks_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
            (int)kSmemBytes);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_chunks_kernel<<<B, kThreads, kSmemBytes, stream>>>(
        scores, lengths, out, nan_flags, topk, B, stride, default_length);
    return cudaGetLastError();
}

}  // namespace dgchunks
}  // namespace rk
