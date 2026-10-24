// deep_gemm's `topk_coarse12`, ported whole.
//
// Why this file exists
// -------------------
// `radix_topk_row_f32` (`radix_row_fp32.cuh`) and `topk_coarse12`
// (`mcDeepGEMM/csrc/kernels/fp32_topk.cu`) answer the same question with the
// same class of algorithm -- histogram the high bits, pick the bin holding
// rank `topk`, rank that bin -- and the ported one is **2.0-2.3x faster** on
// the shapes where deep_gemm routes to it (the ledger's §12 table).  Both are
// two row walks, so the walk count is not the difference; the differences
// that are measurable are:
//
//   * 12 coarse bits against 8, so the bin the refine ranks is 2-3x smaller.
//   * one 16 KB arena doing double duty (coarse histogram, then candidates)
//     against a 14 KB arena plus a ping-pong refine buffer.
//
// The block size is the original's 512 as well.  It was 640 until 2026-10-22,
// and 640 x 3 CTAs is 1920 threads -- under the 2048-thread/AP ceiling, so the
// AP ran 3 CTA/SM where 512 fills four, worth 3.7-6.2% on the fp32 cells with
// the bf16 controls flat (`docs/experiments/20261022_f32_occupancy/`).
//
// The port is deliberately an *extraction*, not a rewrite: the arithmetic,
// the round structure and the overflow handler below are deep_gemm's, so the
// two can be compared without asking which of them is better written.  What
// is dropped is the page-table transform (`Transform = true`), which needs
// the `page_table` / `cu_seqlens_row` / `q_positions` plumbing this
// repository does not synthesize -- the same subtraction the ported
// extraction made.
// What is added is the one thing deep_gemm keeps outside the kernel: this
// runs over `[0, length)` of the row, which on this path is the whole row,
// and it returns *positions in that window*, so the caller's contract half
// (`topk_kernel_radix`) can treat them exactly like the split's answer.
//
// Route selection lives in `routing.cuh` (`f32_coarse12_applies`) and has two
// halves.  The *budget* half -- does this part's shared memory
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

#include <csrc/structs.h>   // ARCH_REGS_PER_AP -- the OUT=2 walk gate
#include <csrc/maca_kernels/common/primitives.cuh>

// The contract half the answer is handed to (`launch_route` below), and through
// it the radix row kernel every route emits through.
#include <csrc/maca_kernels/common/contract.cuh>

namespace topk_select_coarse12 {

// ── deep_gemm's `detail` constants, verbatim ───────────────────────────────
//
// `kMaxTopK` is 2048 in the original and the arena below is sized from
// `kSmemBytes` (the original's `kCoarse12SmemBytes`, renamed here), so this
// port serves `topk <= 2048`.  That is also
// where deep_gemm's own dispatch stops consulting coarse12 for `top_k`, so
// the boundary is the original's rather than a new one.  Callers gate on it
// (`f32_coarse12_applies`), and this header asserts it rather than trusting
// the call site.
constexpr int kThreads = 512;
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

// The second pass reads a list, not the row.
//
// Pass 2 acts only on the elements whose 12-bit bin is at or above the
// threshold -- 2.6% of them at `b4096 V131072 k2048`, so >95% of that pass is
// lines it fetches and does nothing with.  Pass 1 therefore leaves each
// 4-float4 block's max bin in a per-row table, the threshold is compared
// against that table, and the blocks that survive are compacted into a list the
// walk strides over.  All of it mirrors the 16-bit row (`radix_row_bf16.cuh`),
// and the compaction is not optional: a test in front of the walk costs what
// the walk costs, because one kept block in a 64-lane warp puts every lane of
// that warp through the block body (measured on this part:
// `docs/experiments/20261022_f32_pass2_skip/` -- a table-gated walk 2.811 ms
// against the dense walk's 2.782, the same walk over a compacted list 1.489).
constexpr int kBlockElems = 16;                 // = 4 float4, the whole block
constexpr int kBlockVecs = kBlockElems / 4;     // one lane per float4
constexpr int kBlockLanes = 4;                  // lanes per block in the walk
// Past three quarters of the blocks the walk reads most of the row anyway and
// the list is traffic on top of it (the 16-bit row measured 0.63 of the dense
// walk at half the blocks and 1.21 at all of them); the count decides.
constexpr int kListMaxNumerator = 3;
constexpr int kListMaxDenominator = 4;

// The block's max bin from its four lanes' bins.  Two xor steps: a chained
// tree, not three shuffles off the same value (`radix_row_bf16.cuh`'s group max
// paid for that once -- a shuffle of the original gives the max over a subset).
//
// **Two spellings, and the split is what makes the mask honest.**  Pass 1's
// vector loop is per-lane, so its last iteration can run a *prefix* of the
// lanes and a four-lane group can straddle that edge; a full-mask shuffle there
// names lanes that are not executing it -- undefined, and measured: it dropped
// one block per row and mis-selected 5 slots of `b512 V998123 k2048`
// (docs/experiments/20261022_f32_pass2_skip/).  So the loop is split at
// `vector_length / kThreads`, the iteration count every lane runs: in that bulk
// the whole block is live and `block_max_bin` may use the constant mask, while
// the tail -- at most one iteration, a prefix of the lanes -- takes
// `block_max_bin_edge`, which shuffles on `__activemask()` and has a group that
// is not whole claim the maximum bin, so its block is always kept: it walks up
// to one block per row twice and can never drop one.  Guarding *every*
// iteration instead cost 1.4% of the fp32 grid (same note).
// The 16-bit row has no such boundary -- its loops are block-granular and enter
// a whole group or none of it.
__device__ __forceinline__ int block_max_bin(int b0, int b1, int b2, int b3) {
    int m = max(max(b0, b1), max(b2, b3));
    m = max(m, __shfl_xor_sync(~0ULL, m, 1));
    m = max(m, __shfl_xor_sync(~0ULL, m, 2));
    return m;
}

__device__ __forceinline__ int block_max_bin_edge(int b0, int b1, int b2, int b3) {
    int m = max(max(b0, b1), max(b2, b3));
    const unsigned long long live = __activemask();
    m = max(m, __shfl_xor_sync(live, m, 1));
    m = max(m, __shfl_xor_sync(live, m, 2));
    const int lane = static_cast<int>(threadIdx.x) & 63;
    const bool whole = ((live >> (lane & ~(kBlockLanes - 1))) & 0xfULL) == 0xfULL;
    return whole ? m : (kCoarseBins - 1);
}

// The NaN predicate, the key encode and the warp histogram suffix sum come from
// `../common/primitives.cuh`; unqualified here because every call site below
// is spelled that way.
using topk_select_common::is_nan_value;
using topk_select_common::refine_bin;
using topk_select_common::warp_cumsum_histogram;

// One row of `[0, length)`, ranking `requested_topk` of it.
//
// `output` receives *positions in this window*, which is the row's own columns
// on this path (the window always starts at column 0 here -- see the contract
// half in `contract.cuh`).  Ties are broken by whichever order the radix
// passes produce, exactly as in the original: the caller compares as
// multisets, not elementwise.
//
// **Always returns true, and every `return` below is that same `true`.**  The
// original's out-of-arena case is handled by the caller rather than here:
// `topk_coarse12_kernel` checks `length <= topk` (the padding contract) and
// `topk <= kMaxTopK` before calling, and falls back to a `-1`-filled row if
// the call declines -- a `served` variable that is constant today.  The
// `bool` is kept so that the day the arena bound moves back inside this
// function, the call site does not have to change.  deep_gemm asserts instead
// (its dispatch never picks coarse12 there).
__device__ __forceinline__ bool topk_coarse12_row(
    const float *__restrict__ input, int32_t *__restrict__ output,
    int length, int requested_topk, int32_t *__restrict__ nan_flag,
    uint16_t *__restrict__ block_max, uint32_t *__restrict__ block_list)
{
    const int tid = threadIdx.x;
    const unsigned int u_length = static_cast<unsigned int>(length);

    // The arena does double duty: the first pass writes a 4096-wide coarse
    // histogram into it, the second overwrites it with candidates.  One buffer
    // is enough because the **only** state live across that boundary is
    // `histogram[]` -- the 256-bin fold and the 12-bit narrow both finish with
    // `wide_histogram` before the first `candidate_indices` write, and nothing
    // reads the arena again after it.  The overflow handler below re-walks the
    // row with `candidate_indices` live, but it accumulates into `histogram[]`
    // and never touches the arena, which is what keeps that true.  The coarse
    // histogram is `kCoarseBins` words of a `kCandidateCapacity`-word region.
    extern __shared__ int shared_arena[];
    int *wide_histogram = shared_arena;
    int *candidate_indices = shared_arena;

    __shared__ int histogram[kRadix + kHistogramPadding];
    __shared__ int counter;
    __shared__ int kept_block_count;
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

    // The kept-block tables, already offset to this row by the caller (null
    // selects the dense walk).  The
    // summary covers the *vector* region only: the up-to-3 prefix floats, the
    // up-to-3 tail floats and the up-to-3 float4s past the last whole block are
    // outside any block, and pass 2 walks those densely on both arms.
    const bool summarize = block_max != nullptr;
    const int nblk = vector_length / kBlockVecs;

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
    // `nan_flag` is this row's slot in the caller's per-row table, zero when
    // this kernel runs (the dispatcher zeroes the table when it grows, and
    // each call's contract half clears what it read); this only ever raises
    // it.  It is raised before the answer is written, so the contract half --
    // which runs after this kernel -- sees the flag and takes its own NaN path
    // (trap, or the `0x3F3F3F3F` guard) exactly as it would have had it
    // scanned the row itself.
    bool nan_found = false;
    for (int index = tid; index < prefix; index += kThreads) {
        const float value = __ldg(input + index);
        nan_found |= is_nan_value(value);
        atomicAdd(&wide_histogram[refine_bin(value) >> kFirstShift], 1);
    }
    // Two loads in flight per thread where the register file affords it --
    // same measurement as the bf16 row's pair walk (265 -> 956 GB/s at one
    // 512-thread CTA per SM on a C600U): the loop below keeps one float4 in
    // flight per thread otherwise.  The body is spelled out twice inline on
    // purpose -- no function boundary around a load + process pair, per the
    // miscompile documented in docs/experiments/20261019_c600u_pass1/ §4.
    // `bulk_iters` is the iteration count every lane runs (`tid + m * kThreads`
    // stays under `vector_length` for all of them), and it is where the loop
    // splits so the summary's shuffle mask can be the constant one -- see
    // `block_max_bin`.  Everything past it is the tail: at most one iteration.
    int vec = tid;
    int it = 0;
    const int bulk_iters = vector_length / kThreads;
    if constexpr (ARCH_REGS_PER_AP >= 131072) {
        for (; it + 1 < bulk_iters; it += 2, vec += 2 * kThreads) {
            const float4 va = __ldg(vector_input + vec);
            const float4 vb = __ldg(vector_input + vec + kThreads);
            nan_found |= is_nan_value(va.x) || is_nan_value(va.y)
                         || is_nan_value(va.z) || is_nan_value(va.w);
            const int a0 = (int)(refine_bin(va.x) >> kFirstShift);
            const int a1 = (int)(refine_bin(va.y) >> kFirstShift);
            const int a2 = (int)(refine_bin(va.z) >> kFirstShift);
            const int a3 = (int)(refine_bin(va.w) >> kFirstShift);
            atomicAdd(&wide_histogram[a0], 1);
            atomicAdd(&wide_histogram[a1], 1);
            atomicAdd(&wide_histogram[a2], 1);
            atomicAdd(&wide_histogram[a3], 1);
            nan_found |= is_nan_value(vb.x) || is_nan_value(vb.y)
                         || is_nan_value(vb.z) || is_nan_value(vb.w);
            const int b0 = (int)(refine_bin(vb.x) >> kFirstShift);
            const int b1 = (int)(refine_bin(vb.y) >> kFirstShift);
            const int b2 = (int)(refine_bin(vb.z) >> kFirstShift);
            const int b3 = (int)(refine_bin(vb.w) >> kFirstShift);
            atomicAdd(&wide_histogram[b0], 1);
            atomicAdd(&wide_histogram[b1], 1);
            atomicAdd(&wide_histogram[b2], 1);
            atomicAdd(&wide_histogram[b3], 1);
            if (summarize) {
                const int ma = block_max_bin(a0, a1, a2, a3);
                const int mb = block_max_bin(b0, b1, b2, b3);
                if ((tid & (kBlockLanes - 1)) == 0) {
                    block_max[vec >> 2] = (uint16_t)ma;
                    block_max[(vec + kThreads) >> 2] = (uint16_t)mb;
                }
            }
        }
    }
    for (; it < bulk_iters; ++it, vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        nan_found |= is_nan_value(values.x) || is_nan_value(values.y)
                     || is_nan_value(values.z) || is_nan_value(values.w);
        const int c0 = (int)(refine_bin(values.x) >> kFirstShift);
        const int c1 = (int)(refine_bin(values.y) >> kFirstShift);
        const int c2 = (int)(refine_bin(values.z) >> kFirstShift);
        const int c3 = (int)(refine_bin(values.w) >> kFirstShift);
        atomicAdd(&wide_histogram[c0], 1);
        atomicAdd(&wide_histogram[c1], 1);
        atomicAdd(&wide_histogram[c2], 1);
        atomicAdd(&wide_histogram[c3], 1);
        if (summarize) {
            const int mc = block_max_bin(c0, c1, c2, c3);
            if ((tid & (kBlockLanes - 1)) == 0) block_max[vec >> 2] = (uint16_t)mc;
        }
    }
    // The tail: a prefix of the lanes, so its groups can straddle -- the
    // guarded spelling, and only here.
    for (; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        nan_found |= is_nan_value(values.x) || is_nan_value(values.y)
                     || is_nan_value(values.z) || is_nan_value(values.w);
        const int c0 = (int)(refine_bin(values.x) >> kFirstShift);
        const int c1 = (int)(refine_bin(values.y) >> kFirstShift);
        const int c2 = (int)(refine_bin(values.z) >> kFirstShift);
        const int c3 = (int)(refine_bin(values.w) >> kFirstShift);
        atomicAdd(&wide_histogram[c0], 1);
        atomicAdd(&wide_histogram[c1], 1);
        atomicAdd(&wide_histogram[c2], 1);
        atomicAdd(&wide_histogram[c3], 1);
        if (summarize) {
            const int mc = block_max_bin_edge(c0, c1, c2, c3);
            if ((tid & (kBlockLanes - 1)) == 0) block_max[vec >> 2] = (uint16_t)mc;
        }
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

    warp_cumsum_histogram<kRadix>(histogram);
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

    // ── the kept-block list ─────────────────────────────────────────────────
    // One shared atomic per kept block (the 16-bit row's compaction, same
    // shape).  A block whose max bin is under the threshold holds no element
    // pass 2 would act on, so the list is answer-preserving by construction --
    // and the compaction is its own step rather than a test in front of the
    // walk, because a test in front of the walk leaves the decision and the
    // fetch on one chain and every lane of a warp pays for the one member that
    // survived (measured; see `kBlockElems`).
    int nlist = 0;
    bool use_list = false;
    if (summarize && nblk > 0) {
        kept_block_count = 0;
        __syncthreads();
        for (int b = tid; b < nblk; b += kThreads) {
            if (static_cast<int>(block_max[b]) < wide_threshold) continue;
            block_list[atomicAdd(&kept_block_count, 1)] = static_cast<uint32_t>(b);
        }
        __syncthreads();
        nlist = kept_block_count;
        use_list = nlist * kListMaxDenominator <= nblk * kListMaxNumerator;
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
    if (use_list) {
        // One group of four lanes per kept block, one float4 each: the block is
        // a single 64-byte line, so the group's four loads are that line and
        // nothing else.  Four lanes per block also means a 64-lane wave covers
        // 16 blocks per iteration with no lane left out -- the property the
        // table-gated walk does not have.
        const int group = tid / kBlockLanes;
        const int lane_in_block = tid % kBlockLanes;
        for (int j = group; j < nlist; j += kThreads / kBlockLanes) {
            const int block = static_cast<int>(block_list[j]);
            const float4 values = __ldg(vector_input + block * kBlockVecs + lane_in_block);
            const int index = prefix + block * kBlockElems + lane_in_block * 4;
            DG12_COLLECT(values.x, index);
            DG12_COLLECT(values.y, index + 1);
            DG12_COLLECT(values.z, index + 2);
            DG12_COLLECT(values.w, index + 3);
        }
        // The up-to-3 float4s past the last whole block are outside every table
        // entry, so they are walked here rather than listed.
        for (int j = nblk * kBlockVecs + tid; j < vector_length; j += kThreads) {
            const float4 values = __ldg(vector_input + j);
            const int index = prefix + j * 4;
            DG12_COLLECT(values.x, index);
            DG12_COLLECT(values.y, index + 1);
            DG12_COLLECT(values.z, index + 2);
            DG12_COLLECT(values.w, index + 3);
        }
    } else {
    vec = tid;
    if constexpr (ARCH_REGS_PER_AP >= 131072) {
        for (; vec + kThreads < vector_length; vec += 2 * kThreads) {
            const float4 va = __ldg(vector_input + vec);
            const float4 vb = __ldg(vector_input + vec + kThreads);
            const int ia = prefix + vec * 4;
            const int ib = ia + kThreads * 4;
            DG12_COLLECT(va.x, ia);
            DG12_COLLECT(va.y, ia + 1);
            DG12_COLLECT(va.z, ia + 2);
            DG12_COLLECT(va.w, ia + 3);
            DG12_COLLECT(vb.x, ib);
            DG12_COLLECT(vb.y, ib + 1);
            DG12_COLLECT(vb.z, ib + 2);
            DG12_COLLECT(vb.w, ib + 3);
        }
    }
    for (; vec < vector_length; vec += kThreads) {
        const float4 values = __ldg(vector_input + vec);
        const int index = prefix + vec * 4;
        DG12_COLLECT(values.x, index);
        DG12_COLLECT(values.y, index + 1);
        DG12_COLLECT(values.z, index + 2);
        DG12_COLLECT(values.w, index + 3);
    }
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

            warp_cumsum_histogram<kRadix>(histogram);
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

        warp_cumsum_histogram<kRadix>(histogram);
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

// One CTA per row.  `__launch_bounds__` is `kThreads` (512, the original's
// width): the widest cooperative step is the 4096-bin histogram clear and the
// candidate walk, and 512 is what fills the AP -- 640 x 3 CTAs is 1920 threads,
// under the 2048-thread/AP ceiling, so it left 128 of the AP's 2048 thread
// slots unused (measured, `docs/experiments/20261022_f32_occupancy/`).
//
// The width is a template parameter because the kernel is instantiated
// explicitly in the generated TU that owns its launch; the row body below is
// written against the namespace's `kThreads`, which the assertion pins.
template <int kThreads_>
__global__ __launch_bounds__(kThreads_) void topk_coarse12_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    int32_t *__restrict__ out, int32_t *__restrict__ nan_flags, int topk, int B,
    int64_t stride, int default_length, uint16_t *__restrict__ block_max,
    uint32_t *__restrict__ block_list)
{
    static_assert(kThreads_ == kThreads,
                  "`topk_coarse12_row` walks the row at the namespace's kThreads");
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
    // One slot per row.  Zero when this kernel runs -- the dispatcher zeroes
    // the table once when it grows it, and each call's contract half clears
    // the slots it read -- so this kernel only raises it (see
    // `RowParams::nan_flags`).  Null means "the caller is not scanning at all"
    // (`check_nan` false), which is the only case where the contract half does
    // not read the table either.
    int32_t *row_nan = nan_flags == nullptr ? nullptr : nan_flags + row;
    // The row's slice of the skip tables.  Both are indexed by *blocks of the
    // row as the tensor lays it out* (`vocab_size` wide), not by this row's
    // window, so a shorter row still lands on its own slice -- the same rule
    // the 16-bit row's tables follow.  Null means the dispatcher could not
    // allocate them, which the row takes as "walk densely".
    uint16_t *row_bmax = nullptr;
    uint32_t *row_blist = nullptr;
    if (block_max != nullptr && block_list != nullptr) {
        const int64_t blocks_per_row =
            (int64_t)((default_length + kBlockElems - 1) / kBlockElems);
        row_bmax = block_max + (int64_t)row * blocks_per_row;
        row_blist = block_list + (int64_t)row * blocks_per_row;
    }

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

    // Constant today -- `topk_coarse12_row` has no `return false` -- so the
    // `-1` fill below is dead code kept as the seam for the day the arena bound
    // moves back inside the row function.  Named rather than inlined so that
    // seam stays visible.
    const bool served = topk_coarse12_row(row_input, row_out, length, topk,
                                         row_nan, row_bmax, row_blist);
    if (!served) {
        for (int i = threadIdx.x; i < topk; i += kThreads) row_out[i] = -1;
    }
}

// The launcher that owns this kernel's one `<<<>>>`.
template <int kThreads_>
inline cudaError_t launch_coarse12_kernel(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length,
    uint16_t *block_max, uint32_t *block_list, cudaStream_t stream)
{
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    static bool smem_adopted = false;
    const cudaError_t smem_err = topk_select_radix::adopt_dynamic_smem(
        topk_coarse12_kernel<kThreads_>, kSmemBytes, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;
    topk_coarse12_kernel<kThreads_><<<B, (int)kThreads_, kSmemBytes, stream>>>(
        scores, lengths, out, nan_flags, topk, B, stride, default_length,
        block_max, block_list);
    return cudaGetLastError();
}

extern template cudaError_t launch_coarse12_kernel<kThreads>(
    const float *, const int32_t *, int32_t *, int32_t *, int, int, int64_t, int,
    uint16_t *, uint32_t *, cudaStream_t);

inline cudaError_t launch_topk_coarse12(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length,
    uint16_t *block_max, uint32_t *block_list, cudaStream_t stream)
{
    return launch_coarse12_kernel<kThreads>(scores, lengths, out, nan_flags, B,
                                            topk, stride, default_length,
                                            block_max, block_list, stream);
}


// The route's launch and emit.  The kernel answers in row *columns*, which are
// int32 by nature, so it cannot write the caller's buffer directly; the
// columns are staged and the contract half widens them the way it widens the
// split's answer.  The gate that admits this route is
// `deep_select::detail::f32_coarse12_applies` (`routing.cuh`); what is here is
// everything after the decision -- the scratch check, the launch and the
// fallback to the row path.
template <typename OutIdxT>
inline void launch_route(const deep_select::RowParams &params, uint32_t batches,
                         cudaStream_t stream, bool sorted_index,
                         bool sorted_value, bool return_value, int block) {
    int32_t *const staging = params.coarse12_cols;
    bool launched = false;
    if (staging != nullptr) {
        // No split, no workspace of its own, no NaN table of its own: this
        // kernel walks each row itself and answers in row columns, so the
        // contract half below is the only other thing that touches the row
        // -- and it is a pass over the *answer*, not over the input.
        // `params.end_ptr` is the row table the dispatcher already
        // populated with the scratch's all-`vocab_size` entry when the
        // caller passed no `end`, which is the same table the split reads.
        // `params.vocab_size` -- not the row stride -- is the width to
        // scan.  They agree for a full-width tensor and differ the moment
        // the caller hands over a column slice of a wider matrix, which is
        // exactly what the benchmark harness does (`s[:, :L]`): the kernel
        // would then walk past the live prefix and rank columns that are
        // outside the tensor the caller asked about.
        launched = launch_topk_coarse12(
            (const float *)params.input, params.end_ptr, staging,
            params.scan_flags, (int)batches, (int)params.topk,
            (int64_t)(params.stride_input_batch / sizeof(float)),
            (int)params.vocab_size, params.coarse12_block_max,
            params.coarse12_block_list, stream) == cudaSuccess;
    }
    // Hand the answer to the contract half the way the split does, so its
    // per-row `-1` check decides row by row: a row this kernel could not
    // serve, or one whose window is no longer than `topk`, comes back with an
    // empty slot and is re-ranked by the row path.  A launch that did not
    // happen leaves the buffer untouched and answers that way too -- as does
    // an allocation that could not be made.
    //
    // One kernel, so a failed launch means nothing ran: there is no partial
    // work whose scan flag could have survived, which is why this route
    // clears nothing (`clear_flags_on_failure = false`).
    deep_select::detail::launch_staged_or_row<float, OutIdxT>(
        params, batches, stream, sorted_index, sorted_value, return_value, block,
        launched, staging, /*clear_flags_on_failure=*/false);
}

}  // namespace topk_select_coarse12

