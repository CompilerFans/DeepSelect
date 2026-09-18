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
template <int ChunkCount>
inline cudaError_t launch_topk_chunks_impl(
    const float *scores, const int32_t *lengths, int32_t *out, void *workspace,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream);

__host__ __forceinline__ int chunks_for_shape(int rows, int64_t n_cols, int num_sms);

// One launch of the three kernels, for the `NChunks` the shape resolves to.
// The workspace is the caller's (`RowParams::chunks_workspace`), sized by
// `chunks_workspace_bytes` from the same `select_chunk_count`.
inline cudaError_t launch_topk_chunks(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    void *workspace, int B, int topk, int64_t stride, int default_length,
    int num_sms, cudaStream_t stream)
{
    (void)nan_flags;
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    if (workspace == nullptr) return cudaErrorInvalidValue;
    const int chunks = chunks_for_shape(B, (int64_t)default_length, num_sms);
    switch (chunks) {
        case 3: return launch_topk_chunks_impl<3>(scores, lengths, out, workspace,
                                                  B, topk, stride, default_length, stream);
        case 4: return launch_topk_chunks_impl<4>(scores, lengths, out, workspace,
                                                  B, topk, stride, default_length, stream);
        case 5: return launch_topk_chunks_impl<5>(scores, lengths, out, workspace,
                                                  B, topk, stride, default_length, stream);
        case 6: return launch_topk_chunks_impl<6>(scores, lengths, out, workspace,
                                                  B, topk, stride, default_length, stream);
        default: return cudaErrorInvalidValue;
    }
}


// ── the chunked form ───────────────────────────────────────────────────────
//
// The three kernels below are deep_gemm's `topk_chunks_*`, restored after the
// first version of this port dropped them.  That version was one CTA per row
// and it was **slower than the split it was meant to replace** at every width
// (ledger §12.17): 82/116/132 us against deep_gemm's 53/61/63.  The reason is
// not subtle -- a single CTA sweeping 66551 floats twice on a 104-AP part --
// and the conclusion recorded there, that chunking "buys at most 6 CTAs" at
// `b <= 2`, counted the wrong thing: without it the count is **one**.
//
// `select_chunk_count` is deep_gemm's own, verbatim, including its constants.
// It is a *scheduling* heuristic rather than a shape rule: it only adds chunks
// when the last scheduling wave would otherwise be under-filled, which is
// exactly the situation at one or two rows.

constexpr int kMinChunkCount = 3;
constexpr int kMaxChunkCount = 6;
constexpr int64_t kMinElementsPerChunk = 4096;

__host__ __forceinline__ int select_chunk_count(int64_t n_rows, int64_t n_cols,
                                                int num_sms)
{
    const int max_chunks =
        min((int)kMaxChunkCount, static_cast<int>(n_cols / kMinElementsPerChunk));
    if (max_chunks < kMinChunkCount) return 0;

    num_sms = max(num_sms, 1);
    const auto tail_blocks = [num_sms](int64_t blocks) {
        const int waves = static_cast<int>((blocks + num_sms - 1) / num_sms);
        return static_cast<int>(blocks - static_cast<int64_t>(waves - 1) * num_sms);
    };

    // An exact multiple occupies a full last wave, so it never needs chunks.
    const int single_tail = tail_blocks(n_rows);
    int best_chunk_tail = 0;
    for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks)
        best_chunk_tail = max(best_chunk_tail, tail_blocks(n_rows * chunks));
    if (best_chunk_tail <= single_tail) return 0;
    return max_chunks;
}

// One row's shared state between the three kernels, one entry per row.
//
// `NChunks` is a template parameter so the arrays are fixed-size -- this is a
// workspace, not a heap, and the launch knows which instantiation it is because
// `select_chunk_count` returned its value.  The layout is identical across
// instantiations, which is what lets the dispatcher allocate one byte buffer
// sized for whichever `NChunks` this call resolves to.
//
// `candidate_indices` is deep_gemm's own `alignas(16)`, and it is what forces
// the dynamic shared memory below: the array is 16 KB, and the refine stages
// alias it as two 8 KB halves.
template <int ChunkCount>
struct TopKChunksWorkspace {
    int coarse[ChunkCount][kCoarseBins];
    int fine[ChunkCount][kFineBins];
    int guaranteed_bases[ChunkCount];
    int boundary_bases[ChunkCount];
    int threshold;
    int guaranteed_count;
    int boundary_take;
    int candidate_count;
    int arrival;
    // The row's window, resolved once by `topk_chunks_init` and read by both
    // later kernels.  It is here rather than passed as an argument because the
    // three kernels have to agree on it and only the first one has the table in
    // hand -- and because the *refine* needs it to decide, per row, whether the
    // row is one this kernel serves at all.
    int row_length;
    // Set by the coarse merge when the threshold bin is wider than the staging
    // arena.  deep_gemm re-reads the row under a narrowing prefix in that case
    // (its `rescanned` path); this port does not carry that path, so a row it
    // cannot stage is **declined** -- the whole row goes back to the row path,
    // which is a coverage statement and never a wrong answer.
    int declined;
    // Two halves, because the refine ping-pongs them: `[0]` receives the coarse
    // stage's members (written by the coarse kernel's merge CTA, hence device
    // memory rather than shared -- the chunks are separate CTAs) and the two
    // alternate from there.  deep_gemm spells this as
    // `extern __shared__ int staged_indices[][kCandidateCapacity]`, i.e. also
    // two rows; sizing it as one is what made `write_buf[put]` run off the end
    // of the workspace and into the next row's.
    alignas(16) int candidate_indices[2][kCandidateCapacity];
};

template <int ChunkCount>
constexpr size_t chunks_workspace_bytes_for() {
    return sizeof(TopKChunksWorkspace<ChunkCount>);
}

// The dispatcher's sizing entry: one `TopKChunksWorkspace<NChunks>` per row,
// for whichever `NChunks` `select_chunk_count` resolves the shape to.  The
// launch is given the same `select_chunk_count`, so the allocation and the grid
// cannot disagree about which instantiation is in play.
//
// **`batches` multiplies, and it is not decoration.**  The first version of
// this returned `sizeof(TopKChunksWorkspace<NChunks>)` alone -- one row's worth
// -- while the kernels index `workspaces[row]`.  At the band this arm shipped
// with (`batches <= 2`) that is a two-row overflow of a one-row buffer, which
// is why it took a batch the band excludes to see it: `topk_chunks_init` zeroes
// `arrival`/`declined` for row 1 just past the end of the arena, and whether
// that is fatal depends on what the allocator put there and on how many calls
// have run since.  deep_gemm's own sizing has the factor
// (`n_rows * sizeof(...)`); this is the port's copy of it.
//
// `sm_count` is the device's, as everywhere else in this file.
inline size_t chunks_workspace_bytes(uint32_t batches, uint32_t vocab_size,
                                     uint32_t sm_count) {
    switch (select_chunk_count((int64_t)batches, (int64_t)vocab_size, (int)sm_count)) {
        case 3: return (size_t)batches * chunks_workspace_bytes_for<3>();
        case 4: return (size_t)batches * chunks_workspace_bytes_for<4>();
        case 5: return (size_t)batches * chunks_workspace_bytes_for<5>();
        case 6: return (size_t)batches * chunks_workspace_bytes_for<6>();
        default: return 0;
    }
}

// One CTA per (row, chunk): resolves the row's window and its own chunk bounds.
// deep_gemm's Wave-0/Wave-1 shape is kept even though this port has no
// page-table half to resolve in Wave 1 -- the second warp's slot is simply
// unused, and keeping the block shape means the arrival protocol below matches.
//
// The window is clamped into `[0, default_length]` and **stored**, because it
// is the only place the three kernels can agree on it: `lengths` is a table
// that says how much of each row is live (`end` in the public API), while
// `default_length` is how wide the row is.  Reading the table as the *width*,
// or the width as the length, is wrong for every call that has an `end` -- and
// a row whose window is no longer than `topk` is not this kernel's to answer at
// all (see `declined` below).
template <int ChunkCount>
__global__ void topk_chunks_init(TopKChunksWorkspace<ChunkCount> *workspaces,
                                 const int32_t *lengths, int rows,
                                 int default_length)
{
    const int row = blockIdx.x;
    if (row >= rows) return;
    if (threadIdx.x == 0) {
        const int32_t raw = lengths == nullptr ? default_length : __ldg(lengths + row);
        const int length = raw < 0 ? 0 : (raw > default_length ? default_length : raw);
        workspaces[row].row_length = length;
        workspaces[row].arrival = 0;
        workspaces[row].declined = 0;
    }
}

// The coarse histogram, one CTA per (row, chunk), merged by the **last** CTA to
// arrive through the per-row `arrival` counter.  Everything downstream -- the
// threshold, the per-chunk bases -- is computed there, once per row.
template <int ChunkCount>
__global__ __launch_bounds__(kThreads) void topk_chunks_coarse_hist(
    const float *__restrict__ scores, TopKChunksWorkspace<ChunkCount> *workspaces,
    int topk, int64_t stride, int default_length)
{
    const int chunk = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    TopKChunksWorkspace<ChunkCount> &workspace = workspaces[row];
    // The row's *window*, resolved once by `topk_chunks_init`.  Not
    // `default_length` (the tensor's width, which is an upper bound on the
    // window and equal to it only when the caller passed no `end`), and not
    // `stride` (a property of the tensor, larger than either for a column
    // slice).  Both substitutions read correctly on a full-width, no-`end`
    // tensor, which is why the difference is invisible until it is not.
    const int length = workspace.row_length;
    // A row this kernel does not serve contributes nothing: every member of it
    // writes the same per-chunk coarse histogram as an empty one, and the merge
    // below declines the whole row before it computes anything from it.  The
    // early exit is what keeps a window shorter than a chunk from being read
    // past, and it has to be here rather than in the merge because the merge
    // only runs on the last chunk to arrive.
    if (length <= topk) {
        if (tid < kCoarseBins) workspace.coarse[chunk][tid] = 0;
        __threadfence();
        __syncthreads();
        __shared__ int trivial_last;
        if (tid == 0) trivial_last = atomicAdd(&workspace.arrival, 1) == ChunkCount - 1;
        __syncthreads();
        if (!trivial_last) return;
        if (tid == 0) {
            workspace.declined = 1;
            workspace.arrival = 0;
        }
        return;
    }

    __shared__ int histogram[kCoarseBins + kHistogramPadding];
    for (int bin = tid; bin < kCoarseBins; bin += kThreads) histogram[bin] = 0;
    __syncthreads();

    const int chunk_begin = (int)((int64_t)length * chunk / ChunkCount);
    const int chunk_end = (int)((int64_t)length * (chunk + 1) / ChunkCount);
    const float *const row_input = scores + (int64_t)row * stride;
    const float *const chunk_input = row_input + chunk_begin;
    const int chunk_length = chunk_end - chunk_begin;

    const uintptr_t address = reinterpret_cast<uintptr_t>(chunk_input);
    const int prefix_unclamped = static_cast<int>(
        (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
    const int prefix =
        (address & (alignof(float4) - 1)) == 0 ? 0 : min(chunk_length, prefix_unclamped);
    const float4 *vector_input = reinterpret_cast<const float4 *>(chunk_input + prefix);
    const int vector_length = (chunk_length - prefix) / 4;
    const int tail = prefix + vector_length * 4;

    for (int index = tid; index < prefix; index += kThreads)
        atomicAdd(&histogram[coarse_bin(chunk_input[index])], 1);
    for (int index = tid; index < vector_length; index += kThreads) {
        const float4 values = vector_input[index];
        atomicAdd(&histogram[coarse_bin(values.x)], 1);
        atomicAdd(&histogram[coarse_bin(values.y)], 1);
        atomicAdd(&histogram[coarse_bin(values.z)], 1);
        atomicAdd(&histogram[coarse_bin(values.w)], 1);
    }
    for (int index = tail + tid; index < chunk_length; index += kThreads)
        atomicAdd(&histogram[coarse_bin(chunk_input[index])], 1);

    __syncthreads();
    if (tid < kCoarseBins) workspace.coarse[chunk][tid] = histogram[tid];
    __threadfence();
    __syncthreads();

    // The arrival protocol: the last chunk to land owns the merge.  `arrival`
    // is reset by `topk_chunks_compact_refine`'s last CTA for the next call, so
    // this kernel does not have to zero it (and must not -- another chunk of
    // the same row may still be running).
    __shared__ int is_last;
    if (tid == 0) is_last = atomicAdd(&workspace.arrival, 1) == ChunkCount - 1;
    __syncthreads();
    if (!is_last) return;

    // The merge: total the per-chunk histograms, suffix-sum, find the bin
    // holding rank `topk`, then record where each chunk's `guaranteed` and
    // `boundary` members go in the row's output.  Every count here is the
    // *true* count, which is what lets the refine below decide without a second
    // merge.
    int total = 0;
#pragma unroll
    for (int source = 0; source < ChunkCount; ++source)
        total += workspace.coarse[source][tid < kCoarseBins ? tid : 0];
    if (tid < kCoarseBins) histogram[tid] = 0;
    __syncthreads();
    if (tid < kCoarseBins) {
#pragma unroll
        for (int source = 0; source < ChunkCount; ++source)
            histogram[tid] += workspace.coarse[source][tid];
    }
    block_cumsum_histogram_1024(histogram);
    if (tid < kCoarseBins && histogram[tid] >= topk && histogram[tid + 1] < topk) {
        workspace.threshold = tid;
    }
    __syncthreads();
    const int threshold = workspace.threshold;

    __shared__ int chunk_guaranteed[ChunkCount];
    __shared__ int chunk_boundary[ChunkCount];
    if (tid < ChunkCount) {
        int guaranteed = 0;
        for (int bin = threshold + 1; bin < kCoarseBins; ++bin)
            guaranteed += workspace.coarse[tid][bin];
        chunk_guaranteed[tid] = guaranteed;
        chunk_boundary[tid] = workspace.coarse[tid][threshold];
    }
    __syncthreads();
    if (tid == 0) {
        int guaranteed_base = 0;
        int boundary_base = 0;
#pragma unroll
        for (int source = 0; source < ChunkCount; ++source) {
            workspace.guaranteed_bases[source] = guaranteed_base;
            workspace.boundary_bases[source] = boundary_base;
            guaranteed_base += chunk_guaranteed[source];
            boundary_base += chunk_boundary[source];
        }
        workspace.guaranteed_count = guaranteed_base;
        workspace.boundary_take = topk - guaranteed_base;
        workspace.candidate_count = boundary_base;
        // A threshold bin wider than the staging arena is a row this port does
        // not serve: deep_gemm re-reads the row under a narrowing prefix in
        // that case (its `rescanned` path), and this port does not carry that
        // path, so the row is handed back whole.  Declining is a *coverage*
        // decision -- `launch_topk_chunks`' caller turns it into the per-row
        // `-1` that sends the row down the row path -- and it must be taken
        // here, before the refine walks anything, because a truncated arena
        // would otherwise answer for an arbitrary subset of the bin.
        workspace.declined = boundary_base > kCandidateCapacity;
        workspace.arrival = 0;
    }
    (void)total;
}

// The second row walk: writes every element above the threshold bin straight to
// the output, stages the threshold bin's own members, and -- in the last CTA to
// arrive -- resolves the boundary down to individual columns.
template <int ChunkCount>
__global__ __launch_bounds__(kThreads) void topk_chunks_compact_refine(
    const float *__restrict__ scores, int32_t *__restrict__ out,
    TopKChunksWorkspace<ChunkCount> *workspaces,
    int topk, int64_t stride, int default_length)
{
    const int chunk = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    TopKChunksWorkspace<ChunkCount> &workspace = workspaces[row];
    const int length = workspace.row_length;
    const int chunk_begin = (int)((int64_t)length * chunk / ChunkCount);
    const int chunk_end = (int)((int64_t)length * (chunk + 1) / ChunkCount);
    const float *const row_input = scores + (int64_t)row * stride;
    int32_t *const row_output = out + (int64_t)row * topk;

    // A row the coarse stage declined -- a window no longer than `topk`, or a
    // threshold bin wider than the arena -- is answered by the row path, so
    // this kernel writes nothing for it but the per-row `-1` that routes it
    // there.  Chunk 0 owns that write so the row is touched once; the other
    // chunks return without reading the row at all, which is also what keeps a
    // window shorter than a chunk from being read past.
    if (workspace.declined) {
        if (chunk == 0) {
            for (int i = tid; i < topk; i += kThreads) row_output[i] = -1;
        }
        return;
    }

    // The arena is the workspace's own `candidate_indices`, aliased as two
    // halves for the ping-pong below.  It is *device* memory rather than shared
    // here because the chunks are separate CTAs: a shared buffer would not be
    // visible to the CTA that does the merge.
    int *const staging_a = workspace.candidate_indices[0];
    int *const staging_b = workspace.candidate_indices[1];
    int *read_buf = staging_a;
    int *write_buf = staging_b;

    __shared__ int histogram[kFineBins + kHistogramPadding];
    __shared__ int guaranteed_counter;
    __shared__ int boundary_counter;
    __shared__ int candidate_counter;
    __shared__ int threshold_bin_id;
    __shared__ int last_remain;
    __shared__ int is_last;
    __shared__ int count;

    if (tid < kFineBins) histogram[tid] = 0;
    if (tid == 0) {
        guaranteed_counter = 0;
        boundary_counter = 0;
        candidate_counter = 0;
    }
    __syncthreads();

    const int threshold = workspace.threshold;
    const int guaranteed_base = workspace.guaranteed_bases[chunk];
    const int boundary_base = workspace.boundary_bases[chunk];
    const float *const chunk_input = row_input + chunk_begin;
    const int chunk_length = chunk_end - chunk_begin;
    const uintptr_t address = reinterpret_cast<uintptr_t>(chunk_input);
    const int prefix_unclamped = static_cast<int>(
        (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
    const int prefix =
        (address & (alignof(float4) - 1)) == 0 ? 0 : min(chunk_length, prefix_unclamped);
    const float4 *vector_input = reinterpret_cast<const float4 *>(chunk_input + prefix);
    const int vector_length = (chunk_length - prefix) / 4;
    const int tail = prefix + vector_length * 4;

    // `guaranteed` writes go to this chunk's own slice of the row output, so
    // the atomic only has to order them within the chunk.
    const auto compact = [&](int local_index, float value, int bin) {
        const int column = chunk_begin + local_index;
        if (bin > threshold) {
            const int pos = atomicAdd(&guaranteed_counter, 1);
            row_output[guaranteed_base + pos] = column;
        }
        else if (bin == threshold) {
            atomicAdd(&histogram[(refine_bin(value) >> 24) & 0xffu], 1);
            const int local_pos = atomicAdd(&boundary_counter, 1);
            const int global_pos = boundary_base + local_pos;
            if (global_pos < kCandidateCapacity) {
                // Into the *first* arena row: the refine below starts with
                // `read_buf == staging_a`, exactly as the original does with
                // `staged_indices[0]`.
                staging_a[global_pos] = column;
            }
        }
    };
    for (int i = tid; i < prefix; i += kThreads)
        compact(i, chunk_input[i], coarse_bin(chunk_input[i]));
    for (int i = tid; i < vector_length; i += kThreads) {
        const float4 values = vector_input[i];
        const int index = prefix + i * 4;
        compact(index, values.x, coarse_bin(values.x));
        compact(index + 1, values.y, coarse_bin(values.y));
        compact(index + 2, values.z, coarse_bin(values.z));
        compact(index + 3, values.w, coarse_bin(values.w));
    }
    for (int i = tail + tid; i < chunk_length; i += kThreads)
        compact(i, chunk_input[i], coarse_bin(chunk_input[i]));

    __syncthreads();
    if (tid < kFineBins) workspace.fine[chunk][tid] = histogram[tid];
    __threadfence();
    __syncthreads();
    if (tid == 0) is_last = atomicAdd(&workspace.arrival, 1) == ChunkCount - 1;
    __syncthreads();
    if (!is_last) return;

    // ── the merge CTA ──────────────────────────────────────────────────────
    // Everything below runs once per row, on the chunk that arrived last.
    const int boundary_take = workspace.boundary_take;
    if (boundary_take == 0) {
        // Every slot was `guaranteed`, so the answer is already written.
        if (tid == 0) workspace.arrival = 0;
        return;
    }

    // The fine histograms are merged here, and `count` is the *staged* width --
    // capped at `kCandidateCapacity`, unlike `workspace.candidate_count` which
    // is the true width.  A bin wider than the arena is declined by the caller
    // (see `chunks_row_served`), so the two agree whenever this runs.
    if (tid < kFineBins) {
        int sum = 0;
#pragma unroll
        for (int source = 0; source < ChunkCount; ++source)
            sum += workspace.fine[source][tid];
        histogram[tid] = sum;
    }
    if (tid == 0) {
        histogram[kFineBins] = 0;
        count = min(workspace.candidate_count, kCandidateCapacity);
        candidate_counter = 0;
    }
    __syncthreads();

    int remaining = boundary_take;
    warp_cumsum_histogram_256(histogram);
    if (tid < kFineBins && histogram[tid] >= remaining && histogram[tid + 1] < remaining) {
        threshold_bin_id = tid;
        last_remain = remaining - histogram[tid + 1];
    }
    __syncthreads();
    int fine_threshold = threshold_bin_id;
    remaining -= histogram[fine_threshold + 1];

    if (remaining == 0) {
        for (int pos = tid; pos < count; pos += kThreads) {
            const int column = read_buf[pos];
            if (static_cast<int>((refine_bin(row_input[column]) >> 24) & 0xffu)
                > fine_threshold) {
                const int out_pos = atomicAdd(&candidate_counter, 1);
                row_output[workspace.guaranteed_count + out_pos] = column;
            }
        }
        __syncthreads();
        if (tid == 0) workspace.arrival = 0;
        return;
    }

    // Byte 16, then the three rounds over bytes 16, 8 and 0 -- deep_gemm's
    // ordering, which the `remaining` accounting depends on.
    if (tid < kFineBins + kHistogramPadding) histogram[tid] = 0;
    if (tid == 0) boundary_counter = 0;
    __syncthreads();
    for (int pos = tid; pos < count; pos += kThreads) {
        const int column = read_buf[pos];
        const uint32_t key = refine_bin(row_input[column]);
        const int bin = static_cast<int>(key >> 24);
        if (bin > fine_threshold) {
            const int out_pos = atomicAdd(&candidate_counter, 1);
            row_output[workspace.guaranteed_count + out_pos] = column;
        }
        else if (bin == fine_threshold) {
            const int put = atomicAdd(&boundary_counter, 1);
            if (put < kCandidateCapacity) {
                write_buf[put] = column;
                atomicAdd(&histogram[(key >> 16) & 0xffu], 1);
            }        }
    }
    __syncthreads();
    { int *const t = read_buf; read_buf = write_buf; write_buf = t; }
    if (tid == 0) count = min(boundary_counter, kCandidateCapacity);
    __syncthreads();

    for (int round = 0; round < 3; ++round) {
        const int offset = 16 - round * 8;
        if (count == 0) break;
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
            for (int pos = tid; pos < count; pos += kThreads) {
                const int column = read_buf[pos];
                if (static_cast<int>((refine_bin(row_input[column]) >> offset) & 0xffu)
                    > fine_threshold) {
                    const int out_pos = atomicAdd(&candidate_counter, 1);
                    row_output[workspace.guaranteed_count + out_pos] = column;
                }
            }
            __syncthreads();
            if (tid == 0) workspace.arrival = 0;
            return;
        }

        if (tid < kFineBins + kHistogramPadding) histogram[tid] = 0;
        if (tid == 0) boundary_counter = 0;
        __syncthreads();
        for (int pos = tid; pos < count; pos += kThreads) {
            const int column = read_buf[pos];
            const uint32_t key = refine_bin(row_input[column]);
            const int bin = (key >> offset) & 0xffu;
            if (bin > fine_threshold) {
                const int out_pos = atomicAdd(&candidate_counter, 1);
                row_output[workspace.guaranteed_count + out_pos] = column;
            }
            else if (bin == fine_threshold) {
                if (round == 2) {
                    const int left = atomicAdd(&last_remain, -1);
                    if (left > 0) row_output[topk - left] = column;
                }
                else {
                    const int put = atomicAdd(&boundary_counter, 1);
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
        if (tid == 0) count = min(boundary_counter, kCandidateCapacity);
        __syncthreads();
    }
    if (tid == 0) workspace.arrival = 0;
}

// Does the shape reach the chunked form at all?  `select_chunk_count` returning
// 0 means "no chunks", which is the single-CTA arm this file used to be -- and
// that arm is slower than the split, so 0 is a decline rather than a fallback.
__host__ __forceinline__ int chunks_for_shape(int rows, int64_t n_cols,
                                              int num_sms) {
    return select_chunk_count((int64_t)rows, n_cols, num_sms);
}

template <int ChunkCount>
inline cudaError_t launch_topk_chunks_impl(
    const float *scores, const int32_t *lengths, int32_t *out, void *workspace,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream)
{
    auto *ws = reinterpret_cast<TopKChunksWorkspace<ChunkCount> *>(workspace);
    const dim3 grid(ChunkCount, (unsigned)B);
    topk_chunks_init<ChunkCount><<<B, 2 * 64, 0, stream>>>(ws, lengths, B, default_length);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    topk_chunks_coarse_hist<ChunkCount><<<grid, kThreads, 0, stream>>>(
        scores, ws, topk, stride, default_length);
    err = cudaGetLastError();
    if (err != cudaSuccess) return err;
    topk_chunks_compact_refine<ChunkCount><<<grid, kThreads, 0, stream>>>(
        scores, out, ws, topk, stride, default_length);
    return cudaGetLastError();
}

}  // namespace dgchunks
}  // namespace rk
