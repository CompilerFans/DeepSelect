// deep_gemm's `topk_chunks`, ported for the b <= 2 band.
//
// Why this file exists
// -------------------
// `radix_topk_row_f32` (`radix_row_fp32.cuh`) and `topk_coarse12`
// (`topk_coarse12.cuh`) both serve this operator, and neither is the right shape
// at one or two rows.  Measured, interleaved five rounds, device 2, `k = 2048`
// (ledger §12.16.6):
//
//     b     V        ours     deep_gemm    deep_gemm/ours
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
//     synthesize (the same subtraction `topk_coarse12.cuh` makes);
//   * the `rescanned` branch, where deep_gemm re-walks the row when the
//     threshold bin is wider than `kCandidateCapacity`.  This port **declines**
//     the row instead: the coarse merge sets `workspace.declined`, the refine
//     leaves that row's output at `-1`, and the contract half re-ranks it on
//     the row path.  That changes coverage, not correctness, and the branch it
//     removes only fires on a bin wider than 4096, which `randn` at these
//     widths does not produce (the same measurement `kF32OverflowChunkLen`
//     rests on, `routing.cuh`).
//
// **The multi-CTA chunking was originally dropped here and has since been
// restored.**  The paragraph that used to sit in this list said the port was
// one CTA per row, "so it needs no workspace, no arrival protocol and no
// per-row allocation -- and at `b <= 2` the chunking buys at most 6 CTAs on a
// 104-AP part, which is the one thing it cannot be buying."  Every clause of
// that is wrong about the shipped code: the three `topk_chunks_*` kernels
// below are deep_gemm's, with `NChunks in [3, 6]` from `select_chunk_count`,
// the per-row `TopKChunksWorkspace` arena, the `arrival` counter and the
// cross-CTA merge -- see `chunks_workspace_bytes`.  The count was the wrong
// thing to count: without chunking a row is **one** CTA on a 104-AP part, and
// the one-CTA version measured slower than the split it was meant to replace
// at every width (ledger §12.17).  The history is kept in the `chunked form`
// note further down; this list is what the *shipped* code drops, and the only
// thing on it is the `Transform` half and `rescanned`.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <csrc/maca_kernels/common/primitives.cuh>

// This file is the dataflow only: the kernels and their launchers.  The route
// that reaches them -- the scratch check, the launch decision and the handoff
// to the contract half -- is `deep_select::detail::launch_typed_chunks`
// (`routing.cuh`), with the other routes' handoffs, so a dataflow header does
// not depend on the contract half it is emitted through.

namespace topk_select_chunks {

// ── deep_gemm's `detail` constants, verbatim ───────────────────────────────
constexpr int kThreads = 1024;
constexpr int kMaxTopK = 2048;
constexpr int kCoarseBins = 1024;
constexpr int kFineBins = 256;
constexpr int kCandidateCapacity = 4096;
constexpr int kHistogramPadding = 1;
// **UNUSED, and it has no call sites anywhere in this tree** -- grep the name
// before assuming otherwise.  It is deep_gemm's own name for the two-half
// staging arena, and it does not describe this port's: the arena here is
// `TopKChunksWorkspace::candidate_indices` below, **device** memory in the
// workspace rather than a shared buffer, and all three launches pass `0`
// dynamic shared memory.  Nothing else reads it either -- the three
// `static_assert`s beneath it are about `kCandidateCapacity` and the block
// width, not about this.  Kept rather than deleted because it is deep_gemm's
// own constant and a reader comparing the two files should find it where the
// original has it.
constexpr size_t kSmemBytes = 2 * (size_t)kCandidateCapacity * sizeof(int32_t);

static_assert(kFineBins + kHistogramPadding <= kThreads,
              "the fine histogram is cleared and cumsum'd by one thread per "
              "bin, so the block has to cover it");
static_assert(kCoarseBins <= kThreads,
              "same, for the coarse histogram");
static_assert(kCandidateCapacity >= kMaxTopK,
              "the staged candidates must be able to hold a whole top-k "
              "answer -- which is what lets the `remaining == 0` arm write the "
              "boundary out of the staging buffer without a second walk");

// The NaN predicate, the key encode and the 256-bin warp suffix sum come from
// `../common/primitives.cuh`; unqualified here because every call site below
// is spelled that way.  `coarse_bin` below is this dataflow's own *width*: it
// keys the half-rounded value, which is the coarse level's shape, but the fold
// itself is the shared `refine_bin_bits16`.
using topk_select_common::is_nan_value;
using topk_select_common::refine_bin;
using topk_select_common::refine_bin_bits16;
using topk_select_common::warp_cumsum_histogram;

// deep_gemm's `coarse_bin`: the top 10 bits of the **half**-encoded key.  Half,
// not float -- the coarse level only has to put the threshold in the right
// neighbourhood, and the refine below resolves within the bin on the exact
// float bits.  Upstream's `coarse_bins_pair` is its float2 form; this port
// reads `float4`, so the four lanes go through `coarse_bin` one at a time
// exactly as `topk_chunks_coarse_hist` does for its vector tail.
__device__ __forceinline__ int coarse_bin(float value)
{
    const uint16_t bits = __half_as_ushort(__float2half_rn(value));
    return static_cast<int>(refine_bin_bits16(bits) >> 6);
}

// The 256-bin histogram suffix sum the fine level walks is
// `topk_select_common::warp_cumsum_histogram<kFineBins>` (`../common/primitives.cuh`,
// pulled in above); this file's fine histogram is exactly the `[256 + 1]`
// shape it takes.

// The same suffix sum over the 1024-bin coarse level: deep_gemm's
// `warp_cumsum_histogram<kCoarseBins>`, i.e. its **coarse** arm.
//
// **This used to be a block-wide Hillis-Steele walk and that was the wrong
// shape.**  `kCoarseBins` is 1024 and `kThreads` is 1024, so `histogram[tid]`
// is already a complete per-thread value; the coarse arm needs no
// `kItemsPerLane` at all.  The old comment here justified the walk by "1024
// bins over 64 lanes would need 16 items per lane", which describes the *fine*
// arm (`Bins == kFineBins == 256`, 4 items per lane) -- the coarse arm is
// `kWarps = 1024 / 64 = 16` warps each owning 64 consecutive bins.  The
// `static_assert(kWarps <= kWarpSize)` the arm carries (16 <= 64) holds here,
// and `topk_chunks.cuh`'s own `blockDim >= kCoarseBins` assert is what makes
// `histogram[tid]` readable by every thread.
//
// The barrier count is the point, not the instruction count.  The walk was
// 10 rounds x 2 `__syncthreads()` plus one before the loop = **21 barriers**,
// all of them block-wide, in the merge CTA -- the one CTA that every other CTA
// of the row has already finished waiting for.  This arm is **3**.
// `docs/C500-radix-profile.zh.md` §5.3
// measures the fp32 row's own residual the same way: of its ~104 us
// unaccounted, ~103 us is barrier wait and ~1 us is atomics.  Barriers are the
// cost this path was paying.
//
// The padding slot `histogram[kCoarseBins]` is written here rather than
// before a barrier, because the arm's read of it (`lane < kWarps ? ... : 0`
// covers lanes 16..63, which read `warp_totals`, not `histogram`) never
// touches it and the two `__syncthreads()` inside the arm order it against
// every later reader.
__device__ __forceinline__ void block_cumsum_histogram_1024(
    int (&histogram)[kCoarseBins + kHistogramPadding])
{
    constexpr int kWarpSize = 64;
    constexpr uint64_t kWarpMask = 0xffffffffffffffffULL;
    constexpr int kWarps = kCoarseBins / kWarpSize;
    static_assert(kCoarseBins % kWarpSize == 0, "coarse bins must fill whole waves");
    static_assert(kWarps <= kWarpSize, "the second pass is one warp over kWarps lanes");
    static_assert(kCoarseBins == kThreads,
                  "every thread must own one bin for the coarse arm to be exact");

    __shared__ int warp_totals[kWarps];

    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;

    int warp_suffix = histogram[tid];
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_down_sync(kWarpMask, warp_suffix, offset, kWarpSize);
        if (lane + offset < kWarpSize) warp_suffix += other;
    }
    if (lane == 0) warp_totals[warp] = warp_suffix;
    __syncthreads();

    if (warp == 0) {
        const int warp_total = lane < kWarps ? warp_totals[lane] : 0;
        int block_suffix = warp_total;
#pragma unroll
        for (int offset = 1; offset < kWarpSize; offset <<= 1) {
            const int other = __shfl_down_sync(kWarpMask, block_suffix, offset, kWarpSize);
            if (lane + offset < kWarpSize) block_suffix += other;
        }
        if (lane < kWarps) warp_totals[lane] = block_suffix - warp_total;
    }
    __syncthreads();

    histogram[tid] = warp_suffix + warp_totals[warp];
    if (tid == 0) histogram[kCoarseBins] = 0;
    __syncthreads();
}

// One row of `[0, length)`, ranking `requested_topk` of it.
//
// `output` receives *positions in this window*, which on this path is the row's
// own columns (the window always starts at column 0 here), so the caller's
// contract half can treat them exactly like the split's or `coarse12`'s answer.
// Ties are broken by whichever order the passes produce, exactly as in the
// original: the caller compares as multisets.
//
// Returns false when the row is outside what this kernel serves -- a window no
// longer than `topk`, or a threshold bin wider than `kCandidateCapacity`.  The
// caller turns that into the per-row `-1` sentinel and the row path re-ranks
// it, so a `false` is a coverage statement, never a wrong answer.
// One launcher per kernel -- each owns its kernel's single `<<<>>>`, which is
// what puts the device code and its only launch in the same generated TU.  The
// workspace arrives as `void *` and is cast here, as in
// `launch_topk_chunks_impl` below, so these can be declared before
// `TopKChunksWorkspace` is defined.
template <int ChunkCount>
inline cudaError_t launch_chunks_init(void *workspace, const int32_t *lengths,
                                      int rows, int default_length,
                                      cudaStream_t stream);

template <int ChunkCount>
inline cudaError_t launch_chunks_coarse_hist(
    const float *scores, void *workspace, int32_t *nan_flags, int topk,
    int64_t stride, int default_length, int rows, cudaStream_t stream);

template <int ChunkCount>
inline cudaError_t launch_chunks_compact_refine(
    const float *scores, int32_t *out, void *workspace, int topk, int64_t stride,
    int default_length, int rows, cudaStream_t stream);

template <int ChunkCount>
inline cudaError_t launch_topk_chunks_impl(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    void *workspace, int B, int topk, int64_t stride, int default_length,
    cudaStream_t stream);

// ── the chunked form ───────────────────────────────────────────────────────
//
// The three kernels below are deep_gemm's `topk_chunks_*`, restored after the
// first version of this port dropped them.  That version was one CTA per row
// and it was **slower than the split it was meant to replace** at every width
// (ledger §12.17): 82/116/132 us against deep_gemm's 53/61/63.  The reason is
// not subtle -- a single CTA sweeping 66551 floats twice on a 104-AP part --
// and the conclusion recorded there, that chunking "buys at most 6 CTAs" at
// `b <= 2`, counted the wrong thing: without it the count is **one**.  A
// `select_chunk_count` of 0 therefore means the shape does not reach this form
// at all -- a decline, never a fallback to that one-CTA arm.
//
// `select_chunk_count` is deep_gemm's own, including its constants.  It is a
// *scheduling* heuristic rather than a shape rule: it only adds chunks when the
// last scheduling wave would otherwise be under-filled, which is exactly the
// situation at one or two rows.
//
// **"Including its constants" used to overstate this, and the two missing
// stages were a real divergence.**  The port stopped at `return max_chunks`
// while the original (deep_gemm's `select_chunk_count`) has two more: the largest count that
// still fits a single wave, and -- when no count fits one -- the smallest whose
// tail fills at least half a wave, else the widest tail seen.  Both are
// *scheduling* refinements rather than shape rules, which is why they are worth
// having: the comment above them in the original says the waves/chunks-only
// model misses the fixed histogram, arrival and merge cost that every chunk
// pays regardless of how much of the row it walks.
//
// **At this port's band the two stages are inert, and that is a checked claim
// rather than an assumption.**  `b <= 2` against `num_sms = 104` gives 6 either
// way, and an exhaustive walk of `b in {1,2} x V in [2048, 2e6)` at
// `sm_count in {104, 28, 32}` finds **zero** disagreements -- so no measurement
// in this ledger moves, on any of the three families this repository builds.
//
// It is carried anyway because the divergence is **latent**: `select_chunk_count`
// is reachable at any `b`, and it is *not* inert away from the band -- at
// `sm_count = 104` the two functions disagree on 834 of 1,393 `(b, V)` pairs
// over `b in [1,199]` (e.g. `b18-v66551`: 6 before, 5 after).  Those shapes do
// not reach here today because `f32_chunks_applies` stops at `b <= 2`; a future
// widening of that bound would have been silently mis-sized by a function whose
// comment claimed to be the original's.

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

    // Chunks add initialization, synchronization, and merge work.  Only use
    // them when at least one candidate fills the last scheduling wave better
    // than the single-CTA-per-row launch.  An exact multiple occupies a full
    // last wave, so it never needs chunks.
    const int single_tail = tail_blocks(n_rows);
    int best_chunk_tail = 0;
    for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks)
        best_chunk_tail = max(best_chunk_tail, tail_blocks(n_rows * chunks));
    if (best_chunk_tail <= single_tail) return 0;

    // Within a single wave, use as many chunks as possible to expose row-level
    // parallelism without adding another scheduling wave.
    int single_wave_chunks = 0;
    for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks) {
        const int64_t blocks = n_rows * chunks;
        if (blocks <= num_sms) single_wave_chunks = chunks;
    }
    if (single_wave_chunks != 0) return single_wave_chunks;

    // For multi-wave launches, prefer the smallest chunk count with a
    // sufficiently populated tail.  This accounts for the fixed histogram,
    // arrival, and merge cost that a waves/chunks-only model misses.
    int selected = 0;
    int best_tail = -1;
    for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks) {
        const int64_t blocks = n_rows * chunks;
        const int tail = tail_blocks(blocks);
        if (2 * tail >= num_sms) return chunks;
        if (tail > best_tail) {
            selected = chunks;
            best_tail = tail;
        }
    }
    return selected;
}

// The chunk count is a closed set of four instantiations, and two entries have
// to turn it into one: `launch_typed_chunks` (`routing.cuh`) and
// `chunks_workspace_bytes`.  One switch, so an instantiation added to the set
// is added once -- a count the launcher accepts but the sizer does not would
// size an arena smaller than the kernels write.
template <typename Fn>
inline void dispatch_chunk_count(int chunks, Fn &&fn) {
    switch (chunks) {
        case 3: fn.template operator()<3>(); break;
        case 4: fn.template operator()<4>(); break;
        case 5: fn.template operator()<5>(); break;
        case 6: fn.template operator()<6>(); break;
        default: break;
    }
}

// The shape's chunk count and the launch for it live in
// `deep_select::detail::launch_typed_chunks` (`routing.cuh`): each `NChunks`
// case names a concrete kernel, and a non-template body over them is compiled
// into the device image of every including TU.  The workspace is still the
// caller's (`RowParams::chunks_workspace`), sized by `chunks_workspace_bytes`
// from the same `select_chunk_count`.

// One row's shared state between the three kernels, one entry per row.
//
// `NChunks` is a template parameter so the arrays are fixed-size -- this is a
// workspace, not a heap, and the launch knows which instantiation it is because
// `select_chunk_count` returned its value.  The layout is identical across
// instantiations, which is what lets the dispatcher allocate one byte buffer
// sized for whichever `NChunks` this call resolves to.
//
// `candidate_indices` is deep_gemm's own `alignas(16)`, and it is the *arena*:
// 16 KB, which the refine stages alias as two 8 KB halves.  It lives here in
// the workspace, as **device** memory, and not in shared -- the three kernels
// are separate CTAs, so a shared buffer would not be visible to the one that
// does the merge (see the member's own note below).  Nothing here asks for
// dynamic shared memory: all three launches pass `0`.
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
    size_t bytes = 0;  // 0 chunks: the shape declines, and nothing is staged
    dispatch_chunk_count(
        select_chunk_count((int64_t)batches, (int64_t)vocab_size, (int)sm_count),
        [&]<int N>() { bytes = (size_t)batches * chunks_workspace_bytes_for<N>(); });
    return bytes;
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
    int32_t *__restrict__ nan_flags, int topk, int64_t stride,
    int default_length)
{
    const int chunk = blockIdx.x;
    const int row = blockIdx.y;
    const int tid = threadIdx.x;
    TopKChunksWorkspace<ChunkCount> &workspace = workspaces[row];
    // This row's slot in the caller's per-row scan table.  It is zero when this
    // kernel runs -- zeroed once when the dispatcher grew the table, and
    // cleared by the contract half of every earlier call (see
    // `RowParams::nan_flags`) -- so this kernel only ever raises it.  Null
    // means "the caller is not scanning at all" (`check_nan` false), which is
    // the only case where the contract half does not read the table either.
    //
    // The flag is not decoration and it is not ours to skip: the caller hands
    // this kernel's answer to the contract half as `preselected`
    // (`contract.cuh`, `topk_kernel_radix`'s preselected arm), and that arm
    // **reads** the slot instead of scanning the row:
    //
    //     if constexpr (kPreSelected)
    //         nan_local = params.check_nan && nan_vote != 0;
    //
    // where `nan_vote` is this slot, read and cleared at the top of the
    // contract kernel.
    //
    // So a row this arm answers without writing its slot is served a normal
    // top-k, silently, with `abort_when_nan_found` -- the default -- inert.
    // Raising it here is what makes the route enterable under `check_nan` at
    // all, and it is why the scan is folded into pass 1 below rather than left
    // to a separate pass: `topk_coarse12.cuh:223-245` pays the same predicate on
    // words it is already loading (ledger §10.7, ~3% of that pass), and this
    // kernel is already walking the row twice.
    //
    // The `length <= topk` arm above returns *before* this point and writes no
    // slot.  That is not a gap: the contract half's shortcut arm runs before its
    // NaN block (`contract.cuh`) and returns the whole window
    // as the answer without ever reading a flag.
    int32_t *const row_nan = nan_flags == nullptr ? nullptr : nan_flags + row;
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

    // Pass 1, with the contract half's NaN scan folded into it: one predicate
    // per element on words this kernel is already loading, against a whole
    // separate pass over the row.  Each CTA covers its own chunk and ORs its
    // own slot, so the flag is raised by whichever chunk saw the NaN and the
    // last one to arrive does not have to be the one that did.
    //
    // **Guarded on the caller's table, not unconditional.**  `row_nan ==
    // nullptr` is exactly "the caller is not scanning" (`check_nan` false), and
    // a caller that has already established its input is NaN-free should not
    // pay four bit-tests per 16 bytes for a contract it declined -- the same
    // reason `f32_stage1_fuses_scan()` exists on the split.  The branch is
    // uniform and loop-invariant, so it costs a predicate register.
    //
    // What it costs when it *is* on: measured, ~2.3-3.5% of this route's own
    // time (b1-v66551 69.6 -> 71.8 us, b1-v131072 79.6 -> 82.4 us, snapshot
    // clock, device 2, interleaved rounds).  That is the same order as the ~3%
    // of pass 1 `topk_coarse12.cuh` measured for the same fold (ledger §10.7);
    // it is the price of the contract rather than an artifact of this port.
    const bool scan_nan = row_nan != nullptr;
    bool nan_found = false;
    for (int index = tid; index < prefix; index += kThreads) {
        const float value = chunk_input[index];
        if (scan_nan) nan_found |= is_nan_value(value);
        atomicAdd(&histogram[coarse_bin(value)], 1);
    }
    for (int index = tid; index < vector_length; index += kThreads) {
        const float4 values = vector_input[index];
        if (scan_nan)
            nan_found |= is_nan_value(values.x) || is_nan_value(values.y)
                         || is_nan_value(values.z) || is_nan_value(values.w);
        atomicAdd(&histogram[coarse_bin(values.x)], 1);
        atomicAdd(&histogram[coarse_bin(values.y)], 1);
        atomicAdd(&histogram[coarse_bin(values.z)], 1);
        atomicAdd(&histogram[coarse_bin(values.w)], 1);
    }
    for (int index = tail + tid; index < chunk_length; index += kThreads) {
        const float value = chunk_input[index];
        if (scan_nan) nan_found |= is_nan_value(value);
        atomicAdd(&histogram[coarse_bin(value)], 1);
    }
    // Raised here rather than in the merge because the merge runs in one CTA
    // per row and this predicate is per element: `__syncthreads_or` reduces
    // over this block, and every chunk's block gets its own chance to raise
    // the row's flag.  `atomicOr` rather than a store for the same reason --
    // the other chunks of this row are concurrent with this one.
    if (scan_nan && __syncthreads_or((int)nan_found) != 0) {
        if (tid == 0) atomicOr(row_nan, 1);
    }
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
    //
    // **One pass, not two.**  This used to sum into a `total` local, then
    // re-run the identical `ChunkCount`-deep loop into `histogram`, with the
    // first result discarded (`(void)total`).  The original sums once
    // (`histogram[tid] = total`).  The loop reads
    // `ChunkCount * kCoarseBins` ints of **device** memory -- 24 KB per row at
    // `ChunkCount == 6` -- so the duplicate was a second device-memory pass
    // over the whole coarse workspace, taken in the one CTA every other CTA of
    // the row has already finished waiting for.
    //
    // The `tid < kCoarseBins ? tid : 0` guard the duplicate carried is dead at
    // this width (`kThreads == kCoarseBins`, and the cumsum below asserts it);
    // the padding slot `histogram[kCoarseBins]` is written by the cumsum.
    // No barrier is needed between the write and the cumsum: each thread reads
    // back only its own slot, and the cumsum's own leading `__shfl_down_sync`
    // is wave-synchronous.
    if (tid < kCoarseBins) {
        int total = 0;
#pragma unroll
        for (int source = 0; source < ChunkCount; ++source)
            total += workspace.coarse[source][tid];
        histogram[tid] = total;
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
        // decision -- the route's launcher turns it into the per-row
        // `-1` that sends the row down the row path -- and it must be taken
        // here, before the refine walks anything, because a truncated arena
        // would otherwise answer for an arbitrary subset of the bin.
        workspace.declined = boundary_base > kCandidateCapacity;
        workspace.arrival = 0;
    }
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
    // is the true width.  The caller declines a bin wider than the arena
    // (`workspace.declined`, set by the coarse merge), so the two agree
    // whenever this runs.
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
    warp_cumsum_histogram<kFineBins>(histogram);
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
        warp_cumsum_histogram<kFineBins>(histogram);
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

// The window table: one block per row, 128 threads.
template <int ChunkCount>
inline cudaError_t launch_chunks_init(void *workspace, const int32_t *lengths,
                                      int rows, int default_length,
                                      cudaStream_t stream)
{
    auto *ws = reinterpret_cast<TopKChunksWorkspace<ChunkCount> *>(workspace);
    topk_chunks_init<ChunkCount><<<rows, 2 * 64, 0, stream>>>(ws, lengths, rows,
                                                              default_length);
    return cudaGetLastError();
}

extern template cudaError_t launch_chunks_init<3>(void *, const int32_t *, int, int,
                                                  cudaStream_t);
extern template cudaError_t launch_chunks_init<4>(void *, const int32_t *, int, int,
                                                  cudaStream_t);
extern template cudaError_t launch_chunks_init<5>(void *, const int32_t *, int, int,
                                                  cudaStream_t);
extern template cudaError_t launch_chunks_init<6>(void *, const int32_t *, int, int,
                                                  cudaStream_t);

template <int ChunkCount>
inline cudaError_t launch_chunks_coarse_hist(
    const float *scores, void *workspace, int32_t *nan_flags, int topk,
    int64_t stride, int default_length, int rows, cudaStream_t stream)
{
    auto *ws = reinterpret_cast<TopKChunksWorkspace<ChunkCount> *>(workspace);
    const dim3 grid(ChunkCount, (unsigned)rows);
    topk_chunks_coarse_hist<ChunkCount><<<grid, kThreads, 0, stream>>>(
        scores, ws, nan_flags, topk, stride, default_length);
    return cudaGetLastError();
}

extern template cudaError_t launch_chunks_coarse_hist<3>(
    const float *, void *, int32_t *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_coarse_hist<4>(
    const float *, void *, int32_t *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_coarse_hist<5>(
    const float *, void *, int32_t *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_coarse_hist<6>(
    const float *, void *, int32_t *, int, int64_t, int, int, cudaStream_t);

template <int ChunkCount>
inline cudaError_t launch_chunks_compact_refine(
    const float *scores, int32_t *out, void *workspace, int topk, int64_t stride,
    int default_length, int rows, cudaStream_t stream)
{
    auto *ws = reinterpret_cast<TopKChunksWorkspace<ChunkCount> *>(workspace);
    const dim3 grid(ChunkCount, (unsigned)rows);
    topk_chunks_compact_refine<ChunkCount><<<grid, kThreads, 0, stream>>>(
        scores, out, ws, topk, stride, default_length);
    return cudaGetLastError();
}

extern template cudaError_t launch_chunks_compact_refine<3>(
    const float *, int32_t *, void *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_compact_refine<4>(
    const float *, int32_t *, void *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_compact_refine<5>(
    const float *, int32_t *, void *, int, int64_t, int, int, cudaStream_t);
extern template cudaError_t launch_chunks_compact_refine<6>(
    const float *, int32_t *, void *, int, int64_t, int, int, cudaStream_t);

// The three in order.  Each owns one launch; this sequence owns the ordering.
template <int ChunkCount>
inline cudaError_t launch_topk_chunks_impl(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    void *workspace, int B, int topk, int64_t stride, int default_length,
    cudaStream_t stream)
{
    cudaError_t err = launch_chunks_init<ChunkCount>(workspace, lengths, B,
                                                     default_length, stream);
    if (err != cudaSuccess) return err;
    err = launch_chunks_coarse_hist<ChunkCount>(scores, workspace, nan_flags,
                                                topk, stride, default_length, B,
                                                stream);
    if (err != cudaSuccess) return err;
    return launch_chunks_compact_refine<ChunkCount>(
        scores, out, workspace, topk, stride, default_length, B, stream);
}

}  // namespace topk_select_chunks

