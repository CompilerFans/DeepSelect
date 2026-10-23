// The contract half: `RowParams`, the window shortcut, the NaN bit-pattern
// check, the index offsets, the out-of-band fills and the optional ordered
// emit -- one layer every mode shares, over whichever dataflow the routing
// picked.  The radix row kernel below is what emits the answer; which of the
// two row dataflows it ranks with is the dtype's decision (`radix_select_row`),
// not the routing's.
#ifndef CONTRACT_CUH
#define CONTRACT_CUH

#include <cuda_runtime.h>
#include <maca_bfloat16.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <type_traits>

// The family table, for `ARCH_REGS_PER_AP` -- the budget the `kSkipRow`
// predicate below is sized against.
#include <csrc/structs.h>

// The NaN predicates and the order-preserving key encodes, shared with the
// dataflow headers below.
#include <csrc/maca_kernels/common/primitives.cuh>
// The ported C500 dataflow (see the provenance banner in that file).  Its row
// entries are what `radix_select_row` dispatches to, and the launchers below
// are how every route -- the two splits and the two ported ones -- hands its
// answer to `topk_kernel_radix`.
#include <csrc/maca_kernels/common/radix_core.cuh>

namespace deep_select {

// ── tunables ────────────────────────────────────────────────────────────────
// The public contract: `topk <= 4096` (checked in `topk()`), which the radix
// row entries serve as they come -- they take `topk` at runtime and the staging
// buffer below is sized for the whole range.  The ported core's own `kMaxTopK`
// (2048) is the *kernel arm* limit of its static-k dispatch, not the row
// entries'; the chunked gate stays inside it because its merge only compiles
// the k=512/1024 arms.
constexpr int kMaxTopK = 4096;

struct RowParams {
    const void *input;
    void *output_value;             // null when the caller does not want values
    void *output_index;
    const int32_t *end_ptr;         // null means "whole row"
    const int32_t *idx_offset_ptr;  // null means no offset
    // Chunked path only: the merged column indices the split already ranked,
    // `topk` per row.  Null everywhere else.
    const int32_t *preselected;
    // One entry per row, raised by whichever kernel walked the row before the
    // contract half: the split's stage 1 or a route that answers a row without
    // ranking it -- every one of them reaches it through the dispatcher's
    // `scan_flags` table below.  **Read and cleared by the contract half in the
    // same breath**, at the top of `topk_kernel_radix` -- before any arm can
    // return -- which is what lets the writers OR a plain 1: every slot is
    // zero again by the end of the call, so no per-call `cudaMemsetAsync` is
    // needed on any route, and a raised slot can never carry a previous
    // call's answer into this one.  The tables are zeroed once, when the
    // scratch grows them, so the invariant holds on the first use as well.
    // Null when `check_nan` is off or the row path scans inline.
    int32_t *nan_flags;
    // The per-row scan table: allocated and owned by the dispatcher, handed to
    // every route that scans as its `nan_flags` (both splits included -- a
    // split must not put flags in its own workspace, whose offset is a
    // geometry-dependent expression into process-global scratch, so a slot can
    // hold another shape's candidate index; measured: a b1 v262144 bf16 call
    // returned the sentinel after a b6 v1048576 call).  `needs_scan_table` in
    // `maca_topk.cu` decides which calls get one.  Written by the scanning
    // kernel, read and cleared by the contract half (`nan_flags` above is the
    // field it reaches it through).  Null when nothing scans.
    int32_t *scan_flags;
    // The table `nan_abort_kernel` reads, one entry per row: raised by the
    // contract half for a row it found a NaN in.  Distinct from `scan_flags`
    // rather than sharing it because the contract half is its only writer and
    // writes **every** row -- 0 at the top of `topk_kernel_radix`, 1 in the NaN
    // disposition -- which is what lets it go without a per-call
    // `cudaMemsetAsync`: a slot left over from the previous call cannot be
    // read, because no slot is left over.  (That memset is not free: measured
    // on C500, one `mcMemsetAsync` of this table is ~11 us -- a launch, not a
    // bandwidth cost -- and the calls that would pay it are the small-batch
    // ones that gain nothing from the occupancy this change buys.)
    //
    // "Every row" is a property of the launch, and it is the invariant this
    // table rests on: `topk_kernel_radix` is launched `<<<batches, BLOCK>>>`,
    // one CTA per row, on *every* route -- the split, the coarse12 and the
    // chunks arms each hand their answer to that same launch through
    // `launch_typed_radix`, so none of them answers a row without it.  A route
    // that answered rows itself would have to clear its own rows here, and it
    // would need this table zeroed per call.
    //
    // Null unless the call can abort.
    int32_t *nan_abort_flags;
    // The coarse12 route's staging buffer: `n_rows * topk` int32 columns, the
    // row positions that kernel ranks.  It cannot be `output_index` itself
    // because the public entry's default `indices_type` is int64 and the kernel
    // writes int32 -- through the caller's pointer the two widths interleave.
    // The contract half widens these into `output_index` the way it already
    // widens the split's answer.  Allocated by the dispatcher for any call the
    // route can reach; null otherwise, which the route reads as "not staged"
    // and answers with the row path.
    int32_t *coarse12_cols;
    // The chunks route's per-row workspaces, `n_rows * sizeof(TopKChunksWorkspace<NChunks>)`
    // bytes as `void*` so `structs.h` does not have to see the layout.  Null
    // means the allocation failed and the route answers with the row path.
    void *chunks_workspace;
    // The 16-bit row's skip tables, `n_rows * ceil(vocab_size / 64)` entries
    // each: pass 1 leaves one max bin per 64-element block in `row_block_max`
    // (uint16) and the collect walks `row_block_list` (uint32), the blocks that
    // can hold a candidate, instead of the row.  Both rows are indexed by
    // `vocab_size` -- the allocation's row length, not this row's window -- so
    // the table lines up with the row data it describes.  Null on the calls the
    // gate below leaves out, and on every route that does not walk rows.
    uint16_t *row_block_max;
    uint32_t *row_block_list;
    // The fp32 coarse12 route's own skip tables, `n_rows * ceil(vocab_size/16)`
    // entries each.  Separate from the pair above rather than shared with it:
    // the block is 16 elements here and 64 there, and the stride is recomputed
    // from `vocab_size` on both sides, so one buffer serving two granularities
    // would be two rules for one address.  Same contract otherwise -- pass 1
    // leaves one max bin per block, the collect walks the blocks at or above
    // the threshold (compacted into a list, see `topk_coarse12.cuh`) -- and
    // null on every call the route is not reached from.
    uint16_t *coarse12_block_max;
    uint32_t *coarse12_block_list;
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;
    uint32_t vocab_size;
    uint32_t topk;
    // The AP count every grid-sizing decision below reads, supplied by the
    // caller from `torch.cuda.get_device_properties(...)`.  The *family*
    // constants are compile-time (`csrc/structs.h`) -- this artifact was built
    // for exactly one family and `_binding.py` loads the one the device
    // reports -- but the count itself stays an argument: it is the one
    // grid-sizing input whose value must match the device actually in front of
    // the call, and a caller that hands this entry a tensor on another part
    // gets a refusal rather than a grid sized for the artifact's family.
    // Zero means "unknown", which the entry point refuses outright rather than
    // rounding to a plausible wrong grid.
    uint32_t sm_count;
    int32_t idx_fill;
    float value_fill;
    // `check_nan` is whether the row is scanned at all.  The scan is a whole
    // extra pass over the row and a caller that has already established its
    // input is NaN-free pays it for nothing: measured on C500, 22-32% of the
    // five official fp32 cells (7243.7 -> 5618.1 us at b4096-v129280) -- while
    // a caller that has not still gets the full contract by default.
    //
    // What happens to a row that *is* found poisoned is not the kernel's
    // decision: the contract half raises the row's flag, and `nan_abort_kernel`
    // reads it.  The abort answered here instead cost this kernel 15 MT of
    // registers (see that kernel).
    bool check_nan;
};

// ── key encode / decode ─────────────────────────────────────────────────────
// The value-typed key encode and NaN predicate live in
// `../common/primitives.cuh`, so the row kernels' copies of the same bit
// tricks cannot drift from the contract half's.
using topk_select_common::is_nan_value;
using topk_select_common::refine_bin;


// float -> ValueT.  MACA's `__maca_bfloat16` has no implicit float conversion,
// so the bf16 case needs the explicit intrinsic.
template <typename ValueT>
static __device__ __forceinline__ ValueT float_to_value(float x);
template <>
__device__ __forceinline__ float float_to_value<float>(float x) { return x; }
template <>
__device__ __forceinline__ maca_bfloat16 float_to_value<maca_bfloat16>(float x) {
    return __float2bfloat16(x);
}

// ── NaN test ────────────────────────────────────────────────────────────────
// `is_nan_value<ValueT>` -- the raw-bit-pattern test, its contract in
// `../common/primitives.cuh` -- is pulled in above with the key encode.

// Ascending bitonic sort of `n` (power of two) 64-bit keys over the CTA.
template <int BLOCK>
static __device__ __forceinline__ void bitonic_sort_u64(uint64_t *data, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += BLOCK) {
                const int ixj = i ^ j;
                if (ixj > i) {
                    const bool ascending = ((i & k) == 0);
                    const uint64_t a = data[i];
                    const uint64_t b = data[ixj];
                    if ((a > b) == ascending) {
                        data[i] = b;
                        data[ixj] = a;
                    }
                }
            }
            __syncthreads();
        }
    }
}

// ── ordered emit ────────────────────────────────────────────────────────────
//
// Sorts the first `n_out` slots of `selected` by packing each with its sort key
// and running the CTA-wide bitonic network over the packed words.  The low 12
// bits carry the slot, so ties get a deterministic order and the value/index
// pair stays recoverable (`cmp` is always < vocab and never reaches those bits):
//
//   SI: index ascending  -> (index << 12) | slot
//   SV: value descending -> ((~key) << 12) | slot
//
// `n_pad` is `topk` rounded up to a bitonic power of two; the tail is `~0ull`,
// which sorts past every real key and is never emitted.
//
// `sort_buf` comes from the caller: the two selection kernels place it
// differently in dynamic shared memory (see `radix_layout`).
template <typename ValueT, typename OutIdxT, int BLOCK, bool RV, bool SV>
static __device__ __forceinline__ void emit_ordered(
    uint64_t *sort_buf, const uint32_t *selected, const ValueT *input_row,
    OutIdxT *out_index_row, ValueT *out_value_row, const RowParams &params,
    uint32_t n_out, uint32_t topk, int32_t idx_offset) {
    uint32_t n_pad = 1;
    while (n_pad < topk) n_pad <<= 1;
    for (uint32_t i = threadIdx.x; i < n_pad; i += BLOCK) {
        uint64_t pack = ~0ull;  // pads sort last and are never emitted
        if (i < n_out) {
            const uint32_t src = selected[i];
            // Widened before the complement: the 16-bit row's key is
            // `uint16_t`, and `~` on it would promote through `int` and
            // sign-extend into the pack's high bits.
            const uint64_t cmp = SV
                ? (uint64_t)(~static_cast<uint32_t>(refine_bin(__ldg(input_row + src))))
                : (uint64_t)src;
            pack = (cmp << 12) | (uint64_t)i;
        }
        sort_buf[i] = pack;
    }
    __syncthreads();
    bitonic_sort_u64<BLOCK>(sort_buf, (int)n_pad);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < topk; i += BLOCK) {
        const bool valid = i < n_out;
        const uint32_t slot = (uint32_t)(sort_buf[i] & 0xFFFu);
        const uint32_t src = valid ? selected[slot] : 0u;
        out_index_row[i] =
            valid ? (OutIdxT)((int64_t)src + idx_offset) : (OutIdxT)params.idx_fill;
        if (RV) {
            out_value_row[i] =
                valid ? __ldg(input_row + src) : float_to_value<ValueT>(params.value_fill);
        }
    }
}

// ── the operator kernel (one row per CTA) ───────────────────────────────────
//
// The contract layer over the ported radix selection: two passes over the row,
// whatever the key width.
//
// Dynamic shared memory starts with the core's arena -- `s_input_flat` is an
// `extern __shared__` array in the header, so it can only sit at the base --
// followed by the staging buffer the emit reads.  The ordered emit's scratch is
// only live after the arena is dead, so it aliases it; the worst case
// (topk = kMaxTopK, sorted) is 48 KB, what `kSmemBudgetBytes` reserves.
constexpr size_t kRadixArenaBytes =
    (size_t)topk_select_radix::kSmemInputSize * sizeof(uint32_t);

// The 16-bit row's 12-bit level lays its histogram over the arena, so its lead
// region is the histogram's 16 KB rather than the arena's 14,056 B.  At k=512
// the request goes 16,104 -> 18,432 B and static+dynamic 18,432 -> 20,760,
// still under the 32 KB a second CTA needs; at topk >= 4096 both regimes are
// past it already, so no k loses occupancy to this.
constexpr size_t kCoarse12HistBytes =
    (size_t)topk_select_radix::kCoarse12ArenaEntries * sizeof(uint32_t);

static __device__ __forceinline__ void radix_layout(
    uint8_t *base, uint32_t topk, bool sorted, bool wide, uint32_t *&selected,
    uint64_t *&sort_buf) {
    uint32_t n_pad = 1;
    if (sorted) {
        while (n_pad < topk) n_pad <<= 1;
    }
    const size_t sort_bytes =
        sorted ? (size_t)n_pad * sizeof(uint64_t) : (size_t)0;
    const size_t arena_bytes = wide ? kCoarse12HistBytes : kRadixArenaBytes;
    sort_buf = reinterpret_cast<uint64_t *>(base);
    selected = reinterpret_cast<uint32_t *>(
        base + (sort_bytes > arena_bytes ? sort_bytes : arena_bytes));
}

// Both row entries cover every key length and every k **up to the public
// `kMaxTopK` (4096)**, not `topk_select_radix::kMaxTopK` (2048).  The header's 2048 is the
// bf16 *chunked* entries' arm guard, and its readers
// (`launch_topk_bf16_chunked_k` / `launch_topk_bf16_chunked`) are on the way to
// an arm that instantiates `topk` at compile time.  The split's own gate
// (`chunked_bf16_applies` below) admits only `topk` 512 or 1024, so the guard
// never fires; what the two entries called here cover is every key length and
// every k up to 4096, because `radix_topk_row_bf16_b` takes `topk` at runtime
// and `radix_topk_row_f32` does too, and `radix_layout` sizes `selected` and
// `sort_buf` for it.  A reader who finds the 2048 guard and concludes "2048 is
// the real limit" has found a guard on a route this operator does not take; see
// the note above `deep_select::kMaxTopK` for why the two constants differ.
//
// Both also resolve a threshold bin too large for the arena by re-walking the
// row rather than by ranking a truncated candidate set.
// `check_nan` asks the row kernel to fold the contract's NaN scan into pass 1's
// own load and hand the CTA's vote back through `row_nan`.  It is the caller's
// NaN scan either way; what it replaces is `row_has_nan` walking the whole row
// first, which is a third read of a row that is read twice already.
template <typename ValueT, int BLOCK, bool kSkip = true>
static __device__ __forceinline__ void radix_select_row(
    const ValueT *input_row, uint32_t length, int32_t *out_idx, uint32_t topk,
    bool check_nan, bool *row_nan, uint16_t *block_max = nullptr,
    uint32_t *block_list = nullptr) {
    if constexpr (std::is_same<ValueT, maca_bfloat16>::value) {
        if (check_nan)
            topk_select_radix::radix_topk_row_bf16_b<BLOCK, true, kSkip>(input_row, out_idx,
                                                          length, topk, row_nan,
                                                          block_max, block_list);
        else
            topk_select_radix::radix_topk_row_bf16_b<BLOCK, false, kSkip>(
                input_row, out_idx, length, topk, row_nan, block_max,
                block_list);
    } else {
        static_assert(std::is_same<ValueT, float>::value,
                      "the operator serves bfloat16 and float32 only");
        // The fp32 row picks its own block width, one width for every shape.
        static_assert(BLOCK == topk_select_radix::kBlockSize,
                      "the fp32 row runs at topk_select_radix::kBlockSize");
        // `block_max` / `block_list` are not read here: the summary is a
        // property of the 16-bit row's 12-bit coarse level, and the fp32 row's
        // dataflow (and its multi-round rescan) is a different one.
        if (check_nan)
            topk_select_radix::radix_topk_row_f32<true>(input_row, out_idx, length, topk, 0u,
                                         row_nan);
        else
            topk_select_radix::radix_topk_row_f32<false>(input_row, out_idx, length, topk, 0u,
                                          row_nan);
    }
}

// ── the NaN scan, vectorized ────────────────────────────────────────────────
//
// One compare per element, and the bytes have to be read either way, so this
// pass is pure overhead -- which is why it moves vectors like every other pass
// here rather than one element per load.  Both readers of the row go through
// it: the contract half's window check and the chunked path's `nan_scan_kernel`.
//
// Vector while the row slice is 16-byte aligned (it always is -- rows are
// padded to a 1024-byte stride and chunk bases are multiples of 8 elements),
// then a scalar tail.
static __device__ __forceinline__ bool nan_in_block(const float4 &v) {
    return is_nan_value(v.x) || is_nan_value(v.y) || is_nan_value(v.z) ||
           is_nan_value(v.w);
}

static __device__ __forceinline__ bool nan_in_block(const uint4 &v) {
    const maca_bfloat16 *h = reinterpret_cast<const maca_bfloat16 *>(&v);
    bool found = false;
#pragma unroll
    for (int i = 0; i < 8; i++) found |= is_nan_value(h[i]);
    return found;
}

template <typename ValueT>
static __device__ __forceinline__ bool row_has_nan(const ValueT *row,
                                                   uint32_t start,
                                                   uint32_t end) {
    constexpr int kN = (sizeof(ValueT) == 4) ? 4 : 8;
    using VecT = typename std::conditional<sizeof(ValueT) == 4, float4, uint4>::type;
    bool found = false;
    uint32_t done = start;
    if ((reinterpret_cast<uintptr_t>(row + start) & 15u) == 0) {
        const VecT *vrow = reinterpret_cast<const VecT *>(row + start);
        const uint32_t n_vec = (end - start) / kN;
        for (uint32_t k = threadIdx.x; k < n_vec; k += blockDim.x) {
            found |= nan_in_block(__ldg(vrow + k));
        }
        done = start + n_vec * kN;
    }
    for (uint32_t i = done + threadIdx.x; i < end; i += blockDim.x) {
        found |= is_nan_value<ValueT>(__ldg(row + i));
    }
    return found;
}

// The chunked path's NaN scan, on the split's own grid: one CTA per (row,
// chunk).  The contract kernel is one CTA per row by construction, which on a
// small batch is the whole machine parked on six CTAs reading a 12 MB row; this
// asks the same question at the width the row needs.  `flags` is zero when
// this kernel runs (the table is zeroed once, when the scratch grows it) and a
// row's flag is raised by whichever chunk read the bit pattern; the contract
// half reads and clears it.
//
// **Neither split launches this any more.**  Stage 1 reads exactly these bytes
// at exactly this width, so the scan rides its own load instead: fp32 through
// `topk_select_radix::topk_f32_chunk_stage1_kernel`'s `kNan` (the ledger's §10.7 for the A/B:
// the whole walk goes, for ~3% on the pass that absorbs it), the 16-bit split
// through the stage-1 kernels', whose vote feeds this same `flags` table over
// the same window.  What still reaches it is the fp32 *row* path and the fp32
// A/B arm `DS_FUSE_NAN_OFF=1`, which is why it is not deleted.
constexpr int kScanBlock = 256;

template <typename ValueT>
__global__ __launch_bounds__(kScanBlock) void nan_scan_kernel(
    const void *input, const int32_t *lengths, int64_t stride_elems,
    uint32_t vocab_size, uint32_t chunks, int32_t *flags) {
    const uint32_t row = blockIdx.x / chunks;
    const uint32_t chunk = blockIdx.x - row * chunks;
    const uint32_t length = (uint32_t)__ldg(lengths + row);
    // The same chunk geometry the split walks, so the two cover one window.
    const uint32_t raw = (vocab_size + chunks - 1) / chunks;
    const uint32_t chunk_size = (raw + 7u) / 8u * 8u;
    const uint32_t start = chunk * chunk_size;
    if (start >= length) return;  // uniform across the CTA
    const uint32_t end = start + chunk_size < length ? start + chunk_size : length;
    const ValueT *row_ptr = (const ValueT *)((const char *)input +
                                             (uint64_t)row * stride_elems *
                                                 sizeof(ValueT));
    const bool found = row_has_nan(row_ptr, start, end);
    if (__syncthreads_or((int)found) != 0 && threadIdx.x == 0) {
        atomicOr(flags + row, 1);
    }
}

// The grid is the caller's: one CTA per (row, chunk), so the launch is
// `batches * chunks` blocks wide and the kernel's own `blockIdx.x` splits it.
// Not cached in a `static bool` init like the kernel launchers below -- there
// is no `cudaFuncSetAttribute` to do once, so there is nothing to cache.
template <typename ValueT>
inline void launch_nan_scan(const void *input, const int32_t *lengths,
                            int64_t stride_elems, uint32_t vocab_size,
                            uint32_t chunks, int32_t *flags, uint32_t batches,
                            cudaStream_t stream) {
    nan_scan_kernel<ValueT><<<batches * chunks, kScanBlock, 0, stream>>>(
        input, lengths, stride_elems, vocab_size, chunks, flags);
}

extern template void launch_nan_scan<float>(
    const void *, const int32_t *, int64_t, uint32_t, uint32_t, int32_t *,
    uint32_t, cudaStream_t);

// The abort half of the NaN contract, in its own kernel because the contract
// half must not carry it: `printf` + `__trap` answered inline costs the row
// kernel **15 MT of registers on C500** (37 -> 22, one source line at a time),
// and a 1024-thread block stops fitting two CTAs per SM at 32 MT, so the block
// costs the whole kernel a resident CTA on every call, including the calls that
// never reach it.  The `printf` is 13 of the 15 -- dropping it alone takes the
// kernel to 24 MT, dropping the `__trap` alone changes nothing.  With the block
// moved here the row kernel runs two CTAs/SM, and that is the difference
// between 9,310 us and 6,144 us on `b4096 V524288 k512` bf16.
//
// `flags` is `RowParams::nan_abort_flags`: the contract half clears a row's
// entry as the kernel starts and raises it in the NaN disposition, so one
// thread per row here reads this call's answer and the first raised flag prints
// and traps.  A `check_nan` call therefore aborts on the row that poisoned it
// rather than on the row that got scheduled first.
constexpr int kNanAbortBlock = 128;

template <int kBlock>
__global__ __launch_bounds__(kBlock) void nan_abort_kernel(
    const int32_t *flags, uint32_t batches) {
    const uint32_t row = (uint32_t)blockIdx.x * kBlock + threadIdx.x;
    if (row < batches && __ldg(flags + row) != 0) {
        printf("[deep_select] NaN detected. Aborting.\n");
        __trap();
    }
}

template <int kBlock>
inline void launch_nan_abort(const int32_t *flags, uint32_t batches,
                             cudaStream_t stream) {
    nan_abort_kernel<kBlock><<<(batches + kBlock - 1) / kBlock, kBlock, 0,
                               stream>>>(flags, batches);
}

extern template void launch_nan_abort<kNanAbortBlock>(const int32_t *, uint32_t,
                                                      cudaStream_t);

// The seven axes the row kernel is compiled for, as one type.  A configuration
// tuple is the unit of instantiation: one tuple, one generated `instantiations/
// *.cu`, one explicit instantiation of `launch_radix_config<Cfg>` in that file
// (see `scripts/generate_instantiations.py`), and one line in `kRadixTable`
// in `maca_topk.cu` (not in this header -- see the note at its definition).
//
// - `BLOCK` is `topk_select_radix::kBlockSize`, or `topk_select_radix::kLongRowBlockSize`
//   for the regime the ported dispatcher picks the wider block for (see
//   `needs_long_row_bf16`).  Only the 16-bit row has a wide-block arm.
// - `PRE` is for the chunked path: the two-stage split over a long row has
//   already ranked it and left its answer in the workspace (raw column indices,
//   `topk` of them per row, plus the NaN flag its own wider scan raised), so
//   this kernel skips selection and starts from the emit -- the shortcut, the
//   NaN contract, the fills, the offsets and the ordering are the same code
//   either way.  That is what keeps `return_value` / `sorted_index` /
//   `sorted_value` / int64 indices / ragged windows on the split path free of
//   new cases.
template <typename ValueT_, typename OutIdxT_, int BLOCK_, bool SI_, bool RV_,
          bool SV_, bool PRE_>
struct TopkRadixConfig {
    using ValueT = ValueT_;
    using OutIdxT = OutIdxT_;
    static constexpr int BLOCK = BLOCK_;
    static constexpr bool SI = SI_;
    static constexpr bool RV = RV_;
    static constexpr bool SV = SV_;
    static constexpr bool PRE = PRE_;
};

// The dispatch key's two halves: the entry point reads `value_dtype` /
// `index_dtype` off the tensors (`float32 ? 0 : 1` and the output index width),
// and the table below has to be keyed the same way.
template <typename ValueT>
struct radix_value_dtype;
template <>
struct radix_value_dtype<float> {
    static constexpr int value = 0;
};
template <>
struct radix_value_dtype<maca_bfloat16> {
    static constexpr int value = 1;
};
template <typename OutIdxT>
struct radix_index_dtype;
template <>
struct radix_index_dtype<int32_t> {
    static constexpr int value = 0;
};
template <>
struct radix_index_dtype<int64_t> {
    static constexpr int value = 1;
};

template <typename Cfg>
__global__ __launch_bounds__(Cfg::BLOCK) void topk_kernel_radix(RowParams params) {
    using ValueT = typename Cfg::ValueT;
    using OutIdxT = typename Cfg::OutIdxT;
    constexpr int BLOCK = Cfg::BLOCK;
    constexpr bool SI = Cfg::SI;
    constexpr bool RV = Cfg::RV;
    constexpr bool SV = Cfg::SV;
    constexpr bool kPreSelected = Cfg::PRE;
    const int tid = threadIdx.x;
    const uint32_t row = blockIdx.x;

    __shared__ int32_t row_offset;
    // This row's NaN vote, as the kernel that walked the row left it.  Read
    // and cleared together at the top of the kernel -- see `nan_flags` in
    // `RowParams`.
    __shared__ int32_t nan_vote;
    extern __shared__ uint8_t arena_raw[];
    uint32_t *selected;
    uint64_t *sort_buf;
    radix_layout(arena_raw, params.topk, SI || SV,
                 std::is_same<ValueT, maca_bfloat16>::value, selected, sort_buf);

    if (tid == 0) {
        row_offset =
            params.idx_offset_ptr ? __ldg(params.idx_offset_ptr + row) : 0;
        // Every row clears its own abort flag here, before anything in this
        // CTA can return, so `nan_abort_kernel` reads this call's answer and
        // the table never needs zeroing between calls (see
        // `RowParams::nan_abort_flags`).
        if (params.nan_abort_flags != nullptr) params.nan_abort_flags[row] = 0;
        // The vote a split or selector kernel raised for this row: read, then
        // clear.  The clear is what lets the writers OR a plain 1 -- the slot
        // is zero again before this call's kernel completes, so no route needs
        // a per-call `cudaMemsetAsync` and a previous call's answer can never
        // be read as this one's.  The `length <= topk` shortcut below returns
        // without looking, and rows that short are never NaN-checked, but the
        // clear still has to happen here rather than at the read: this is the
        // only point on every path, and a slot left raised would poison the
        // next call that uses this row.
        if (params.check_nan && params.nan_flags != nullptr) {
            nan_vote = params.nan_flags[row];
            params.nan_flags[row] = 0;
        } else {
            nan_vote = 0;
        }
    }
    __syncthreads();
    const int32_t idx_offset = row_offset;

    const uint32_t length = params.end_ptr ? (uint32_t)__ldg(params.end_ptr + row)
                                           : params.vocab_size;
    const uint32_t topk = params.topk;
    const ValueT *input_row = (const ValueT *)((const char *)params.input +
                                              (uint64_t)row * params.stride_input_batch);
    OutIdxT *out_index_row = (OutIdxT *)((char *)params.output_index +
                                        (uint64_t)row * params.stride_output_index_batch);
    ValueT *out_value_row =
        RV ? (ValueT *)((char *)params.output_value +
                        (uint64_t)row * params.stride_output_value_batch)
           : nullptr;

    // Shortcut contract: the window is no longer
    // than k, so the whole window is the answer.  Index order is what
    // `sorted_index` asks for and what the modes that leave the order
    // unspecified accept; `sorted_value` still goes through the ordered emit.
    if (length <= topk) {
        if constexpr (SV) {
            for (uint32_t i = tid; i < length; i += BLOCK) selected[i] = i;
            __syncthreads();
            emit_ordered<ValueT, OutIdxT, BLOCK, RV, SV>(
                sort_buf, selected, input_row, out_index_row, out_value_row,
                params, length, topk, idx_offset);
        } else {
            for (uint32_t i = tid; i < topk; i += BLOCK) {
                const bool valid = i < length;
                out_index_row[i] =
                    valid ? (OutIdxT)((int64_t)i + idx_offset) : (OutIdxT)params.idx_fill;
                if (RV) {
                    out_value_row[i] =
                        valid ? __ldg(input_row + i) : float_to_value<ValueT>(params.value_fill);
                }
            }
        }
        return;
    }

    // ── NaN detection ───────────────────────────────────────────────────────
    // A route that answered this row without ranking it left its vote in the
    // per-row table -- read and cleared at the top of this kernel, into
    // `nan_vote`.  The row path instead folds the scan into pass 1 below, so
    // either way the input is walked exactly once.
    //
    // `params.check_nan` is uniform across the grid, so skipping the
    // `__syncthreads_or` skips it in every thread of every CTA -- no barrier is
    // left half-entered.  When it is false the flag tables are neither written
    // nor read.
    bool nan_local = false;
    // The row kernel folds the scan into pass 1 and writes the vote here; only
    // the preselected arm has it already, from the flag table the earlier
    // kernel raised.  Asking `row_has_nan` as well would walk the row a third
    // time.
    const bool fold_nan = params.check_nan && !kPreSelected;
    if constexpr (kPreSelected) {
        nan_local = params.check_nan && nan_vote != 0;
        if (nan_local) {
            if (tid == 0) {
                out_index_row[0] = (OutIdxT)0x3F3F3F3F;
                if (params.nan_abort_flags != nullptr)
                    params.nan_abort_flags[row] = 1;
            }
            return;
        }
    }

    // ── selection ───────────────────────────────────────────────────────────
    // Exactly `topk` slots come out filled: everything above the threshold bin
    // plus the first `topk - excess` of its members, which are bit-identical.
    // `length > topk` holds here, so the count is `topk` and not one less.
    bool rerank = false;
    if constexpr (kPreSelected) {
        // The split already ranked this row; take its answer and check it is
        // one the merge could fill.  A slot the merge could not fill (-1: every
        // value in that window is at or below the bf16 floor the chunk stage
        // pads with) sends the whole row back through the row dataflow, so a
        // floor-valued row is answered by the same code as any other and never
        // from an empty slot.
        const int32_t *merged = params.preselected + (uint64_t)row * topk;
        for (uint32_t i = tid; i < topk; i += BLOCK) {
            const int32_t src = merged[i];
            rerank |= (src < 0);
            selected[i] = (uint32_t)src;
        }
        rerank = __syncthreads_or((int)rerank) != 0;
    }
    if (!kPreSelected || rerank) {
        // This row's slice of the skip tables, on the tables' own row length
        // (`vocab_size`), so a row with a shorter window still finds its own.
        const uint32_t blocks_per_row = (params.vocab_size + 63u) / 64u;
        uint16_t *row_block_max =
            params.row_block_max != nullptr
                ? params.row_block_max + (uint64_t)row * blocks_per_row : nullptr;
        uint32_t *row_block_list =
            params.row_block_list != nullptr
                ? params.row_block_list + (uint64_t)row * blocks_per_row : nullptr;
        // `kSkip` is `true` for the 32-bit-index row on every family, and for
        // the 64-bit row only where the register file can afford the summary:
        // the tables cost the body ~3 MT over the bare one, and a 1024-thread
        // block keeps its second CTA per SM at <= 64 MT -- within reach of the
        // 128 KiB-regfile parts (measured 131072 regs/AP on a C600U,
        // `structs.h`'s `ARCH_REGS_PER_AP`), not of the C500's 65536.  The
        // host allocates the tables under the same predicate
        // (`skip_tables` in `maca_topk.cu`) -- the two sites must agree, or an
        // allocated-but-unread table is a walk that reads a summary nothing
        // wrote.
        constexpr bool kSkipRow = std::is_same<OutIdxT, int32_t>::value ||
                                  ARCH_REGS_PER_AP >= 131072;
        radix_select_row<ValueT, BLOCK, kSkipRow>(
            input_row, length, (int32_t *)selected, topk, fold_nan, &nan_local,
            row_block_max, row_block_list);
    }

    // The fold's vote lands only now, after the kernel has ranked the row.  The
    // slot it owns is row 0's and the disposition is the one the preselected
    // arm ran above; what differs is when the answer arrived.  Neither arm
    // aborts from here: the flag raised below is what `nan_abort_kernel` reads,
    // and the abort belongs in that kernel rather than in this one (see it for
    // the 15 MT this costs when inlined).
    if (nan_local) {
        if (tid == 0) {
            out_index_row[0] = (OutIdxT)0x3F3F3F3F;
            if (params.nan_abort_flags != nullptr)
                params.nan_abort_flags[row] = 1;
        }
        return;
    }

    // ── emit ────────────────────────────────────────────────────────────────
    const uint32_t n_out = topk;

    if constexpr (!SI && !SV) {
        for (uint32_t i = tid; i < topk; i += BLOCK) {
            const uint32_t src = selected[i];
            out_index_row[i] = (OutIdxT)((int64_t)src + idx_offset);
            if (RV) out_value_row[i] = __ldg(input_row + src);
        }
        return;
    }
    if constexpr (SI || SV) {
        emit_ordered<ValueT, OutIdxT, BLOCK, RV, SV>(
            sort_buf, selected, input_row, out_index_row, out_value_row, params,
            n_out, topk, idx_offset);
    }
}

// ── host-side launch ────────────────────────────────────────────────────────
namespace detail {

// The dynamic-smem ceiling every mode is configured with.  It is the retired
// byte-wise path's `Arena` (kMaxTopK * (4 + 8) bytes), which was the worst case
// over all of them; `cudaFuncSetAttribute` only has to admit what a launch
// actually asks for, and every mode asks for less than this.
constexpr size_t kSmemBudgetBytes =
    (size_t)kMaxTopK * (sizeof(uint32_t) + sizeof(uint64_t));

// Bytes the radix path reserves: the core arena (or the ordered scratch, which
// aliases it and can be larger) followed by the staging buffer.
//
// The staging buffer holds the `topk` selected indices the core writes (the
// shortcut path, the merge hand-off and the row selector all fill exactly
// `topk` entries), so it is sized by `topk` and not by the `kMaxTopK` ceiling.
// At k=512 that is 2 KB instead of 16 KB, and the difference decides the
// occupancy the launch gets: the static shared state plus the arena plus this
// buffer has to stay under half of `smemPerSM` for a second CTA to be resident
// (32 KB static+dynamic per CTA on C500's 64 KB SM).
inline size_t radix_smem_bytes(uint32_t topk, bool sorted, bool wide) {
    uint32_t n_pad = 1;
    if (sorted) {
        while (n_pad < topk) n_pad <<= 1;
    }
    const size_t sort_bytes =
        sorted ? (size_t)n_pad * sizeof(uint64_t) : (size_t)0;
    const size_t arena_bytes = wide ? kCoarse12HistBytes : kRadixArenaBytes;
    const size_t lead = sort_bytes > arena_bytes ? sort_bytes : arena_bytes;
    return lead + sizeof(uint32_t) * topk;
}

// One launch site for one configuration tuple: the row path selects, the
// chunked path picks up the indices the split left in `params.preselected`.
// Both dtypes have a split (16-bit above, fp32 below) and the preselected arm
// is dtype-agnostic -- it reads `preselected` / `nan_flags` and skips selection
// -- so the two differ only in the config's `PRE` member, which is why the
// dispatcher below keys the table on it rather than branching.
//
// **The body owns the launch, and that is what makes a tuple one file.** The
// explicit instantiation in `instantiations/*.cu` defines this function, so the
// kernel's device code and the only launch that reaches it are compiled
// together; the `extern template` below keeps this header's users from
// instantiating it a second time, which would duplicate that device code.
template <typename Cfg>
inline void launch_radix_config(const RowParams &params, uint32_t batches,
                                cudaStream_t stream, size_t smem) {
    // Every mode is configured with the one ceiling; each launch then asks for
    // exactly what its own mode needs (`radix_smem_bytes`).
    const cudaError_t rc = cudaFuncSetAttribute(
        (const void *)topk_kernel_radix<Cfg>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBudgetBytes);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[deep_select] radix smem attribute: %s\n",
                     cudaGetErrorString(rc));
    }
    topk_kernel_radix<Cfg><<<batches, Cfg::BLOCK, smem, stream>>>(params);
}

// ── the launcher lookup, defined in `maca_topk.cu` ─────────────────────────
//
// The table that maps a configuration tuple to its launcher lives in
// `maca_topk.cu`, not here, and the reason is an artifact-size one: every
// entry takes `&launch_radix_config<Cfg>`, and a translation unit that merely
// *sees* that table instantiates all of its rows in the device pass -- an
// `extern template` is honored by the host pass and ignored there, so a TU
// including this header with no table of its own still carried all 180
// `topk_kernel_radix` kernels (measured 2026-10-23: a TU that defines one
// launcher yielded 5.2 MB of device bitcode holding 180 of them).  With the
// table out of this header, an instantiation TU's device image holds the one
// kernel its `instantiations/*.cu` names.  Keep the table there.
//
// The two spellings of the key are the same number by construction: the table's
// `value_dtype` / `index_dtype` come from the trait above, and the entry point
// derives its pair from the tensor dtypes the same way.
using radix_launcher_fn = void (*)(const RowParams &, uint32_t, cudaStream_t,
                                   size_t);

radix_launcher_fn radix_launcher_for(int value_dtype, int index_dtype,
                                     int block, bool si, bool rv, bool sv,
                                     bool pre);

template <typename ValueT, typename OutIdxT>
void launch_typed_radix(const RowParams &params, uint32_t batches,
                        cudaStream_t stream, bool sorted_index,
                        bool sorted_value, bool return_value, int block,
                        bool preselected = false) {
    const size_t smem = radix_smem_bytes(
        params.topk, sorted_index || sorted_value,
        std::is_same<ValueT, maca_bfloat16>::value);
    // The fp32 row has a single width, so only the 16-bit row selects between
    // the two blocks -- and it selects `kLongRowBlockSize` **or** the default,
    // not "whatever `block` says": a caller that passes anything else gets
    // `kBlockSize`, which is what keeps `block` a two-valued axis rather than
    // an open one.  A width outside the table would find no row at all.
    const int width =
        std::is_same<ValueT, maca_bfloat16>::value &&
                block == (int)topk_select_radix::kLongRowBlockSize
            ? (int)topk_select_radix::kLongRowBlockSize
            : (int)topk_select_radix::kBlockSize;
    // `sorted_value` implies `return_value` and is exclusive with
    // `sorted_index` (both checked at the entry), so of the eight combinations
    // these three expressions produce only the five the table carries.
    const bool si = sorted_index || sorted_value;
    const bool sv = sorted_value;
    const bool rv = return_value || sorted_value;
    radix_launcher_fn launch = radix_launcher_for(
        radix_value_dtype<ValueT>::value, radix_index_dtype<OutIdxT>::value,
        width, si, rv, sv, preselected);
    // A miss means this image's table and the expressions above disagree, and
    // the call then has no correct output to produce -- so it fails loudly
    // rather than returning with the caller's buffer unwritten.
    if (launch == nullptr) {
        std::fprintf(stderr,
                     "[deep_select] no radix instantiation for "
                     "{value_dtype=%d, index_dtype=%d, block=%d, si=%d, rv=%d, "
                     "sv=%d, pre=%d}\n",
                     radix_value_dtype<ValueT>::value,
                     radix_index_dtype<OutIdxT>::value, width, (int)si, (int)rv,
                     (int)sv, (int)preselected);
        std::abort();
    }
    launch(params, batches, stream, smem);
}

}  // namespace detail

}  // namespace deep_select

#endif  // CONTRACT_CUH
