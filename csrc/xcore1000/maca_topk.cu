// MACA-native row-wise top-K for DeepSelect -- the shipping backend.
//
// Implements the `topk` contract (`deep_select.interface.topk`) on primitives
// that exist on MACA.  One contract layer is shared by every mode: the
// `length <= topk` shortcut, the NaN bit-pattern check, the index offsets, the
// out-of-band fills, and the optional ordered emit.  One selection dataflow
// feeds it for both dtypes -- the ported radix core in `radix_core.cuh` (see
// its provenance banner): one histogram pass over the high key byte plus one
// vectorized collect pass that refines the threshold bin in shared memory, so
// two passes over the row whatever the key width.  The 16-bit row walks a bf16
// key, the 32-bit row an fp32 one; the dtype picks the row entry, not the
// dataflow.
//
// Two things to know before editing:
//
//   * Portable primitives only: `__shfl_down_sync`, `atomicAdd`, `__ldg`,
//     `__syncthreads`, `__syncthreads_or`.  No inline asm, no TMA, no mbarrier,
//     no cluster.
//   * The wave is 64 lanes, not 32.  Any mask or `/ 32` inherited from the
//     CUDA-era kernels is suspect -- see `csrc/xcore1600/` for what that looks
//     like when it is wrong.
//
// The upstream `v3` (bf16) and `v3_fp32` trees are ported under `csrc/xcore1600/`
// and are NOT built; `v3_cluster` is deleted (MACA has no cluster launch), so
// the shapes it served fall through to this kernel -- the same path every other
// shape takes, with `topk <= 1024 <= 4096` and no vocabulary bound, so they are
// served, only without the cluster-specific scheduling.

#include <cuda_runtime.h>
// [MACA] 数据类型取 MACA 原生头，不经过 cu-bridge 的 <cuda_bf16.h> 兼容层
// （那里只是 `typedef maca_bfloat16 __nv_bfloat16;`）。本文件其余部分只用
// maca_bfloat16 / maca_bfloat162 这两个原生名字。
#include <maca_bfloat16.h>

#include <cstdint>
#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <vector>

#include "structs.h"
// The ported C500 dataflow (see the provenance banner in that file).  It is
// included here so that this translation unit -- the one `setup.py` builds --
// is what proves it compiles under the mxcc/cu-bridge toolchain, and so that
// the bf16 selection path below is a header-only dependency.
#include "radix_core.cuh"
#include "dg_coarse12.cuh"
#include "dg_chunks.cuh"

namespace deep_select_maca {

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
    // `topk` per row, and one NaN flag per row raised by the split's own scan.
    // Null everywhere else.
    const int32_t *preselected;
    const int32_t *nan_flags;
    // The per-row scan table the *other* new route needs: unlike `nan_flags`
    // (owned by whichever split built it, inside that split's workspace), this
    // one is allocated and zeroed by the dispatcher for any call that can
    // reach a route answering a row without ranking it -- currently the
    // coarse12 arm, which folds the scan into its own pass 1.  Written by that
    // kernel, read by the contract half.  Null when nothing scans.
    int32_t *scan_flags;
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
    // `check_nan` is whether the row is scanned at all; `abort_on_nan` is what
    // happens once one is found, and is inert when nothing scans.  Two flags
    // rather than one because the scan is a whole extra pass over the row and a
    // caller that has already established its input is NaN-free pays it for
    // nothing: measured on C500, 22-32% of the five official fp32 cells
    // (7243.7 -> 5618.1 us at b4096-v129280) -- while a caller that has not
    // still gets the full contract by default.
    bool check_nan;
    bool abort_on_nan;
};

// ── key encode / decode ─────────────────────────────────────────────────────
template <typename ValueT>
static __device__ __forceinline__ uint32_t key_of(ValueT v);

template <>
__device__ __forceinline__ uint32_t key_of<float>(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

template <>
__device__ __forceinline__ uint32_t key_of<maca_bfloat16>(maca_bfloat16 v) {
    const uint32_t bits = (uint32_t)__bfloat16_as_ushort(v);
    return (bits & 0x8000u) ? (~bits & 0xFFFFu) : (bits | 0x8000u);
}


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
//
// Test the raw BIT PATTERN, not the key.  The order-preserving encode sends the
// two signed NaNs to opposite ends of the key space, so key comparison catches
// only the single encoding 0x7FFFFFFF; a 16-bit bf16 key cannot even equal a
// 32-bit all-ones constant.  Exponent all-ones with a non-zero payload is
// exactly the NaN set, either sign, quiet or signaling.
//
// (`v != v` would be the obvious spelling but the build enables
// `--use_fast_math`.)
template <typename ValueT>
static __device__ __forceinline__ bool is_nan_value(ValueT v);

template <>
__device__ __forceinline__ bool is_nan_value<float>(float v) {
    const uint32_t bits = __float_as_uint(v);
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

template <>
__device__ __forceinline__ bool is_nan_value<maca_bfloat16>(maca_bfloat16 v) {
    const uint32_t bits = (uint32_t)__bfloat16_as_ushort(v);
    return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0u;
}

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
            const uint64_t cmp = SV
                ? (uint64_t)(~key_of<ValueT>(__ldg(input_row + src)))
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
    (size_t)rk::kSmemInputSize * sizeof(uint32_t);

// The 16-bit row's 12-bit level lays its histogram over the arena, so its lead
// region is the histogram's 16 KB rather than the arena's 14,056 B.  At k=512
// the request goes 16,104 -> 18,432 B and static+dynamic 18,432 -> 20,760,
// still under the 32 KB a second CTA needs; at topk >= 4096 both regimes are
// past it already, so no k loses occupancy to this.
constexpr size_t kCoarse12HistBytes =
    (size_t)rk::kCoarse12ArenaEntries * sizeof(uint32_t);

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

// Both row entries cover every key length and every k up to `rk::kMaxTopK`, and
// both resolve a threshold bin too large for the arena by re-walking the row
// rather than by ranking a truncated candidate set.
template <typename ValueT, int BLOCK>
static __device__ __forceinline__ void radix_select_row(
    const ValueT *input_row, uint32_t length, int32_t *out_idx, uint32_t topk) {
    if constexpr (std::is_same<ValueT, maca_bfloat16>::value) {
        rk::radix_topk_row_bf16_b<BLOCK>(input_row, out_idx, length, topk);
    } else {
        static_assert(std::is_same<ValueT, float>::value,
                      "the operator serves bfloat16 and float32 only");
        // The fp32 row picks its own block width, one width for every shape.
        static_assert(BLOCK == rk::kBlockSize,
                      "the fp32 row runs at rk::kBlockSize");
        rk::radix_topk_row_f32(input_row, out_idx, length, topk);
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
// asks the same question at the width the row needs.  `flags` is zeroed by the
// caller and a row's flag is raised by whichever chunk read the bit pattern.
//
// **The fp32 split no longer launches this.**  Stage 1's pass 1 reads exactly
// these bytes at exactly this width, so the scan rides its own load instead
// (`rk::topk_f32_chunk_stage1_kernel`'s `kNan`, and the ledger's §10.7 for the
// A/B: the whole walk goes, for ~3% on the pass that absorbs it).  What is
// left here is the 16-bit split's scan, the fp32 row path's, and the A/B arm
// `DS_FUSE_NAN_OFF=1` reaches -- which is why it is not deleted.
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

// BLOCK is `rk::kBlockSize`, or `rk::kLongRowBlockSize` for the regime the
// ported dispatcher picks the wider block for (see `needs_long_row_bf16`).
//
// `kPreSelected` is for the chunked path: the two-stage split over a long row
// has already ranked it and left its answer in the workspace (raw column
// indices, `topk` of them per row, plus the NaN flag its own wider scan raised),
// so this kernel skips selection and starts from the emit -- the shortcut, the
// NaN contract, the fills, the offsets and the ordering are the same code
// either way.  That is what keeps `return_value` / `sorted_index` /
// `sorted_value` / int64 indices / ragged windows on the split path free of
// new cases.
template <typename ValueT, typename OutIdxT, int BLOCK, bool SI, bool RV, bool SV,
          bool kPreSelected = false>
__global__ __launch_bounds__(BLOCK) void topk_kernel_radix(RowParams params) {
    const int tid = threadIdx.x;
    const uint32_t row = blockIdx.x;

    __shared__ int32_t row_offset;
    extern __shared__ uint8_t arena_raw[];
    uint32_t *selected;
    uint64_t *sort_buf;
    radix_layout(arena_raw, params.topk, SI || SV,
                 std::is_same<ValueT, maca_bfloat16>::value, selected, sort_buf);

    if (tid == 0) {
        row_offset =
            params.idx_offset_ptr ? __ldg(params.idx_offset_ptr + row) : 0;
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
    // The chunked path has already answered this question across the whole row
    // with the machine busy (`nan_scan_kernel`), which is the difference that
    // matters when the batch is small: this CTA alone would be the only thing
    // reading the row.
    //
    // `params.check_nan` is uniform across the grid, so skipping the
    // `__syncthreads_or` skips it in every thread of every CTA -- no barrier is
    // left half-entered.  When it is false the flag tables are neither written
    // (`nan_scan_kernel` is not launched) nor read (this branch), which is why
    // the chunked arm can skip its memset as well.
    bool nan_local = false;
    if constexpr (kPreSelected) {
        nan_local = params.check_nan && params.nan_flags[row] != 0;
    } else if (params.check_nan) {
        nan_local = row_has_nan(input_row, 0, length);
        nan_local = __syncthreads_or((int)nan_local) != 0;
    }
    if (nan_local) {
        if (params.abort_on_nan) {
            if (tid == 0) printf("[deep_select] NaN detected. Aborting.\n");
            __trap();
        }
        if (tid == 0) out_index_row[0] = (OutIdxT)0x3F3F3F3F;
        return;
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
        radix_select_row<ValueT, BLOCK>(input_row, length, (int32_t *)selected, topk);
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


template <typename ValueT, typename OutIdxT, int BLOCK, bool SI, bool RV, bool SV,
          bool PRE = false>
inline void set_radix_attr() {
    // Every mode is configured with the one ceiling; each launch then asks for
    // exactly what its own mode needs (`radix_smem_bytes`).
    const cudaError_t rc = cudaFuncSetAttribute(
        (const void *)topk_kernel_radix<ValueT, OutIdxT, BLOCK, SI, RV, SV, PRE>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBudgetBytes);
    if (rc != cudaSuccess) {
        std::fprintf(stderr, "[deep_select] radix smem attribute: %s\n",
                     cudaGetErrorString(rc));
    }
}

// One launch site for both dataflows of a mode: the row path selects, the
// chunked path picks up the indices the split left in `params.preselected`.
// Both dtypes have a split (16-bit above, fp32 below) and the preselected arm
// is dtype-agnostic -- it reads `preselected` / `nan_flags` and skips
// selection -- so there is no `if constexpr` narrowing it here.
template <typename ValueT, typename OutIdxT, int BLOCK, bool SI, bool RV, bool SV>
inline void launch_radix(const RowParams &params, uint32_t batches,
                         cudaStream_t stream, size_t smem, bool preselected) {
    if (preselected) {
        set_radix_attr<ValueT, OutIdxT, BLOCK, SI, RV, SV, true>();
        topk_kernel_radix<ValueT, OutIdxT, BLOCK, SI, RV, SV, true>
            <<<batches, BLOCK, smem, stream>>>(params);
    } else {
        set_radix_attr<ValueT, OutIdxT, BLOCK, SI, RV, SV>();
        topk_kernel_radix<ValueT, OutIdxT, BLOCK, SI, RV, SV, false>
            <<<batches, BLOCK, smem, stream>>>(params);
    }
}

// The ported dispatcher's long-row regime is the only thing that differs
// between the two block widths (`rk::needs_long_row_bf16`); the batch is the
// launch's row count, one row per CTA.
inline int radix_block_for(uint32_t batches, uint32_t vocab_size, uint32_t topk) {
    const rk::TopKConfig cfg{(int)batches, (int)vocab_size, (int)topk};
    return rk::needs_long_row_bf16(cfg) ? rk::kLongRowBlockSize : rk::kBlockSize;
}

// ── the chunked split, for rows too long for one CTA to carry ───────────────
//
// One row per CTA leaves the machine idle when the batch is small and the rows
// long: B=6 at L=1M is 6 CTAs for the whole device.  The split ranks each row
// across `chunked_chunks(params)` CTAs and merges.  It wins by 11x at B=6/L=1M and
// loses above ~256 rows, which is the batch bound.  Upstream's own gate
// (`needs_chunked`: L >= 1M) is the same idea with a higher floor; 262144 is
// where the row count is still the problem here.  Measurements in
// `docs/C500-radix-perf-ledger.zh.md`.
constexpr uint32_t kChunkedMaxBatches = 64;
constexpr uint32_t kChunkedMinVocab = 262144;

// The split is SM-count-sensitive: a grid of `kBatch * chunks` CTAs leaves
// `ctas mod SM` SMs idle unless it is a whole number of waves, and the same 16
// is a different fraction of a wave on a 104-AP C500, a 28-SM C600 and a
// 32-SM C600U.  `params.sm_count` (the caller's, from the device) makes that
// decidable at launch time -- and it is the *device's* number rather than
// `ARCH_SM_COUNT` on purpose, because a caller that hands this entry a tensor
// on another part should get a grid sized for the part in front of it, not for
// the family the artifact was compiled for.
//
// The rule: **keep the measured count where it already fills at least half of
// its last wave, and otherwise round up to a whole number of waves.**  The
// half-wave floor is what keeps C500 exactly where it was measured.
// The fp32 split's work target, derived rather than tabulated.
//
// The split exists to fill a machine a short batch leaves empty, so its size
// is a *machine* property and the only question the sweep had to answer is
// what multiple of the SM count it is.  On C500 that fit is 256 (2.46 x 104
// APs), rounded to 5/2.  A per-family table would be three rows of which one
// is measured; this is the one measured fact, written once.
//
// **It reads the device's count, not `ARCH_SM_COUNT`, and that is deliberate.**
// The artifact is per-family now, so the macro *could* be used here, and it
// would be wrong to: this is the one rule whose whole purpose is to fill the
// machine in front of the call, and a 1600 image reached by a caller with a
// 1000 tensor would then size the grid for 32 APs on a 104-AP part.  The
// family constants answer questions about the *artifact*; this answers a
// question about the *run*.
inline uint32_t f32_chunk_work_target(uint32_t sm_count) {
    return sm_count * 5 / 2;
}

inline int wave_filled_chunks(int base_chunks, int sm_count) {
    constexpr int kBatch = 6;                      // the split's gate is small
    const int kSms = sm_count;
    const int ctas = kBatch * base_chunks;
    const int last_wave = ctas % kSms;
    if (last_wave == 0 || last_wave * 2 >= kSms) return base_chunks;
    return ((ctas + kSms - 1) / kSms) * kSms / kBatch;
}

// [MACA] SM-count-sensitive.  C500's 104 APs are what 16 was measured against
// (b6 x 16 = 96 CTAs = 92% of the last wave, and the chunk sweep shows 16-32
// flat there); C600 (28) and C600U (32) round it by `wave_filled_chunks`.
inline int chunked_chunks(const RowParams &params) {
    return wave_filled_chunks(16, (int)params.sm_count);
}

struct ChunkedWorkspace {
    int32_t *candidate_indices;
    maca_bfloat16 *candidate_values;
    int32_t *merged;
    int32_t *nan_flags;
};

// The split is 16-bit only, and only the two static-k arms of the merge are
// compiled, which is the whole gate: a 16-bit caller that satisfies it can
// always be answered by the split (the caller supplies the row table).
inline bool chunked_bf16_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0 || batches > kChunkedMaxBatches) return false;
    if (params.vocab_size < kChunkedMinVocab) return false;
    return params.topk == 512 || params.topk == 1024;
}

inline size_t chunked_workspace_bytes(uint32_t batches, uint32_t topk,
                                      uint32_t chunks) {
    const size_t candidates = (size_t)batches * chunks * topk;
    // The merge reads `candidate_values` with the vectorized row path, so the
    // two candidate arrays are 16-byte apart at the seam.
    const size_t indices_bytes = candidates * sizeof(int32_t);
    const size_t values_bytes = candidates * sizeof(maca_bfloat16);
    return indices_bytes + values_bytes + (size_t)batches * topk * sizeof(int32_t) +
           (size_t)batches * sizeof(int32_t);
}

inline ChunkedWorkspace chunked_workspace(void *base, uint32_t batches,
                                          uint32_t topk, uint32_t chunks) {
    const size_t candidates = (size_t)batches * chunks * topk;
    ChunkedWorkspace ws{};
    ws.candidate_indices = (int32_t *)base;
    ws.candidate_values = (maca_bfloat16 *)((char *)base + candidates * sizeof(int32_t));
    ws.merged = (int32_t *)((char *)ws.candidate_values + candidates * sizeof(maca_bfloat16));
    ws.nan_flags = (int32_t *)((char *)ws.merged + (size_t)batches * topk * sizeof(int32_t));
    return ws;
}

template <typename ValueT, typename OutIdxT>
void launch_typed_radix(const RowParams &params, uint32_t batches,
                        cudaStream_t stream, bool sorted_index,
                        bool sorted_value, bool return_value, int block,
                        bool preselected = false) {
    const size_t smem = radix_smem_bytes(
        params.topk, sorted_index || sorted_value,
        std::is_same<ValueT, maca_bfloat16>::value);
    auto run = [&](auto si, auto sv, auto rv) {
        constexpr bool SI = decltype(si)::value;
        constexpr bool SV = decltype(sv)::value;
        constexpr bool RV = decltype(rv)::value;
        // The fp32 row has a single width, so only the 16-bit row compiles the
        // wide-block arm (a runtime branch would instantiate it for fp32 too,
        // where it cannot be launched).
        if constexpr (std::is_same<ValueT, maca_bfloat16>::value) {
            if (block == rk::kLongRowBlockSize) {
                launch_radix<ValueT, OutIdxT, rk::kLongRowBlockSize, SI, RV, SV>(
                    params, batches, stream, smem, preselected);
            } else {
                launch_radix<ValueT, OutIdxT, rk::kBlockSize, SI, RV, SV>(
                    params, batches, stream, smem, preselected);
            }
        } else {
            launch_radix<ValueT, OutIdxT, rk::kBlockSize, SI, RV, SV>(
                params, batches, stream, smem, preselected);
        }
    };
    using T = std::true_type;
    using F = std::false_type;
    // sorted_value implies return_value (validated by the caller).
    if (sorted_value) {
        run(T{}, T{}, T{});
    } else if (sorted_index) {
        if (return_value) run(T{}, F{}, T{});
        else run(T{}, F{}, F{});
    } else {
        if (return_value) run(F{}, F{}, T{});
        else run(F{}, F{}, F{});
    }
}

// ── the fp32 split ──────────────────────────────────────────────────────────
//
// Same shape as the 16-bit split and the same gate, so the scratch and the
// `end_ptr` route are one code path.  The chunk count is a function of the
// batch because the two row regimes disagree: long rows prefer fewer chunks
// monotonically (hence the constant 2 above 64 rows), short rows want
// `largest power of two <= K / batches`, where K is the caller's
// `f32_chunk_work_target(params.sm_count)`.  Sweeps and caveats:
// `docs/C500-radix-perf-ledger.zh.md`.
//
// At an SM count other than the one measured, the large-batch arm's merge CTA
// per row becomes a much larger share of the waves.  The fix the sweep points
// at is dropping that CTA (`chunks + 1` -> `chunks`) -- a change to the merge
// launch, not a constant, so it is not made on arithmetic alone.
namespace {
// The work target is `f32_chunk_work_target(sm_count)` above -- `sm_count * 5
// / 2`, the 2.5 being the only fit ever measured (on C500).  This comment sent
// readers to `deep_select/_arch.py`'s `F32_CHUNK_WORK_TARGET` until
// 2026-09-16; no such symbol exists there or anywhere, and the module itself
// went on 2026-09-17 -- the SM count is read at the one place it is used.
//
// `DEEP_SELECT_F32_CHUNK32` raises the ceiling from 16 to 32 for the b6/V=1M
// cell the V-sweep flagged (parity plan §17.3).  A ceiling, not a count, so
// b32/b64 -- already at their optimum per the sweep -- are unaffected.
inline bool f32_chunk_ceiling_32() {
    static const bool on = [] {
        const char *v = std::getenv("DEEP_SELECT_F32_CHUNK32");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}
inline constexpr int kF32ChunkCeiling = 16;
inline int f32_chunks_for(uint32_t batches, int ceiling, uint32_t work_target) {
    const uint32_t b = batches == 0 ? 1u : batches;
    int c = 2;
    while (c < ceiling && (uint64_t)(c * 2) * b <= work_target) c *= 2;
    return c;
}
inline int f32_chunks_small_batch(uint32_t batches, int ceiling,
                                  uint32_t work_target) {
    return f32_chunks_for(batches, ceiling, work_target);
}

// **The chunk count's own rule cannot see `topk`, and above 512 it is the
// wrong axis.**  Stage 2's merge reads `num_chunks * topk` candidates per row
// -- `candidate_stride` in `rk::launch_topk_f32_chunks_stage2` -- and
// `f32_chunks_for` sizes itself from the batch alone.  The batch rule is
// already at the optimum for `topk <= 512`; above that the merge weights the
// count and a `topk`-aware cap is what §8.6 added.  Small-batch tier only --
// above `kF32ChunksFewBatches` the count is the large-batch rule and never
// reaches this.
//
// §8.6's version of that cap pinned the count at 8 and was measured over a
// sweep that is now superseded: it was wrong at both ends.  What this one is
// bounded by is the **chunk length**, and every length measured wants a count
// that is a multiple of two -- so the count is written as the product of two
// independent doubled quantities rather than as one number.
//
// `kF32ChunkSplitCeiling` (32) bounds the second one and is a *smem* bound, not
// a fit.  The arena is `kF32SmemInputSize` = 1757 slots and the 8-bit coarse
// level's threshold bin is not `L / 256` wide but about `L / 58` -- the suffix
// count reaches `topk` several bins below the mean, so it stops in the wider
// tail of the distribution (ratio flat at 57.9 / 56.8 / 58.2 for `L` in
// {87384, 65536, 131072}).  A bin that does not fit falls through to
// `radix_topk_row_f32_rescan`, which re-walks the whole window once per key
// byte -- five passes instead of two.  So `L / 58 <= 1757`, i.e. **L <= 101906**.
// The largest power of two under that is 65536.  That is the whole constant:
// two geometry facts, no measured value.
//
// **There used to be a `topk`-aware cap here and it was wrong in both
// directions.**  It read `c_default * topk * 10 <= vocab` and returned 8 when
// that failed.  Measured, interleaved repeats, `kk.bench`, contract checked,
// b1 and b6, `k = 2048`, medians of 2-3 rounds (ledger §10.9):
//
//     V        library ran    optimum   delta     -> V/c     optimum's chunk
//     98304    8              8            0%       12288
//     114688   8              4        -17.2%       28672
//     131072   8              4        -17.2%       32768
//     196608   8              6         -7.6%       32768
//     229376   8              4         -8.4%       57344
//     262144   8              6         -4.1%       43690
//     327680   16             6        -31.7%       54613
//     393216   16             6        -24.7%       65536
//     458752   16             6        -26.3%       76459
//     524288   16             8        -18.8%       65536
//
// The cap fired far too little (it exempted V >= 327680 and left the batch
// rule's 16, worth -18.8% .. -31.7%) and, where it did fire, pinned the count
// at 8 -- which is above the optimum at every V below 131072 on this ladder.
// The 8 was not a measurement of this tier's optimum; it was the merge weight
// `topk` puts on it, and that is a different axis.  **This is not an artefact
// of the fused NaN scan**: the `DS_FUSE_NAN_OFF` cross (`ab_fuse_chunks.py`,
// b1/b6 at V = 65536, 129280) puts both arms' optimum at the same count, so
// §8.6's sweep was already pointing low on its own.
//
// **What the optimum actually is: where stage 1 and stage 2 balance.**  They
// move in opposite directions in `c` -- stage 1 falls (`V / c` per CTA, more
// CTAs to hide the latency) and stage 2 rises (`num_chunks * topk` candidates
// read per row, and it is one CTA per row either way).  Setting the two
// derivatives against each other gives `c = theta * sqrt(V / topk)`, and the
// ladder above is that curve with `theta = 1/2`:
//
//     V        0.5*sqrt(V/2048)   nearest even   measured optimum
//     98304    3.46               4              8   (4 is 6.4% off)
//     114688   3.74               4              4
//     131072   4.00               4              4
//     196608   4.90               4              6   (1.4% off)
//     229376   5.29               6              4   (2.3% off)
//     262144   5.66               6              6
//     327680   6.32               6              6
//     393216   6.93               6              6
//     458752   7.48               8              6   (1.0% off)
//     524288   8.00               8              8
//
// Worst cell 6.4%, mean under 1.5%, and the two structural instincts are both
// in it: the count grows as `sqrt(V)` (more chunks only pay while stage 1's
// per-CTA latency is what is hiding) and falls as `1/sqrt(topk)` (a bigger
// merge is more expensive per candidate read).
//
// `theta = 1/2` is the one fitted number and it is fitted on `topk = 2048`
// only -- the `topk` axis of the curve is *predicted*, not measured, so the
// formula is applied only where `topk > 512` (the same gate §8.6 used) and the
// result is a cap: the caller takes `min` with the batch rule, so a cell the
// formula would raise past its measured count is unaffected.
//
// **`kF32ChunkCurveFloor` is a discontinuity in the data, not in the model.**
// The curve predicts 4 at both `V = 98304` and `V = 114688`, but the measured
// optimum steps 8 -> 4 across that interval (`0.0972` against `0.1034` at the
// low end; `0.1438` against `0.1550` at the high end, and the stage split flips
// sign between them -- s1's slope dominates at 98304, s2's at 114688). Above
// the floor the curve is used; below it the count is the measured 8, which is
// what both cells below want. Between those two V there is no measurement, so
// the boundary is placed on the measured one.
//
// `work_target` is the batch rule's own cap (`c * b <= 2.5 * sm_count`): the
// split exists to fill a machine a short batch leaves empty, so at a large
// batch it has nothing to buy and must not be forced up.
inline constexpr uint32_t kF32ChunkSplitCeiling = 32;
inline constexpr uint32_t kF32ChunkCurveFloor = 114688;
// The curve is fitted on `k = 2048` only, and it is applied on that axis only.
// For `512 < topk < 2048` §8.6's measurements still stand and still disagree
// with the curve: at `V = 262144, k = 1024` the batch rule's own 16 measures
// 99.5 us against 138.4 us for the curve's 8, so extrapolating down the `topk`
// axis would replace a measured optimum with a predicted one.
inline constexpr uint32_t kF32ChunkCurveTopK = 2048;

inline uint32_t f32_isqrt(uint64_t v) {
    uint32_t r = 0;
    for (uint32_t bit = 1u << 30; bit != 0; bit >>= 1) {
        if ((uint64_t)(r + bit) * (r + bit) <= v) r += bit;
    }
    return r;
}

// Round a half-integer up to the next even value: the ladder's optima are all
// even (they come in `c` and `c/2` pairs of a 2-way split), and an odd count
// makes `chunk_size = ceil(L/c)` uneven, which the chunk geometry's 8-element
// rounding then amplifies.
inline int f32_chunk_round_even(uint32_t c) {
    if (c < 2u) return 2;
    return (c & 1u) ? (int)(c + 1u) : (int)c;
}

inline int f32_topk_chunk_cap(uint32_t topk, uint32_t vocab, uint32_t c_default,
                              uint32_t work_target) {
    if (topk <= 512) return 0;   // 0 = no cap; keeps `DEEP_SELECT_F32_CHUNK32` in force
    if (vocab >= kF32ChunkCurveFloor && topk == kF32ChunkCurveTopK) {
        // `c = 0.5 * sqrt(V / topk)`, in integers: `c = isqrt(V / (4 * topk))`.
        // Nothing else: the merge-share guard below is *not* allowed to exempt
        // a cell here, and that is the point of splitting the two cases.
        int c = f32_chunk_round_even(f32_isqrt((uint64_t)vocab / (4ull * topk)));
        if ((uint64_t)c > work_target) c = f32_chunk_round_even(work_target);
        if (c > (int)kF32ChunkSplitCeiling) c = (int)kF32ChunkSplitCeiling;
        return c;
    }
    // §8.6's rule for everything the curve does not cover.  The guard is
    // `c_default * topk * 10 <= vocab` -- the merge reading under a tenth of the
    // row is already at the batch rule's own count, so capping it is a pure
    // loss.  **The curve above had to be split out of this because the guard
    // fires exactly at the cells the ladder says the curve should decide**: at
    // `k = 2048` the product is `8 * 2048 * 10 = 163840` (cap in force) or
    // `16 * 2048 * 10 = 327680` (exempt), so every `V >= 327680` took the
    // `return 0` arm and ran the batch rule's 16.  That is why `rule_ab.py`
    // read `V = 393216 / 524288` at `-0.2% / +0.2%` while the ladder's optimum
    // there is worth `-24.7% / -18.8%` (in force 0.2859 / 0.2959 against
    // 0.2152 / 0.2414 at `c = 6 / 8`).
    if ((uint64_t)c_default * topk * 10 <= vocab) return 0;
    return 8;   // §8.6's measured cap for `512 < topk < 2048`
}

// ── the chunk count above `kF32ChunksFewBatches` ────────────────────────────
//
// This used to be the constant 2.  What bounds it is the chunk *length*, and
// the bound is a shared-memory one.
//
// `radix_topk_row_f32` stages the members of the coarse threshold bin into an
// arena of `kF32SmemInputSize` slots.  A bin that does not fit falls through to
// `radix_topk_row_f32_rescan`, which re-walks the whole window once per key byte
// with no early exit -- five passes over the chunk instead of two.  Measured
// directly: on `b128 V=524288 k=2048`, disabling that branch (the `if (false
// && ...)` probe) takes `c = 1` from 4.18 ms to 0.95 ms, and `c = 4` from
// 3.80 ms to 0.96 ms, while `c = 6` is unchanged at ~1.05 ms -- i.e. the whole
// cliff is the rescan, and it is gone once a chunk clears the arena.
//
// So the question is only: for a row of `V` split `c` ways, does a chunk of
// `V / c` keep its threshold bin under the arena?  The 8-bit coarse level has
// 256 bins, and on a randn row the threshold bin is not `L / 256` wide but
// about `L / 58` -- the suffix count reaches `topk` several bins below the
// mean, so the bin it stops at is the wider tail of the distribution.  The
// ratio is flat at 57.9 / 56.8 / 58.2 for `L` in {87384, 65536, 131072}, which
// is exactly the range this rule decides in (measured with numpy over seeded
// randn draws, the same distribution `tests/lib.py` generates).
//
//     overflow when  L / 58 > kF32SmemInputSize  =>  L > kF32OverflowChunkLen
//
// Both factors are device geometry (`kF32SmemInputSize` from `kSMEM`, `kRadix`)
// except the 58, which is the measurement.  The count is then the smallest `c`
// that clears it:
//
//     V = 131072 -> 1.29 -> 2      (unchanged; the sweep is flat here)
//     V = 262144 -> 2.57 -> 3      (measured optimum 3)
//     V = 524288 -> 5.14 -> 6      (measured optimum 6)
//
// and both measured optima land where the rule says they do, with no fitted
// boundary: `c = 5` at `V = 524288` would put every chunk 3% over the arena
// (104858 against 101906) and the merge pays for it, 5.77 ms against 0.22 ms.
//
// Measured on C500, `k = 2048`, `kk.bench`, contract checked, 3-5 repeats,
// min-of-medians:
//
//     V=524288  b256 7.26 -> 2.05 ms   b4096 90.1 -> 29.6 ms   (c=2 -> 6)
//     V=262144  b256 3.81 -> 1.12 ms   b4096 46.5 -> 15.1 ms   (c=2 -> 3)
//
// `V < 262144` is untouched, which is where the sweep has the whole gate
// losing to the row path anyway (`b4096-v65536` is flat at 3.62 ms for every
// `c`); that is the batch bound at `topk_worth_splitting_f32`, not this.
//
// **Provenance, not a device test.**  Both inputs to the rule are device
// geometry -- the arena and the AP count the grid is sized against -- and the
// two fitted numbers below were read off one machine.  C600/C600U disagree on
// both (28/32 APs against 104, and a different smem budget), so they keep
// `c = 2`.  `__MACA_ARCH__` looks like the obvious discriminator and is the
// wrong one: it is defined only in the *device* pass, and this is a host
// function, so every architecture would take the `#else` branch.
//
// This is the same shape as the coarse12 gate's provenance half: a constant
// naming *where a number was measured*, not what the device is.  It reads
// `sm_count` because that is the only machine identity this file has -- unlike
// the coarse12 gate, there is no budget question here to answer instead, since
// the rule's own first factor already carries the arena.
//
// The first factor is not written down: it is `kF32SmemInputSize`, the arena
// the kernel actually has, and transcribing its value here is what let the
// small-batch branch above go without it for so long.  Deriving it means a
// `-DKSMEM_BYTES=` build moves this bound with the arena instead of leaving it
// behind -- the two numbers agreed at 1757 only because nothing had changed
// `kSMEM`.  The second factor, 58, is the one measured quantity here and is
// the only one that stays a literal.
inline constexpr uint32_t kF32OverflowChunkLen = rk::kF32SmemInputSize * 58u;
// The 58 was measured against the default arena, and `f32_chunks_large_batch`
// applies this bound to *every* `topk` -- so a build that moves `kSMEM` would
// carry the old ratio into a new arena without anything noticing.  The default
// build has nothing to check against (58 *is* the measurement); this fires only
// when `-DKSMEM_BYTES=` has actually moved the arena.
#ifdef KSMEM_BYTES
static_assert(rk::kF32SmemInputSize == 1757,
              "KSMEM_BYTES moved the arena, so kF32OverflowChunkLen's 58 is no "
              "longer the measured value: re-derive it against the new "
              "kF32SmemInputSize (the ladder is in the ledger, SS12.5) before "
              "shipping this build.");
#endif
// The AP count the coarse12 ladder was measured on, and the same kind of
// constant as the `sm_count` test just above -- a number naming a
// measurement's provenance, not a claim about what a device is.  There used to
// be a `kC500SmCount` here as well, for the coarse12 gate's device half; that
// half is now the budget comparison in `f32_coarse12_applies`, and the only
// question left that `sm_count` can answer is "was this the machine the
// numbers came from".  One constant, one job.
inline constexpr uint32_t kF32Coarse12MeasuredSmCount = 104;
inline int f32_chunks_large_batch(uint32_t vocab_size, uint32_t sm_count) {
    // Same shape as the coarse12 gate's provenance half, and the same reason:
    // the `58` below was read off this device.
    if (sm_count != kF32Coarse12MeasuredSmCount) return 2;
    // `c = 2` is the floor: `launch_topk_f32_chunks_stage1` rejects
    // `num_chunks <= 1`, so fewer than two is not a split at all.
    const int c = (int)((vocab_size + kF32OverflowChunkLen - 1) / kF32OverflowChunkLen);
    return c < 2 ? 2 : c;
}

// The batch above which the chunk count stops being a parallelism knob: 64 is
// the split's own small-batch ceiling, a property of the gate rather than a
// fitted value.  At 6 rows the machine is empty and chunks are the only CTAs
// there are; at 256 rows the grid already covers the machine twice over.
inline constexpr uint32_t kF32ChunksFewBatches = 64;

// Where the coarse12 route takes over.  The batch floor is not one number,
// because the measured crossing is not one number: it moves with the width.
// A row walk is only worth the machine when there are enough CTAs to fill it,
// and how many that is depends on how much work each CTA has -- which is `V`.
// The ladder below is `k = 2048`, paired medians, `B/A < 1` meaning the route
// won; the ledger's §12 has the full table.
//
//      V        b16     b24     b32     b48     b64     b96    b128
//      8192    0.821   0.801   1.015   0.763   0.724     -      -
//     32768    0.630     -       -       -       -       -      -
//     65536    0.897     -       -       -       -       -      -
//    131072    0.910   0.908   0.839     -       -       -      -
//    196608    1.029     -     0.925   0.891   0.796     -      -
//    262144    1.182   1.092   1.048   1.042   0.915   0.813  0.672
//    393216    1.407     -     1.255   1.184   1.031     -      -
//    524288    1.525   1.469   1.319   0.277   0.253   0.840    -
//
// Two regimes fall out of it, and one constant cannot hold both.  At or below
// `V = 131072` the route is already ahead at `b16` and never measured behind.
// `196608` is the first width where `b16` loses (1.029), so `131072` is the
// boundary: everything at or under it takes the 16 floor, everything above
// takes 64.  The two widths between them are unmeasured and are deliberately
// left on the wide side, which is the conservative direction.
//
// Above the boundary the crossing sits between `b48` and `b64` at `262144` and
// above `b64` at `393216`, so 64 is the one value that is a win at every wide
// cell measured except `393216` (1.031 -- the flat part of the crossing, +3%).
// `524288` wants a floor near 36, but that cell is the split's own cliff --
// `b32` is 1.319 and `b40` is 0.277 against a baseline whose stage 1 jumps
// 0.22 ms to 1.68 ms between them -- so it is not a crossing to site a
// constant on, and the 64 floor takes `b64` there at -74.7%.
// The wide-width floor is a product, not a count, because the crossing moves
// with `topk` as well as with the width.  Measured crossing batch, `V = 262144`,
// paired medians (`B/A < 1` = the route won):
//
//     topk    crossing    batches * topk
//     2048    ~56         114688
//     1024    ~112        114688
//     512     ~95         48640   (and ~190 at V = 524288 -> 97280)
//
// The first two agree to the digit; `topk = 512` crosses earlier than the
// product would put it, so this floor is **conservative there by design** --
// it declines `b128` at `V = 262144 k = 512` (-12.2%, a win left on the table)
// in exchange for never entering the two cells where the route loses badly at
// that `topk` (`b64 V = 524288` is +55.7%, `b128 V = 524288` is +4.1%,
// `b64 V = 262144` is +17.6% -- all three excluded).
inline constexpr uint64_t kF32Coarse12WorkTarget = 114688;
inline constexpr uint32_t kF32Coarse12MinBatchesNarrow = 16;
inline constexpr uint32_t kF32Coarse12NarrowVocab = 131072;

// The width floor is the one value in this gate that is not a crossing at all.
// The port wins at every width the split is legal at -- `V = 4096` is -29.1%,
// `8192` is -45.9%, `16384` is -53.4%, and the `(V, k)` sweeps above that are
// all wins -- so there is nothing to site it on.  What sets it instead is the
// original's own bound: `select_topk_policy` never routes to coarse12 below
// `n_cols = 2049`.  Below that the split is the only arm anyone has measured,
// so the gate matches the original rather than inventing a floor.  (2048, not
// 2049: the facade's own stride guard already rejects anything that is not a
// multiple of 256 floats, and 2048 is the largest legal width under 2049.)
inline constexpr uint32_t kF32Coarse12MinVocab = 2048;

// ── the deep_gemm coarse12 route ────────────────────────────────────────────
//
// `dg_coarse12.cuh` carries `topk_coarse12`, extracted whole from
// `mcDeepGEMM/csrc/kernels/fp32_topk.cu`.  It answers one row in one CTA and
// writes *positions in the row*, which is exactly what the contract half below
// consumes, so it replaces the whole split for the shapes it serves rather than
// sitting beside it.
//
// **Budget-gated, and for once the reason is not a measurement.**
// The kernel's 16 KB arena holds a 4096-bin coarse histogram first and the
// candidate array second, so it needs 16 KB of dynamic shared memory.  That is
// `kF32RowSmemBytes`-sized on C500 and is not on a device with a smaller
// budget -- and unlike every other C500-only rule in this file, a smaller
// budget does not make it slower, it makes it **wrong**.
//
// **Two questions, and this predicate used to answer both with one number.**
//
// *Does the route fit?*  A budget question, and it is answered against
// `ARCH_SMEM_PER_AP_BYTES` -- this artifact's own family's budget, a
// compile-time constant (`csrc/structs.h`).  The predicate used to read
// `sm_count != 104`, which is a fact about the C500 that has nothing to do
// with shared memory; the two agreed on every device anyone had looked at, and
// that agreement was the whole argument for the old form.  In between it read
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` at the entry point, which is the
// same number obtained one call later and from the wrong place: the artifact
// already knows which family it was built for, so asking the driver at launch
// time was a second source for a fact the build had.
//
// *Is the route faster here?*  A measurement, and this one is not adaptive --
// the crossing in the tables above is a C500 ladder, and applying it to a
// machine nobody measured would be a guess dressed as a rule.  So it is
// guarded by the device it was measured on, named for what it is.
//
// Splitting them is the point, and it still is under the macro: the budget
// half is now a *compile-time* property of the family, while the performance
// half stays a runtime property of the machine in front of the call.  A family
// 1600 image carries the 128 KiB budget without carrying the C500 ladder --
// the two halves move independently, which is the whole reason they are two
// halves.  An artifact whose `ARCH_SMEM_PER_AP_BYTES` is too small for the
// arena takes the `#if` and never routes at all, and that is a *build* fact
// now: it cannot be talked out of at runtime by a device that reports more.
//
// Why it is worth having at all: on the shapes deep_gemm routes here it is
// 2.0-2.3x faster than the row kernel (ledger §12).  The three measured
// differences are 12 coarse bits against 8, 640 threads against 512, and one
// 16 KB arena against 14 KB plus a ping-pong refine buffer.
// `DEEP_SELECT_F32_COARSE12`: `1` forces the route onto every shape that is
// otherwise *legal* for it, `0` denies it, unset takes the ladder.  In the same
// shape as `DEEP_SELECT_F32_CHUNKS` below, and for the same reason: the ladder
// is a set of measured crossings, and re-siting one means measuring the route on
// the shapes the current floor declines -- which is otherwise a source edit and a
// three-artifact rebuild per probe.  It replaces the floor and **nothing else**:
// the arena budget, the `topk` bound and the width bound are still enforced, so
// a forced route is a legal route.  It does not change any default, and it
// cannot reach a family the ladder was not measured on -- the `sm_count` test
// above it runs first.
inline int f32_coarse12_override() {
    static const int v = [] {
        const char *s = std::getenv("DEEP_SELECT_F32_COARSE12");
        if (s == nullptr || s[0] == '\0') return -1;
        return std::atoi(s) != 0 ? 1 : 0;
    }();
    return v;
}

inline bool f32_coarse12_applies(const RowParams &params, uint32_t batches) {
    // ── the budget half: this family's compile-time budget, and the request ──
    // `kSmemBytes` is the route's whole request (histogram over candidates);
    // `kF32RowSmemBytes` is what the *split* it replaces needs.  Both have to
    // fit, because which one runs is what this predicate is deciding.
    //
    // `if constexpr`, not a runtime test: on a family whose budget cannot hold
    // the arena the answer is false for every shape, and the build says so
    // rather than every call re-deriving it.  It also keeps the comparison
    // honest -- an unsigned constant against an unsigned constant is decided
    // by the compiler, so a family that fails this half is a compile-time
    // fact visible in the generated code rather than a branch.
    constexpr uint32_t kNeeded =
        rk::dg12::kSmemBytes > rk::kF32RowSmemBytes
            ? (uint32_t)rk::dg12::kSmemBytes
            : (uint32_t)rk::kF32RowSmemBytes;
    if constexpr (ARCH_SMEM_PER_AP_BYTES < kNeeded) {
        return false;
    } else {
        // ── the performance half: where the ladder below was measured ──
        if (params.sm_count != kF32Coarse12MeasuredSmCount) return false;
        // The arena's bound, not a fitted one: `topk_coarse12_row` serves at
        // most what the candidate half can hold, and there is no overflow path
        // in the *staged* refine that can make a larger answer correct.
        if (params.topk > (uint32_t)rk::dg12::kMaxTopK) return false;
        if (batches == 0) return false;
        // The original's own width bound (`n_cols >= 2049`).  It matters more
        // now than it did: the narrow arm below is `return batches >= 16` with
        // no other `V` test, so without this line a 1024-wide row would route,
        // which is below every width anyone has measured on either arm.
        if (params.vocab_size < kF32Coarse12MinVocab) return false;
        // The two arms below are the only part of this predicate that is a
        // *measured floor* rather than a bound -- everything above is a limit
        // the route would be wrong or impossible outside of.  So this is where
        // the A/B knob goes, and it is the whole of what it replaces.
        if (f32_coarse12_override() >= 0) return f32_coarse12_override() != 0;
        // Route on the shape deep_gemm itself routes on
        // (`select_topk_policy`), with the floor taken from the measured
        // crossing on this device.  The narrow-width arm is why there are two
        // arms and not one: at or below `V = 131072` the route is ahead at
        // `b16` and never measured behind, so the width decides which floor
        // applies, and only the wide one is a function of `topk`.  The tables
        // above have both.
        if (params.vocab_size <= kF32Coarse12NarrowVocab)
            return batches >= kF32Coarse12MinBatchesNarrow;
        return (uint64_t)batches * params.topk >= kF32Coarse12WorkTarget;
    }
}

// ── the deep_gemm chunks route: the `b <= 2` band ───────────────────────────
//
// `dg_chunks.cuh` carries `topk_chunks`, deep_gemm's third kernel family, in the
// same shape as `dg_coarse12.cuh` carries the second.  It is here because the
// two arms this file already has are both the wrong shape at one or two rows,
// and that is measured rather than argued (ledger §12.16.6, interleaved five
// rounds on device 2, `k = 2048`):
//
//     b     V        split    deep_gemm    dg/split
//     1     66551     90.8      53.0        0.584x
//     2     66551     93.4      54.1        0.580x
//     4     66551     93.8     360.8        3.844x
//     1    107520    174.8      60.6        0.347x
//     1    131072    135.0      62.7        0.464x
//
// **The cliff is deep_gemm's, at `b = 4`.**  Its own `topk_chunks_compact_refine`
// goes from 24.4 us at one row to 310.7 us at sixteen -- 12.7x per row -- while
// the CTA count only goes from 6 to 96 on a 104-AP part.  So this is not a
// fallback to reach for whenever we are behind: it is a `b <= 2` special case,
// and the batch bound below is sited on the measured step, not on a guess.
//
// Why it wins there: deep_gemm walks the row a second time and writes every
// element **above** the threshold bin straight to the output in that same walk
// (its `guaranteed` half), staging only the threshold bin's members.  The split
// instead materializes `num_chunks * topk` candidates and merges them, and at
// `k = 2048, V = 66551` the elements above the threshold bin are already almost
// the whole answer -- which is why the sweep over `DEEP_SELECT_F32_CHUNKS`
// (2..32, ledger §12.16.6) moves the split by 46% and never reaches this.
//
// **Not `constexpr`-gated, and it does not need to be**: the arena is 32 KB and
// every family this repository builds has at least 64 KB per AP
// (`csrc/structs.h`), so the budget half of `f32_coarse12_applies` would be
// satisfied by all three.  The one question is whether the machine in front of
// the call is one this was measured on, which is the same `sm_count` test the
// coarse12 tuning half uses and for the same reason.
inline constexpr uint32_t kF32ChunksMaxBatches = 2;
inline constexpr uint32_t kF32ChunksMinVocab = 2048;

// `DEEP_SELECT_F32_CHUNKS_ROUTE`: `1` forces this arm on, `0` denies it, unset
// takes the band.  Same shape and same reason as `DEEP_SELECT_F32_COARSE12` --
// the band is a measured step and re-siting it means measuring the arm on the
// batches it declines.  It replaces the band and nothing else.
inline int f32_chunks_route_override() {
    static const int v = [] {
        const char *s = std::getenv("DEEP_SELECT_F32_CHUNKS_ROUTE");
        if (s == nullptr || s[0] == '\0') return -1;
        return std::atoi(s) != 0 ? 1 : 0;
    }();
    return v;
}

inline bool f32_chunks_applies(const RowParams &params, uint32_t batches) {
    // ── the provenance half: where the b1/b2 ladder was measured ──
    if (params.sm_count != kF32Coarse12MeasuredSmCount) return false;
    if (params.topk > (uint32_t)rk::dgchunks::kMaxTopK) return false;
    // The original's own width bound, as on the coarse12 arm: `2048`, not
    // `2049`, because the facade already rejects a stride that is not a
    // multiple of 256 floats.
    if (params.vocab_size < kF32ChunksMinVocab) return false;
    // ── the measured band ──
    if (f32_chunks_route_override() >= 0) return f32_chunks_route_override() != 0;
    if (batches == 0 || batches > kF32ChunksMaxBatches) return false;
    return true;
}
}  // namespace

// `topk` and `vocab_size` are parameters here, not read off `params`, because
// every caller that sizes the workspace must get the same answer as the one
// that launches: a workspace sized for one chunk count and a grid launched with
// another is a write past the arena, and the gate at `topk_launch` compares the
// two.
int f32_chunked_chunks(uint32_t batches, uint32_t topk, uint32_t vocab_size,
                       uint32_t sm_count) {
    const uint32_t work_target = f32_chunk_work_target(sm_count);
    static const int override_n = [] {
        const char *v = std::getenv("DEEP_SELECT_F32_CHUNKS");
        if (v != nullptr && v[0] != '\0') {
            const int d = std::atoi(v);
            return (d >= 2 && d <= 256) ? d : 0;
        }
        return 0;
    }();
    if (override_n != 0) return override_n;   // A/B knob; does not change default
    if (batches > kF32ChunksFewBatches)
        return f32_chunks_large_batch(vocab_size, sm_count);
    const int ceiling = f32_chunk_ceiling_32() ? 32 : kF32ChunkCeiling;
    const int c = f32_chunks_small_batch(batches, ceiling, work_target);
    const int cap = f32_topk_chunk_cap(topk, vocab_size, (uint32_t)c, work_target);
    int n = (cap != 0 && c > cap) ? cap : c;
    // **The arena floor, which this branch never had.**  `f32_chunks_large_batch`
    // above is derived from exactly one thing -- the largest chunk whose
    // threshold bin still fits `kF32SmemInputSize` -- but the small-batch
    // branch below `kF32ChunksFewBatches` was sized from the batch alone and
    // never learned it.  The omission is not a slope, it is a cliff, and it
    // lands precisely on the branch boundary: at `V = 524288` the batch rule
    // gives `c = 4` for `33 <= batches <= 64`, which is a 131072-element chunk
    // against the `kF32OverflowChunkLen` bound, so `radix_topk_row_f32_rescan`
    // runs five passes where it should run two.  Measured, split alone,
    // `k = 2048`: `b32` is 0.3732 ms (c = 8, 65536/chunk, no rescan) and `b33`
    // is 1.77 ms (c = 4); `b65` takes the branch above, gets 6, and is fine
    // again.  So the two branches disagreed about the same row at their own
    // seam.  With the floor, split alone: `b40` **-77.6%**, `b64` **-72.5%**,
    // and the two controls that have no overflow (`b32`, `b40 V = 262144`) at
    // -0.1% / -0.3%.
    //
    // Applied *after* the cap and as a floor, not a replacement: the batch rule
    // and the curve still decide the count wherever they are already above the
    // bound.
    //
    // **The `topk` guard is the measurement; the branch above is geometry.**
    // `f32_chunks_large_batch` applies the same arena bound to *every* `topk`,
    // because that is the only way it can be a derivation rather than a table.
    // Here the bound has to be checked against a measurement instead: the 58 is
    // a `k = 2048` value, and at `k = 512` the threshold bin is narrower and
    // `c = 4` never overflowed -- so raising it there is free in the arena and
    // the merge pays for it, measured `b64 V = 524288 k = 512` at **+14.3%**
    // against -0.1% / -0.3% on the controls.  So this arm fires where its
    // constant was derived, and `k = 1024` keeps the `k = 2048` verdict for
    // want of a measurement rather than for want of a reason.
    if (topk == kF32ChunkCurveTopK) {
        const uint32_t arena =
            (vocab_size + kF32OverflowChunkLen - 1) / kF32OverflowChunkLen;
        if ((uint32_t)n < arena) n = (int)arena;
    }
    return n;
}

// The batch bound has no single value: the split costs a merge over `batches`
// CTAs on an otherwise idle machine and saves the row kernel's per-CTA time, so
// which way it goes is measured per shape.  The floor was 262144 and the chunk
// count above is why it could drop to the small-batch tier's value -- with the
// count now a constant 2 above 64 rows the merge is 3 CTAs per row at every
// shape, so the cost stops scaling with the gate.  Measured over the eight
// cells the change reaches: -12.7% total, with the two neutral cells being the
// contract (b6 was already open, b4096-v262144 already served).
//
// **The band `32768 <= V < 65536` is what that unification newly serves**, and
// it was measured before being opened: six controls at `V >= 65536`, which the
// change cannot reach, hold to |delta| <= 0.3%.  Every cell in the band wins,
// -5.6% to -27.7%, the magnitude falling as the batch rises -- the same curve
// every other result here shows.  Full tables in
// `docs/C500-radix-perf-ledger.zh.md`.
//
// There used to be four constants here, a small-batch tier (`V >= 65536`,
// `batches <= 64`) and a large-batch one (`V >= 262144`, later 65536,
// `batches <= 4096`).  Lowering the large tier's floor to 32768 makes the small
// branch unreachable (it would need `V < 32768` and `V >= 65536` at once), so
// the predicate is one condition and the two constants below are what is left.
// A constant that cannot select anything is a knob a reader will try to turn.
constexpr uint32_t kF32ChunkedMinVocab = 32768;
constexpr uint32_t kF32ChunkedMaxBatches = 4096;

// **When is a split worth its own two extra stages?**  The row kernel already
// runs one CTA per row, so a split adds a scan and a merge over
// `batches * (chunks + 1)` CTAs and buys exactly one thing: parallelism the row
// kernel did not have.  That is a function of the row length against the
// machine, not of `topk` -- the gain rises with `vocab` and falls with `batches`,
// and large-batch short rows are the corner where the split has nothing to buy.
//
// The model: the split's own stages cost `B * (chunks + 1) * fixed` for a fixed
// per-CTA cost, and a row costs `V * c` at a per-element rate `c`; with
// chunks = 2 that makes rows win below `V < B * 6.0e6`.  A two-term model is
// not trustworthy near the corner it was fitted at, so it is **coupled to the
// measured region**: inside the sweep the sweep decides, and only outside it
// does the model speak.  The sweep's verdicts, and the one cell it leaves
// ambiguous (`b768-v65536`, which the model sends to the split and the gate
// follows), are in `docs/C500-radix-perf-ledger.zh.md` -- recorded there rather
// than smoothed over.

inline bool topk_worth_splitting_f32(const RowParams &params,
                                     uint32_t batches) {
    if (params.topk <= 0 || params.topk > (int)rk::kF32MaxTopK) return false;
    if (batches == 0) return false;
    // V=32768 rows, and every row of V<=65536 at a batch past the 768 tier.
    if (params.vocab_size <= kF32ChunkedMinVocab) return false;
    if (params.vocab_size <= 65536 && batches > 256) return false;
    // So does b4096 at V=65536.  (`batches == 4096` is the gate's own ceiling.)
    if (params.vocab_size <= 65536 && batches == kF32ChunkedMaxBatches) return false;
    return true;
}

// The split is fp32-only (the caller's `value_dtype == 0`) and its merge only
// compiles the k=512/1024 arms, which is the whole gate.
inline bool chunked_f32_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0) return false;
    // `end_ptr` absent means the row really is `vocab_size` long, so the cap is
    // the vocab either way -- and it is the only length the host knows without
    // a device read.
    if (params.vocab_size < kF32ChunkedMinVocab) return false;
    if (batches > kF32ChunkedMaxBatches) return false;
    // The topk clause that used to sit here (`== 512 || == 1024`) was a *policy*
    // bound, not an instantiation one -- the fp32 chunk engine is dynamic-k
    // (`rk::f32_chunk_engine_supports`) -- so it is replaced by a measured one
    // rather than by a wider constant.  Plan §21.3 opened it to
    // `topk <= kF32MaxTopK` and got -71..-82% on the long rows but up to +26.5%
    // back on short ones at a large batch; the cause is the shape of what a
    // split buys.
    if (params.topk == 512 || params.topk == 1024) return true;   // measured
    return topk_worth_splitting_f32(params, batches);
}

inline size_t chunked_f32_workspace_bytes(uint32_t batches, uint32_t topk,
                                          int chunks) {
    return rk::chunked_f32_workspace_bytes(batches, topk, (uint32_t)chunks) +
           (size_t)batches * sizeof(int32_t);
}

struct ChunkedF32Workspace {
    int32_t *lengths;
    int32_t *cols;
    float *vals;
    int32_t *merged;
};

inline ChunkedF32Workspace chunked_f32_workspace(void *base, uint32_t batches,
                                                 uint32_t topk, int chunks) {
    ChunkedF32Workspace ws{};
    const size_t candidates = (size_t)batches * chunks * topk;
    ws.lengths = (int32_t *)base;
    ws.cols = ws.lengths + batches;
    ws.vals = (float *)(ws.cols + candidates);
    ws.merged = (int32_t *)(ws.vals + candidates);
    return ws;
}

// Split, merge, then run the row kernel's contract half over the merged
// answer.  `params` must satisfy `chunked_f32_applies`, workspace included.
//
// The row table is passed through as `end_ptr` rather than being consumed here
// -- the split's own kernels take it as `lengths` -- which is what keeps the
// dispatch below (`params.end_ptr != nullptr`) meaningful.
template <typename OutIdxT>
void launch_typed_f32_chunked(const RowParams &params, uint32_t batches,
                              cudaStream_t stream, bool sorted_index,
                              bool sorted_value, bool return_value, int block,
                              void *workspace) {
    const int chunks = f32_chunked_chunks(batches, params.topk,
                                          params.vocab_size, params.sm_count);
    const ChunkedF32Workspace ws = chunked_f32_workspace(
        workspace, batches, params.topk, chunks);
    // `ws.lengths` is the NaN-flag table here, not the split's row table --
    // that one is `params.end_ptr`, passed through below.  It is memset and
    // then OR-ed per row, so the length is whichever ran.  `end_ptr` being
    // non-null is the dispatch's precondition and the dispatcher only reaches
    // here with the scratch's own all-`vocab_size` table, but clearing first
    // makes the scan independent of that rather than merely consistent with
    // it.  Both the clearing and the scan go when `check_nan` is off: the row
    // kernel reads this table only under that same flag.
    //
    // With the scan fused, this table is still built the same way -- zeroed
    // here, OR-ed per row -- and only stage 1's own pass 1 walks the row.  One
    // row walk instead of two; the contract the flag answers is unchanged.
    const size_t table_bytes = (size_t)batches * sizeof(int32_t);
    const bool fuse_scan = params.check_nan && rk::f32_stage1_fuses_scan();
    if (params.check_nan) {
        cudaMemsetAsync(ws.lengths, 0, table_bytes, stream);
        if (!fuse_scan) {
            nan_scan_kernel<float><<<batches * chunks, kScanBlock, 0, stream>>>(
                params.input, params.end_ptr,
                (int64_t)(params.stride_input_batch / sizeof(float)),
                params.vocab_size, (uint32_t)chunks, ws.lengths);
        }
    }
    const cudaError_t rc = rk::launch_topk_f32_chunked(
        (const float *)params.input, params.end_ptr, ws.cols, ws.merged,
        (int)batches, (int)params.vocab_size, (int)params.topk, chunks, stream,
        // The row stride is in bytes at this layer and in elements there.
        (int64_t)(params.stride_input_batch / sizeof(float)),
        fuse_scan, ws.lengths);
    RowParams merged = params;
    const bool preselected = rc == cudaSuccess;
    merged.preselected = preselected ? ws.merged : nullptr;
    merged.nan_flags = preselected ? ws.lengths : nullptr;
    launch_typed_radix<float, OutIdxT>(merged, batches, stream, sorted_index,
                                       sorted_value, return_value, block,
                                       preselected);
}

// Split, merge, then run the row kernel's contract half over the merged
// answer.  `params` must satisfy `chunked_bf16_applies`, workspace included.
template <typename OutIdxT>
void launch_typed_chunked(const RowParams &params, uint32_t batches,
                          cudaStream_t stream, bool sorted_index,
                          bool sorted_value, bool return_value, int block,
                          void *workspace) {
    const int chunks = chunked_chunks(params);
    const ChunkedWorkspace ws =
        chunked_workspace(workspace, batches, params.topk, (uint32_t)chunks);
    if (params.check_nan) {
        cudaMemsetAsync(ws.nan_flags, 0, (size_t)batches * sizeof(int32_t), stream);
        nan_scan_kernel<maca_bfloat16><<<batches * chunks, kScanBlock, 0, stream>>>(
            params.input, params.end_ptr,
            (int64_t)(params.stride_input_batch / sizeof(maca_bfloat16)),
            params.vocab_size, (uint32_t)chunks, ws.nan_flags);
    }
    const cudaError_t rc = rk::launch_topk_bf16_chunked(
        (const maca_bfloat16 *)params.input, params.end_ptr, ws.merged,
        ws.candidate_indices, ws.candidate_values, (int)batches,
        (int)params.vocab_size, (int)params.topk, chunks, stream,
        // The row stride is in bytes at this layer and in elements there.
        (int64_t)(params.stride_input_batch / sizeof(maca_bfloat16)));
    RowParams merged = params;
    // A split that could not be launched leaves the workspace unfilled; the row
    // dataflow answers instead of the contract reading an empty slot.  The gate
    // keeps this unreachable (the split rejects only what it was gated on).
    const bool preselected = rc == cudaSuccess;
    merged.preselected = preselected ? ws.merged : nullptr;
    merged.nan_flags = preselected ? ws.nan_flags : nullptr;
    launch_typed_radix<maca_bfloat16, OutIdxT>(merged, batches, stream,
                                               sorted_index, sorted_value,
                                               return_value, block, preselected);
}

}  // namespace detail

void topk_launch(const RowParams &params, int64_t batches, void *stream,
                 int value_dtype, int index_dtype, bool sorted_index,
                 bool sorted_value, bool return_value,
                 void *chunked_workspace, size_t chunked_bytes) {
    auto *cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const uint32_t n = (uint32_t)batches;
    const bool rv = return_value || sorted_value;
    // Both dtypes run the ported radix dataflow; only the block width differs,
    // and the fp32 row has a single width.
    const int block = (value_dtype == 0)
                          ? (int)rk::kBlockSize
                          : detail::radix_block_for(n, params.vocab_size,
                                                    params.topk);
    // Long rows at a small batch go through the split; the caller sized the
    // workspace for exactly this gate, and the split reads 16-bit input.
    if (value_dtype == 1 && chunked_workspace != nullptr &&
        params.end_ptr != nullptr &&
        chunked_bytes >= detail::chunked_workspace_bytes(
                             n, params.topk, (uint32_t)detail::chunked_chunks(params)) &&
        detail::chunked_bf16_applies(params, n)) {
        if (index_dtype == 0) {
            detail::launch_typed_chunked<int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                chunked_workspace);
        } else {
            detail::launch_typed_chunked<int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                chunked_workspace);
        }
        return;
    }
    // **Both index widths, and that is the point.**  The gate here used to be
    // `index_dtype == 0`, which made the whole route unreachable through the
    // public facade: `deep_select.topk`'s default `indices_type` is
    // `torch.int64`, so every default call took the split and this kernel was
    // never launched.  The first A/B of this route measured 0.0% for exactly
    // that reason.
    //
    // The kernel itself is width-agnostic -- it answers in row *columns*, which
    // are int32 by nature -- so widening means staging those columns through
    // the dispatcher's scratch and letting the contract half do the widening it
    // already does for the split.  Writing them straight into `output_index`
    // would be wrong for int64 (`int32_t*` arithmetic over an int64 buffer), so
    // the scratch is not an optimization here, it is what makes the arm legal.
    // ── the `b <= 2` arm: deep_gemm's chunks kernel ────────────────────────
    // Same handoff as the coarse12 arm below it -- answer in row columns, staged
    // through the scratch, contract half reads the per-row `-1` -- because it is
    // the same class of kernel: one CTA per row, walking the row itself.  Tested
    // *before* coarse12 so the two cannot both claim a cell; the batch bands are
    // disjoint by construction (`<= 2` here, `>= 16` or a 114688 product there),
    // but the order makes that a fact about the code rather than about two
    // constants that have to keep agreeing.
    if (value_dtype == 0 && detail::f32_chunks_applies(params, n)) {
        int32_t *const staging = params.coarse12_cols;
        if (staging != nullptr) {
            const cudaError_t rc = rk::dgchunks::launch_topk_chunks(
                (const float *)params.input, params.end_ptr, staging,
                params.scan_flags, params.chunks_workspace, (int)n,
                (int)params.topk,
                (int64_t)(params.stride_input_batch / sizeof(float)),
                (int)params.vocab_size, (int)params.sm_count, cuda_stream);
            if (rc == cudaSuccess) {
                RowParams merged = params;
                merged.preselected = staging;
                merged.nan_flags = params.scan_flags;
                if (index_dtype == 0) {
                    detail::launch_typed_radix<float, int32_t>(
                        merged, n, cuda_stream, sorted_index, sorted_value, rv,
                        block, /*preselected=*/true);
                } else {
                    detail::launch_typed_radix<float, int64_t>(
                        merged, n, cuda_stream, sorted_index, sorted_value, rv,
                        block, /*preselected=*/true);
                }
                return;
            }
        }
        if (index_dtype == 0) {
            detail::launch_typed_radix<float, int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                /*preselected=*/false);
        } else {
            detail::launch_typed_radix<float, int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                /*preselected=*/false);
        }
        return;
    }
    if (value_dtype == 0 && detail::f32_coarse12_applies(params, n)) {
        int32_t *const staging = params.coarse12_cols;
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
            const cudaError_t rc = rk::dg12::launch_topk_coarse12(
                (const float *)params.input, params.end_ptr, staging,
                params.scan_flags, (int)n, (int)params.topk,
                (int64_t)(params.stride_input_batch / sizeof(float)),
                (int)params.vocab_size, cuda_stream);
            if (rc == cudaSuccess) {
                // Hand the answer to the contract half the way the split does,
                // so its per-row `-1` check decides row by row: a row this
                // kernel could not serve, or one whose window is no longer
                // than `topk`, comes back with an empty slot and is re-ranked
                // by the row path.
                //
                // The scan table goes with it, and that is not optional: this
                // kernel answers a row *without* ranking it, so under
                // `check_nan` the contract half must read a flag rather than
                // scan -- and the flag is the one that kernel raised while
                // reading the row anyway.
                RowParams merged = params;
                merged.preselected = staging;
                merged.nan_flags = params.scan_flags;
                if (index_dtype == 0) {
                    detail::launch_typed_radix<float, int32_t>(
                        merged, n, cuda_stream, sorted_index, sorted_value, rv,
                        block, /*preselected=*/true);
                } else {
                    detail::launch_typed_radix<float, int64_t>(
                        merged, n, cuda_stream, sorted_index, sorted_value, rv,
                        block, /*preselected=*/true);
                }
                return;
            }
        }
        // A launch that did not happen leaves the buffer untouched, so the row
        // path answers instead -- the same fallback the split takes.  So does
        // an allocation that could not be made.
        if (index_dtype == 0) {
            detail::launch_typed_radix<float, int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                /*preselected=*/false);
        } else {
            detail::launch_typed_radix<float, int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                /*preselected=*/false);
        }
        return;
    }
    if (value_dtype == 0 && chunked_workspace != nullptr &&
        params.end_ptr != nullptr &&
        chunked_bytes >= detail::chunked_f32_workspace_bytes(
                             n, params.topk,
                             detail::f32_chunked_chunks(n, params.topk,
                                                        params.vocab_size,
                                                        params.sm_count)) &&
        detail::chunked_f32_applies(params, n) &&
        // The engine's own bound, kept separate from the policy predicate so
        // the two cannot be confused (they were one expression until §21.2).
        // The row path serves anything the engine cannot.
        rk::f32_chunk_engine_supports((int)params.topk)) {
        if (index_dtype == 0) {
            detail::launch_typed_f32_chunked<int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                chunked_workspace);
        } else {
            detail::launch_typed_f32_chunked<int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block,
                chunked_workspace);
        }
        return;
    }
    if (value_dtype == 0) {  // float32
        if (index_dtype == 0) {
            detail::launch_typed_radix<float, int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block);
        } else {
            detail::launch_typed_radix<float, int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block);
        }
    } else {  // bfloat16
        if (index_dtype == 0) {
            detail::launch_typed_radix<maca_bfloat16, int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block);
        } else {
            detail::launch_typed_radix<maca_bfloat16, int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv, block);
        }
    }
}

}  // namespace deep_select_maca

// ── the tvm-ffi entry points ────────────────────────────────────────────────
//
// Torch-free: this TU no longer includes <torch/extension.h> or
// <ATen/cuda/CUDAContext.h>, so the extension carries no torch DT_NEEDED
// entry and its behavior is not tied to the host's torch build.  The contract
// checks that used to be `TORCH_CHECK` are `DS_HOST_CHECK`, the tensors are
// `tvm::ffi::TensorView`, and the stream comes from the FFI environment.
#include "../ffi/ffi_error.h"
#include "../ffi/ffi_tensor.h"

#include <tvm/ffi/extra/c_env_api.h>

#include <utility>
#include <vector>

namespace deep_select {

using namespace deep_select_maca;
namespace dsf = deep_select::ffi;

// ── the chunked path's scratch, held across calls ───────────────────────────
//
// `cudaMalloc`/`cudaFree` on this runtime cost ~70-100 us per *pair* at this
// size class, and the cost is non-monotonic in size (a 12 MB allocation is
// nearly free; 300 KB is the worst case).  The split needs this workspace plus
// a row-length table on every call, so a fresh pair each time put a ~265 us
// host floor in front of a ~96 us kernel -- `e2e = max(host, device)`.
//
// So they are cached process-wide and **grown only**.  Growing rather than
// keying by shape is the design: a caller alternating between two batches would
// realloc on every switch under a shape-keyed cache, which is the cost this
// exists to remove.
//
// Three properties make reuse safe rather than merely fast:
//
//   * `chunked_workspace(base, batches, topk)` walks forward from `base` by the
//     *geometry of this call*, not by the capacity, so a buffer larger than this
//     call needs is correct by construction.
//   * the split's kernels write every slot they read back -- stage 1 fills each
//     candidate slot it is asked for, and `nan_flags` is memset per call --
//     so nothing is inherited from the previous call.
//   * the lengths table is skipped only when `(batches, vocab_size)` -- both
//     arguments of this call -- already match the pair it was last filled for.
//     Its content is `[vocab_size] * batches`, a pure function of that pair, so
//     the pair *is* the content; see `lengths_epoch` at the fill.
//
// The high-water mark is bounded by the gate that admits the split
// (`chunked_bf16_applies`: batches <= 64, topk <= 1024) -- ~6.3 MB of workspace
// and 256 B of table, held for the life of the process.  A bounded, one-time
// footprint in exchange for removing a per-call host cost larger than the
// kernel it fronts.  Freeing instead of holding gives most of the win back:
// free-then-malloc *should* be cheap, and measured on this runtime it is not.

struct ChunkedScratch {
    std::mutex mu;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    int32_t *lengths = nullptr;
    size_t lengths_count = 0;
    // The `(batches, vocab_size)` the device table was last **filled** for.
    // `kNoEpoch` means "unknown, must refill"; it is not a fabricated `(0, 0)`,
    // which a call with `batches == 0` would match and then skip the fill on.
    static constexpr int64_t kNoEpoch = -1;
    int64_t lengths_epoch_batches = kNoEpoch;
    int64_t lengths_epoch_vocab = kNoEpoch;
    // One `int32_t` per row, zeroed once per call that scans.  This is the
    // table `topk_kernel_radix`'s preselected arm reads to decide whether the
    // row is poisoned -- and it is *read on every preselected path*, while
    // only the two splits used to own one (each embedded in its own workspace
    // layout).  The coarse12 route answers a row without ranking it, so it can
    // hand back a row a scan would have rejected, and it had no table to read:
    // `nan_flags` stayed null and the first `check_nan` read faulted.  This
    // buffer is that table for every route that does not bring its own.
    int32_t *scan_flags = nullptr;
    size_t scan_flags_count = 0;
    // The coarse12 route's answer buffer: `n_rows * topk` int32 columns, which
    // are the row *positions* the kernel ranks.  It is separate from
    // `output_index` because the public entry's default `indices_type` is
    // int64, and a kernel writing int32 through the caller's pointer would
    // interleave.  Grown like the rest and never shrunk.
    int32_t *coarse12_cols = nullptr;
    size_t coarse12_cols_count = 0;
    // The chunks route's per-row workspaces.  Separate from `coarse12_cols`
    // because the two have different shapes and different lifetimes: this one
    // is `n_rows * sizeof(TopKChunksWorkspace<NChunks>)` of opaque bytes, that
    // one is `n_rows * topk` int32 columns of answer.  One byte buffer serves
    // every `NChunks` instantiation -- the launch knows which one it is, and
    // the layout is identical for all of them.
    void *chunks_workspace = nullptr;
    size_t chunks_workspace_bytes = 0;
};

ChunkedScratch &chunked_scratch() {
    static ChunkedScratch s;
    return s;
}

tvm::ffi::Array<int64_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
            OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

void topk(const tvm::ffi::TensorView &input, int64_t topk,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &end,
          bool sorted_value, bool sorted_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_value,
          const tvm::ffi::TensorView &output_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_idx_offset,
          int64_t idx_oob_fill_value, double value_oob_fill_value,
          bool return_value, bool abort_when_nan_found, bool check_nan,
          // AP count of the device this call runs on, from the caller's
          // `get_device_properties`.  The last argument rather than one near
          // `topk` because it is the only one the caller derives from the
          // *device* rather than from the problem -- and the only device fact
          // that stays an argument now that the family constants are
          // compile-time (`csrc/structs.h`).  The shared-memory budget is
          // `ARCH_SMEM_PER_AP_BYTES`; the AP count cannot be, because the grid
          // has to be sized for the part in front of the call and not for the
          // one the image was built for.
          int64_t sm_count) {
    DS_HOST_CHECK(input.ndim() == 2, "input must be 2-D, got ", input.ndim());
    const int64_t batches = dsf::size(input, 0);
    const int64_t vocab_size = dsf::size(input, 1);
    const bool float32 = dsf::is_float32(input);
    const bool bfloat16 = dsf::is_bfloat16(input);

    DS_HOST_CHECK(topk > 0, "topk must > 0");
    DS_HOST_CHECK(topk <= (int64_t)kMaxTopK, "topk must be <= ", kMaxTopK);
    DS_HOST_CHECK(!(sorted_value && !return_value),
                  "`return_value` must be enabled when `sorted_value` is True");
    DS_HOST_CHECK(!(sorted_value && sorted_index),
                  "`sorted_value` and `sorted_index` cannot be used at the same time");
    DS_HOST_CHECK(float32 || bfloat16,
                  "input dtype must be float32 or bfloat16");
    DS_HOST_CHECK(dsf::is_index_type(output_index),
                  "output_index dtype must be int32 or int64");
    DS_HOST_CHECK(dsf::stride(input, 1) == 1, "input.stride(1) must be 1");
    DS_HOST_CHECK(dsf::stride(input, 0) * (int64_t)dsf::element_size(input) %
                          (int64_t)INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
                  "input.stride(0) must be a multiple of ",
                  INPUT_STRIDE_ALIGNMENT_REQUIREMENT, " bytes");

    // Every output row is addressed as `row * stride(0) + column`, so a
    // last-dimension stride other than 1 (or a row that is too short) writes
    // outside the columns the caller owns.  Upstream rejects both
    // (api.cu KU_CHECK_LAST_DIM_CONTIGUOUS / KU_CHECK_SHAPE) -- without the
    // check the result is silently scrambled, so refuse instead.
    auto check_out_tensor = [&](const char what[],
                                const tvm::ffi::TensorView &t) {
        DS_HOST_CHECK(t.device().device_id == input.device().device_id,
                      what, " must be on the same device as `input`");
        DS_HOST_CHECK(dsf::stride(t, 1) == 1, what, ".stride(1) must be 1");
        DS_HOST_CHECK(dsf::size(t, 0) == batches && dsf::size(t, 1) >= topk,
                      what, " must be at least (batch_size, topk) = (",
                      batches, ", ", topk, ")");
    };
    check_out_tensor("output_index", output_index);
    if (return_value) {
        DS_HOST_CHECK(output_value.has_value(),
                      "`output_value` must not be `None` when `return_value` is True");
        const tvm::ffi::TensorView &ov = output_value.value();
        DS_HOST_CHECK(dsf::same_dtype(ov.dtype(), input.dtype()),
                      "output_value dtype must match input dtype");
        check_out_tensor("output_value", ov);
    }
    // The per-row tables are read as `table[row]`, so `stride(0)` must be 1 and
    // the tensor must be on the device (a wrong-device pointer faults on read).
    auto check_row_table = [&](const char what[],
                               const tvm::ffi::TensorView &t) {
        DS_HOST_CHECK(t.device().device_id == input.device().device_id,
                      what, " must be on the same device as `input`");
        DS_HOST_CHECK(t.ndim() == 1 && dsf::size(t, 0) == batches &&
                          dsf::stride(t, 0) == 1,
                      what, " must be a contiguous tensor of `batch_size` entries");
    };
    if (end.has_value()) check_row_table("end", end.value());
    if (output_idx_offset.has_value()) {
        check_row_table("output_idx_offset", output_idx_offset.value());
    }

    const tvm::ffi::TensorView &ov =
        return_value ? output_value.value() : output_index;

    RowParams p{};
    p.input = dsf::const_data_ptr(input);
    p.output_value = return_value ? dsf::data_ptr(ov) : nullptr;
    p.output_index = dsf::data_ptr(output_index);
    p.end_ptr = end.has_value() ? dsf::data_ptr<int32_t>(end.value()) : nullptr;
    p.idx_offset_ptr = output_idx_offset.has_value()
                           ? dsf::data_ptr<int32_t>(output_idx_offset.value())
                           : nullptr;
    // The kernel offsets rows in *bytes*; `stride` counts elements.
    p.stride_input_batch = (uint64_t)dsf::stride(input, 0) * dsf::element_size(input);
    p.stride_output_value_batch =
        return_value ? (uint64_t)dsf::stride(ov, 0) * dsf::element_size(ov) : 0;
    p.stride_output_index_batch =
        (uint64_t)dsf::stride(output_index, 0) * dsf::element_size(output_index);
    p.vocab_size = (uint32_t)vocab_size;
    p.topk = (uint32_t)topk;
    // A zero here sizes every grid below to nothing -- an empty answer, not a
    // crash -- and the caller cannot fail to know it (it reads the number off
    // the device).  Refuse rather than round it to something plausible.
    DS_HOST_CHECK(sm_count > 0, "sm_count must be > 0 (got ", sm_count, ")");
    p.sm_count = (uint32_t)sm_count;
    // The *other* machine number -- the shared-memory budget -- is not read
    // here and not passed here: it is `ARCH_SMEM_PER_AP_BYTES`, a compile-time
    // constant of the family this artifact was built for (`csrc/structs.h`).
    // It used to be a `cudaDeviceGetAttribute` on this line, which is the same
    // number obtained a call later and from a place that did not need to be
    // asked: the artifact already knows its family, `_binding.py` loads the
    // artifact matching the device, and the only way the two can disagree is a
    // call that crossed devices -- which this file would rather refuse than
    // silently re-tune for.
    p.idx_fill = (int32_t)idx_oob_fill_value;
    p.value_fill = (float)value_oob_fill_value;
    p.check_nan = check_nan;
    p.abort_on_nan = abort_when_nan_found;

    const int value_dtype = float32 ? 0 : 1;
    const int index_dtype = dsf::same_dtype(output_index.dtype(), dsf::kInt32) ? 0 : 1;
    // The FFI environment holds torch's current stream while the python
    // facade is inside `tvm_ffi.use_torch_stream()` (deep_select/_binding.py).
    // Outside it TVMFFIEnvGetStream reports the null handle, which is the
    // legacy default stream -- the same thing the torch build launched on
    // when no stream was set, so this is not a behavior change.
    const cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(
        (int32_t)dsf::device_type(input), dsf::device_index(input));

    // Both buffers are built HERE rather than taken from the caller.  The
    // lengths table is an internal detail of the split's stage 1 -- `end`
    // absent already means the whole row -- so keeping it here leaves the
    // public entry a pure DLTensor boundary.  The workspace is raw cudaMalloc
    // rather than a torch tensor, because a torch tensor here is exactly the
    // coupling the tvm-ffi migration removes; it is synchronous on the calling
    // thread, so it is ordered before the launches on any stream, and cached
    // across calls -- see `ChunkedScratch`.
    static const bool kCacheScratch = [] {
        const char *v = std::getenv("DEEP_SELECT_NO_SCRATCH_CACHE");
        return !(v != nullptr && v[0] == '1' && v[1] == '\0');
    }();

    // The cache is process-wide while the kernels here have no thread-safety
    // contract of their own, so the scratch carries its own lock rather than
    // assuming the caller serializes.  Held across the launches, because the
    // buffers are not done being read when this function returns.
    struct ScratchGuard {
        std::mutex *m;
        explicit ScratchGuard(std::mutex *mu) : m(mu) { if (m) m->lock(); }
        ~ScratchGuard() { if (m) m->unlock(); }
    };
    ChunkedScratch &scratch = chunked_scratch();
    ScratchGuard scratch_guard(kCacheScratch ? &scratch.mu : nullptr);
    bool scratch_borrowed = false;

    int32_t *lengths = nullptr;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    // One scratch arm per split.  The two differ in dtype and in the workspace
    // layout they derive from it, so they are separate branches over a shared
    // allocator/table front half; the fp32 arm takes the same `p.end_ptr` route
    // (its split requires `end_ptr != nullptr`).
    const bool bf16_split =
        value_dtype == 1 && detail::chunked_bf16_applies(p, (uint32_t)batches);
    const bool f32_split =
        value_dtype == 0 && detail::chunked_f32_applies(p, (uint32_t)batches);
    // The routes that answer a row *without* ranking it -- the splits and the
    // coarse12 arm -- read the per-row scan table in the contract half.  The
    // shape gate above is not enough to know whether one of them will be taken
    // (the splits are reached through `topk_launch`'s own predicates and its
    // workspace check), so the table is provided for the whole `check_nan`
    // class and only when a route really asks for it: staying inside that class
    // costs one `cudaMemsetAsync` over `batches` ints, and it is that memset
    // alone that makes reading a slot legal.  A plain row run under
    // `check_nan` scans the row itself and never reads this, but the memset it
    // pays for is 255x smaller than the row scan it replaces (see
    // `f32_coarse12_applies`), so the class is the right place to draw the line.
    const bool needs_scan_table =
        check_nan && (bf16_split || f32_split ||
                      (value_dtype == 0 && (detail::f32_coarse12_applies(p, (uint32_t)batches) ||
                                            detail::f32_chunks_applies(p, (uint32_t)batches))));
    // The coarse12 answer buffer, allocated by the same rule: a call the route
    // can reach gets one, so `topk_launch` can stage the columns it ranks
    // whatever the caller's index width is.  Allocated up front rather than
    // inside the dispatch because the dispatch is the kernel layer and this is
    // a host allocation, and because a failure here has to fall back to the row
    // path rather than fault.  The `b <= 2` chunks arm shares it -- same kind of
    // answer (row columns), same contract half, and the two gates are disjoint.
    const bool needs_coarse12_cols =
        value_dtype == 0 && (detail::f32_coarse12_applies(p, (uint32_t)batches) ||
                             detail::f32_chunks_applies(p, (uint32_t)batches));
    if (needs_coarse12_cols) {
        const size_t need_cols = (size_t)batches * (size_t)p.topk;
        if (need_cols > scratch.coarse12_cols_count) {
            int32_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(
                cudaMalloc(&grown, need_cols * sizeof(int32_t)));
            if (scratch.coarse12_cols)
                DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.coarse12_cols));
            scratch.coarse12_cols = grown;
            scratch.coarse12_cols_count = need_cols;
        }
    }
    // The chunks route's workspace, on the same rule: sized for the row count
    // this call has, allocated once and grown, and its absence is a fallback to
    // the row path rather than a fault.  The count itself comes from
    // `select_chunk_count`, which needs the device's SM count -- read here
    // rather than passed in, because this is the sizing half and the launch
    // half has to agree with it exactly.
    if (value_dtype == 0 && detail::f32_chunks_applies(p, (uint32_t)batches)) {
        const size_t need_ws = rk::dgchunks::chunks_workspace_bytes(
            (uint32_t)batches, p.vocab_size, p.sm_count);
        if (need_ws > scratch.chunks_workspace_bytes) {
            void *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_ws));
            if (scratch.chunks_workspace)
                DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.chunks_workspace));
            scratch.chunks_workspace = grown;
            scratch.chunks_workspace_bytes = need_ws;
        }
    }
    if (bf16_split || f32_split) {
        const int f32_chunks =
            detail::f32_chunked_chunks((uint32_t)batches, p.topk,
                                       p.vocab_size, p.sm_count);
        const size_t need_lengths = (size_t)batches * sizeof(int32_t);
        const size_t need_workspace =
            f32_split
                ? detail::chunked_f32_workspace_bytes((uint32_t)batches,
                                                      (uint32_t)topk, f32_chunks)
                : detail::chunked_workspace_bytes(
                      (uint32_t)batches, (uint32_t)topk,
                      (uint32_t)detail::chunked_chunks(p));

        if (!kCacheScratch) {
            if (!end.has_value()) {
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&lengths, need_lengths));
                std::vector<int32_t> host_lengths((size_t)batches,
                                                  (int32_t)vocab_size);
                DS_CUDA_RUNTIME_CHECK(cudaMemcpy(lengths, host_lengths.data(),
                                                 need_lengths,
                                                 cudaMemcpyHostToDevice));
                p.end_ptr = lengths;
            }
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, need_workspace));
            workspace_bytes = need_workspace;
        } else {
            // Grow-only.  A `cudaFree` here would be correct but would give back
            // exactly the cost this buffer exists to avoid, so the peak is held.
            if (need_workspace > scratch.workspace_bytes) {
                void *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_workspace));
                if (scratch.workspace) DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.workspace));
                scratch.workspace = grown;
                scratch.workspace_bytes = need_workspace;
            }
            if (need_lengths > scratch.lengths_count * sizeof(int32_t)) {
                int32_t *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_lengths));
                if (scratch.lengths) DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.lengths));
                scratch.lengths = grown;
                scratch.lengths_count = (size_t)batches;
                scratch.lengths_epoch_batches = ChunkedScratch::kNoEpoch;
                scratch.lengths_epoch_vocab = ChunkedScratch::kNoEpoch;
            }
            if (!end.has_value()) {
                // `[vocab_size] * batches` is a pure function of two arguments of
                // this call, so "is the table already right" is a question about
                // that pair -- `lengths_epoch` is set only by the fill below, and
                // clearing it on a resize makes "no epoch" mean "must refill"
                // rather than a fabricated `(0, 0)`.
                if (scratch.lengths_epoch_batches != batches ||
                    scratch.lengths_epoch_vocab != (int64_t)vocab_size) {
                    std::vector<int32_t> host_lengths((size_t)batches,
                                                      (int32_t)vocab_size);
                    DS_CUDA_RUNTIME_CHECK(cudaMemcpy(scratch.lengths,
                                                     host_lengths.data(),
                                                     need_lengths,
                                                     cudaMemcpyHostToDevice));
                    scratch.lengths_epoch_batches = batches;
                    scratch.lengths_epoch_vocab = (int64_t)vocab_size;
                }
                p.end_ptr = scratch.lengths;
            }
            workspace = scratch.workspace;
            workspace_bytes = need_workspace;
            scratch_borrowed = true;
        }
    }

    // The per-row scan table, owned here rather than by the route that reads
    // it.  Only `check_nan` calls that can reach a route answering a row
    // without ranking it get one (`needs_scan_table` above), the table is
    // zeroed on the current stream so the kernel that raises a flag is ordered
    // after it, and it is *not* UNinitialized state a reader could trip over
    // -- a slot is read on the first call that reads it at all.
    //
    // Zeroing per call rather than tagging an epoch is deliberate: the flag
    // lives for exactly one call (it answers "is this row poisoned" for the
    // input as it is now), so a table reused across calls would have to be
    // cleared in the same place anyway, and a flag left raised from a previous
    // call is the one failure mode that is silent.
    if (needs_scan_table) {
        if (p.scan_flags == nullptr) {
            if ((size_t)batches > scratch.scan_flags_count) {
                int32_t *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, (size_t)batches * sizeof(int32_t)));
                if (scratch.scan_flags)
                    DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.scan_flags));
                scratch.scan_flags = grown;
                scratch.scan_flags_count = (size_t)batches;
            }
            p.scan_flags = scratch.scan_flags;
        }
        DS_CUDA_RUNTIME_CHECK(cudaMemsetAsync(p.scan_flags, 0,
                                              (size_t)batches * sizeof(int32_t),
                                              stream));
    }
    if (needs_coarse12_cols) p.coarse12_cols = scratch.coarse12_cols;
    if (value_dtype == 0 && detail::f32_chunks_applies(p, (uint32_t)batches))
        p.chunks_workspace = scratch.chunks_workspace;

    // The cached branch hands out the scratch's pointers directly, so the guard
    // must not free them -- it owns only what this call allocated itself (the
    // uncached branch, and nothing at all once `DEEP_SELECT_NO_SCRATCH_CACHE` is
    // off).
    struct FreeIfSet {
        void *p;
        ~FreeIfSet() { if (p) cudaFree(p); }
    } free_lengths{scratch_borrowed ? nullptr : (void *)lengths},
        free_workspace{scratch_borrowed ? nullptr : workspace};

    topk_launch(p, batches, (void *)stream, value_dtype, index_dtype,
                sorted_index, sorted_value, return_value, workspace,
                workspace_bytes);
    DS_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace deep_select

#include "../ffi/ffi_entries.h"
