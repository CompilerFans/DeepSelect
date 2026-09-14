// MACA-native row-wise top-K for DeepSelect.
//
// Reimplements the DeepSelect `topk` operator (`deep_select.interface.topk`)
// using only primitives that exist on the MetaX MACA platform.
//
// Relationship to the upstream `csrc/cuda_kernels/` trees as of 2026-09-11:
//   * `v3` (bf16) and `v3_fp32` have since been ported to MACA — TMA tensor-map
//     loads -> all-thread cooperative `ldg` + `__syncthreads`, mbarrier
//     pipeline -> single buffer, inline PTX -> MACA builtins/plain C++.  Both
//     now compile for xcore1000, but `setup.py` still builds neither: this file
//     is the shipping backend.  See `csrc/cuda_kernels/common_parts.cuh`.
//   * `v3_cluster` is deleted outright (MACA has no cluster launch and no TMA).
//
// ── the contract layer, and the one dataflow under it ───────────────────────
//
// Every mode shares one contract layer: the `length <= topk` shortcut, the NaN
// bit-pattern check (`abort_when_nan_found` / the 0x3F3F3F3F guard), the index
// offsets, the out-of-band fills, and the optional ordered emit.  One selection
// dataflow feeds it for both dtypes: the ported radix core
// (`radix_core.cuh`, provenance banner in that file) -- one histogram pass over
// the high key byte plus one vectorized collect pass that refines the threshold
// bin in shared memory, so two passes over the row whatever the key width.
// The 16-bit row walks a bf16 key, the 32-bit row a fp32 one; the dtype picks
// the row entry, not the dataflow.
//
// What that replaced, for the record: this file used to walk the key one byte
// at a time (most significant first).  Each round histogrammed the current byte
// over the elements still matching the confirmed prefix, suffix-scanned it, and
// took the bin the k-th remaining element fell in as the pivot, appending
// everything strictly above it to the answer.  The pivot bin was not carried
// between rounds -- the next round rescanned the row under the extended prefix,
// which is what makes the refine exact (carrying only the ties that fit would
// discard precisely the values the next byte has to rank).  That is 2*R passes
// over the row, R = 4 for fp32 and 2 for bf16, against the ported core's 2, and
// it was element-at-a-time with no vectorized body.  The ported core keeps the
// same no-truncation rule in a different shape: a threshold bin too large for
// the candidate arena falls back to a full-row rescan.
//
// Portable primitives only: `__shfl_down_sync`, `atomicAdd`, `__ldg`,
// `__syncthreads`, `__syncthreads_or`.  No inline asm, no TMA, no mbarrier,
// no cluster.
//
// The wave width is MACA's 64 lanes, matching the in-tree reference.  The
// CUDA-era code in this repo assumed 32, which does not hold on this target.
//
// ── coverage: the `v3_cluster` dispatch arm is dropped ──────────────────────
// The original dispatch (`csrc/xcore1600/api.cu`) sends
//     batch_size <= 6 && vocab_size >= 512K && topk <= 1024
// bf16 shapes to `topk_select_bf16_cluster`, a cluster-cooperative kernel.
// MACA has no cluster launch (mcErrorInvalidConfiguration for any cluster
// dim; see deep_jit/backend/maca/kernel.hpp), so that kernel and its dispatch
// arm have been deleted, and
// those shapes fall through to this single general kernel -- the same path
// every other shape takes.  They are NOT a hole: `topk <= 1024 <= 4096` holds
// and vocabulary is unbounded here (the kernel makes 2 passes over the row
// regardless of length), so the arm's shapes are served, only without the
// cluster-specific scheduling the original gave them.  Covered by the official
// performance grid's indexer cases (`tests/test.py --perf-only`: bf16,
// `topk` 512 and 1024, `vocab` up to 1M, `batch` 6..4096), which check the
// selection before they time it.
//
// The other CUDA-era dispatch arms keep their behaviour: `v3` (bf16) and
// `v3_fp32` both become this kernel, with the fp32/bf16 split now a template
// parameter rather than a separate source tree.

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
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;
    uint32_t vocab_size;
    uint32_t topk;
    int32_t idx_fill;
    float value_fill;
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
// NaN is NOT the all-ones key.  The order-preserving encode sends the two
// signed NaNs to opposite ends of the key space (fp32 0x7FFFFFFF -> 0xFFFFFFFF,
// but 0xFFFFFFFF -> 0x00000000), and a bf16 key is only 16 bits wide, so it can
// never equal a 32-bit all-ones constant at all.  Comparing keys therefore
// catches nothing but the single fp32 encoding 0x7FFFFFFF.
//
// Test the raw bit pattern instead: exponent all ones with a non-zero payload
// is exactly the set of NaNs, either sign, quiet or signaling -- the same set
// upstream detects with `set.nan.f32.f32` / `set.nan.bf16x2.bf16x2`.  (`v != v`
// would be the obvious spelling, but the build enables `--use_fast_math`.)
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
// Emits the first `n_out` slots of `selected` in the order the caller asked
// for, by packing each one with its sort key and running the CTA-wide bitonic
// network over the packed words.  The low 12 bits carry the slot, so ties keep
// a deterministic order and a value - index pair can be recovered exactly
// (`cmp` is always < vocab, so it never reaches those 12 bits).
//
//   SI: index ascending  -> (index << 12) | slot
//   SV: value descending -> ((~key) << 12) | slot
//
// `n_pad` is `topk` rounded up to a bitonic power of two; the tail is padded
// with `~0ull`, which sorts past every real key and is never emitted.
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
// The contract layer -- the `length <= topk` shortcut, the NaN path, the fills,
// the offsets, the ordered emit -- over the ported radix selection
// (`radix_core.cuh`): two passes over the row (a histogram of the high key
// byte, then one vectorized collect that writes everything above the threshold
// bin straight to the output and refines that bin in shared memory), whatever
// the key width.
//
// Dynamic shared memory starts with the core's arena -- `s_input_flat` is an
// `extern __shared__` array inside the header, so it can only sit at the base --
// and is followed by the staging buffer the emit reads.  The ordered emit's
// scratch is only live after the arena is dead, so it aliases it; the worst
// case (topk = kMaxTopK, sorted) is 48 KB, the same budget as the retired
// byte-wise path's `Arena` (and what `kSmemBudgetBytes` reserves).
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
// The bit-pattern test is one compare per element and the bytes have to be read
// either way, so the pass is pure overhead in the best case.  Written one
// element per load it was also the only pass in the row dataflow that moved no
// vector: the radix passes move eight 16-bit (or four 32-bit) elements per load,
// this moved one.  Both readers of the row go through here -- the contract
// half's own window check and the chunked path's `nan_scan_kernel`.
//
// Vector while the row slice is 16-byte aligned (it always is: rows are padded
// to a 1024-byte stride and the chunk bases are multiples of 8 elements), then
// a scalar tail for the remainder.
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
    bool nan_local = false;
    if constexpr (kPreSelected) {
        nan_local = params.nan_flags[row] != 0;
    } else {
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
// are long: B=6 at L=1M is 6 CTAs for the whole device.  The ported split ranks
// each row across `kChunkedChunks` CTAs and merges, and this is where
// DeepSelect enters it.
//
// Measured on C500 (bf16, k=512, cudaEvent, warm) against the row path:
// B=6/L=1M 2.03 -> 0.18 ms, B=6/L=262144 0.43 -> 0.074 ms; at B=256/L=1M the
// same split is 1.34x, which the batch bound keeps out.  Upstream's own gate
// (`needs_chunked`: L >= 1M) is the same idea with a higher floor; 262144 is
// where the row count is still the problem here.
constexpr uint32_t kChunkedMaxBatches = 64;
constexpr uint32_t kChunkedMinVocab = 262144;

// The chunk count is SM-count-sensitive: a grid of `kBatch * chunks` CTAs
// leaves `ctas mod SM` SMs idle when it is not a whole number of waves, and the
// same 16 is a different fraction of a wave on a 104-AP C500, a 28-SM C600 and
// a 32-SM C600U.  `NATIVE_SM_COUNT` (csrc/structs.h) is the compile-time fact
// that makes this decidable per build.
//
// The rule, stated so it can be argued with: **keep the measured count where it
// already fills at least half of its last wave, and otherwise round up to the
// next count that fills a whole number of waves.**  The half-wave floor is what
// keeps C500 exactly where it was measured -- b6 x 16 = 96 CTAs over 104 APs
// leaves 96 of the last wave's 104 APs busy (92%), so 16 is kept and no number
// recorded anywhere moves.  An architecture whose SM count would leave that grid
// under half a wave gets a count that fills it instead.
constexpr int wave_filled_chunks(int base_chunks) {
    constexpr int kBatch = 6;                      // the split's gate is small
    constexpr int kSms = (int)NATIVE_SM_COUNT;
    const int ctas = kBatch * base_chunks;
    const int last_wave = ctas % kSms;
    if (last_wave == 0 || last_wave * 2 >= kSms) return base_chunks;
    return ((ctas + kSms - 1) / kSms) * kSms / kBatch;
}

// [MACA] SM-count-sensitive.  C500's 104 APs are what 16 was measured against
// (b6 x 16 = 96 CTAs = 92% of the last wave, and the chunk sweep shows 16-32
// flat there); C600 (28) and C600U (32) round it by `wave_filled_chunks`.
constexpr int kChunkedChunks = wave_filled_chunks(16);

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

inline size_t chunked_workspace_bytes(uint32_t batches, uint32_t topk) {
    const size_t candidates = (size_t)batches * kChunkedChunks * topk;
    // The merge reads `candidate_values` with the vectorized row path, so the
    // two candidate arrays are 16-byte apart at the seam.
    const size_t indices_bytes = candidates * sizeof(int32_t);
    const size_t values_bytes = candidates * sizeof(maca_bfloat16);
    return indices_bytes + values_bytes + (size_t)batches * topk * sizeof(int32_t) +
           (size_t)batches * sizeof(int32_t);
}

inline ChunkedWorkspace chunked_workspace(void *base, uint32_t batches,
                                          uint32_t topk) {
    const size_t candidates = (size_t)batches * kChunkedChunks * topk;
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
// Same shape as the 16-bit split below, and deliberately the same gate to
// start with.  The fp32 row kernel is slower per row than the 16-bit one, so
// the split probably starts paying at a shorter row than 262144 -- but that is
// a measurement this change does not have, and the two splits sharing a gate
// also shares one code path for the scratch and the `end_ptr` route.
// The 16-bit split's 16, which is also what `nan_scan_kernel` is launched with,
// so the two splits agree about the chunk geometry.  Measured on the cells the
// gate admits (b6, kernel time, C500, CUDA_VISIBLE_DEVICES=3):
//
//   chunks        8      16      24      32
//   v262144     97.8    84.7    83.5    87.7
//   v524288    144.2   108.3   107.7   104.2
//
// Flat from 16 up; 8 is short of CTAs (48).  24 is within noise of 16 and 32
// trades the two columns off, so the shared value is the pick *on C500*.
//
// On a part with a different SM count the same 16 is a different fraction of a
// wave, so the two splits round *up* to a whole number of waves on whatever
// architecture the extension was built for (`NATIVE_SM_COUNT`, csrc/structs.h
// -- 104 on C500, 28 on C600, 32 on C600U, all compile-time).  This is the one
// place where a number the split is made of is SM-count-sensitive.
//
// **The bodies below were measured on C500 ONLY.**  What follows is therefore
// not a re-sweep on C600/C600U -- it is the arithmetic of the existing
// measurements carried to 28 and 32 SMs, with the cells where it stops being
// justified marked as such.  Both split geometry functions are pure, so this is
// computable here rather than assertable.
//
// `f32_chunks_small_batch(batches)` used to be the fixed `wave_filled_chunks(16)`
// and is now `f32_chunks_for(batches)` -- see its definition below, and
// `NATIVE_F32_CHUNK_WORK_TARGET` in `csrc/structs.h` for the measurement.  What
// follows is that older rule's SM-count arithmetic, kept because it is why the
// fixed value was believed to be right, and because the reason it was wrong is
// the same reason this family of constants exists:
//
//   family        SMs  b6x16 % SMs   result   b6x(chunks+1) % SMs
//   C500          104        96       16        102  (98%)   measured
//   C600           28        84       32        198  (7%)
//   C600U          32        96       16        102  (19%)
//   C600 (b8)      28        96       16        136  (24%)
//
// That rule keys on `kBatch = 6` because 6 is the split's gate, so it answers
// for b6 -- and it answers *only* for b6.  At b24 the same 16 chunks is 21%
// slower than 8, and at b48/b64 it is 27%/31% slower than 4; the old note below
// ("the small-batch arm keeps the value that was measured on it") was true of
// b6 and silently extended to a whole tier.  The work-target form removes that
// extension: it is a function of both quantities the optimum actually depends
// on, and it reduces to 16 at b6, which is where the measurement was taken.
//
// `f32_chunks_large_batch()` = 2 (constant): the chunk count produced
// `2 * batches` CTAs, and the merge `batches`.  So on a 28-SM C600 the merge is
// 28 whole waves at b4096/28 = 146.3, and 146 of those 147 waves are merge -- a
// 2:1 work ratio.  On a 32-SM C600U the same ratio holds at 128 waves + 128.
// **This is the one place where the C500 measurement does not carry: the C500
// sweep that picked 2 ran the merge 39 waves deep against 78 chunk waves.**
// The remedy that the sweep itself points at is to drop the merge's CTA per row
// (`chunks + 1` -> `chunks`) for the large-batch arm, which the sweep already
// supports (chunks = 1: 524.7 us on C500, statistically indistinguishable from 2
// at b256-v262144, and never measured at b4096).  That is a change to the merge
// kernel's launch, not a constant, so it is not made on this arithmetic alone.
namespace {
// The measured chunk counts.  Both are sweeps on C500 with the contract checked
// at every point and two alternating passes per point (`sweep18.py` for these
// calls, `chunk_sweep.py` for the wider grid).  The two regimes disagree, and
// the disagreement is the whole reason this is a function of the batch:
//
//   chunks   b6-v262144   b256-v262144   b4096-v262144   b4096-v524288
//        1           --           524.7           --              --
//        2           --           525.2        7,112          13,358
//        3           --              --        7,476          13,645
//        4           --           666.9        9,226          13,747
//        6           --              --        9,645          14,396
//        8        97.8           680.2        9,850          17,864
//       12           --           721.9           --              --
//       16        84.7           760.6       11,316          19,241
//       24        83.5           898.8           --              --
//       32        87.7           988.0       15,093          22,154
//       64       119.4              --       21,985          29,800
//
// One direction *at large*: the long-row cells above all prefer FEWER chunks
// monotonically, which is why the large-batch arm is a constant 2.  The
// short-row columns are a **different** curve, and the fixed 16 was wrong on
// them: `chunks = 16` is not the optimum at every batch, because what the
// chunks are for is filling the machine, and 16 chunks overshoot badly once
// `batches * (chunks + 1)` is already past a couple of waves.
//
// Measured at `V = 32768`, one binary, only `DEEP_SELECT_F32_CHUNKS` varying,
// 3 alternating rounds, median (`chunk_boundary.py`; the full table is at
// `NATIVE_F32_CHUNK_WORK_TARGET`'s definition in `csrc/structs.h`): the optimum
// is `16, 8, 8, 4/8, 4, 4, 4, 2, 2, ...` for batches `6, 16, 24, 32, 40, 48,
// 64, 80, 96, ...`.  That is `largest power of two <= K / batches`, with K a
// property of the machine.  Against the fixed 16 the losses were 21.0% at
// b24-v65536, 26.9% at b48-v65536 and **31.0% at b64-v65536** -- all on shapes
// whose gate was already open, so this costs no new path.
constexpr int f32_chunks_for(uint32_t batches) {
    const uint32_t b = batches == 0 ? 1u : batches;
    int c = 2;
    while (c < 16 && (uint64_t)(c * 2) * b <= NATIVE_F32_CHUNK_WORK_TARGET) c *= 2;
    return c;
}
constexpr int f32_chunks_small_batch(uint32_t batches) {
    return f32_chunks_for(batches);
}
constexpr int f32_chunks_large_batch() { return 2; }

// The batch at which the chunk count stops being a parallelism knob.  It is the
// split's own small-batch ceiling (64), which is a property of the gate rather
// than a fitted value: at 6 rows the machine is empty and chunks are the only
// CTAs there are; at 256 the grid is 256 * chunks CTAs and 256 final modules
// already covers C500's 104 APs twice over.
inline constexpr uint32_t kF32ChunksFewBatches = 64;
}  // namespace

int f32_chunked_chunks(uint32_t batches) {
    static const int override_n = [] {
        const char *v = std::getenv("DEEP_SELECT_F32_CHUNKS");
        if (v != nullptr && v[0] != '\0') {
            const int d = std::atoi(v);
            return (d >= 2 && d <= 256) ? d : 0;
        }
        return 0;
    }();
    if (override_n != 0) return override_n;   // A/B knob; does not change default
    return batches <= kF32ChunksFewBatches ? f32_chunks_small_batch(batches)
                                           : f32_chunks_large_batch();
}

// The batch bound has no single value, because what the split costs is a merge
// over `batches` CTAs on an otherwise idle machine, and what it saves is the
// row kernel's per-CTA time.  Both are measurable, and the measured gate has
// two tiers (kernel time, C500, `maca_arm.py`):
//
//   batches   vocab      row us   split us   speedup
//       6     65536       124.5       64.8     1.92x      <- b6 tier
//       6    129280       256.1       69.1     3.71x
//       6    262144       425.8       84.9     5.02x
//     256     65536       326.1      394.3     0.83x      <- b256 LOSES here
//     256    262144      1044.3      762.7     1.37x      <- b256 tier opens
//     256    524288      1924.4     1282.2     1.50x
//
// and at b256 the split *loses* below 262144 (394 vs 326) while winning above
// it.  That "loses" half no longer holds, and the reason is the chunk count
// above: when this floor was measured the split's cost was `chunks` CTAs per
// row, so at 16 chunks b256-v65536 was 256 * 17 = 4,352 CTAs deep in a
// chunk-merge the machine had no room to hide.  With the count now a constant 2
// for every batch above 64, the merge is 3 CTAs per row at every shape, so the
// cost no longer scales with the gate and the floor is free to drop to the
// small-batch tier's value.  Measured (kernel time, C500, three alternating
// rounds, `floor_ab.py`; both sides contract-checked, 16546/16552/16544 vs
// 18955/18963/18949 us over the eight cells):
//
//   cell              floor 262144   floor 65536
//   b4096-v  65536       3485.1        2738.7      -21.4%
//   b4096-v 129280       5845.5        4802.4      -17.8%
//   b  256-v  65536       330.2         217.9      -34.0%   <- was the "loses" cell
//   b  256-v 129280       534.6         366.4      -31.5%
//   b  512-v 129280       846.4         684.3      -19.1%
//   b  768-v  65536       741.5         564.2      -23.9%
//   b    6-v  65536        66.3          66.4       +0.2%
//   b 4096-v 262144      7113.1        7110.3       -0.0%
//   TOTAL               18962.7       16551.4        -12.7%
//
// The two neutral cells are the contract: b6 already opened (the tier is `<=`
// on both) and b4096-v262144 was already served.  Both tiers now sit on the
// measured side of the same knee, which is the small-batch tier's, and no
// longer on two different ones.
//
// **The band `32768 <= V < 65536` is what that unification newly serves**, and
// it was measured before being opened (kernel time, C500, round-robin A/B: both
// arms in every round, order alternated, 5 rounds, median of the paired
// per-round ratios; `ab_b.py`).  Six controls at `V >= 65536`, which the change
// cannot reach, hold to |delta| <= 0.3% with per-round spreads <= 1.4% -- that
// is the run's own noise floor, and it is what makes the rest readable:
//
//   cell              head us   cand us   delta    GB/s  h -> c    %1W h -> c
//   b   6-v 32768        71.6      57.6   -19.6%    22.0 -> 27.3    1.3 -> 1.7
//   b  32-v 32768        81.3      62.1   -23.5%   103.2 -> 135.1   6.3 -> 8.2
//   b  64-v 32768        91.0      75.0   -17.8%   184.4 -> 223.7  11.2 -> 13.6
//   b 256-v 32768       181.8     149.3   -17.1%   369.1 -> 449.5  22.4 -> 27.2
//   b1024-v 32768       496.3     468.0    -5.6%   540.9 -> 573.6  32.8 -> 34.8
//   b4096-v 32768      1858.4    1712.2    -7.8%   577.8 -> 627.1  35.0 -> 38.0
//   b 256-v 49152       255.9     184.9   -27.7%   393.4 -> 544.4  23.8 -> 33.0
//   b4096-v 49152      2659.8    2200.0   -17.4%   605.5 -> 732.1  36.7 -> 44.4
//
// Logical GB/s is `2 * batches * V * 4 / time` (the split's chunk stage reads
// the row once and the merge reads `2 * topk` candidates), so `%1W` is the
// distance to the best case, not an occupancy.
//
// The magnitude falls as the batch rises, and that is the same curve every
// other result in this file shows: at b6 the row kernel has 6 CTAs on 104 APs
// and the split's 6 * 17 = 102 CTAs are a near-perfect wave; at b4096 the row
// kernel already has 39 waves of its own and the split is buying a shorter
// dependency chain per row, not a fuller machine.  It never goes negative, so
// no second knee is opened -- but the b1024/b4096 cells are the ones to
// re-measure if this band ever grows.
//
// ── the two tiers have collapsed into one ───────────────────────────────────
// There used to be four constants here: a small-batch tier (`V >= 65536`,
// `batches <= 64`) and a large-batch tier (`V >= 262144` -> later 65536,
// `batches <= 4096`).  With the large tier's floor at 32768 the small branch is
// **unreachable** -- it needs `V < 32768` and `V >= 65536` at once -- so the
// predicate is one condition, and the two constants below are what is left of
// it.  The tiers are gone rather than kept as documentation because a constant
// that cannot select anything is a knob a reader will try to turn; the history
// is the tables above.
//
// The measured band used the surviving cap throughout (b256/b1024/b4096 at
// V=32768 are all `batches <= 4096` cells and all wins), so this spelling is
// the one the numbers above were taken on, not a wider claim.
constexpr uint32_t kF32ChunkedMinVocab = 32768;
constexpr uint32_t kF32ChunkedMaxBatches = 4096;

// The split is fp32-only (the caller's `value_dtype == 0`) and its merge only
// compiles the k=512/1024 arms, which is the whole gate.
inline bool chunked_f32_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0) return false;
    // `end_ptr` absent means the row really is `vocab_size` long, so the cap is
    // the vocab either way -- and it is the only length the host knows without
    // a device read.
    if (params.vocab_size < kF32ChunkedMinVocab) return false;
    if (batches > kF32ChunkedMaxBatches) return false;
    return params.topk == 512 || params.topk == 1024;
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
    const int chunks = f32_chunked_chunks(batches);
    const ChunkedF32Workspace ws = chunked_f32_workspace(
        workspace, batches, params.topk, chunks);
    // `nan_flags` is the table.  It is memset and then OR-ed per row, so the
    // length is whichever ran.  `end_ptr` being non-null is the dispatch's
    // precondition and the dispatcher only reaches here with the scratch's own
    // all-`vocab_size` table, but clearing first makes the scan independent of
    // that rather than merely consistent with it.
    const size_t table_bytes = (size_t)batches * sizeof(int32_t);
    cudaMemsetAsync(ws.lengths, 0, table_bytes, stream);
    nan_scan_kernel<float><<<batches * chunks, kScanBlock, 0, stream>>>(
        params.input, params.end_ptr,
        (int64_t)(params.stride_input_batch / sizeof(float)),
        params.vocab_size, (uint32_t)chunks, ws.lengths);
    const cudaError_t rc = rk::launch_topk_f32_chunked(
        (const float *)params.input, params.end_ptr, ws.cols, ws.merged,
        (int)batches, (int)params.vocab_size, (int)params.topk, chunks, stream,
        // The row stride is in bytes at this layer and in elements there.
        (int64_t)(params.stride_input_batch / sizeof(float)));
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
    const ChunkedWorkspace ws = chunked_workspace(workspace, batches, params.topk);
    cudaMemsetAsync(ws.nan_flags, 0, (size_t)batches * sizeof(int32_t), stream);
    nan_scan_kernel<maca_bfloat16><<<batches * kChunkedChunks, kScanBlock, 0, stream>>>(
        params.input, params.end_ptr,
        (int64_t)(params.stride_input_batch / sizeof(maca_bfloat16)),
        params.vocab_size, kChunkedChunks, ws.nan_flags);
    const cudaError_t rc = rk::launch_topk_bf16_chunked(
        (const maca_bfloat16 *)params.input, params.end_ptr, ws.merged,
        ws.candidate_indices, ws.candidate_values, (int)batches,
        (int)params.vocab_size, (int)params.topk, kChunkedChunks, stream,
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
        chunked_bytes >= detail::chunked_workspace_bytes(n, params.topk) &&
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
    if (value_dtype == 0 && chunked_workspace != nullptr &&
        params.end_ptr != nullptr &&
        chunked_bytes >= detail::chunked_f32_workspace_bytes(
                             n, params.topk, detail::f32_chunked_chunks(n)) &&
        detail::chunked_f32_applies(params, n)) {
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
// `cudaMalloc`/`cudaFree` on this runtime are ~70-100 us per *pair* at the size
// class the candidate workspace lands in, and -- measured, not assumed -- the
// cost is **non-monotonic in size**, so a 12 MB allocation is nearly free while
// 300 KB is the worst case (`docs/C500-to-parity-plan.zh.md` item 0,
// `malloc_sweep.cu`).  The split needs this workspace plus a row-length table on
// every call, so paying a fresh pair each time put a ~265 us host floor in front
// of a ~96 us kernel at b6-v262144-k512 -- `e2e = max(host, device)`.
//
// So they are cached process-wide and **grown only**.  Growing rather than
// keying by shape is the whole design: a caller that alternates between two
// batches would realloc on every switch under a shape-keyed cache, which is the
// cost this exists to remove; a high-water-mark buffer never does.
//
// Three properties make reuse safe rather than merely fast:
//
//   * `chunked_workspace(base, batches, topk)` derives the layout by walking
//     forward from `base` by the *geometry of this call*, not by the capacity,
//     so a buffer larger than this call needs is correct by construction.
//   * the split's own kernels write every slot they read back -- stage 1 fills
//     each candidate slot it is asked for, and `nan_flags` is memset per call --
//     so nothing is inherited from the previous call.
//   * the lengths table is only skipped when `(batches, vocab_size)` -- both
//     arguments of this call -- already match the pair the table was last filled
//     for.  The table's content is `[vocab_size] * batches`, a pure function of
//     that pair, so the pair *is* the content; see `lengths_epoch` at the fill.
//
// The high-water mark is bounded by the gate that admits the split
// (`chunked_bf16_applies`: batches <= 64, topk <= 1024), i.e. ~6.3 MB for the
// workspace and 256 B for the table, held for the life of the process.  That is
// the trade: a bounded, one-time footprint in exchange for removing a per-call
// host cost that is larger than the kernel it fronts.
//
// One design point, since the natural instinct is to keep the `cudaFree` and
// tolerate the pair: free-then-malloc of the same size *should* be cheap (that
// is what an allocator does).  Measured on this runtime it is not -- 307,224 B
// costs 69.9 us/pair with the free included and 100.3 us/pair interleaved with
// device work, while the *same* loop over 12,582,912 B costs 1.2 us.  Freeing
// would therefore give back most of the win, so the buffer is held.
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
          bool return_value, bool abort_when_nan_found) {
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
    p.idx_fill = (int32_t)idx_oob_fill_value;
    p.value_fill = (float)value_oob_fill_value;
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

    // The chunked split wants the row lengths as a table and a candidate
    // workspace.  The lengths table is built HERE rather than by the caller:
    // it is an internal detail of the split's stage 1 (the whole row, which is
    // what `end` absent already means), it is at most `batch_size` int32s, and
    // keeping it here leaves the public entry a pure DLTensor boundary.
    //
    // The workspace is also ours: raw cudaMalloc rather than a torch tensor,
    // because a torch tensor here is exactly the coupling this migration
    // removes.  It is a synchronous allocation on the calling thread, so it is
    // ordered before the launches on any stream.
    // The workspace is also ours: raw cudaMalloc rather than a torch tensor,
    // because a torch tensor here is exactly the coupling this migration
    // removes.  It is cached across calls rather than freed at the end -- see
    // `ChunkedScratch` for why the allocation, not the kernel, was the floor.
    // `DEEP_SELECT_NO_SCRATCH_CACHE=1` restores the free-per-call behavior, which
    // is what makes the claim measurable as an A/B rather than argued.
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
    if (bf16_split || f32_split) {
        const int f32_chunks = detail::f32_chunked_chunks((uint32_t)batches);
        const size_t need_lengths = (size_t)batches * sizeof(int32_t);
        const size_t need_workspace =
            f32_split
                ? detail::chunked_f32_workspace_bytes((uint32_t)batches,
                                                      (uint32_t)topk, f32_chunks)
                : detail::chunked_workspace_bytes((uint32_t)batches,
                                                  (uint32_t)topk);

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
