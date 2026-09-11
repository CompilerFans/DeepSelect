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
// The original dispatch (`csrc/api.cpp:128`) sends
//     batch_size <= 6 && vocab_size >= 512K && topk <= 1024
// bf16 shapes to `topk_select_bf16_cluster`, a cluster-cooperative kernel.
// MACA has no cluster launch (mcErrorInvalidConfiguration for any cluster
// dim; see deep_jit/backend/maca/kernel.hpp), so that kernel and its dispatch
// arm have been deleted, and
// those shapes fall through to this single general kernel -- the same path
// every other shape takes.  They are NOT a hole: `topk <= 1024 <= 4096` holds
// and vocabulary is unbounded here (the kernel makes 2 passes over the row
// regardless of length), so the arm's shapes are served, only without the
// cluster-specific scheduling the original gave them.  Covered by the
// `b=4 v=524288 k=1024 bf16` case in `tests/check_maca.py`.
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

static __device__ __forceinline__ void radix_layout(
    uint8_t *base, uint32_t topk, bool sorted, uint32_t *&selected,
    uint64_t *&sort_buf) {
    uint32_t n_pad = 1;
    if (sorted) {
        while (n_pad < topk) n_pad <<= 1;
    }
    const size_t sort_bytes =
        sorted ? (size_t)n_pad * sizeof(uint64_t) : (size_t)0;
    sort_buf = reinterpret_cast<uint64_t *>(base);
    selected = reinterpret_cast<uint32_t *>(
        base + (sort_bytes > kRadixArenaBytes ? sort_bytes : kRadixArenaBytes));
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
    bool found = false;
    for (uint32_t i = start + threadIdx.x; i < end; i += kScanBlock) {
        found |= is_nan_value<ValueT>(__ldg(row_ptr + i));
    }
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
    radix_layout(arena_raw, params.topk, SI || SV, selected, sort_buf);

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
        for (uint32_t i = tid; i < length; i += BLOCK) {
            nan_local |= is_nan_value<ValueT>(__ldg(input_row + i));
        }
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
inline size_t radix_smem_bytes(uint32_t topk, bool sorted) {
    uint32_t n_pad = 1;
    if (sorted) {
        while (n_pad < topk) n_pad <<= 1;
    }
    const size_t sort_bytes =
        sorted ? (size_t)n_pad * sizeof(uint64_t) : (size_t)0;
    const size_t lead = sort_bytes > kRadixArenaBytes ? sort_bytes : kRadixArenaBytes;
    return lead + sizeof(uint32_t) * kMaxTopK;
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
template <typename ValueT, typename OutIdxT, int BLOCK, bool SI, bool RV, bool SV>
inline void launch_radix(const RowParams &params, uint32_t batches,
                         cudaStream_t stream, size_t smem, bool preselected) {
    if (preselected) {
        // The split is 16-bit only, so no other dtype instantiates this arm
        // (an fp32 instantiation could not be launched).
        if constexpr (std::is_same<ValueT, maca_bfloat16>::value) {
            set_radix_attr<ValueT, OutIdxT, BLOCK, SI, RV, SV, true>();
            topk_kernel_radix<ValueT, OutIdxT, BLOCK, SI, RV, SV, true>
                <<<batches, BLOCK, smem, stream>>>(params);
        }
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
constexpr int kChunkedChunks = 16;

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
    const size_t smem =
        radix_smem_bytes(params.topk, sorted_index || sorted_value);
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

// ── torch extension entry points ────────────────────────────────────────────
#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

std::pair<uint32_t, uint32_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
            OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

void topk(torch::Tensor &input, int topk, c10::optional<torch::Tensor> &begin,
          c10::optional<torch::Tensor> &end, bool sorted_value,
          bool sorted_index, c10::optional<torch::Tensor> &output_value,
          torch::Tensor &output_index,
          c10::optional<torch::Tensor> &output_idx_offset,
          int idx_oob_fill_value, float value_oob_fill_value,
          bool return_value, bool abort_when_nan_found) {
    using namespace deep_select_maca;

    const int64_t batches = input.size(0);
    const int64_t vocab_size = input.size(1);
    const at::ScalarType value_t = input.scalar_type();

    TORCH_CHECK(topk > 0, "topk must > 0");
    TORCH_CHECK(topk <= kMaxTopK, "topk must be <= ", kMaxTopK);
    TORCH_CHECK(!(sorted_value && !return_value),
                "`return_value` must be enabled when `sorted_value` is True");
    TORCH_CHECK(!(sorted_value && sorted_index),
                "`sorted_value` and `sorted_index` cannot be used at the same time");
    TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
    TORCH_CHECK(value_t == at::kFloat || value_t == at::kBFloat16,
                "input dtype must be float32 or bfloat16");
    TORCH_CHECK(output_index.scalar_type() == at::kInt ||
                    output_index.scalar_type() == at::kLong,
                "output_index dtype must be int32 or int64");
    TORCH_CHECK(input.stride(1) == 1, "input.stride(1) must be 1");
    TORCH_CHECK(input.stride(0) * input.element_size() %
                        (int64_t)INPUT_STRIDE_ALIGNMENT_REQUIREMENT ==
                    0,
                "input.stride(0) must be a multiple of ",
                INPUT_STRIDE_ALIGNMENT_REQUIREMENT, " bytes");

    // Every output row is addressed as `row * stride(0) + column`, so a
    // last-dimension stride other than 1 (or a row that is too short) writes
    // outside the columns the caller owns.  Upstream rejects both
    // (api.cpp KU_CHECK_LAST_DIM_CONTIGUOUS / KU_CHECK_SHAPE) -- without the
    // check the result is silently scrambled, so refuse instead.
    auto check_out_tensor = [&](const char what[], const torch::Tensor &t) {
        TORCH_CHECK(t.device() == input.device(),
                    what, " must be on the same device as `input`");
        TORCH_CHECK(t.stride(1) == 1, what, ".stride(1) must be 1");
        TORCH_CHECK(t.size(0) == batches && t.size(1) >= topk,
                    what, " must be at least (batch_size, topk) = (",
                    batches, ", ", topk, ")");
    };
    check_out_tensor("output_index", output_index);
    if (return_value) {
        TORCH_CHECK(output_value.has_value(),
                    "`output_value` must not be `None` when `return_value` is True");
        TORCH_CHECK(output_value->scalar_type() == value_t,
                    "output_value dtype must match input dtype");
        check_out_tensor("output_value", *output_value);
    }
    // The per-row tables are read as `table[row]`, so `stride(0)` must be 1 and
    // the tensor must be on the device (a wrong-device pointer faults on read).
    auto check_row_table = [&](const char what[], const torch::Tensor &t) {
        TORCH_CHECK(t.device() == input.device(),
                    what, " must be on the same device as `input`");
        TORCH_CHECK(t.numel() == batches && t.stride(0) == 1,
                    what, " must be a contiguous tensor of `batch_size` entries");
    };
    if (end.has_value()) check_row_table("end", *end);
    if (output_idx_offset.has_value()) {
        check_row_table("output_idx_offset", *output_idx_offset);
    }

    RowParams p{};
    p.input = input.data_ptr();
    p.output_value = return_value ? output_value->data_ptr() : nullptr;
    p.output_index = output_index.data_ptr();
    p.end_ptr = end.has_value() ? end->data_ptr<int32_t>() : nullptr;
    p.idx_offset_ptr =
        output_idx_offset.has_value() ? output_idx_offset->data_ptr<int32_t>()
                                      : nullptr;
    // The kernel offsets rows in *bytes*; `Tensor::stride` counts elements.
    p.stride_input_batch = (uint64_t)input.stride(0) * input.element_size();
    p.stride_output_value_batch =
        return_value ? (uint64_t)output_value->stride(0) * output_value->element_size() : 0;
    p.stride_output_index_batch =
        (uint64_t)output_index.stride(0) * output_index.element_size();
    p.vocab_size = (uint32_t)vocab_size;
    p.topk = (uint32_t)topk;
    p.idx_fill = idx_oob_fill_value;
    p.value_fill = value_oob_fill_value;
    p.abort_on_nan = abort_when_nan_found;

    const int value_dtype = (value_t == at::kFloat) ? 0 : 1;
    const int index_dtype = (output_index.scalar_type() == at::kInt) ? 0 : 1;
    auto stream = at::cuda::getCurrentCUDAStream().stream();

    // The chunked split wants the row lengths as a table and a candidate
    // workspace; both are sized here, where the gate can be asked before the
    // launch.  A shape outside the gate pays neither.
    torch::Tensor lengths_holder, workspace_holder;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    if (value_dtype == 1 &&
        deep_select_maca::detail::chunked_bf16_applies(p, (uint32_t)batches)) {
        if (!end.has_value()) {
            // `end` absent means the whole row, which is exactly the table the
            // split's stage-1 reads per row.
            lengths_holder = torch::full({input.size(0)}, vocab_size,
                                         input.options().dtype(at::kInt));
            p.end_ptr = lengths_holder.data_ptr<int32_t>();
        }
        workspace_bytes = deep_select_maca::detail::chunked_workspace_bytes(
            (uint32_t)batches, (uint32_t)topk);
        workspace_holder = torch::empty({(int64_t)workspace_bytes},
                                        input.options().dtype(at::kByte));
        workspace = workspace_holder.data_ptr();
    }

    topk_launch(p, batches, (void *)stream, value_dtype, index_dtype,
                sorted_index, sorted_value, return_value, workspace,
                workspace_bytes);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "topk launch failed: ",
                cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
    m.def("get_alignment_requirement", &get_alignment_requirement);
}
