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
    // The one machine number every grid-sizing decision below reads, supplied
    // by the caller from `deep_select/_arch.py`'s `SM_COUNT`.  An argument
    // rather than a compile-time constant because one extension serves all
    // three families, and not read from the driver because the device already
    // reported its architecture to the caller (torch), which names the family
    // this is keyed by.  Zero means "unknown", which `resolve_sm_count` turns
    // into an error rather than a plausible wrong grid.
    uint32_t sm_count;
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
// long: B=6 at L=1M is 6 CTAs for the whole device.  The split ranks each row
// across `chunked_chunks(params)` CTAs and merges.  It wins by 11x at B=6/L=1M and
// loses above ~256 rows, which is the batch bound.  Upstream's own gate
// (`needs_chunked`: L >= 1M) is the same idea with a higher floor; 262144 is
// where the row count is still the problem here.  Measurements in
// `docs/C500-radix-perf-ledger.zh.md`.
constexpr uint32_t kChunkedMaxBatches = 64;
constexpr uint32_t kChunkedMinVocab = 262144;

// The chunk count is SM-count-sensitive: a grid of `kBatch * chunks` CTAs
// leaves `ctas mod SM` SMs idle unless it is a whole number of waves, and the
// same 16 is a different fraction of a wave on a 104-AP C500, a 28-SM C600 and
// a 32-SM C600U.  `params.sm_count` (the caller's, from the device) makes that
// decidable at launch time.
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
// The `DEEP_SELECT_F32_CHUNKS` sweep behind the work target is tabulated at
// `deep_select/_arch.py`'s `F32_CHUNK_WORK_TARGET` -- do not re-derive it.
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
inline int f32_chunks_large_batch() { return 2; }

// The batch above which the chunk count stops being a parallelism knob: 64 is
// the split's own small-batch ceiling, a property of the gate rather than a
// fitted value.  At 6 rows the machine is empty and chunks are the only CTAs
// there are; at 256 rows the grid already covers the machine twice over.
inline constexpr uint32_t kF32ChunksFewBatches = 64;
}  // namespace

int f32_chunked_chunks(uint32_t batches, uint32_t sm_count) {
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
    const int ceiling = f32_chunk_ceiling_32() ? 32 : kF32ChunkCeiling;
    return batches <= kF32ChunksFewBatches
               ? f32_chunks_small_batch(batches, ceiling, work_target)
               : f32_chunks_large_batch();
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
    const int chunks = f32_chunked_chunks(batches, params.sm_count);
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
    const int chunks = chunked_chunks(params);
    const ChunkedWorkspace ws =
        chunked_workspace(workspace, batches, params.topk, (uint32_t)chunks);
    cudaMemsetAsync(ws.nan_flags, 0, (size_t)batches * sizeof(int32_t), stream);
    nan_scan_kernel<maca_bfloat16><<<batches * chunks, kScanBlock, 0, stream>>>(
        params.input, params.end_ptr,
        (int64_t)(params.stride_input_batch / sizeof(maca_bfloat16)),
        params.vocab_size, (uint32_t)chunks, ws.nan_flags);
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
    if (value_dtype == 0 && chunked_workspace != nullptr &&
        params.end_ptr != nullptr &&
        chunked_bytes >= detail::chunked_f32_workspace_bytes(
                             n, params.topk,
                             detail::f32_chunked_chunks(n, params.sm_count)) &&
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
          bool return_value, bool abort_when_nan_found,
          // SM count of the device this call runs on, from the caller's
          // architecture table (`deep_select/_arch.py`).  The last argument
          // rather than one near `topk` because it is the only one the caller
          // derives from the *device* rather than from the problem.
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
    if (bf16_split || f32_split) {
        const int f32_chunks =
            detail::f32_chunked_chunks((uint32_t)batches, p.sm_count);
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
