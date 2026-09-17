// ── reference extraction ────────────────────────────────────────────────────
//
// Upstream: `csrc/xcore1000/maca_topk.cu`.  Everything above the host entry at
// the bottom of this file is upstream lines 1-1108, verbatim, with exactly two
// edits, both in the include block: `"structs.h"` is dropped (see the note
// where it stood) and `"radix_core.cuh"` becomes `"xcore1000_radix_core.cuh"`
// (the file it is copied to here).
//
// What upstream had after line 1108, and what is not here:
//
//   * 1112-1421  the tvm-ffi entry `deep_select::topk(TensorView, ...)` --
//                dtype/shape/stride checks, the process-wide grow-only
//                `cudaMalloc` scratch cache under a mutex, the length table it
//                builds when the caller passes no `end`, and
//                `TVMFFIEnvGetStream` for the stream.  Replaced by the plain
//                host entry at the bottom of this file; the parts of it that
//                are the dataflow's own input (the length table, the
//                workspace sizing, the NaN-flag table) are carried over there.
//   * 1125-1425  the `deep_select` namespace and the `../ffi/ffi_entries.h`
//                registration tail.
//
// The kernel layer is untouched, so the whole contract surface is still here:
// the `length <= topk` shortcut, the NaN bit-pattern check and its two
// `abort_on_nan` behaviors, the out-of-band fills, the ordered emit, the
// `RowParams`-by-value ABI, `nan_scan_kernel`, and every routing predicate
// (`chunked_bf16_applies`, `chunked_f32_applies`, `topk_worth_splitting_f32`,
// `f32_chunked_chunks`, `chunked_chunks`, `radix_block_for`).
// ────────────────────────────────────────────────────────────────────────────

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

// [reference] upstream includes "structs.h" here.  Nothing in the kernel layer
// reads it: the only users were `INPUT_STRIDE_ALIGNMENT_REQUIREMENT` and
// `OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT` in the tvm-ffi edge (upstream lines
// 1185, 1220-1222), which this file does not carry.  Its other effect was to
// pull in `maca_bfloat16.h`; `radix_core.cuh` includes that itself, which is
// where the 16-bit arms still get it.  Dropped so the reference needs nothing
// from `csrc/`.
// The ported C500 dataflow (see the provenance banner in that file).  It is
// included here so that this translation unit -- the one `setup.py` builds --
// is what proves it compiles under the mxcc/cu-bridge toolchain, and so that
// the bf16 selection path below is a header-only dependency.
#include "xcore1000_radix_core.cuh"

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
// The work target is `f32_chunk_work_target(sm_count)` above -- `sm_count * 5
// / 2`, the 2.5 being the only fit ever measured (on C500).  This comment sent
// readers to `deep_select/_arch.py`'s `F32_CHUNK_WORK_TARGET` until
// 2026-09-16; no such symbol exists there or anywhere, and the table it named
// was dissolved when the per-architecture builds went (the per-family SM
// counts it multiplied are what `_arch.py` still carries, in `SM_COUNT`).
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

// **The chunk count's own rule cannot see `topk`, and that is the axis it is
// wrong on above 512.**  Stage 2's merge reads `num_chunks * topk` candidates
// per row -- `candidate_stride` in `rk::launch_topk_f32_chunks_stage2` -- and
// `f32_chunks_for` sizes itself from the batch alone.  Measured over ~20 cells
// (`V` in {65536, 66551, 107520, 129280, 131072, 262144}, `batches <= 16`,
// `kk.bench`, contract checked; the sweeps are in ledger §8.6): the batch rule
// is already at the optimum for `topk <= 512`, and capping at 8 above that is
// worth **+4.8% .. +40.0%**.  Small-batch tier only -- above
// `kF32ChunksFewBatches` the count is the constant 2 and never reaches this.
//
// **8, not 4, and that is the one place the sweep overturned the obvious
// answer.**  4 wins at `V = 129280` (112.2 vs 118.8 us at k=1024) but loses at
// `V = 65536` (75.3 vs 70.7 us) and at every `V = 66551` cell (90.5 vs 88.2 us
// at b1); at `V = 262144` it is worse than the *batch rule's own* 16 by 39%
// (138.4 vs 99.5 us at b6-k1024).  8 has the best worst cell.
inline int f32_topk_chunk_cap(uint32_t topk, uint32_t vocab, int c_default) {
    if (topk <= 512) return 0;   // 0 = no cap; keeps `DEEP_SELECT_F32_CHUNK32` in force
    // Only where the merge is a real share of the row: `c * topk` candidates
    // against `vocab` elements.  The three cells below a tenth are exactly the
    // three where the batch rule's own count is the optimum (`V = 262144`,
    // k = 1024 -- 6.25%, at b1/b6/b16), so capping them is a pure loss
    // (-9.3% / -2.4% / -4.2%).  `10` sits between that 6.25% and the 12.5% of
    // the nearest win; the threshold is fitted, the bracket on both sides of
    // it is measured.
    if ((uint64_t)c_default * topk * 10 <= vocab) return 0;
    return 8;
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
//     overflow when  L / 58 > kF32SmemInputSize  =>  L > 1757 * 58 = 101906
//
// Both factors are device geometry (`kF32SmemInputSize` from `kSMEM`, `kRadix`)
// except the 58.  The count is then the smallest `c` that clears it:
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
// **C500 only, and the discriminator is `sm_count`, not an arch macro.**  Both
// inputs to the rule are device geometry -- the arena and the AP count the grid
// is sized against -- and C600/C600U disagree on both (28/32 APs against 104,
// and a different smem budget).  `__MACA_ARCH__` looks like the obvious gate and
// is the wrong one: it is defined only in the *device* pass, and this is a host
// function, so every architecture would take the `#else` branch.  `sm_count` is
// already the file's own answer to "which part is this" (`f32_chunk_work_target`
// above, and the grid sizing in `chunked_chunks`), so it is used here too.
inline constexpr uint32_t kF32OverflowChunkLen = 1757u * 58u;   // 101906
inline constexpr uint32_t kC500SmCount = 104;
inline int f32_chunks_large_batch(uint32_t vocab_size, uint32_t sm_count) {
    if (sm_count != kC500SmCount) return 2;   // C600/C600U untouched; see above
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
    const int cap = f32_topk_chunk_cap(topk, vocab_size, c);
    return (cap != 0 && c > cap) ? cap : c;
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
    const size_t table_bytes = (size_t)batches * sizeof(int32_t);
    if (params.check_nan) {
        cudaMemsetAsync(ws.lengths, 0, table_bytes, stream);
        nan_scan_kernel<float><<<batches * chunks, kScanBlock, 0, stream>>>(
            params.input, params.end_ptr,
            (int64_t)(params.stride_input_batch / sizeof(float)),
            params.vocab_size, (uint32_t)chunks, ws.lengths);
    }
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
// ── the reference's host entry ──────────────────────────────────────────────
//
// `ds_topk` (declared in `xcore1000_ds_topk.h`).  What it does that upstream's
// `deep_select::topk()` did:
//
//   * fills `RowParams` from the arguments.  Fixed at the upstream defaults
//     this reference does not expose: fp32 input, int32 indices, no values
//     out, no index offset (so the `length <= topk` shortcut skips
//     `emit_ordered` and writes plain ascending indices), `idx_fill = -1`,
//     `check_nan = true`, `abort_on_nan = false`.
//   * evaluates the split's gates in the same order upstream does, so the
//     workspace below is sized for the arm that will actually run.  The
//     equality that matters is the one upstream enforces with `DS_HOST_CHECK`:
//     `topk_launch` re-derives `f32_chunked_chunks(...)` from `params` and
//     refuses the split unless `chunked_bytes` is at least what that count
//     needs.  Both sides call the same functions here, so they agree.
//   * builds the per-row length table when the caller passes none -- upstream
//     internals, not a caller-visible detail: `end` absent already means "the
//     whole row", and the split requires an `end_ptr`.
//   * allocates the workspace.  Upstream holds a grow-only one across calls
//     with a per-buffer `cudaMalloc` the first time a shape needs it
//     (`ChunkedScratch`), and that is what is reproduced below, lock included
//     -- a per-call `cudaMalloc`/`cudaFree` would be outside the dataflow and
//     would put the driver's own allocator in the timed region.
//
// The NaN-flag table is the first `batches * sizeof(int32_t)` bytes of the
// workspace (`ChunkedF32Workspace::lengths`): `nan_scan_kernel` accumulates
// into it and `topk_kernel_radix` reads it in its `kPreSelected` arm.  Upstream
// memsets it only under `params.check_nan`, which is true here, so the cast is
// live and not a reinterpretation of the length table.

// ── error reporting ─────────────────────────────────────────────────────────
//
// The two macros the body below uses live upstream in `../ffi/ffi_error.h`
// (`DS_HOST_CHECK`, `DS_CUDA_RUNTIME_CHECK`), where they raise
// `tvm::ffi::Error` so the runtime can surface the message as a python
// exception.  There is no FFI here, so they are re-spelled with the same names
// and the same fail-loudly intent: message to stderr, then `std::abort()`.  A
// `void` C entry has no error channel to return through, and a reference that
// ignored a failed `cudaMalloc` would print a GB/s number for a run that never
// happened.
#define DS_REF_RAISE(...)                                                      \
    do {                                                                       \
        std::fprintf(stderr, "[ds_topk] %s:%d: ", __FILE_NAME__, __LINE__);    \
        std::fprintf(stderr, __VA_ARGS__);                                     \
        std::fprintf(stderr, "\n");                                            \
        std::abort();                                                          \
    } while (0)

#define DS_HOST_CHECK(cond, ...)                                               \
    do {                                                                       \
        if (!(cond)) DS_REF_RAISE(__VA_ARGS__);                                \
    } while (0)

#define DS_CUDA_RUNTIME_CHECK(cmd)                                             \
    do {                                                                       \
        const cudaError_t _ds_err = (cmd);                                     \
        if (_ds_err != cudaSuccess)                                            \
            DS_REF_RAISE("%s failed: %s", #cmd, cudaGetErrorString(_ds_err));  \
    } while (0)

namespace {

// Upstream's `ChunkedScratch` (maca_topk.cu, the `deep_select` namespace), minus
// the fields only the tvm-ffi layer touched.  Both buffers are grow-only and
// shared by every call in the process, so the kernel arguments must not be
// resized between the launches of one call -- which is why the lock is held
// across them rather than around the allocation alone.
struct ChunkedScratch {
    static constexpr int64_t kNoEpoch = -1;

    int32_t *lengths = nullptr;
    size_t lengths_count = 0;
    // `[vocab_size] * batches` is a pure function of two arguments of the call
    // that fills this table, so "is it already right" is a question about that
    // pair and nothing else. `kNoEpoch` means "must refill".
    int64_t lengths_epoch_batches = kNoEpoch;
    int64_t lengths_epoch_vocab = kNoEpoch;

    void *workspace = nullptr;
    size_t workspace_bytes = 0;

    // The cache is process-wide while the kernels have no thread-safety
    // contract of their own, so it carries its own lock rather than assuming
    // the caller serializes.
    std::mutex mu;
};

ChunkedScratch &chunked_scratch() {
    static ChunkedScratch scratch;
    return scratch;
}

struct ScratchGuard {
    std::mutex *m;
    explicit ScratchGuard(std::mutex *mu) : m(mu) { if (m) m->lock(); }
    ~ScratchGuard() { if (m) m->unlock(); }
};

}  // namespace

extern "C" void ds_topk(const float* scores, const int32_t* lengths, int32_t* out,
                        int n_rows, int n_cols, int top_k, int sm_count) {
    const int value_dtype = 0;   // float32  (bf16 would be 1)
    const int index_dtype = 0;   // int32    (int64 would be 1)
    const bool sorted_index = false;
    const bool sorted_value = false;
    const bool return_value = false;
    const int64_t batches = (int64_t)n_rows;

    RowParams p{};
    p.input = scores;
    p.output_value = nullptr;
    p.output_index = out;
    p.end_ptr = nullptr;          // filled in below when the split runs
    p.idx_offset_ptr = nullptr;
    p.preselected = nullptr;
    p.nan_flags = nullptr;
    p.stride_input_batch = (uint64_t)n_cols * sizeof(float);
    p.stride_output_value_batch = 0;
    p.stride_output_index_batch = (uint64_t)top_k * sizeof(int32_t);
    p.vocab_size = (uint32_t)n_cols;
    p.topk = (uint32_t)top_k;
    p.sm_count = (uint32_t)sm_count;
    p.idx_fill = -1;
    p.value_fill = 0.0f;
    p.check_nan = true;
    p.abort_on_nan = false;

    if (n_rows <= 0 || n_cols <= 0 || top_k <= 0 ||
        top_k > kMaxTopK || sm_count <= 0) {
        DS_HOST_CHECK(n_rows > 0 && n_cols > 0,
                      "bad shape: n_rows=%d n_cols=%d", n_rows, n_cols);
        DS_HOST_CHECK(top_k > 0 && top_k <= kMaxTopK,
                      "topk must be in (0, %d], got %d", kMaxTopK, top_k);
        DS_HOST_CHECK(sm_count > 0, "sm_count must be > 0 (got %d)", sm_count);
    }

    const bool f32_split = detail::chunked_f32_applies(p, (uint32_t)batches);
    const int f32_chunks = detail::f32_chunked_chunks(
        (uint32_t)batches, p.topk, p.vocab_size, p.sm_count);
    const size_t need_lengths = (size_t)batches * sizeof(int32_t);
    const size_t need_workspace =
        f32_split ? detail::chunked_f32_workspace_bytes((uint32_t)batches,
                                                        (uint32_t)top_k,
                                                        f32_chunks)
                  : 0;

    // The cache is the upstream default path; `DEEP_SELECT_NO_SCRATCH_CACHE=1`
    // turns it off, which is the knob upstream carries for the same reason.
    static const bool cache_scratch = [] {
        const char *v = std::getenv("DEEP_SELECT_NO_SCRATCH_CACHE");
        return !(v != nullptr && v[0] == '1' && v[1] == '\0');
    }();
    ChunkedScratch &scratch = chunked_scratch();
    ScratchGuard scratch_guard(cache_scratch ? &scratch.mu : nullptr);

    int32_t *row_lengths = nullptr;
    void *workspace = nullptr;
    bool borrowed = false;

    if (f32_split) {
        if (cache_scratch) {
            // Grow-only.  A free here would be correct and would give back
            // exactly the cost this buffer exists to avoid, so the peak is
            // held.  Also clear the epoch on a resize: then "no epoch" means
            // "must refill" instead of a fabricated `(0, 0)`.
            if (need_workspace > scratch.workspace_bytes) {
                void *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_workspace));
                if (scratch.workspace)
                    DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.workspace));
                scratch.workspace = grown;
                scratch.workspace_bytes = need_workspace;
            }
            if (need_lengths > scratch.lengths_count * sizeof(int32_t)) {
                int32_t *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_lengths));
                if (scratch.lengths)
                    DS_CUDA_RUNTIME_CHECK(cudaFree(scratch.lengths));
                scratch.lengths = grown;
                scratch.lengths_count = (size_t)batches;
                scratch.lengths_epoch_batches = ChunkedScratch::kNoEpoch;
                scratch.lengths_epoch_vocab = ChunkedScratch::kNoEpoch;
            }
            if (lengths == nullptr) {
                if (scratch.lengths_epoch_batches != batches ||
                    scratch.lengths_epoch_vocab != (int64_t)n_cols) {
                    std::vector<int32_t> host_lengths((size_t)batches,
                                                      (int32_t)n_cols);
                    DS_CUDA_RUNTIME_CHECK(cudaMemcpy(scratch.lengths,
                                                     host_lengths.data(),
                                                     need_lengths,
                                                     cudaMemcpyHostToDevice));
                    scratch.lengths_epoch_batches = batches;
                    scratch.lengths_epoch_vocab = (int64_t)n_cols;
                }
                row_lengths = scratch.lengths;
            } else {
                row_lengths = const_cast<int32_t *>(lengths);
            }
            workspace = scratch.workspace;
            borrowed = true;
        } else {
            if (lengths == nullptr) {
                // What upstream builds for an `end`-less caller:
                // `[vocab_size] * batches`.
                int32_t *own = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&own, need_lengths));
                std::vector<int32_t> host_lengths((size_t)batches, (int32_t)n_cols);
                DS_CUDA_RUNTIME_CHECK(cudaMemcpy(own, host_lengths.data(),
                                                 need_lengths,
                                                 cudaMemcpyHostToDevice));
                row_lengths = own;
            } else {
                row_lengths = const_cast<int32_t *>(lengths);
            }
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, need_workspace));
        }
        p.end_ptr = row_lengths;
    }

    topk_launch(p, batches, /*stream=*/nullptr, value_dtype, index_dtype,
                sorted_index, sorted_value, return_value, workspace,
                need_workspace);
    DS_CUDA_RUNTIME_CHECK(cudaGetLastError());

    // The cached branch hands out the scratch's own pointers, so it owns only
    // what this call allocated itself -- which is nothing once the cache is on.
    if (!borrowed) {
        if (workspace) DS_CUDA_RUNTIME_CHECK(cudaFree(workspace));
        if (lengths == nullptr && row_lengths)
            DS_CUDA_RUNTIME_CHECK(cudaFree(row_lengths));
    }
}

}  // namespace deep_select_maca
