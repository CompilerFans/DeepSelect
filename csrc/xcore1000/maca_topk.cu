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
// Algorithm (modeled on the in-tree MACA reference
// `mcDeepGEMM/csrc/kernels/fp32_topk.cu`):
//
//   1. Map each value onto an order-preserving unsigned key, so unsigned key
//      order equals value order.  NaN is *not* ranked by this order (the encode
//      sends +NaN to the largest key and -NaN to the smallest), so NaN presence
//      is detected separately, by bit pattern, and takes over the row:
//        fp32: bits & 0x80000000 ? ~bits : bits | 0x80000000
//        bf16: the same trick on 16 bits -- no fp32->half conversion needed.
//   2. Walk the key one byte at a time, most significant first.  Each round
//      histograms the current byte over the elements that still match the
//      confirmed high-byte prefix, suffix-scans the histogram, and takes the
//      bin the k-th remaining element falls in as the pivot.  Elements
//      strictly above the pivot are final: they are appended to `selected`.
//      The pivot bin itself is not carried -- the next round rescans the row
//      under the extended prefix.  On the last round the pivot bin holds
//      bit-identical keys, so any `take` of them complete the answer.
//
//      Each round is two full passes over the row (histogram, collect), so
//      the kernel performs 2*R passes, R = 4 for fp32 and 2 for bf16 -- the
//      same shape as the in-tree reference.  Rescanning is what makes the
//      refine exact: carrying only the ties that fit (`take` of them) would
//      discard precisely the values the next byte has to rank.
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
// and vocabulary is unbounded here (the kernel makes 2 passes over the row per
// key byte regardless of length), so the arm's shapes are served, only without
// the cluster-specific scheduling the original gave them.  Covered by the
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

namespace deep_select_maca {

// ── tunables ────────────────────────────────────────────────────────────────
constexpr int kMaxTopK = 4096;      // public contract: `topk <= 4096`
constexpr int kRadix = 256;         // bins per round (one key byte)
constexpr int kThreads = 256;       // 4 MACA waves of 64 lanes
constexpr int kWarpSize = 64;       // MACA wave width
constexpr int kMaxWaves = kThreads / kWarpSize;
// MACA waves are 64 lanes wide, so a shuffle over the whole wave needs all 64
// mask bits.  The CUDA-era 0xFFFFFFFF names only lanes 0..31; with it, every
// reduction silently returns wrong values for the upper half (measured:
// an inclusive suffix sum over an all-ones 64-lane wave came out as 64 for
// lanes 0..31 instead of 64-i).
constexpr unsigned long long kWarpMask = 0xFFFFFFFFFFFFFFFFull;

struct RowParams {
    const void *input;
    void *output_value;             // null when the caller does not want values
    void *output_index;
    const int32_t *end_ptr;         // null means "whole row"
    const int32_t *idx_offset_ptr;  // null means no offset
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

template <typename ValueT>
static __device__ __forceinline__ ValueT value_of_key(uint32_t key);

template <>
__device__ __forceinline__ float value_of_key<float>(uint32_t key) {
    const uint32_t bits = (key & 0x80000000u) ? ~key : (key & 0x7FFFFFFFu);
    return __uint_as_float(bits);
}

template <>
__device__ __forceinline__ maca_bfloat16 value_of_key<maca_bfloat16>(uint32_t key) {
    const uint32_t key16 = key & 0xFFFFu;
    const uint16_t bits =
        (key16 & 0x8000u) ? (uint16_t)(~key16 & 0xFFFFu) : (uint16_t)(key16 & 0x7FFFu);
    return __ushort_as_bfloat16(bits);
}

// fp32 -> 4 key bytes, bf16 -> 2 key bytes.
template <typename ValueT>
static constexpr int kNumRounds = (int)sizeof(ValueT);

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

// ── block-wide inclusive suffix scan over the 256-bin histogram ─────────────
//
// On return `histogram[b]` = number of counted elements whose current key byte
// is >= b.  One bin per thread, so this is a two-level shuffle scan.
static __device__ __forceinline__ void suffix_scan(uint32_t *histogram,
                                                   uint32_t *wave_totals) {
    const int tid = threadIdx.x;
    const int lane = tid % kWarpSize;
    const int wave = tid / kWarpSize;

    const uint32_t own = histogram[tid];
    uint32_t acc = own;
#pragma unroll
    for (int off = 1; off < kWarpSize; off <<= 1) {
        const uint32_t other =
            (uint32_t)__shfl_down_sync(kWarpMask, acc, off, kWarpSize);
        if (lane + off < kWarpSize) acc += other;
    }
    if (lane == 0) wave_totals[wave] = acc;

    __syncthreads();
    if (tid == 0) {
        uint32_t higher = 0;
        for (int w = kMaxWaves - 1; w >= 0; --w) {
            const uint32_t cur = wave_totals[w];
            wave_totals[w] = higher;
            higher += cur;
        }
    }
    __syncthreads();
    histogram[tid] = own + (acc - own) + wave_totals[wave];
    // Sentinel: the pivot test reads histogram[kRadix] for the top bin.
    if (tid == 0) histogram[kRadix] = 0;
    __syncthreads();
}

// Scratch the kernel allocates in dynamic shared memory.
//
// `selected[0..k)` holds the answer as it is accumulated: each round appends
// the elements strictly above its pivot, then the last round appends `take`
// bit-identical pivot-bin elements.  `k` therefore lands exactly on `topk`.
//
// `sort_buf` backs the ordered emit (sorted_index / sorted_value) only; it is
// dead weight in the other modes, but it is a compile-time member either way,
// so the arena is a flat 16 KB + 32 KB.  `sort_slots` (4096) is `kMaxTopK`
// rounded up to the bitonic power of two.
struct Arena {
    uint32_t selected[kMaxTopK];           // final answer, descending value order
    uint64_t sort_buf[kMaxTopK];
};

// Ascending bitonic sort of `n` (power of two) 64-bit keys over the CTA.
static __device__ __forceinline__ void bitonic_sort_u64(uint64_t *data, int n) {
    for (int k = 2; k <= n; k <<= 1) {
        for (int j = k >> 1; j > 0; j >>= 1) {
            for (int i = threadIdx.x; i < n; i += kThreads) {
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
template <typename ValueT, typename OutIdxT, bool RV, bool SV>
static __device__ __forceinline__ void emit_ordered(
    Arena &arena, const uint32_t *selected, const ValueT *input_row,
    OutIdxT *out_index_row, ValueT *out_value_row, const RowParams &params,
    uint32_t n_out, uint32_t topk, int32_t idx_offset) {
    uint64_t *sort_buf = arena.sort_buf;
    uint32_t n_pad = 1;
    while (n_pad < topk) n_pad <<= 1;
    for (uint32_t i = threadIdx.x; i < n_pad; i += kThreads) {
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
    bitonic_sort_u64(sort_buf, (int)n_pad);
    __syncthreads();

    for (uint32_t i = threadIdx.x; i < topk; i += kThreads) {
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
// RV = also produce output values
// SI = sorted_index: emit indices ascending
// SV = sorted_value: emit values descending (implies RV)
template <typename ValueT, typename OutIdxT, bool SI, bool RV, bool SV>
__global__ __launch_bounds__(kThreads) void topk_kernel(RowParams params) {
    const int tid = threadIdx.x;
    const uint32_t row = blockIdx.x;

    // hist[0] is used for every round; the +1 slot is the zero sentinel the
    // pivot test reads at bin kRadix-1.  (Only one buffer is needed now that
    // each round's histogram is built in its own pass.)
    __shared__ uint32_t histogram[kRadix + 1];
    __shared__ uint32_t wave_totals[kMaxWaves];
    __shared__ uint32_t pivot_bin;
    __shared__ uint32_t counter;
    __shared__ uint32_t ties_seen;
    __shared__ uint32_t num_selected;
    __shared__ int32_t row_offset;
    extern __shared__ uint8_t arena_raw[];
    Arena &arena = *reinterpret_cast<Arena *>(arena_raw);
    uint32_t *selected = arena.selected;

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

    // Shortcut contract: the window is no longer than k, so the whole window is
    // the answer.  Emitting it in index order is what `sorted_index` asks for
    // and what the modes that leave the order unspecified accept; `sorted_value`
    // is the exception and goes through the ordered emit below.  The NaN check
    // is deliberately skipped on this path either way, matching the original
    // operator (`abort_when_nan_found` is documented as ignored when
    // `end <= topk`).
    if (length <= topk) {
        if constexpr (SV) {
            // The window is the answer, but not yet in the requested order:
            // index order is ascending, `sorted_value` wants value descending.
            // Hand the whole window to the ordered emit as the selection.
            for (uint32_t i = tid; i < length; i += kThreads) selected[i] = i;
            __syncthreads();
            emit_ordered<ValueT, OutIdxT, RV, SV>(
                arena, selected, input_row, out_index_row, out_value_row, params,
                length, topk, idx_offset);
        } else {
            // Index order is already what `sorted_index` asks for, and the
            // other modes leave the order unspecified.
            for (uint32_t i = tid; i < topk; i += kThreads) {
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
    bool nan_local = false;
    for (uint32_t i = tid; i < length; i += kThreads) {
        nan_local |= is_nan_value<ValueT>(__ldg(input_row + i));
    }
    if (__syncthreads_or((int)nan_local) != 0) {
        if (params.abort_on_nan) {
            if (tid == 0) printf("[deep_select] NaN detected. Aborting.\n");
            __trap();
        }
        if (tid == 0) out_index_row[0] = (OutIdxT)0x3F3F3F3F;
        return;
    }

    constexpr int kRounds = kNumRounds<ValueT>;
    constexpr int kBitsPerRound = 8;

    if (tid == 0) {
        counter = 0;
        num_selected = 0;
    }
    __syncthreads();

    // ── refine rounds ───────────────────────────────────────────────────────
    // `remaining` is how many of the top-k are still unaccounted for.  Each
    // round narrows by one key byte:
    //   A. histogram the byte under the confirmed byte prefix, suffix-scan it,
    //      and take the bin holding the `remaining`-th element as the pivot;
    //   B. collect everything strictly above the pivot into `selected`, and on
    //      the last byte also the first `take` elements of the pivot bin (they
    //      are bit-identical, so any `take` of them finish the answer).
    //
    // The pivot bin itself is *not* carried between rounds: the next round
    // rescans the row under the extended prefix.  Carrying a truncated subset
    // of the ties would throw away exactly the values the next byte has to rank,
    // and the whole bin does not fit in a fixed buffer (a single top-byte bin
    // holds `length/256`-ish elements, unbounded in `topk`).
    uint32_t remaining = topk;
    uint32_t prefix_mask = 0;   // key bits above the current byte that must match
    uint32_t prefix_value = 0;

#pragma unroll
    for (int round = 0; round < kRounds; ++round) {
        if (remaining == 0) break;
        const int shift = (kRounds - 1 - round) * kBitsPerRound;
        const bool is_last = (round == kRounds - 1);
        const bool filtered = (prefix_mask != 0);

        // ── A. histogram the current byte over the live prefix ──────────────
        for (int b = tid; b < kRadix; b += kThreads) histogram[b] = 0;
        __syncthreads();
        for (uint32_t i = tid; i < length; i += kThreads) {
            const uint32_t key = key_of<ValueT>(__ldg(input_row + i));
            if (filtered && (key & prefix_mask) != prefix_value) continue;
            atomicAdd(&histogram[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        suffix_scan(histogram, wave_totals);

        if (tid < kRadix) {
            // The remaining-th element (from the top) falls in the largest bin
            // b whose inclusive suffix count still covers it.
            if (histogram[tid] >= remaining
                && histogram[tid + 1] < remaining) {
                pivot_bin = (uint32_t)tid;
            }
        }
        __syncthreads();
        const uint32_t pivot = pivot_bin;
        const uint32_t excess = histogram[pivot + 1];
        const uint32_t take = remaining - excess;   // pivot-bin picks still needed

        // ── B. collect ──────────────────────────────────────────────────────
        // `excess` is exact (the histogram is frozen behind the barrier above),
        // so the collected count needs no atomic tally: exactly `excess`
        // elements satisfy `bin > pivot`.
        const uint32_t base = num_selected;
        if (tid == 0) {
            counter = 0;
            ties_seen = 0;
        }
        __syncthreads();
        for (uint32_t i = tid; i < length; i += kThreads) {
            const uint32_t key = key_of<ValueT>(__ldg(input_row + i));
            if (filtered && (key & prefix_mask) != prefix_value) continue;
            const uint32_t bin = (key >> shift) & 0xFFu;
            if (bin > pivot) {
                selected[base + atomicAdd(&counter, 1u)] = i;
            } else if (is_last && bin == pivot) {
                // Ties on the final byte: any `take` of them are equally valid.
                const uint32_t pos = atomicAdd(&ties_seen, 1u);
                if (pos < take) selected[base + excess + pos] = i;
            }
        }
        __syncthreads();
        if (tid == 0) num_selected = base + excess + (is_last ? take : 0u);
        __syncthreads();
        if (is_last) break;   // `remaining` is now 0: the answer is complete
        remaining -= excess;

        prefix_mask |= (uint32_t)0xFFu << shift;
        prefix_value |= pivot << shift;
    }

    // ── emit ────────────────────────────────────────────────────────────────
    const uint32_t n_out = num_selected;

    if constexpr (!SI && !SV) {
        for (uint32_t i = tid; i < topk; i += kThreads) {
            const bool valid = i < n_out;
            const uint32_t src = valid ? selected[i] : 0u;
            out_index_row[i] =
                valid ? (OutIdxT)((int64_t)src + idx_offset) : (OutIdxT)params.idx_fill;
            if (RV) {
                out_value_row[i] =
                    valid ? __ldg(input_row + src) : float_to_value<ValueT>(params.value_fill);
            }
        }
        return;
    }
    if constexpr (SI || SV) {
        emit_ordered<ValueT, OutIdxT, RV, SV>(
            arena, selected, input_row, out_index_row, out_value_row, params,
            n_out, topk, idx_offset);
    }
}

// ── host-side launch ────────────────────────────────────────────────────────
namespace detail {

inline size_t arena_bytes() { return sizeof(Arena); }

template <typename ValueT, typename OutIdxT, bool SI, bool RV, bool SV>
inline cudaError_t configure() {
    return cudaFuncSetAttribute(
        (const void *)topk_kernel<ValueT, OutIdxT, SI, RV, SV>,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)arena_bytes());
}

template <typename ValueT, typename OutIdxT>
void launch_typed(const RowParams &params, uint32_t batches, cudaStream_t stream,
                  bool sorted_index, bool sorted_value, bool return_value) {
    auto run = [&](auto si, auto sv, auto rv) {
        constexpr bool SI = decltype(si)::value;
        constexpr bool SV = decltype(sv)::value;
        constexpr bool RV = decltype(rv)::value;
        const size_t smem = arena_bytes();
        const cudaError_t rc = configure<ValueT, OutIdxT, SI, RV, SV>();
        if (rc != cudaSuccess) {
            std::fprintf(stderr, "[deep_select] smem attribute: %s\n",
                         cudaGetErrorString(rc));
        }
        topk_kernel<ValueT, OutIdxT, SI, RV, SV>
            <<<batches, kThreads, smem, stream>>>(params);
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

}  // namespace detail

void topk_launch(const RowParams &params, int64_t batches, void *stream,
                 int value_dtype, int index_dtype, bool sorted_index,
                 bool sorted_value, bool return_value) {
    auto *cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const uint32_t n = (uint32_t)batches;
    const bool rv = return_value || sorted_value;
    if (value_dtype == 0) {  // float32
        if (index_dtype == 0) {
            detail::launch_typed<float, int32_t>(params, n, cuda_stream,
                                                 sorted_index, sorted_value, rv);
        } else {
            detail::launch_typed<float, int64_t>(params, n, cuda_stream,
                                                 sorted_index, sorted_value, rv);
        }
    } else {  // bfloat16
        if (index_dtype == 0) {
            detail::launch_typed<maca_bfloat16, int32_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv);
        } else {
            detail::launch_typed<maca_bfloat16, int64_t>(
                params, n, cuda_stream, sorted_index, sorted_value, rv);
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

    topk_launch(p, batches, (void *)stream, value_dtype, index_dtype,
                sorted_index, sorted_value, return_value);
    cudaError_t err = cudaGetLastError();
    TORCH_CHECK(err == cudaSuccess, "topk launch failed: ",
                cudaGetErrorString(err));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
    m.def("get_alignment_requirement", &get_alignment_requirement);
}
