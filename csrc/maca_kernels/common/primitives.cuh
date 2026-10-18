// Device primitives shared by this operator's dataflows, one definition each.
//
// `common/` carries three of them -- `topk_select_radix`'s radix rows, `topk_select_coarse12`'s
// selector and `topk_select_chunks`' chunked merge -- and each was ported or written
// against its own copy of the same bit tricks, so the bodies could drift with
// nothing to catch it.  What is here is what every one of them spelled the
// same way: the NaN bit-pattern test, the order-preserving key, and the warp
// histogram suffix sum.  Everything is integer-only and architecture-
// independent; nothing here reads a family macro, so a family's artifact
// compiles the same code from this file.
//
// **The names are the source code's own, and they are the ones to keep.**
// `refine_bin` and `warp_cumsum_histogram` are deep_gemm's (`fp32_topk`, the
// file the `topk_select_coarse12` / `topk_select_chunks` dataflows were ported from, whose call
// sites spell both the same way); the NaN tests are **this port's** bit-pattern
// spelling, in upstream's `topk_select_common` namespace -- upstream has no bit
// test (it detects NaN with PTX `set.nan.bf16`), and the ported tree carries
// the twin in `legacy/bit_utils.cuh`.  The key transform is upstream's
// `distort` under deep_gemm's name: same operation, and the two lineages spell
// it differently, so renaming either side diverges silently from the code the
// dataflows are re-merged with.
//
// Siblings that look like these but are not the same operation stay where they
// are: `topk_select_radix::float_to_uint8` / `bf16_to_uint8` (the radix row's 8-bit coarse
// level, fp16-rounded for fp32) and `topk_select_radix::run_cumsum_warp` (a 256-thread
// two-level scan with a different barrier contract).  deep_gemm's template
// carries a second arm -- the `kCoarseBins` walk that needs a thread per bin --
// which `topk_chunks.cuh`'s `block_cumsum_histogram_1024` already implements and
// has no second caller for.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>
#include <maca_bfloat16.h>

namespace topk_select_common {

// ── NaN, raw bit pattern ────────────────────────────────────────────────────
//
// Exponent all-ones with a non-zero payload: every NaN encoding, either sign,
// quiet or signaling.  NOT a comparison against the key -- the encode below
// sends the two signed NaNs to opposite ends of the key space, so key
// comparison catches only `0x7FFFFFFF` -- and not `v != v`, which the build's
// `--use_fast_math` rules out.
__device__ __forceinline__ bool is_nan_bits32(uint32_t bits) {
    return (bits & 0x7F800000u) == 0x7F800000u && (bits & 0x007FFFFFu) != 0u;
}

// The 16-bit sibling: the same test in 8+7 bits, on the value's own bit
// pattern (so it serves bf16; fp16 would key differently -- sign bit aside,
// only the exponent width differs, and both are all-ones for NaN).
__device__ __forceinline__ bool bf16_is_nan(uint16_t bits) {
    return (bits & 0x7F80u) == 0x7F80u && (bits & 0x007Fu) != 0u;
}

// The value-typed form, for callers that hold the value rather than its bits
// (`row_has_nan`'s vector and scalar walks).  Two specializations, because the
// caller is generic over the row's dtype.
template <typename ValueT>
__device__ __forceinline__ bool is_nan_value(ValueT v);

template <>
__device__ __forceinline__ bool is_nan_value<float>(float v) {
    return is_nan_bits32(__float_as_uint(v));
}

template <>
__device__ __forceinline__ bool is_nan_value<maca_bfloat16>(maca_bfloat16 v) {
    return bf16_is_nan(__bfloat16_as_ushort(v));
}

// ── the order-preserving key ────────────────────────────────────────────────
//
// Sign-magnitude to ascending unsigned: every bit of a negative input flips,
// only the sign bit of a positive one does.
__device__ __forceinline__ uint32_t refine_bin(float value) {
    const uint32_t bits = __float_as_uint(value);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

// The 16-bit sibling, overloaded on the value type.  On MACA the sign branch
// is an xor rather than a select, so it cannot diverge: positive keys xor
// 0x8000, negative keys xor 0xffff.
__device__ __forceinline__ uint16_t refine_bin(maca_bfloat16 value) {
    const uint16_t bits = __bfloat16_as_ushort(value);
#ifdef __MACACC__
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint16_t mask = static_cast<uint16_t>(0x8000u | (0u - sign));
    return static_cast<uint16_t>(bits ^ mask);
#else
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits)
                           : static_cast<uint16_t>(bits | 0x8000);
#endif
}

// ── the warp histogram suffix sum ───────────────────────────────────────────
//
// Each of the 64 lanes owns `kBins / 64` consecutive bins, suffix-sums them in
// registers, then one shuffle pass gives each lane its offset; `histogram[kBins]`
// is the padding slot the shape carries.  Reads and writes `histogram[0..kBins-1]`
// and ends with a `__syncthreads()`, so the caller's next read of the histogram
// is ordered after it.
//
// This is deep_gemm's one-warp arm.  Not to be confused with
// `topk_select_radix::run_cumsum_warp`: that one spreads 256 bins over the block's warps,
// adds a second scan over the warp totals and writes them into the same
// padding region; this one is a single warp and one barrier.
template <uint32_t kBins>
__device__ __forceinline__ void warp_cumsum_histogram(int (&histogram)[kBins + 1]) {
    constexpr int kWarpSize = 64;
    constexpr uint64_t kWarpMask = 0xffffffffffffffffULL;
    constexpr int kItemsPerLane = static_cast<int>(kBins) / kWarpSize;
    static_assert(kBins % 64u == 0 && kBins <= 64u * 64u,
                  "one warp owns kBins/64 consecutive bins: kBins must be a "
                  "multiple of 64 and at most 4096");

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

}  // namespace topk_select_common
