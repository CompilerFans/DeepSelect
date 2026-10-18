#pragma once

#include <cstdint>

// ── the wave ────────────────────────────────────────────────────────────────
//
// MACA's wave is 64 lanes, and every mask-based collective honors its mask:
// `0xFFFFFFFF` names physical lanes 0..31 and nothing else.  There is no
// "logical group of 32" on this hardware -- see
// `skills/maca-wave64-port/SKILL.md` §3.
#define MACA_WARP_SIZE 64u
// Spell the type: MACA ships `__reduce_*_sync(uint64_t, ...)` and
// `__reduce_*_sync(unsigned, ...)`, so an unsuffixed literal is ambiguous and
// fails to compile; `uint64_t` is `unsigned long` here, a distinct type.
#define MACA_FULL_MASK ((uint64_t)0xFFFFFFFFFFFFFFFFull)

// The kernels `static_assert` their launch config against these.
template<typename T>
__device__ __forceinline__ T warp_level_inclusive_prefix_sum(T x, uint32_t lane_idx) {
    static_assert(sizeof(T) == 4);
    #pragma unroll
    for (uint32_t i = 1; i <= MACA_WARP_SIZE / 2; i <<= 1) {
        uint32_t t = __shfl_up_sync(MACA_FULL_MASK, (uint32_t)x, i);
        if (lane_idx >= i) x += (T)t;
    }
    return x;
}

template<typename T>
__device__ __forceinline__ T warp_level_exclusive_prefix_sum(T x, uint32_t lane_idx) {
    T inclusive_prefix_sum = warp_level_inclusive_prefix_sum(x, lane_idx);
    return inclusive_prefix_sum - x;
}

template<typename T>
__device__ __forceinline__ T warp_level_inclusive_suffix_sum(T x, uint32_t lane_idx) {
    static_assert(sizeof(T) == 4);
    #pragma unroll
    for (uint32_t i = 1; i <= MACA_WARP_SIZE / 2; i <<= 1) {
        uint32_t t = __shfl_down_sync(MACA_FULL_MASK, (uint32_t)x, i);
        if (lane_idx + i < MACA_WARP_SIZE) x += (T)t;
    }
    return x;
}


template<typename T>
__device__ __forceinline__ T warp_level_exclusive_suffix_sum(T x, uint32_t lane_idx) {
    T inclusive_suffix_sum = warp_level_inclusive_suffix_sum(x, lane_idx);
    return inclusive_suffix_sum - x;
}
