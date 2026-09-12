#pragma once

#include <cstdint>

// ── the wave ────────────────────────────────────────────────────────────────
//
// MACA's wave is 64 lanes.  This tree was ported from CUDA assuming 32 and ran
// wrong *silently* as a result, so the width lives in one place and every site
// that depends on it says so.
//
// History, because the old comment here was the origin of the bug: it asserted
// that "C500's wave is 64 lanes, but ballot/reduce/shfl group by 32, so a scan
// written for 32 has correct semantics".  Measured (`skills/maca-wave64-port/
// scripts/wave64_probe.sh`), the opposite holds: every mask-based collective
// honors its mask, and `0xFFFFFFFF` names physical lanes 0..31 of the wave and
// nothing else.  A "logical group of 32" has no encoding on this hardware at
// all -- see `skills/maca-wave64-port/SKILL.md` §3.
#define MACA_WARP_SIZE 64u
// Spell the type: MACA ships `__reduce_*_sync(uint64_t, ...)` AND
// `__reduce_*_sync(unsigned, ...)`, so an unsuffixed literal is ambiguous and
// fails to compile.  `unsigned long long` is a distinct type from `uint64_t`
// (`unsigned long`) on this platform, hence the cast.
#define MACA_FULL_MASK ((uint64_t)0xFFFFFFFFFFFFFFFFull)

// Every site is written against these so the CUDA-era 32 is nowhere left.
// `static_assert` in the kernels pins the launch config against them.
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
