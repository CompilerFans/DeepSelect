#pragma once

#include <cstdint>

// [MACA] 原来这两个球内扫描是内联 PTX（shfl.sync.up/down.b32 + 谓词加）。
//   MACA 汇编器不认 PTX，改用等价的 __shfl_*_sync 内建（已探针实测可用）：
//     * shfl.sync.up.b32 ... 0, 0xffffffff —— 段界 0、全掩码，源 lane 越界时
//       谓词为假、目标值不变 ⇒ 等价于 `if (lane_idx >= i) x += t`
//     * shfl.sync.down.b32 ... 31, 0xffffffff —— 段界 31，越界谓词为假 ⇒
//       等价于 `if (lane_idx + i < 32) x += t`
//   上游全文按 32 线程/warp 假设（lane_idx = threadIdx.x % 32、ballot 用 32 位掩码），
//   这里的掩码同样取 0xFFFFFFFF。**前提是 MACA 的 warp 宽度确为 32** ——
//   若硬件实际是 64 路波前，球内扫描的语义会变，需按 64 重写。本机实测：
//   C500 的波前是 64 lane，但 ballot/reduce/shfl 按 32 lane 分组，所以按 32
//   写的扫描语义成立。
template<typename T>
__device__ __forceinline__ T warp_level_inclusive_prefix_sum(T x, uint32_t lane_idx) {
    static_assert(sizeof(T) == 4);
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFFu, (uint32_t)x, i);
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
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        uint32_t t = __shfl_down_sync(0xFFFFFFFFu, (uint32_t)x, i);
        if (lane_idx + i < 32) x += (T)t;
    }
    return x;
}


template<typename T>
__device__ __forceinline__ T warp_level_exclusive_suffix_sum(T x, uint32_t lane_idx) {
    T inclusive_suffix_sum = warp_level_inclusive_suffix_sum(x, lane_idx);
    return inclusive_suffix_sum - x;
}
