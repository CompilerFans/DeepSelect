#pragma once

#include <cstdint>
#include <cstring>
// [MACA] 原为 cu-bridge 的 <cuda_bf16.h>（它只是 `typedef maca_bfloat16 __nv_bfloat16;`
//   的兼容层）。直接用 MACA 原生头，数据类型不经过兼容层。
#include <maca_bfloat16.h>

namespace topk_select_common {

__device__ __forceinline__
uint32_t bf16x2_to_u32(const maca_bfloat162 &v) {
    static_assert(sizeof(maca_bfloat162) == sizeof(uint32_t));
    return *reinterpret_cast<const uint32_t *>(&v);
}

__device__ __forceinline__
maca_bfloat162 u32_to_bf16x2(uint32_t u) {
    static_assert(sizeof(maca_bfloat162) == sizeof(uint32_t));
    maca_bfloat162 v;
    *reinterpret_cast<uint32_t *>(&v) = u;
    return v;
}

// Distort and un-distort: Map IEEE 754 floating-point total order to unsigned integer order.
// IEEE 754 bit representation (sign-magnitude with biased exponent and mantissa) does not
// preserve value order when interpreted as unsigned integer, and distort() fixes this:
//   - For positive values (sign bit = 0): flip only the sign bit, mapping all positives to [0x80..., 0xFF...]
//   - For negative values (sign bit = 1): flip all bits (~), inverting the magnitude order so that
//     more negative (smaller value) => smaller uint
// After distortion, all negatives (0x00... ~ 0x7F...) < all positives (0x80... ~ 0xFF...) in
// unsigned integer comparison, which matches the floating-point total order.
//
// un_distort() is the exact inverse: flips the sign bit back for positives, flips all bits back for negatives.
template<typename UIntValueT>
__device__ __forceinline__
UIntValueT distort(const UIntValueT &x) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        UIntValueT mask = (x&0x8000) ? 0xFFFF : 0x8000;
        return x ^ mask;
    } else {
        // mask = (x >> 31) | 0x80000000 is 0x80000000 for positives and 0xFFFFFFFF for negatives
        //
        // [MACA] 原为 `shr.s32 t, x, 31` + `lop3.b32 d, x, t, 0x80000000, 0x1e`。
        //   已核对 0x1e={1,2,3,4} 恰是该 lop3 的真值表（c 是常量 0x80000000，
        //   只有第 31 位为 1，故 index5 是不可达项）：
        //     正数 t=0: index=(a<<2)|(b<<1) → a=0→bit0=0, a=1→bit4=1 ⇒ d=x
        //     负数 t=全1: bits0-30 index=(a<<2)|2 → a=0→bit2=0, a=1→bit6=0 ⇒ d=0
        //   合起来正是 d = x ^ ((x>>31算术) | 0x80000000)，与上面的注释一致。
        uint32_t mask = ((uint32_t)((int32_t)x >> 31)) | 0x80000000u;
        return (UIntValueT)((uint32_t)x ^ mask);
    }
}

// ---------------------------------------------------------------------------
// 逐半字比较：结果为每半字 0xFFFF / 0x0000 的掩码，对应 PTX 的
//   set.gt/.eq.bf16x2 与 set.gtu.s32.bf16x2。
//
// **这些是浮点比较，不是整数比较。** 上游用 `set.*.bf16x2`，其 CmpOp 是
// 浮点语义的（`.gt`/`.eq` 有序：任一操作数为 NaN 时为假；`.gtu` 无序：
// 任一操作数为 NaN 时为真）。把位型当 int16 读**不等价**：
//   * NaN：0x7F81..0x7FFF 作 int16 是最大的正数，于是 "+NaN > 任何正数" 为真，
//     而浮点 `.gt` 为假；-NaN（0xFF81..0xFFFF）作 int16 是 -127..-1，同样失真。
//   * ±0：+0=0x0000 与 -0=0x8000 作 int16 是 0 与 -32768，浮点却相等。
// 故一律经 `bf16_bits_to_float` 用真正的浮点比较，逐位对齐 PTX。
// ---------------------------------------------------------------------------
__device__ __forceinline__
float bf16_bits_to_float(uint16_t bits) {
    // bf16 就是同值 fp32 的高 16 位，左移补齐即可；inf/NaN/denormal 都原样传递。
    return __uint_as_float((uint32_t)bits << 16);
}

__device__ __forceinline__
uint32_t bf16x2_gt_mask_float(uint32_t a, uint32_t b) {   // PTX set.gt.bf16x2
    uint32_t m0 = bf16_bits_to_float((uint16_t)a)        > bf16_bits_to_float((uint16_t)b)        ? 0xFFFFu : 0u;
    uint32_t m1 = bf16_bits_to_float((uint16_t)(a >> 16)) > bf16_bits_to_float((uint16_t)(b >> 16)) ? 0xFFFFu : 0u;
    return m0 | (m1 << 16);
}

__device__ __forceinline__
uint32_t bf16x2_gtu_mask_float(uint32_t a, uint32_t b) {  // PTX set.gtu.bf16x2
    // gtu = "greater than, unordered"：`a > b` 或任一操作数为 NaN 时为真。
    // 这是主循环的命中判据，也是 0f03b68 之后 NaN 检测的唯一入口：NaN 必命中，
    // 因而必落进 incoming buffer，由 census 报出来。
    float a0 = bf16_bits_to_float((uint16_t)a),        b0 = bf16_bits_to_float((uint16_t)b);
    float a1 = bf16_bits_to_float((uint16_t)(a >> 16)), b1 = bf16_bits_to_float((uint16_t)(b >> 16));
    uint32_t m0 = (a0 > b0 || a0 != a0 || b0 != b0) ? 0xFFFFu : 0u;
    uint32_t m1 = (a1 > b1 || a1 != a1 || b1 != b1) ? 0xFFFFu : 0u;
    return m0 | (m1 << 16);
}

// bf16 半字是否为 NaN：指数全 1 且尾数非 0（对应 PTX 的 set.nan.bf16）
__device__ __forceinline__
bool bf16_is_nan(uint16_t h) {
    return (h & 0x7F80u) == 0x7F80u && (h & 0x007Fu) != 0u;
}

__device__ __forceinline__
uint32_t bf16x2_eq_mask_float(uint32_t a, uint32_t b) {   // PTX set.eq.bf16x2
    // 浮点相等：NaN 不等于任何值（包括自身），+0 与 -0 相等。按位型比较这两条都不成立。
    uint32_t m0 = bf16_bits_to_float((uint16_t)a)        == bf16_bits_to_float((uint16_t)b)        ? 0xFFFFu : 0u;
    uint32_t m1 = bf16_bits_to_float((uint16_t)(a >> 16)) == bf16_bits_to_float((uint16_t)(b >> 16)) ? 0xFFFFu : 0u;
    return m0 | (m1 << 16);
}

// 逐半字**按位**相等。指针历史：这是 `bf16x2_gt_mask_float` 的整数表亲，只用于
// `histogram_radix_lsb_for_pivot_msb` 里比较抽出来的高字节（见那里的注释）——
// 那种场合两边都是 0x00hh 形态、不是 bf16 值，浮点比较反而是错的。
__device__ __forceinline__
uint32_t bf16x2_eq_mask(uint32_t a, uint32_t b) {
    uint32_t m0 = (uint16_t)a == (uint16_t)b ? 0xFFFFu : 0u;
    uint32_t m1 = (uint16_t)(a >> 16) == (uint16_t)(b >> 16) ? 0xFFFFu : 0u;
    return m0 | (m1 << 16);
}

// A SIMD-like version of `distort`, which uses more efficient instructions for calculation
//
// [MACA] 原 16 位分支是一段内联 PTX（prmt.b32 造掩码 + or/xor）。MACA 汇编器不认
//   PTX，这里改用逐半字写法。注意：它与标量 `distort<uint16_t>` **必须逐位一致** ——
//   `histogram_radix_msb` 用它算桶，而 `histogram_radix_lsb_for_pivot_msb` /
//   `compute_pivot_and_quota` 是按标量 distort 的正/负号约定去 un-distort 桶号的，
//   两处若不一致，桶序和 pivot 就会错位（静默错误结果）。
//   本函数直接逐半字调用标量 `distort`，把"一致性"变成结构上不可违反的事，
//   代价是 4 条左右额外的指令；后续若要回到 SIMD，请以本函数为逐位参照实现。
template<typename UIntValueT>
__device__ __forceinline__
void distort_x2(UIntValueT result[2], const UIntValueT input[2]) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        uint16_t in[2];
        memcpy(in, input, sizeof(in));
        uint16_t out[2] = { distort<uint16_t>(in[0]), distort<uint16_t>(in[1]) };
        memcpy(result, out, sizeof(out));
    } else {
        result[0] = distort(input[0]);
        result[1] = distort(input[1]);
    }
}

template<typename UIntValueT>
__device__ __forceinline__
UIntValueT un_distort(const UIntValueT &x) {
    static_assert(sizeof(UIntValueT) == 2 || sizeof(UIntValueT) == 4);
    if constexpr (sizeof(UIntValueT) == 2) {
        UIntValueT mask = (x&0x8000) ? 0x8000 : 0xFFFF;
        return x ^ mask;
    } else {
        // mask = ~(x >> 31) | 0x80000000 is 0xFFFFFFFF for positives and 0x80000000 for negatives
        //
        // [MACA] 原为 `shr.s32` + `lop3.b32 ..., 0x4b`。0x4b={0,1,3,6} 已核对为
        //   d = x ^ (~t | 0x80000000) 的真值表（正数→~x，负数→翻符号位），
        //   即上面注释的精确逆。
        uint32_t mask = (~(uint32_t)((int32_t)x >> 31)) | 0x80000000u;
        return (UIntValueT)((uint32_t)x ^ mask);
    }
}

} // namespace topk_select_common
