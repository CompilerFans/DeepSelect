// MACA (xcore1000 / xcore1500 / xcore1600) 适配层：把上游 CUDA 内核依赖的两个
// Hopper 专有原语换成 MACA 等价实现。
//
// 被替换的两样东西，以及为什么必须替换而不是「靠 cu-bridge 兜底」：
//
//   1. TMA (cute::SM90_TMA_LOAD_3D::copy)。cu-bridge 提供同名符号，但它的
//      `cp.async.bulk.tensor` PTX 被 `#if defined(CUTE_ARCH_TMA_SM90_ENABLED)`
//      包住，而该宏在 MACA 上不定义 —— 于是它退化成 CUTE_RUNTIME_ASSERT，
//      也就是「编译通过、运行时搬运完全不发生」。靠它兜底会得到能跑但结果全错的
//      内核，比编译失败危险。这里改用 MACA 的 global->shared 异步搬运内建：
//      __builtin_mxc_ldg_b128_bsm + __builtin_mxc_barrier_and_wait4。
//
//   2. 事务屏障 (cutlass::arch::ClusterTransactionBarrier)。mctlass 里没有
//      cutlass/arch/barrier.h，这个类型真不存在。这里用 shared 上的
//      ticket 计数器实现同语义的相位屏障。
//
// 两个等价性论证：
//
//   * 数据布局：TMA 128B swizzle 的硬件效果就是「把 16 字节块的索引 c 映射到
//     c ^ ((c >> 3) & 7)」—— 正是上游已有的 sw_b128()。消费者的读取侧用
//     sw_elem()（作用在元素索引上）表示同一个置换，两者是同一 swizzle 的两种
//     视角。因此只要在写入 shared 时按 sw_b128 计算目标块号，消费者代码可以
//     完全不动。
//
//   * 屏障相位：mbarrier 的 wait(phase) 语义是「等到相位位翻转成 != phase」。
//     用单调递增的 ticket 计数器实现时，每 expected 个 ticket 让 seq 加一，
//     seq 的奇偶性即相位位，两者逐次对应，无需复位计数器（复位会与下一轮的
//     arrive 竞态）。
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

// 收敛 GCC 向量类型：MACA 内建返回的是 __attribute__((vector_size(N*4))) 的
// 向量类型，不是 uint2/uint4。写错类型会得到 "no viable conversion" 并看起来
// 像「内建不存在」。
typedef unsigned int maca_v1u32 __attribute__((vector_size(4)));
typedef unsigned int maca_v2u32 __attribute__((vector_size(8)));
typedef unsigned int maca_v4u32 __attribute__((vector_size(16)));

namespace maca_compat {

// ---------------------------------------------------------------------------
// 事务屏障：cutlass::arch::ClusterTransactionBarrier 的 MACA 等价物
// ---------------------------------------------------------------------------
// 上游用到的接口只有三个：init / arrive_and_expect_tx / wait。
// 字段：expected 在本对象生命周期内不变；ticket 单调递增且永不复位；seq 每凑齐
// expected 个 ticket 加一，其奇偶性即 mbarrier 的相位位。
struct transac_bar_t {
    uint32_t expected;
    uint32_t ticket;
    uint32_t seq;

    // 对应 mbarrier.init：设定每轮的到达次数。必须在任何 arrive 之前、
    // 且由单线程调用后经 __syncthreads() 广播。
    __device__ __forceinline__ void init(uint32_t num_arrivals) {
        expected = num_arrivals;
        ticket = 0u;
        seq = 0u;
    }

    // 对应 mbarrier.arrive.expect_tx。num_tx_bytes 在 MACA 路径上不参与判定：
    // 上游用它让硬件知道「还要等多少字节到齐」，而这里的数据落地由调用方在
    // arrive 之前用 __builtin_mxc_barrier_and_wait4(scope=0) 显式等到，因此
    // 「到达」与「数据就绪」在调用点上已经合一，字节数只保留语义不再驱动同步。
    __device__ __forceinline__ void arrive_and_expect_tx(uint32_t num_tx_bytes) {
        (void)num_tx_bytes;
        // 让本线程此前写入的 shared 数据（数据由异步拷贝落地，但完成序由本线程
        // 观察）对同 block 的其他线程可见。ticket 的原子序不能替代这一条：
        // atomicAdd 只保证计数本身有序，不保证数据。
        __threadfence_block();
        uint32_t my_ticket = atomicAdd(&ticket, 1u);
        if ((my_ticket + 1u) % expected == 0u) {
            __threadfence_block();
            atomicAdd(&seq, 1u);
        }
    }

    // 对应 mbarrier.try_wait.parity(phase)：等到相位位 != phase。
    // 上游的用法是 buffer 每回绕一次 phase ^= 1，与本实现逐次对应。
    //
    // 自旋之后的 __threadfence_block() 不是可有可无的：arrive 侧的栅栏只把
    // 「写者的数据」排到「写者的到达」之前（release），观察者看到 seq 之后
    // 还必须有一次获取栅栏，才能保证自己读到的是那些数据而不是陈旧值。
    // 缺了它，相位推进会正确、数据可见性却会错——这正是最初那版实现的表现：
    // 屏障按序放行，但读到的 shared 内容不对。mbarrier 在这点上是
    // acquire/release 成对的，本实现必须同样成对。
    __device__ __forceinline__ void wait(uint32_t phase) const {
        while ((__ldcg(&seq) & 1u) == phase) {
            __nanosleep(32);
        }
        __threadfence_block();
    }
};

// ---------------------------------------------------------------------------
// global -> shared 异步拷贝：SM90_TMA_LOAD_3D::copy 的 MACA 等价物
// ---------------------------------------------------------------------------

// 发起一个 16 字节块的异步拷贝并返回完成标志。
// 调用方负责在数据可见之前，对每个返回的标志调用 wait_flag()。
__device__ __forceinline__ maca_v4u32 issue_b128(
        void *smem_dst, const void *gmem_src) {
    // mask 只能是 -1（当前无意义）；pred_neg=false 表示使用活跃线程掩码；
    // is_async=false 表示由编译器插入完成序，这里我们复用返回值显式等待。
    return __builtin_mxc_ldg_b128_bsm(smem_dst, const_cast<void *>(gmem_src),
                                      0, (size_t)-1, false, false, false, false);
}

// scope=0：仅内存栅栏语义，不含跨 warp 指令屏障。
// 上游的发起代码位于 `if (warp_idx >= NUM_ISSUE_WARPS) return;` 之后，只有部分
// warp 参与，因此这里不能使用 scope=1（那等同 __syncthreads()，会造成死锁）。
__device__ __forceinline__ void wait_flag(maca_v4u32 flag) {
    __builtin_mxc_barrier_and_wait4(0, flag);
}

// 一个段 = NUM_ELEMS_PER_SEG 个元素 = NUM_ELEMS_PER_128b*8 个 16B 块。
// 把这一段从 gmem 拷到 smem_seg，目标块号按 128B swizzle 重排，使得消费者的
// sw_elem() 视图无需改动。
//
//   tid_in_warp: 本 warp 内负责本段的线程序号（0..31）
//   warp_size:   参与本段拷贝的线程数
// 返回本线程发起的所有标志之一；多块时逐块等待（见下方 wait_all 用法说明）。
template<uint32_t NUM_ELEMS_PER_SEG, typename ValueT, uint32_t NUM_CHUNKS_PER_SEG>
__device__ __forceinline__ void copy_seg_swizzled(
        ValueT *smem_seg, const ValueT *gmem_seg, uint32_t tid, uint32_t nthreads) {
    static_assert(NUM_ELEMS_PER_SEG % (16 / sizeof(ValueT)) == 0);
    // 与上游 SWIZZLE 常量一致：chunk 索引的三位与「行」的三位异或
    constexpr uint32_t CHUNKS_PER_ROW = 128 / 16;  // 128B swizzle 单元 = 8 个 16B 块
    static_assert(NUM_CHUNKS_PER_SEG % CHUNKS_PER_ROW == 0);

    for (uint32_t c = tid; c < NUM_CHUNKS_PER_SEG; c += nthreads) {
        // 128B swizzle：块号 c ^ ((c >> 3) & 7)（等价于上游的 sw_b128）
        uint32_t dst_chunk = c ^ ((c >> 3) & (CHUNKS_PER_ROW - 1u));
        constexpr uint32_t CHUNK_ELEMS = 16 / sizeof(ValueT);
        maca_v4u32 flag = issue_b128(smem_seg + dst_chunk * CHUNK_ELEMS,
                                     gmem_seg + c * CHUNK_ELEMS);
        wait_flag(flag);
    }
}

// ---------------------------------------------------------------------------
// 内联 PTX 的等价实现
// ---------------------------------------------------------------------------
// MACA 汇编器只认 MACA ISA，任何内联 PTX 都无法编译；以下是有对应内建的几条。
// （bf16x2 的比较/算术用 cu-bridge 提供的 __h* 系列，见各调用点原位注释。）

// prmt.b32：按字节选择器从 {s0,s1} 中取 4 个字节拼成结果
__device__ __forceinline__ uint32_t prmt_b32(uint32_t s0, uint32_t s1, uint32_t sel) {
    return __builtin_mxc_byte_perm(s0, s1, sel);
}

// bfe.u32 -> ubfe；bfe.s32 -> sbfe（上游的 bfe.u32 用法见各调用点）
__device__ __forceinline__ uint32_t bfe_u32(uint32_t src, uint32_t pos, uint32_t len) {
    return __builtin_mxc_ubfe(src, pos, len);
}

// mad.lo.u32 的低 32 位乘加，直接写表达式即可（MACA 有原生整数乘加）
__device__ __forceinline__ uint32_t mad_lo_u32(uint32_t a, uint32_t b, uint32_t c) {
    return a * b + c;
}

}  // namespace maca_compat
