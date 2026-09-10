/*
Common data types and macros that are used across the kerutils library.
*/
#pragma once

// [MACA] 原来是 cu-bridge 的兼容层头（<cuda_bf16.h> / <cuda_fp8.h>）。
// cu-bridge 里的 `typedef maca_bfloat16 __nv_bfloat16;` 说明它本就是把 MACA
// 原生类型换个 CUDA 名字再导出；这里直接取原生头，数据类型因此只依赖 MACA
// 自身（<maca_bfloat16.h> 会连带包含 <maca_fp16.h>），不经过任何兼容层。
//   实测 `-I/opt/maca/include` 即可解析 `<maca_bfloat16.h>`。
// 原头部还有两行 cutlass 依赖，也已不需要：
//   * <cutlass/bfloat16.h> 只为 `using bf16 = cutlass::bfloat16_t;` 引入，
//     而 bf16 这个别名在 kerutils 与 DeepSelect 里都没有任何使用点（死代码）；
//   * arch/barrier.h 在 MACA 上不存在（mctlass/arch 下没有 barrier.h）。
// 依赖 cutlass 的 sm90/sm100 intrinsics 由 KERUTILS_ENABLE_SM*9x* 宏门控，
// 而那些宏以 __CUDA_ARCH__ 为前提 —— MACA 上不成立，故不会编到。
#include <maca_bfloat16.h>

#include <cute/config.hpp>  // For CUTE_DEVICE

namespace kerutils {

// Cache hints
enum class CacheHint {
    EVICT_FIRST,
    EVICT_NORMAL,
    EVICT_LAST,
    EVICT_UNCHANGED,
    NO_ALLOCATE
};

// Prefetch size
enum class PrefetchSize {
    B64,
    B128,
    B256
};

// [MACA] 此处原有 5 个别名：`nvbf16/nvbf16x2`（__nv_ 形态的 bf16）与
//   `nve4m3/nve4m3x2/nve4m3x4`（__nv_ 形态的 fp8）。已全部删除 ——
//   它们在 kerutils 与 DeepSelect 全仓库都没有任何使用点（死代码），
//   保留只会把 cu-bridge 的兼容层重新拖回每个编译单元。
//   需要 bf16 的地方直接用 <maca_bfloat16.h> 的原生名
//   `maca_bfloat16` / `maca_bfloat162`。
//
// [MACA] 此处原有 `using bf16 = cutlass::bfloat16_t;`，已删 —— 同上，死代码。

// [MACA] 原为 `using transac_bar_t = cutlass::arch::ClusterTransactionBarrier;`。
//   MACA 上这个类型不存在（mctlass 里没有 cutlass/arch/barrier.h），而且它承载的
//   事务屏障语义也无所依附：MACA 没有 TMA，没有 mbarrier，也没有生产者/消费者
//   warp specialization。原先依赖它的预取流水线已在 common_parts.cuh 中整段拆除，
//   改用「全线程协作 ldg 装载 + __syncthreads()」。
//   这里不再提供该别名 —— 保留它只会让已拆掉的流水线代码继续编译通过。
//   （如果将来某处仍需要它，应当显式实现，而不是从这里重新导出。）

// ---------------------------------------------------------------------------
// 以下两个原在 device/cuda/sm80/helpers.cuh 与 sm80/intrinsics.cuh 里，
// 那两个头整组是内联 PTX（MACA 汇编器不认），已不再包含，故在此给出 MACA 实现。
// ---------------------------------------------------------------------------

// 16 字节写 shared。
//   原实现：asm volatile("st.shared.b128 [%0], %1;" :: "r"(smem_addr), "q"(__int128 val));
//   MACA 汇编器只认 MACA ISA，且 xcore1000 不支持 __int128（'Int128 or UInt128
//   is not supported on this target'）。改为按 16 字节向量直接存 —— 同一块 shared
//   内存、同样的 16 字节，语义逐位等价。
CUTE_DEVICE
void st_shared(void* ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

// 16 字节写 shared，按两个 64 位半字给出（供原先构造 __int128 字面量的调用点使用）
CUTE_DEVICE
void st_shared(void* ptr, uint64_t lo, uint64_t hi) {
    *reinterpret_cast<uint2*>(ptr) = make_uint2((unsigned)lo, (unsigned)hi);
}

// 原实现：asm("trap;") / sm80 的 trap()。MACA 有 __trap() 内建（实测可用）。
CUTE_DEVICE
void trap() {
    __trap();
}

// 原为 cutlass::canonical_warp_idx_sync()（CUTLASS 里由 %warpid 硬件寄存器导出）。
// MACA 没有该寄存器。上游内核全程按 32 线程/warp 的假设走
// （lane_idx = threadIdx.x % 32，NUM_WARPS = NUM_THREADS / 32，
//  shuffle/ballot 的掩码都是 32 位），所以这里要的正是「block 内的 32 线程组序号」，
// 直接由 threadIdx.x 导出。内核在取该值之前没有 warp 退出，两者语义一致。
CUTE_DEVICE
uint32_t canonical_warp_idx_sync() {
    return threadIdx.x / 32u;
}

}

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800))
#define KERUTILS_ENABLE_SM80
#elif (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 800))
static_assert(false, "kerutils doesn't support SM architectures below SM80");
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900))
#define KERUTILS_ENABLE_SM90
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900 && __CUDA_ARCH__ < 1000))
#define KERUTILS_ENABLE_SM90A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000))
#define KERUTILS_ENABLE_SM100
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1000 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM100A
#endif

#if (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 1030 && __CUDA_ARCH__ < 1200))
#define KERUTILS_ENABLE_SM103A
#endif

#if (defined(__CLION_IDE__) || defined(__VSCODE_IDE__))
#define KERUTILS_ENABLE_SM80
#define KERUTILS_ENABLE_SM90
#define KERUTILS_ENABLE_SM90A
#define KERUTILS_ENABLE_SM100
#define KERUTILS_ENABLE_SM100A
#define KERUTILS_ENABLE_SM103A
#endif
