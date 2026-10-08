/*
Common data types and macros that are used across the kerutils library.

[MACA] The MACA arm of `device/`, the sibling of `cuda/common.h` and
`ascend/common.h` rather than a variant of either.  `device/device.cuh`
includes exactly one of the three, chosen by `KERUTILS_IS_BUILD_ON_*`, so the
arms never coexist in a translation unit and each can differ from its siblings
where the platforms differ instead of accumulating `#ifdef`s.

What each arm owes its callers is the same set: the cache-hint and prefetch
enums, and the handful of primitives the shared device headers call.  A
platform whose headers cannot supply a primitive implements it here.
*/
#pragma once

// MACA's native bf16, from the toolkit's own header.  `maca_bfloat16.h` lives
// at `<toolkit>/include/common/`, which cucc puts on the include path (it
// pulls `<maca_fp16.h>` in with it).  The CUDA arm reaches the same type names
// through cu-bridge's `<cuda_bf16.h>` / `<cuda_fp8.h>`; this arm takes the
// type from MACA directly, so nothing here depends on the compatibility layer.
//
// **No cutlass on this arm, and the platform is what decides that.**
// `/opt/maca/include/cutlass/` does not exist -- the toolkit's cutlass-derived
// library is `include/mctlass/` -- so the CUDA arm's `<cutlass/bfloat16.h>`
// and `<cutlass/arch/barrier.h>` do not resolve here, and the aliases they
// feed (`bf16`, `transac_bar_t`) have no counterpart to alias: nothing in this
// tree uses them, and `ClusterTransactionBarrier` names a cluster mechanism
// MACA does not have.  `<cute/...>` *is* available (`include/cute/`) and is
// wanted -- it is where `CUTE_DEVICE` comes from.
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

// ---------------------------------------------------------------------------
// Primitives.  The CUDA arm gets these from `cuda/common.h` plus the sm80
// headers it includes; this arm has no sm80 header (they are inline PTX) and
// implements the ones the shared device headers call.
// ---------------------------------------------------------------------------

// 16 bytes to shared memory.
//   The sm80 original is `st.shared.b128` with the payload as an `__int128`
//   literal ('q' operand constraint).  Neither half is available here: the
//   assembler takes MACA ISA only, and this target has no `__int128`
//   ('Int128 or UInt128 is not supported on this target').  Storing a 16-byte
//   vector writes the same 16 bytes to the same shared address.
CUTE_DEVICE
void st_shared(void* ptr, uint4 val) {
    *reinterpret_cast<uint4*>(ptr) = val;
}

// The same 16 bytes, given as two 64-bit halves -- for call sites that build
// the payload from two registers rather than assembling a vector.
CUTE_DEVICE
void st_shared(void* ptr, uint64_t lo, uint64_t hi) {
    *reinterpret_cast<uint2*>(ptr) = make_uint2((unsigned)lo, (unsigned)hi);
}

// The CUDA arm's is `asm("trap;")`; MACA has the builtin.
CUTE_DEVICE
void trap() {
    __trap();
}

// The CUDA arm's is `cutlass::canonical_warp_idx_sync()`, which reads the
// `%warpid` register.  **MACA has no such register, and the number the callers
// want is not MACA's wave index.**  The ported kernels in
// `csrc/maca_kernels/xcore1600/` are written throughout against CUDA's
// 32-lane warp (`lane_idx = threadIdx.x % 32`, `NUM_WARPS = NUM_THREADS / 32`,
// 32-bit ballot and shuffle masks), so what they index is the thread's
// 32-thread group within the block, which is `threadIdx.x / 32u`.  Returning
// MACA's real wave index here would double it and mis-size every
// `warp_cnt[NUM_WARPS]` array that reads it.  No thread has exited before this
// is called, so it also satisfies `canonical_warp_idx_sync`'s requirement that
// the whole warp take part.
CUTE_DEVICE
uint32_t canonical_warp_idx_sync() {
    return threadIdx.x / 32u;
}

}

// The CUDA arm ends with a block defining `KERUTILS_ENABLE_SM80/90/100/...`
// from `__CUDA_ARCH__`.  It is deliberately *absent* here, and the reason is
// the opposite of the obvious one: **`__CUDA_ARCH__` is defined on this
// platform, as 800.**  cucc passes `-Xdevice -D__CUDA_ARCH__=800` so that
// CUDA-dialect source takes its sm80 paths, so the first condition of that
// block (`>= 800`) holds here and `KERUTILS_ENABLE_SM80` *would* come on --
// which is what pulls in the `cuda/sm80/` PTX headers this arm does not list.
//
// So it is the include list, not a macro test, that keeps them out, and
// re-adding a `cuda/sm*` header to this arm silently re-enables the block.
// `ref/kerutils_platform_arms/probe.cu` measures both compilation passes and
// prints this table; do not re-derive it from the toolchain's documentation.
