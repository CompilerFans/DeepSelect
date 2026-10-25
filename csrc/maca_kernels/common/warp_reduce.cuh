// Wave and group reductions with CUB's `WarpReduce` semantics, lowered to
// MACA's row shuffle + lane read instead of the shared-memory-pipe permute.
//
// Why not the spellings this tree already carries:
//
// - `__shfl_*_sync` is a wrapper: measured from the emitted device assembly,
//   a 5-step `__shfl_down_sync` scan costs 71 instructions where the same
//   scan on the raw builtin costs 44 -- every call re-derives `__lane_id()`
//   and the width-bound arithmetic, and the compiler does not hoist it.  And
//   the instruction both lower to, `sm_bperm_b32`, travels the shared-memory
//   pipe (the `bsm` in the name), so a reduction in a hot walk competes with
//   the walk's own shared atomics for one pipe (the hist1a summary walk
//   issues 3 of them per 64-element block).
// - `__builtin_mxc_bsm_bpermute` drops the wrapper but keeps the pipe.
//
// What the guide (OG-26013-000-F5_V01, §5.1) offers instead is the 16-lane
// row shuffle: the wave's 64 lanes are 4 rows of 16, `mov_shfl` shifts/
// rotates/mirrors within a row in the ALU pipe, and `readlane` lifts one
// lane's register into a scalar (SALU).  A full-wave reduction is then 4
// chained row shifts + 4 vector ops (each row's reduction lands on the row's
// leader lane), 4 `readlane` + 3 scalar ops -- 15 ALU/SALU instructions,
// zero shared-memory traffic, and the result is a *scalar*: uniform across
// the wave with no broadcast step at all.
//
// Contract:
//
// - Full participation.  Every lane of the wave executes the call (rmsk/bmsk
//   are fixed at 0xf); the walks this serves run full waves.  bc=1 keeps a
//   lane's own value on out-of-row reads, which is correct because only the
//   leader lanes' values are ever consumed.
// - `warp_reduce_*` returns the reduction as a uniform scalar (all lanes).
// - `group8_reduce_*` leaves each 8-lane group's reduction on the group's
//   leader lane only: `(tx & 7) == DS_GROUP8_LEADER_IN8`.  This is the
//   `DS_GROUP_MAX8_TREE` shape (the hist1a summary walk).
// - Keys are compared as uint32.  mov_shfl moves bits without interpreting
//   them; float values must arrive pre-encoded (`refine_bin`).
//
// **Direction, established by probe and SASS** (`docs/experiments/
// 20261021_mov_shfl_tree/tree_probe`'s lane table): Row Shift Left
// (modes 0x10d) reads lane i+d into lane i, so a chained reduce lands on the
// row's/group's FIRST lane -- this header's default.  Row Shift Right
// (0x11d) reads lane i-d; `-DDS_ROW_SHIFT_RIGHT` flips every mode and leader
// constant from the one spelling below.  The same table carries an edge
// caveat: out-of-row reads do NOT honor the guide's bc promise (the rows'
// top lanes returned a stale cross-lane value; lane 0 returned garbage
// 0x1E00000) -- only reads that stay strictly inside the row are
// trustworthy.  Every chain here reads in-row lanes only by construction,
// which is also why non-leader partials must never be read.  SASS gate
// PASSED: mov_shfl lowers to one `mov_b32 ... row_shl:N rmsk bmsk bc`
// (ALU pipe, DPP form), readlane to `rl_b32 sD, ...` (SALU); no sm_/ds_/bsm
// instruction anywhere in the probe kernel.
//
// The `#else` arm keeps the upstream-CUDA dual-build form (32-lane
// `__shfl_down_sync` chain + broadcast), matching `radix_common.cuh`'s
// `kWarpSize` convention; this tree compiles only the MACA arm
// (`DEEP_SELECT_IS_BUILD_ON_MACA` is asserted in `csrc/structs.h`).

#pragma once

#include <cstdint>

namespace topk_select_common {

// ── the direction spelling (see the banner) ────────────────────────────────
#ifdef DS_ROW_SHIFT_RIGHT
// the guide's Row Shift Right modes (0x11d): lane i reads lane i-d, so the
// reduction lands on each row's/group's LAST lane.
#define DS_ROW_SH(d) (0x110 + (d))
constexpr int DS_ROW_LEADER_IN16 = 15;
constexpr int DS_GROUP8_LEADER_IN8 = 7;
#else
// default, the guide's Row Shift Left modes (0x10d): lane i reads lane i+d,
// leaders are the first lanes.
#define DS_ROW_SH(d) (0x100 + (d))
constexpr int DS_ROW_LEADER_IN16 = 0;
constexpr int DS_GROUP8_LEADER_IN8 = 0;
#endif

// ── reduction ops, cub::Max/Min/Sum's shape ─────────────────────────────────
struct ReduceMaxU32 {
    __device__ __forceinline__ uint32_t operator()(uint32_t a,
                                                   uint32_t b) const {
        return a > b ? a : b;
    }
};
struct ReduceMinU32 {
    __device__ __forceinline__ uint32_t operator()(uint32_t a,
                                                   uint32_t b) const {
        return a < b ? a : b;
    }
};
struct ReduceSumU32 {
    __device__ __forceinline__ uint32_t operator()(uint32_t a,
                                                   uint32_t b) const {
        return a + b;
    }
};

// ── full-wave reduce -> uniform scalar ──────────────────────────────────────
template <typename Op>
__device__ __forceinline__ uint32_t warp_reduce_u32(uint32_t v, Op op) {
#ifdef __MACACC__
    // 4 chained row shifts: the row reduction lands on each row's leader.
    uint32_t o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(8), 0xf,
                                                  0xf, 1);
    v = op(v, o);
    o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(4), 0xf, 0xf, 1);
    v = op(v, o);
    o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(2), 0xf, 0xf, 1);
    v = op(v, o);
    o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(1), 0xf, 0xf, 1);
    v = op(v, o);
    // Row leaders -> scalars; the rest is SALU, and the result is uniform.
    const uint32_t r0 = (uint32_t)__builtin_mxc_readlane(
        (int)v, 0 + DS_ROW_LEADER_IN16);
    const uint32_t r1 = (uint32_t)__builtin_mxc_readlane(
        (int)v, 16 + DS_ROW_LEADER_IN16);
    const uint32_t r2 = (uint32_t)__builtin_mxc_readlane(
        (int)v, 32 + DS_ROW_LEADER_IN16);
    const uint32_t r3 = (uint32_t)__builtin_mxc_readlane(
        (int)v, 48 + DS_ROW_LEADER_IN16);
    return op(op(r0, r1), op(r2, r3));
#else
    v = op(v, __shfl_down_sync(0xFFFFFFFFu, v, 16));
    v = op(v, __shfl_down_sync(0xFFFFFFFFu, v, 8));
    v = op(v, __shfl_down_sync(0xFFFFFFFFu, v, 4));
    v = op(v, __shfl_down_sync(0xFFFFFFFFu, v, 2));
    v = op(v, __shfl_down_sync(0xFFFFFFFFu, v, 1));
    return __shfl_sync(0xFFFFFFFFu, v, 0);
#endif
}

__device__ __forceinline__ uint32_t warp_reduce_max_u32(uint32_t v) {
    return warp_reduce_u32(v, ReduceMaxU32{});
}
__device__ __forceinline__ uint32_t warp_reduce_min_u32(uint32_t v) {
    return warp_reduce_u32(v, ReduceMinU32{});
}
__device__ __forceinline__ uint32_t warp_reduce_sum_u32(uint32_t v) {
    return warp_reduce_u32(v, ReduceSumU32{});
}

// ── 8-lane group reduce -> group leaders ────────────────────────────────────
// Result valid only on each group's leader lane, `(tx & 7) ==
// DS_GROUP8_LEADER_IN8`; other lanes hold partials over shifted subsets and
// must not be read.  This is the DS_GROUP_MAX8_TREE replacement: same
// chaining, same leaders, off the shared-memory pipe.
template <typename Op>
__device__ __forceinline__ uint32_t group8_reduce_u32(uint32_t v, Op op) {
#ifdef __MACACC__
    uint32_t o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(4), 0xf,
                                                  0xf, 1);
    v = op(v, o);
    o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(2), 0xf, 0xf, 1);
    v = op(v, o);
    o = (uint32_t)__builtin_mxc_mov_shfl((int)v, DS_ROW_SH(1), 0xf, 0xf, 1);
    v = op(v, o);
    return v;
#else
    v = op(v, __shfl_down_sync(0xFFu << (threadIdx.x & 0x18u), v, 4));
    v = op(v, __shfl_down_sync(0xFFu << (threadIdx.x & 0x18u), v, 2));
    v = op(v, __shfl_down_sync(0xFFu << (threadIdx.x & 0x18u), v, 1));
    return v;
#endif
}

__device__ __forceinline__ uint32_t group8_reduce_max_u32(uint32_t v) {
    return group8_reduce_u32(v, ReduceMaxU32{});
}
__device__ __forceinline__ uint32_t group8_reduce_min_u32(uint32_t v) {
    return group8_reduce_u32(v, ReduceMinU32{});
}
__device__ __forceinline__ uint32_t group8_reduce_sum_u32(uint32_t v) {
    return group8_reduce_u32(v, ReduceSumU32{});
}

}  // namespace topk_select_common
