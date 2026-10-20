// ── provenance ──────────────────────────────────────────────────────────────
// Ported from the standalone C500 radix-TopK project (`dsa_topk`,
// `csrc/radix_topk.cuh`, commit 61ab380c77b81669718bfb11b95a583b0e661001).
// Two passes over the row: one histogram, one collect-and-stage that refines
// the threshold bin in shared memory.
//
// Three deviations from that source:
//
//   1. The 16-bit path speaks bf16, not fp16.  DeepSelect's dtypes are bf16
//      and fp32, so the fp16 key encode became `refine_bin`'s 16-bit overload
//      (`../common/primitives.cuh`), reading the value's own bit pattern
//      (`__bfloat16_as_ushort`) instead of fp16's.  The key transform is
//      otherwise unchanged -- it only needs the sign bit at bit 15, which both
//      formats have.  This is NOT the same as `float_to_uint8`, which rounds
//      fp32 through fp16 before binning and so has no bf16 counterpart.
//   2. `radix_topk_row_f32_rescan` is added, called from
//      `radix_topk_row_f32` when its coarse bin does not fit the candidate
//      arena.  Without it that case ranks only the members it managed to
//      stage and mis-answers quietly -- see its own comment.  The rest of the
//      fp32 half is verbatim.
//   3. The two chunked launchers take the row stride.  The stage-1 kernels
//      always did; only the launchers assumed a packed row, and DeepSelect's
//      callers hand over padded rows (`get_stride_requirement()` is 1024 B).
//
// ────────────────────────────────────────────────────────────────────────────
/*
 * Radix TopK CUDA Kernel Template
 *
 * 数据集: fp32 路径原样保留；16 位路径已改为 bf16（见上）。
 * 包含 warp-level 前缀和优化。
 *
 * 使用方式: #include <csrc/maca_kernels/common/radix_core.cuh>
 */

#ifndef RADIX_TOPK_CUH
#define RADIX_TOPK_CUH

#include <csrc/maca_kernels/common/radix_common.cuh>
#include <csrc/maca_kernels/common/radix_row_bf16.cuh>
#include <csrc/maca_kernels/common/radix_row_fp32.cuh>
#include <csrc/maca_kernels/common/radix_chunked.cuh>

#endif  // RADIX_TOPK_CUH
