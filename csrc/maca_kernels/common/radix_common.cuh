#ifndef RADIX_COMMON_CUH
#define RADIX_COMMON_CUH

#include <cuda_runtime.h>
#include <maca_bfloat16.h>   // the 16-bit path's value type
#include <cstdint>

#include <csrc/maca_kernels/common/launch.cuh>
#include <csrc/maca_kernels/common/primitives.cuh>

namespace topk_select_radix {

// The bf16 *chunked* entries' arm guard -- not the array bound, and not the
// public contract.  Its two readers (`launch_topk_bf16_chunked_k` and
// `launch_typed_chunked`, which picks between the arms) are on the way to an
// arm that instantiates `topk` at compile time.  The runtime-k row entries (`radix_topk_row_bf16_b`,
// `radix_topk_row_f32`) do **not** read it, and the public contract is
// `deep_select::kMaxTopK` = 4096 in `maca_topk.cu` -- which is why the fp32
// side has its own `kF32MaxTopK` (4096) below rather than reusing this.
//
// A 4096 bf16 topk reaches `radix_topk_row_bf16_b` and is answered there; this
// constant does not bound it.  That guard is the most misleading thing in the
// file for a reader asking "what is the top-k limit", so it is called out here
// rather than left to be discovered.
//
// There are four constants with this name in the tree (2048 here,
// `topk_chunks.cuh`, `topk_coarse12.cuh`; 4096 in `maca_topk.cu`) and nothing links
// them.  That is deliberate -- each is its own arm's bound -- but it means a
// grep for `kMaxTopK` does not answer "what is the top-k limit".
constexpr int kMaxTopK = 2048;
#ifndef KBLOCK_SIZE
constexpr int kBlockSize = 512;
#else
constexpr int kBlockSize = KBLOCK_SIZE;
#endif
#ifndef KCHUNK_BLOCK_SIZE
constexpr int kChunkBlockSize = 1024;
#else
constexpr int kChunkBlockSize = KCHUNK_BLOCK_SIZE;
#endif
// smem 大小：可通过编译参数 -DKSMEM_BYTES= 覆盖
//
// **The two lines this comment used to carry were wrong about the shipped
// build.**  They read "RTX 4080: 48KB" and "MetaX C500: 32KB", but nothing in
// this tree passes `-DKSMEM_BYTES` and no build entry defines it -- so the
// `__MACACC__` arm is what ships, and that is **16 KB**, not 32.  The 48 KB
// arm is the non-MACA fallback that no build here takes.
//
// 32 KB was not merely unused, it is unbuildable -- and so is 48 KB, the
// `#else` arm below.  Neither is reachable through a build entry here, but a
// reader who reaches for `-DKSMEM_BYTES` should know the ceiling is not a
// preference.  Measured on this tree, both passes, with `-fsyntax-only`:
//
//     -DKSMEM_BYTES=16384  compiles
//     -DKSMEM_BYTES=32768  static_assert fails: '16384 >= 30440'
//     -DKSMEM_BYTES=49152  static_assert fails: '16384 >= 46824'
//
// The failing assert is this file's
// `kCoarse12ArenaEntries * sizeof(uint32_t) >= kSmemInputSize * sizeof(uint32_t)`
// -- the 16-bit coarse level's 12-bit histogram aliases the *whole* dynamic
// arena, so 4096 bins (16,384 B) is a hard floor and any `kSMEM` that grows the
// arena past it fails to compile.  `kSmemInputSize` is
// `(kSMEM - kSmemStaticBytes) / 4`, i.e. 14,056 B / 30,440 B / 46,824 B for
// the three values above.  So the 16 KB arm is not a conservative choice that
// a bigger part could raise: on this source it is the only buildable one.
// (Both larger builds fail on the same assert -- the 48 KB `#else` with
// `'16384 >= 46824'`, 32 KB with `'16384 >= 30440'` -- which is why they are
// unbuilt rather than merely untaken.)
//
// Read `kF32SmemInputSize` (below) for what 16 KB actually buys.
//
// `maca_topk.cu`'s `#ifdef KSMEM_BYTES` assert has the same shape of problem
// from the other side -- see the note there.
#ifndef KSMEM_BYTES
#ifdef __MACACC__
constexpr size_t kSMEM = 16 * 1024;
#else
constexpr size_t kSMEM = 48 * 1024;
#endif
#else
constexpr size_t kSMEM = KSMEM_BYTES;
#endif

// ============================================================
// 类型转换
// ============================================================

// The NaN predicates and the key encodes come from `../common/primitives.cuh`,
// the launch primitive from `../common/launch.cuh` (one definition each, shared
// with the two ported dataflows and `maca_topk.cu`); they are pulled in
// unqualified because every call site below is spelled that way.
using topk_select_common::is_nan_bits32;
using topk_select_common::bf16_is_nan;
using topk_select_common::refine_bin;
using topk_select_common::adopt_dynamic_smem;

#ifdef __MACACC__
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif
static_assert(kBlockSize >= 256 && kBlockSize % kWarpSize == 0,
              "kBlockSize must cover the 256-bin scan and whole wavefronts");
static_assert(kChunkBlockSize >= 256 && kChunkBlockSize % kWarpSize == 0,
              "kChunkBlockSize must cover the 256-bin scan and whole wavefronts");
constexpr int kLongRowBlockSize = 1024;
static_assert(kLongRowBlockSize >= 256 && kLongRowBlockSize <= 1024 &&
                  kLongRowBlockSize % kWarpSize == 0,
              "kLongRowBlockSize must cover the 256-bin scan and whole wavefronts");

constexpr uint32_t kRadix = 256;
// Static smem, as the floor the dynamic arena is sized against:
// `s_histogram_buf[2][256+32]` plus the six words `s_counter`,
// `s_threshold_bin_id`, `s_high_threshold_bin_id`, `s_num_input[2]` and
// `s_last_remain`.  It is a floor, not the row kernel's footprint: the 16-bit
// row adds `s_wide_above` and `s_nlist`, and `topk_kernel_radix` adds its own
// `row_offset` / `nan_vote` -- each carries those in *its* static share.  The
// number sizes the dynamic region, which is what the two constants below need
// it for.
constexpr uint32_t kSmemStaticBytes =
    2 * (kRadix + 32) * sizeof(uint32_t) + 6 * sizeof(uint32_t);
static_assert(kSMEM >= kSmemStaticBytes, "KSMEM_BYTES is smaller than static shared state");
constexpr uint32_t kSmemInputSize = (kSMEM - kSmemStaticBytes) / sizeof(int32_t);

// The fp32 row's dynamic footprint: the generic arena, cut into two
// ping-pong tiles (the staging half and the refine's half).  The fp32 row's
// static block *is* the floor above, so the generic arena is exactly the
// space the floor leaves; the 16-bit row instead sizes its own arena from
// `kCoarse12ArenaEntries` (below) and adds two words of static.
constexpr uint32_t kF32SmemInputSize = kSmemInputSize / 2;
constexpr size_t kF32RowSmemBytes =
    2 * (size_t)kF32SmemInputSize * sizeof(uint32_t);
// The two tiles *are* the whole generic arena, so the fp32 row's request is the
// same 14,056 B asked for either way; an odd entry count would silently drop
// one and shrink the arena the row has always been given.
static_assert(kF32RowSmemBytes == (size_t)kSmemInputSize * sizeof(uint32_t),
              "the fp32 row's two tiles must cover the whole generic arena");

// The 16-bit row resolves its coarse level at 12 bits, not 8, over the same
// 16 KB the 8-bit level used for its candidate arena -- the histogram is dead
// by the time the arena is written, so one region carries both.  Twelve bits
// leave four, which is exactly the rest of a bf16 key: the two levels together
// determine the key, so the refine never ranks past the fine ties.
//
// The 8-bit level made the threshold bin wider than the arena on ordinary rows,
// so every one of them paid a third row walk.  See
// `docs/C500-radix-perf-ledger.zh.md` §2.3.
constexpr uint32_t kKeyBits = 16;
constexpr uint32_t kCoarse12Bits = 12;
constexpr uint32_t kCoarse12Shift = kKeyBits - kCoarse12Bits;      // 4
constexpr uint32_t kCoarse12Bins = 1u << kCoarse12Bits;            // 4096
constexpr uint32_t kCoarse12SubBins = 1u << kCoarse12Shift;        // 16, one per high byte
constexpr uint32_t kCoarse12HistBytes = kCoarse12Bins * sizeof(uint32_t);
constexpr uint32_t kCoarse12ArenaEntries = kCoarse12HistBytes / sizeof(uint32_t);
static_assert(kCoarse12ArenaEntries * sizeof(uint32_t) >= (size_t)kSmemInputSize * sizeof(uint32_t),
              "the 16-bit arena must not be smaller than the region it aliases");

// ── the dynamic region: one description, read by both sides ─────────────────
//
// A launch of the row kernel gets one `extern __shared__` region and the
// kernel carves it itself.  Two live shapes share that region, and the
// sharing is a schedule rather than a layout trick -- a `__syncthreads()`
// separates each pair of uses:
//
//   [ sort scratch ][ selected[topk] ]   ordered emit; the arena is dead
//   [ arena        ][ selected[topk] ]   unordered emit
//
// and the 16-bit row lays its 12-bit histogram over the arena's own bytes
// (`radix_row_bf16.cuh`'s `s_wide = s_input_flat`), once the histogram has
// been folded and narrowed and before the staging writes begin.
//
// **A `union` cannot state that.**  It is sized where it is declared, so it
// cannot be sized per call: it would have to hold `kSmemBudgetBytes` -- the
// `cudaFuncSetAttribute` ceiling, the ordered emit at `topk = 4096`, 48 KB --
// where the same call at `k = 512` asks for 18,432 B.  With the row kernel's
// 2,596 B of static beside it that is 51,748 B, one resident CTA per AP on a
// 64 KiB AP, against three for the 21,028 B the call actually asks for.  A
// static `__shared__` union cannot reach the region at all: `s_input_flat` is
// `extern __shared__`, so the arena sits at the region's base by construction
// (`contract.cuh`'s `radix_layout`).  Nor is the offset a kernel argument --
// an argument can disagree with what was allocated, and the disagreement is a
// silent overlap, where a `__host__ __device__` function read by both sides
// cannot.  So:
//
//   `radix_arena_bytes`  the dtype's row dataflow, on its own
//   `radix_lead_bytes`   where `selected` begins: the sort scratch or the
//                        arena, whichever is larger
//   `radix_smem_plan`    the composed budget: arena, sort, lead, total
//
// **A launcher requests `arena_bytes` when the dataflow answers to global
// memory and `total_bytes` when the contract half follows in the same
// kernel.**  The slots after the lead belong to the contract (`selected` is
// its staging), so the split's stages -- whose candidates live in the
// caller's arrays -- ask for the arena alone, and `topk_kernel_radix` asks for
// the total.  The two selectors keep their own regions: `topk_coarse12.cuh`
// and `topk_chunks.cuh` are separate kernels, each asserting its own
// histogram/arena pair.
__host__ __device__ constexpr size_t radix_arena_bytes(bool wide) {
    return wide ? kCoarse12HistBytes : kF32RowSmemBytes;
}

// The ordered emit's scratch: one 64-bit slot per `topk`, rounded up to the
// bitonic network's power of two.  Zero when nothing is sorted.
__host__ __device__ constexpr size_t radix_sort_bytes(uint32_t topk,
                                                      bool sorted) {
    uint32_t n_pad = 1;
    if (sorted) {
        while (n_pad < topk) n_pad <<= 1;
    }
    return sorted ? (size_t)n_pad * sizeof(uint64_t) : (size_t)0;
}

static __host__ __device__ constexpr __forceinline__ size_t radix_lead_bytes(
    uint32_t topk, bool sorted, bool wide) {
    const size_t sort_bytes = radix_sort_bytes(topk, sorted);
    const size_t arena_bytes = radix_arena_bytes(wide);
    return sort_bytes > arena_bytes ? sort_bytes : arena_bytes;
}

struct SmemPlan {
    size_t arena_bytes;  // the row dataflow, alone
    size_t sort_bytes;   // ordered-emit scratch; 0 unless the emit sorts
    size_t lead_bytes;   // max of the two: `selected` starts here
    size_t total_bytes;  // lead + the `topk` index slots after it
};

// The staging is sized by `topk`, not by the `kMaxTopK` ceiling: at k=512 that
// is 2 KB instead of 16 KB, and the difference decides the occupancy the
// launch gets.
__host__ __device__ constexpr SmemPlan radix_smem_plan(uint32_t topk,
                                                       bool sorted, bool wide) {
    const size_t lead = radix_lead_bytes(topk, sorted, wide);
    return SmemPlan{radix_arena_bytes(wide), radix_sort_bytes(topk, sorted), lead,
                    lead + sizeof(uint32_t) * (size_t)topk};
}

// The split's chunk geometry: `ceil(length / chunks)` rounded up to 8 elements.
// One definition because a chunk is one window to everything that reads it --
// both chunked splits stage by this size, and `nan_scan_kernel`
// (`contract.cuh`) is gridded by it, one CTA per (row, chunk) -- so a change
// here must move all of them together.  The rounding keeps every chunk base
// 8-element aligned for the vectorized row walk, and it lets the last chunk
// reach past a row's `length`, which every reader clamps against.
template <typename T>
__host__ __device__ __forceinline__ T chunk_size_for(T length, T chunks) {
    return ((length + chunks - 1) / chunks + 7) / 8 * 8;
}

/*
 * run_cumsum_warp: 256-element reverse inclusive scan, one buffer, warp
 * shuffles.  Each warp scans its own 32 elements, the warp totals go to
 * s_histogram_buf[0][257..264] (256 stays a sentinel), warp 0 scans those
 * eight, and each warp adds its offset -- three __syncthreads (the two above
 * plus the one that publishes the per-bin result, the last barrier in this
 * function), against eight in the naive form.  Returns the calling thread's
 * bin suffix so the caller can avoid reading a neighbouring bin before another
 * barrier.
 *
 * **The 16-bit path uses this, not fp32.**  Its only caller is
 * `radix_topk_row_bf16_b` (`radix_row_bf16.cuh`), twice per row, once per
 * pass.  `run_cumsum` (the double-buffered Hillis-Steele form) is what
 * `radix_topk_row_f32` calls, at three sites in `radix_row_fp32.cuh`.  The
 * sentence here used to say the reverse, which is the kind of thing a reader
 * trusts.
 */
__device__ __forceinline__ uint32_t run_cumsum_warp(
    uint32_t s_histogram_buf[2][256 + 32], uint32_t tx, uint32_t& exclusive_suffix)
{
    constexpr int NUM_WARPS = 256 / kWarpSize;  // 8 (warp=32) or 4 (warp=64)
    const uint32_t warp_id = tx / kWarpSize;
    const uint32_t lane = tx % kWarpSize;

    // Only the first 256 threads own histogram bins; larger blocks keep the
    // remaining warps at barriers without executing empty shuffle scans.
    uint32_t self_count = 0;
    uint32_t val = 0;
    if (tx < 256) {
        self_count = s_histogram_buf[0][tx];
        val = self_count;
        #pragma unroll
        for (int i = 1; i <= kWarpSize / 2; i <<= 1) {
#ifdef __MACACC__
            uint32_t n = __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, val, i);
#else
            uint32_t n = __shfl_down_sync(0xFFFFFFFF, val, i);
#endif
            if (lane + i < kWarpSize) val += n;
        }
    }

    // warp 总和写入 padding 区域
    if (lane == 0 && warp_id < NUM_WARPS)
        s_histogram_buf[0][257 + warp_id] = val;
    __syncthreads();

    // warp 0 对 NUM_WARPS 个总和做 reverse scan
    if (warp_id == 0) {
        uint32_t ws = (tx < NUM_WARPS) ? s_histogram_buf[0][257 + tx] : 0;
        #pragma unroll
        for (int i = 1; i < NUM_WARPS; i <<= 1) {
#ifdef __MACACC__
            uint32_t n = __shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, ws, i);
#else
            uint32_t n = __shfl_down_sync(0xFFFFFFFF, ws, i);
#endif
            if (tx + i < NUM_WARPS) ws += n;
        }
        if (tx < NUM_WARPS) s_histogram_buf[0][257 + tx] = ws;
    }
    __syncthreads();

    // 加上更高 bin 所在 warp 的总数
    if (tx < 256 && warp_id + 1 < NUM_WARPS)
        val += s_histogram_buf[0][257 + warp_id + 1];

    if (tx < 256)
        s_histogram_buf[0][tx] = val;
    __syncthreads();
    exclusive_suffix = val - self_count;
    return val;
}

}  // namespace topk_select_radix

#endif  // RADIX_COMMON_CUH
