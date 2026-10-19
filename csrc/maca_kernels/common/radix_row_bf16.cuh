#ifndef RADIX_ROW_BF16_CUH
#define RADIX_ROW_BF16_CUH

#include "structs.h"   // ARCH_REGS_PER_AP -- the pair-walk's family gate
#include "radix_common.cuh"

// The block-max summary's step-down tree as a MACRO, so the text the compiler
// sees is byte-identical to the inline form it replaced.  A function form
// (`group_max8(key, tx)`, `__forceinline__`) was measured BROKEN IN SITU
// 2026-10-19: instruction-identical to the inline text in an isolated kernel
// (0/16384 blocks differ, `maxtree_bisect.cu` in
// `docs/experiments/20261019_c600u_pass1/`), but in the full row kernel it
// computes a low block max on ~35% of blocks -- the tree reads other lanes'
// registers, and the function boundary shifts the compiler's schedule around
// it.  Never extract this into a function again.
#ifdef __MACACC__
// No mask: only lane 0 is read and the tree chains, so the group-edge
// contamination `(tx + delta) & 63` can introduce reaches lanes >= 5 only,
// and lane 0's chain (0,1,2,4) never reads one.
#define DS_GROUP_MAX8_TREE(key)                                       \
    do {                                                              \
        uint32_t o_ = (uint32_t)__builtin_mxc_bsm_bpermute(           \
            (int)(((tx + 1u) & 63u) << 2), (int)(key));               \
        key = (key) > o_ ? (key) : o_;                                \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                    \
            (int)(((tx + 2u) & 63u) << 2), (int)(key));               \
        key = (key) > o_ ? (key) : o_;                                \
        o_ = (uint32_t)__builtin_mxc_bsm_bpermute(                    \
            (int)(((tx + 4u) & 63u) << 2), (int)(key));               \
        key = (key) > o_ ? (key) : o_;                                \
    } while (0)
#endif

namespace topk_select_radix {

__device__ __forceinline__ uint8_t bf16_to_uint8(maca_bfloat16 x) {
    return static_cast<uint8_t>(refine_bin(x) >> 8);
}

__device__ __forceinline__ bool bf16x8_is_aligned(const maca_bfloat16* input)
{
    return (reinterpret_cast<uintptr_t>(input) & (sizeof(uint4) - 1)) == 0;
}

__device__ __forceinline__ void hist_add_bf16_aligned(
    uint32_t* s_histogram, const maca_bfloat16* input, uint32_t idx)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        atomicAdd(&s_histogram[bf16_to_uint8(h[i])], 1u);
}

// Public standalone helper: retain vector loads for aligned offsets and avoid
// an illegal uint4 access when a caller supplies an odd row stride.
//
// **No call sites in this tree.**  The live bf16 walkers test
// `bf16x8_is_aligned` themselves and call `hist_add_bf16_aligned` directly
// (:1367, :1140, :1547 and the bf16 row entries), so this wrapper is the
// un-inlined form of a check they already make.  Kept because it is the one
// place the "aligned fast path, scalar fallback" pair is stated as a unit, and
// because the fp32 lane's missing guard is the bug this shape would have
// prevented (ledger §12.20.6).
__device__ __forceinline__ void hist_add_bf16(
    uint32_t* s_histogram, const maca_bfloat16* input, uint32_t idx)
{
    if (bf16x8_is_aligned(input + idx)) {
        hist_add_bf16_aligned(s_histogram, input, idx);
        return;
    }
    #pragma unroll
    for (int i = 0; i < 8; i++)
        atomicAdd(&s_histogram[bf16_to_uint8(__ldg(input + idx + i))], 1u);
}

// ============================================================
// 寄存器直方图 + Warp shuffle 合并
// ============================================================

/*
 * Intended design: one register histogram per warp (256 bins), 256/kWarpSize
 * bins per thread, each element's bin shuffled to its owner thread, reduced
 * within the warp and written to smem once -- no shared atomicAdd at all.
 *
 * NOT IMPLEMENTED, AND NOT CALLED.  The bodies below move one element per
 * shuffle, so each warp records about 1/64 of what it reads.  They are kept as
 * the shape of the idea, not as a working histogram; see
 * `docs/C500-radix-handover.zh.md` §9 for the measurement (1.5%) that killed it.
 * `hist_add_bf16_reg` and `hist_reg_to_smem` have **no call sites anywhere in
 * this tree** -- grep the names before assuming otherwise.  They are the only
 * block in this file that is both dead and known-broken, which is why the
 * warning is here and not on the k2048 kernels below.
 */

// 12 位粗层：一次 uint4 搬 8 个元素，每个元素一个原子加。
//
// Measured, so the next person does not re-open the merge idea (ledger §12.23).
// Same vectorized walk, same 4,096-bin coarse level, 134.2M elements, uniform
// bf16 keys, device 2: walk with no atomic 88.3 us, walk + one shared atomic
// per element 183.6 us.  So this loop is **95 us of atomic** and 88 us of walk
// -- the atomic is half the cost, which is what makes the merge idea tempting.
//
// It does not work.  `__match_any_sync` on MACA is a 32-iteration `cmp_lg_u32`
// software loop (2,048 instructions per 64-element warp step, 16/element); the
// atomic it would save is ~0.7/element.  Match-and-merge measures 2,133 us
// against the walk's 88 -- 20x, and identical for uniform and clustered keys
// because the loop is fixed-length.  Removing the atomic from the merge (1 us
// saved) is what proves the 2,045 us is the match itself.  Any de-atomized
// form has to skip `__match_any_sync` entirely: a per-thread private histogram
// with one flush at the end is the only shape left, and its risk is the
// per-element register dependency chain, not conflicts (that is what P4 hit).
// `kNan` folds the contract's NaN scan into this very load -- the same trade
// `hist_add_f32` makes, and for the same reason: the bytes are read anyway, so
// the scan is one compare per element here instead of a whole extra row walk.
template <bool kNan = false>
__device__ __forceinline__ bool hist_add_bf16_wide(
    uint32_t* s_wide, const maca_bfloat16* input, uint32_t idx,
    uint32_t* max_key = nullptr)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    // Both halves of each 32-bit word at once.  The ordered-key flip is a
    // per-half mask -- 0x8000 for a non-negative half, 0xffff for a negative
    // one, i.e. `0x8000 + 0x7fff * sign` -- and both halves' masks pack into
    // one word: the arithmetic shift puts each half's sign at bit 0 or bit 16,
    // so one multiply by 0x7fff builds the pair with no carry between halves
    // (0x7fff + 0x7fff0000 = 0x7fff7fff).
    //
    // The NaN test rides the raw word: `(h & 0x7fff) > 0x7f80` is
    // `bf16_is_nan` exactly, and it holds for both halves at once because
    // `| 0x80008000` puts each half's top bit within reach of one subtraction
    // of 0x7f817f81 -- the low half cannot borrow into the high one, since
    // 0x8000 - 0x7f81 = 0x7f >= 0.  Both forms are verified against the
    // per-element ones on all 65,536 half patterns.
    uint32_t z = 0u;
    // Fully unrolled, and that is a measured constraint rather than taste: the
    // walk is memory-latency bound on a long row, and a partial unroll halves
    // the loads in flight per thread.  `unroll 2` costs the row kernel +92% at
    // `b4096 x 1048576 k512` bf16 (+14% at 16384, where the row is too short to
    // be latency bound) while saving 3-4 MT of registers, which is the wrong
    // trade at every length.
    #pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp =
            w[i] ^ ((((w[i] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
        atomicAdd(&s_wide[(kp >> kCoarse12Shift) & 0x0FFFu], 1u);
        atomicAdd(&s_wide[kp >> (16 + kCoarse12Shift)], 1u);
        // The block-max summary (see `radix_topk_row_bf16_b`) wants the largest
        // key in this thread's 8 elements; both halves are already in `kp`, and
        // the key's order is the bin's, so one max over keys is one max over
        // 12-bit bins without shifting either of them.
        if (max_key != nullptr) {
            const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
            const uint32_t m = lo > hi ? lo : hi;
            *max_key = m > *max_key ? m : *max_key;
        }
        if (kNan) {
            z |= (((w[i] & 0x7FFF7FFFu) | 0x80008000u) - 0x7F817F81u)
                 & 0x80008000u;
        }
    }
    return z != 0u;
}

// FP16: 寄存器直方图 + warp shuffle
__device__ __forceinline__ void hist_add_bf16_reg(
    uint32_t* r_hist,  // 每线程持有 kBinsPerThread 个 bin
    const maca_bfloat16* input, uint32_t idx, uint32_t lane)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);

    #pragma unroll
    for (int i = 0; i < 8; i++) {
        uint8_t bin = bf16_to_uint8(h[i]);
        uint32_t owner = bin / kBinsPerThread;  // 哪个线程拥有这个 bin
        uint32_t local = bin % kBinsPerThread;   // 在 owner 的哪个位置

        // 用 shuffle 将 bin 值发送给 owner
        // 所有线程同时执行 shuffle, owner 收到 bin 值后累加
#ifdef __MACACC__
        uint32_t recv_bin = __shfl_sync(0xFFFFFFFFFFFFFFFFULL, bin, owner);
#else
        uint32_t recv_bin = __shfl_sync(0xFFFFFFFF, bin, owner);
#endif
        if (lane == owner) {
            // owner 线程累加对应的寄存器
            // 注意: 这里需要一个映射 recv_bin -> r_hist 索引
            // 由于 recv_bin 可能来自任意线程, 需要按 bin 值累加
            r_hist[recv_bin % kBinsPerThread] += 1;
        }
    }
}

// 将寄存器直方图写入 smem
__device__ __forceinline__ void hist_reg_to_smem(
    uint32_t* r_hist,
    uint32_t* s_histogram,
    uint32_t lane)
{
    #pragma unroll
    for (int i = 0; i < kBinsPerThread; i++) {
        uint32_t bin = lane * kBinsPerThread + i;
        if (bin < 256 && r_hist[i] > 0) {
            atomicAdd(&s_histogram[bin], r_hist[i]);
        }
    }
}

// ============================================================
// FP16 radix topk row
// ============================================================

// One member of the overflow row's threshold bin: everything strictly above the
// fine threshold is selected outright, and the ties fill what is left of the
// window from the top down -- the arena path's rule, applied to a re-walked row
// instead of a staged candidate list.
__device__ __forceinline__ void overflow_emit_member(
    uint32_t idx, uint32_t key, uint32_t high_threshold_bin,
    uint32_t threshold_bin, uint32_t remain_topk, uint32_t topk, int32_t* output,
    uint32_t* s_counter, int32_t* s_last_remain)
{
    if ((key >> kCoarse12Shift) != high_threshold_bin) return;
    const uint32_t low = key & (kCoarse12SubBins - 1u);
    if (low > threshold_bin) {
        output[atomicAdd(s_counter, 1u)] = static_cast<int32_t>(idx);
    } else if (low == threshold_bin && remain_topk != 0) {
        auto p = atomicAdd(s_last_remain, -1);
        if (p > 0) output[topk - p] = static_cast<int32_t>(idx);
    }
}

// `kNan` / `row_nan` mirror `radix_topk_row_f32`: the contract's NaN scan rides
// pass 1's own load, and the block-wide vote is handed back so the caller does
// not walk the row a second time to ask the same question.
// One summary entry per 64 elements -- 8 consecutive lanes' `uint4`, i.e. one
// 128-byte line.  It is the granularity the collect walk already moves in, so
// a skipped entry is exactly a line the walk would have fetched and found
// nothing to do with.
constexpr uint32_t kSkipBlockElems = 64;

// Walk the blocks a summary named, with the aligned row walk's own layout: the
// eight lanes of a group take one `uint4` each out of the same 64-element block
// and `body(idx, raw)` sees the same elements, in the same per-lane order, that
// the row walk hands it.  What changed is only which blocks the group visits --
// a list of the ones that can hold a candidate, instead of the whole row.
//
// This is the shape the lever needs, not a detail: a test in front of the walk
// leaves the decision and the line on one latency chain and measures at 0.45 of
// the full walk even when 96% of the blocks are skipped, while this decoupled
// form reaches 0.14 (docs/experiments/20261010_skim_rate/).
template <uint32_t BLOCK_SIZE, typename F>
__device__ __forceinline__ void walk_kept_blocks(const maca_bfloat16* input,
                                                 const uint32_t* list,
                                                 uint32_t n, F&& body) {
    const uint32_t lane = threadIdx.x & 7u;
    const uint32_t group = threadIdx.x >> 3;
#pragma unroll 2
    for (uint32_t j = group; j < n; j += BLOCK_SIZE / 8u) {
        const uint32_t base = list[j] << 6;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + base) + lane);
        const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
#pragma unroll
        for (int i = 0; i < 8; i++) body(base + lane * 8u + (uint32_t)i, h[i]);
    }
}

template <uint32_t BLOCK_SIZE, bool kNan = false, bool kSkip = true>
__device__ __forceinline__ void radix_topk_row_bf16_b(
    const maca_bfloat16* input, int32_t* output, uint32_t length, uint32_t topk,
    bool* row_nan = nullptr,
    // The skip summary and its block list: pass 1 leaves each 64-element
    // block's max key in `block_max` (2 bytes per block) and the collect
    // compacts the blocks that can hold a candidate into `block_list`
    // (`block_max`'s owner allocates; 4 bytes per block) and walks that instead
    // of the row.  A block whose max bin is below the threshold holds no
    // element either walk would act on, so the list is answer-preserving by
    // construction.  Both null -- every other caller -- leaves this function
    // exactly as it was.
    //
    // `kSkip` is the caller saying whether this build of the row carries the
    // round at all: false compiles the summary, the list and the walks out.
    // The 64-bit-index row is the one caller that says false -- its epilogue
    // already costs it two registers more than the 32-bit row's, and carrying
    // the summary on top of that puts it over the 32 MT a 1024-thread block
    // needs for its second CTA per SM (measured with `--resource-usage`).
    uint16_t* block_max = nullptr, uint32_t* block_list = nullptr)
{
    constexpr uint32_t RADIX = kRadix;
    constexpr uint32_t SMEM_INPUT_SIZE = kCoarse12ArenaEntries;

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    __shared__ uint32_t s_high_threshold_bin_id;
    __shared__ uint32_t s_num_input[2];
    __shared__ uint32_t s_wide_above;
    __shared__ int32_t s_last_remain;
    // The kept-block list's population count: eight lanes write it before the
    // walk reads it, on the barrier between them.
    __shared__ uint32_t s_nlist;
    extern __shared__ uint32_t s_input_flat[];
    // The coarse histogram is laid over the candidate arena: `s_input_flat`
    // holds the 4,096 bins until the narrow is done, and the staging writes
    // only start after the barrier that follows it.
    uint32_t* const s_wide = s_input_flat;

    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = topk;
    auto& s_histogram = s_histogram_buf[0];

    for (uint32_t b = tx; b < kCoarse12Bins; b += BLOCK_SIZE) s_wide[b] = 0;
    __syncthreads();

    // C500 has an intermittent race in the mixed vector-plus-tail path for
    // 1024-thread blocks. Use one uniform load mode for odd-length rows.
    const bool input_aligned = (length & 7u) == 0 && bf16x8_is_aligned(input);
    uint32_t vec_len = length / 8 * 8;
    // Whole blocks: the summary, the list and every walk that follows one cover
    // `[0, nb << 6)`, and each of them owes the rest of the row to its own tail
    // loop.  The three tail loops have to agree on where that starts -- a
    // histogram that stops at `vec_len` is short of the elements the collect's
    // tail then finds, and the count it derives is what the stage is sized by.
    const uint32_t nb = vec_len >> 6;
    const uint32_t list_end = nb << 6;
    // The summary is built on the aligned walk only, and that is not a
    // limitation: eight consecutive lanes there cover exactly one 64-element
    // block, which is the whole reason the alignment of the walk and the
    // summary granularity can be the same number.  A row that takes the scalar
    // walk gets no summary and is walked whole by the collect.
    const bool summary_wanted = kSkip && block_max != nullptr && input_aligned;
    bool nan_local = false;
    if (input_aligned) {
        if (summary_wanted) {
            // The same walk, the same addresses: `blk` is `idx >> 6` with the
            // bound hoisted to the 8-lane group, so a group enters and leaves
            // the body together.  That is what makes the group's own mask (and
            // not the full wave's) the exact set of participating lanes --
            // including in the last, partially filled block of a row that is
            // not 64-aligned, where some lanes of the group have no element to
            // read.
            const uint32_t group = tx >> 3;
            const uint32_t lane = tx & 7u;
            // Whole blocks only.  The list walk moves a `uint4` per lane and
            // must not read past the row, so the elements past the last whole
            // block stay with the tail loop both walks already carry -- and
            // with this bound the guarded load below has nothing left to guard.
            const uint32_t step = BLOCK_SIZE / 8;
            uint32_t blk = group;
#ifdef __MACACC__
            // Two blocks in flight per iteration where the register file
            // affords it: the walk keeps one outstanding uint4 per thread,
            // which is a memory-latency cap, not a bandwidth one -- the
            // measured read curve on a C600U is 265 -> 956 -> 1390 GB/s for
            // 1/2/4 outstanding at one 512-thread CTA per SM (1024 -> 1616 at
            // two).  The second block is the whole gap on under-filled grids
            // and ~6% once the shared atomics cap.  The C500's 32 MT wall
            // keeps the single-issue walk.
            if constexpr (ARCH_REGS_PER_AP >= 131072) {
                for (; ((blk + step + 1u) << 6) <= vec_len; blk += 2 * step) {
                    uint32_t key0 = 0, key1 = 0;
                    nan_local |= hist_add_bf16_wide<kNan>(
                        s_wide, input, (blk << 6) + lane * 8u, &key0);
                    nan_local |= hist_add_bf16_wide<kNan>(
                        s_wide, input, ((blk + step) << 6) + lane * 8u, &key1);
                    DS_GROUP_MAX8_TREE(key0);
                    DS_GROUP_MAX8_TREE(key1);
                    if (lane == 0u) {
                        block_max[blk] = (uint16_t)(key0 >> kCoarse12Shift);
                        block_max[blk + step] =
                            (uint16_t)(key1 >> kCoarse12Shift);
                    }
                }
            }
#endif
            for (; ((blk + 1u) << 6) <= vec_len; blk += step) {
                const uint32_t idx = (blk << 6) + lane * 8u;
                uint32_t key = 0;
                nan_local |= hist_add_bf16_wide<kNan>(s_wide, input, idx, &key);
#ifdef __MACACC__
                DS_GROUP_MAX8_TREE(key);
#else
                const unsigned gmask = 0xFFu << (tx & 0x18u);
                // The tree is chained -- each shuffle reads what the step before
                // it produced.  Three shuffles taken off the *same* value would
                // give lane 0 the max over lanes 0,1,2,4 only, and a summary
                // that can under-estimate a block's largest key is not a
                // skip hint: a block holding a candidate would be compacted
                // away, so the collect would count and stage a member of the
                // threshold bin that the walk never reaches.
                uint32_t other = (uint32_t)__shfl_down_sync(gmask, key, 1);
                key = key > other ? key : other;
                other = (uint32_t)__shfl_down_sync(gmask, key, 2);
                key = key > other ? key : other;
                other = (uint32_t)__shfl_down_sync(gmask, key, 4);
                key = key > other ? key : other;
#endif
                if (lane == 0u)
                    block_max[blk] = (uint16_t)(key >> kCoarse12Shift);
            }
        } else {
            for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8)
                nan_local |= hist_add_bf16_wide<kNan>(s_wide, input, idx);
        }
        // The summary walk above is block-granular, so the row's last partial
        // block is this loop's, and it starts where the collect's tail does --
        // at `list_end`, not at `vec_len`.  Starting it at `vec_len` leaves
        // `[list_end, vec_len)` out of the coarse histogram whenever the two
        // differ, and those elements are ones the collect's tail goes on to
        // find: the stage is then sized by a count short of what the walk
        // writes, which runs it past the topk slots it owns.  It is also the
        // only NaN vote those elements get.
        const uint32_t hist_tail_start = summary_wanted ? list_end : vec_len;
        for (uint32_t idx = hist_tail_start + tx; idx < length; idx += BLOCK_SIZE) {
            const maca_bfloat16 raw = __ldg(input + idx);
            atomicAdd(&s_wide[refine_bin(raw) >> kCoarse12Shift], 1u);
            if (kNan) nan_local |= bf16_is_nan(__bfloat16_as_ushort(raw));
        }
    } else {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const maca_bfloat16 raw = __ldg(input + idx);
            atomicAdd(&s_wide[refine_bin(raw) >> kCoarse12Shift], 1u);
            if (kNan) nan_local |= bf16_is_nan(__bfloat16_as_ushort(raw));
        }
    }
    __syncthreads();
    // The histogram is complete and the CTA's NaN vote is in, on one barrier --
    // the same reduction `row_has_nan` runs, only across a block.
    if (kNan && row_nan != nullptr)
        *row_nan = __syncthreads_or((int)nan_local) != 0;

    // Fold 4,096 -> 256, one bin per high byte, so the scan stays the same
    // 256-wide warp scan the 8-bit level used; thread 0 then narrows inside the
    // byte the scan lands on.  This is the last read of the coarse histogram.
    if (tx < RADIX) {
        uint32_t c = 0;
        #pragma unroll
        for (uint32_t s = 0; s < kCoarse12SubBins; s++)
            c += s_wide[(tx << kCoarse12Shift) + s];
        s_histogram[tx] = c;
    } else if (tx == RADIX) {
        s_histogram[RADIX] = 0;
        // The refine's landing state, initialized here so that a crossing the
        // scan cannot find leaves a defined bin behind rather than whatever the
        // previous launch on this SM left in shared memory.
        s_threshold_bin_id = 0;
        s_last_remain = 0;
    }
    __syncthreads();

    uint32_t exclusive_suffix = 0;
    uint32_t inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);

    if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
        s_high_threshold_bin_id = tx; s_num_input[0] = 0; s_counter = 0;
    }
    __syncthreads();

    // The scan named a high byte; the 12-bit threshold is the sub-bin inside it
    // where the count crosses `remain_topk`.  `above` ends as the count of keys
    // strictly above that threshold, which is what the emit owes the window.
    if (tx == 0) {
        const uint32_t high = s_high_threshold_bin_id;
        uint32_t above = s_histogram[high + 1];
        uint32_t sub = 0;
        for (int s = (int)kCoarse12SubBins - 1; s >= 0; --s) {
            const uint32_t c = s_wide[(high << kCoarse12Shift) + (uint32_t)s];
            // The bin has to carry what the window still wants, and the test is
            // against the window `remain_topk` itself -- not against what is
            // left once the byte above has been counted.  Testing the remainder
            // fires a sub-bin early whenever the byte has slack, which leaves
            // the window short and the refine with no bin to land on.
            if (above + c > remain_topk) { sub = (uint32_t)s; break; }
            above += c;
        }
        s_high_threshold_bin_id = (high << kCoarse12Shift) | sub;
        s_wide_above = above;
        // The kept-block list's population, cleared here so the barrier that
        // publishes the threshold publishes this too and the list build below
        // needs only the one barrier it ends on.
        s_nlist = 0;
    }
    __syncthreads();

    // ── the kept-block list ─────────────────────────────────────────────────
    // A block whose max bin is at or above the threshold is one the walks below
    // can act on; every other block is a line they would fetch and do nothing
    // with.  Those are 96% of the row on a long one, so compaction turns the
    // second read of the row into a read of a list -- and the compaction has to
    // be its own step: a test in front of the walk leaves the decision and the
    // fetch on one latency chain and measures at 0.45 of the full walk, while
    // this decoupled form reaches 0.14 (docs/experiments/20261010_skim_rate/).
    //
    // Past three quarters of the blocks it stops paying -- the walk reads most
    // of the row anyway and the list is traffic on top of it (measured: 0.63 at
    // half the blocks, 1.21 at all of them) -- so the count decides and the list
    // is not written past the cap.  `nlist` counts every kept block either way.
    const uint32_t coarse_threshold = s_high_threshold_bin_id;
    const uint32_t list_cap = nb - (nb >> 2);
    uint32_t nlist = 0;
    bool use_list = false;
    if (summary_wanted) {
        for (uint32_t b = tx; b < nb; b += BLOCK_SIZE) {
            if (block_max[b] < coarse_threshold) continue;
            const uint32_t pos = atomicAdd(&s_nlist, 1u);
            if (pos < list_cap) block_list[pos] = b;
        }
        __syncthreads();
        nlist = s_nlist;
        // `list_cap > 0` matters on a row with no whole block at all: there the
        // vector walk is the better of the two, and the list covers nothing.
        use_list = list_cap > 0 && nlist <= list_cap;
    }

    {
        const auto threshold_bin = coarse_threshold;
        remain_topk -= s_wide_above;
        if (remain_topk == 0) {
            if (use_list) {
                walk_kept_blocks<BLOCK_SIZE>(
                    input, block_list, nlist,
                    [&](uint32_t idx, maca_bfloat16 raw) {
                        if ((refine_bin(raw) >> kCoarse12Shift) > threshold_bin)
                            output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                    });
                // What the list does not cover: the row past its last whole
                // block.
                for (uint32_t idx = list_end + tx; idx < length; idx += BLOCK_SIZE)
                    if ((refine_bin(__ldg(input + idx)) >> kCoarse12Shift) > threshold_bin)
                        output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else {
                for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                    if ((refine_bin(__ldg(input + idx)) >> kCoarse12Shift) > threshold_bin)
                        output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        // One element of the collect, in the shape both walks hand it: the row
        // walk's `idx + i` and the list walk's `base + lane*8 + i` are the same
        // index for the same element.
        const auto collect_one = [&](uint32_t idx, maca_bfloat16 raw) {
            const uint32_t bin = refine_bin(raw) >> kCoarse12Shift;
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                // The count is over the bin, the staging is over the arena.
                // Gating both on the arena made the refine's histogram partial
                // whenever the bin overflowed, which is what the full-row
                // rebuild below used to repair.
                const uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                if (pos < SMEM_INPUT_SIZE) s_input_flat[pos] = idx;
                atomicAdd(&s_histogram[refine_bin(raw) & (kCoarse12SubBins - 1u)], 1u);
            }
        };
        if (use_list) {
            walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, collect_one);
        } else if (input_aligned) {
            for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                #pragma unroll
                for (int i = 0; i < 8; i++) collect_one(idx + i, h[i]);
            }
        } else {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                collect_one(idx, __ldg(input + idx));
        }
        // The tail of the row: the list walk covers whole blocks and the row
        // walk covers `[0, vec_len)`, so what is left starts at `list_end` and
        // `vec_len` respectively -- and the scalar walk has already covered the
        // whole row.
        const uint32_t tail_start =
            use_list ? list_end : (input_aligned ? vec_len : length);
        for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
            collect_one(idx, __ldg(input + idx));
        __syncthreads();
    }

    {
        const bool overflow = s_num_input[0] > SMEM_INPUT_SIZE;
        // No rebuild pass here: the fine histogram above already counted every
        // member of the threshold bin, staged or not, so the refine below ranks
        // the bin itself rather than the part of it that fit.
        exclusive_suffix = 0;
        inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);
        if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
            s_threshold_bin_id = tx;
            s_last_remain = static_cast<int32_t>(remain_topk - exclusive_suffix);
        }
        __syncthreads();
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (overflow) {
            // The rescan answers for the coarse threshold bin's members, and the
            // list is the blocks that can hold one of those, so it walks the
            // list on the same terms as the collect above.
            const auto high_threshold_bin = coarse_threshold;
            const auto emit_member = [&](uint32_t idx, maca_bfloat16 raw) {
                overflow_emit_member(idx, refine_bin(raw), high_threshold_bin,
                                     threshold_bin, remain_topk, topk, output,
                                     &s_counter, &s_last_remain);
            };
            if (use_list) {
                walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, emit_member);
            } else if (input_aligned) {
                for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) emit_member(idx + i, h[i]);
                }
            } else {
                for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                    emit_member(idx, __ldg(input + idx));
            }
            // Same tail split as the collect above.
            const uint32_t emit_tail_start =
                use_list ? list_end : (input_aligned ? vec_len : length);
            for (uint32_t idx = emit_tail_start + tx; idx < length;
                 idx += BLOCK_SIZE)
                emit_member(idx, __ldg(input + idx));
            __syncthreads(); return;
        }
        const auto num = s_num_input[0];
        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
                auto idx = s_input_flat[i];
                if ((refine_bin(__ldg(input + idx)) & (kCoarse12SubBins - 1u)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
            __syncthreads(); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
            auto idx = s_input_flat[i];
            uint32_t bin = refine_bin(__ldg(input + idx)) & (kCoarse12SubBins - 1u);
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                auto p = atomicAdd(&s_last_remain, -1);
                if (p > 0) output[topk - p] = static_cast<int32_t>(idx);
            }
        }
        __syncthreads();
    }
}

}  // namespace topk_select_radix

#endif  // RADIX_ROW_BF16_CUH
