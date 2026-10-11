#ifndef RADIX_ROW_FP32_CUH
#define RADIX_ROW_FP32_CUH

#include <cuda_fp16.h>       // float_to_uint8: the fp32 path's fp16 rounding

#include "radix_common.cuh"

namespace topk_select_radix {

// 与 SGLang 原始实现一致：先转 half 再取高 8 位
__device__ __forceinline__ uint8_t float_to_uint8(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

// ============================================================
// 向量化直方图 (smem atomicAdd)
// ============================================================

// `hist_add_f32`'s call sites are the *unit-strided* walk: pass 1 (:584-590)
// takes `tx, tx+BS, tx+2BS, tx+3BS` float4 units per lane, so adjacent lanes of
// one load instruction are 16 bytes apart and a 128-byte segment carries 128
// useful bytes.
//
// **The `tx*4` walk this note used to describe is a different loop.**  It is
// pass 2's two open-coded `__ldg(float4*)` loops (:626, :641), which take four
// CONSECUTIVE float4 per thread -- 64 bytes between adjacent lanes, 32 useful
// bytes per 128-byte segment.  They do not call this function, and they were
// left alone on purpose when pass 1 was coalesced (`eaa0131`).
//
// The stride is real and deliberate -- but it is NOT why pass 1 is short of
// the wall.  Coalescing that walk is a 65% win in a microbench and -0.7% in
// the kernel: at the kernel's occupancy the two patterns are 7% apart, not
// 65%, because pass 1 issues one shared-bucket atomic per element and that is
// what it is waiting on.  See `docs/C500-radix-perf-ledger.zh.md` before
// changing it.
//
// `kNan` is the contract's NaN scan folded into this very load.  The two walks
// (`radix_topk_row_f32`'s pass 1 and `nan_scan_kernel`) read exactly the same
// bytes at exactly the same vector width, so the scan is a whole second row
// walk bought for four bit-tests per 16 bytes.  `kNan = false` is the shipped
// helper with none of it, and it is the default so the 16-bit path -- which has
// no such scan -- compiles to what it always did.
template <bool kNan = false>
__device__ __forceinline__ bool hist_add_f32(
    uint32_t* s_histogram, const float* input, uint32_t idx)
{
    const float4 v = __ldg(reinterpret_cast<const float4*>(input + idx));
    atomicAdd(&s_histogram[float_to_uint8(v.x)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.y)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.z)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.w)], 1u);
    if (!kNan) return false;
    return is_nan_bits32(__float_as_uint(v.x)) | is_nan_bits32(__float_as_uint(v.y))
         | is_nan_bits32(__float_as_uint(v.z)) | is_nan_bits32(__float_as_uint(v.w));
}

// One element of the fp32 row's second pass.  Both walkers ask every element
// the same question -- `float_to_uint8(raw) > threshold_bin` -- and both are
// read-bound, so both move four elements per load exactly as pass 1 does:
// one element per load tops out at the 4-byte streaming rate, well under the
// 16-byte wall every other pass here reaches, and costs four times the
// dependent load stalls per row.
//
// The arena write stays capacity-limited while `s_num_input[0]` counts EVERY
// member of the bin.  That split is what makes the overflow test sound, and
// the rescan rebuilds the fine histogram it discards -- so the clamped
// histogram add inside the same branch is deliberate.
__device__ __forceinline__ void stage_f32_lane(
    float raw, uint32_t idx, uint32_t threshold_bin, int32_t* output,
    uint32_t* s_counter, uint32_t* s_input_flat, uint32_t* s_num_input0,
    uint32_t* s_histogram, uint32_t smem_input_size, uint32_t chunk_begin = 0)
{
    const uint32_t bin = float_to_uint8(raw);
    if (bin > threshold_bin) {
        output[atomicAdd(s_counter, 1u)] =
            static_cast<int32_t>(idx + chunk_begin);
    } else if (bin == threshold_bin) {
        const uint32_t pos = atomicAdd(s_num_input0, 1u);
        if (pos < smem_input_size) {
            s_input_flat[pos] = idx;
            atomicAdd(&s_histogram[(refine_bin(raw) >> 24) & 0xFFu], 1u);
        }
    }
}

// ============================================================
// 前缀和
// ============================================================

/*
 * run_cumsum: 256 元素 inclusive scan (双缓冲 Hillis-Steele)
 */
__device__ __forceinline__ void run_cumsum(
    uint32_t s_histogram_buf[2][256 + 32], uint32_t tx)
{
    #pragma unroll 8
    for (int i = 0; i < 8; ++i) {
        if (tx < 256) {
            const auto j = 1 << i;
            const auto k = i & 1;
            auto value = s_histogram_buf[k][tx];
            if (tx + j < 256) value += s_histogram_buf[k][tx + j];
            s_histogram_buf[k ^ 1][tx] = value;
        }
        __syncthreads();
    }
}

// ============================================================
// FP32 radix topk row
// ============================================================

// The fp32 row's dynamic shared-memory requirement, as seen from a launch
// site.  `radix_topk_row_f32` derives the same number from `kSMEM`, and both
// callers (the row entry and the chunked split's stage kernels) must request
// exactly it -- so it lives here rather than being re-spelled at each launch.
//
// The static half is `s_histogram_buf[2][256+32]` plus the six words
// `s_counter`, `s_threshold_bin_id`, `s_high_threshold_bin_id`,
// `s_num_input[2]` and `s_last_remain`.
constexpr uint32_t kF32StaticBytes =
    2 * (kRadix + 32) * sizeof(uint32_t) + 6 * sizeof(uint32_t);
// Two of these tiles the dynamic region, one for the arena and one for the
// refine's ping-pong staging; `radix_topk_row_f32` names the halves by
// multiplying this.
constexpr uint32_t kF32SmemInputSize =
    (kSMEM - kF32StaticBytes) / (2 * sizeof(uint32_t));
constexpr size_t kF32RowSmemBytes =
    2 * (size_t)kF32SmemInputSize * sizeof(uint32_t);

// Overflow resolution for the fp32 row below.  Do not remove this path.
//
// The staging pass of `radix_topk_row_f32` keeps only the first
// `SMEM_INPUT_SIZE` members of the coarse threshold bin and drops the rest,
// after which the refinement ranks that subset rather than the bin.  The
// answer still has `topk` distinct indices, so nothing looks wrong: the values
// are simply from the wrong place.  Measured on a uniform row of L=129280,
// k=512, the bin holds 8025 members against an arena of 1757 and **391 of the
// 512 picks land below the row's true k-th largest** -- which is why this path
// is not a fallback but part of the answer for that shape.
//
// The 16-bit path handles the same case by re-walking the row -- see the
// `overflow` branch of `radix_topk_row_bf16_b`.  For a four-byte key that
// becomes: rank the members of the coarse bin by the exact fp32 key bytes, one
// level per round, each round a histogram pass and an emit-and-narrow pass.
// The candidate set shrinks by construction, so the tail slots fill from the
// last byte's ties exactly as the arena path fills them.
//
// `excess_coarse` members above the coarse bin are already emitted and
// `s_counter` counts them; `remain` are still needed, all from inside the bin.
__device__ __forceinline__ void radix_topk_row_f32_rescan(
    const float* input, int32_t* output, uint32_t length, uint32_t topk,
    uint32_t coarse_bin, uint32_t remain,
    uint32_t (&s_histogram_buf)[2][kRadix + 32], uint32_t& s_counter,
    int32_t& s_last_remain, uint32_t chunk_begin = 0)
{
    constexpr uint32_t BLOCK_SIZE = kBlockSize;
    const uint32_t tx = threadIdx.x;
    auto& s_histogram = s_histogram_buf[0];
    __shared__ uint32_t s_threshold_bin_id;

    uint32_t prefix_mask = 0, prefix_value = 0;
    #pragma unroll
    for (int round = 0; round < 4; ++round) {
        const int shift = 24 - round * 8;
        const bool is_last = (round == 3);
        const bool filtered = (prefix_mask != 0);

        for (uint32_t b = tx; b < kRadix + 1; b += BLOCK_SIZE) s_histogram[b] = 0;
        __syncthreads();
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float raw = __ldg(input + idx);
            if (float_to_uint8(raw) != coarse_bin) continue;
            const uint32_t key = refine_bin(raw);
            if (filtered && (key & prefix_mask) != prefix_value) continue;
            atomicAdd(&s_histogram[(key >> shift) & 0xFFu], 1u);
        }
        __syncthreads();
        run_cumsum(s_histogram_buf, tx);

        if (tx < kRadix && s_histogram[tx] > remain && s_histogram[tx + 1] <= remain) {
            s_threshold_bin_id = tx;
            s_last_remain = static_cast<int32_t>(remain - s_histogram[tx + 1]);
        }
        __syncthreads();
        const uint32_t pivot = s_threshold_bin_id;
        remain -= s_histogram[pivot + 1];

        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            const float raw = __ldg(input + idx);
            if (float_to_uint8(raw) != coarse_bin) continue;
            const uint32_t key = refine_bin(raw);
            if (filtered && (key & prefix_mask) != prefix_value) continue;
            const uint32_t bin = (key >> shift) & 0xFFu;
            if (bin > pivot) {
                output[atomicAdd(&s_counter, 1u)] =
                    static_cast<int32_t>(idx + chunk_begin);
            } else if (is_last && bin == pivot) {
                // The whole key is now known: any `remain` of the ties are
                // equally valid, and the tail slots take exactly that many.
                const int32_t p = atomicAdd(&s_last_remain, -1);
                if (p > 0)
                    output[topk - p] = static_cast<int32_t>(idx + chunk_begin);
            }
        }
        __syncthreads();
        if (is_last || remain == 0) break;
        prefix_mask |= (uint32_t)0xFFu << shift;
        prefix_value |= pivot << shift;
    }
}

// `chunk_begin` converts a position in *this* call's window to its column in
// the row.  It is 0 for the row entry, where the two are the same; the chunked
// fp32 split passes a chunk's column offset and gets the row columns the
// contract half needs.  It is applied only at the emit, which is the single
// point where an index becomes an answer -- the arena, the refine and the
// rescan all work on positions in the window either way.
//
// `kNan` folds the contract's NaN scan into pass 1's own load, and `row_nan`
// is where the answer lands (thread-uniform, written before the function's
// first return).  The two are one parameter's worth of design: a caller that
// wants the scan gets it for free off a load it was making anyway, and a caller
// that does not (`kNan = false`, the default, and every 16-bit caller) gets the
// shipped kernel with none of it.  A template rather than a runtime branch
// precisely so the false case stays free.
template <bool kNan = false>
__device__ __forceinline__ void radix_topk_row_f32(
    const float* input, int32_t* output, uint32_t length, uint32_t topk,
    uint32_t chunk_begin = 0, bool* row_nan = nullptr)
{
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t BLOCK_SIZE = kBlockSize;

    // smem 布局：直方图 + 候选索引
    // 静态变量: s_histogram_buf(2304) + s_counter/s_threshold_bin_id/
    //           s_high_threshold_bin_id (12) + s_num_input(8) + s_last_remain(4)
    // 动态区: arena 与 refine 的 ping-pong 暂存，各一半
    constexpr uint32_t STATIC_BYTES = kF32StaticBytes;
    constexpr uint32_t SMEM_INPUT_SIZE = kF32SmemInputSize;

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    // **Write-only in this function.**  Set at :650 alongside
    // `s_threshold_bin_id`, never read in `radix_topk_row_f32`, its rescan or
    // `stage_f32_lane` -- the fp32 path narrows with `s_histogram` directly.
    // The readers are the 16-bit row entries, which declare their own copy
    // (:1178, :1401).  Kept because the static-smem accounting above is
    // written against this declaration list: deleting it would move
    // `kF32StaticBytes` and with it `kF32SmemInputSize`.
    __shared__ uint32_t s_high_threshold_bin_id;
    __shared__ uint32_t s_num_input[2];
    __shared__ int32_t s_last_remain;
    extern __shared__ uint32_t s_input_flat[];

    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = topk;
    auto& s_histogram = s_histogram_buf[0];

    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    uint32_t vec_len = length / 4 * 4;
    // Four float4 per lane per leg, but *unit-strided* between the four: the
    // leg takes `tx, tx+BS, tx+2BS, tx+3BS` (float4 units) rather than
    // `4tx .. 4tx+3`.  Same bytes, same instruction count -- but each load
    // instruction's 64 lanes now sit 16 B apart instead of 64 B, which is the
    // difference between 998.7 and 1650.9 GB/s on pass 1's own shape (measured;
    // see the note above `hist_add_f32`).
    uint32_t n4 = vec_len / 4;
    uint32_t i4 = tx;
    bool nan_local = false;
    for (; i4 + 3u * BLOCK_SIZE < n4; i4 += 4u * BLOCK_SIZE) {
        nan_local |= hist_add_f32<kNan>(s_histogram, input, (i4 + 0u * BLOCK_SIZE) * 4u);
        nan_local |= hist_add_f32<kNan>(s_histogram, input, (i4 + 1u * BLOCK_SIZE) * 4u);
        nan_local |= hist_add_f32<kNan>(s_histogram, input, (i4 + 2u * BLOCK_SIZE) * 4u);
        nan_local |= hist_add_f32<kNan>(s_histogram, input, (i4 + 3u * BLOCK_SIZE) * 4u);
    }
    for (; i4 < n4; i4 += BLOCK_SIZE)
        nan_local |= hist_add_f32<kNan>(s_histogram, input, i4 * 4u);
    for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE) {
        const float raw = __ldg(input + idx);
        atomicAdd(&s_histogram[float_to_uint8(raw)], 1u);
        if (kNan) nan_local |= is_nan_bits32(__float_as_uint(raw));
    }
    __syncthreads();
    // Two things on the same barrier: the histogram is complete, and the CTA's
    // NaN vote is in.  `__syncthreads_or` is the same reduction
    // `nan_scan_kernel` runs, only across a block instead of a grid -- and it
    // is why the flag can be handed back as a thread-uniform bool.
    if (kNan && row_nan != nullptr)
        *row_nan = __syncthreads_or((int)nan_local) != 0;
    run_cumsum(s_histogram_buf, tx);

    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx; s_high_threshold_bin_id = tx; s_num_input[0] = 0; s_counter = 0;
    }
    __syncthreads();

    {
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t idx = tx * 4; idx < vec_len; idx += BLOCK_SIZE * 4) {
                const float4 v = __ldg(reinterpret_cast<const float4 *>(input + idx));
                if (float_to_uint8(v.x) > threshold_bin) output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + chunk_begin);
                if (float_to_uint8(v.y) > threshold_bin) output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + 1 + chunk_begin);
                if (float_to_uint8(v.z) > threshold_bin) output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + 2 + chunk_begin);
                if (float_to_uint8(v.w) > threshold_bin) output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + 3 + chunk_begin);
            }
            for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
                if (float_to_uint8(__ldg(input + idx)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + chunk_begin);
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (uint32_t idx = tx * 4; idx < vec_len; idx += BLOCK_SIZE * 4) {
            const float4 v = __ldg(reinterpret_cast<const float4 *>(input + idx));
            stage_f32_lane(v.x, idx,     threshold_bin, output, &s_counter,
                           s_input_flat, &s_num_input[0], s_histogram, SMEM_INPUT_SIZE, chunk_begin);
            stage_f32_lane(v.y, idx + 1, threshold_bin, output, &s_counter,
                           s_input_flat, &s_num_input[0], s_histogram, SMEM_INPUT_SIZE, chunk_begin);
            stage_f32_lane(v.z, idx + 2, threshold_bin, output, &s_counter,
                           s_input_flat, &s_num_input[0], s_histogram, SMEM_INPUT_SIZE, chunk_begin);
            stage_f32_lane(v.w, idx + 3, threshold_bin, output, &s_counter,
                           s_input_flat, &s_num_input[0], s_histogram, SMEM_INPUT_SIZE, chunk_begin);
        }
        for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
            stage_f32_lane(__ldg(input + idx), idx, threshold_bin, output,
                           &s_counter, s_input_flat, &s_num_input[0],
                           s_histogram, SMEM_INPUT_SIZE, chunk_begin);
        __syncthreads();
    }

    // A threshold bin that does not fit the arena is ranked over the row
    // instead of inside it; see `radix_topk_row_f32_rescan`.
    if (s_num_input[0] > SMEM_INPUT_SIZE) {
        radix_topk_row_f32_rescan(input, output, length, topk,
                                  s_threshold_bin_id, remain_topk,
                                  s_histogram_buf, s_counter, s_last_remain,
                                  chunk_begin);
        return;
    }

    #pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        const auto r_idx = round % 2;
        const auto r_off = r_idx * SMEM_INPUT_SIZE;
        const auto num = s_num_input[r_idx] < SMEM_INPUT_SIZE ? s_num_input[r_idx] : SMEM_INPUT_SIZE;
        run_cumsum(s_histogram_buf, tx);
        if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
            s_threshold_bin_id = tx; s_num_input[r_idx ^ 1] = 0;
            s_last_remain = static_cast<int32_t>(remain_topk - s_histogram[tx + 1]);
        }
        __syncthreads();
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
                auto idx = s_input_flat[r_off + i];
                if (((refine_bin(__ldg(input + idx)) >> (24 - round * 8)) & 0xFF) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] =
                        static_cast<int32_t>(idx + chunk_begin);
            }
            __syncthreads(); break;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        const auto r_off_next = (r_idx ^ 1) * SMEM_INPUT_SIZE;
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
            auto idx = s_input_flat[r_off + i];
            float raw = __ldg(input + idx);
            auto offset = 24 - round * 8;
            auto bin = (refine_bin(raw) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] =
                    static_cast<int32_t>(idx + chunk_begin);
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    auto p = atomicAdd(&s_last_remain, -1);
                    if (p > 0)
                        output[topk - p] = static_cast<int32_t>(idx + chunk_begin);
                } else {
                    uint32_t p = atomicAdd(&s_num_input[r_idx ^ 1], 1u);
                    if (p < SMEM_INPUT_SIZE) {
                        s_input_flat[r_off_next + p] = idx;
                        atomicAdd(&s_histogram[(refine_bin(raw) >> (offset - 8)) & 0xFF], 1u);
                    }
                }
            }
        }
        __syncthreads();
    }
}

// FP32 k=2048 path shaped after SGLang's original TopK kernel.
// This is intentionally narrow: it targets SGLang-shape correctness/perf
// comparison without changing the generic FP32/FP16 dispatch paths.
//
// **Dead in this tree, and kept on purpose.**  Nothing calls
// `launch_topk_f32_k2048_b1024_c500` -- the only entry that would launch the
// two kernels below -- so `radix_topk_row_f32_k2048_b1024` and its warp-scan
// twin are compiled but never dispatched.  This block was the first shape of
// the fp32 row and the generic `radix_topk_row_f32` (:538) replaced it; the
// refactor that removed the dispatch left the kernels, and the removal was
// deliberate rather than an oversight.  `ref/ds/README.md` lists both in its
// resource table and says the same thing, so this is the port's only
// *recorded* dead code.
//
// It is not removed because the question it exists to answer is not "is it
// called" but "was the generic row a regression against SGLang's shape" --
// and the answer lives in the two kernels' resource numbers, not in a call
// graph.  Deleting ~380 lines to save nothing measurable would also delete
// that.  A reader looking for what is *live* should start at
// `radix_topk_row_f32`; a reader changing `kBlockSize` should know these
// two carry `__launch_bounds__(kF32K2048BlockSize)` and will move with it.
constexpr int kF32K2048BlockSize = 1024;
constexpr int kF32K2048TopK = 2048;
#ifndef KF32_K2048_SMEM_BYTES
constexpr size_t kF32K2048Smem = 32 * 1024;
#else
constexpr size_t kF32K2048Smem = KF32_K2048_SMEM_BYTES;
#endif
#ifndef KF32_K2048_WARP_SCAN_MAX_LEN
constexpr int kF32K2048WarpScanMaxLen = 32768;
#else
constexpr int kF32K2048WarpScanMaxLen = KF32_K2048_WARP_SCAN_MAX_LEN;
#endif

__device__ __forceinline__ void radix_topk_row_f32_k2048_b1024(
    const float* __restrict__ input, int32_t* __restrict__ output, int row_start, int length)
{
    int remain_topk = kF32K2048TopK;
    constexpr int RADIX = 256;
    constexpr int SMEM_INPUT_SIZE = kF32K2048Smem / (2 * sizeof(int32_t));

    alignas(128) __shared__ int32_t s_histogram_buf[2][RADIX + 128];
    alignas(128) __shared__ int32_t s_counter;
    alignas(128) __shared__ int32_t s_threshold_bin_id;
    alignas(128) __shared__ int32_t s_num_input[2];

    auto& s_histogram = s_histogram_buf[0];
    extern __shared__ int32_t s_input_idx[][SMEM_INPUT_SIZE];

    const int tx = threadIdx.x;

    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += kF32K2048BlockSize) {
        const auto bin = float_to_uint8(input[idx + row_start]);
        atomicAdd(&s_histogram[bin], 1);
    }
    __syncthreads();

    const auto run_cumsum_i32 = [&] {
        #pragma unroll 8
        for (int i = 0; i < 8; ++i) {
            if (tx < RADIX) {
                const auto j = 1 << i;
                const auto k = i & 1;
                auto value = s_histogram_buf[k][tx];
                if (tx < RADIX - j) {
                    value += s_histogram_buf[k][tx + j];
                }
                s_histogram_buf[k ^ 1][tx] = value;
            }
            __syncthreads();
        }
    };

    run_cumsum_i32();
    if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
        s_threshold_bin_id = tx;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    const auto high_threshold_bin = s_threshold_bin_id;
    remain_topk -= s_histogram[high_threshold_bin + 1];

    if (remain_topk == 0) {
        for (int idx = tx; idx < length; idx += kF32K2048BlockSize) {
            const auto bin = static_cast<int>(float_to_uint8(input[idx + row_start]));
            if (bin > high_threshold_bin) {
                const auto pos = atomicAdd(&s_counter, 1);
                output[pos] = idx;
            }
        }
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (int idx = tx; idx < length; idx += kF32K2048BlockSize) {
        const auto raw_input = input[idx + row_start];
        const auto bin = static_cast<int>(float_to_uint8(raw_input));
        if (bin > high_threshold_bin) {
            const auto pos = atomicAdd(&s_counter, 1);
            output[pos] = idx;
        } else if (bin == high_threshold_bin) {
            const auto pos = atomicAdd(&s_num_input[0], 1);
            if (pos < SMEM_INPUT_SIZE) {
                s_input_idx[0][pos] = idx;
                const auto sub_bin = (refine_bin(raw_input) >> 24) & 0xFF;
                atomicAdd(&s_histogram[sub_bin], 1);
            }
        }
    }
    __syncthreads();

    #pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        __shared__ int32_t s_last_remain;
        const auto r_idx = round % 2;

        const auto raw_num_input = s_num_input[r_idx];
        const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        run_cumsum_i32();
        if (tx < RADIX && s_histogram[tx] > remain_topk && s_histogram[tx + 1] <= remain_topk) {
            s_threshold_bin_id = tx;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = remain_topk - s_histogram[tx + 1];
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];

        if (remain_topk == 0) {
            for (int i = tx; i < num_input; i += kF32K2048BlockSize) {
                const auto idx = s_input_idx[r_idx][i];
                const auto offset = 24 - round * 8;
                const auto bin = (refine_bin(input[idx + row_start]) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    const auto pos = atomicAdd(&s_counter, 1);
                    output[pos] = idx;
                }
            }
            __syncthreads();
            break;
        }

        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (int i = tx; i < num_input; i += kF32K2048BlockSize) {
            const auto idx = s_input_idx[r_idx][i];
            const auto raw_input = input[idx + row_start];
            const auto offset = 24 - round * 8;
            const auto bin = (refine_bin(raw_input) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                const auto pos = atomicAdd(&s_counter, 1);
                output[pos] = idx;
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    const auto pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        output[kF32K2048TopK - pos] = idx;
                    }
                } else {
                    const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1);
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_idx[r_idx ^ 1][pos] = idx;
                        const auto sub_bin = (refine_bin(raw_input) >> (offset - 8)) & 0xFF;
                        atomicAdd(&s_histogram[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

// The static-k2048 pair, carried by the artifact but routed to by nothing.
// Both are templates on their block size so the generated TU that owns their
// launch can hold the explicit instantiation, exactly as the row kernel is:
// a non-template `__global__` in a header is defined in every TU that
// includes it.
template <int kBlock>
__global__ __launch_bounds__(kBlock) void topk_f32_kernel_k2048_b1024(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int64_t score_stride, int B)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;
    const int32_t length = lengths[bid];
    const int32_t row_start = row_starts ? row_starts[bid] : 0;
    const float* row = scores + static_cast<int64_t>(bid) * score_stride;
    int32_t* out = indices + static_cast<int64_t>(bid) * kF32K2048TopK;

    if (length <= kF32K2048TopK) {
        for (int i = threadIdx.x; i < kF32K2048TopK; i += kBlock)
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1;
        return;
    }

    radix_topk_row_f32_k2048_b1024(row, out, row_start, length);
}

template <int kBlock>
inline cudaError_t launch_f32_k2048_b1024(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_kernel_k2048_b1024<kBlock>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kF32K2048Smem);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_f32_kernel_k2048_b1024<kBlock><<<B, (int)kBlock, kF32K2048Smem, stream>>>(
        scores, lengths, row_starts, indices, L, B);
    return cudaGetLastError();
}

extern template cudaError_t launch_f32_k2048_b1024<kF32K2048BlockSize>(
    const float*, const int32_t*, const int32_t*, int32_t*, int, int, cudaStream_t);

__device__ __forceinline__ void radix_topk_row_f32_k2048_b1024_warp_scan(
    const float* __restrict__ input, int32_t* __restrict__ output, int row_start, int length)
{
    uint32_t remain_topk = kF32K2048TopK;
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t SMEM_INPUT_SIZE = kF32K2048Smem / (2 * sizeof(uint32_t));

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    __shared__ uint32_t s_threshold_exclusive_count;
    __shared__ uint32_t s_num_input[2];
    __shared__ int32_t s_last_remain;
    extern __shared__ uint32_t s_input_flat[];

    const uint32_t tx = threadIdx.x;
    auto& s_histogram = s_histogram_buf[0];

    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (uint32_t idx = tx; idx < static_cast<uint32_t>(length); idx += kF32K2048BlockSize) {
        atomicAdd(&s_histogram[float_to_uint8(__ldg(input + row_start + idx))], 1u);
    }
    __syncthreads();

    uint32_t exclusive_suffix = 0;
    uint32_t inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);
    if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
        s_threshold_bin_id = tx;
        s_threshold_exclusive_count = exclusive_suffix;
        s_num_input[0] = 0;
        s_counter = 0;
    }
    __syncthreads();

    const auto high_threshold_bin = s_threshold_bin_id;
    remain_topk -= s_threshold_exclusive_count;

    if (remain_topk == 0) {
        for (uint32_t idx = tx; idx < static_cast<uint32_t>(length); idx += kF32K2048BlockSize) {
            if (float_to_uint8(__ldg(input + row_start + idx)) > high_threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
        }
        __syncthreads();
        return;
    }

    __syncthreads();
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    for (uint32_t idx = tx; idx < static_cast<uint32_t>(length); idx += kF32K2048BlockSize) {
        const float raw = __ldg(input + row_start + idx);
        const uint32_t bin = float_to_uint8(raw);
        if (bin > high_threshold_bin) {
            output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
        } else if (bin == high_threshold_bin) {
            const uint32_t pos = atomicAdd(&s_num_input[0], 1u);
            if (pos < SMEM_INPUT_SIZE) {
                s_input_flat[pos] = idx;
                atomicAdd(&s_histogram[(refine_bin(raw) >> 24) & 0xFF], 1u);
            }
        }
    }
    __syncthreads();

    #pragma unroll 4
    for (int round = 0; round < 4; ++round) {
        const auto r_idx = round % 2;
        const auto r_off = r_idx * SMEM_INPUT_SIZE;
        const auto r_off_next = (r_idx ^ 1) * SMEM_INPUT_SIZE;
        const auto raw_num_input = s_num_input[r_idx];
        const auto num_input = raw_num_input < SMEM_INPUT_SIZE ? raw_num_input : SMEM_INPUT_SIZE;

        exclusive_suffix = 0;
        inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);
        if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
            s_threshold_bin_id = tx;
            s_threshold_exclusive_count = exclusive_suffix;
            s_num_input[r_idx ^ 1] = 0;
            s_last_remain = static_cast<int32_t>(remain_topk - exclusive_suffix);
        }
        __syncthreads();

        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_threshold_exclusive_count;

        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num_input; i += kF32K2048BlockSize) {
                const auto idx = s_input_flat[r_off + i];
                const auto offset = 24 - round * 8;
                const auto bin = (refine_bin(__ldg(input + row_start + idx)) >> offset) & 0xFF;
                if (bin > threshold_bin) {
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                }
            }
            __syncthreads();
            break;
        }

        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (uint32_t i = tx; i < num_input; i += kF32K2048BlockSize) {
            const auto idx = s_input_flat[r_off + i];
            const float raw = __ldg(input + row_start + idx);
            const auto offset = 24 - round * 8;
            const auto bin = (refine_bin(raw) >> offset) & 0xFF;
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    const auto pos = atomicAdd(&s_last_remain, -1);
                    if (pos > 0) {
                        output[kF32K2048TopK - pos] = static_cast<int32_t>(idx);
                    }
                } else {
                    const auto pos = atomicAdd(&s_num_input[r_idx ^ 1], 1u);
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_flat[r_off_next + pos] = idx;
                        atomicAdd(&s_histogram[(refine_bin(raw) >> (offset - 8)) & 0xFF], 1u);
                    }
                }
            }
        }
        __syncthreads();
    }
}

template <int kBlock>
__global__ __launch_bounds__(kBlock) void topk_f32_kernel_k2048_b1024_warp_scan(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int64_t score_stride, int B)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;
    const int32_t length = lengths[bid];
    const int32_t row_start = row_starts ? row_starts[bid] : 0;
    const float* row = scores + static_cast<int64_t>(bid) * score_stride;
    int32_t* out = indices + static_cast<int64_t>(bid) * kF32K2048TopK;

    if (length <= kF32K2048TopK) {
        for (int i = threadIdx.x; i < kF32K2048TopK; i += kBlock)
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1;
        return;
    }

    radix_topk_row_f32_k2048_b1024_warp_scan(row, out, row_start, length);
}

template <int kBlock>
inline cudaError_t launch_f32_k2048_b1024_warp_scan(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_kernel_k2048_b1024_warp_scan<kBlock>,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kF32K2048Smem);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_f32_kernel_k2048_b1024_warp_scan<kBlock>
        <<<B, (int)kBlock, kF32K2048Smem, stream>>>(
            scores, lengths, row_starts, indices, L, B);
    return cudaGetLastError();
}

extern template cudaError_t launch_f32_k2048_b1024_warp_scan<kF32K2048BlockSize>(
    const float*, const int32_t*, const int32_t*, int32_t*, int, int, cudaStream_t);

inline cudaError_t launch_topk_f32_k2048_b1024_c500(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int score_stride, int common_length, cudaStream_t stream)
{
    if (common_length > 0 && common_length <= kF32K2048WarpScanMaxLen) {
        return launch_f32_k2048_b1024_warp_scan<kF32K2048BlockSize>(
            scores, lengths, row_starts, indices, B, score_stride, stream);
    }
    return launch_f32_k2048_b1024<kF32K2048BlockSize>(
        scores, lengths, row_starts, indices, B, score_stride, stream);
}

}  // namespace topk_select_radix

#endif  // RADIX_ROW_FP32_CUH
