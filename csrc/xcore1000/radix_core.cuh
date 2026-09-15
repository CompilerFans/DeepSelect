// ── provenance ──────────────────────────────────────────────────────────────
// Ported from the standalone C500 radix-TopK project (`dsa_topk`,
// `csrc/radix_topk.cuh`, commit 61ab380c77b81669718bfb11b95a583b0e661001).
// Two passes over the row: one histogram, one collect-and-stage that refines
// the threshold bin in shared memory.
//
// Three deviations from that source:
//
//   1. The 16-bit path speaks bf16, not fp16.  DeepSelect's dtypes are bf16
//      and fp32, so `half_to_uintN` became `bf16_to_uintN`, reading the
//      value's own bit pattern (`__bfloat16_as_ushort`) instead of fp16's.
//      The key transform is otherwise unchanged -- it only needs the sign bit
//      at bit 15, which both formats have.  This is NOT the same as
//      `float_to_uint8`, which rounds fp32 through fp16 before binning and so
//      has no bf16 counterpart.
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
 * 使用方式: #include "radix_core.cuh"
 */

#ifndef RADIX_TOPK_CUH
#define RADIX_TOPK_CUH

#include <cuda_runtime.h>
#include <cuda_fp16.h>       // float_to_uint8: the fp32 path's fp16 rounding
#include <maca_bfloat16.h>   // the 16-bit path's value type
#include <cstdint>

namespace rk {

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
// smem 大小：可通过编译参数 -DKSMEM_BYTES=32768 覆盖
// RTX 4080: 48KB (100KB/SM, 选48KB放2 block/SM)
// MetaX C500: 32KB (高延迟，选32KB增加并行度)
#ifndef KSMEM_BYTES
#ifdef __MACACC__
constexpr size_t kSMEM = 16 * 1024;
#else
constexpr size_t kSMEM = 48 * 1024;
#endif
#else
constexpr size_t kSMEM = KSMEM_BYTES;
#endif

#ifndef KTOPK_COMPACT_BF16_INDICES
#define KTOPK_COMPACT_BF16_INDICES 1
#endif

// Compact indices are only dispatched for this measured 16-bit-safe window.
#ifndef KCOMPACT_BF16_MAX_LENGTH
#define KCOMPACT_BF16_MAX_LENGTH 65535
#endif
static_assert(KCOMPACT_BF16_MAX_LENGTH <= 65535,
              "compact FP16 indices require a uint16-safe length");
constexpr uint32_t kCompactBF16MaxLength = KCOMPACT_BF16_MAX_LENGTH;

// ============================================================
// 类型转换
// ============================================================

// 与 SGLang 原始实现一致：先转 half 再取高 8 位
__device__ __forceinline__ uint8_t float_to_uint8(float x) {
    __half h = __float2half_rn(x);
    uint16_t bits = __half_as_ushort(h);
    uint16_t key = (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
    return static_cast<uint8_t>(key >> 8);
}

__device__ __forceinline__ uint32_t float_to_uint32(float x) {
    uint32_t bits = __float_as_uint(x);
    return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ uint16_t bf16_to_uint16(maca_bfloat16 x) {
    uint16_t bits = __bfloat16_as_ushort(x);
#ifdef __MACACC__
    // C500: express the ordered-key transform without a divergent sign
    // branch.  Positive keys xor 0x8000; negative keys xor 0xffff.
    const uint32_t sign = static_cast<uint32_t>(bits >> 15);
    const uint16_t mask = static_cast<uint16_t>(0x8000u | (0u - sign));
    return static_cast<uint16_t>(bits ^ mask);
#else
    return (bits & 0x8000) ? static_cast<uint16_t>(~bits) : static_cast<uint16_t>(bits | 0x8000);
#endif
}

__device__ __forceinline__ uint8_t bf16_to_uint8(maca_bfloat16 x) {
    return static_cast<uint8_t>(bf16_to_uint16(x) >> 8);
}


// ============================================================
// 向量化直方图 (smem atomicAdd)
// ============================================================

// `hist_add_f32`'s call sites walk `input + tx*4`: one thread takes four
// CONSECUTIVE float4 per leg, so adjacent lanes of a single load instruction
// are 64 bytes apart and a 128-byte segment carries 32 useful bytes.
//
// That stride is real and deliberate -- but it is NOT why pass 1 is short of
// the wall.  Coalescing this walk is a 65% win in a microbench and -0.7% in
// the kernel: at the kernel's occupancy the two patterns are 7% apart, not
// 65%, because pass 1 issues one shared-bucket atomic per element and that is
// what it is waiting on.  See `docs/C500-radix-perf-ledger.zh.md` before
// changing it.
__device__ __forceinline__ void hist_add_f32(
    uint32_t* s_histogram, const float* input, uint32_t idx)
{
    float4 v = __ldg(reinterpret_cast<const float4*>(input + idx));
    atomicAdd(&s_histogram[float_to_uint8(v.x)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.y)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.z)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.w)], 1u);
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
            atomicAdd(&s_histogram[(float_to_uint32(raw) >> 24) & 0xFFu], 1u);
        }
    }
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
 * NOT IMPLEMENTED.  The bodies below move one element per shuffle, so each
 * warp records about 1/64 of what it reads.  They are kept as the shape of the
 * idea, not as a working histogram; see `docs/C500-radix-handover.zh.md` §9.
 */

#ifdef __MACACC__
constexpr int kWarpSize = 64;
#else
constexpr int kWarpSize = 32;
#endif
constexpr int kBinsPerThread = 256 / kWarpSize;  // 8 (warp=32) or 4 (warp=64)
static_assert(kBlockSize >= 256 && kBlockSize % kWarpSize == 0,
              "kBlockSize must cover the 256-bin scan and whole wavefronts");
static_assert(kChunkBlockSize >= 256 && kChunkBlockSize % kWarpSize == 0,
              "kChunkBlockSize must cover the 256-bin scan and whole wavefronts");
constexpr int kLongRowBlockSize = 1024;
static_assert(kLongRowBlockSize >= 256 && kLongRowBlockSize <= 1024 &&
                  kLongRowBlockSize % kWarpSize == 0,
              "kLongRowBlockSize must cover the 256-bin scan and whole wavefronts");

constexpr uint32_t kRadix = 256;
// Static smem: histogram_buf[2][256+32] + s_counter + s_threshold_bin_id + s_high_threshold_bin_id + s_num_input[2] + s_last_remain
constexpr uint32_t kSmemStaticBytes = 2 * (kRadix + 32) * sizeof(uint32_t)
                                    + sizeof(uint32_t) + sizeof(uint32_t)
                                    + sizeof(uint32_t) + 2 * sizeof(uint32_t) + sizeof(int32_t);
static_assert(kSMEM >= kSmemStaticBytes, "KSMEM_BYTES is smaller than static shared state");
constexpr uint32_t kSmemInputSize = (kSMEM - kSmemStaticBytes) / sizeof(int32_t);

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

// 12 位粗层：一次 uint4 搬 8 个元素，每个元素一个原子加。
__device__ __forceinline__ void hist_add_bf16_wide(
    uint32_t* s_wide, const maca_bfloat16* input, uint32_t idx)
{
    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
    #pragma unroll
    for (int i = 0; i < 8; i++)
        atomicAdd(&s_wide[bf16_to_uint16(h[i]) >> kCoarse12Shift], 1u);
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

/*
 * run_cumsum_warp: 256-element reverse inclusive scan, one buffer, warp
 * shuffles.  Each warp scans its own 32 elements, the warp totals go to
 * s_histogram_buf[0][257..264] (256 stays a sentinel), warp 0 scans those
 * eight, and each warp adds its offset -- two __syncthreads, against eight in
 * the naive form.  Returns the calling thread's bin suffix so the caller can
 * avoid reading a neighbouring bin before another barrier.
 *
 * fp32 uses this; the 16-bit path uses run_cumsum.
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
            const uint32_t key = float_to_uint32(raw);
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
            const uint32_t key = float_to_uint32(raw);
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
__device__ __forceinline__ void radix_topk_row_f32(
    const float* input, int32_t* output, uint32_t length, uint32_t topk,
    uint32_t chunk_begin = 0)
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
    for (; i4 + 3u * BLOCK_SIZE < n4; i4 += 4u * BLOCK_SIZE) {
        hist_add_f32(s_histogram, input, (i4 + 0u * BLOCK_SIZE) * 4u);
        hist_add_f32(s_histogram, input, (i4 + 1u * BLOCK_SIZE) * 4u);
        hist_add_f32(s_histogram, input, (i4 + 2u * BLOCK_SIZE) * 4u);
        hist_add_f32(s_histogram, input, (i4 + 3u * BLOCK_SIZE) * 4u);
    }
    for (; i4 < n4; i4 += BLOCK_SIZE)
        hist_add_f32(s_histogram, input, i4 * 4u);
    for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
        atomicAdd(&s_histogram[float_to_uint8(__ldg(input + idx))], 1u);
    __syncthreads();
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
                if (((float_to_uint32(__ldg(input + idx)) >> (24 - round * 8)) & 0xFF) > threshold_bin)
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
            auto bin = (float_to_uint32(raw) >> offset) & 0xFF;
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
                        atomicAdd(&s_histogram[(float_to_uint32(raw) >> (offset - 8)) & 0xFF], 1u);
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
                const auto sub_bin = (float_to_uint32(raw_input) >> 24) & 0xFF;
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
                const auto bin = (float_to_uint32(input[idx + row_start]) >> offset) & 0xFF;
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
            const auto bin = (float_to_uint32(raw_input) >> offset) & 0xFF;
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
                        const auto sub_bin = (float_to_uint32(raw_input) >> (offset - 8)) & 0xFF;
                        atomicAdd(&s_histogram[sub_bin], 1);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kF32K2048BlockSize) void topk_f32_kernel_k2048_b1024(
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
        for (int i = threadIdx.x; i < kF32K2048TopK; i += kF32K2048BlockSize)
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1;
        return;
    }

    radix_topk_row_f32_k2048_b1024(row, out, row_start, length);
}

inline cudaError_t launch_topk_f32_k2048_b1024(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_kernel_k2048_b1024,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kF32K2048Smem);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_f32_kernel_k2048_b1024<<<B, kF32K2048BlockSize, kF32K2048Smem, stream>>>(
        scores, lengths, row_starts, indices, L, B);
    return cudaGetLastError();
}

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
                atomicAdd(&s_histogram[(float_to_uint32(raw) >> 24) & 0xFF], 1u);
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
                const auto bin = (float_to_uint32(__ldg(input + row_start + idx)) >> offset) & 0xFF;
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
            const auto bin = (float_to_uint32(raw) >> offset) & 0xFF;
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
                        atomicAdd(&s_histogram[(float_to_uint32(raw) >> (offset - 8)) & 0xFF], 1u);
                    }
                }
            }
        }
        __syncthreads();
    }
}

__global__ __launch_bounds__(kF32K2048BlockSize) void topk_f32_kernel_k2048_b1024_warp_scan(
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
        for (int i = threadIdx.x; i < kF32K2048TopK; i += kF32K2048BlockSize)
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1;
        return;
    }

    radix_topk_row_f32_k2048_b1024_warp_scan(row, out, row_start, length);
}

inline cudaError_t launch_topk_f32_k2048_b1024_warp_scan(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_kernel_k2048_b1024_warp_scan,
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            kF32K2048Smem);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_f32_kernel_k2048_b1024_warp_scan<<<B, kF32K2048BlockSize, kF32K2048Smem, stream>>>(
        scores, lengths, row_starts, indices, L, B);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_f32_k2048_b1024_c500(
    const float* scores, const int32_t* lengths, const int32_t* row_starts,
    int32_t* indices, int B, int score_stride, int common_length, cudaStream_t stream)
{
    if (common_length > 0 && common_length <= kF32K2048WarpScanMaxLen) {
        return launch_topk_f32_k2048_b1024_warp_scan(
            scores, lengths, row_starts, indices, B, score_stride, stream);
    }
    return launch_topk_f32_k2048_b1024(
        scores, lengths, row_starts, indices, B, score_stride, stream);
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

template <uint32_t BLOCK_SIZE>
__device__ __forceinline__ void radix_topk_row_bf16_b(
    const maca_bfloat16* input, int32_t* output, uint32_t length, uint32_t topk)
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
    if (input_aligned) {
        for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8)
            hist_add_bf16_wide(s_wide, input, idx);
        for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
            atomicAdd(&s_wide[bf16_to_uint16(__ldg(input + idx)) >> kCoarse12Shift], 1u);
    } else {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
            atomicAdd(&s_wide[bf16_to_uint16(__ldg(input + idx)) >> kCoarse12Shift], 1u);
    }
    __syncthreads();

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
    }
    __syncthreads();

    {
        const auto threshold_bin = s_high_threshold_bin_id;
        remain_topk -= s_wide_above;
        if (remain_topk == 0) {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                if ((bf16_to_uint16(__ldg(input + idx)) >> kCoarse12Shift) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        if (input_aligned) {
            for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    maca_bfloat16 raw = h[i];
                    uint32_t bin = bf16_to_uint16(raw) >> kCoarse12Shift;
                    if (bin > threshold_bin) {
                        output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + i);
                    } else if (bin == threshold_bin) {
                        // The count is over the bin, the staging is over the
                        // arena.  Gating both on the arena made the refine's
                        // histogram partial whenever the bin overflowed, which
                        // is what the full-row rebuild below used to repair.
                        uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                        if (pos < SMEM_INPUT_SIZE) s_input_flat[pos] = idx + i;
                        atomicAdd(&s_histogram[bf16_to_uint16(raw) & (kCoarse12SubBins - 1u)], 1u);
                    }
                }
            }
            for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE) {
                maca_bfloat16 raw = __ldg(input + idx);
                uint32_t bin = bf16_to_uint16(raw) >> kCoarse12Shift;
                if (bin > threshold_bin) {
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                } else if (bin == threshold_bin) {
                    uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                    if (pos < SMEM_INPUT_SIZE) s_input_flat[pos] = idx;
                    atomicAdd(&s_histogram[bf16_to_uint16(raw) & (kCoarse12SubBins - 1u)], 1u);
                }
            }
        } else {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
                maca_bfloat16 raw = __ldg(input + idx);
                uint32_t bin = bf16_to_uint16(raw) >> kCoarse12Shift;
                if (bin > threshold_bin) {
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                } else if (bin == threshold_bin) {
                    uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                    if (pos < SMEM_INPUT_SIZE) s_input_flat[pos] = idx;
                    atomicAdd(&s_histogram[bf16_to_uint16(raw) & (kCoarse12SubBins - 1u)], 1u);
                }
            }
        }
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
            const auto high_threshold_bin = s_high_threshold_bin_id;
            if (input_aligned) {
                for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                    uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                    #pragma unroll
                    for (int i = 0; i < 8; i++)
                        overflow_emit_member(idx + i, bf16_to_uint16(h[i]),
                                             high_threshold_bin, threshold_bin,
                                             remain_topk, topk, output,
                                             &s_counter, &s_last_remain);
                }
                for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
                    overflow_emit_member(idx, bf16_to_uint16(__ldg(input + idx)),
                                         high_threshold_bin, threshold_bin,
                                         remain_topk, topk, output, &s_counter,
                                         &s_last_remain);
            } else {
                for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                    overflow_emit_member(idx, bf16_to_uint16(__ldg(input + idx)),
                                         high_threshold_bin, threshold_bin,
                                         remain_topk, topk, output, &s_counter,
                                         &s_last_remain);
            }
            __syncthreads(); return;
        }
        const auto num = s_num_input[0];
        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
                auto idx = s_input_flat[i];
                if ((bf16_to_uint16(__ldg(input + idx)) & (kCoarse12SubBins - 1u)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
            __syncthreads(); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
            auto idx = s_input_flat[i];
            uint32_t bin = bf16_to_uint16(__ldg(input + idx)) & (kCoarse12SubBins - 1u);
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

__device__ __forceinline__ void radix_topk_row_bf16(
    const maca_bfloat16* input, int32_t* output, uint32_t length, uint32_t topk)
{
    radix_topk_row_bf16_b<kBlockSize>(input, output, length, topk);
}

template <uint32_t TOPK, uint32_t BLOCK_SIZE = kBlockSize, bool USE_COMPACT_INDICES = false>
__device__ __forceinline__ void radix_topk_row_bf16_k(
    const maca_bfloat16* input, int32_t* output, uint32_t length)
{
    static_assert(TOPK <= kMaxTopK, "TOPK exceeds kMaxTopK");
    constexpr uint32_t RADIX = kRadix;
    constexpr uint32_t SMEM_INPUT_SIZE = kSmemInputSize;

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    __shared__ uint32_t s_high_threshold_bin_id;
    __shared__ uint32_t s_num_input[2];
    __shared__ int32_t s_last_remain;
    extern __shared__ uint32_t s_input_flat[];

    const uint32_t tx = threadIdx.x;
    constexpr bool use_compact_indices =
        KTOPK_COMPACT_BF16_INDICES && USE_COMPACT_INDICES;
    constexpr uint32_t input_capacity =
        use_compact_indices ? SMEM_INPUT_SIZE * 2 : SMEM_INPUT_SIZE;
    uint16_t* const s_input_compact = reinterpret_cast<uint16_t*>(s_input_flat);
    uint32_t remain_topk = TOPK;
    auto& s_histogram = s_histogram_buf[0];

    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();

    // C500 has an intermittent race in the mixed vector-plus-tail path for
    // 1024-thread blocks. Use one uniform load mode for odd-length rows.
    const bool input_aligned = (length & 7u) == 0 && bf16x8_is_aligned(input);
    uint32_t vec_len = length / 8 * 8;
    if (input_aligned) {
        for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8)
            hist_add_bf16_aligned(s_histogram, input, idx);
        for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE)
            atomicAdd(&s_histogram[bf16_to_uint8(__ldg(input + idx))], 1u);
    } else {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
            atomicAdd(&s_histogram[bf16_to_uint8(__ldg(input + idx))], 1u);
    }
    __syncthreads();
    uint32_t exclusive_suffix = 0;
    uint32_t inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);

    if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
        s_threshold_bin_id = tx; s_high_threshold_bin_id = tx; s_num_input[0] = 0; s_counter = 0;
    }
    __syncthreads();

    {
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                if (bf16_to_uint8(__ldg(input + idx)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        if (input_aligned) {
            for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                #pragma unroll
                for (int i = 0; i < 8; i++) {
                    maca_bfloat16 raw = h[i];
                    uint32_t bin = bf16_to_uint8(raw);
                    if (bin > threshold_bin) {
                        output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx + i);
                    } else if (bin == threshold_bin) {
                        uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                        if (pos < input_capacity) {
                            if (use_compact_indices) s_input_compact[pos] = static_cast<uint16_t>(idx + i);
                            else s_input_flat[pos] = idx + i;
                            atomicAdd(&s_histogram[bf16_to_uint16(raw) & 0xFF], 1u);
                        }
                    }
                }
            }
            for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE) {
                maca_bfloat16 raw = __ldg(input + idx);
                uint32_t bin = bf16_to_uint8(raw);
                if (bin > threshold_bin) {
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                } else if (bin == threshold_bin) {
                    uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                    if (pos < input_capacity) {
                        if (use_compact_indices) s_input_compact[pos] = static_cast<uint16_t>(idx);
                        else s_input_flat[pos] = idx;
                        atomicAdd(&s_histogram[bf16_to_uint16(raw) & 0xFF], 1u);
                    }
                }
            }
        } else {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
                maca_bfloat16 raw = __ldg(input + idx);
                uint32_t bin = bf16_to_uint8(raw);
                if (bin > threshold_bin) {
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                } else if (bin == threshold_bin) {
                    uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                    if (pos < input_capacity) {
                        if (use_compact_indices) s_input_compact[pos] = static_cast<uint16_t>(idx);
                        else s_input_flat[pos] = idx;
                        atomicAdd(&s_histogram[bf16_to_uint16(raw) & 0xFF], 1u);
                    }
                }
            }
        }
        __syncthreads();
    }

    {
        const bool overflow = s_num_input[0] > input_capacity;
        if (overflow) {
            if (tx < RADIX + 1) s_histogram[tx] = 0;
            __syncthreads();
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
                const maca_bfloat16 raw = __ldg(input + idx);
                const uint32_t key = bf16_to_uint16(raw);
                if ((key >> 8) == s_high_threshold_bin_id)
                    atomicAdd(&s_histogram[key & 0xFF], 1u);
            }
            __syncthreads();
        }
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
            const auto high_threshold_bin = s_high_threshold_bin_id;
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
                const maca_bfloat16 raw = __ldg(input + idx);
                const uint32_t key = bf16_to_uint16(raw);
                if ((key >> 8) == high_threshold_bin) {
                    const uint32_t low = key & 0xFF;
                    if (low > threshold_bin) {
                        output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
                    } else if (low == threshold_bin && remain_topk != 0) {
                        auto p = atomicAdd(&s_last_remain, -1);
                        if (p > 0) output[TOPK - p] = static_cast<int32_t>(idx);
                    }
                }
            }
            __syncthreads(); return;
        }
        const auto num = s_num_input[0];
        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
                const uint32_t idx = use_compact_indices
                    ? static_cast<uint32_t>(s_input_compact[i]) : s_input_flat[i];
                if ((bf16_to_uint16(__ldg(input + idx)) & 0xFF) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
            __syncthreads(); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
            const uint32_t idx = use_compact_indices
                ? static_cast<uint32_t>(s_input_compact[i]) : s_input_flat[i];
            uint32_t bin = bf16_to_uint16(__ldg(input + idx)) & 0xFF;
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                auto p = atomicAdd(&s_last_remain, -1);
                if (p > 0) output[TOPK - p] = static_cast<int32_t>(idx);
            }
        }
        __syncthreads();
    }
}

// ============================================================
// FP16 radix topk row (寄存器直方图版本 - 后合并模式)
// ============================================================

__device__ __forceinline__ void radix_topk_row_bf16_reg(
    const maca_bfloat16* input, int32_t* output, uint32_t length, uint32_t topk)
{
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t BLOCK_SIZE = kBlockSize;

    constexpr uint32_t STATIC_BYTES = 2 * (RADIX + 32) * sizeof(uint32_t)
                                    + sizeof(uint32_t) + sizeof(uint32_t)
                                    + 2 * sizeof(uint32_t) + sizeof(int32_t);
    constexpr uint32_t SMEM_INPUT_SIZE = (kSMEM - STATIC_BYTES) / (2 * sizeof(int32_t));

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    __shared__ uint32_t s_num_input[2];
    __shared__ int32_t s_last_remain;
    extern __shared__ uint32_t s_input_flat[];

    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx % kWarpSize;
    const uint32_t warp_id = tx / kWarpSize;
    uint32_t remain_topk = topk;
    auto& s_histogram = s_histogram_buf[0];

    // Stage 1: 本地累积寄存器直方图 (无 shuffle)
    // 每个线程只累积自己拥有的 bins: [my_bin_start, my_bin_start + kBinsPerThread)
    const uint32_t my_bin_start = tx * kBinsPerThread;  // 注意: 用 tx 而非 lane，避免跨 warp 冲突
    uint32_t r_hist[kBinsPerThread] = {0};

    // Keep the register variant consistent with the production row paths.
    const bool input_aligned = (length & 7u) == 0 && bf16x8_is_aligned(input);
    uint32_t vec_len = length / 8 * 8;
    if (input_aligned) {
        for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
            uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
            const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
            #pragma unroll
            for (int i = 0; i < 8; i++) {
                uint32_t bin = bf16_to_uint8(h[i]);
                if (bin >= my_bin_start && bin < my_bin_start + kBinsPerThread)
                    r_hist[bin - my_bin_start]++;
            }
        }
        for (uint32_t idx = vec_len + tx; idx < length; idx += BLOCK_SIZE) {
            uint32_t bin = bf16_to_uint8(__ldg(input + idx));
            if (bin >= my_bin_start && bin < my_bin_start + kBinsPerThread)
                r_hist[bin - my_bin_start]++;
        }
    } else {
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            uint32_t bin = bf16_to_uint8(__ldg(input + idx));
            if (bin >= my_bin_start && bin < my_bin_start + kBinsPerThread)
                r_hist[bin - my_bin_start]++;
        }
    }

    // Stage 1.5: 将寄存器直方图合并到 smem
    if (tx < RADIX + 1) s_histogram[tx] = 0;
    __syncthreads();
    for (uint32_t b = 0; b < kBinsPerThread; b++) {
        uint32_t bin = lane * kBinsPerThread + b;
        if (bin < RADIX && r_hist[b] > 0)
            atomicAdd(&s_histogram[bin], r_hist[b]);
    }
    __syncthreads();
    uint32_t exclusive_suffix = 0;
    uint32_t inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);

    if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
        s_threshold_bin_id = tx; s_num_input[0] = 0; s_counter = 0;
    }
    __syncthreads();

    // Stage 1.5 + Stage 2: 与标准版本相同
    {
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                if (bf16_to_uint8(__ldg(input + idx)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            maca_bfloat16 raw = __ldg(input + idx);
            uint32_t bin = bf16_to_uint8(raw);
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                if (pos < SMEM_INPUT_SIZE) {
                    s_input_flat[pos] = idx;
                    atomicAdd(&s_histogram[bf16_to_uint16(raw) & 0xFF], 1u);
                }
            }
        }
        __syncthreads();
    }

    {
        const auto num = s_num_input[0] < SMEM_INPUT_SIZE ? s_num_input[0] : SMEM_INPUT_SIZE;
        exclusive_suffix = 0;
        inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);
        if (tx < RADIX && inclusive_suffix > remain_topk && exclusive_suffix <= remain_topk) {
            s_threshold_bin_id = tx;
            s_last_remain = static_cast<int32_t>(remain_topk - exclusive_suffix);
        }
        __syncthreads();
        const auto threshold_bin = s_threshold_bin_id;
        remain_topk -= s_histogram[threshold_bin + 1];
        if (remain_topk == 0) {
            for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
                auto idx = s_input_flat[i];
                if ((bf16_to_uint16(__ldg(input + idx)) & 0xFF) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            }
            __syncthreads(); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {
            auto idx = s_input_flat[i];
            uint32_t bin = bf16_to_uint16(__ldg(input + idx)) & 0xFF;
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

}  // namespace rk

namespace rk {

__global__ void topk_bf16_kernel_runtime(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int64_t score_stride, int topk, int B)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;
    const int32_t length = lengths[bid];
    const maca_bfloat16* row = scores + bid * score_stride;
    int32_t* out = indices + bid * topk;
    if (length <= topk) {
        for (int i = threadIdx.x; i < topk; i += kBlockSize)
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1;
        return;
    }
    radix_topk_row_bf16(row, out, length, topk);
}

#define RK_DEFINE_TOPK_BF16_KERNEL(NAME, TOPK_VALUE, COMPACT_VALUE) \
__global__ void NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int64_t score_stride, int topk, int B) \
{ \
    (void)topk; \
    constexpr uint32_t TOPK = TOPK_VALUE; \
    const int bid = blockIdx.x; \
    if (bid >= B) return; \
    const int32_t length = lengths[bid]; \
    const maca_bfloat16* row = scores + bid * score_stride; \
    int32_t* out = indices + bid * TOPK; \
    if (length <= static_cast<int32_t>(TOPK)) { \
        for (int i = threadIdx.x; i < static_cast<int>(TOPK); i += kBlockSize) \
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1; \
        return; \
    } \
    radix_topk_row_bf16_k<TOPK, kBlockSize, COMPACT_VALUE>(row, out, length); \
}

RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k50, 50, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k100, 100, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k512, 512, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k1024, 1024, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k1536, 1536, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k2028, 2028, false)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k2048, 2048, false)

RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k50_compact, 50, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k100_compact, 100, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k512_compact, 512, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k1024_compact, 1024, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k1536_compact, 1536, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k2028_compact, 2028, true)
RK_DEFINE_TOPK_BF16_KERNEL(topk_bf16_kernel_k2048_compact, 2048, true)

#undef RK_DEFINE_TOPK_BF16_KERNEL

// Long-row variants are intentionally narrow: the dispatch below only uses
// k=512/1024, where the measured 1024-thread benefit outweighs the extra
// launch footprint. All other top-k values keep the regular specialization.
#define RK_DEFINE_TOPK_BF16_LONG_KERNEL(NAME, TOPK_VALUE, COMPACT_VALUE) \
__global__ void NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int64_t score_stride, int topk, int B) \
{ \
    (void)topk; \
    constexpr uint32_t TOPK = TOPK_VALUE; \
    const int bid = blockIdx.x; \
    if (bid >= B) return; \
    const int32_t length = lengths[bid]; \
    const maca_bfloat16* row = scores + bid * score_stride; \
    int32_t* out = indices + bid * TOPK; \
    if (length <= static_cast<int32_t>(TOPK)) { \
        for (int i = threadIdx.x; i < static_cast<int>(TOPK); i += kLongRowBlockSize) \
            out[i] = (i < length) ? static_cast<int32_t>(i) : -1; \
        return; \
    } \
    radix_topk_row_bf16_k<TOPK, kLongRowBlockSize, COMPACT_VALUE>(row, out, length); \
}

RK_DEFINE_TOPK_BF16_LONG_KERNEL(topk_bf16_kernel_k512_long, 512, false)
RK_DEFINE_TOPK_BF16_LONG_KERNEL(topk_bf16_kernel_k1024_long, 1024, false)
RK_DEFINE_TOPK_BF16_LONG_KERNEL(topk_bf16_kernel_k512_compact_long, 512, true)
RK_DEFINE_TOPK_BF16_LONG_KERNEL(topk_bf16_kernel_k1024_compact_long, 1024, true)

#undef RK_DEFINE_TOPK_BF16_LONG_KERNEL

template <uint32_t TOPK>
__global__ void topk_bf16_chunk_stage1_kernel_k(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* candidate_indices,
    maca_bfloat16* candidate_values, int64_t score_stride, int B,
    int num_chunks, int chunk_size)
{
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;

    const int32_t length = lengths[bid];
    const int start = chunk * chunk_size;
    const int chunk_len = (start < length) ? min(chunk_size, static_cast<int>(length - start)) : 0;
    constexpr int BLOCK_SIZE = kChunkBlockSize;
    constexpr int TOPK_I = static_cast<int>(TOPK);
    const int candidate_stride = num_chunks * TOPK_I;
    int32_t* chunk_indices = candidate_indices + bid * candidate_stride + chunk * TOPK_I;
    maca_bfloat16* chunk_values = candidate_values + bid * candidate_stride + chunk * TOPK_I;
    const maca_bfloat16* row = scores + bid * score_stride;
    const maca_bfloat16 invalid_value = __float2bfloat16(-65504.0f);

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < TOPK_I; i += BLOCK_SIZE) {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
        return;
    }

    if (chunk_len <= TOPK_I) {
        for (int i = threadIdx.x; i < TOPK_I; i += BLOCK_SIZE) {
            if (i < chunk_len) {
                const int32_t idx = start + i;
                chunk_indices[i] = idx;
                chunk_values[i] = __ldg(row + idx);
            } else {
                chunk_indices[i] = -1;
                chunk_values[i] = invalid_value;
            }
        }
        return;
    }

    radix_topk_row_bf16_k<TOPK, kChunkBlockSize>(row + start, chunk_indices, chunk_len);
    for (int i = threadIdx.x; i < TOPK_I; i += BLOCK_SIZE) {
        const int32_t local_idx = chunk_indices[i];
        if (local_idx >= 0 && local_idx < chunk_len) {
            const int32_t idx = start + local_idx;
            chunk_indices[i] = idx;
            chunk_values[i] = __ldg(row + idx);
        } else {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
    }
}

__global__ void topk_bf16_chunk_stage1_kernel(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* candidate_indices,
    maca_bfloat16* candidate_values, int64_t score_stride, int topk, int B,
    int num_chunks, int chunk_size)
{
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;

    const int32_t length = lengths[bid];
    const int start = chunk * chunk_size;
    const int chunk_len = (start < length) ? min(chunk_size, static_cast<int>(length - start)) : 0;
    const int candidate_stride = num_chunks * topk;
    constexpr int BLOCK_SIZE = kChunkBlockSize;
    int32_t* chunk_indices = candidate_indices + bid * candidate_stride + chunk * topk;
    maca_bfloat16* chunk_values = candidate_values + bid * candidate_stride + chunk * topk;
    const maca_bfloat16* row = scores + bid * score_stride;
    const maca_bfloat16 invalid_value = __float2bfloat16(-65504.0f);

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
        return;
    }

    if (chunk_len <= topk) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            if (i < chunk_len) {
                const int32_t idx = start + i;
                chunk_indices[i] = idx;
                chunk_values[i] = __ldg(row + idx);
            } else {
                chunk_indices[i] = -1;
                chunk_values[i] = invalid_value;
            }
        }
        return;
    }

    radix_topk_row_bf16_b<kChunkBlockSize>(row + start, chunk_indices, chunk_len, topk);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t local_idx = chunk_indices[i];
        if (local_idx >= 0 && local_idx < chunk_len) {
            const int32_t idx = start + local_idx;
            chunk_indices[i] = idx;
            chunk_values[i] = __ldg(row + idx);
        } else {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
    }
}

template <uint32_t TOPK>
__global__ void topk_bf16_chunk_stage2_kernel_k(
    const maca_bfloat16* candidate_values, const int32_t* candidate_indices,
    int32_t* indices, int candidate_stride, int B)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;

    const maca_bfloat16* values = candidate_values + bid * candidate_stride;
    const int32_t* candidates = candidate_indices + bid * candidate_stride;
    int32_t* out = indices + bid * TOPK;
    constexpr int BLOCK_SIZE = kChunkBlockSize;

    radix_topk_row_bf16_k<TOPK, kChunkBlockSize>(values, out, candidate_stride);
    for (int i = threadIdx.x; i < static_cast<int>(TOPK); i += BLOCK_SIZE) {
        const int32_t candidate_pos = out[i];
        out[i] = (candidate_pos >= 0 && candidate_pos < candidate_stride)
               ? candidates[candidate_pos]
               : -1;
    }
}

__global__ void topk_bf16_chunk_stage2_kernel(
    const maca_bfloat16* candidate_values, const int32_t* candidate_indices,
    int32_t* indices, int candidate_stride, int topk, int B)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;

    const maca_bfloat16* values = candidate_values + bid * candidate_stride;
    const int32_t* candidates = candidate_indices + bid * candidate_stride;
    int32_t* out = indices + bid * topk;
    constexpr int BLOCK_SIZE = kChunkBlockSize;

    radix_topk_row_bf16_b<kChunkBlockSize>(values, out, candidate_stride, topk);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t candidate_pos = out[i];
        out[i] = (candidate_pos >= 0 && candidate_pos < candidate_stride)
               ? candidates[candidate_pos]
               : -1;
    }
}

inline cudaError_t launch_topk_bf16_runtime(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int B, int L, int topk, cudaStream_t stream)
{
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_bf16_kernel_runtime, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_bf16_kernel_runtime<<<B, kBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, topk, B);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_bf16_k50(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_bf16_kernel_k50, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_bf16_kernel_k50<<<B, kBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, 50, B);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_bf16_k100(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int B, int L, cudaStream_t stream)
{
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_bf16_kernel_k100, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_bf16_kernel_k100<<<B, kBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, 100, B);
    return cudaGetLastError();
}

#define RK_DEFINE_TOPK_BF16_LAUNCH(NAME, KERNEL_NAME, TOPK_VALUE) \
inline cudaError_t NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int B, int L, cudaStream_t stream) \
{ \
    static bool init = false; \
    if (!init) { \
        cudaError_t err = cudaFuncSetAttribute( \
            KERNEL_NAME, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM); \
        if (err != cudaSuccess) return err; \
        init = true; \
    } \
    KERNEL_NAME<<<B, kBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, TOPK_VALUE, B); \
    return cudaGetLastError(); \
}

RK_DEFINE_TOPK_BF16_LAUNCH(launch_topk_bf16_k512, topk_bf16_kernel_k512, 512)
RK_DEFINE_TOPK_BF16_LAUNCH(launch_topk_bf16_k1024, topk_bf16_kernel_k1024, 1024)
RK_DEFINE_TOPK_BF16_LAUNCH(launch_topk_bf16_k1536, topk_bf16_kernel_k1536, 1536)
RK_DEFINE_TOPK_BF16_LAUNCH(launch_topk_bf16_k2028, topk_bf16_kernel_k2028, 2028)
RK_DEFINE_TOPK_BF16_LAUNCH(launch_topk_bf16_k2048, topk_bf16_kernel_k2048, 2048)

#define RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(NAME, KERNEL_NAME, FALLBACK_NAME, TOPK_VALUE) \
inline cudaError_t NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int B, int L, cudaStream_t stream) \
{ \
    if (KTOPK_COMPACT_BF16_INDICES && \
        (kSMEM > 16 * 1024 || L > static_cast<int>(kCompactBF16MaxLength))) \
        return FALLBACK_NAME(scores, lengths, indices, B, L, stream); \
    static bool init = false; \
    if (!init) { \
        cudaError_t err = cudaFuncSetAttribute( \
            KERNEL_NAME, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM); \
        if (err != cudaSuccess) return err; \
        init = true; \
    } \
    KERNEL_NAME<<<B, kBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, TOPK_VALUE, B); \
    return cudaGetLastError(); \
}

RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k50_compact, topk_bf16_kernel_k50_compact, launch_topk_bf16_k50, 50)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k100_compact, topk_bf16_kernel_k100_compact, launch_topk_bf16_k100, 100)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k512_compact, topk_bf16_kernel_k512_compact, launch_topk_bf16_k512, 512)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k1024_compact, topk_bf16_kernel_k1024_compact, launch_topk_bf16_k1024, 1024)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k1536_compact, topk_bf16_kernel_k1536_compact, launch_topk_bf16_k1536, 1536)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k2028_compact, topk_bf16_kernel_k2028_compact, launch_topk_bf16_k2028, 2028)
RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH(
    launch_topk_bf16_k2048_compact, topk_bf16_kernel_k2048_compact, launch_topk_bf16_k2048, 2048)

#define RK_DEFINE_TOPK_BF16_LONG_LAUNCH(NAME, KERNEL_NAME, TOPK_VALUE) \
inline cudaError_t NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int B, int L, cudaStream_t stream) \
{ \
    static bool init = false; \
    if (!init) { \
        cudaError_t err = cudaFuncSetAttribute( \
            KERNEL_NAME, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM); \
        if (err != cudaSuccess) return err; \
        init = true; \
    } \
    KERNEL_NAME<<<B, kLongRowBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, TOPK_VALUE, B); \
    return cudaGetLastError(); \
}

RK_DEFINE_TOPK_BF16_LONG_LAUNCH(
    launch_topk_bf16_k512_long, topk_bf16_kernel_k512_long, 512)
RK_DEFINE_TOPK_BF16_LONG_LAUNCH(
    launch_topk_bf16_k1024_long, topk_bf16_kernel_k1024_long, 1024)

#define RK_DEFINE_TOPK_BF16_LONG_COMPACT_LAUNCH(NAME, KERNEL_NAME, FALLBACK_NAME, TOPK_VALUE) \
inline cudaError_t NAME( \
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices, \
    int B, int L, cudaStream_t stream) \
{ \
    if (KTOPK_COMPACT_BF16_INDICES && \
        (kSMEM > 16 * 1024 || L > static_cast<int>(kCompactBF16MaxLength))) \
        return FALLBACK_NAME(scores, lengths, indices, B, L, stream); \
    static bool init = false; \
    if (!init) { \
        cudaError_t err = cudaFuncSetAttribute( \
            KERNEL_NAME, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM); \
        if (err != cudaSuccess) return err; \
        init = true; \
    } \
    KERNEL_NAME<<<B, kLongRowBlockSize, kSMEM, stream>>>(scores, lengths, indices, L, TOPK_VALUE, B); \
    return cudaGetLastError(); \
}

RK_DEFINE_TOPK_BF16_LONG_COMPACT_LAUNCH(
    launch_topk_bf16_k512_compact_long, topk_bf16_kernel_k512_compact_long,
    launch_topk_bf16_k512_long, 512)
RK_DEFINE_TOPK_BF16_LONG_COMPACT_LAUNCH(
    launch_topk_bf16_k1024_compact_long, topk_bf16_kernel_k1024_compact_long,
    launch_topk_bf16_k1024_long, 1024)

#undef RK_DEFINE_TOPK_BF16_LONG_COMPACT_LAUNCH
#undef RK_DEFINE_TOPK_BF16_LONG_LAUNCH

#undef RK_DEFINE_TOPK_BF16_COMPACT_LAUNCH
#undef RK_DEFINE_TOPK_BF16_LAUNCH

template <uint32_t TOPK>
inline cudaError_t launch_topk_bf16_chunked_k(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int num_chunks, cudaStream_t stream,
    int64_t score_stride = 0)
{
    if (TOPK > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_bf16_chunk_stage1_kernel_k<TOPK>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        err = cudaFuncSetAttribute(
            topk_bf16_chunk_stage2_kernel_k<TOPK>, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }

    const int raw_chunk_size = (L + num_chunks - 1) / num_chunks;
    const int chunk_size = (raw_chunk_size + 7) / 8 * 8;
    const int candidate_stride = num_chunks * static_cast<int>(TOPK);
    // Deviation 3: the row stride can be wider than the window (`L`), which is
    // how a caller with padded input rows reaches this path; upstream passes
    // `L` and so can only split packed rows.
    const int64_t stride = score_stride > 0 ? score_stride : (int64_t)L;
    topk_bf16_chunk_stage1_kernel_k<TOPK><<<B * num_chunks, kChunkBlockSize, kSMEM, stream>>>(
        scores, lengths, candidate_indices, candidate_values, stride, B, num_chunks, chunk_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    topk_bf16_chunk_stage2_kernel_k<TOPK><<<B, kChunkBlockSize, kSMEM, stream>>>(
        candidate_values, candidate_indices, indices, candidate_stride, B);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_bf16_chunked(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride = 0)
{
    if (topk > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    switch (topk) {
    case 512:
        return launch_topk_bf16_chunked_k<512>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream, score_stride);
    case 1024:
        return launch_topk_bf16_chunked_k<1024>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream, score_stride);
    default:
        break;
    }

    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_bf16_chunk_stage1_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        err = cudaFuncSetAttribute(
            topk_bf16_chunk_stage2_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }

    const int raw_chunk_size = (L + num_chunks - 1) / num_chunks;
    // Keep each chunk base aligned for the uint4 FP16 vectorized row path.
    const int chunk_size = (raw_chunk_size + 7) / 8 * 8;
    const int64_t stride = score_stride > 0 ? score_stride : (int64_t)L;
    topk_bf16_chunk_stage1_kernel<<<B * num_chunks, kChunkBlockSize, kSMEM, stream>>>(
        scores, lengths, candidate_indices, candidate_values, stride, topk, B, num_chunks, chunk_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    topk_bf16_chunk_stage2_kernel<<<B, kChunkBlockSize, kSMEM, stream>>>(
        candidate_values, candidate_indices, indices, num_chunks * topk, topk, B);
    return cudaGetLastError();
}

inline int recommend_topk_bf16_chunks(int B, int L, int topk)
{
    if (topk == 1024 && B <= 64 && L >= 1048576)
        return 12;
    if (topk == 512 && B <= 64 && L >= 1048576)
        return 13;
    return 1;
}

inline size_t topk_bf16_chunked_workspace_bytes(int B, int topk, int num_chunks)
{
    if (num_chunks <= 1) return 0;
    return static_cast<size_t>(B) * num_chunks * topk * (sizeof(int32_t) + sizeof(maca_bfloat16));
}

// ============================================================
// 配置驱动 dispatch
// ============================================================
//
// The `-DKSMEM_BYTES=` size per workload is now decided by the launchers, not
// by the caller: see `radix_smem_bytes()` in `maca_topk.cu`.
//
// The chunked path needs extra workspace: B * chunks * topk * (4 + 2) bytes.
//
struct TopKConfig {
    int B;
    int L;
    int topk;
};

inline bool needs_chunked(const TopKConfig& cfg)
{
    // dsv4 decode: B 小、L 极大，单 block 并行度不足
    return cfg.B <= 64 && cfg.L >= 1048576 && cfg.topk >= 512;
}

inline int recommend_chunks(const TopKConfig& cfg)
{
    if (!needs_chunked(cfg)) return 1;
    return recommend_topk_bf16_chunks(cfg.B, cfg.L, cfg.topk);
}

inline size_t workspace_bytes(const TopKConfig& cfg)
{
    int chunks = recommend_chunks(cfg);
    return topk_bf16_chunked_workspace_bytes(cfg.B, cfg.topk, chunks);
}

inline bool needs_compact_bf16_indices(const TopKConfig& cfg)
{
#if KTOPK_COMPACT_BF16_INDICES
    // The upper bound keeps every possible row offset representable in uint16;
    // bucket overflow is handled by the overflow-safe full-row fallback.
    if (kSMEM > 16 * 1024) return false;
    return cfg.L >= 32768 && cfg.L <= static_cast<int>(kCompactBF16MaxLength);
#else
    (void)cfg;
    return false;
#endif
}

inline bool needs_long_row_bf16(const TopKConfig& cfg)
{
    // The measured crossover is specific to the 16 KiB C500 row variant. A
    // larger arena, a short batch, or an unsupported k keeps the 512 path so
    // compact-fit and small-row latency do not regress.
    if (kBlockSize != 512 || kSMEM > 16 * 1024 || cfg.B <= 0 || cfg.B > 4096 ||
        cfg.L < 60000)
        return false;
    return cfg.topk == 512 || cfg.topk == 1024;
}

// 单行 dispatch: 按 topk 值选择特化 kernel
inline cudaError_t launch_topk_bf16_single_row(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int B, int L, int topk, cudaStream_t stream)
{
    const TopKConfig cfg{B, L, topk};
    if (needs_long_row_bf16(cfg)) {
        if (needs_compact_bf16_indices(cfg)) {
            switch (topk) {
            case 512:
                return launch_topk_bf16_k512_compact_long(
                    scores, lengths, indices, B, L, stream);
            case 1024:
                return launch_topk_bf16_k1024_compact_long(
                    scores, lengths, indices, B, L, stream);
            default:
                break;
            }
        }
        switch (topk) {
        case 512:
            return launch_topk_bf16_k512_long(scores, lengths, indices, B, L, stream);
        case 1024:
            return launch_topk_bf16_k1024_long(scores, lengths, indices, B, L, stream);
        default:
            break;
        }
    }
    if (needs_compact_bf16_indices(cfg)) {
        switch (topk) {
        case 50: return launch_topk_bf16_k50_compact(scores, lengths, indices, B, L, stream);
        case 100: return launch_topk_bf16_k100_compact(scores, lengths, indices, B, L, stream);
        case 512: return launch_topk_bf16_k512_compact(scores, lengths, indices, B, L, stream);
        case 1024: return launch_topk_bf16_k1024_compact(scores, lengths, indices, B, L, stream);
        case 1536: return launch_topk_bf16_k1536_compact(scores, lengths, indices, B, L, stream);
        case 2028: return launch_topk_bf16_k2028_compact(scores, lengths, indices, B, L, stream);
        case 2048: return launch_topk_bf16_k2048_compact(scores, lengths, indices, B, L, stream);
        default: break;
        }
    }
    switch (topk) {
    case 50: return launch_topk_bf16_k50(scores, lengths, indices, B, L, stream);
    case 100: return launch_topk_bf16_k100(scores, lengths, indices, B, L, stream);
    case 512: return launch_topk_bf16_k512(scores, lengths, indices, B, L, stream);
    case 1024: return launch_topk_bf16_k1024(scores, lengths, indices, B, L, stream);
    case 1536: return launch_topk_bf16_k1536(scores, lengths, indices, B, L, stream);
    case 2028: return launch_topk_bf16_k2028(scores, lengths, indices, B, L, stream);
    case 2048: return launch_topk_bf16_k2048(scores, lengths, indices, B, L, stream);
    default: return launch_topk_bf16_runtime(scores, lengths, indices, B, L, topk, stream);
    }
}

// 统一 dispatch 入口: 自动选择 single-row 或 chunked 路径
inline cudaError_t launch_topk_bf16_dispatch(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    const TopKConfig& cfg, cudaStream_t stream)
{
    if (needs_chunked(cfg)) {
        int chunks = recommend_chunks(cfg);
        return launch_topk_bf16_chunked(
            scores, lengths, indices, candidate_indices, candidate_values,
            cfg.B, cfg.L, cfg.topk, chunks, stream);
    }
    return launch_topk_bf16_single_row(
        scores, lengths, indices, cfg.B, cfg.L, cfg.topk, stream);
}

// 兼容旧接口: 无 chunked 支持的简化 dispatch
inline cudaError_t launch_topk_bf16_dispatch(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int B, int L, int topk, cudaStream_t stream)
{
    return launch_topk_bf16_single_row(scores, lengths, indices, B, L, topk, stream);
}

// stage 1's answer is a *value*, so the merge can rank `chunks * topk` of them
// and map positions back.  A slot stage 1 could not fill takes `-inf`, which
// cannot appear in a real input: `is_nan_value` gates NaNs out before any
// selection runs, so no element is NaN and -inf is unreachable from the row.
__global__ __launch_bounds__(kBlockSize) void topk_f32_chunk_stage1_kernel(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int64_t score_stride, int topk, int B, int num_chunks, int chunk_size,
    int candidate_stride)
{
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;

    const int32_t length = lengths[bid];
    const int start = chunk * chunk_size;
    const int chunk_len = (start < length)
                              ? min(chunk_size, static_cast<int>(length - start))
                              : 0;
    constexpr int BLOCK_SIZE = kBlockSize;
    int32_t *chunk_cols = cols + bid * candidate_stride + chunk * topk;
    float *chunk_vals = vals + bid * candidate_stride + chunk * topk;
    const float *row = scores + (int64_t)bid * score_stride;

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            chunk_cols[i] = -1;
            chunk_vals[i] = -__builtin_huge_valf();
        }
        return;
    }

    if (chunk_len <= topk) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            if (i < chunk_len) {
                chunk_cols[i] = start + i;
                chunk_vals[i] = __ldg(row + start + i);
            } else {
                chunk_cols[i] = -1;
                chunk_vals[i] = -__builtin_huge_valf();
            }
        }
        return;
    }

    // The row entry, on this chunk's window, answering in row columns.  Its
    // `-1` sentinel is a column index that cannot be real, so it becomes the
    // same empty-slot marker the two arms above write.
    radix_topk_row_f32(row + start, chunk_cols, (uint32_t)chunk_len,
                       (uint32_t)topk, (uint32_t)start);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t col = chunk_cols[i];
        if (col >= start && col < start + chunk_len) {
            chunk_vals[i] = __ldg(row + col);
        } else {
            chunk_cols[i] = -1;
            chunk_vals[i] = -__builtin_huge_valf();
        }
    }
}

// The merge.  It ranks the candidate *values* -- so its answer is a position
// into the candidate arrays, which is what lets `cols` be mapped back -- and
// rewrites `topk` columns.
__global__ __launch_bounds__(kBlockSize) void topk_f32_chunk_stage2_kernel(
    const float *vals, const int32_t *cols, int32_t *out, int topk, int B,
    int candidate_stride)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;

    const float *row_vals = vals + bid * candidate_stride;
    const int32_t *row_cols = cols + bid * candidate_stride;
    int32_t *row_out = out + (int64_t)bid * topk;
    constexpr int BLOCK_SIZE = kBlockSize;

    radix_topk_row_f32(row_vals, row_out, (uint32_t)candidate_stride,
                       (uint32_t)topk);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t pos = row_out[i];
        row_out[i] = (pos >= 0 && pos < candidate_stride) ? row_cols[pos] : -1;
    }
}

// ── the fp32 chunked split ──────────────────────────────────────────────────
//
// The 16-bit split above ranks each chunk with `radix_topk_row_bf16_k`, whose
// output is a position *inside the chunk*, so its merge ranks `chunks * topk`
// candidate values and maps positions back.  The fp32 row entry writes row
// columns instead (its `chunk_begin` is the chunk's column offset), so the
// split carries the columns alongside the values:
//
//   cols[batches * chunks * topk]  int32  the row columns stage 1 selected
//   vals[batches * chunks * topk]  float  their values, -inf for an empty slot
//   out [batches * topk]           int32  the merge's answer, one column each
//
// Stage 2 ranks `vals` and maps the resulting position through `cols`.
// Everything else -- the grid, the `-1` sentinel, the contract half's `rerank`
// fallback -- is the 16-bit split's.
//
// The chunk count, the work target behind it and the reason a short batch is
// split at all are in `csrc/structs.h` (NATIVE_F32_CHUNK_WORK_TARGET).
constexpr size_t kF32ChunkBlocks = kBlockSize;   // `radix_topk_row_f32`'s width
// The split's chunk count is the caller's (`deep_select_maca::kChunkedChunks`,
// currently 16, which is also what `nan_scan_kernel` is launched with).  It is
// only ever used for *sizing* here -- the kernels take it as an argument -- so
// this header does not need the caller's constant, just a ceiling to bound
// `uint32_t` arithmetic with.  64 covers any value the dispatcher could pick.
constexpr uint32_t kF32MaxTopK = 4096u;

inline size_t chunked_f32_workspace_bytes(uint32_t batches, uint32_t topk,
                                          uint32_t chunks) {
    const size_t candidates = (size_t)batches * chunks * topk;
    return candidates * sizeof(int32_t) + candidates * sizeof(float) +
           (size_t)batches * topk * sizeof(int32_t);
}

inline int32_t *chunked_f32_cols(void *base, uint32_t batches, uint32_t topk,
                                 uint32_t chunks) {
    (void)batches; (void)topk; (void)chunks;
    return (int32_t *)base;
}

inline float *chunked_f32_vals(void *base, uint32_t batches, uint32_t topk,
                              uint32_t chunks) {
    return (float *)(chunked_f32_cols(base, batches, topk, chunks) +
                     (size_t)batches * chunks * topk);
}

inline int32_t *chunked_f32_out(void *base, uint32_t batches, uint32_t topk,
                                uint32_t chunks) {
    return (int32_t *)(chunked_f32_vals(base, batches, topk, chunks) +
                       (size_t)batches * chunks * topk);
}

// One CTA per (batch, chunk).  The chunk geometry is the 16-bit split's --
// `raw = ceil(L / chunks)` rounded up to 8 elements -- so the two agree about
// where a chunk starts, which is what lets `nan_scan_kernel` be shared.
inline cudaError_t launch_topk_f32_chunks_stage1(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride)
{
    if (topk > (int)kF32MaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_chunk_stage1_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }
    const int raw_chunk_size = (L + num_chunks - 1) / num_chunks;
    const int chunk_size = (raw_chunk_size + 7) / 8 * 8;
    const int candidate_stride = num_chunks * topk;
    topk_f32_chunk_stage1_kernel<<<B * num_chunks, (int)kF32ChunkBlocks, kSMEM,
                                   stream>>>(
        scores, lengths, cols, vals, score_stride, topk, B, num_chunks,
        chunk_size, candidate_stride);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_f32_chunks_stage2(
    const float *vals, const int32_t *cols, int32_t *out,
    int B, int topk, int num_chunks, cudaStream_t stream)
{
    if (topk > (int)kF32MaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_f32_chunk_stage2_kernel,
            cudaFuncAttributeMaxDynamicSharedMemorySize, kSMEM);
        if (err != cudaSuccess) return err;
        init = true;
    }
    const int candidate_stride = num_chunks * topk;
    topk_f32_chunk_stage2_kernel<<<B, (int)kF32ChunkBlocks, kSMEM, stream>>>(
        vals, cols, out, topk, B, candidate_stride);
    return cudaGetLastError();
}

// stage 1, then stage 2.  `cols_base` is the start of the candidate arena the
// split carved out (`chunked_f32_cols`); `out` must have room for `topk`
// columns per row.  They are separate arguments because the caller keeps its
// own state (the row-length table, the per-row NaN flags) in front of the
// arena, so the arena is not at the base of the caller's workspace.
//
// This is the *engine* bound, not the policy one: `chunked_f32_applies` (in
// `maca_topk.cu`) decides whether to split and calls this to find out whether
// the split can serve the topk.  Keeping them separate is what stops the policy
// from looking more expensive than the engine actually is.
inline bool f32_chunk_engine_supports(int topk) {
    // The fp32 split is dynamic-k: stage1/stage2 are templated on `kBlockSize`
    // alone and take `topk` as an argument (`radix_topk_row_f32`'s `topk` is a
    // runtime parameter too), so the engine's bound is the array limit, not a
    // switch.  The `512 || 1024` that used to sit in `chunked_f32_applies` was
    // a *policy* bound from where the split was measured, not an instantiation
    // one -- which is what plan §21.2 says and this makes explicit.
    return topk > 0 && topk <= (int)kF32MaxTopK;
}

inline cudaError_t launch_topk_f32_chunked(
    const float *scores, const int32_t *lengths, void *cols_base, int32_t *out,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride)
{
    const uint32_t b = (uint32_t)B, k = (uint32_t)topk, c = (uint32_t)num_chunks;
    int32_t *cols = chunked_f32_cols(cols_base, b, k, c);
    float *vals = chunked_f32_vals(cols_base, b, k, c);
    cudaError_t err = launch_topk_f32_chunks_stage1(
        scores, lengths, cols, vals, B, L, topk, num_chunks, stream, score_stride);
    if (err != cudaSuccess) return err;
    return launch_topk_f32_chunks_stage2(
        vals, cols, out, B, topk, num_chunks, stream);
}

}  // namespace rk

#endif  // RADIX_TOPK_CUH
