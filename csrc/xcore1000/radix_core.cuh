// ── provenance ──────────────────────────────────────────────────────────────
// Ported from the standalone C500 radix-TopK project at
//   /home/compiler_gfx/dsa_topk, csrc/radix_topk.cuh
//   commit 61ab380c77b81669718bfb11b95a583b0e661001  (blob 75af81c7ae1e44cc0b5baf8f8dd02a75ea1aa4dd)
//
// Why it is here: this is a two-pass dataflow (one histogram pass, one
// collect-and-stage pass, with the low bits refined inside a shared-memory
// arena) where the incumbent `maca_topk.cu` re-walks the row twice per key
// byte -- 8 passes for a 4-byte key, 4 for a 2-byte one.  Same operator, same
// hardware, 158..280 GB/s measured there against ~21 GB/s here.
//
// ── the one deviation from that file: the 16-bit path speaks bf16, not fp16 ──
// DeepSelect has no fp16: its two dtypes are bf16 and fp32, and the 16-bit
// path here serves bf16.  The retarget is a type substitution and nothing
// else -- `half_to_uintN` -> `bf16_to_uintN` reading the value's own bit
// pattern (`__bfloat16_as_ushort`, a reinterpret) instead of fp16's, and the
// `f16`-named symbols renamed to `bf16`.  The key transform is unchanged: it
// only depends on the sign bit being bit 15, which holds for both.
// Deliberately NOT copied from `float_to_uint8`: that one rounds fp32 through
// fp16 before binning, which a bf16 input has no source for.
// The fp32 half of the file (`float_to_uint*`, `radix_topk_row_f32*`, the
// k2048 family) is still verbatim, and is what fp32 will be built on.
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

__device__ __forceinline__ void hist_add_f32(
    uint32_t* s_histogram, const float* input, uint32_t idx)
{
    float4 v = __ldg(reinterpret_cast<const float4*>(input + idx));
    atomicAdd(&s_histogram[float_to_uint8(v.x)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.y)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.z)], 1u);
    atomicAdd(&s_histogram[float_to_uint8(v.w)], 1u);
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
 * 设计思路:
 *   - 每个 warp 维护一个独立的寄存器直方图 (256 bins)
 *   - 每个线程持有 256/kWarpSize 个 bin
 *   - 读取元素后, 用 __shfl_sync 将 bin 发给 owner 线程
 *   - owner 线程在寄存器中累加
 *   - 最后 warp 内归约, 写入 smem
 *
 * 优势: 避免 smem atomicAdd 竞争, 寄存器访问零延迟
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
 * run_cumsum_warp: 256 元素 reverse inclusive scan (warp shuffle)
 *
 * 方案B: 单缓冲 warp scan
 *   - 8 个 warp 各处理 32 元素, warp 内用 __shfl_down_sync
 *   - warp 总和写入 s_histogram_buf[0][257..264] (保留 256 作为哨兵)
 *   - __syncthreads
 *   - warp 0 对 8 个总和做 scan
 *   - __syncthreads
 *   - 各 warp 加偏移
 *   → 2 次 __syncthreads (vs 原来 8 次)
 *   - 返回当前线程 bin 的 inclusive/exclusive suffix，调用方可避免
 *     在额外 barrier 前读取相邻 bin。
 *
 * 注意: FP32 使用 run_cumsum, FP16 使用 run_cumsum_warp
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

__device__ __forceinline__ void radix_topk_row_f32(
    const float* input, int32_t* output, uint32_t length, uint32_t topk)
{
    constexpr uint32_t RADIX = 256;
    constexpr uint32_t BLOCK_SIZE = kBlockSize;

    // smem 布局：直方图 + 候选索引
    // 静态变量: s_histogram_buf(2304) + s_counter(4) + s_threshold_bin_id(4)
    //           + s_num_input(8) + s_last_remain(4) = 2324 bytes
    // 候选索引: 剩余空间
    constexpr uint32_t STATIC_BYTES = 2 * (RADIX + 32) * sizeof(uint32_t)
                                    + sizeof(uint32_t) + sizeof(uint32_t)
                                    + 2 * sizeof(uint32_t) + sizeof(int32_t);
    constexpr uint32_t SMEM_INPUT_SIZE = (kSMEM - STATIC_BYTES) / (2 * sizeof(int32_t));

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
    for (uint32_t idx = tx * 4; idx < vec_len; idx += BLOCK_SIZE * 4)
        hist_add_f32(s_histogram, input, idx);
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
            for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                if (float_to_uint8(__ldg(input + idx)) > threshold_bin)
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;
        __syncthreads();
        for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE) {
            float raw = __ldg(input + idx);
            uint32_t bin = float_to_uint8(raw);
            if (bin > threshold_bin) {
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                uint32_t pos = atomicAdd(&s_num_input[0], 1u);
                if (pos < SMEM_INPUT_SIZE) {
                    s_input_flat[pos] = idx;
                    atomicAdd(&s_histogram[(float_to_uint32(raw) >> 24) & 0xFF], 1u);
                }
            }
        }
        __syncthreads();
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
                    output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
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
                output[atomicAdd(&s_counter, 1u)] = static_cast<int32_t>(idx);
            } else if (bin == threshold_bin) {
                if (round == 3) {
                    auto p = atomicAdd(&s_last_remain, -1);
                    if (p > 0) output[topk - p] = static_cast<int32_t>(idx);
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

template <uint32_t BLOCK_SIZE>
__device__ __forceinline__ void radix_topk_row_bf16_b(
    const maca_bfloat16* input, int32_t* output, uint32_t length, uint32_t topk)
{
    constexpr uint32_t RADIX = kRadix;
    constexpr uint32_t SMEM_INPUT_SIZE = kSmemInputSize;

    __shared__ uint32_t s_histogram_buf[2][RADIX + 32];
    __shared__ uint32_t s_counter;
    __shared__ uint32_t s_threshold_bin_id;
    __shared__ uint32_t s_high_threshold_bin_id;
    __shared__ uint32_t s_num_input[2];
    __shared__ int32_t s_last_remain;
    extern __shared__ uint32_t s_input_flat[];
    // FP16 needs only one candidate index buffer; overflow falls back to a full rescan.

    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = topk;
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
                        if (pos < SMEM_INPUT_SIZE) {
                            s_input_flat[pos] = idx + i;
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
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_flat[pos] = idx;
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
                    if (pos < SMEM_INPUT_SIZE) {
                        s_input_flat[pos] = idx;
                        atomicAdd(&s_histogram[bf16_to_uint16(raw) & 0xFF], 1u);
                    }
                }
            }
        }
        __syncthreads();
    }

    {
        const bool overflow = s_num_input[0] > SMEM_INPUT_SIZE;
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
                        if (p > 0) output[topk - p] = static_cast<int32_t>(idx);
                    }
                }
            }
            __syncthreads(); return;
        }
        const auto num = s_num_input[0];
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
    int B, int L, int num_chunks, cudaStream_t stream)
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
    topk_bf16_chunk_stage1_kernel_k<TOPK><<<B * num_chunks, kChunkBlockSize, kSMEM, stream>>>(
        scores, lengths, candidate_indices, candidate_values, L, B, num_chunks, chunk_size);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) return err;

    topk_bf16_chunk_stage2_kernel_k<TOPK><<<B, kChunkBlockSize, kSMEM, stream>>>(
        candidate_values, candidate_indices, indices, candidate_stride, B);
    return cudaGetLastError();
}

inline cudaError_t launch_topk_bf16_chunked(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int topk, int num_chunks, cudaStream_t stream)
{
    if (topk > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    switch (topk) {
    case 512:
        return launch_topk_bf16_chunked_k<512>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream);
    case 1024:
        return launch_topk_bf16_chunked_k<1024>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream);
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
    topk_bf16_chunk_stage1_kernel<<<B * num_chunks, kChunkBlockSize, kSMEM, stream>>>(
        scores, lengths, candidate_indices, candidate_values, L, topk, B, num_chunks, chunk_size);
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
// 推荐编译配置:
//   搜广推 (B≤6000, L=10K, k≤1024):  -DKSMEM_BYTES=16384  (16KB, 高 occupancy)
//   DSA (B=4096, L=50K, k=512~2048): -DKSMEM_BYTES=49152  (48KB, 大候选缓冲)
//   dsv4 decode (B≤64, L≥1M):        -DKSMEM_BYTES=49152  (48KB, chunked 路径)
//   dsv4 prefill (B=128~1K, L≤128K): -DKSMEM_BYTES=49152  (48KB)
//
// chunked 路径需要额外 workspace: B * chunks * topk * (4 + 2) bytes
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

}  // namespace rk

#endif  // RADIX_TOPK_CUH
