// Sampled-threshold one-pass fp32 row (LiteTopK's sample-filter-select,
// arXiv 2607.11976; attribution per DeepSelect issue #13).
//
// Why this file exists
// --------------------
// `topk_coarse12.cuh` (deep_gemm's port) serves the big fp32 cells with two
// full-row walks, and the phase probe of 2026-10-21 (pod 30752,
// `docs/experiments/20261021_coarse12_phase/`) measured both at the same
// ~600-670 GB/s in situ -- pass 1 43%, collect 49%, arena 5.6% of the
// kernel.  The atomics are NOT binding (a replica of pass 1's walk hits
// 1226-1310 GB/s), so the lever is deleting one walk, and the sampled
// threshold is what makes one walk exact:
//
//   1. histogram a contiguous `length/16` prefix at 12 bits and take the bin
//      holding the sample's m-th largest key (m ~ 1.4k/16);
//   2. ONE full-row walk staging {key > bin floor} into a 4096-entry smem
//      arena (measured ~2500-3300 members on the official distribution;
//      underflow is structurally impossible, overflow retries at a 20-bit
//      threshold below);
//   3. four 8-bit refinement rounds over the arena (keys re-gathered from
//      the row, L2-hot) -- exact to the full key; true ties fill
//      arbitrarily, which the contract allows.
//
// Prototype numbers (`docs/experiments/20261021_f32_sample_route/`, official
// shape, contract/set exact): 5.42 / 4.07 / 4.69 ms on the three big fp32
// cells against coarse12's 7.60 / 6.49 / 6.60 on the same toolkit --
// 1.40-1.60x.
//
// Fallback design
// ---------------
// The row function returns false only where the route genuinely cannot
// answer: candidates underflow topk (measured ~0 on the official
// distribution) or still overflow the 4096 arena after the 20-bit retry
// (also ~0).  `topk_sample_kernel` then serves the row with
// `topk_coarse12_row` INLINE -- one launch, per-row fallback, no second
// dispatch, and every row's answer is exact either way.
//
// The NaN vote is fused into the filter walk exactly as in
// `topk_coarse12.cuh`: ranking continues (the encode puts NaNs at the key
// extremes), and the contract half does the disposition from `nan_flag`.

#pragma once

#include <cstdint>
#include <cuda_runtime.h>

#include <csrc/maca_kernels/common/primitives.cuh>
#include <csrc/maca_kernels/common/topk_coarse12.cuh>

namespace topk_select_sample {

constexpr int kThreads = 640;
constexpr int kMaxTopK = topk_select_coarse12::kMaxTopK;
constexpr int kCoarseBits = 12;
constexpr int kCoarseBins = 1 << kCoarseBits;   // 4096
constexpr int kCap = 4096;                      // arena entries (16 KB)
constexpr size_t kSmemBytes = 32 * 1024;        // arena idx + scratch idx
constexpr int kSampleShift = 4;                 // sample = length >> 4

using topk_select_common::is_nan_bits32;
using topk_select_common::refine_bin;

// Suffix-cumsum of a 256-bin smem histogram, parallel: 64 lanes x 4 bins,
// local tail-sum, a 6-step guarded Hillis-Steele suffix scan, add back.
// MACA's `__shfl_down_sync` returns the lane's OWN current value on an
// out-of-range read (measured 2026-10-21, `test_shfl.cu` in the experiment
// dir), so the add is guarded on the lane, or the bottom lanes double.
__device__ __forceinline__ void suffix256(int *h, int tx) {
    if (tx >= 64) return;
    const int base = (int)(tx * 4);
    int v0 = h[base], v1 = h[base + 1], v2 = h[base + 2], v3 = h[base + 3];
    int tot = v0 + v1 + v2 + v3;
    const unsigned long long m = 0xFFFFFFFFFFFFFFFFULL;
    int suf = tot;
    #pragma unroll
    for (int d = 1; d < 64; d <<= 1) {
        const int o = __shfl_down_sync(m, suf, d);
        if (tx + d < 64) suf += o;
    }
    const int excl = suf - tot;
    h[base]     = excl + tot;
    h[base + 1] = excl + v1 + v2 + v3;
    h[base + 2] = excl + v2 + v3;
    h[base + 3] = excl + v3;
}

// After suffix256 (h[c] = count(>= c)): the unique bin where the count
// crosses `m`, one thread per bin, one winner.  `out[0]` is scratch.
__device__ __forceinline__ int find_crossing(const int *h, int m, int *out,
                                             int tx) {
    if (tx == 0) out[0] = -1;
    __syncthreads();
    if (tx < 256) {
        const int prev = tx < 255 ? h[tx + 1] : 0;
        if (prev < m && h[tx] >= m) out[0] = (int)tx;
    }
    __syncthreads();
    return out[0];
}

// Fold a 4096-bin smem histogram to 256 coarse bins (parallel, into `h256`),
// suffix-cumsum downward, find the coarse bin where the count crosses `m`,
// and narrow over its 16 sub-bins.  out[0] = the 12-bit bin, out[1] = the
// count strictly above it.
__device__ __forceinline__ void narrow12(const int *h4096, int m, int *h256,
                                         int *out, int tx) {
    for (int c = tx; c < 256; c += kThreads) {
        int sum = 0;
        #pragma unroll
        for (int j = 0; j < 16; j++) sum += h4096[(c << 4) + j];
        h256[c] = sum;
    }
    __syncthreads();
    suffix256(h256, tx);
    const int c = find_crossing(h256, m, out, tx);
    if (tx == 0) {
        const int prev = c < 255 ? h256[c + 1] : 0;
        int sub_run = prev, sub = 15, bin = c << 4;
        for (; sub >= 0; sub--) {
            const int sc = h4096[(c << 4) + sub];
            if (sub_run < m && sub_run + sc >= m) { bin = (c << 4) + sub; break; }
            sub_run += sc;
        }
        out[0] = bin;
        out[1] = sub_run;
    }
    __syncthreads();
}

// One row of `[0, length)`, ranking `topk` of it, by the sampled-threshold
// one-pass.  `output` receives positions in the window, exactly like
// `topk_coarse12_row`.  Returns false where the route cannot answer
// (underflow, or overflow past the 20-bit retry); the kernel wrapper then
// serves the row with `topk_coarse12_row`.
__device__ __forceinline__ bool topk_sample_row(
    const float *__restrict__ input, int32_t *__restrict__ output,
    uint32_t length, uint32_t topk, int32_t *__restrict__ nan_flag) {
    extern __shared__ int smem[];
    int *arena_idx = smem;                 // 4096
    int *scratch_idx = smem + 4096;        // 4096
    __shared__ int hist[256 + 16];
    __shared__ int counter, staged, bcount, sh_thr, sh_above;
    __shared__ int nan_local;
    const int tx = threadIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(input);

    if (tx == 0) {
        counter = 0;
        staged = 0;
        nan_local = 0;
    }

    // ── sample phase: 12-bit histogram of the contiguous prefix ────────────
    const int sample_len = (int)(length >> kSampleShift);
    const int m_sample =
        max(1, (int)((topk >= 1024 ? 14u * topk : 16u * topk) / 160u));
    for (int b = tx; b < kCoarseBins; b += kThreads) arena_idx[b] = 0;
    __syncthreads();
    const int svec = sample_len >> 2;
    for (int v = tx; v < svec; v += kThreads) {
        const float4 a = __ldg(row4 + v);
        atomicAdd(&arena_idx[refine_bin(a.x) >> 20], 1);
        atomicAdd(&arena_idx[refine_bin(a.y) >> 20], 1);
        atomicAdd(&arena_idx[refine_bin(a.z) >> 20], 1);
        atomicAdd(&arena_idx[refine_bin(a.w) >> 20], 1);
    }
    for (int i = (svec << 2) + tx; i < sample_len; i += kThreads)
        atomicAdd(&arena_idx[refine_bin(__ldg(input + i)) >> 20], 1);
    __syncthreads();
    narrow12(arena_idx, m_sample, hist, hist + 256, tx);
    const int bin12 = hist[256];
    uint32_t T = (uint32_t)bin12 << 20;

    // ── filter + stage walk: one full-row pass, {key > T} into the arena ───
    // NaN vote fused on the loaded words, like topk_coarse12's pass 1.
    // OUT=2: two float4 in flight (measured +8-14% on this shape).
    const int vlen = (int)(length >> 2);
    int nan_found = 0;
    for (int retry = 0; retry < 2; retry++) {
        if (tx == 0) staged = 0;
        __syncthreads();
        int v = tx;
        for (; v + kThreads < vlen; v += 2 * kThreads) {
            const float4 a = __ldg(row4 + v);
            const float4 b = __ldg(row4 + v + kThreads);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t bits = e == 0 ? __float_as_uint(a.x)
                                    : e == 1 ? __float_as_uint(a.y)
                                    : e == 2 ? __float_as_uint(a.z)
                                             : __float_as_uint(a.w);
                nan_found |= (int)is_nan_bits32(bits);
                const uint32_t k =
                    (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                if (k > T) {
                    const int pos = atomicAdd(&staged, 1);
                    if (pos < kCap) arena_idx[pos] = (v << 2) + e;
                }
            }
            const int vb = v + kThreads;
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t bits = e == 0 ? __float_as_uint(b.x)
                                    : e == 1 ? __float_as_uint(b.y)
                                    : e == 2 ? __float_as_uint(b.z)
                                             : __float_as_uint(b.w);
                nan_found |= (int)is_nan_bits32(bits);
                const uint32_t k =
                    (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                if (k > T) {
                    const int pos = atomicAdd(&staged, 1);
                    if (pos < kCap) arena_idx[pos] = (vb << 2) + e;
                }
            }
        }
        for (; v < vlen; v += kThreads) {
            const float4 a = __ldg(row4 + v);
            #pragma unroll
            for (int e = 0; e < 4; e++) {
                const uint32_t bits = e == 0 ? __float_as_uint(a.x)
                                    : e == 1 ? __float_as_uint(a.y)
                                    : e == 2 ? __float_as_uint(a.z)
                                             : __float_as_uint(a.w);
                nan_found |= (int)is_nan_bits32(bits);
                const uint32_t k =
                    (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
                if (k > T) {
                    const int pos = atomicAdd(&staged, 1);
                    if (pos < kCap) arena_idx[pos] = (v << 2) + e;
                }
            }
        }
        // scalar tail of the row (length not a multiple of 4)
        for (int i = (vlen << 2) + tx; i < (int)length; i += kThreads) {
            const float f = __ldg(input + i);
            const uint32_t bits = __float_as_uint(f);
            nan_found |= (int)is_nan_bits32(bits);
            const uint32_t k =
                (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            if (k > T) {
                const int pos = atomicAdd(&staged, 1);
                if (pos < kCap) arena_idx[pos] = i;
            }
        }
        __syncthreads();
        int n = staged;
        if (n < (int)topk) return false;      // underflow: coarse12 serves
        if (n <= kCap) break;                 // fits: on to the refinement
        if (retry == 1) return false;         // still over: coarse12 serves
        // 20-bit retry: a second sample pass resolves T inside bin12 and the
        // walk repeats with ~the sub-bin fewer candidates (~90 at 20 bits vs
        // the full ~1400 of the 12-bit bin).
        for (int b = tx; b < 256; b += kThreads) hist[b] = 0;
        __syncthreads();
        const uint32_t prefix = (uint32_t)bin12;
        for (int v2 = tx; v2 < svec; v2 += kThreads) {
            const float4 a = __ldg(row4 + v2);
            const uint32_t k0 = refine_bin(a.x), k1 = refine_bin(a.y),
                           k2 = refine_bin(a.z), k3 = refine_bin(a.w);
            if ((k0 >> 20) == prefix) atomicAdd(&hist[(k0 >> 12) & 0xffu], 1);
            if ((k1 >> 20) == prefix) atomicAdd(&hist[(k1 >> 12) & 0xffu], 1);
            if ((k2 >> 20) == prefix) atomicAdd(&hist[(k2 >> 12) & 0xffu], 1);
            if ((k3 >> 20) == prefix) atomicAdd(&hist[(k3 >> 12) & 0xffu], 1);
        }
        for (int i = (svec << 2) + tx; i < sample_len; i += kThreads) {
            const uint32_t k = refine_bin(__ldg(input + i));
            if ((k >> 20) == prefix) atomicAdd(&hist[(k >> 12) & 0xffu], 1);
        }
        __syncthreads();
        suffix256(hist, tx);
        {
            const int remain = m_sample - hist[257];
            const int sub = find_crossing(hist, remain, hist + 256, tx);
            if (tx == 0)
                sh_thr = (int)(((uint32_t)bin12 << 8 | (uint32_t)sub) << 12);
        }
        __syncthreads();
        T = (uint32_t)sh_thr;
        __syncthreads();
    }

    if (nan_flag != nullptr && __syncthreads_or(nan_found) != 0) {
        if (tx == 0) atomicOr(nan_flag, 1);
    }
    __syncthreads();
    int n = staged;

    // ── refinement: four rounds of 8 bits (offsets 24, 16, 8, 0) ───────────
    // Keys re-gathered from the row (L2-hot).  Rounds alternate
    // arena -> scratch -> arena -> scratch, never overlapping; boundary
    // counts (<= n <= kCap) always fit the scratch.
    int remaining = (int)topk;
    const int *src = arena_idx;
    int *dst = scratch_idx;
    #pragma unroll
    for (int round = 0; round < 4; round++) {
        const int off = 24 - round * 8;
        for (int b = tx; b < 256; b += kThreads) hist[b] = 0;
        if (tx == 0) bcount = 0;
        __syncthreads();
        for (int i = tx; i < n; i += kThreads) {
            const uint32_t key = refine_bin(__ldg(input + src[i]));
            atomicAdd(&hist[(key >> off) & 0xffu], 1);
        }
        __syncthreads();
        suffix256(hist, tx);
        {
            const int t_ = find_crossing(hist, remaining, hist + 256, tx);
            if (tx == 0) {
                sh_thr = t_;
                sh_above = t_ < 255 ? hist[t_ + 1] : 0;
            }
            __syncthreads();
        }
        const int thr = sh_thr;
        if (round < 3) {
            for (int i = tx; i < n; i += kThreads) {
                const int idx = src[i];
                const uint32_t key = refine_bin(__ldg(input + idx));
                const int sub = (int)((key >> off) & 0xffu);
                if (sub > thr) {
                    output[atomicAdd(&counter, 1)] = idx;
                } else if (sub == thr) {
                    const int pos = atomicAdd(&bcount, 1);
                    dst[pos] = idx;   // pos < n <= kCap always
                }
            }
            __syncthreads();
            remaining -= sh_above;
            n = bcount;
            const int *tmp_ = src;
            src = dst;
            dst = (int *)tmp_;
            __syncthreads();
        } else {
            // final round: emit strictly-above, then fill the rest of the
            // tied boundary arbitrarily (equal full keys: contract-valid).
            for (int i = tx; i < n; i += kThreads) {
                const int idx = src[i];
                const uint32_t key = refine_bin(__ldg(input + idx));
                if ((int)((key >> off) & 0xffu) > thr)
                    output[atomicAdd(&counter, 1)] = idx;
            }
            __syncthreads();
            for (int i = tx; i < n; i += kThreads) {
                const int idx = src[i];
                const uint32_t key = refine_bin(__ldg(input + idx));
                if ((int)((key >> off) & 0xffu) == thr) {
                    const int pos = atomicAdd(&counter, 1);
                    if (pos < (int)topk) output[pos] = idx;
                }
            }
            __syncthreads();
        }
    }
    return true;
}

// One CTA per row, mirroring `topk_coarse12_kernel` -- including its inline
// fallback: a row the sample route declines is served by `topk_coarse12_row`
// in the same launch.
template <int kThreads_>
__global__ __launch_bounds__(kThreads_) void topk_sample_kernel(
    const float *__restrict__ scores, const int32_t *__restrict__ lengths,
    int32_t *__restrict__ out, int32_t *__restrict__ nan_flags, int topk, int B,
    int64_t stride, int default_length) {
    static_assert(kThreads_ == kThreads,
                  "`topk_sample_row` walks the row at the namespace's kThreads");
    const int row = blockIdx.x;
    if (row >= B) return;
    const int length =
        lengths == nullptr ? default_length : __ldg(lengths + row);
    const float *row_input = scores + (int64_t)row * stride;
    int32_t *row_out = out + (int64_t)row * topk;
    int32_t *row_nan = nan_flags == nullptr ? nullptr : nan_flags + row;

    if (length <= topk) {
        for (int i = threadIdx.x; i < topk; i += kThreads)
            row_out[i] = i < length ? i : -1;
        return;
    }
    if (!topk_sample_row(row_input, row_out, (uint32_t)length,
                         (uint32_t)topk, row_nan)) {
        topk_select_coarse12::topk_coarse12_row(row_input, row_out,
                                                (uint32_t)length,
                                                (uint32_t)topk, row_nan);
    }
}

// The launcher that owns this kernel's one `<<<>>>`.
template <int kThreads_>
inline cudaError_t launch_sample_kernel(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream) {
    if (topk > kMaxTopK) return cudaErrorInvalidValue;
    static bool init = false;
    if (!init) {
        cudaError_t err = cudaFuncSetAttribute(
            topk_sample_kernel<kThreads_>,
            cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBytes);
        if (err != cudaSuccess) return err;
        init = true;
    }
    topk_sample_kernel<kThreads_>
        <<<B, kThreads_, kSmemBytes, stream>>>(scores, lengths, out, nan_flags,
                                               topk, B, stride, default_length);
    return cudaGetLastError();
}

extern template cudaError_t launch_sample_kernel<kThreads>(
    const float *, const int32_t *, int32_t *, int32_t *, int, int, int64_t,
    int, cudaStream_t);

inline cudaError_t launch_topk_sample(
    const float *scores, const int32_t *lengths, int32_t *out, int32_t *nan_flags,
    int B, int topk, int64_t stride, int default_length, cudaStream_t stream)
{
    return launch_sample_kernel<kThreads>(scores, lengths, out, nan_flags, B,
                                          topk, stride, default_length, stream);
}

// The route's launch and emit, mirroring `topk_select_coarse12::launch_route`:
// the kernel answers in row columns, staged through the dispatcher's scratch,
// and the contract half widens/orders them -- with the same NaN-table
// requirement, since this kernel also answers a row without ranking it.
template <typename OutIdxT>
inline void launch_route(const deep_select::RowParams &params, uint32_t batches,
                         cudaStream_t stream, bool sorted_index,
                         bool sorted_value, bool return_value, int block) {
    int32_t *const staging = params.coarse12_cols;
    if (staging != nullptr) {
        const cudaError_t rc = launch_topk_sample(
            (const float *)params.input, params.end_ptr, staging,
            params.scan_flags, (int)batches, (int)params.topk,
            (int64_t)(params.stride_input_batch / sizeof(float)),
            (int)params.vocab_size, stream);
        if (rc == cudaSuccess) {
            deep_select::RowParams merged = params;
            merged.preselected = staging;
            merged.nan_flags = params.scan_flags;
            deep_select::detail::launch_typed_radix<float, OutIdxT>(
                merged, batches, stream, sorted_index, sorted_value, return_value,
                block, /*preselected=*/true);
            return;
        }
    }
    // A launch that did not happen leaves the buffer untouched, so the row
    // path answers instead -- the same fallback the coarse12 route takes.
    deep_select::detail::launch_typed_radix<float, OutIdxT>(
        params, batches, stream, sorted_index, sorted_value, return_value, block,
        /*preselected=*/false);
}

}  // namespace topk_select_sample
