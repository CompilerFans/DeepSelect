// fp32 sampled-threshold one-pass prototype (LiteTopK sample-filter-select,
// arXiv 2607.11976; attribution per DeepSelect issue #13).
//
// One CTA per row: (1) histogram a contiguous V/16 sample at 12 bits, fold +
// narrow to the m-th-largest 12-bit bin, then a second sample pass resolves
// the threshold to 20 bits (T = floor of that key at bit 12); (2) ONE
// full-row walk staging {key > T} into the smem arena (idx32 + key32, cap
// 4096); (3) in-smem refinement: three rounds (8+8+4 bits) over the staged
// candidates reading keys from SMEM (no global re-gather), exact to the full
// key; true ties at the full key fill arbitrarily (contract-valid).  Falls
// back (reported, driver re-serves the row via deep_select) on staged < k
// (underflow), staged > 4096 (overflow), or a NaN vote.
//
// Build:  cucc -O3 --offload-arch=xcore1600 -shared -fPIC proto_sample.cu -o /tmp/proto.so
#include <cstdio>
#include <cstdint>

#ifndef DS_PROTO_THREADS
#define DS_PROTO_THREADS 640
#endif

constexpr int kThreads = DS_PROTO_THREADS;
constexpr int kCoarseBits = 12;
constexpr int kCoarseBins = 1 << kCoarseBits;   // 4096
constexpr int kCap = 4096;

__device__ __forceinline__ uint32_t key_of(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

__device__ __forceinline__ bool nan_bits(uint32_t b) {
    return (b & 0x7F800000u) == 0x7F800000u && (b & 0x007FFFFFu) != 0u;
}

// Suffix-cumsum of a 256-bin smem histogram, parallel: 64 lanes x 4 bins,
// local tail-sum, a 6-step warp scan (shfl_down, suffix form), add back.
// Lane l owns bins [4l, 4l+4); after the call h[i] holds count(bins >= i's
// bin... i.e. the downward suffix).  Must be called by the FIRST 64 lanes.
__device__ __forceinline__ void suffix256(int *h, int tx) {
    if (tx >= 64) return;
    const int base = (int)(tx * 4);
    int v0 = h[base], v1 = h[base + 1], v2 = h[base + 2], v3 = h[base + 3];
    // lane total, then exclusive suffix scan of lane totals
    int tot = v0 + v1 + v2 + v3;
    const unsigned long long m = 0xFFFFFFFFFFFFFFFFULL;
    int o;
    int suf = tot;
    #pragma unroll
    for (int d = 1; d < 64; d <<= 1) {
        o = __shfl_down_sync(m, suf, d);
        // out-of-range reads return the lane's OWN current value on MACA
        // (measured: shfl_down lane63 d1 = own) -- adding it doubles the
        // total.  The guard is on the add, not the shuffle.
        if (tx + d < 64) suf += o;
    }
    // suf = suffix sum including own; exclusive = suf - tot
    const int excl = suf - tot;
    h[base]     = excl + tot;          // suffix(>= base)
    h[base + 1] = excl + v1 + v2 + v3;
    h[base + 2] = excl + v2 + v3;
    h[base + 3] = excl + v3;
}


// After suffix256 (h[c] = count(>= c)): the unique bin where the count
// crosses m, found in parallel -- one thread per bin, one winner.  A serial
// 256-iteration scan here cost ~76 us/row over the refinement's four
// rounds (measured 2026-10-21, pod 30752).
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
// suffix-cumsum downward (thread 0), find the bin where the count crosses
// `m`, and narrow over its 16 sub-bins.  Returns the 12-bit bin via the
// shared `out` slots: [0] = bin12, [1] = count strictly above it.
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

// counters layout per row: [0] staged count, [1] path (1 ok, 2 fallback),
// [2] nan votes, [3] total ticks, [4] sample ticks, [5] filter ticks,
// [6] refine ticks, [7] emitted count
extern "C" __global__ void __launch_bounds__(kThreads) proto_sample_topk(
    const float *__restrict__ input, int vocab, int topk, int m_sample,
    int sample_len, int *__restrict__ out_idx, int *__restrict__ counters) {
    extern __shared__ int smem[];
    int *arena_idx = smem;                 // 4096  (16 KB)
    int *scratch_idx = smem + 4096;        // 4096  (16 KB; 32 KB total)
    // The scratch must take ANY boundary count: round 0's boundary (the
    // k-th key's 8-bit bin among the candidates) is ~1400-2900, over the
    // old 2048 -- and boundary <= n <= 4096 by construction, so 4096 makes
    // a spill impossible.  32 KB + ~1 KB static keeps 3 CTAs/SM.
    // keys are RE-GATHERED from the row during refinement (L2-hot: the
    // filter walk has just read them) instead of a smem key arena, so the
    // kernel stays at 3 CTAs/SM (24 KB + ~1 KB static).
    __shared__ int hist[256 + 16];
    __shared__ int counter, staged, bcount, t_key, t_above, nan_count,
        path_code;
    __shared__ long long t0, t1, t2;
    const int tx = threadIdx.x;
    const float *row = input + (size_t)blockIdx.x * vocab;
    int *out = out_idx + (size_t)blockIdx.x * topk;

    if (tx == 0) {
        counter = 0;
        staged = 0;
        nan_count = 0;
        path_code = 1;
        t0 = clock64();
    }
    // sample phase: 12-bit histogram of the contiguous prefix, in arena_idx
    for (int b = tx; b < kCoarseBins; b += kThreads) arena_idx[b] = 0;
    __syncthreads();
    const int svec = sample_len >> 2;
    const float4 *row4 = reinterpret_cast<const float4 *>(row);
    for (int v = tx; v < svec; v += kThreads) {
        const float4 a = __ldg(row4 + v);
        atomicAdd(&arena_idx[key_of(a.x) >> 20], 1);
        atomicAdd(&arena_idx[key_of(a.y) >> 20], 1);
        atomicAdd(&arena_idx[key_of(a.z) >> 20], 1);
        atomicAdd(&arena_idx[key_of(a.w) >> 20], 1);
    }
    // scalar tail of the sample
    for (int i = (svec << 2) + tx; i < sample_len; i += kThreads)
        atomicAdd(&arena_idx[key_of(__ldg(row + i)) >> 20], 1);
    __syncthreads();

    // fold + suffix + narrow (parallel fold, serial 256-bin suffix)
    int bin12, above_sample;
    {
        narrow12(arena_idx, m_sample, hist, hist + 256, tx);
        bin12 = hist[256];
        above_sample = hist[257];
    }
    (void)above_sample;

    // The threshold is the 12-bit bin FLOOR: candidates = count(key > T) =
    // count above bin12 (~1100-1900) + bin12's full population (~1400) =
    // ~2500-3300, inside the 4096 arena.  (An earlier draft resolved T to 20
    // bits with a second sample pass; the arithmetic above makes it
    // unnecessary -- the 20-bit sub-bin is only ~90 of the bin's ~1400.)
    const uint32_t T = (uint32_t)bin12 << 20;
    if (tx == 0 && blockIdx.x < 4) {
        counters[blockIdx.x * 12 + 8] = (int)(uint32_t)bin12;
        counters[blockIdx.x * 12 + 9] = (int)((uint32_t)T >> 12);
        counters[blockIdx.x * 12 + 10] = m_sample;
    }
    if (tx == 0) t1 = clock64();

    // filter + stage walk: one full-row pass, {key > T} into the arena
    const int vlen = vocab >> 2;
    int nan_local = 0;
    uint32_t bad_bits = 0;
    for (int v = tx; v < vlen; v += kThreads) {
        const float4 a = __ldg(row4 + v);
        #pragma unroll
        for (int e = 0; e < 4; e++) {
            const uint32_t bits = e == 0 ? __float_as_uint(a.x)
                                : e == 1 ? __float_as_uint(a.y)
                                : e == 2 ? __float_as_uint(a.z)
                                         : __float_as_uint(a.w);
            if (nan_bits(bits)) { nan_local = 1; if (bad_bits == 0) bad_bits = bits; }
            const uint32_t k = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
            if (k > T) {
                const int pos = atomicAdd(&staged, 1);
                if (pos < kCap) arena_idx[pos] = (v << 2) + e;
            }
        }
    }
    for (int i = (vlen << 2) + tx; i < vocab; i += kThreads) {
        const uint32_t bits = __float_as_uint(__ldg(row + i));
        nan_local |= (int)nan_bits(bits);
        const uint32_t k = (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
        if (k > T) {
            const int pos = atomicAdd(&staged, 1);
            if (pos < kCap) arena_idx[pos] = i;
        }
    }
    if (__syncthreads_or(nan_local) != 0) {
        if (tx == 0) {
            atomicAdd(&nan_count, 1);
            counters[blockIdx.x * 12 + 11] = (int)bad_bits;
        }
    }
    __syncthreads();
    if (tx == 0) t2 = clock64();
    int n = staged;
    if (tx == 0 && (n < topk || n > kCap || nan_count != 0)) path_code = 2;
    __syncthreads();
    if (path_code == 2) {
        if (tx == 0) {
            counters[blockIdx.x * 12 + 0] = n;
            counters[blockIdx.x * 12 + 1] = 2;
            counters[blockIdx.x * 12 + 2] = nan_count;
            counters[blockIdx.x * 12 + 3] = (int)(clock64() - t0);
            counters[blockIdx.x * 12 + 4] = (int)(t1 - t0);
            counters[blockIdx.x * 12 + 5] = (int)(t2 - t1);
            counters[blockIdx.x * 12 + 6] = (int)(clock64() - t2);
            counters[blockIdx.x * 12 + 7] = 0;
        }
        return;
    }

    // refinement: FOUR rounds of 8 bits (offsets 24, 16, 8, 0) over the
    // staged candidates.  Keys are re-gathered from the row (L2-hot).  Round
    // sources alternate arena -> scratch -> arena -> scratch, never
    // overlapping; boundary counts (<= n <= 4096) always fit the 4096
    // scratch -- the spill arm is dead and kept only as a report path.
    int remaining = topk;
    const int *src = arena_idx;
    int *dst = scratch_idx;
    #pragma unroll
    for (int round = 0; round < 4; round++) {
        const int off = 24 - round * 8;
        for (int b = tx; b < 256; b += kThreads) hist[b] = 0;
        if (tx == 0) bcount = 0;
        __syncthreads();
        for (int i = tx; i < n; i += kThreads) {
            const uint32_t key = key_of(__ldg(row + src[i]));
            atomicAdd(&hist[(key >> off) & 0xffu], 1);
        }
        __syncthreads();
        suffix256(hist, tx);
        {
            const int t_ = find_crossing(hist, remaining, hist + 256, tx);
            if (tx == 0) {
                hist[256] = t_;
                hist[257] = t_ < 255 ? hist[t_ + 1] : 0;
            }
            __syncthreads();
        }
        const int thr = hist[256];
        if (round < 3) {
            for (int i = tx; i < n; i += kThreads) {
                const int idx = src[i];
                const uint32_t key = key_of(__ldg(row + idx));
                const int sub = (int)((key >> off) & 0xffu);
                if (sub > thr) {
                    out[atomicAdd(&counter, 1)] = idx;
                } else if (sub == thr) {
                    const int pos = atomicAdd(&bcount, 1);
                    if (pos < 4096) dst[pos] = idx;
                }
            }
            __syncthreads();
            if (bcount > 4096) {   // scratch spill: report, driver re-serves
                if (tx == 0) {
                    counters[blockIdx.x * 12 + 0] = n;
                    counters[blockIdx.x * 12 + 1] = 2;
                    counters[blockIdx.x * 12 + 2] = nan_count;
                    counters[blockIdx.x * 12 + 3] = (int)(clock64() - t0);
                    counters[blockIdx.x * 12 + 4] = (int)(t1 - t0);
                    counters[blockIdx.x * 12 + 5] = (int)(t2 - t1);
                    counters[blockIdx.x * 12 + 6] = (int)(clock64() - t2);
                    counters[blockIdx.x * 12 + 7] = counter;
                }
                return;
            }
            remaining -= hist[257];
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
                const uint32_t key = key_of(__ldg(row + idx));
                if ((int)((key >> off) & 0xffu) > thr)
                    out[atomicAdd(&counter, 1)] = idx;
            }
            __syncthreads();
            for (int i = tx; i < n; i += kThreads) {
                const int idx = src[i];
                const uint32_t key = key_of(__ldg(row + idx));
                if ((int)((key >> off) & 0xffu) == thr) {
                    const int pos = atomicAdd(&counter, 1);
                    if (pos < topk) out[pos] = idx;
                }
            }
            __syncthreads();
        }
    }

    if (tx == 0) {
        counters[blockIdx.x * 12 + 0] = staged;
        counters[blockIdx.x * 12 + 1] = path_code;
        counters[blockIdx.x * 12 + 2] = nan_count;
        counters[blockIdx.x * 12 + 3] = (int)(clock64() - t0);
        counters[blockIdx.x * 12 + 4] = (int)(t1 - t0);
        counters[blockIdx.x * 12 + 5] = (int)(t2 - t1);
        counters[blockIdx.x * 12 + 6] = (int)(clock64() - t2);
        counters[blockIdx.x * 12 + 7] = counter;
    }
}

// Host-side launcher so the 48 KB dynamic arena is one call for ctypes.
extern "C" int proto_launch(const void *input, int rows, int vocab, int topk,
                            int m_sample, int sample_len, void *out_idx,
                            void *counters) {
    static bool configured = false;
    if (!configured) {
        const cudaError_t e = cudaFuncSetAttribute(
            proto_sample_topk, cudaFuncAttributeMaxDynamicSharedMemorySize,
            32 * 1024);
        if (e != cudaSuccess) return (int)e;
        configured = true;
    }
    proto_sample_topk<<<rows, kThreads, 32 * 1024>>>(
        reinterpret_cast<const float *>(input), vocab, topk, m_sample,
        sample_len, reinterpret_cast<int *>(out_idx),
        reinterpret_cast<int *>(counters));
    return (int)cudaGetLastError();
}
