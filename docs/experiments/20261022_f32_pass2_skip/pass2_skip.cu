// Does the coarse12 second pass have to read the whole row?
//
// Pass 2 (the collect) reads every element of the row to act on the ~k+bin ones
// at or above the 12-bit threshold -- 2.6% of them at b4096 V131072 k2048.  The
// bf16 row already answers the same question with a block-max summary and a
// kept-block list (`radix_row_bf16.cuh`); this probe asks whether the fp32
// collect walk has the same structure to win, at 16/32/64-element blocks.
//
// Self-contained; compile on the pod in seconds:
//   cucc -O3 --offload-arch=xcore1600 -shared -fPIC pass2_skip.cu -o /tmp/ps.so
//
// Modes, all one CTA per row, 512 threads, 16 KB dynamic shared arena:
//   0 dense collect     -- coarse12 pass 2 as it is written today
//   1 listed, G=16      -- walk only the kept blocks
//   2 listed, G=32
//   3 listed, G=64
//   4 listed G=16, load-only (the listed walk's ceiling: no collect work)
//
// 2026-10-23: G=4 and G=8 added.  G=16 was chosen against G=32/G=64 before the
// finer direction had been measured; the finer it is, the less of the row is
// kept, and the summary that feeds it gets cheaper to compute (G=4 needs no
// shuffle at all).  So the trade is a bigger table and more list entries
// against a smaller pass-2 walk.
//   5 listed, G=4       -- one lane per float4
//   6 listed, G=8       -- two lanes per block
//
// The kept-block lists are built by the driver (torch) with the *true* 12-bit
// threshold bin, so this measures the walk, not the summary.  Production also
// has to build the summary in pass 1 and compact the list; the bf16 row pays
// that in `row_block_max` / `row_block_list` and its ledger is
// docs/experiments/20261021_blockhist/.
#include <cstdio>
#include <cstdint>

constexpr int kThreads = 512;
constexpr int kCoarseBins = 4096;
constexpr int kArenaCap = 2048;      // 16 KB / 8 B, the coarse12 arena

__device__ __forceinline__ uint32_t key_of(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

// The collect rule, shared by every mode: above the threshold goes to `out`,
// in the threshold bin goes to the arena and the refine histogram.
struct Collect {
    int *counter;      // smem, above-threshold count
    int *num_input;    // smem, threshold-bin count
    int *arena;        // smem, kArenaCap entries
    int *hist;         // smem, 256 bins
    int *out;          // global, this row's answer slots
    int thr;

    __device__ __forceinline__ void operator()(float v, int index) const {
        const uint32_t key = key_of(v);
        const int bin = (int)(key >> 20);
        if (bin > thr) {
            const int pos = atomicAdd(counter, 1);
            out[pos] = index;
        } else if (bin == thr) {
            const int pos = atomicAdd(num_input, 1);
            if (pos < kArenaCap) {
                arena[pos] = index;
                atomicAdd(&hist[(key >> 12) & 0xffu], 1);
            }
        }
    }
};

// Mode 0: the dense walk, spelled the way the kernel spells it (two float4 in
// flight where the register file affords it).
__global__ __launch_bounds__(kThreads) void dense_collect(
    const float *__restrict__ in, const int *__restrict__ thr_bin, int len,
    int out_stride, int *__restrict__ out, int *__restrict__ above,
    int *__restrict__ inbin) {
    extern __shared__ int smem[];
    int *arena = smem;                 // 2048
    int *hist = smem + kArenaCap;      // 256 (aliases nothing here)
    __shared__ int counter;
    __shared__ int num_input;
    const int row = blockIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(in + (int64_t)row * len);
    if (threadIdx.x == 0) { counter = 0; num_input = 0; }
    if (threadIdx.x < 256) hist[threadIdx.x] = 0;
    __syncthreads();

    const Collect c{&counter, &num_input, arena, hist,
                    out + (int64_t)row * out_stride, thr_bin[row]};
    const int vec_len = len >> 2;
    int vec = threadIdx.x;
    for (; vec + kThreads < vec_len; vec += 2 * kThreads) {
        const float4 a = __ldg(row4 + vec);
        const float4 b = __ldg(row4 + vec + kThreads);
        const int ia = vec * 4;
        const int ib = ia + kThreads * 4;
        c(a.x, ia); c(a.y, ia + 1); c(a.z, ia + 2); c(a.w, ia + 3);
        c(b.x, ib); c(b.y, ib + 1); c(b.z, ib + 2); c(b.w, ib + 3);
    }
    for (; vec < vec_len; vec += kThreads) {
        const float4 a = __ldg(row4 + vec);
        const int ia = vec * 4;
        c(a.x, ia); c(a.y, ia + 1); c(a.z, ia + 2); c(a.w, ia + 3);
    }
    __syncthreads();
    if (threadIdx.x == 0) { above[row] = counter; inbin[row] = num_input; }
}

// Modes 1-4: walk the kept-block list.  One group of G/4 lanes owns one block,
// so a warp covers 64/G * ... blocks per iteration and each group's four lanes
// read the block's G/4 float4 back to back.
template <int G, bool kLoadOnly>
__global__ __launch_bounds__(kThreads) void listed_collect(
    const float *__restrict__ in, const int *__restrict__ thr_bin,
    const int *__restrict__ list, const int *__restrict__ list_off,
    const int *__restrict__ list_len, int len, int out_stride,
    int *__restrict__ out, int *__restrict__ above, int *__restrict__ inbin) {
    extern __shared__ int smem[];
    int *arena = smem;
    int *hist = smem + kArenaCap;
    __shared__ int counter;
    __shared__ int num_input;
    const int row = blockIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(in + (int64_t)row * len);
    if (threadIdx.x == 0) { counter = 0; num_input = 0; }
    if (threadIdx.x < 256) hist[threadIdx.x] = 0;
    __syncthreads();

    const Collect c{&counter, &num_input, arena, hist,
                    out + (int64_t)row * out_stride, thr_bin[row]};
    constexpr int kLanesPerBlock = G / 4;              // 4, 8, 16
    constexpr int kGroups = kThreads / kLanesPerBlock; // 128, 64, 32
    const int group = threadIdx.x / kLanesPerBlock;
    const int lane_in_block = threadIdx.x % kLanesPerBlock;
    const int n = list_len[row];
    const int *row_list = list + list_off[row];
    uint32_t acc = 0;
    for (int j = group; j < n; j += kGroups) {
        const int block = row_list[j];
        const float4 v = __ldg(row4 + block * kLanesPerBlock + lane_in_block);
        if (kLoadOnly) {
            acc ^= key_of(v.x) ^ key_of(v.y) ^ key_of(v.z) ^ key_of(v.w);
        } else {
            const int ia = block * G + lane_in_block * 4;
            c(v.x, ia); c(v.y, ia + 1); c(v.z, ia + 2); c(v.w, ia + 3);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        // Load-only has no collect state: writing the (zero) counters here
        // would overwrite the previous mode's counts, which is how the first
        // version of this probe reported a spurious count mismatch.
        if (!kLoadOnly) {
            above[row] = counter;
            inbin[row] = num_input;
        } else if (acc == 0xDEADBEEFu) {
            above[row] = -1;   // keeps the walk alive; never taken
        }
    }
}

// ── the host-side entry points ─────────────────────────────────────────────

template <typename Kernel, typename Launch>
static float bench(Kernel k, Launch launch, int smem, int iters) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    cudaFuncSetAttribute(k, cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    for (int r = 0; r < 3; r++) launch();
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < iters; r++) launch();
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

extern "C" void probe_dense(const float *in, const int *thr, const int *list,
                            const int *off, const int *llen, int rows, int len,
                            int out_stride, int *out, int *above, int *inbin,
                            int iters) {
    const int smem = (kArenaCap + 256) * (int)sizeof(int);
    const float ms = bench(dense_collect,
                           [&] { dense_collect<<<rows, kThreads, smem>>>(
                                     in, thr, len, out_stride, out, above, inbin); },
                           smem, iters);
    const double gb = (double)rows * len * 4.0 / 1e9;
    printf("  dense collect                    %8.3f ms  %7.1f GB/s\n", ms,
           gb / (ms / 1e3));
}

// The listed walk at G=4 with two entries in flight per lane.  The walk is a
// dependent pair of loads -- `list[j]`, then `data[list[j]]` -- and at one
// entry per lane it measured 0.812 ms for 189 MB, i.e. latency, not bandwidth.
// Two independent entries per iteration give the scheduler two chains.
template <bool kLoadOnly>
__global__ __launch_bounds__(kThreads) void listed_collect_ilp2(
    const float *__restrict__ in, const int *__restrict__ thr_bin,
    const int *__restrict__ list, const int *__restrict__ list_off,
    const int *__restrict__ list_len, int len, int out_stride,
    int *__restrict__ out, int *__restrict__ above, int *__restrict__ inbin) {
    extern __shared__ int smem[];
    int *arena = smem;
    int *hist = smem + kArenaCap;
    __shared__ int counter;
    __shared__ int num_input;
    const int row = blockIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(in + (int64_t)row * len);
    if (threadIdx.x == 0) { counter = 0; num_input = 0; }
    if (threadIdx.x < 256) hist[threadIdx.x] = 0;
    __syncthreads();

    const Collect c{&counter, &num_input, arena, hist,
                    out + (int64_t)row * out_stride, thr_bin[row]};
    const int n = list_len[row];
    const int *row_list = list + list_off[row];
    uint32_t acc = 0;
    int j = threadIdx.x;
    for (; j + kThreads < n; j += 2 * kThreads) {
        const int b0 = row_list[j];
        const int b1 = row_list[j + kThreads];
        const float4 v0 = __ldg(row4 + b0);
        const float4 v1 = __ldg(row4 + b1);
        if (kLoadOnly) {
            acc ^= key_of(v0.x) ^ key_of(v1.y) ^ key_of(v0.z) ^ key_of(v1.w);
        } else {
            const int i0 = b0 * 4;
            const int i1 = b1 * 4;
            c(v0.x, i0); c(v0.y, i0 + 1); c(v0.z, i0 + 2); c(v0.w, i0 + 3);
            c(v1.x, i1); c(v1.y, i1 + 1); c(v1.z, i1 + 2); c(v1.w, i1 + 3);
        }
    }
    for (; j < n; j += kThreads) {
        const int b0 = row_list[j];
        const float4 v0 = __ldg(row4 + b0);
        if (kLoadOnly) {
            acc ^= key_of(v0.x) ^ key_of(v0.y) ^ key_of(v0.z) ^ key_of(v0.w);
        } else {
            const int i0 = b0 * 4;
            c(v0.x, i0); c(v0.y, i0 + 1); c(v0.z, i0 + 2); c(v0.w, i0 + 3);
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        if (!kLoadOnly) { above[row] = counter; inbin[row] = num_input; }
        else if (acc == 0xDEADBEEFu) above[row] = -1;
    }
}

// One call per granularity: each G has its own kept-block list, so the
// dispatch has to be C-side (a template cannot be chosen from Python).
extern "C" void probe_listed(const float *in, const int *thr, const int *list,
                             const int *off, const int *llen, int rows, int len,
                             int out_stride, int *out, int *above, int *inbin,
                             int iters, int g, long kept, long total_blocks) {
    const int smem = (kArenaCap + 256) * (int)sizeof(int);
    const double gb = (double)rows * len * 4.0 / 1e9;
    const double frac = (double)kept / (double)total_blocks;
    const double kept_gb = gb * frac;
    float ms_collect = -1.f, ms_load = -1.f;
    switch (g) {
    case 4:
        ms_collect = bench(listed_collect<4, false>, [&] {
            listed_collect<4, false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect<4, true>, [&] {
            listed_collect<4, true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    case 44:  // G=4, two entries in flight
        ms_collect = bench(listed_collect_ilp2<false>, [&] {
            listed_collect_ilp2<false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect_ilp2<true>, [&] {
            listed_collect_ilp2<true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    case 8:
        ms_collect = bench(listed_collect<8, false>, [&] {
            listed_collect<8, false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect<8, true>, [&] {
            listed_collect<8, true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    case 16:
        ms_collect = bench(listed_collect<16, false>, [&] {
            listed_collect<16, false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect<16, true>, [&] {
            listed_collect<16, true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    case 32:
        ms_collect = bench(listed_collect<32, false>, [&] {
            listed_collect<32, false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect<32, true>, [&] {
            listed_collect<32, true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    case 64:
        ms_collect = bench(listed_collect<64, false>, [&] {
            listed_collect<64, false><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        ms_load = bench(listed_collect<64, true>, [&] {
            listed_collect<64, true><<<rows, kThreads, smem>>>(
                in, thr, list, off, llen, len, out_stride, out, above, inbin); },
            smem, iters);
        break;
    default:
        printf("  (no G=%d)\n", g);
        return;
    }
    // GB/s is quoted two ways: against the whole row (comparable to the dense
    // walk, which is what the decision is about) and against the bytes the
    // listed walk actually touches (its own efficiency).
    const char *label = g == 44 ? "4x2" : nullptr;
    char gbuf[8];
    if (!label) { snprintf(gbuf, sizeof(gbuf), "%d", g); label = gbuf; }
    printf("  listed G=%-3s collect  %8.3f ms  %7.1f GB/s row  %7.1f GB/s kept"
           "  (kept %ld/%ld = %.1f%%)\n",
           label, ms_collect, gb / (ms_collect / 1e3), kept_gb / (ms_collect / 1e3),
           kept, total_blocks, 100.0 * frac);
    printf("  listed G=%-3s load-only %7.3f ms  %7.1f GB/s row  %7.1f GB/s kept"
           "  (walk ceiling)\n",
           label, ms_load, gb / (ms_load / 1e3), kept_gb / (ms_load / 1e3));
}

// ── the shipping design: a block-max table, no compaction ──────────────────
//
// The listed walks above need a *compacted* list, which costs an atomic per
// kept block and a second table.  A table-driven walk needs neither: pass 2
// strides over `bmax[]` (one uint16 per 4 float4 = 16 elements), skips a block
// whose max bin is under the threshold, and reads the block only otherwise.
// The table read is coalesced (2 B per lane) and is 1/32 of the row's traffic.

constexpr unsigned long long kFullMask = ~0ull;

// Two xor steps over four consecutive lanes give the 16-element block's max
// bin.  The walk hands lane `t` float4 `t`, so lanes 4k..4k+3 are block k.
__device__ __forceinline__ int block_bin_max(int b0, int b1, int b2, int b3) {
    int m = max(max(b0, b1), max(b2, b3));
    m = max(m, __shfl_xor_sync(kFullMask, m, 1));
    m = max(m, __shfl_xor_sync(kFullMask, m, 2));
    return m;
}

// The same four-lane max with the platform's permute instead of the shuffle
// wrapper: `sm_bperm_b32` is the instruction both spellings emit, so the
// difference is the wrapper's per-call lane-index arithmetic (~5 insns).
// The hardware index is byte-addressed (dest[n] = data[index[n]/4 % 64]).
__device__ __forceinline__ int block_bin_max_bp(int b0, int b1, int b2, int b3) {
    int m = max(max(b0, b1), max(b2, b3));
    const int self = (int)threadIdx.x;
    m = max(m, __builtin_mxc_bsm_bpermute(((self ^ 1) << 2), m));
    m = max(m, __builtin_mxc_bsm_bpermute(((self ^ 2) << 2), m));
    return m;
}

__device__ __forceinline__ int bin_of(float v) { return (int)(key_of(v) >> 20); }

// Mode: pass 1 as the kernel writes it today (histogram only).
template <int kSummary>   // 0 none, 1 shfl_xor G=16, 2 bsm_bpermute G=16,
                          // 3 G=4 (no shuffle), 4 G=8 (one xor step)
__global__ __launch_bounds__(kThreads) void pass1_hist(
    const float *__restrict__ in, int len, int *__restrict__ hist_out,
    unsigned short *__restrict__ bmax) {
    extern __shared__ int wide[];                       // 4096 bins
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(in + (int64_t)row * len);
    const int vec_len = len >> 2;
    // One table entry per 16/8/4 elements, so the row stride follows the grain.
    const int nblk = kSummary == 4 ? (vec_len >> 1)
                   : kSummary == 3 ? vec_len
                                   : (vec_len >> 2);
    unsigned short *row_bmax = kSummary ? bmax + (int64_t)row * nblk : nullptr;
    for (int b = tid; b < kCoarseBins; b += kThreads) wide[b] = 0;
    __syncthreads();

    int vec = tid;
    for (; vec + kThreads < vec_len; vec += 2 * kThreads) {
        const float4 va = __ldg(row4 + vec);
        const float4 vb = __ldg(row4 + vec + kThreads);
        atomicAdd(&wide[bin_of(va.x)], 1);
        atomicAdd(&wide[bin_of(va.y)], 1);
        atomicAdd(&wide[bin_of(va.z)], 1);
        atomicAdd(&wide[bin_of(va.w)], 1);
        atomicAdd(&wide[bin_of(vb.x)], 1);
        atomicAdd(&wide[bin_of(vb.y)], 1);
        atomicAdd(&wide[bin_of(vb.z)], 1);
        atomicAdd(&wide[bin_of(vb.w)], 1);
        if constexpr (kSummary == 1) {
            const int ma = block_bin_max(bin_of(va.x), bin_of(va.y), bin_of(va.z), bin_of(va.w));
            const int mb = block_bin_max(bin_of(vb.x), bin_of(vb.y), bin_of(vb.z), bin_of(vb.w));
            if ((tid & 3) == 0) {
                row_bmax[(vec >> 2)] = (unsigned short)ma;
                row_bmax[((vec + kThreads) >> 2)] = (unsigned short)mb;
            }
        }
        if constexpr (kSummary == 2) {
            const int ma = block_bin_max_bp(bin_of(va.x), bin_of(va.y), bin_of(va.z), bin_of(va.w));
            const int mb = block_bin_max_bp(bin_of(vb.x), bin_of(vb.y), bin_of(vb.z), bin_of(vb.w));
            if ((tid & 3) == 0) {
                row_bmax[(vec >> 2)] = (unsigned short)ma;
                row_bmax[((vec + kThreads) >> 2)] = (unsigned short)mb;
            }
        }
        if constexpr (kSummary == 3) {
            // G=4: the lane's own float4 is the block -- no shuffle, 4x the stores.
            row_bmax[vec] = (unsigned short)max(max(bin_of(va.x), bin_of(va.y)),
                                                max(bin_of(va.z), bin_of(va.w)));
            row_bmax[vec + kThreads] = (unsigned short)max(max(bin_of(vb.x), bin_of(vb.y)),
                                                           max(bin_of(vb.z), bin_of(vb.w)));
        }
        if constexpr (kSummary == 4) {
            // G=8: two lanes per block, one xor step, even lanes store.
            int ma = max(max(bin_of(va.x), bin_of(va.y)), max(bin_of(va.z), bin_of(va.w)));
            int mb = max(max(bin_of(vb.x), bin_of(vb.y)), max(bin_of(vb.z), bin_of(vb.w)));
            ma = max(ma, __shfl_xor_sync(kFullMask, ma, 1));
            mb = max(mb, __shfl_xor_sync(kFullMask, mb, 1));
            if ((tid & 1) == 0) {
                row_bmax[vec >> 1] = (unsigned short)ma;
                row_bmax[(vec + kThreads) >> 1] = (unsigned short)mb;
            }
        }
    }
    for (; vec < vec_len; vec += kThreads) {
        const float4 va = __ldg(row4 + vec);
        atomicAdd(&wide[bin_of(va.x)], 1);
        atomicAdd(&wide[bin_of(va.y)], 1);
        atomicAdd(&wide[bin_of(va.z)], 1);
        atomicAdd(&wide[bin_of(va.w)], 1);
        if constexpr (kSummary == 1) {
            const int ma = block_bin_max(bin_of(va.x), bin_of(va.y), bin_of(va.z), bin_of(va.w));
            if ((tid & 3) == 0) row_bmax[(vec >> 2)] = (unsigned short)ma;
        }
        if constexpr (kSummary == 2) {
            const int ma = block_bin_max_bp(bin_of(va.x), bin_of(va.y), bin_of(va.z), bin_of(va.w));
            if ((tid & 3) == 0) row_bmax[(vec >> 2)] = (unsigned short)ma;
        }
        if constexpr (kSummary == 3) {
            row_bmax[vec] = (unsigned short)max(max(bin_of(va.x), bin_of(va.y)),
                                                max(bin_of(va.z), bin_of(va.w)));
        }
        if constexpr (kSummary == 4) {
            int ma = max(max(bin_of(va.x), bin_of(va.y)), max(bin_of(va.z), bin_of(va.w)));
            ma = max(ma, __shfl_xor_sync(kFullMask, ma, 1));
            if ((tid & 1) == 0) row_bmax[vec >> 1] = (unsigned short)ma;
        }
    }
    __syncthreads();
    if (tid < 256) {                                    // fold, 16 sub-bins each
        int s = 0;
#pragma unroll
        for (int j = 0; j < 16; ++j) s += wide[tid * 16 + j];
        hist_out[(int64_t)row * 256 + tid] = s;
    }
}

// Mode: pass 2 walking the table, one lane per 16-element block.
template <bool kLoadOnly>
__global__ __launch_bounds__(kThreads) void table_collect(
    const float *__restrict__ in, const int *__restrict__ thr_bin,
    const unsigned short *__restrict__ bmax, int len, int out_stride,
    int *__restrict__ out, int *__restrict__ above, int *__restrict__ inbin) {
    extern __shared__ int smem[];
    int *arena = smem;
    int *hist = smem + kArenaCap;
    __shared__ int counter;
    __shared__ int num_input;
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const float4 *row4 = reinterpret_cast<const float4 *>(in + (int64_t)row * len);
    const int vec_len = len >> 2;
    const int nblk = vec_len >> 2;
    const unsigned short *row_bmax = bmax + (int64_t)row * nblk;
    if (tid == 0) { counter = 0; num_input = 0; }
    if (tid < 256) hist[tid] = 0;
    __syncthreads();

    const Collect c{&counter, &num_input, arena, hist,
                    out + (int64_t)row * out_stride, thr_bin[row]};
    const int thr = thr_bin[row];
    uint32_t acc = 0;
#pragma unroll 2
    for (int blk = tid; blk < nblk; blk += kThreads) {
        if ((int)row_bmax[blk] < thr) continue;
        const float4 *p = row4 + blk * 4;
        const float4 a = __ldg(p);
        const float4 b = __ldg(p + 1);
        const float4 d = __ldg(p + 2);
        const float4 e = __ldg(p + 3);
        if (kLoadOnly) {
            acc ^= key_of(a.x) ^ key_of(b.y) ^ key_of(d.z) ^ key_of(e.w);
        } else {
            const int ia = blk * 16;
            c(a.x, ia);     c(a.y, ia + 1);  c(a.z, ia + 2);  c(a.w, ia + 3);
            c(b.x, ia + 4); c(b.y, ia + 5);  c(b.z, ia + 6);  c(b.w, ia + 7);
            c(d.x, ia + 8); c(d.y, ia + 9);  c(d.z, ia + 10); c(d.w, ia + 11);
            c(e.x, ia + 12); c(e.y, ia + 13); c(e.z, ia + 14); c(e.w, ia + 15);
        }
    }
    // The float4s past the last whole block (vec_len % 4 of them) and any
    // scalar remainder are not covered by the table; walk them densely.
    for (int vec = (nblk * 4) + tid; vec < vec_len; vec += kThreads) {
        const float4 a = __ldg(row4 + vec);
        if (!kLoadOnly) {
            const int ia = vec * 4;
            c(a.x, ia); c(a.y, ia + 1); c(a.z, ia + 2); c(a.w, ia + 3);
        }
    }
    __syncthreads();
    if (tid == 0) {
        if (!kLoadOnly) { above[row] = counter; inbin[row] = num_input; }
        else if (acc == 0xDEADBEEFu) above[row] = -1;
    }
}

extern "C" void probe_pass1(const float *in, int rows, int len,
                            unsigned short *bmax, int *hist_out, int iters,
                            int with_summary) {
    const int smem = kCoarseBins * (int)sizeof(int);
    float ms;
    const char *name = "hist only";
    if (with_summary == 1) {
        name = "+summary shfl";
        ms = bench(pass1_hist<1>, [&] {
            pass1_hist<1><<<rows, kThreads, smem>>>(in, len, hist_out, bmax); },
            smem, iters);
    } else if (with_summary == 2) {
        name = "+summary bperm";
        ms = bench(pass1_hist<2>, [&] {
            pass1_hist<2><<<rows, kThreads, smem>>>(in, len, hist_out, bmax); },
            smem, iters);
    } else if (with_summary == 3) {
        name = "+summary G=4";
        ms = bench(pass1_hist<3>, [&] {
            pass1_hist<3><<<rows, kThreads, smem>>>(in, len, hist_out, bmax); },
            smem, iters);
    } else if (with_summary == 4) {
        name = "+summary G=8";
        ms = bench(pass1_hist<4>, [&] {
            pass1_hist<4><<<rows, kThreads, smem>>>(in, len, hist_out, bmax); },
            smem, iters);
    } else {
        ms = bench(pass1_hist<0>, [&] {
            pass1_hist<0><<<rows, kThreads, smem>>>(in, len, hist_out, bmax); },
            smem, iters);
    }
    const double gb = (double)rows * len * 4.0 / 1e9;
    printf("  pass1 %-15s         %8.3f ms  %7.1f GB/s\n",
           name, ms, gb / (ms / 1e3));
}

extern "C" void probe_table(const float *in, const int *thr,
                            const unsigned short *bmax, int rows, int len,
                            int out_stride, int *out, int *above, int *inbin,
                            int iters) {
    const int smem = (kArenaCap + 256) * (int)sizeof(int);
    const float ms = bench(table_collect<false>, [&] {
        table_collect<false><<<rows, kThreads, smem>>>(
            in, thr, bmax, len, out_stride, out, above, inbin); }, smem, iters);
    const float ms_l = bench(table_collect<true>, [&] {
        table_collect<true><<<rows, kThreads, smem>>>(
            in, thr, bmax, len, out_stride, out, above, inbin); }, smem, iters);
    const double gb = (double)rows * len * 4.0 / 1e9;
    printf("  table collect                    %8.3f ms  %7.1f GB/s\n", ms, gb / (ms / 1e3));
    printf("  table load-only                  %8.3f ms  %7.1f GB/s\n", ms_l, gb / (ms_l / 1e3));
}
