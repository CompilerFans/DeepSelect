// One-pass no-histogram topk walk probe (the upstream DeepSelect dataflow).
//
// Arms over the production walk geometry (one CTA per row, 512 threads,
// 8-lane groups covering 64-element blocks):
//
//   A  bare read (wall reference)
//   B  + key xform + compare against a FIXED smem threshold (no staging)
//   C  + stage above-threshold elements (atomic into arena) -- isolates the
//      staging cost with the threshold pre-set to the true k-th
//   D  the full one-pass dataflow: running threshold + compact when the
//      candidate arena fills past k + B2
//
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1600 onepass_probe.cu -o /tmp/opp
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

constexpr int kThreads = 512;
constexpr int kTopK = 512;
constexpr int kB2 = 512;                    // compact trigger: k + B2
constexpr int kArena = kTopK + kB2 + 64;    // one block of slack
constexpr int kRadix = 256;

__global__ void init_keys(uint16_t* p, size_t n) {
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride)
        p[i] = (uint16_t)((i * 2654435761u) >> 13);
}
__device__ __forceinline__ uint32_t xform16(uint16_t h) {
    return (uint32_t)(h ^ (((h >> 15) * 0x7FFFu) + 0x8000u));
}
// the production's packed word transform, one uint32 = two keys
__device__ __forceinline__ uint32_t xform32(uint32_t w) {
    return w ^ ((((w >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
}

template <int ARM>
__global__ void __launch_bounds__(kThreads) onepass(
    const uint16_t* __restrict__ rows, uint32_t row_elems, uint32_t n_rows,
    uint32_t* __restrict__ arena_out, uint32_t true_thresh_key,
    uint32_t* __restrict__ dbg) {
    __shared__ uint32_t s_arena[kArena];
    __shared__ uint32_t s_count;
    __shared__ uint32_t s_thresh;
    __shared__ uint32_t s_hist[kRadix + 1];
    __shared__ uint32_t s_pick;

    const uint32_t row = blockIdx.x;
    const uint16_t* input = rows + (size_t)row * row_elems;
    const uint32_t tx = threadIdx.x;
    if (tx == 0) {
        s_count = 0;
        s_thresh = ARM >= 3 ? 0u : true_thresh_key;   // D starts at -inf
    }
    __syncthreads();

    // block order: affine permutation over the 64-element blocks (odd stride =
    // bijective), the random-order stand-in
    const uint32_t nblk = row_elems >> 6;
    const uint32_t group = tx >> 3, lane = tx & 7u;
    const uint32_t step = kThreads / 8;

    for (uint32_t j0 = group; j0 < nblk; j0 += step) {
        const uint32_t blk = (j0 * 8191u) % nblk;   // shuffled-ish block order
        const uint32_t base = blk << 6;
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + base) + lane);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
        if constexpr (ARM == 0) {
            // bare: fold the loads
            if ((v.x ^ v.w) == 0xDEADBEEFu) dbg[blockIdx.x] = 1;
            continue;
        }
        uint32_t kv[8];
#pragma unroll
        for (int i = 0; i < 4; i++) {
            kv[2 * i] = w[i] & 0xFFFFu;      // raw u16 keys, uniform
            kv[2 * i + 1] = w[i] >> 16;
        }
        const uint32_t T = s_thresh;   // smem broadcast
        if constexpr (ARM == 1) {
            // filter only
            uint32_t m = 0;
#pragma unroll
            for (int i = 0; i < 8; i++) m |= (kv[i] > T) ? kv[i] : 0u;
            if (m == 0xFFFFFFFFu) dbg[blockIdx.x] = 1;   // never taken on random u16
            continue;
        }
        // stage the members above T
#pragma unroll
        for (int i = 0; i < 8; i++) {
            if (kv[i] > T) {
                const uint32_t pos = atomicAdd(&s_count, 1u);
                if (pos < kArena) s_arena[pos] = kv[i];
            }
        }
        if constexpr (ARM == 3) {
            // compact when the arena crosses k + B2 (checked once per group
            // iteration, by the whole CTA -- upstream's granularity)
            __syncthreads();
            if (s_count < kTopK + kB2) continue;
            // radix-select the arena to the top-k: two 8-bit rounds,
            // histogram parallel, pick by one thread scanning 256 bins.
            const uint32_t m = s_count < kArena ? s_count : kArena;
            uint32_t prefix_keep = 0;   // elements strictly above the running prefix
            uint32_t prefix = 0;        // selected key prefix (8 bits per round)
            for (int round = 0; round < 2; round++) {
                const uint32_t shift = round == 0 ? 8 : 0;
                if (tx < kRadix + 1) s_hist[tx] = 0;
                __syncthreads();
                for (uint32_t i = tx; i < m; i += kThreads) {
                    const uint32_t key = s_arena[i];
                    if (round == 0 || (key >> 8) == prefix)
                        atomicAdd(&s_hist[(key >> shift) & 0xFFu], 1u);
                }
                __syncthreads();
                if (tx == 0) {
                    const uint32_t need = kTopK - prefix_keep;
                    uint32_t acc = 0, pick = 0, above = 0;
                    for (int b = 255; b >= 0; b--) {
                        if (acc + s_hist[b] >= need) { pick = b; break; }
                        acc += s_hist[b];
                    }
                    s_pick = pick;
                    s_hist[kRadix] = prefix_keep + acc;  // count strictly above pick
                }
                __syncthreads();
                prefix_keep = s_hist[kRadix];
                prefix = round == 0 ? s_pick : ((prefix << 8) | s_pick);
            }
            if (tx == 0) {
                s_thresh = prefix;
                s_count = 0;
            }
            __syncthreads();
        }
    }
    __syncthreads();
    if (ARM >= 2 && tx == 0) {
        arena_out[blockIdx.x] = s_count;
        dbg[blockIdx.x] = s_thresh;
    }
}

int main(int argc, char** argv) {
    const uint32_t n_rows = 4096;
    const uint32_t row_elems = 1048576;
    const size_t bytes = (size_t)n_rows * row_elems * 2;
    uint16_t* in; uint32_t* arena; uint32_t* dbg;
    CK(cudaMalloc(&in, bytes));
    CK(cudaMalloc(&arena, n_rows * 4)); CK(cudaMalloc(&dbg, n_rows * 4));
    init_keys<<<8192, 256>>>(in, bytes / 2);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    // the true k-th key for memset data: all elements identical -> any key;
    // use a mid key so the filter is active but stages ~nothing for B/C.
    const uint32_t true_t = 65535u - (65535u * kTopK) / 1048576u;
    printf("true_thresh_key=%08x (all-equal data: filter at T-1 stages all)\n",
           true_t);
    auto run = [&](int arm, const char* name) {
        cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        float ms;
        auto launch = [&] {
            switch (arm) {
            case 0: onepass<0><<<n_rows, kThreads>>>(in, row_elems, n_rows, arena, true_t - 1, dbg); break;
            case 1: onepass<1><<<n_rows, kThreads>>>(in, row_elems, n_rows, arena, true_t - 1, dbg); break;
            case 2: onepass<2><<<n_rows, kThreads>>>(in, row_elems, n_rows, arena, true_t, dbg); break;
            case 3: onepass<3><<<n_rows, kThreads>>>(in, row_elems, n_rows, arena, 0, dbg); break;
            }
        };
        launch();
        CK(cudaEventRecord(t0));
        for (int r = 0; r < 3; r++) launch();
        CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
        CK(cudaEventElapsedTime(&ms, t0, t1)); CK(cudaGetLastError());
        ms /= 3;
        printf("arm %s: %9.3f ms  %7.1f GB/s\n", name, ms, bytes / (ms * 1e-3) / 1e9);
    };
    run(0, "bare      ");
    run(1, "filter    ");
    run(2, "filter+stage");
    run(3, "full onepass");
    return 0;
}
