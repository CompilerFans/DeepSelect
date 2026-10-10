// Does warp-aggregated slot allocation beat one shared atomic per element?
// This is the collect's shape: N elements, 1024 threads, each above-threshold
// element takes a slot from one shared counter and stores to output[slot].
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <maca_bfloat16.h>

// (a) today's shape: one atomicAdd per element
__global__ void k_plain(const uint32_t* __restrict__ keys, int32_t* out, uint32_t len,
                        uint32_t threshold, uint32_t* total) {
    __shared__ uint32_t s_counter;
    if (threadIdx.x == 0) s_counter = 0;
    __syncthreads();
    for (uint32_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
        if (keys[idx] > threshold) out[atomicAdd(&s_counter, 1u)] = (int32_t)idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) total[blockIdx.x] = s_counter;
}
// (b) warp-aggregated: one atomic per warp per iteration
__global__ void k_agg(const uint32_t* __restrict__ keys, int32_t* out, uint32_t len,
                      uint32_t threshold, uint32_t* total) {
    __shared__ uint32_t s_counter;
    if (threadIdx.x == 0) s_counter = 0;
    __syncthreads();
    for (uint32_t idx = threadIdx.x; idx < len; idx += blockDim.x) {
        const bool hit = keys[idx] > threshold;
        const unsigned long long live = __activemask();
        const unsigned long long m = __ballot_sync(live, (int)hit);
        const unsigned rank = (unsigned)__popcll(m & ((1ull << (threadIdx.x & 63)) - 1ull));
        const unsigned cnt  = (unsigned)__popcll(m);
        unsigned base = 0;
        if (rank == 0 && cnt) base = atomicAdd(&s_counter, cnt);
        base = (unsigned)__shfl_sync(live, (int)base, (int)(threadIdx.x & ~63));
        if (hit) out[base + rank] = (int32_t)idx;
    }
    __syncthreads();
    if (threadIdx.x == 0) total[blockIdx.x] = s_counter;
}
static const uint32_t* g_keys; static int32_t* g_out; static uint32_t* g_tot;
static uint32_t g_len = 8192, g_thr = 0;
static void a() { k_plain<<<6, 1024>>>(g_keys, g_out, g_len, g_thr, g_tot); }
static void b() { k_agg<<<6, 1024>>>(g_keys, g_out, g_len, g_thr, g_tot); }
template <typename F> static float rate(int iters, F body) {
    for (int i = 0; i < 5; i++) body();
    cudaDeviceSynchronize();
    cudaEvent_t x, y; cudaEventCreate(&x); cudaEventCreate(&y);
    cudaEventRecord(x);
    for (int i = 0; i < iters; i++) body();
    cudaEventRecord(y); cudaEventSynchronize(y);
    float ms = 0; cudaEventElapsedTime(&ms, x, y);
    return ms * 1000.0f / iters;
}
int main() {
    cudaSetDevice(0);
    const uint32_t N = 1u << 20;
    uint32_t* keys; cudaMalloc(&keys, N * 4); cudaMalloc(&g_out, N * 4);
    cudaMalloc(&g_tot, 64);
    uint32_t* h = (uint32_t*)malloc(N * 4); uint32_t s = 7;
    for (uint32_t i = 0; i < N; i++) { s = s * 1664525u + 1013904223u; h[i] = s; }
    cudaMemcpy(keys, h, N * 4, cudaMemcpyHostToDevice); g_keys = keys; free(h);
    // threshold chosen so ~1/16 of elements hit, i.e. ~512 of 8192 per CTA
    printf("%-26s %9s %9s\n", "case", "plain us", "agg us");
    for (int frac : {2, 16, 256}) {          // 1/frac of elements take a slot
        g_thr = 0xffffffffu - (uint32_t)(0xffffffffu / frac);
        for (uint32_t L : {8192u, 65536u, 262144u}) {
            g_len = L;
            char nm[64]; snprintf(nm, sizeof nm, "len=%-7u 1/%d hit", L, frac);
            printf("%-26s %9.2f %9.2f\n", nm, rate(1000, a), rate(1000, b));
        }
    }
    return 0;
}
