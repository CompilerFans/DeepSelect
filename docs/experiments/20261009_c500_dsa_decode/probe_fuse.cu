// Two questions, both decisive for whether folding the merge into stage 1 pays:
//   1. Is the ~13 us fixed cost per _b *invocation* or per *kernel*?
//   2. Does the coarse12 level (4096 bins) own that fixed cost, or does the
//      static-k row (_k, 256 bins) pay it too?
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <maca_bfloat16.h>
#include "radix_core.cuh"

static const maca_bfloat16* g_in;
static int32_t* g_out;
static int g_grid = 6;
static uint32_t g_len = 8192;
static const size_t S512 = rk::kCoarse12HistBytes + 4 * 512;

template <int TOPK>
__global__ void k_b1(const maca_bfloat16* __restrict__ in, int32_t* out, uint32_t len) {
    rk::radix_topk_row_bf16_b<1024, false, false>(
        in + (size_t)blockIdx.x * len, out + (size_t)blockIdx.x * TOPK, len, (uint32_t)TOPK);
}
// Two _b calls in one kernel, on two halves of the row, two output buffers.
template <int TOPK>
__global__ void k_b2(const maca_bfloat16* __restrict__ in, int32_t* out, uint32_t len) {
    const size_t base = (size_t)blockIdx.x * len;
    rk::radix_topk_row_bf16_b<1024, false, false>(
        in + base, out + (size_t)blockIdx.x * TOPK, len, (uint32_t)TOPK);
    rk::radix_topk_row_bf16_b<1024, false, false>(
        in + base + len, out + (size_t)blockIdx.x * TOPK + TOPK, len, (uint32_t)TOPK);
}
template <int TOPK>
__global__ void k_k1(const maca_bfloat16* __restrict__ in, int32_t* out, uint32_t len) {
    rk::radix_topk_row_bf16_k<TOPK, 1024, false>(
        in + (size_t)blockIdx.x * len, out + (size_t)blockIdx.x * TOPK, len);
}

static float rate(int iters, void (*body)()) {
    for (int i = 0; i < 5; i++) body();
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) body();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / iters;
}
static void b_b1() { k_b1<512><<<g_grid, 1024, S512>>>(g_in, g_out, g_len); }
static void b_b2() { k_b2<512><<<g_grid, 1024, S512>>>(g_in, g_out, g_len); }
static void b_k1() { k_k1<512><<<g_grid, 1024, rk::kSMEM>>>(g_in, g_out, g_len); }
static void b_b1_x2() { b_b1(); b_b1(); }

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    printf("device=%s sm=%d\n", prop.name, prop.multiProcessorCount);
    const uint32_t kMaxLen = 1u << 20; const int kRows = 256;
    size_t n = (size_t)kMaxLen * kRows;
    maca_bfloat16* in; int32_t* out;
    cudaMalloc(&in, n * 2); cudaMalloc(&out, (size_t)kRows * 4096 * 4);
    { float* h = (float*)malloc(n * 4); uint32_t s = 12345;
      for (size_t i = 0; i < n; i++) { s = s * 1664525u + 1013904223u;
        float u1 = ((s >> 8) & 0xffffff) / (float)0x1000000;
        s = s * 1664525u + 1013904223u;
        float u2 = ((s >> 8) & 0xffffff) / (float)0x1000000;
        h[i] = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.2831853f * u2) * 3.0f; }
      maca_bfloat16* hb = (maca_bfloat16*)malloc(n * 2);
      for (size_t i = 0; i < n; i++) hb[i] = __float2bfloat16(h[i]);
      cudaMemcpy(in, hb, n * 2, cudaMemcpyHostToDevice); free(h); free(hb); }
    g_in = in; g_out = out;
    cudaFuncSetAttribute(k_b1<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, S512);
    cudaFuncSetAttribute(k_b2<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, S512);
    cudaFuncSetAttribute(k_k1<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, rk::kSMEM);

    printf("\n%-30s %9s %9s\n", "op (grid=6)", "us", "x/one");
    for (uint32_t L : {2048u, 8192u, 65536u}) {
        g_len = L;
        float one = rate(500, b_b1), two = rate(500, b_b2), x2 = rate(500, b_b1_x2);
        float kk  = rate(500, b_k1);
        printf("\nlen=%-7u  _b x1   %9.2f\n", L, one);
        printf("           _b x2 in one kernel %9.2f  (%+.2f over one)\n", two, two - one);
        printf("           _b x2, two launches %9.2f  (%+.2f over one)\n", x2, x2 - one);
        printf("           _k x1   %9.2f  (%+.2f vs _b)\n", kk, kk - one);
    }
    return 0;
}
