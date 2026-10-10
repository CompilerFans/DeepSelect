// What does the coarse12 row cost as a function of row length, at a fixed grid?
// `_b` is the kernel both chunk stages rank with, so this is the merge's own
// cost curve, not a model of it.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <maca_bfloat16.h>
#include "radix_core.cuh"

template <int TOPK>
__global__ void rank_kernel(const maca_bfloat16* __restrict__ in, int32_t* out,
                            uint32_t len) {
    rk::radix_topk_row_bf16_b<1024, false, false>(
        in + (size_t)blockIdx.x * len, out + (size_t)blockIdx.x * TOPK, len,
        (uint32_t)TOPK);
}

__global__ void k_empty() {}

static int g_grid;
static uint32_t g_len;
static const maca_bfloat16* g_in;
static int32_t* g_out;

template <int TOPK>
static float time_rank(int grid, uint32_t len, int iters,
                       const maca_bfloat16* in, int32_t* out) {
    const size_t smem = rk::kCoarse12HistBytes + sizeof(uint32_t) * (size_t)TOPK;
    cudaFuncSetAttribute(rank_kernel<TOPK>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    for (int i = 0; i < 3; i++)
        rank_kernel<TOPK><<<grid, 1024, smem>>>(in, out, len);
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++)
        rank_kernel<TOPK><<<grid, 1024, smem>>>(in, out, len);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / iters;
}

static float time_empty(int grid, int iters) {
    for (int i = 0; i < 3; i++) k_empty<<<grid, 1024>>>();
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) k_empty<<<grid, 1024>>>();
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / iters;
}

int main() {
    cudaSetDevice(0);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, 0);
    const int sm = prop.multiProcessorCount;
    printf("device=%s sm=%d\n", prop.name, sm);

    const uint32_t kMaxLen = 1u << 20;
    const int kRows = 256;
    size_t n = (size_t)kMaxLen * kRows;
    maca_bfloat16* in; int32_t* out;
    cudaMalloc(&in, n * sizeof(maca_bfloat16));
    cudaMalloc(&out, (size_t)kRows * 2048 * sizeof(int32_t));
    // Normal-ish data, so the coarse level sees the crowding a real row has.
    { float* h = (float*)malloc(n * sizeof(float));
      uint32_t s = 12345;
      for (size_t i = 0; i < n; i++) {
          s = s * 1664525u + 1013904223u;
          float u1 = ((s >> 8) & 0xffffff) / (float)0x1000000;
          s = s * 1664525u + 1013904223u;
          float u2 = ((s >> 8) & 0xffffff) / (float)0x1000000;
          h[i] = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.2831853f * u2) * 3.0f;
      }
      maca_bfloat16* hb = (maca_bfloat16*)malloc(n * 2);
      for (size_t i = 0; i < n; i++) hb[i] = __float2bfloat16(h[i]);
      cudaMemcpy(in, hb, n * 2, cudaMemcpyHostToDevice);
      free(h); free(hb); }

    // Launch floor at this block size / smem.
    for (int g : {1, 6, 16, 48, 96, 104})
        printf("empty            grid=%-4d %8.2f us\n", g, time_empty(g, 1000));
    printf("\n");

    const uint32_t lens[] = {2048, 4096, 8192, 16384, 32768, 65536, 131072, 262144};
    for (int g : {1, 6, 16, 48, 96}) {
        printf("--- grid=%d  (topk=512) ---\n", g);
        for (uint32_t L : lens) {
            g_grid = g; g_len = L; g_in = in; g_out = out;
            float us = time_rank<512>(g, L, 500, in, out);
            printf("  len=%-8u %8.2f us   %6.2f us/CTA  %7.2f GB/s/CTA\n",
                   L, us, us / g, (double)L * 2.0 / (us / g) / 1000.0);
        }
        printf("\n");
    }
    return 0;
}
