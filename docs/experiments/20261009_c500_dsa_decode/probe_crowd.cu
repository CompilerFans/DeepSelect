// Is the collect's cost the serialized staging counter?  Then it must scale
// with how many members the threshold bin holds -- which is exactly what the
// merge's input (a row's extreme tail) maximizes.
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <maca_bfloat16.h>
#include "radix_core.cuh"
static const maca_bfloat16* g_in; static int32_t* g_out;
static const size_t S512 = rk::kCoarse12HistBytes + 4 * 512;
template <int TOPK>
__global__ void k_b1(const maca_bfloat16* __restrict__ in, int32_t* out, uint32_t len) {
    rk::radix_topk_row_bf16_b<1024, false, false>(
        in + (size_t)blockIdx.x * len, out + (size_t)blockIdx.x * TOPK, len, (uint32_t)TOPK);
}
template <typename F> static float rate(int iters, F body) {
    for (int i = 0; i < 5; i++) body();
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) body();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / iters;
}
static uint32_t g_len = 8192;
static void run() { k_b1<512><<<6, 1024, S512>>>(g_in, g_out, g_len); }

int main() {
    cudaSetDevice(0);
    const int kRows = 64; const uint32_t kMaxLen = 1u << 20;
    size_t n = (size_t)kMaxLen * kRows;
    maca_bfloat16* in; int32_t* out;
    cudaMalloc(&in, n * 2); cudaMalloc(&out, (size_t)kRows * 4096 * 4);
    cudaFuncSetAttribute(k_b1<512>, cudaFuncAttributeMaxDynamicSharedMemorySize, S512);
    maca_bfloat16* hb = (maca_bfloat16*)malloc(n * 2);
    g_out = out;

    struct Case { const char* name; int kind; };
    Case cases[] = {{"gaussian", 0}, {"uniform keys", 1}, {"all equal", 2}, {"two values", 3}};
    printf("%-14s %8s\n", "data", "us");
    for (Case c : cases) {
        uint32_t s = 12345;
        for (size_t i = 0; i < n; i++) {
            float v;
            if (c.kind == 0) {
                s = s * 1664525u + 1013904223u; float u1 = ((s >> 8) & 0xffffff) / (float)0x1000000;
                s = s * 1664525u + 1013904223u; float u2 = ((s >> 8) & 0xffffff) / (float)0x1000000;
                v = sqrtf(-2.0f * logf(u1 + 1e-9f)) * cosf(6.2831853f * u2) * 3.0f;
            } else if (c.kind == 1) {
                s = s * 1664525u + 1013904223u; v = (float)((s >> 8) & 0xffff) / 65536.0f * 8.0f - 4.0f;
            } else if (c.kind == 2) { v = 1.0f; }
            else { v = (i & 1) ? 1.5f : 1.0f; }
            hb[i] = __float2bfloat16(v);
        }
        cudaMemcpy(in, hb, n * 2, cudaMemcpyHostToDevice);
        g_in = in;
        for (uint32_t L : {8192u, 65536u}) { g_len = L; printf("%-14s len=%-7u %8.2f\n", c.name, L, rate(500, run)); }
    }
    return 0;
}
