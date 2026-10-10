// Barrier cost vs grid and block size -- is it the barrier, or the grid?
#include <cstdio>
#include <cuda_runtime.h>
template <int N> __global__ void k_bar(int* out) {
    int x = threadIdx.x;
    #pragma unroll 1
    for (int i = 0; i < N; i++) x += (int)__syncthreads_count(0);
    if (x == 0xdeadbeef) out[0] = x;      // one barrier per iteration, value unused
}
static int* g_out;
static int G = 6, B = 1024;
template <typename F>
static float rate(int iters, F body) {
    for (int i = 0; i < 5; i++) body();
    cudaDeviceSynchronize();
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) body();
    cudaEventRecord(b); cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    return ms * 1000.0f / iters;
}
int main() {
    cudaSetDevice(0); cudaMalloc(&g_out, 16);
    for (int g : {1, 6, 104}) {
        for (int blk : {128, 256, 512, 1024}) {
            G = g; B = blk;
            float t8  = rate(1000, [] { k_bar<8><<<G, B>>>(g_out); });
            float t72 = rate(1000, [] { k_bar<72><<<G, B>>>(g_out); });
            printf("grid=%-4d block=%-5d  N=8 %7.2f  N=72 %7.2f   per barrier %6.0f ns\n",
                   g, blk, t8, t72, (t72 - t8) / 64.0 * 1000.0);
        }
    }
    return 0;
}
