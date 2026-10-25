#include <cstdio>
#include <cuda_runtime.h>
__global__ void spin(unsigned long long *out, int n) {
    unsigned long long acc = threadIdx.x + 1, t0 = clock64();
    for (int i = 0; i < n; i++) acc = acc * 6364136223846793005ull + 1442695040888963407ull;
    if (threadIdx.x == 0) out[blockIdx.x] = clock64() - t0;
    if (acc == 1) out[0] = acc;
}
int main() {
    unsigned long long *o; cudaMalloc(&o, 8);
    cudaEvent_t a, b; cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e30f;
    for (int i = 0; i < 5; i++) {
        cudaEventRecord(a); spin<<<1, 32>>>(o, 2000000); cudaEventRecord(b);
        cudaEventSynchronize(b); float ms; cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    unsigned long long c; cudaMemcpy(&c, o, 8, cudaMemcpyDeviceToHost);
    printf("cycles %llu  wall %.3f ms  -> timer %.2f GHz  (%.2f cycles/mul-add)\n",
           c, best, c / (best * 1e6), c / 2000000.0);
    return 0;
}
