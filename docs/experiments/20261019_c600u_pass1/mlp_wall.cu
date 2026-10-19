// MLP scaling probe: read wall as a function of outstanding uint4 loads per
// thread, on the C600-U.  The bf16 row kernel's walk keeps exactly ONE uint4 in
// flight per thread; the fp8 ceiling probe's read arm used four.  This sweep
// quantifies what a software-pipelined walk could buy back.
//
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1600 mlp_wall.cu -o /tmp/mlp_wall
// Run:   /tmp/mlp_wall [size_mb]
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

template <int OUT>
__global__ void __launch_bounds__(512) walk_read(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + (OUT - 1) * stride < n_vec; i += OUT * stride) {
        uint4 v[OUT];
#pragma unroll
        for (int j = 0; j < OUT; j++) v[j] = __ldg(in + i + j * stride);
#pragma unroll
        for (int j = 0; j < OUT; j++) acc ^= v[j].x ^ v[j].y ^ v[j].z ^ v[j].w;
    }
    for (; i < n_vec; i += stride) { uint4 a = __ldg(in + i); acc ^= a.x; }
    if (acc == 0xFFFFFFFFu) out[blockIdx.x] = acc;
}

template <int OUT>
float run_one(const uint4* in, size_t n_vec, uint32_t* out, int blocks, int reps) {
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    walk_read<OUT><<<blocks, 512>>>(in, n_vec, out);   // warmup
    CK(cudaEventRecord(t0));
    for (int r = 0; r < reps; r++) walk_read<OUT><<<blocks, 512>>>(in, n_vec, out);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
    CK(cudaGetLastError());
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / reps;
}

int main(int argc, char** argv) {
    const size_t bytes = (argc > 1 ? atoi(argv[1]) : 256) * (size_t)1048576;
    const size_t n_vec = bytes / sizeof(uint4);
    uint4* in; uint32_t* out;
    CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&out, 4096));
    CK(cudaMemset(in, 0x5a, bytes));
    const int reps = 10;
    printf("%8s %8s %10s %10s\n", "blocks", "outst", "ms", "GB/s");
    const int block_grid[] = {28, 56, 112, 224, 448};
    for (int bi = 0; bi < 5; bi++) {
        const int blocks = block_grid[bi];
        float ms;
        ms = run_one<1>(in, n_vec, out, blocks, reps);
        printf("%8d %8d %10.3f %10.1f\n", blocks, 1, ms, bytes / (ms * 1e-3) / 1e9);
        ms = run_one<2>(in, n_vec, out, blocks, reps);
        printf("%8d %8d %10.3f %10.1f\n", blocks, 2, ms, bytes / (ms * 1e-3) / 1e9);
        ms = run_one<4>(in, n_vec, out, blocks, reps);
        printf("%8d %8d %10.3f %10.1f\n", blocks, 4, ms, bytes / (ms * 1e-3) / 1e9);
        ms = run_one<8>(in, n_vec, out, blocks, reps);
        printf("%8d %8d %10.3f %10.1f\n", blocks, 8, ms, bytes / (ms * 1e-3) / 1e9);
    }
    cudaFree(in); cudaFree(out);
    return 0;
}
