// Bisect: why does the 4096-bin + key-transform walk run at 185 GB/s while the
// fp8 probe's 256-bin arm C runs at 683 GB/s on the same part?
// Arms: {256, 4096} bins x {raw, xform} x {1,2} outstanding, 112 blocks.
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1600 hist_bisect.cu -o /tmp/hb
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)
constexpr int kThreads = 512;

// arm R256: fp8 probe's exact arm C -- 256 bins, bin = high byte of each u16
__global__ void __launch_bounds__(kThreads) walk_r256(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[256];
    for (int t = threadIdx.x; t < 256; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; j++) atomicAdd(&s_hist[h[j] >> 8], 1u);
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// arm X256: 256 bins, production key transform, bin = key >> 8 per half-word
__global__ void __launch_bounds__(kThreads) walk_x256(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[256];
    for (int t = threadIdx.x; t < 256; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t kp =
                w[j] ^ ((((w[j] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
            atomicAdd(&s_hist[(kp >> 8) & 0xFFu], 1u);
            atomicAdd(&s_hist[kp >> 24], 1u);
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// arm R4096: 4096 bins, no transform (bin = (h[j]*77) & 4095 -- spread, no xform)
__global__ void __launch_bounds__(kThreads) walk_r4096(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; j++) atomicAdd(&s_hist[(h[j] * 77u) >> 4 & 0xFFFu], 1u);
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// arm X4096: production shape -- 4096 bins + key transform
__global__ void __launch_bounds__(kThreads) walk_x4096(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec;
         i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
        for (int j = 0; j < 4; j++) {
            const uint32_t kp =
                w[j] ^ ((((w[j] >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
            atomicAdd(&s_hist[(kp >> 4) & 0x0FFFu], 1u);
            atomicAdd(&s_hist[kp >> 20], 1u);
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

__global__ void init_uniform(uint4* in, size_t n_vec) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_vec) return;
    uint4 v;
    v.x = (uint32_t)(4 * i + 0) * 2654435761u;
    v.y = (uint32_t)(4 * i + 1) * 2654435761u;
    v.z = (uint32_t)(4 * i + 2) * 2654435761u;
    v.w = (uint32_t)(4 * i + 3) * 2654435761u;
    in[i] = v;
}

typedef void (*kern_t)(const uint4*, size_t, uint32_t*);
float time_it(kern_t k, const uint4* in, size_t n_vec, uint32_t* out,
              int blocks, int reps) {
    cudaEvent_t t0, t1; CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    k<<<blocks, kThreads>>>(in, n_vec, out);
    CK(cudaEventRecord(t0));
    for (int r = 0; r < reps; r++) k<<<blocks, kThreads>>>(in, n_vec, out);
    CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
    CK(cudaGetLastError()); cudaEventDestroy(t0); cudaEventDestroy(t1);
    return ms / reps;
}

int main(int argc, char** argv) {
    const size_t bytes = (argc > 1 ? atoi(argv[1]) : 256) * (size_t)1048576;
    const size_t n_vec = bytes / sizeof(uint4);
    uint4* in; uint32_t* out;
    CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&out, 4096));
    init_uniform<<<(unsigned)((n_vec + 255) / 256), 256>>>(in, n_vec);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const struct { const char* name; kern_t k; } arms[] = {
        {"r256", walk_r256}, {"x256", walk_x256},
        {"r4096", walk_r4096}, {"x4096", walk_x4096},
    };
    printf("%8s %10s %10s\n", "arm", "ms", "GB/s");
    for (int rep = 0; rep < 2; rep++)
        for (auto& a : arms) {
            float ms = time_it(a.k, in, n_vec, out, 112, 10);
            printf("%8s %10.3f %10.1f\n", a.name, ms, bytes/(ms*1e-3)/1e9);
        }
    cudaFree(in); cudaFree(out);
    return 0;
}
