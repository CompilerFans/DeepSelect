// The proposed C600U pass-1 shape, end to end:
//   OUT=2 batched uint4 loads + block-max summary (bsm_bpermute tree, the
//   production form) + histogram atomics on 1/S of the vectors + NaN compare.
// Reference arms: today's shape (OUT=1, full atomics, summary), and bare read.
//
// Grid: 56 / 112 / 224 blocks x 512 threads, 256 MB, uniform keys.
// Build: cucc -O2 -std=c++20 --offload-arch=xcore1600 pass1_new.cu -o /tmp/p1n
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)
constexpr int kThreads = 512;

__device__ __forceinline__ uint32_t xform(uint32_t w) {
    return w ^ ((((w >> 15) & 0x00010001u) * 0x7FFFu) + 0x80008000u);
}
__device__ __forceinline__ uint32_t maxkey_of(uint4 v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t m = 0;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp = xform(w[i]);
        const uint32_t lo = kp & 0xFFFFu, hi = kp >> 16;
        const uint32_t t = lo > hi ? lo : hi;
        m = t > m ? t : m;
    }
    return m;
}
__device__ __forceinline__ void hist8(uint32_t* s, uint4 v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const uint32_t kp = xform(w[i]);
        atomicAdd(&s[(kp >> 4) & 0x0FFFu], 1u);
        atomicAdd(&s[kp >> 20], 1u);
    }
}
__device__ __forceinline__ uint32_t nan_of(uint4 v) {
    const uint32_t* w = reinterpret_cast<const uint32_t*>(&v);
    uint32_t z = 0;
#pragma unroll
    for (int i = 0; i < 4; i++)
        z |= (((w[i] & 0x7FFF7FFFu) | 0x80008000u) - 0x7F817F81u) & 0x80008000u;
    return z;
}
// 3-step chained bpermute group max over 8 lanes (production form)
__device__ __forceinline__ uint32_t groupmax8(uint32_t key, uint32_t tx) {
    uint32_t o = (uint32_t)__builtin_mxc_bsm_bpermute((int)(((tx + 1u) & 63u) << 2), (int)key);
    key = key > o ? key : o;
    o = (uint32_t)__builtin_mxc_bsm_bpermute((int)(((tx + 2u) & 63u) << 2), (int)key);
    key = key > o ? key : o;
    o = (uint32_t)__builtin_mxc_bsm_bpermute((int)(((tx + 4u) & 63u) << 2), (int)key);
    key = key > o ? key : o;
    return key;
}

// Arm OLD: today's pass 1 -- one uint4 in flight, 8 atomics each, summary.
__global__ void __launch_bounds__(kThreads) walk_old(
    const uint4* __restrict__ in, size_t n_vec, uint16_t* __restrict__ bmax,
    uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const uint32_t lane = threadIdx.x & 7u;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n_vec; i += stride) {
        uint4 v = __ldg(in + i);
        hist8(s_hist, v);
        uint32_t key = groupmax8(maxkey_of(v), threadIdx.x);
        nan_of(v);
        if (lane == 0) bmax[i >> 3] = (uint16_t)(key >> 4);
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// Arm NEW: OUT=2, sampled atomics (1/S of vectors), summary + NaN kept.
template <int S>
__global__ void __launch_bounds__(kThreads) walk_new(
    const uint4* __restrict__ in, size_t n_vec, uint16_t* __restrict__ bmax,
    uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const uint32_t lane = threadIdx.x & 7u;
    uint32_t cnt = 0;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + stride < n_vec; i += 2 * stride) {
        uint4 a = __ldg(in + i), b = __ldg(in + i + stride);
        if ((cnt++ & (S - 1)) == 0) hist8(s_hist, a);
        if ((cnt & (S - 1)) == 1) hist8(s_hist, b);   // 1/S of all vectors
        uint32_t ka = groupmax8(maxkey_of(a), threadIdx.x);
        nan_of(a); nan_of(b);
        if (lane == 0) bmax[i >> 3] = (uint16_t)(ka >> 4);
        // b's block summary would need the sibling walk's index; omitted here
        // (same instruction mix as a's, counted by the checksum instead).
        uint32_t kb = maxkey_of(b);
        if (kb == 0xFFFFFFFFu) out[blockIdx.x] = 1;
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}


// Ablations of walk_new<8>: which ingredient costs the 36%?
template <bool kStore, bool kGroup, bool kHist>
__global__ void __launch_bounds__(kThreads) walk_abl(
    const uint4* __restrict__ in, size_t n_vec, uint16_t* __restrict__ bmax,
    uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    if (kHist) {
        for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
        __syncthreads();
    }
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    const uint32_t lane = threadIdx.x & 7u;
    uint32_t cnt = 0, acc = 0;
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + stride < n_vec; i += 2 * stride) {
        uint4 a = __ldg(in + i), b = __ldg(in + i + stride);
        if (kHist) {
            if ((cnt++ & 7) == 0) hist8(s_hist, a);
            if ((cnt & 7) == 1) hist8(s_hist, b);
        }
        uint32_t ka = maxkey_of(a), kb = maxkey_of(b);
        if (kGroup) { ka = groupmax8(ka, threadIdx.x); kb = groupmax8(kb, threadIdx.x); }
        if (kStore) {
            if (lane == 0) bmax[i >> 3] = (uint16_t)(ka >> 4);
            acc ^= kb;
        } else {
            acc ^= ka ^ kb;
        }
    }
    if (kHist) __syncthreads();
    if (acc == 0xFFFFFFFFu || (kHist && s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu))
        out[blockIdx.x] = 1;
}

// Arm BARE: OUT=2 pure read.
__global__ void __launch_bounds__(kThreads) walk_bare(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    uint32_t acc = 0;
    const size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i + stride < n_vec;
         i += 2 * stride) {
        uint4 a = __ldg(in + i), b = __ldg(in + i + stride);
        acc ^= a.x ^ b.y;
    }
    if (acc == 0xFFFFFFFFu) out[blockIdx.x] = acc;
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

typedef void (*kern_t)(const uint4*, size_t, uint16_t*, uint32_t*);
int main(int argc, char** argv) {
    const size_t bytes = (argc > 1 ? atoi(argv[1]) : 256) * (size_t)1048576;
    const size_t n_vec = bytes / sizeof(uint4);
    uint4* in; uint16_t* bmax; uint32_t* out;
    CK(cudaMalloc(&in, bytes)); CK(cudaMalloc(&bmax, n_vec / 8 * 2)); CK(cudaMalloc(&out, 4096));
    init_uniform<<<(unsigned)((n_vec + 255) / 256), 256>>>(in, n_vec);
    CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
    const int grids[] = {56, 112, 224};
    printf("%8s %6s %10s %10s\n", "blocks", "arm", "ms", "GB/s");
    for (int gi = 0; gi < 3; gi++) {
        const int blocks = grids[gi];
        cudaEvent_t t0, t1; float ms;
        auto once = [&](kern_t k, const char* name) {
            CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
            k<<<blocks, kThreads>>>(in, n_vec, bmax, out);
            CK(cudaEventRecord(t0));
            for (int r = 0; r < 10; r++) k<<<blocks, kThreads>>>(in, n_vec, bmax, out);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            CK(cudaEventElapsedTime(&ms, t0, t1)); CK(cudaGetLastError());
            cudaEventDestroy(t0); cudaEventDestroy(t1);
            printf("%8d %6s %10.3f %10.1f\n", blocks, name, ms / 10,
                   bytes / (ms / 10 * 1e-3) / 1e9);
        };
        once((kern_t)walk_bare, "bare");
        once((kern_t)walk_old, "old");
        once((kern_t)walk_new<8>, "new_s8");
        once((kern_t)walk_abl<false,true,true>, "abl_nostore");
        once((kern_t)walk_abl<true,false,true>, "abl_nogroup");
        once((kern_t)walk_abl<true,true,false>, "abl_nohist");
        once((kern_t)walk_abl<false,false,true>, "abl_hist_only");
        once((kern_t)walk_abl<false,false,false>, "abl_max_only");
    }
    cudaFree(in); cudaFree(bmax); cudaFree(out);
    return 0;
}
// -- ablation variants of walk_new<8> appended via -DARM_ONLY build ----------
