// FP8 pass-1 ceiling probe: at 1 byte/element, is the histogram walk bound by
// DRAM or by the shared-atomic issue rate?
//
// Arms over the same grid-stride walk (16 B per load per thread):
//
//   A  read-only, 4 uint4 in flight  -- the streaming wall, in bytes/s.
//   B  16 shared atomics per uint4   -- the naive fp8 pass-1 port (16 fp8 per
//                                       uint4, one histogram increment per
//                                       element).  Bin = the byte itself (the
//                                       fp8 order-preserving encode is one
//                                       xor, free).
//   C  8 shared atomics per uint4    -- today's bf16 pass-1 pattern, the
//                                       calibration arm.
//
// B and C run twice: on uniform bytes (bins spread over all 256 entries) and
// on crowded bytes (7/8 of the buffer one bin -- the shape
// NormalFloatDistribution crowds the bf16 high byte with, which is the DSA
// workload's own distribution).  The pair separates per-atomic *issue* from
// same-bin *contention*.
//
// Bins live in a per-CTA 256-entry shared histogram, exactly like the row
// kernel's pass 1.  Timing by CUDA events over a device-side repeat loop; the
// histogram feeds the checksum so nothing is elided.
//
// Build (this box, C600-U):
//   cucc -O2 -std=c++20 --offload-arch=xcore1600 fp8_pass1_ceiling.cu -o /tmp/fp8_probe

#include <cstdint>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

constexpr int kThreads = 512;          // the row kernel's block width

__global__ void __launch_bounds__(kThreads) walk_read(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    uint32_t acc = 0;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x; i + 3 * stride < n_vec;
         i += 4 * stride) {
        uint4 a = __ldg(in + i);
        uint4 b = __ldg(in + i + stride);
        uint4 c = __ldg(in + i + 2 * stride);
        uint4 d = __ldg(in + i + 3 * stride);
        acc ^= a.x ^ b.y ^ c.z ^ d.w;
    }
    if (acc == 0xFFFFFFFFu) out[blockIdx.x] = acc;   // never true; keeps the loads
}

template <int kAtomicsPerVec>   // 16 = fp8 arm, 8 = bf16 arm
__global__ void __launch_bounds__(kThreads) walk_hist(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[256];
    for (int t = threadIdx.x; t < 256; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        if constexpr (kAtomicsPerVec == 16) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; j++) atomicAdd(&s_hist[b[j]], 1u);
        } else {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
            for (int j = 0; j < 8; j++) atomicAdd(&s_hist[h[j] >> 8], 1u);
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// D: ballot-aggregated atomics -- per j-round the 64 lanes hold one byte each;
// loop over the round's distinct bins, and each distinct bin costs the warp one
// ballot + one shuffle + one shared atomic instead of one atomic per lane.
// Uniform 8-bit data has ~56 distinct bins per 64 lanes (no win, more wrapper
// instructions); crowded data has a handful (the win this arm measures).
// `__match_any_sync` is the banned spelling (265 insns, emulated); this loop
// uses the native ballot only.
__global__ void __launch_bounds__(kThreads) walk_hist_ballot(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[256];
    for (int t = threadIdx.x; t < 256; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    const unsigned long long live = __activemask();
    const int lane = threadIdx.x & 63;
    // lane-major: lane L takes the tile's L-th uint4, so one ballot round
    // aggregates one byte position across the wave.
    const int warp = threadIdx.x >> 6, warps = kThreads >> 6;
    size_t tiles = n_vec / 64;
    for (size_t t = blockIdx.x * warps + warp; t < tiles;
         t += (size_t)gridDim.x * warps) {
        uint4 v = __ldg(in + t * 64 + lane);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
        for (int j = 0; j < 16; j++) {
            const uint8_t my = b[j];
            bool done = false;
            unsigned long long remaining = live;
            while (remaining) {
                const int leader = __ffsll((long long)remaining) - 1;
                const uint8_t lbin = (uint8_t)__shfl_sync(live, (uint32_t)my, leader);
                const unsigned long long same =
                    __ballot_sync(live, !done && my == lbin) & remaining;
                if (lane == leader) atomicAdd(&s_hist[lbin], (uint32_t)__popcll(same));
                done |= (same >> lane) & 1;
                remaining &= ~same;
            }
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// E: per-warp replicated histograms (8 warps x 256 bins x 4 B = 8 KB/CTA).
// Kills *contention* (each warp increments its own copy) but not *issue* (still
// one atomic per element) -- the arm that separates the two on crowded data.
// The walk is what is measured; the end-of-kernel reduce over the 8 copies is
// 8 KB of shared reads, negligible, and omitted.
__global__ void __launch_bounds__(kThreads) walk_hist_repl(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[8][256];
    for (int t = threadIdx.x; t < 8 * 256; t += kThreads)
        s_hist[t >> 8][t & 255] = 0;
    __syncthreads();
    const int warp = threadIdx.x >> 6;
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
        for (int j = 0; j < 16; j++) atomicAdd(&s_hist[warp][b[j]], 1u);
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 7][threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// F/G: within-thread same-bin pre-aggregation -- the thread's own vector load
// is aggregated in registers before any atomic is issued: each distinct bin in
// the load costs one atomic carrying its in-load count.  Fixed triangular
// compare (28 compares for 8 elems, 120 for 16), no divergence, no transport.
// Uniform data buys ~nothing (few dups) and pays the compares; crowded data --
// the DSA/NormalFloat shape, where elements crowd one high-byte bin -- cuts
// the atomics by the crowding factor, exactly where B/C hurt most.
template <int kElems>
__global__ void __launch_bounds__(kThreads) walk_hist_agg(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[256];
    for (int t = threadIdx.x; t < 256; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        uint8_t bin[kElems];
        if constexpr (kElems == 16) {
            const uint8_t* b = reinterpret_cast<const uint8_t*>(&v);
#pragma unroll
            for (int j = 0; j < 16; j++) bin[j] = b[j];
        } else {
            const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
            for (int j = 0; j < 8; j++) bin[j] = (uint8_t)(h[j] >> 8);
        }
#pragma unroll
        for (int j = 0; j < kElems; j++) {
            bool leader = true;
#pragma unroll
            for (int m = 0; m < j; m++) leader &= (bin[m] != bin[j]);
            uint32_t cnt = 0;
#pragma unroll
            for (int m = j; m < kElems; m++) cnt += (bin[m] == bin[j]);
            if (leader) atomicAdd(&s_hist[bin[j]], cnt);
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 255] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

// C12/F12: the LIVE kernel's pass-1 shape -- elements are 16-bit keys, the
// histogram is 4096 bins of `key >> 4` (coarse12), 16 KB of shared.  The 8-bit
// arms above overstate crowding: NormalFloat crowds one *high byte*, which
// spreads over ~16 sub-bins at 12 bits, relaxing same-bin serialization.
__global__ void __launch_bounds__(kThreads) walk_hist12(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
#pragma unroll
        for (int j = 0; j < 8; j++) atomicAdd(&s_hist[h[j] >> 4], 1u);
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

__global__ void __launch_bounds__(kThreads) walk_hist12_agg(
    const uint4* __restrict__ in, size_t n_vec, uint32_t* __restrict__ out) {
    __shared__ uint32_t s_hist[4096];
    for (int t = threadIdx.x; t < 4096; t += kThreads) s_hist[t] = 0;
    __syncthreads();
    for (size_t i = blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec; i += (size_t)gridDim.x * blockDim.x) {
        uint4 v = __ldg(in + i);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
        uint16_t bin[8];
#pragma unroll
        for (int j = 0; j < 8; j++) bin[j] = h[j] >> 4;
#pragma unroll
        for (int j = 0; j < 8; j++) {
            bool leader = true;
#pragma unroll
            for (int m = 0; m < j; m++) leader &= (bin[m] != bin[j]);
            uint32_t cnt = 0;
#pragma unroll
            for (int m = j; m < 8; m++) cnt += (bin[m] == bin[j]);
            if (leader) atomicAdd(&s_hist[bin[j]], cnt);
        }
    }
    __syncthreads();
    if (s_hist[threadIdx.x & 4095] == 0xFFFFFFFFu) out[blockIdx.x] = 1;
}

template <typename F>
float time_kernel(F&& launch, const char* name, size_t bytes, int reps) {
    cudaEvent_t t0, t1;
    CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
    launch();                                  // warm-up
    CK(cudaDeviceSynchronize());
    CK(cudaEventRecord(t0));
    for (int r = 0; r < reps; r++) launch();
    CK(cudaEventRecord(t1));
    CK(cudaEventSynchronize(t1));
    float ms = 0; CK(cudaEventElapsedTime(&ms, t0, t1));
    float us = ms * 1000.0f / reps;
    printf("%-40s %9.1f us  %8.1f GB/s\n", name, us, bytes / us / 1e3);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
    return us;
}

static uint32_t __float_as_uint_host(float x) { union { float f; uint32_t u; } u; u.f = x; return u.u; }

// The kernel's bf16 order-preserving key, host-side: truncate to bf16 bits,
// then the sign-flip encode (`refine_bin` in primitives.cuh).
static uint16_t host_key16(float x) {
    uint16_t bits = (uint16_t)(__float_as_uint_host(x) >> 16);
    return (bits & 0x8000u) ? (uint16_t)~bits : (uint16_t)(bits | 0x8000u);
}
static uint8_t host_key_hi8(float x) { return (uint8_t)(host_key16(x) >> 8); }

int main() {
    int dev = 0; CK(cudaGetDevice(&dev));
    cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, dev));
    const int ctas = p.multiProcessorCount * 8;   // 224: the repo's wall arm's grid

    const size_t bytes = size_t(1) << 28;      // 256 MB, streaming (no L2 reuse)
    uint8_t* buf; CK(cudaMalloc(&buf, bytes));
    uint32_t* out; CK(cudaMalloc(&out, ctas * sizeof(uint32_t)));
    uint8_t* host = (uint8_t*)malloc(bytes);
    srand(20261018);

    size_t n_vec = bytes / sizeof(uint4);
    int reps = 5;
    printf("device %s, %d APs, grid %d x %d threads, buffer %zu MB\n",
           p.name, p.multiProcessorCount, ctas, kThreads, bytes >> 20);

    // The wall, once.
    for (size_t i = 0; i < bytes; i++) host[i] = (uint8_t)rand();      // uniform
    CK(cudaMemcpy(buf, host, bytes, cudaMemcpyHostToDevice));
    time_kernel([&]{ walk_read<<<ctas, kThreads>>>((const uint4*)buf, n_vec, out); },
                "A read-only x4 in flight (the wall)", bytes, reps);

    // Three distributions: uniform / normal (the official generator's shape:
    // ~27% of elements crowd one high-byte bin) / extreme (87.5% one bin).
    // The bf16 arms run on 16-bit keys (the live kernel's element, coarse12
    // bins); the fp8 arms on bytes.  Each gets its own faithful buffer.
    uint8_t* buf8;  CK(cudaMalloc(&buf8, bytes));
    uint8_t* host8 = (uint8_t*)malloc(bytes);
    uint16_t* host16 = (uint16_t*)malloc(bytes);
    struct Dist { const char* name; int kind; };
    const Dist dists[] = {{"uniform", 0}, {"normal~27%", 1}, {"crowded87.5%", 2}};
    for (const auto& d : dists) {
        if (d.kind == 0) {
            for (size_t i = 0; i < bytes; i++) host8[i] = (uint8_t)rand();
            for (size_t i = 0; i < bytes / 2; i++) host16[i] = (uint16_t)rand();
        } else if (d.kind == 1) {
            // Box-Muller N(0,1), through the kernel's own key encodes.
            for (size_t k = 0; k < bytes; k += 2) {
                float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
                float u2 = (rand() + 0.0f) / (RAND_MAX + 1.0f);
                float r = sqrtf(-2.0f * logf(u1)), t = 6.2831853f * u2;
                host8[k] = host_key_hi8(r * cosf(t));
                if (k + 1 < bytes) host8[k + 1] = host_key_hi8(r * sinf(t));
            }
            for (size_t k = 0; k < bytes / 2; k += 2) {
                float u1 = (rand() + 1.0f) / (RAND_MAX + 2.0f);
                float u2 = (rand() + 0.0f) / (RAND_MAX + 1.0f);
                float r = sqrtf(-2.0f * logf(u1)), t = 6.2831853f * u2;
                host16[k] = host_key16(r * cosf(t));
                if (k + 1 < bytes / 2) host16[k + 1] = host_key16(r * sinf(t));
            }
        } else {
            for (size_t i = 0; i < bytes; i++)
                host8[i] = (rand() & 7) ? 0x80 : (uint8_t)rand();
            for (size_t i = 0; i < bytes / 2; i++)
                host16[i] = (rand() & 7) ? (uint16_t)0x8080u : (uint16_t)rand();
        }
        CK(cudaMemcpy(buf8, host8, bytes, cudaMemcpyHostToDevice));
        CK(cudaMemcpy(buf, host16, bytes, cudaMemcpyHostToDevice));
        printf("── %s ──\n", d.name);
        time_kernel([&]{ walk_hist12<<<ctas, kThreads>>>((const uint4*)buf, n_vec, out); },
                    "C12 bf16 plain, coarse12 bins (live shape)", bytes, reps);
        time_kernel([&]{ walk_hist12_agg<<<ctas, kThreads>>>((const uint4*)buf, n_vec, out); },
                    "F12 bf16 in-thread agg, coarse12 bins", bytes, reps);
        time_kernel([&]{ walk_hist<8><<<ctas, kThreads>>>((const uint4*)buf8, n_vec, out); },
                    "C  bf16 plain (8 atomics/16B, 256 bins)", bytes, reps);
        time_kernel([&]{ walk_hist_agg<8><<<ctas, kThreads>>>((const uint4*)buf8, n_vec, out); },
                    "F  bf16 in-thread agg (256 bins)", bytes, reps);
        time_kernel([&]{ walk_hist<16><<<ctas, kThreads>>>((const uint4*)buf8, n_vec, out); },
                    "B  fp8 plain (16 atomics/16B)", bytes, reps);
        time_kernel([&]{ walk_hist_agg<16><<<ctas, kThreads>>>((const uint4*)buf8, n_vec, out); },
                    "G  fp8 in-thread agg", bytes, reps);
    }
    CK(cudaGetLastError());
    return 0;
}
