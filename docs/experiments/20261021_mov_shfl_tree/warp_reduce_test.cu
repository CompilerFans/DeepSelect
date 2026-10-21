// Correctness + rate for csrc/maca_kernels/common/warp_reduce.cuh.
// Build (repo root on the include path, per the tree's include discipline):
//   cucc -O3 --offload-arch=xcore1600 -I<repo-root> \
//        docs/experiments/20261021_mov_shfl_tree/warp_reduce_test.cu -o /tmp/wrt
//   /tmp/wrt
// SASS receipt:
//   mxcc -aop -S -maca-device-only -I<repo-root> \
//        docs/experiments/20261021_mov_shfl_tree/warp_reduce_test.cu -o /tmp/wrt.s
//   -> the header's reduce must contain NO sm_/ds_-class instruction; count
//      the three forms' instructions between their labels.
//
// Exit status 0 iff every semantic check passes.  If the full-wave checks
// fail with the leaders looking shifted to the rows' other ends, the
// hardware's Row Shift direction is the reverse of the header's assumption:
// rebuild with -DDS_ROW_SHIFT_LEFT (one spelling flips every mode and
// leader), do NOT edit call sites.
#include <cstdio>
#include <cstdint>
#include <cstdlib>

#include <csrc/maca_kernels/common/warp_reduce.cuh>

using topk_select_common::warp_reduce_max_u32;
using topk_select_common::warp_reduce_min_u32;
using topk_select_common::warp_reduce_sum_u32;
using topk_select_common::group8_reduce_max_u32;
using topk_select_common::DS_GROUP8_LEADER_IN8;

// ── correctness ─────────────────────────────────────────────────────────────
__global__ void check_kernel(const uint32_t *__restrict__ in,
                             uint32_t *__restrict__ out_max,
                             uint32_t *__restrict__ out_min,
                             uint32_t *__restrict__ out_sum,
                             uint32_t *__restrict__ out_g8) {
    const int tx = (int)threadIdx.x;
    const uint32_t v = in[blockIdx.x * 64 + tx];
    const uint32_t mx = warp_reduce_max_u32(v);
    const uint32_t mn = warp_reduce_min_u32(v);
    const uint32_t sm = warp_reduce_sum_u32(v & 0xFFFFFu);
    const uint32_t g8 = group8_reduce_max_u32(v);
    if (tx == 0) {
        out_max[blockIdx.x] = mx;
        out_min[blockIdx.x] = mn;
        out_sum[blockIdx.x] = sm;
    }
    if ((tx & 7) == DS_GROUP8_LEADER_IN8)
        out_g8[blockIdx.x * 8 + (tx >> 3)] = g8;
}

// ── the three full-wave reduce forms ────────────────────────────────────────
// FORM 0: the __shfl_down_sync wrapper chain (the measured 71-insn class).
// The mask travels as a typed variable: MACA ships 32- and 64-bit overloads
// of these wrappers, and an unsuffixed literal is ambiguous (CLAUDE.md).
__device__ __forceinline__ uint32_t red_wrapper(uint32_t v) {
    const unsigned long long m = 0xFFFFFFFFFFFFFFFFULL;
    uint32_t o;
    o = __shfl_down_sync(m, v, 32); v = v > o ? v : o;
    o = __shfl_down_sync(m, v, 16); v = v > o ? v : o;
    o = __shfl_down_sync(m, v, 8);  v = v > o ? v : o;
    o = __shfl_down_sync(m, v, 4);  v = v > o ? v : o;
    o = __shfl_down_sync(m, v, 2);  v = v > o ? v : o;
    o = __shfl_down_sync(m, v, 1);  v = v > o ? v : o;
    return __shfl_sync(m, v, 0);
}

// FORM 1: the hand-written bsm_bpermute xor butterfly (the 42-44-insn class,
// same smem-pipe instruction without the wrapper's index math).
__device__ __forceinline__ uint32_t red_bpermute(uint32_t v, uint32_t tx) {
    uint32_t o;
    #pragma unroll
    for (int d = 32; d > 0; d >>= 1) {
        o = (uint32_t)__builtin_mxc_bsm_bpermute(
            (int)(((tx ^ (uint32_t)d) & 63u) << 2), (int)v);
        v = v > o ? v : o;
    }
    return v;
}

// FORM 2: the header (mov_shfl row shifts + readlane scalars).
template <int FORM>
__device__ __forceinline__ uint32_t red_form(uint32_t v, uint32_t tx) {
    if (FORM == 0) return red_wrapper(v);
    if (FORM == 1) return red_bpermute(v, tx);
    return warp_reduce_max_u32(v);
}

// ── rate: dependent chain (latency) and independent sets (throughput) ───────
template <int FORM>
__global__ void lat_kernel(uint32_t *out, int iters) {
    uint32_t v = 0x9E3779B9u + threadIdx.x * 2654435761u;
    for (int i = 0; i < iters; i++)
        v = red_form<FORM>(v, threadIdx.x) ^ (threadIdx.x & 63u);
    if (v == 0xDEADBEEFu) out[0] = v;
}

template <int FORM>
__global__ void thr_kernel(uint32_t *out, int iters) {
    const uint32_t tx = threadIdx.x;
    const uint32_t a0 = 0x9E3779B9u + tx, a1 = 0x85EBCA6Bu + tx * 3u,
                   a2 = 0xC2B2AE35u + tx * 5u, a3 = 0x27D4EB2Fu + tx * 7u;
    uint32_t s = 0;
    for (int i = 0; i < iters; i++) {
        s += red_form<FORM>(a0 + (uint32_t)i, tx);
        s += red_form<FORM>(a1 + (uint32_t)i, tx);
        s += red_form<FORM>(a2 + (uint32_t)i, tx);
        s += red_form<FORM>(a3 + (uint32_t)i, tx);
    }
    if (s == 0xDEADBEEFu) out[0] = s;
}

template <int FORM>
static double time_one(uint32_t *out, int iters, bool latency) {
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    if (latency) lat_kernel<FORM><<<1, 64>>>(out, 512);
    else thr_kernel<FORM><<<1, 64>>>(out, 512);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    if (latency) lat_kernel<FORM><<<1, 64>>>(out, iters);
    else thr_kernel<FORM><<<1, 64>>>(out, iters);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return (double)ms;
}

int main() {
    // correctness over 8 independent waves
    const int waves = 8;
    uint32_t h_in[64 * waves];
    srand(20261021);
    for (int i = 0; i < 64 * waves; i++)
        h_in[i] = ((uint32_t)rand() << 12) ^ (uint32_t)rand();
    uint32_t *d_in, *d_max, *d_min, *d_sum, *d_g8, *d_out;
    cudaMalloc(&d_in, sizeof(h_in));
    cudaMalloc(&d_max, waves * 4);
    cudaMalloc(&d_min, waves * 4);
    cudaMalloc(&d_sum, waves * 4);
    cudaMalloc(&d_g8, waves * 8 * 4);
    cudaMalloc(&d_out, 4);
    cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice);
    check_kernel<<<waves, 64>>>(d_in, d_max, d_min, d_sum, d_g8);
    uint32_t h_max[waves], h_min[waves], h_sum[waves], h_g8[waves * 8];
    cudaMemcpy(h_max, d_max, sizeof(h_max), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_min, d_min, sizeof(h_min), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_sum, d_sum, sizeof(h_sum), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_g8, d_g8, sizeof(h_g8), cudaMemcpyDeviceToHost);
    const cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("KERNEL ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }
    int bad = 0;
    for (int w = 0; w < waves; w++) {
        uint32_t em = 0, en = 0xFFFFFFFFu, es = 0;
        for (int l = 0; l < 64; l++) {
            const uint32_t v = h_in[w * 64 + l];
            em = v > em ? v : em;
            en = v < en ? v : en;
            es += v & 0xFFFFFu;
        }
        if (h_max[w] != em) { printf("wave %d max: got %u want %u\n", w, h_max[w], em); bad++; }
        if (h_min[w] != en) { printf("wave %d min: got %u want %u\n", w, h_min[w], en); bad++; }
        if (h_sum[w] != es) { printf("wave %d sum: got %u want %u\n", w, h_sum[w], es); bad++; }
        for (int g = 0; g < 8; g++) {
            uint32_t eg = 0;
            for (int l = 0; l < 8; l++) {
                const uint32_t v = h_in[w * 64 + g * 8 + l];
                eg = v > eg ? v : eg;
            }
            if (h_g8[w * 8 + g] != eg) {
                printf("wave %d group %d max: got %u want %u\n", w, g,
                       h_g8[w * 8 + g], eg);
                bad++;
            }
        }
    }
    printf("correctness: %s (%d mismatches over %d waves x {max,min,sum,8xg8})\n",
           bad ? "FAIL" : "PASS", bad, waves);
    if (bad) return 1;

    // rate: three forms, latency and throughput
    static const char *names[3] = {"shfl_down_sync wrapper",
                                   "bsm_bpermute butterfly",
                                   "mov_shfl+readlane (header)"};
    const int lat_iters = 16384, thr_iters = 8192;
    for (int f = 0; f < 3; f++) {
        double lat_ms = 0, thr_ms = 0;
        switch (f) {
        case 0:
            lat_ms = time_one<0>(d_out, lat_iters, true);
            thr_ms = time_one<0>(d_out, thr_iters, false);
            break;
        case 1:
            lat_ms = time_one<1>(d_out, lat_iters, true);
            thr_ms = time_one<1>(d_out, thr_iters, false);
            break;
        default:
            lat_ms = time_one<2>(d_out, lat_iters, true);
            thr_ms = time_one<2>(d_out, thr_iters, false);
            break;
        }
        printf("%-28s latency %7.2f ns/reduce   throughput %7.2f ns/reduce\n",
               names[f], lat_ms * 1e6 / lat_iters, thr_ms * 1e6 / (thr_iters * 4));
    }
    cudaFree(d_in); cudaFree(d_max); cudaFree(d_min); cudaFree(d_sum);
    cudaFree(d_g8); cudaFree(d_out);
    return 0;
}
