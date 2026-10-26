// Phase split of the fp32 Sampler row (`topk_coarse12_row`) on the shapes the
// README's fp32 figure plots: vocab 129280, topk 512.
//
// One CTA per row, the row replicated, clock64() timers at the function's own
// barriers (a copy of topk_coarse12.cuh patched by ./f32phase_patch.py).
//
//   0 pass 1   the 4,096-bin histogram + the block summary + the NaN vote
//   1 scan     the 256-bin fold, the cumsum, the 12-bit narrow
//   2 list     the kept-block list
//   3 pass 2   the collect over the list (or the dense walk)
//   4 refine   the arena's three rounds (or the overflow rescan)
//
// Usage: ./f32phaseprobe [V] [k] [rows]

#include <csrc/maca_kernels/common/topk_coarse12.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace topk_select_coarse12;

#ifdef DS_PHASE_TIMERS
extern __device__ unsigned long long ds_f32_phase[8];
#endif

__global__ __launch_bounds__(kThreads) void rowk(const float *__restrict__ in,
                                                 int32_t *__restrict__ out,
                                                 int length, int topk,
                                                 int32_t *__restrict__ nanf,
                                                 uint16_t *__restrict__ bmax,
                                                 uint32_t *__restrict__ blist) {
#ifdef DS_PHASE_TIMERS
    const unsigned long long t0 = (threadIdx.x == 0u) ? clock64() : 0ull;
#endif
    const int nblk = length / kBlockElems + 1;
    topk_coarse12_row(in + (size_t)blockIdx.x * length, out + (size_t)blockIdx.x * topk,
                      length, topk, nanf + blockIdx.x,
                      bmax + (size_t)blockIdx.x * nblk,
                      blist + (size_t)blockIdx.x * nblk);
#ifdef DS_PHASE_TIMERS
    if (threadIdx.x == 0u) atomicAdd(&ds_f32_phase[4], clock64() - t0);
#endif
}

__global__ void repl(float4 *dst, const float4 *src, size_t n_per_row, int rows) {
    const size_t n = n_per_row * (size_t)(rows - 1);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[n_per_row + i] = src[i % n_per_row];
}

int main(int argc, char **argv) {
    const int V = argc > 1 ? atoi(argv[1]) : 129280;
    const int K = argc > 2 ? atoi(argv[2]) : 512;
    const int ROWS = argc > 3 ? atoi(argv[3]) : 1120;
    const int nblk = V / kBlockElems + 1;

    std::vector<float> row(V);
    uint64_t st = 0x243F6A8885A308D3ull;
    for (int i = 0; i + 1 < V; i += 2) {
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        const double u1 = std::max((double)(st >> 11) / 9007199254740992.0, 1e-12);
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        const double u2 = (double)(st >> 11) / 9007199254740992.0;
        const double r = std::sqrt(-2.0 * std::log(u1));
        row[i] = (float)(r * std::cos(6.283185307179586 * u2));
        if (i + 1 < V) row[i + 1] = (float)(r * std::sin(6.283185307179586 * u2));
    }

    float *in;
    int32_t *out, *nanf;
    uint16_t *bmax;
    uint32_t *blist;
    const size_t row_bytes = (size_t)V * sizeof(float);
    cudaMalloc(&in, row_bytes * ROWS);
    cudaMalloc(&out, (size_t)K * 4 * ROWS);
    cudaMalloc(&nanf, sizeof(int32_t) * ROWS);
    cudaMalloc(&bmax, sizeof(uint16_t) * (size_t)nblk * ROWS);
    cudaMalloc(&blist, sizeof(uint32_t) * (size_t)nblk * ROWS);
    cudaMemcpy(in, row.data(), row_bytes, cudaMemcpyHostToDevice);
    if (ROWS > 1) repl<<<168, 512>>>((float4 *)in, (const float4 *)in, V / 4, ROWS);
    cudaMemset(nanf, 0, sizeof(int32_t) * ROWS);

    cudaFuncSetAttribute(rowk, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)kSmemBytes);
    const size_t smem = kSmemBytes;
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, rowk, kThreads, smem);
    const int concurrent = occ * 28;

    rowk<<<ROWS, kThreads, smem>>>(in, out, V, K, nanf, bmax, blist);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) { printf("ERROR: %s\n", cudaGetErrorString(err)); return 1; }
    printf("V %d  k %d  rows %d  kThreads %d  smem %zu B  occupancy %d CTA/AP "
           "(%d concurrent)  nblk %d\n", V, K, ROWS, kThreads, smem, occ, concurrent, nblk);

    // Ramp, then measure: an idle AP runs at a low clock and a two-millisecond
    // probe measures the idle clock.
    for (int i = 0; i < 300; i++)
        rowk<<<ROWS, kThreads, smem>>>(in, out, V, K, nanf, bmax, blist);
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    float best = 1e30f, last = 0;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a);
        rowk<<<ROWS, kThreads, smem>>>(in, out, V, K, nanf, bmax, blist);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0; cudaEventElapsedTime(&ms, a, b); last = ms;
        if (ms < best) best = ms;
    }
    const double bytes = (double)ROWS * V * 4.0;
    const double waves = (double)ROWS / (double)concurrent;
    const double row_wall = best * 1e3 / waves;
    printf("%.3f ms (last %.3f)  %.1f GB/s  wave %.1f  row wall %.3f us  "
           "row CTA %.1f us\n",
           best, last, bytes / (best * 1e6), waves, row_wall, best * 1e3 / waves);

#ifdef DS_PHASE_TIMERS
    unsigned long long ph[8] = {0};
    cudaMemcpyFromSymbol(ph, ds_f32_phase, sizeof(ph));
    const double n = 331.0 * ROWS;
    static const char *names[5] = {"0 pass 1    ", "1 scan      ", "2 list      ",
                                   "3 pass 2    ", "4 refine+all"};
    double sum = 0;
    for (int i = 0; i < 4; i++) sum += (double)ph[i];
    printf("%s %12s %8s\n", "phase", "cycles/row", "%");
    for (int i = 0; i < 4; i++)
        printf("%s %12.0f %7.1f%%\n", names[i], (double)ph[i] / n,
               (double)ph[i] / (double)ph[4] * 100);
    printf("%s %12.0f %7.1f%%   (whole row)\n", names[4], (double)ph[4] / n, 100.0);
    printf("refine = whole - (0..3)   %10.0f cycles  %5.1f%%\n",
           ((double)ph[4] - sum) / n, ((double)ph[4] - sum) / (double)ph[4] * 100);
    printf("clock %.2f GHz (from whole/wall)\n",
           (double)ph[4] / n / row_wall / 1e3);
#endif
    return 0;
}
