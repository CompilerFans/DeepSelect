// Phase split of the production row dataflow, measured on the real body.
//
// The kernel below is `radix_topk_row_bf16_b<1024, true, true>` -- the exact
// instantiation the C600U runs for the official cells -- with `clock64()`
// timers at every barrier (the header is a copy under /root/bisect, patched by
// /tmp/phase_patch.py under -DDS_PHASE_TIMERS).  One CTA per row, so the
// per-phase numbers are per-row costs, and the phases are:
//
//   0 zero the 4096-bin histogram
//   1 pass 1: the ADJ4 summary walk (+ its tail)
//   2 scan_threshold #1 (fold 4096 -> 256, warp scan, narrow)
//   3 the kept-block list build
//   4 the element histogram over the listed blocks
//   5 scan_threshold #2
//   6 the collect (emit above, stage the threshold bin)
//   7 = the whole function; 7 - (0..6) is the refine/emit tail
//
// Usage: ./phaseprobe [V] [k] [rows]

#include <csrc/maca_kernels/common/radix_core.cuh>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace topk_select_radix;

#ifdef DS_PHASE_TIMERS
extern __device__ unsigned long long ds_phase[8];
#endif

__global__ __launch_bounds__(1024) void rowk(const maca_bfloat16 *__restrict__ in,
                                             int32_t *__restrict__ out,
                                             uint32_t length, uint32_t topk,
                                             bool *__restrict__ nanflags,
                                             uint16_t *__restrict__ bmax,
                                             uint32_t *__restrict__ blist) {
#ifdef DS_PHASE_TIMERS
    const unsigned long long t0 = (threadIdx.x == 0u) ? clock64() : 0ull;
#endif
    const uint32_t nb = length >> 6;
    radix_topk_row_bf16_b<1024u, true, true>(
        in + (size_t)blockIdx.x * length, out + (size_t)blockIdx.x * topk, length,
        topk, nanflags + blockIdx.x, bmax + (size_t)blockIdx.x * nb,
        blist + (size_t)blockIdx.x * nb);
#ifdef DS_PHASE_TIMERS
    if (threadIdx.x == 0u) atomicAdd(&ds_phase[7], clock64() - t0);
#endif
}

__global__ void repl(uint4 *dst, const uint4 *src, size_t n_per_row, int rows) {
    const size_t n = n_per_row * (size_t)(rows - 1);
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n;
         i += (size_t)gridDim.x * blockDim.x)
        dst[n_per_row + i] = src[i % n_per_row];
}

static uint16_t to_bf16(float f) {
    uint32_t u;
    __builtin_memcpy(&u, &f, 4);
    const uint32_t lsb = (u >> 16) & 1u;
    u += 0x7FFFu + lsb;  // round to nearest even
    return (uint16_t)(u >> 16);
}

int main(int argc, char **argv) {
    const uint32_t V = argc > 1 ? (uint32_t)atoi(argv[1]) : (1u << 20);
    const uint32_t K = argc > 2 ? (uint32_t)atoi(argv[2]) : 512u;
    const int ROWS = argc > 3 ? atoi(argv[3]) : 1120;
    const uint32_t BLOCK = 1024;
    const uint32_t nb = V >> 6;

    // One standard-normal row, replicated: the kept-block fraction is a
    // property of the distribution and the block size, and every row having
    // the same one is what makes the per-row average meaningful.
    std::vector<uint16_t> row(V);
    uint64_t st = 0x243F6A8885A308D3ull;
    for (uint32_t i = 0; i + 1 < V; i += 2) {
        double u1 = 0, u2 = 0;
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        u1 = (double)(st >> 11) / 9007199254740992.0;
        st ^= st << 13; st ^= st >> 7; st ^= st << 17;
        u2 = (double)(st >> 11) / 9007199254740992.0;
        if (u1 < 1e-12) u1 = 1e-12;
        const double r = std::sqrt(-2.0 * std::log(u1));
        row[i] = to_bf16((float)(r * std::cos(6.283185307179586 * u2)));
        if (i + 1 < V)
            row[i + 1] = to_bf16((float)(r * std::sin(6.283185307179586 * u2)));
    }

    maca_bfloat16 *in;
    int32_t *out;
    bool *nanflags;
    uint16_t *bmax;
    uint32_t *blist;
    const size_t row_bytes = (size_t)V * 2;
    cudaMalloc(&in, row_bytes * ROWS);
    cudaMalloc(&out, (size_t)K * 4 * ROWS);
    cudaMalloc(&nanflags, ROWS);
    cudaMalloc(&bmax, (size_t)nb * 2 * ROWS);
    cudaMalloc(&blist, (size_t)nb * 4 * ROWS);
    cudaMemcpy(in, row.data(), row_bytes, cudaMemcpyHostToDevice);
    repl<<<168, 1024>>>((uint4 *)in, (const uint4 *)in, V / 8, ROWS);
    cudaMemset(nanflags, 0, ROWS);

    const size_t smem = radix_smem_plan(K, false, true).total_bytes;
    cudaFuncSetAttribute(rowk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);

    rowk<<<ROWS, BLOCK, smem>>>(in, out, V, K, nanflags, bmax, blist);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("ERROR: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // One row's answer, for a sanity check that this is the real dataflow.
    std::vector<int32_t> h((size_t)K * 4 * ROWS);
    cudaMemcpy(h.data(), out, (size_t)K * 4 * ROWS, cudaMemcpyDeviceToHost);
    int in_range = 0;
    for (uint32_t i = 0; i < K; i++)
        in_range += (h[i] >= 0 && (uint32_t)h[i] < V);
    printf("V %u  k %u  rows %d  nb %u  smem %zu B  row0 idx in range %d/%u\n", V,
           K, ROWS, nb, smem, in_range, K);

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e30f;
    for (int i = 0; i < 10; i++) {
        cudaEventRecord(a);
        rowk<<<ROWS, BLOCK, smem>>>(in, out, V, K, nanflags, bmax, blist);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        if (ms < best) best = ms;
    }
    const double bytes = (double)ROWS * V * 2;
    printf("%.3f ms   %.2f us/row   %.1f GB/s (one pass, %.1f%% of 1545)\n", best,
           best * 1e3 / ROWS, bytes / (best * 1e6),
           bytes / (best * 1e6) / 1545 * 100);

#ifdef DS_PHASE_TIMERS
    unsigned long long ph[8] = {0};
    cudaMemcpyFromSymbol(ph, ds_phase, sizeof(ph));
    static const char *names[8] = {"0 zero hist ", "1 pass1 walk", "2 scan #1   ",
                                   "3 list build", "4 pass2 hist", "5 scan #2   ",
                                   "6 collect   ", "7 whole     "};
    printf("%s  %12s %10s %8s\n", "phase", "cycles/row", "us/row", "%");
    double sum = 0;
    for (int i = 0; i < 7; i++) sum += (double)ph[i];
    // ds_phase accumulated over the warm-up plus the 10 timed calls, so the
    // shares are what matter; 7 is the whole function.
    for (int i = 0; i < 7; i++) {
        const double cyc = (double)ph[i] / 11.0 / ROWS;
        printf("%s  %12.0f %10.3f %7.1f%%\n", names[i], cyc, cyc / 1e3,
               (double)ph[i] / (double)ph[7] * 100);
    }
    const double tot = (double)ph[7] / 11.0 / ROWS;
    printf("%s  %12.0f %10.3f %7.1f%%\n", names[7], tot, tot / 1e3, 100.0);
    printf("refine/emit (7 - 0..6) %10.3f us/row  %5.1f%%\n",
           (tot - sum / 11.0 / ROWS) / 1e3,
           ((double)ph[7] - sum) / (double)ph[7] * 100);
#endif
    return 0;
}
