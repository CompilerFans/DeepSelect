// Phase split of the production row dataflow *after* the staging walk and the
// packed arena, measured on the real body.
//
// The kernel below is `radix_topk_row_bf16_b<512, true, true>` -- the
// instantiation the C600-U runs for the official long-row cells at batch >= 256
// (`radix_block_for` -> `kBlockSize` on a 28/32-AP part) -- with `clock64()`
// timers at every barrier (the header is a copy, patched by
// ./phase_patch2.py under -DDS_PHASE_TIMERS).  One CTA per row, so the
// per-phase numbers are per-row costs, and the phases are:
//
//   0 pass 1: the ADJ4 summary walk, its tail, and the NaN vote
//   1 scan #1: the fold of the block counts, the scan, the narrow
//   2 the kept-block list build
//   3 the staging walk over the list, plus the row's tail
//   4 scan #2: the arena pass, the scan, the narrow
//   5 = the whole function; 5 - (0..4) is the refine/emit tail
//
//   8 / 9 / 10 split the two scans into fold / cumsum / narrow, and 11 is the
//   arena pass that precedes scan #2.  Slots 6/7 count the rows that got a
//   list and the rows that kept the arena; 13/14 sum `nlist` and `num_staged`.
//
// The per-row wall is `wall / (ROWS / concurrent)`, and `concurrent` is the
// driver's own occupancy answer, not a hand division.
//
// Usage: ./phaseprobe2 [V] [k] [rows] [row.bin]
// `row.bin` is raw bf16 (uint16, little endian) of V elements; without it the
// row is a Box-Muller normal.

#include <csrc/maca_kernels/common/radix_core.cuh>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <vector>

using namespace topk_select_radix;

#ifdef DS_PHASE_TIMERS
extern __device__ unsigned long long ds_phase[16];
#endif
#ifdef DS_ONEPASS
extern __device__ unsigned int ds_probe_cut;
#endif

__global__ __launch_bounds__(512) void rowk(const maca_bfloat16 *__restrict__ in,
                                            int32_t *__restrict__ out,
                                            uint32_t length, uint32_t topk,
                                            bool *__restrict__ nanflags,
                                            uint16_t *__restrict__ bmax,
                                            uint32_t *__restrict__ blist) {
#ifdef DS_PHASE_TIMERS
    const unsigned long long t0 = (threadIdx.x == 0u) ? clock64() : 0ull;
#endif
    const uint32_t nb = length >> 6;
    radix_topk_row_bf16_b<512u, true, true>(
        in + (size_t)blockIdx.x * length, out + (size_t)blockIdx.x * topk, length,
        topk, nanflags + blockIdx.x, bmax + (size_t)blockIdx.x * nb,
        blist + (size_t)blockIdx.x * nb);
#ifdef DS_PHASE_TIMERS
    if (threadIdx.x == 0u) atomicAdd(&ds_phase[5], clock64() - t0);
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
    const uint32_t V = argc > 1 ? (uint32_t)atoi(argv[1]) : (1u << 17);
    const uint32_t K = argc > 2 ? (uint32_t)atoi(argv[2]) : 512u;
    const int ROWS = argc > 3 ? atoi(argv[3]) : 1120;
    const char *rowfile = argc > 4 ? argv[4] : nullptr;
    const uint32_t BLOCK = 512;
    const uint32_t nb = V >> 6;

    // One row, replicated: the kept-block fraction is a property of the
    // distribution and the block size, and every row having the same one is
    // what makes the per-row average meaningful.  `row.bin` is what the
    // harness itself generated, which is the arm production takes.
    std::vector<uint16_t> row(V);
    if (rowfile != nullptr) {
        FILE *f = fopen(rowfile, "rb");
        if (f == nullptr || fread(row.data(), 2, V, f) != V) {
            printf("ERROR: cannot read %s\n", rowfile);
            return 1;
        }
        fclose(f);
    } else {
        uint64_t st = 0x243F6A8885A308D3ull;
        for (uint32_t i = 0; i + 1 < V; i += 2) {
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            const double u1 = std::max((double)(st >> 11) / 9007199254740992.0, 1e-12);
            st ^= st << 13; st ^= st >> 7; st ^= st << 17;
            const double u2 = (double)(st >> 11) / 9007199254740992.0;
            const double r = std::sqrt(-2.0 * std::log(u1));
            row[i] = to_bf16((float)(r * std::cos(6.283185307179586 * u2)));
            if (i + 1 < V)
                row[i + 1] = to_bf16((float)(r * std::sin(6.283185307179586 * u2)));
        }
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
    if (ROWS > 1) repl<<<168, 512>>>((uint4 *)in, (const uint4 *)in, V / 8, ROWS);
    cudaMemset(nanflags, 0, ROWS);

#ifdef DS_ONEPASS
    // The cut a one-pass dataflow would have to guess: the bin of the
    // 1.15 x topk-th largest element, so the staged set is a superset of the
    // answer and everything downstream stays exact.  The bin is taken off an
    // order-preserving key, which is what `refine_bin` computes in the kernel.
    {
        std::vector<uint16_t> keys(row);
        for (uint32_t i = 0; i < V; i++)
            keys[i] = (keys[i] & 0x8000u) ? (uint16_t)~keys[i]
                                          : (uint16_t)(keys[i] | 0x8000u);
        const uint32_t rank = (uint32_t)((double)K * 1.15);
        std::nth_element(keys.begin(), keys.begin() + (V - rank), keys.end());
        const uint32_t cut = keys[V - rank] >> 4;
        cudaMemcpyToSymbol(ds_probe_cut, &cut, sizeof(cut));
        printf("one-pass probe: cut bin %u (rank %u of %u)\n", cut, rank, V);
    }
#endif

    const size_t smem = radix_smem_plan(K, false, true).total_bytes;
    cudaFuncSetAttribute(rowk, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         (int)smem);
    int occ = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, rowk, (int)BLOCK,
                                                  smem);
    const int concurrent = occ * 28;

    // The tables are allocated by the call site only when the row is long and
    // dense enough (`kSkipMinVocab`, `kSkipMinRowPerSlot`); a probe that always
    // hands them over measures a summary the caller never asks for.
    const bool tables = V >= 65536u && (uint64_t)V >= (uint64_t)K * 128u;
    rowk<<<ROWS, BLOCK, smem>>>(in, out, V, K, nanflags,
                                tables ? bmax : nullptr,
                                tables ? blist : nullptr);
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
    printf("V %u  k %u  rows %d  nb %u  smem %zu B  occupancy %d CTA/AP "
           "(%d concurrent)  row0 idx in range %d/%u  data %s\n",
           V, K, ROWS, nb, smem, occ, concurrent, in_range, K,
           rowfile != nullptr ? rowfile : "normal");

    // Ramp first: the AP idles at a low clock and needs sustained work to reach
    // boost, and a two-millisecond probe measures the idle clock (a factor of
    // 3.5 in wall time, and a memory-latency bias in the phase shares).  The
    // cycles are clock-invariant; the wall is not, so `best` is reported beside
    // `last` and the ramp is the difference.
    for (int i = 0; i < 300; i++)
        rowk<<<ROWS, BLOCK, smem>>>(in, out, V, K, nanflags,
                                    tables ? bmax : nullptr,
                                    tables ? blist : nullptr);
    cudaDeviceSynchronize();

    cudaEvent_t a, b;
    cudaEventCreate(&a);
    cudaEventCreate(&b);
    float best = 1e30f, last = 0;
    for (int i = 0; i < 30; i++) {
        cudaEventRecord(a);
        rowk<<<ROWS, BLOCK, smem>>>(in, out, V, K, nanflags,
                                    tables ? bmax : nullptr,
                                    tables ? blist : nullptr);
        cudaEventRecord(b);
        cudaEventSynchronize(b);
        float ms = 0;
        cudaEventElapsedTime(&ms, a, b);
        last = ms;
        if (ms < best) best = ms;
    }
    const double bytes = (double)ROWS * V * 2;
    const double waves = (double)ROWS / (double)concurrent;
    const double row_wall_us = best * 1e3 / waves;
    printf("%.3f ms (last %.3f)   %.1f GB/s (one pass, %.1f%% of 1545)   "
           "wave %.1f   row wall %.3f us\n",
           best, last, bytes / (best * 1e6), bytes / (best * 1e6) / 1545 * 100,
           waves, row_wall_us);

#ifdef DS_PHASE_TIMERS
    unsigned long long ph[16] = {0};
    cudaMemcpyFromSymbol(ph, ds_phase, sizeof(ph));
    static const char *names[6] = {"0 pass1 walk", "1 scan #1   ", "2 list build",
                                   "3 stage walk", "4 scan #2   ", "5 whole     "};
    // ds_phase accumulated over the one launch before the ramp, the 300 ramp
    // launches and the 30 timed ones.
    const double launches = 331.0;
    const double n = launches * ROWS;
    printf("%s  %12s %10s %8s\n", "phase", "cycles/row", "kcycles", "%");
    double sum = 0;
    for (int i = 0; i < 5; i++) sum += (double)ph[i];
    for (int i = 0; i < 5; i++) {
        const double cyc = (double)ph[i] / n;
        printf("%s  %12.0f %10.3f %7.1f%%\n", names[i], cyc, cyc / 1e3,
               (double)ph[i] / (double)ph[5] * 100);
    }
    const double tot = (double)ph[5] / n;
    printf("%s  %12.0f %10.3f %7.1f%%\n", names[5], tot, tot / 1e3, 100.0);
    // The AP clock, from the two independent readings of one row: `whole`
    // cycles and the wall it took.  A share taken at a low clock over-weights
    // the phases that wait on memory, so this is what says the ramp landed.
    printf("clock from whole/wall: %.2f GHz\n", tot / row_wall_us / 1e3);
    printf("refine/emit (5 - 0..4) %10.3f us/CTA  %5.1f%%\n",
           (tot - sum / n) / 1e3, ((double)ph[5] - sum) / (double)ph[5] * 100);
    static const char *sub[4] = {"8 fold      ", "9 cumsum    ", "10 narrow bar",
                                 "11 arena pass"};
    for (int i = 8; i < 12; i++)
        printf("%s  %12.0f %10.3f %7.1f%%   (both scans)\n", sub[i - 8],
               (double)ph[i] / n, (double)ph[i] / n / 1e3,
               (double)ph[i] / (double)ph[5] * 100);
    printf("%s  %12.0f %10.3f %7.1f%%   (both scans)\n", "12 narrow loop",
           (double)ph[12] / n, (double)ph[12] / n / 1e3,
           (double)ph[12] / (double)ph[5] * 100);
    static const char *cnt[3] = {"13 count hist", "14 collect   ", "15 sampled cut"};
    for (int i = 13; i < 16; i++)
        printf("%s  %12.0f %10.3f %7.1f%%   (counting arm)\n", cnt[i - 13],
               (double)ph[i] / n, (double)ph[i] / n / 1e3,
               (double)ph[i] / (double)ph[5] * 100);
    printf("rows with list %.0f/%.0f, rows staged %.0f/%.0f\n",
           (double)ph[6] / launches, (double)ROWS, (double)ph[7] / launches,
           (double)ROWS);
#endif
    return 0;
}
