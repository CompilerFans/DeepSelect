// wave64_probe -- measure MACA's actual wave semantics on the device in front
// of you, instead of reading them off the headers.
//
// WHY THIS EXISTS
// ---------------
// A CUDA kernel ported to MACA usually arrives assuming CUDA's 32-lane warp:
// `lane_idx = threadIdx.x % 32`, `NUM_WARPS = NUM_THREADS / 32`, and
// `0xFFFFFFFF` masks everywhere.  On MACA none of those hold, and the failure
// is *silent* -- no compile error, no assert, just wrong numbers that change
// run to run.  This program measures the real behavior so the port can be
// fixed against evidence.
//
// MEASURED RESULTS (MACA 3.8.1.3, MetaX C600-U, cucc/mxcc, 2026-09-12)
// --------------------------------------------------------------------
//     warpSize                        = 64
//     __lane_id()                     = threadIdx.x % 64
//     __activemask()                  = 0xffffffffffffffff  (full wave)
//     __ballot_sync(0xFFFFFFFF, 1)    = 0x00000000ffffffff  (low half only!)
//     __ballot_sync(0xF...F, 1)       = 0xffffffffffffffff
//     __popc(ballot)                  = counts only the low 32 bits
//     __reduce_add_sync(0xFFFFFFFF, 1)= 32        (not 64)
//     __reduce_add_sync(full, 1)      = 64
//     __shfl_up_sync(0xFFFFFFFF, ...)  lane 32 cannot see lane 31
//     __any_sync(0xFFFFFFFF, ...)      ignores lanes 32..63
//
// The load-bearing detail: `__ballot_sync(mask, pred)` lowers to
// `__builtin_mxc_sicmp(pred, 0, ICMP_NE) & mask` -- ONE comparison covering the
// whole 64-lane wave, then a bitwise AND with the mask.  So a 32-bit mask is
// not "group by 32"; it is "throw away lanes 32..63".  `__reduce_add_sync`,
// `__any_sync` and `__shfl_*_sync` follow the same rule: a lane outside the
// mask gets its own value back.
//
// BUILD AND RUN
// -------------
//     scripts/run_probe.sh                  # on the native target
//     scripts/run_probe.sh xcore1600        # or name it
//
// Compiles with the repository's own device toolchain (cu-bridge's cucc), so
// the semantics reported are the ones the kernels under audit were built with.

#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>

// ── 1. identity of the wave ─────────────────────────────────────────────────

__global__ void k_identity(int *warp_size, int *lane_id, unsigned long long *active) {
    const unsigned tx = threadIdx.x;
    if (tx == 0) {
        *warp_size = (int)warpSize;
        *active = __activemask();
    }
    lane_id[tx] = (int)__lane_id();
}

// ── 2. what a mask means to each collective ─────────────────────────────────
//
// Vote only from lanes 0, 32 and 40, and value 1 only from lanes 0 and 32.
// A 32-lane grouping and a 64-lane grouping then give answers that cannot be
// confused: 0x1 (own group only) vs 0x10100000001 (both groups).

__global__ void k_collectives(unsigned long long *ballot32, unsigned long long *ballot64,
                              unsigned *red32, unsigned *red64,
                              unsigned *any32, unsigned *any64,
                              int *shfl32) {
    const unsigned tx = threadIdx.x;

    const unsigned votes = (tx == 0) || (tx == 32) || (tx == 40);
    ballot32[tx] = __ballot_sync(0xFFFFFFFFu, votes);
    ballot64[tx] = __ballot_sync(0xFFFFFFFFFFFFFFFFull, votes);

    const unsigned val = (tx == 0) || (tx == 32) ? 1u : 0u;
    // NB: the two overloads are `(uint64_t, ...)` and `(unsigned, ...)`, so an
    // unsuffixed 0xFFFFFFFF is ambiguous and will not compile. Spell the type.
    red32[tx] = __reduce_add_sync_impl((uint64_t)0xFFFFFFFFull, val);
    red64[tx] = __reduce_add_sync_impl((uint64_t)0xFFFFFFFFFFFFFFFFull, val);

    any32[tx] = __any_sync(0xFFFFFFFFu, tx == 40) ? 1u : 0u;
    any64[tx] = __any_sync(0xFFFFFFFFFFFFFFFFull, tx == 40) ? 1u : 0u;

    shfl32[tx] = __shfl_up_sync(0xFFFFFFFFu, (int)(tx == 31 ? 777 : tx), 1);
}

// ── 3. does the port's own scan helper survive a 64-lane wave? ──────────────

// Verbatim from csrc/xcore1600/utils.cuh: the 32-lane form the port uses.
template<typename T>
__device__ __forceinline__ T scan32(T x, uint32_t lane_idx) {
    #pragma unroll
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFFu, (uint32_t)x, i);
        if (lane_idx >= i) x += (T)t;
    }
    return x;
}

// The 64-lane form the port needs.
template<typename T>
__device__ __forceinline__ T scan64(T x, uint32_t lane_idx) {
    #pragma unroll
    for (uint32_t i = 1; i <= 32; i <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFFFFFFFFFFull, (uint32_t)x, i);
        if (lane_idx >= i) x += (T)t;
    }
    return x;
}

__global__ void k_scan(int *as_port, int *as_fixed, int *expected) {
    const unsigned tx = threadIdx.x;
    as_port[tx] = scan32<int>(1, tx % 32);   // what the port does today
    as_fixed[tx] = scan64<int>(1, tx);       // what a 64-lane port must do
    expected[tx] = tx + 1;                   // inclusive prefix sum of 1
}

int main() {
    int *warp_size, *lane_id, *shfl32, *as_port, *as_fixed, *expected;
    unsigned long long *active, *ballot32, *ballot64;
    unsigned *red32, *red64, *any32, *any64;

    cudaMalloc(&warp_size, sizeof(int));
    cudaMalloc(&lane_id, 64 * sizeof(int));
    cudaMalloc(&active, sizeof(unsigned long long));
    cudaMalloc(&ballot32, 64 * sizeof(unsigned long long));
    cudaMalloc(&ballot64, 64 * sizeof(unsigned long long));
    cudaMalloc(&red32, 64 * sizeof(unsigned));
    cudaMalloc(&red64, 64 * sizeof(unsigned));
    cudaMalloc(&any32, 64 * sizeof(unsigned));
    cudaMalloc(&any64, 64 * sizeof(unsigned));
    cudaMalloc(&shfl32, 64 * sizeof(int));
    cudaMalloc(&as_port, 64 * sizeof(int));
    cudaMalloc(&as_fixed, 64 * sizeof(int));
    cudaMalloc(&expected, 64 * sizeof(int));

    k_identity<<<1, 64>>>(warp_size, lane_id, active);
    k_collectives<<<1, 64>>>(ballot32, ballot64, red32, red64, any32, any64, shfl32);
    k_scan<<<1, 64>>>(as_port, as_fixed, expected);

    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) {
        printf("FAILED: %s\n", cudaGetErrorString(e));
        return 1;
    }

    int ws, hl[64], hs[64], hp[64], hf[64], he[64];
    unsigned long long ha;
    unsigned long long hb32[64], hb64[64];
    unsigned hr32[64], hr64[64], hy32[64], hy64[64];

    cudaMemcpy(&ws, warp_size, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&ha, active, sizeof(ha), cudaMemcpyDeviceToHost);
    cudaMemcpy(hl, lane_id, sizeof(hl), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb32, ballot32, sizeof(hb32), cudaMemcpyDeviceToHost);
    cudaMemcpy(hb64, ballot64, sizeof(hb64), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr32, red32, sizeof(hr32), cudaMemcpyDeviceToHost);
    cudaMemcpy(hr64, red64, sizeof(hr64), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy32, any32, sizeof(hy32), cudaMemcpyDeviceToHost);
    cudaMemcpy(hy64, any64, sizeof(hy64), cudaMemcpyDeviceToHost);
    cudaMemcpy(hs, shfl32, sizeof(hs), cudaMemcpyDeviceToHost);
    cudaMemcpy(hp, as_port, sizeof(hp), cudaMemcpyDeviceToHost);
    cudaMemcpy(hf, as_fixed, sizeof(hf), cudaMemcpyDeviceToHost);
    cudaMemcpy(he, expected, sizeof(he), cudaMemcpyDeviceToHost);

    printf("device: %s\n\n", "see mx-smi for the part name");

    printf("== identity ==\n");
    printf("  warpSize                 : %d\n", ws);
    printf("  __lane_id() lane 0/31/32/40/63 : %d %d %d %d %d\n",
           hl[0], hl[31], hl[32], hl[40], hl[63]);
    printf("  __activemask()           : 0x%016llx (%d lanes)\n", ha,
           __builtin_popcountll(ha));

    printf("\n== collectives, voting from lanes 0/32/40, value 1 from 0/32 ==\n");
    printf("  ballot(0xFFFFFFFF)       : 0x%016llx   want 0x1 if grouped by 32,\n",
           hb32[0]);
    printf("                             0x10100000001 if 64-wide\n");
    printf("  ballot(full)             : 0x%016llx\n", hb64[0]);
    printf("  reduce(0xFFFFFFFF, 1@0,32): lane0=%u lane32=%u lane40=%u lane63=%u\n",
           hr32[0], hr32[32], hr32[40], hr32[63]);
    printf("     ^ 1 = low half only (lanes 32..63 excluded), 2 = whole wave\n");
    printf("  reduce(full, 1@0,32)     : lane0=%u lane32=%u lane40=%u lane63=%u\n",
           hr64[0], hr64[32], hr64[40], hr64[63]);
    printf("  any(tx==40, 0xFFFFFFFF)  : lane0=%u lane32=%u lane40=%u\n",
           hy32[0], hy32[32], hy32[40]);
    printf("     ^ 0 everywhere = lane 40 is outside the mask\n");
    printf("  any(tx==40, full)        : lane0=%u lane32=%u lane40=%u\n",
           hy64[0], hy64[32], hy64[40]);
    printf("  shfl_up(0xFFFFFFFF,1)@31 : %d   (777 = read lane 31, 31 = kept own)\n",
           hs[31]);
    printf("  shfl_up(0xFFFFFFFF,1)@32 : %d\n", hs[32]);

    printf("\n== the port's own scan (inclusive prefix sum of 1) ==\n");
    printf("  lane        :");
    for (int i = 0; i < 64; i += 8) printf("%5d", i);
    printf("\n  as the port :");
    for (int i = 0; i < 64; i += 8) printf("%5d", hp[i]);
    printf("   <- every lane >= 32 restarts at 1\n");
    printf("\n  64-lane fix :");
    for (int i = 0; i < 64; i += 8) printf("%5d", hf[i]);
    printf("\n  expected    :");
    for (int i = 0; i < 64; i += 8) printf("%5d", he[i]);
    printf("\n");

    int bad = 0, fixed_bad = 0;
    for (int i = 0; i < 64; ++i) {
        if (hp[i] != he[i]) ++bad;
        if (hf[i] != he[i]) ++fixed_bad;
    }
    printf("\n  VERDICT: port scan wrong on %d/64 lanes; 64-lane scan wrong on %d/64\n",
           bad, fixed_bad);
    return 0;
}
