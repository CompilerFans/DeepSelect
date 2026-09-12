// cub_vs_builtin_probe -- can the port's cross-lane primitives be replaced by
// CUB, or by a MACA builtin, and which is cheaper?
//
// MEASURED (MACA 3.8.1.3, MetaX C600-U, cucc/mxcc, 2026-09-12)
// -----------------------------------------------------------
//   inclusive prefix sum of 1 over a 64-lane wave, wrong lanes out of 64:
//
//     port's current 32-lane scan : 32/64 wrong
//     cub::WarpScan<int>          :  0/64 wrong
//     bsm_bpermute butterfly      :  0/64 wrong
//
//   full-wave sum of 1:  cub::WarpReduce = 64 (correct), bpermute = 64
//
//   device instruction count (`mxcc -aop -S -maca-device-only`, same flags as
//   the build):
//
//     cub scan 66   cub reduce 69   bpermute scan 54   bpermute reduce 42
//
//   So CUB is a legitimate answer -- it gets the width right -- and the builtin
//   is ~1.3-1.6x cheaper, because CUB's shuffles carry the __shfl_*_sync
//   wrappers' per-call index arithmetic.
//
//   __builtin_mxc_readlane(x, src): every src returned the caller's OWN value,
//   from lanes 0, 31, 32 and 63 alike.  Do not use it to replace a shuffle.
//
// BUILD: same toolchain as the build --
//   cucc cub_vs_builtin_probe.cu -o probe --offload-arch=<target> -O3 -std=c++17
// (the repo's skills/maca-wave64-port/scripts/run_probe.sh does this for
// wave64_probe.cu; this file is compiled the same way.)

// Can the port's cross-lane primitives be replaced by CUB / MACA builtins?
//
// Three candidate replacements for `csrc/xcore1600/utils.cuh`'s 32-lane scan,
// measured side by side on a 64-lane wave against a reference computed with no
// cross-lane primitive at all:
//
//   1. cub::WarpScan / cub::WarpReduce  -- MACA's CUB, which claims width 64
//   2. a hand-written bsm_bpermute butterfly (the tree's own idiom)
//   3. __builtin_mxc_mov_shfl / readlane / writelane
//
// plus the port's current 32-lane form, so the failure is visible next to the
// fixes.
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#include <cub/cub.cuh>

// ── the port's current form, verbatim from csrc/xcore1600/utils.cuh ─────────
template<typename T>
__device__ __forceinline__ T scan_port(T x, uint32_t lane_idx) {
    for (uint32_t i = 1; i <= 16; i <<= 1) {
        uint32_t t = __shfl_up_sync(0xFFFFFFFFu, (uint32_t)x, i);
        if (lane_idx >= i) x += (T)t;
    }
    return x;
}

// ── 1. CUB ──────────────────────────────────────────────────────────────────
// MACA's CUB requires explicit temp storage (this CUB generation does).
__shared__ cub::WarpScan<int>::TempStorage g_scan_tmp;
__shared__ cub::WarpReduce<int>::TempStorage g_red_tmp;

template<typename T>
__device__ __forceinline__ T scan_cub(T x) {
    T out = x;
    cub::WarpScan<T> ws(g_scan_tmp);
    ws.InclusiveSum(x, out);
    return out;
}

template<typename T>
__device__ __forceinline__ T reduce_cub(T x) {
    cub::WarpReduce<T> wr(g_red_tmp);
    return wr.Sum(x);
}

// ── 2. bsm_bpermute butterfly ───────────────────────────────────────────────
// The hardware index is byte-addressed (dest[n] = data[index[n]/4 % 64]),
// hence the <<2.  A 64-lane Hillis-Steele scan in 6 steps.
__device__ __forceinline__ int scan_bpermute(int x) {
    const unsigned lane = __lane_id();
    #pragma unroll
    for (int d = 1; d <= 32; d <<= 1) {
        int n = __builtin_mxc_bsm_bpermute(((lane - d) & 63) << 2, x);
        // lanes below d have no source; bpermute wraps, so mask the add
        if (lane >= (unsigned)d) x += n;
    }
    return x;
}

__device__ __forceinline__ int reduce_bpermute(int x) {
    const unsigned lane = __lane_id();
    #pragma unroll
    for (int d = 32; d >= 1; d >>= 1) {
        int n = __builtin_mxc_bsm_bpermute(((lane + d) & 63) << 2, x);
        x += n;
    }
    return x;
}

// ── 3. mov_shfl / readlane / writelane ──────────────────────────────────────
// readlane: read one arbitrary lane's value.  Whether it reaches lanes 32..63
// of a 64-lane wave is the question -- so probe several source lanes.
__device__ __forceinline__ int readlane_probe(int x, int src) {
    return __builtin_mxc_readlane(x, src);
}

// ── kernels ─────────────────────────────────────────────────────────────────
__global__ void k_scan(int *port, int *cub, int *bperm, int *want) {
    const unsigned tx = threadIdx.x;
    port[tx]  = scan_port<int>(1, tx % 32);
    cub[tx]   = scan_cub<int>(1);
    bperm[tx] = scan_bpermute(1);
    want[tx]  = tx + 1;
}

__global__ void k_reduce(int *cub, int *bperm, int *want) {
    int c = reduce_cub<int>(1);
    int b = reduce_bpermute(1);
    if (threadIdx.x == 0) { cub[0] = c; bperm[0] = b; want[0] = 64; }
}

__global__ void k_readlane(int *out) {
    const unsigned tx = threadIdx.x;
    // every lane reads four source lanes; a src >= 32 is the question
    if (tx == 0) {
        out[0] = readlane_probe(1000 + (int)tx, 0);
        out[1] = readlane_probe(1000 + (int)tx, 31);
        out[2] = readlane_probe(1000 + (int)tx, 32);
        out[3] = readlane_probe(1000 + (int)tx, 63);
    }
}

int main() {
    int *port, *cubr, *bperm, *want, *o0, *o1, *o2, *txo;
    cudaMalloc(&port, 256); cudaMalloc(&cubr, 256); cudaMalloc(&bperm, 256);
    cudaMalloc(&want, 256);
    cudaMalloc(&o0, 256); cudaMalloc(&o1, 256); cudaMalloc(&o2, 256);
    cudaMalloc(&txo, 256);

    k_scan<<<1, 64>>>(port, cubr, bperm, want);
    k_reduce<<<1, 64>>>(o0, o1, o2);
    k_readlane<<<1, 64>>>(txo);
    const cudaError_t e = cudaDeviceSynchronize();
    if (e != cudaSuccess) { printf("FAILED: %s\n", cudaGetErrorString(e)); return 1; }

    int hp[64], hc[64], hb[64], hw[64], r[3], rl[64];
    cudaMemcpy(hp, port, 256, cudaMemcpyDeviceToHost);
    cudaMemcpy(hc, cubr, 256, cudaMemcpyDeviceToHost);
    cudaMemcpy(hb, bperm, 256, cudaMemcpyDeviceToHost);
    cudaMemcpy(hw, want, 256, cudaMemcpyDeviceToHost);
    { int t; cudaMemcpy(&t, o0, 4, cudaMemcpyDeviceToHost); r[0] = t; }  // cub
    { int t; cudaMemcpy(&t, o1, 4, cudaMemcpyDeviceToHost); r[1] = t; }  // bpermute
    { int t; cudaMemcpy(&t, o2, 4, cudaMemcpyDeviceToHost); r[2] = t; }  // want
    cudaMemcpy(rl, txo, 256, cudaMemcpyDeviceToHost);

    printf("inclusive prefix sum of 1 over a 64-lane wave\n");
    printf("  lane          :");
    for (int i = 0; i < 64; i += 8) printf("%5d", i);
    printf("\n  port (32-lane):");
    for (int i = 0; i < 64; i += 8) printf("%5d", hp[i]);
    printf("\n  cub::WarpScan :");
    for (int i = 0; i < 64; i += 8) printf("%5d", hc[i]);
    printf("\n  bsm_bpermute  :");
    for (int i = 0; i < 64; i += 8) printf("%5d", hb[i]);
    printf("\n  expected      :");
    for (int i = 0; i < 64; i += 8) printf("%5d", hw[i]);

    int bp = 0, bc = 0, bb = 0;
    for (int i = 0; i < 64; ++i) {
        bp += (hp[i] != hw[i]);
        bc += (hc[i] != hw[i]);
        bb += (hb[i] != hw[i]);
    }
    printf("\n  wrong lanes -- port: %d/64   cub: %d/64   bpermute: %d/64\n",
           bp, bc, bb);

    printf("\nfull-wave sum of 1: cub=%d  bpermute=%d  want=%d\n", r[0], r[1], r[2]);
    printf("__builtin_mxc_readlane(x, src), x = 1000+lane, from lane 0:\n");
    printf("  src=0 -> %d (want 1000)   src=31 -> %d (want 1031)\n", rl[0], rl[1]);
    printf("  src=32 -> %d (want 1032)  src=63 -> %d (want 1063)\n", rl[2], rl[3]);
    return 0;
}
