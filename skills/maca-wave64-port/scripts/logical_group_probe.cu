// logical_group_probe -- can a 32-lane-logical-group design be salvaged on a
// 64-lane wave by fixing only the mask?
//
// MEASURED (MACA 3.8.1.3, MetaX C600-U, 2026-09-12).  x = lane+1 distinct per
// logical lane; want = 528 (in-group sum of 1..32).  "wrong" = lanes whose
// result was not 528.
//
//   block   logical groups  reduction mask used
//   -----   --------------  ---------------------------------------------
//      64        2          0xFFFFFFFF  32/64 wrong   group-mask   0/64 wrong
//     128        4          0xFFFFFFFF  64/128 wrong  group-mask  64/128 wrong
//     256        8          0xFFFFFFFF 128/256 wrong  group-mask 192/256 wrong
//     512       16          0xFFFFFFFF 256/512 wrong  group-mask 448/512 wrong
//
//   __activemask() is the PHYSICAL wave mask and is 0xffffffffffffffff for
//   every block size, so using it makes the reduction MORE wrong, not less:
//   at 128 threads it sums two logical groups (128/128 wrong).
//
// THE CONCLUSION, and it is structural rather than a bug: for 128 threads the
// logical group-mask is right for groups 0 and 1 and impossible for groups 2
// and 3.  Logical group g owns physical lanes 32g..32g+31, and a 64-bit mask
// can name only the first two of those windows -- from group 2 up the mask is
// not just wrong, it is inexpressible, and the shift is UB.  The scheme has no
// representation on this hardware.
//
// So the correct model is the PHYSICAL one, and the fix is the plain wide one:
//   lane_idx  = threadIdx.x % 64        NUM_WARPS = NUM_THREADS / 64
//   mask      = 0xFFFFFFFFFFFFFFFFull  (or __activemask() where converged)
// A warp-scope operation can only reach lanes that share a wave anyway, so
// widening to 64 is not merely a patch -- it is what the algorithm needs.

// Does __activemask() cover a whole multi-group block, and does it fix the
// unguarded collectives the port relies on?
#include <cstdint>
#include <cstdio>
#include <cuda_runtime.h>
#define LW 32

__device__ __forceinline__ unsigned long long gmask(unsigned tid) {
    const unsigned g = tid / LW;
    return g < 2 ? (0xFFFFFFFFull << (LW * g)) : 0ull;
}

__global__ void k(unsigned long long *live, unsigned *red_lo, unsigned *red_act,
                  unsigned *red_gmask, unsigned *want, int nthreads) {
    const unsigned tid = threadIdx.x, lane = tid % LW;
    const unsigned x = lane + 1;               // distinct per logical lane
    live[tid] = __activemask();
    red_lo[tid]   = __reduce_add_sync_impl((uint64_t)0xFFFFFFFFull, x);
    red_act[tid]  = __reduce_add_sync_impl((uint64_t)__activemask(), x);
    red_gmask[tid] = __reduce_add_sync_impl((uint64_t)gmask(tid), x);
    want[tid] = (tid < (unsigned)nthreads) ? 528u : 0u;
}

int main() {
    for (int nt : {64, 128, 256, 512}) {
        unsigned long long *live; unsigned *a,*b,*c,*w;
        cudaMalloc(&live, nt*8); cudaMalloc(&a,nt*4); cudaMalloc(&b,nt*4);
        cudaMalloc(&c,nt*4); cudaMalloc(&w,nt*4);
        k<<<1,nt>>>(live,a,b,c,w,nt);
        if (cudaDeviceSynchronize()!=cudaSuccess){printf("FAIL %s\n",cudaGetErrorString(cudaGetLastError()));return 1;}
        unsigned long long hl[512]; unsigned ha[512],hb[512],hc[512],hw[512];
        cudaMemcpy(hl,live,nt*8,cudaMemcpyDeviceToHost);
        cudaMemcpy(ha,a,nt*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hb,b,nt*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hc,c,nt*4,cudaMemcpyDeviceToHost);
        cudaMemcpy(hw,w,nt*4,cudaMemcpyDeviceToHost);
        int ba=0,bb=0,bc=0;
        for(int i=0;i<nt;i++){ba+=(ha[i]!=hw[i]);bb+=(hb[i]!=hw[i]);bc+=(hc[i]!=hw[i]);}
        printf("block %3d threads (%2d logical groups): live[0]=0x%016llx popcount=%d\n",
               nt, nt/LW, hl[0], __builtin_popcountll(hl[0]));
        printf("   reduce wrong -- 0xFFFFFFFF: %3d/%d    __activemask(): %3d/%d    group-mask: %3d/%d\n",
               ba,nt,bb,nt,bc,nt);
    }
    return 0;
}
