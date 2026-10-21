// Semantics probe for __builtin_mxc_mov_shfl -- run ON DEVICE, print the lane
// table. Self-contained executable:
//   cucc -O3 --offload-arch=xcore1600 tree_probe.cu -o /tmp/tree_probe
//   /tmp/tree_probe
//   mxcc -aop -S -maca-device-only tree_probe.cu -o /tmp/tree_probe.s
//
// What it establishes (the walk A/B's rate does NOT depend on these, the
// kernel patch DOES):
//   1. Row Shift direction: does 0x111 ("shift right by 1") make lane i read
//      lane i+1 or lane i-1?  -> which lanes are the group leaders.
//   2. bc behavior on out-of-row reads (bc=1 keep own / bc=0 zero-fill).
//   3. rmsk/bmsk masking semantics (source always readable, mask gates the
//      DESTINATION lanes only -- AMD DPP convention; verify).
//   4. The emitted instruction class (from the .s): mov_shfl must NOT lower
//      to sm_bperm_b32 / a shared-memory op -- that is the whole premise.
#include <cstdio>
#include <cstdint>

__global__ void probe_kernel(int *out) {
    const int tx = (int)threadIdx.x;
    const int v = 1000 + tx;  // source lane encoded in the value
    // row shift right, amounts 1/2/4, bc=1 (keep own on out-of-range)
    out[0 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x111, 0xf, 0xf, 1);
    out[1 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x112, 0xf, 0xf, 1);
    out[2 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x114, 0xf, 0xf, 1);
    // row shift left, amounts 1/2/4, bc=1
    out[3 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x101, 0xf, 0xf, 1);
    out[4 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x102, 0xf, 0xf, 1);
    out[5 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x104, 0xf, 0xf, 1);
    // bc=0 zero-fill on out-of-range
    out[6 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x111, 0xf, 0xf, 0);
    // bmsk: only banks 0+2 (lanes 0-3, 8-11 of each row) receive
    out[7 * 64 + tx] = __builtin_mxc_mov_shfl(v, 0x111, 0xf, 0x5, 1);
    // readlane / readfirstlane
    out[8 * 64 + tx] = __builtin_mxc_readlane(v, 7);
    out[9 * 64 + tx] = __builtin_mxc_readfirstlane(v);
}

int main() {
    int *d = nullptr;
    cudaMalloc(&d, 10 * 64 * sizeof(int));
    probe_kernel<<<1, 64>>>(d);
    int h[10 * 64];
    cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
    static const char *names[10] = {
        "shr1 bc1", "shr2 bc1", "shr4 bc1", "shl1 bc1", "shl2 bc1",
        "shl4 bc1", "shr1 bc0", "shr1 bmsk=0x5", "readlane(7)", "readfirstlane"};
    for (int t = 0; t < 10; t++) {
        printf("%-14s lane 0-3: %4d %4d %4d %4d | lane 7-9: %4d %4d %4d"
               " | lane 14-17: %4d %4d %4d %4d | lane 63: %4d\n",
               names[t], h[t * 64 + 0], h[t * 64 + 1], h[t * 64 + 2],
               h[t * 64 + 3], h[t * 64 + 7], h[t * 64 + 8], h[t * 64 + 9],
               h[t * 64 + 14], h[t * 64 + 15], h[t * 64 + 16], h[t * 64 + 17],
               h[t * 64 + 63]);
    }
    cudaFree(d);
    return 0;
}
