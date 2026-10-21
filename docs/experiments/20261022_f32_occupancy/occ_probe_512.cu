// Driver occupancy for the real coarse12 kernel at the landed width (512).
// Same instrument as /tmp/occ_c12.cu (which measured 640), same header, so the
// two answers are comparable.  Run with CUDA_VISIBLE_DEVICES=1.
#include <cstdio>
#include <csrc/maca_kernels/common/topk_coarse12.cuh>

int main() {
    int n16 = -1, n0 = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n16, topk_select_coarse12::topk_coarse12_kernel<512>, 512, 16384);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n0, topk_select_coarse12::topk_coarse12_kernel<512>, 512, 0);
    printf("coarse12<512> occupancy: dyn_smem=16KB -> %d CTA/SM (%d threads), dyn_smem=0 -> %d CTA/SM (%d threads)\n",
           n16, n16 * 512, n0, n0 * 512);
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) printf("last error: %s\n", cudaGetErrorString(e));
    return 0;
}
