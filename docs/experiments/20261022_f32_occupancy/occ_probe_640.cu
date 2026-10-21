#include <cstdio>
#include <csrc/maca_kernels/common/topk_coarse12.cuh>
int main() {
    int n_smem_16k = -1, n_smem_0 = -1, n_block_512 = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n_smem_16k, topk_select_coarse12::topk_coarse12_kernel<640>, 640, 16384);
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &n_smem_0, topk_select_coarse12::topk_coarse12_kernel<640>, 640, 0);
    printf("coarse12<640> occupancy: dyn_smem=16KB -> %d CTA/SM, dyn_smem=0 -> %d CTA/SM\n",
           n_smem_16k, n_smem_0);
    return 0;
}
