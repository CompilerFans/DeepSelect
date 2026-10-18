// The launcher's one definition, for the link.
//
// Since P5 ("one configuration tuple, one generated .cu, one dispatch table")
// `topk_coarse12.cuh` declares `launch_coarse12_kernel<kThreads>` `extern
// template` and carries no definition: the shipping extension defines it in the
// generated `csrc/maca_kernels/xcore<N>/instantiations/coarse12_kernel.cu`.
// This probe's link line is just its own directory's TUs, so the definition has
// to come from a file here -- same explicit instantiation the generator writes.
// (Without it the probe fails at link with an undefined
// `launch_coarse12_kernel<640>`; that breakage is P5's, not the P6 rename's.)

#include "../../csrc/maca_kernels/common/topk_coarse12.cuh"

namespace topk_select_coarse12 {

template cudaError_t launch_coarse12_kernel<kThreads>(
    const float *scores, const int32_t *lengths, int32_t *out,
    int32_t *nan_flags, int B, int topk, int64_t stride,
    int default_length, cudaStream_t stream);

}  // namespace topk_select_coarse12
