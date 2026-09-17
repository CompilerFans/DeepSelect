// The kernel TU's external surface, for the driver.
//
// `csrc/xcore1000/dg_coarse12.cuh` is the header of record; this declares only
// the wrappers `xcore1000_dg_coarse12.cu` defines around it.  Kept apart so the
// driver's translation unit does not contain the kernel -- see that file.
#pragma once

#include <cstdint>
#include <cuda_runtime.h>

namespace rk {
namespace dg12_ref {

cudaError_t launch(const float *scores, const int32_t *lengths, int32_t *out,
                   int32_t *nan_flags, int B, int topk, int64_t stride,
                   int default_length, cudaStream_t stream);

int smem_bytes();
int candidate_capacity();
int threads();
int coarse_bins();

}  // namespace dg12_ref
}  // namespace rk
