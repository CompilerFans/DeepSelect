// Kernel side of the port driver.
//
// Split from `main.cu` the way `ref/deep_gemm` splits its driver from
// `xcore1000_fp32_topk.cu`, and for the same reason: a `__global__` definition
// in two translation units is two host stubs for one kernel, which the linker
// rejects.  So this TU owns the kernel and the driver only calls it.
//
// What is under measurement is `csrc/xcore1000/dg_coarse12.cuh` **as the
// shipping extension compiles it** -- included here, not copied.  Everything in
// it is `inline`, so a second TU needs one external symbol to link against,
// which is what the wrapper below is; the `static_assert`s in the header are
// therefore checked in a TU that contains nothing else, which is the property
// that makes it reusable at all.
#include "../../csrc/xcore1000/dg_coarse12.cuh"

namespace rk {
namespace dg12_ref {

// `default_length` is part of the shipping signature, not a driver convenience:
// the driver here passes `n_cols` for both `stride` and `default_length` because
// its rows are contiguous.  The product needs them separately -- the harness
// slices `s[:, :L]`, so a row's stride is the parent tensor's width while its
// length is `L`.
cudaError_t launch(const float *scores, const int32_t *lengths, int32_t *out,
                   int32_t *nan_flags, int B, int topk, int64_t stride,
                   int default_length, cudaStream_t stream)
{
    return rk::dg12::launch_topk_coarse12(scores, lengths, out, nan_flags, B,
                                          topk, stride, default_length, stream);
}

// The two constants the driver prints: the arena the header sizes itself from,
// and the candidate capacity that falls out of it.  Returned rather than
// repeated so the driver cannot disagree with the kernel about them.
int smem_bytes() { return (int)rk::dg12::kSmemBytes; }
int candidate_capacity() { return rk::dg12::kCandidateCapacity; }
int threads() { return rk::dg12::kThreads; }
int coarse_bins() { return rk::dg12::kCoarseBins; }

}  // namespace dg12_ref
}  // namespace rk
