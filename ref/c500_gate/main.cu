// The gate, printed.
//
//   ./main
//
// The shipping dispatch decides "does the coarse12 route exist in this image"
// at *build* time -- `ARCH_SMEM_PER_AP_BYTES` against the kernel's request, a
// comparison of two compile-time constants (`csrc/structs.h`, `if constexpr`
// in `f32_coarse12_applies`) -- and "is this shape worth it" by the measured
// crossing.  Those are numbers and a boolean, and this driver is the place
// they are stated together with the device's own report of itself, so the next
// person to change a threshold can see what it was.
//
// What this driver checks has changed with the predicate.  It used to feed the
// device's `cudaDevAttrMaxSharedMemoryPerBlockOptin` *into* the gate; the gate
// takes no budget now, so the device's report is printed *beside* the constant
// the gate was compiled with, and the check is that the two agree.  A C500
// image on a part with a smaller arena is the failure that matters, and it is
// a mismatch between exactly those two numbers.
//
// Build (via ref/build_all.sh):
//   cucc -O2 -std=c++20 --offload-arch=xcore1000 main.cu xcore1000_gate.cu -o main

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>

extern "C" bool ref_c500_gate(uint32_t sm_count, uint32_t batches,
                              uint32_t vocab_size, uint32_t topk);
extern "C" int ref_candidate_capacity();
extern "C" int ref_smem_bytes();
extern "C" int ref_row_smem_bytes();
extern "C" int ref_c500_smem_per_ap();
extern "C" int ref_measured_sm_count();
extern "C" int ref_arch_smem_per_ap();
extern "C" int ref_arch_sm_count();
extern "C" int ref_needed_smem();
extern "C" int ref_budget_half();

namespace {

struct Shape {
  uint32_t batches;
  uint32_t vocab;
  uint32_t topk;
  const char *why;
};

// One row per decision the gate makes, including the cells that were measured
// to decide it.  `why` is the reason, not a description.
const Shape kShapes[] = {
    {256, 524288, 2048, "the deep_gemm coarse12 cell"},
    {256, 131072, 2048, "narrow arm: the route is ahead at b16 (measured)"},
    {16, 131072, 2048, "at the narrow floor"},
    {15, 131072, 2048, "just below the narrow floor"},
    {16, 262144, 2048, "wide arm: 16*2048 = 32768 < 114688, so no"},
    {56, 262144, 2048, "wide arm, at the measured crossing (56*2048)"},
    {55, 262144, 2048, "one batch below it"},
    {64, 262144, 512, "k=512: 64*512 = 32768, the product declines it"},
    {256, 524288, 4096, "topk above the arena capacity"},
    {1024, 2048, 2048, "at the original's own width bound"},
    {1024, 2047, 2048, "one column below it"},
};

}  // namespace

int main()
{
  int device = 0;
  cudaDeviceProp prop{};
  if (cudaGetDevice(&device) != cudaSuccess
      || cudaGetDeviceProperties(&prop, device) != cudaSuccess) {
    std::fprintf(stderr, "FAIL: cannot read device properties\n");
    return 1;
  }

  // The device's own answer, read the same way a device query would read it.
  // It is *not* an input to the gate any more -- it is the thing the gate's
  // compiled-in constant is checked against.
  int smem_per_ap = 0;
  const cudaError_t attr = cudaDeviceGetAttribute(
      &smem_per_ap, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);

  std::printf("device %s | APs %u\n", prop.name, prop.multiProcessorCount);
  std::printf("smem/AP (optin) %d bytes%s\n", smem_per_ap,
              attr == cudaSuccess ? "" : "  [attribute query FAILED]");
  std::printf("compiled-in budget (ARCH_SMEM_PER_AP_BYTES) %d bytes\n",
              ref_arch_smem_per_ap());
  std::printf("the gate compares it against max(route %d, split %d) = %d\n",
              ref_smem_bytes(), ref_row_smem_bytes(), ref_needed_smem());
  std::printf("arena %d bytes -> %d candidates (a top-k answer must fit here)\n",
              ref_smem_bytes(), ref_candidate_capacity());
  std::printf("budget half of the gate: %s (a compile-time fact in this image)\n",
              ref_budget_half() ? "passes" : "FAILS");
  std::printf("provenance half: the ladder was measured at %u APs; this device "
              "reports %u -- %s\n", ref_measured_sm_count(),
              prop.multiProcessorCount,
              (uint32_t)prop.multiProcessorCount == ref_measured_sm_count()
                  ? "same, so the tuning applies"
                  : "different, so the tuning is withheld");

  std::printf("\n%-8s %-8s %-6s %-6s  %s\n", "batches", "vocab", "topk", "gate",
              "why");
  int mismatches = 0;
  for (const Shape &s : kShapes) {
    const bool got = ref_c500_gate((uint32_t)prop.multiProcessorCount,
                                   s.batches, s.vocab, s.topk);
    std::printf("%-8u %-8u %-6u %-6s  %s\n", s.batches, s.vocab, s.topk,
                got ? "yes" : "no", s.why);
  }

  // The claims the gate's own comments make, checked rather than asserted in
  // prose: the arena holds a whole top-k answer, it holds the 4096-bin coarse
  // histogram in the same words, and the device's own report is the number the
  // image was compiled with.
  if (ref_candidate_capacity() < 2048) {
    std::fprintf(stderr, "FAIL: %d candidates cannot hold kMaxTopK=2048\n",
                 ref_candidate_capacity());
    ++mismatches;
  }
  if ((long)ref_smem_bytes() < 4096L * 4L) {
    std::fprintf(stderr,
                 "FAIL: %d bytes cannot hold the 4096-bin coarse histogram\n",
                 ref_smem_bytes());
    ++mismatches;
  }
  if (ref_arch_sm_count() != ref_measured_sm_count()) {
    std::fprintf(stderr,
                 "FAIL: the image's AP count (%d) is not the one the ladder was "
                 "measured on (%d) -- the budget half would pass and the "
                 "tuning half would silently apply\n",
                 ref_arch_sm_count(), ref_measured_sm_count());
    ++mismatches;
  }
  // The one that matters: a mismatch here means an image is being run on a
  // part whose arena is not the one it was compiled for.  On this device the
  // two agree by construction; the check is that they still do.
  if (attr == cudaSuccess && smem_per_ap != ref_arch_smem_per_ap()) {
    std::fprintf(stderr,
                 "FAIL: this device reports %d bytes/AP but this image was "
                 "compiled for %d -- `_binding.family_suffix()` should have "
                 "loaded a different artifact\n",
                 smem_per_ap, ref_arch_smem_per_ap());
    ++mismatches;
  } else if (attr == cudaSuccess && smem_per_ap != ref_c500_smem_per_ap()) {
    std::fprintf(stderr,
                 "NOTE: this device reports %d bytes/AP, not the C500's %d -- "
                 "the threshold still applies, but the ladder behind it was "
                 "measured on a %d-byte device\n",
                 smem_per_ap, ref_c500_smem_per_ap(), ref_c500_smem_per_ap());
  }
  std::printf("\n%s\n", mismatches == 0 ? "gate constants consistent"
                                        : "gate constants INCONSISTENT");
  return mismatches == 0 ? 0 : 1;
}
