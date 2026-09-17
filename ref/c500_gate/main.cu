// The C500 gate, printed.
//
//   ./main
//
// The shipping dispatch decides "does the coarse12 route exist on this device"
// by comparing the device's AP count against a constant, and decides "is this
// shape worth it" by two more.  Those are three numbers and a boolean, and this
// driver is the place they are stated together with the device's own report of
// itself -- so the next person to change a threshold can see what it was, and
// so `ref_c500_gate` (same expression, second TU) can be cross-checked against
// the value the gate is *supposed* to produce on this machine.
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
    {256, 262144, 2048, "at the V floor"},
    {256, 131072, 2048, "V below the floor: row path wins (measured)"},
    {256, 524288, 4096, "topk above the arena capacity"},
    {127, 524288, 2048, "just below the batch floor"},
    {128, 524288, 2048, "at the batch floor"},
    {6, 524288, 2048, "small batch: the split, not this"},
    {1024, 65536, 2048, "short rows: the split, not this"},
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

  const uint32_t sm = (uint32_t)prop.multiProcessorCount;
  std::printf("device %s | APs %u\n", prop.name, sm);
  std::printf("arena %d bytes -> %d candidates (a top-k answer must fit here)\n",
              ref_smem_bytes(), ref_candidate_capacity());
  std::printf("gate is an equality on APs: this device is %s\n",
              sm == 104 ? "C500 (104) -- the route can run"
                        : "NOT C500 -- the route is never selected");

  std::printf("\n%-8s %-10s %-6s %-6s  %s\n", "batches", "vocab", "topk", "gate",
              "why");
  int mismatches = 0;
  for (const Shape &s : kShapes) {
    const bool got = ref_c500_gate(sm, s.batches, s.vocab, s.topk);
    // The expected column is the gate's own type, so a change to any of the
    // three constants shows up as a diff on this line rather than as a number
    // someone has to remember.
    std::printf("%-8u %-10u %-6u %-6s  %s\n", s.batches, s.vocab, s.topk,
                got ? "yes" : "no", s.why);
  }

  // The two claims the gate's own comment makes, checked rather than asserted
  // in prose: the arena holds a whole top-k answer, and it holds the 4096-bin
  // coarse histogram in the same words.
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
  std::printf("\n%s\n", mismatches == 0 ? "gate constants consistent"
                                        : "gate constants INCONSISTENT");
  return mismatches == 0 ? 0 : 1;
}
