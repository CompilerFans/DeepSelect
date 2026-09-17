// Witness for the large-row defect, not a timing driver.
//
// For a given (n_rows, n_cols) it searches the returned TopK of each row for the
// bit pattern ref-tester saw, by re-running with an explicit per-row sequence
// length equal to n_cols (so dispatch stays on the radix path) and scanning a
// prefix of the output. Prints the bad slot indices, their row positions, their
// scores and the true kth value, so the pattern can be identified from data
// rather than guessed at.
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include "xcore1000_topk_v1.h"

#define CK(e) do{cudaError_t r=(e); if(r){printf("ERR %s @%d\n",cudaGetErrorString(r),__LINE__);exit(1);} }while(0)

int main(int argc, char** argv) {
  const int rows = argc > 1 ? atoi(argv[1]) : 1;
  const int cols = argc > 2 ? atoi(argv[2]) : 65536;
  const int scan = argc > 3 ? atoi(argv[3]) : 512;
  const size_t n = (size_t)rows * cols;
  std::vector<float> h(n);
  std::mt19937 rng(0xC0FFEEu + (uint32_t)(rows * 31 + cols));
  std::uniform_real_distribution<float> d(-1.0f, 1.0f);
  for (auto& v : h) v = d(rng);

  float* ds; int32_t *dl, *dout, *dp;
  CK(cudaMalloc(&ds, n * 4));
  CK(cudaMalloc(&dl, rows * 4));
  CK(cudaMalloc(&dout, (size_t)rows * 512 * 4));
  CK(cudaMalloc(&dp, (size_t)cols * 4));
  std::vector<int32_t> lens(rows, cols);
  std::vector<int32_t> page(cols);
  for (int i = 0; i < cols; ++i) page[i] = i;
  CK(cudaMemcpy(ds, h.data(), n * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dl, lens.data(), rows * 4, cudaMemcpyHostToDevice));
  CK(cudaMemcpy(dp, page.data(), (size_t)cols * 4, cudaMemcpyHostToDevice));
  CK(cudaMemset(dout, 0xFF, (size_t)rows * 512 * 4));
  mcoplib_topk_transform(ds, dl, dout, dp, rows, cols, 512, 0);
  CK(cudaDeviceSynchronize());
  std::vector<int32_t> out((size_t)rows * 512);
  CK(cudaMemcpy(out.data(), dout, out.size() * 4, cudaMemcpyDeviceToHost));

  for (int r = 0; r < rows; ++r) {
    const float* row = h.data() + (size_t)r * cols;
    std::vector<int32_t> ord(cols);
    for (int i = 0; i < cols; ++i) ord[i] = i;
    std::nth_element(ord.begin(), ord.begin() + 511, ord.end(),
                     [row](int32_t a, int32_t b) { return row[a] > row[b]; });
    const float kth = row[ord[511]];
    const int32_t slot_of_kth = ord[511];
    int nbad = 0;
    printf("row %d: kth=%.9g (slot %d)\n", r, (double)kth, slot_of_kth);
    for (int i = 0; i < scan; ++i) {
      const int32_t g = out[(size_t)r * 512 + i];
      if (g < 0 || g >= cols) { printf("  slot %d: OUT OF RANGE %d\n", i, g); ++nbad; continue; }
      const float v = row[g];
      if (v < kth) {
        printf("  slot %d: idx %d val %.9g  kth-slot %d  idx-kth %d  within-512-of-kth %d\n",
               i, g, (double)v, slot_of_kth, g - slot_of_kth,
               std::abs(g - slot_of_kth) < 512);
        ++nbad;
      }
    }
    printf("  -> %d of %d scanned slots hold a value below the true kth\n", nbad, scan);
  }
  return 0;
}
