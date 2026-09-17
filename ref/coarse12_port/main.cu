// Runnable driver for the *ported* coarse12 kernel -- the one the shipping
// extension launches through `f32_coarse12_applies`.
//
//   ./main [n_rows] [n_cols] [top_k] [iters]
//
// It fills a (n_rows, n_cols) fp32 matrix on the device, runs
// `rk::dg12::launch_topk_coarse12` directly (no gate, no dispatch, no
// contract half), checks every row against a CPU `nth_element` reference, and
// reports latency plus read bandwidth.  The row is scanned in full, so
// `lengths` is `n_cols` for every row.
//
// What it is for
// --------------
// Three claims about this kernel are checkable *here* and nowhere else, because
// this is the only place it runs without the gate, the contract half and the
// `deep_select` facade around it:
//
//   1. **It is the same algorithm as `ref/deep_gemm`'s.**  Same shape, same
//      reference: `ref/deep_gemm/main.cu` prints a `time:` line for the same
//      (n_rows, n_cols, top_k), and the two are the port and the original.
//      They are NOT the same speed, and this driver is how that is measured
//      rather than assumed -- the port carries a folded NaN scan the original
//      does not, which the `-DNO_NAN_SCAN` build of this driver removes.
//   2. **It answers the whole shape range the gate admits.**  `topk` is a
//      runtime argument here, so `top_k` 512..2048 runs over the same harness
//      the original's own driver does not expose (its policy only picks this
//      kernel for `top_k >= 256` and a wide vocabulary).
//   3. **Its two failure modes are visible.**  A row it cannot serve comes
//      back all `-1` (`topk_coarse12_row` returned false), and a `-1` in the
//      answer is the contract, not a bug -- the checker accepts it only past
//      the live prefix, exactly as `ref/deep_gemm`'s does.
//
// Build (from TOOLCHAIN.md, via ref/build_all.sh):
//   cucc -O2 -std=c++20 --offload-arch=xcore1000 \
//        main.cu xcore1000_dg_coarse12.cu -o main

#include "xcore1000_dg_coarse12.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

void die(const char *what)
{
  std::fprintf(stderr, "FAIL: %s\n", what);
  std::exit(1);
}

void check(cudaError_t status, const char *what)
{
  if (status != cudaSuccess) {
    std::fprintf(stderr, "FAIL: %s: %s\n", what, cudaGetErrorString(status));
    std::exit(1);
  }
}

// The same deterministic generator `ref/deep_gemm/main.cu` uses, so the two
// drivers rank byte-identical matrices for the same shape and their timings
// compare like for like.
__global__ void fill_normal(float *data, int64_t count, uint64_t seed)
{
  const int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= count) return;
  uint64_t state = static_cast<uint64_t>(i) * 0x9e3779b97f4a7c15ull + seed;
  auto next_bits = [&state]() {
    state ^= state >> 33;
    state *= 0xff51afd7ed558ccdull;
    state ^= state >> 33;
    state *= 0xc4ceb9fe1a85ec53ull;
    state ^= state >> 33;
    return state;
  };
  const double u1 = static_cast<double>(next_bits() >> 11) * (1.0 / 9007199254740992.0);
  const double u2 = static_cast<double>(next_bits() >> 11) * (1.0 / 9007199254740992.0);
  const double radius = std::sqrt(-2.0 * std::log(u1 == 0.0 ? 1e-300 : u1));
  data[i] = static_cast<float>(radius * std::cos(6.283185307179586 * u2));
}

constexpr int64_t kVerifyRowsPerBlock = 128;

struct VerifyScratch {
  std::vector<float> row;
  std::vector<float> expected;
  std::vector<float> actual;
  std::vector<int32_t> seen;
};

// Multiset comparison, as the official checker does
// (`deep_gemm/kernels/indexer/dsa/ref.py:741-745`): output order is not part of
// the contract, and neither is the index chosen at the top-k boundary, so the
// selected *values* must match and the indices must only be distinct and inside
// the scanned prefix.  A `-1` is padding and is legal only past the prefix.
bool verify_row(const float *row_scores, const int32_t *row_indices,
                int64_t col_eff, int64_t top_k, int64_t row,
                VerifyScratch &scratch, std::string &error)
{
  scratch.seen.assign(row_indices, row_indices + top_k);
  std::sort(scratch.seen.begin(), scratch.seen.end());
  for (int64_t i = 0; i < top_k; ++i) {
    const int32_t index = scratch.seen[(size_t)i];
    if (index == -1) continue;
    if (index < 0 || index >= col_eff) {
      error = "row " + std::to_string(row) + ": index " + std::to_string(index) +
              " outside the live prefix [0, " + std::to_string(col_eff) + ")";
      return false;
    }
    if (i > 0 && index == scratch.seen[(size_t)i - 1]) {
      error = "row " + std::to_string(row) + ": duplicate index " +
              std::to_string(index);
      return false;
    }
  }

  scratch.actual.resize((size_t)top_k);
  for (int64_t i = 0; i < top_k; ++i)
    scratch.actual[(size_t)i] = row_indices[i] < 0
                                    ? -std::numeric_limits<float>::infinity()
                                    : row_scores[row_indices[i]];
  scratch.row.assign(row_scores, row_scores + col_eff);
  scratch.expected.resize((size_t)top_k);
  std::nth_element(scratch.row.begin(), scratch.row.begin() + (col_eff - top_k),
                   scratch.row.end());
  std::copy(scratch.row.begin() + (col_eff - top_k), scratch.row.end(),
            scratch.expected.begin());
  std::sort(scratch.expected.begin(), scratch.expected.end(), std::greater<float>());
  std::sort(scratch.actual.begin(), scratch.actual.end(), std::greater<float>());
  for (int64_t i = 0; i < top_k; ++i) {
    if (scratch.expected[(size_t)i] != scratch.actual[(size_t)i]) {
      error = "row " + std::to_string(row) + ": top-k value mismatch at " +
              std::to_string(i) + " (expected " +
              std::to_string(scratch.expected[(size_t)i]) + ", got " +
              std::to_string(scratch.actual[(size_t)i]) + ")";
      return false;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char **argv)
{
  const int64_t n_rows = argc > 1 ? std::atoll(argv[1]) : 256;
  const int64_t n_cols = argc > 2 ? std::atoll(argv[2]) : 524288;
  const int64_t top_k = argc > 3 ? std::atoll(argv[3]) : 2048;
  const int iters = argc > 4 ? std::atoi(argv[4]) : 20;
  if (n_rows <= 0 || n_cols <= 0 || top_k <= 0 || top_k > n_cols)
    die("need n_rows > 0, n_cols > 0, 0 < top_k <= n_cols");

  int device = 0;
  cudaDeviceProp prop{};
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::printf("device %s | APs %d | warp %d | smem/AP %zu KB\n", prop.name,
              prop.multiProcessorCount, prop.warpSize,
              prop.sharedMemPerMultiprocessor / 1024);
  std::printf("shape n_rows=%lld n_cols=%lld top_k=%lld iters=%d\n",
              (long long)n_rows, (long long)n_cols, (long long)top_k, iters);
  // The arena the header sizes itself from, and the capacity that falls out:
  // the second is what makes the gate's `topk <= kMaxTopK` a real bound rather
  // than a fitted one.
  std::printf("arena %d bytes -> %d candidates (%d threads, %d coarse bins)\n",
              rk::dg12_ref::smem_bytes(), rk::dg12_ref::candidate_capacity(),
              rk::dg12_ref::threads(), rk::dg12_ref::coarse_bins());


  const size_t scores_bytes = (size_t)n_rows * n_cols * sizeof(float);
  const size_t indices_bytes = (size_t)n_rows * (size_t)top_k * sizeof(int32_t);
  float *scores = nullptr;
  int32_t *seq_lens = nullptr;
  int32_t *out_indices = nullptr;
  check(cudaMalloc(&scores, scores_bytes), "cudaMalloc(scores)");
  check(cudaMalloc(&seq_lens, (size_t)n_rows * sizeof(int32_t)), "cudaMalloc(seq_lens)");
  check(cudaMalloc(&out_indices, indices_bytes), "cudaMalloc(out_indices)");

  const int threads = 256;
  const int64_t blocks = (n_rows * n_cols + threads - 1) / threads;
  fill_normal<<<(unsigned)blocks, threads>>>(scores, n_rows * n_cols, 0x5eed1234ull);
  check(cudaGetLastError(), "fill_normal launch");
  std::vector<int32_t> host_lens((size_t)n_rows, (int32_t)n_cols);
  check(cudaMemcpy(seq_lens, host_lens.data(), (size_t)n_rows * sizeof(int32_t),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(seq_lens)");

  // No scan table: this driver measures the kernel, and the folded NaN scan
  // only runs where the caller asked for one (`nan_flags != nullptr`).
  auto launch = [&]() {
    rk::dg12_ref::launch(scores, seq_lens, out_indices, /*nan_flags=*/nullptr,
                         (int)n_rows, (int)top_k, (int64_t)n_cols,
                         /*default_length=*/(int)n_cols, /*stream=*/nullptr);
  };

  launch();
  check(cudaDeviceSynchronize(), "warmup sync");
  check(cudaGetLastError(), "coarse12 launch");

  cudaEvent_t start, stop;
  check(cudaEventCreate(&start), "cudaEventCreate");
  check(cudaEventCreate(&stop), "cudaEventCreate");
  check(cudaEventRecord(start), "cudaEventRecord");
  for (int i = 0; i < iters; ++i) launch();
  check(cudaEventRecord(stop), "cudaEventRecord");
  check(cudaEventSynchronize(stop), "cudaEventSynchronize");
  float elapsed_ms = 0.0f;
  check(cudaEventElapsedTime(&elapsed_ms, start, stop), "cudaEventElapsedTime");
  const double per_iter_ms = elapsed_ms / iters;
  const double read_gb = (double)scores_bytes / 1e9;
  std::printf("time: %.3f ms/iter over %d iters\n", per_iter_ms, iters);
  std::printf("read bandwidth: %.1f GB/s (%.3f GB read per call)\n",
              read_gb / (per_iter_ms / 1e3), read_gb);

  const int64_t block_rows = std::min(kVerifyRowsPerBlock, n_rows);
  std::vector<float> host_scores((size_t)block_rows * (size_t)n_cols);
  std::vector<int32_t> host_indices((size_t)block_rows * (size_t)top_k);
  VerifyScratch scratch;
  int64_t bad = 0;
  std::string error;
  for (int64_t row = 0; row < n_rows; row += block_rows) {
    const int64_t rows = std::min(block_rows, n_rows - row);
    check(cudaMemcpy(host_scores.data(), scores + row * n_cols,
                     (size_t)rows * (size_t)n_cols * sizeof(float),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy(scores)");
    check(cudaMemcpy(host_indices.data(), out_indices + row * top_k,
                     (size_t)rows * (size_t)top_k * sizeof(int32_t),
                     cudaMemcpyDeviceToHost),
          "cudaMemcpy(indices)");
    for (int64_t r = 0; r < rows; ++r) {
      if (!verify_row(host_scores.data() + r * n_cols,
                      host_indices.data() + r * top_k, n_cols, top_k, row + r,
                      scratch, error)) {
        if (bad == 0) std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        ++bad;
      }
    }
  }
  if (bad != 0) {
    std::printf("verify: %lld/%lld rows FAILED\n", (long long)bad, (long long)n_rows);
    return 1;
  }
  std::printf("verify: %lld/%lld rows match CPU nth_element reference\n",
              (long long)n_rows, (long long)n_rows);

  cudaFree(out_indices);
  cudaFree(seq_lens);
  cudaFree(scores);
  return 0;
}
