// Runnable driver for the extracted deep_gemm fp32 indexer TopK selector.
//
//   ./main [n_rows] [n_cols] [top_k] [iters]
//
// Allocates a (n_rows, n_cols) randn fp32 score matrix on the device, fills
// seq_lens with n_cols (scan every column), runs `deep_gemm_topk_selector`,
// copies the result back in row blocks, checks it against a CPU
// `std::nth_element` reference, and reports latency and read bandwidth.
//
// Build (whole program, from TOOLCHAIN.md):
//   cucc -O2 -std=c++20 --offload-arch=xcore1000 \
//        main.cu xcore1000_fp32_topk.cu -o main

#include "xcore1000_fp32_topk.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
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

// Deterministic pseudo-normal deviates, so every run verifies the same matrix.
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
  // Box-Muller over two uniforms in (0, 1].
  const double u1 = static_cast<double>(next_bits() >> 11) * (1.0 / 9007199254740992.0);
  const double u2 = static_cast<double>(next_bits() >> 11) * (1.0 / 9007199254740992.0);
  const double radius = std::sqrt(-2.0 * std::log(u1 == 0.0 ? 1e-300 : u1));
  data[i] = static_cast<float>(radius * std::cos(6.283185307179586 * u2));
}

const char *policy_name(int32_t policy)
{
  switch (policy) {
    case 1: return "Chunks";
    case 2: return "Single";
    case 3: return "Coarse12";
  }
  return "Auto(unresolved)";
}

constexpr int64_t kVerifyRowsPerBlock = 128;

// Reusable scratch for the reference, so the verification loop allocates once.
struct VerifyScratch {
  std::vector<float> row;       // the whole row, reduced in place by nth_element
  std::vector<float> expected;  // the reference top-k values, descending
  std::vector<float> actual;    // the kernel's top-k values, descending
  std::vector<int32_t> seen;    // the kernel's indices, sorted
};

// Checks one row against the CPU reference.
//
// Only the first `col_eff = seq_lens[row]` columns are live and the selector
// scans exactly that prefix (`get_row_begin_length` with `begin == 0`).  When
// the prefix is shorter than `top_k` -- both the `length <= top_k` arm of every
// kernel and any `seq_len < top_k` shape -- the tail of the row is padded with
// -1 indices and -inf values, and the reference does the same.
//
// The comparison is a *multiset* one, and deliberately so: output order is not
// part of this kernel's contract, and neither is the index choice at the top-k
// boundary.  The three internal radix passes place a boundary bin's members in
// whatever order they happen to be appended in, and the official checker
// (deep_gemm/kernels/indexer/dsa/ref.py:741-745, `check_ref_indexer_topk_selector`)
// says as much: "Comparison is value-based (sorted per row) because top-k is
// not unique when duplicate values exist: indices may differ but the selected
// value multiset must match the reference."
bool verify_row(
    const float *row_scores, const int32_t *row_indices, int64_t n_cols,
    int64_t col_eff, int64_t top_k, int64_t row, VerifyScratch &scratch,
    std::string &error)
{
  scratch.seen.assign(row_indices, row_indices + top_k);
  std::sort(scratch.seen.begin(), scratch.seen.end());
  for (int64_t i = 0; i < top_k; ++i) {
    const int32_t index = scratch.seen[(size_t)i];
    if (index == -1) continue;  // padding, only legal past the live prefix
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
  if (col_eff <= top_k) {
    std::sort(scratch.row.begin(), scratch.row.end(), std::greater<float>());
    for (int64_t i = 0; i < top_k; ++i)
      scratch.expected[(size_t)i] =
          i < col_eff ? scratch.row[(size_t)i]
                      : -std::numeric_limits<float>::infinity();
  }
  else {
    std::nth_element(
        scratch.row.begin(), scratch.row.begin() + (col_eff - top_k),
        scratch.row.end());
    std::copy(
        scratch.row.begin() + (col_eff - top_k), scratch.row.end(),
        scratch.expected.begin());
    std::sort(scratch.expected.begin(), scratch.expected.end(), std::greater<float>());
  }
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

// Writes the exact bytes the kernel selected from, plus the kernel's answer, so
// a cross-checker can load this driver's input into its own process and compare
// against a different TopK implementation without having to reproduce the
// generator.  Returns false (with `error` set) on any I/O failure.
bool dump_selection(
    const char *prefix, const float *scores, const int32_t *out_indices,
    int64_t n_rows, int64_t n_cols, int64_t top_k, std::string &error)
{
  const std::string scores_path = std::string(prefix) + ".scores.f32";
  const std::string idx_path = std::string(prefix) + ".idx.i32";
  std::FILE *scores_file = std::fopen(scores_path.c_str(), "wb");
  if (scores_file == nullptr) {
    error = "cannot open " + scores_path;
    return false;
  }
  std::FILE *idx_file = std::fopen(idx_path.c_str(), "wb");
  if (idx_file == nullptr) {
    std::fclose(scores_file);
    error = "cannot open " + idx_path;
    return false;
  }

  // Streamed in blocks so a large score matrix does not need a host mirror.
  const int64_t block_rows = std::min((int64_t)256, n_rows);
  std::vector<float> host_scores((size_t)block_rows * n_cols);
  std::vector<int32_t> host_indices((size_t)block_rows * top_k);
  for (int64_t row = 0; row < n_rows; row += block_rows) {
    const int64_t rows = std::min(block_rows, n_rows - row);
    if (cudaMemcpy(host_scores.data(), scores + row * n_cols,
                   (size_t)rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost)
            != cudaSuccess
        || cudaMemcpy(host_indices.data(), out_indices + row * top_k,
                      (size_t)rows * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost)
               != cudaSuccess) {
      std::fclose(scores_file);
      std::fclose(idx_file);
      error = "cudaMemcpy failed while dumping";
      return false;
    }
    if (std::fwrite(host_scores.data(), sizeof(float), (size_t)rows * n_cols,
                    scores_file)
            != (size_t)rows * n_cols
        || std::fwrite(host_indices.data(), sizeof(int32_t), (size_t)rows * top_k,
                       idx_file)
               != (size_t)rows * top_k) {
      std::fclose(scores_file);
      std::fclose(idx_file);
      error = "short write";
      return false;
    }
  }
  std::fclose(scores_file);
  std::fclose(idx_file);
  std::printf(
      "dump: %s.scores.f32 (%lld x %lld fp32) | %s.idx.i32 (%lld x %lld int32)\n",
      prefix, (long long)n_rows, (long long)n_cols, prefix, (long long)n_rows,
      (long long)top_k);
  return true;
}

int main(int argc, char **argv)
{
  // `--dump-prefix P` writes P.scores.f32 + P.idx.i32 and does not run the
  // normal verify/timing path.  Everything else is positional.
  const char *dump_prefix = nullptr;
  int argi = 1;
  if (argi + 1 < argc && std::strcmp(argv[argi], "--dump-prefix") == 0) {
    dump_prefix = argv[argi + 1];
    argi += 2;
  }
  const int64_t n_rows = argi < argc ? std::atoll(argv[argi]) : 1024;
  const int64_t n_cols = argi + 1 < argc ? std::atoll(argv[argi + 1]) : 4096;
  const int64_t top_k = argi + 2 < argc ? std::atoll(argv[argi + 2]) : 128;
  const int iters = argi + 3 < argc ? std::atoi(argv[argi + 3]) : 20;
  const int64_t seq_len = argi + 4 < argc ? std::atoll(argv[argi + 4]) : 0;  // 0 -> n_cols
  if (n_rows <= 0 || n_cols <= 0 || top_k <= 0 || top_k > n_cols)
    die("need n_rows > 0, n_cols > 0, 0 < top_k <= n_cols");
  if (seq_len < 0 || seq_len > n_cols)
    die("need 0 <= seq_len <= n_cols");

  int device = 0;
  cudaDeviceProp prop{};
  check(cudaGetDevice(&device), "cudaGetDevice");
  check(cudaGetDeviceProperties(&prop, device), "cudaGetDeviceProperties");
  std::printf(
      "device %s | APs %d | warp %d | regs/AP %d | smem/AP %zu KB | L2 %d MB\n",
      prop.name, prop.multiProcessorCount, prop.warpSize, prop.regsPerMultiprocessor,
      prop.sharedMemPerMultiprocessor / 1024, prop.l2CacheSize / (1024 * 1024));
  std::printf("shape n_rows=%lld n_cols=%lld top_k=%lld seq_len=%lld iters=%d\n",
              (long long)n_rows, (long long)n_cols, (long long)top_k,
              (long long)(seq_len > 0 ? seq_len : n_cols), iters);

  // Workspace the chunks policy needs.  The chunk count (3..6) is chosen inside
  // the launch from the device AP count, so size for the largest.
  const int32_t chunk_count = deep_gemm::indexer::deep_gemm_topk_chunk_count(
      n_rows, n_cols, prop.multiProcessorCount);
  int64_t chunks_workspace_bytes = 0;
  for (int chunks = 3; chunks <= 6; ++chunks)
    chunks_workspace_bytes = std::max(
        chunks_workspace_bytes,
        static_cast<int64_t>(
            deep_gemm::indexer::deep_gemm_topk_chunks_workspace_bytes(n_rows, chunks)));
  std::printf(
      "chunks chunk_count=%d (APs=%d) | workspace bytes (max over 3..6): %lld\n",
      chunk_count, prop.multiProcessorCount, (long long)chunks_workspace_bytes);

  const int32_t policy = deep_gemm::indexer::deep_gemm_topk_policy(
      n_rows, n_cols, (int32_t)top_k, prop.multiProcessorCount);
  std::printf("selected policy: %s (%d)\n", policy_name(policy), policy);

  const size_t scores_bytes = (size_t)n_rows * n_cols * sizeof(float);
  const size_t indices_bytes = (size_t)n_rows * top_k * sizeof(int32_t);
  const size_t values_bytes = (size_t)n_rows * top_k * sizeof(float);

  float *scores = nullptr;
  int32_t *seq_lens = nullptr;
  int32_t *out_indices = nullptr;
  float *out_values = nullptr;
  int32_t *chunks_workspace = nullptr;
  check(cudaMalloc(&scores, scores_bytes), "cudaMalloc(scores)");
  check(cudaMalloc(&seq_lens, (size_t)n_rows * sizeof(int32_t)), "cudaMalloc(seq_lens)");
  check(cudaMalloc(&out_indices, indices_bytes), "cudaMalloc(out_indices)");
  check(cudaMalloc(&out_values, values_bytes), "cudaMalloc(out_values)");
  if (chunks_workspace_bytes > 0)
    check(cudaMalloc(&chunks_workspace, (size_t)chunks_workspace_bytes),
          "cudaMalloc(chunks_workspace)");

  const int threads = 256;
  const int64_t blocks = (n_rows * n_cols + threads - 1) / threads;
  fill_normal<<<(unsigned)blocks, threads>>>(scores, n_rows * n_cols, 0x5eed1234ull);
  check(cudaGetLastError(), "fill_normal launch");
  // seq_lens drives how much of each row is scanned; `n_cols` means "all of it".
  // A `seq_len` below `top_k` exercises the `length <= top_k` padding arm.
  std::vector<int32_t> host_lens(
      (size_t)n_rows, seq_len > 0 ? seq_len : (int32_t)n_cols);
  check(cudaMemcpy(seq_lens, host_lens.data(), (size_t)n_rows * sizeof(int32_t),
                   cudaMemcpyHostToDevice),
        "cudaMemcpy(seq_lens)");

  auto launch = [&]() {
    deep_gemm::indexer::deep_gemm_topk_selector(
        scores, seq_lens, out_indices, out_values, chunks_workspace, n_rows, n_cols,
        (int32_t)top_k, /*stream=*/nullptr);
  };

  launch();
  check(cudaDeviceSynchronize(), "warmup sync");
  check(cudaGetLastError(), "selector launch");

  if (dump_prefix != nullptr) {
    std::string dump_error;
    if (!dump_selection(
            dump_prefix, scores, out_indices, n_rows, n_cols, top_k, dump_error))
      die(dump_error.c_str());
    cudaFree(chunks_workspace);
    cudaFree(out_values);
    cudaFree(out_indices);
    cudaFree(seq_lens);
    cudaFree(scores);
    return 0;
  }

  cudaEvent_t start;
  cudaEvent_t stop;
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

  // Verify every row, streaming the matrix back in blocks so the host does not
  // need the whole (n_rows, n_cols) matrix resident.
  std::string error;
  VerifyScratch scratch;
  const int64_t block_rows = std::min(kVerifyRowsPerBlock, n_rows);
  std::vector<float> host_scores((size_t)block_rows * n_cols);
  std::vector<int32_t> host_indices((size_t)block_rows * top_k);
  for (int64_t row = 0; row < n_rows; row += block_rows) {
    const int64_t rows = std::min(block_rows, n_rows - row);
    check(cudaMemcpy(host_scores.data(), scores + row * n_cols,
                     (size_t)rows * n_cols * sizeof(float), cudaMemcpyDeviceToHost),
          "cudaMemcpy(scores block)");
    check(cudaMemcpy(host_indices.data(), out_indices + row * top_k,
                     (size_t)rows * top_k * sizeof(int32_t), cudaMemcpyDeviceToHost),
          "cudaMemcpy(indices block)");
    for (int64_t r = 0; r < rows; ++r) {
      if (!verify_row(host_scores.data() + r * n_cols, host_indices.data() + r * top_k,
                      n_cols, host_lens[(size_t)(row + r)], top_k, row + r, scratch,
                      error)) {
        std::fprintf(stderr, "FAIL: %s\n", error.c_str());
        return 1;
      }
    }
  }
  std::printf("verify: %lld/%lld rows match CPU nth_element reference\n",
              (long long)n_rows, (long long)n_rows);

  cudaFree(chunks_workspace);
  cudaFree(out_values);
  cudaFree(out_indices);
  cudaFree(seq_lens);
  cudaFree(scores);
  return 0;
}
