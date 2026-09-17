// Runnable driver for the torch-free mcoplib SGLang TopK extraction.
//
// Build (TOOLCHAIN.md whole-program line):
//   cucc -O2 -std=c++20 --offload-arch=xcore1000 -I. main.cu \
//        build/xcore1000_topk_v1.o -o build/main && ./build/main
//
// Exercises every dispatch arm of mcoplib_topk_transform:
//
//   naive        seq_len <= TopK (512)          -> pad/transform only
//   histogram    max(seq_len) <= 16384          -> single-pass 4096-bin
//   radix        max(seq_len) > 16384           -> two-pass radix-256
//   mixed        one long row forces every row  -> documents the per-launch
//                                                  (not per-row) dispatch
//
// Every row is checked against a CPU std::nth_element reference. Timing is
// best-of-R interleaved rounds, with the buffer re-staged before each timed
// round (the TLB cost of touching 256 MB of fresh device pages is real and must
// not be hidden by reuse). The C500 clock is not locked (persistence off), so a
// single measurement drifts by up to 2x between runs; the min over interleaved
// rounds is the only stable figure.

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>

#include "xcore1000_topk_v1.h"

#define CUDA_CHECK(expr)                                                        \
  do {                                                                          \
    const cudaError_t _e = (expr);                                              \
    if (_e != cudaSuccess) {                                                    \
      std::fprintf(stderr, "CUDA error %s at %s:%d: %s\n", cudaGetErrorName(_e),\
                   __FILE__, __LINE__, cudaGetErrorString(_e));                 \
      std::exit(1);                                                             \
    }                                                                           \
  } while (0)

namespace {

constexpr int kTopK = 512;
// topk_v1_hist4096::kMaxLen (topk_v1_histogram_4096.cuh:34), restated here
// because main.cu must not pull in the kernel header.
constexpr int kHist4096MaxLen = 16384;
// The CPU reference runs std::nth_element over every checked row, so the shape
// grid is bounded by how many rows can be checked, not by how many can be run.
// A reduced check is reported on stdout rather than passed off as a full one.
constexpr int kMaxCheckRows = 512;
// 4 interleaved rounds is enough to pick a stable minimum: the C500 clock is not
// locked (persistence off), but the running minimum converges after a couple of
// rounds and extra rounds only add wall time.
constexpr int kRounds = 4;

// Narrow-range uniform scores. The range is chosen so that the FP16 coarse
// histograms both kernels build stay well under their capacity: the selection
// resolves in exact FP32, but a coarse bin holding more than TopK(512) values
// on the histogram path, or more than the 4096-entry staging buffer on the
// radix path, would hit those kernels' documented limits rather than a
// selection bug. max_coarse_bin_occupancy() prints the actual occupancy so the
// margin is visible.
void fill_scores(std::vector<float>& h, uint32_t seed) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
  for (auto& v : h) v = dist(rng);
}

// 8-bit sign-magnitude key over an fp16 round-to-nearest-even of x, derived on
// the host to match convert_to_uint8 (xcore1000_topk_v1.cu:80-85), which calls
// __float2half_rn. Rounding matters here: the mantissa is rounded, not
// truncated, so this must reproduce RNE including the tie-to-even case and the
// carry out of the mantissa into the exponent. A truncating version shifts
// values across 8-bit bin boundaries and reports band occupancies that are a
// few percent off -- enough to move the answer at the crossover length.
uint32_t fp16_key8(float x) {
  const uint32_t f = *reinterpret_cast<const uint32_t*>(&x);
  const uint32_t sign = f >> 31;
  const uint32_t exp = (f >> 23) & 0xFF;
  const uint32_t mant = f & 0x7FFFFF;
  uint16_t h;
  if (exp == 0xFF) {
    h = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
  } else {
    int32_t newexp = static_cast<int32_t>(exp) - 127 + 15;
    if (newexp >= 31) {
      h = static_cast<uint16_t>((sign << 15) | 0x7C00);  // overflow -> inf
    } else if (newexp <= 0) {
      h = static_cast<uint16_t>(sign << 15);  // underflow -> 0 (range is +/-1)
    } else {
      uint32_t hi = mant >> 13;        // kept mantissa bits
      const uint32_t low = mant & 0x1FFF;  // discarded bits
      // round-to-nearest, ties to even
      if (low > 0x1000u || (low == 0x1000u && (hi & 1u))) {
        ++hi;
        if (hi == 0x400u) {  // carry out of the mantissa
          hi = 0;
          if (++newexp >= 31) {
            h = static_cast<uint16_t>((sign << 15) | 0x7C00);
            goto rounded;
          }
        }
      }
      h = static_cast<uint16_t>((sign << 15) | (static_cast<uint32_t>(newexp) << 10) | hi);
    }
  }
rounded:;
  const uint16_t key =
      (h & 0x8000) ? static_cast<uint16_t>(~h) : static_cast<uint16_t>(h | 0x8000);
  return key >> 8;
}

// Occupancy of the threshold's own 8-bit FP16 coarse bin, over the worst row:
// the count of elements sharing the 8-bit key of the row's 512th-largest value.
// This is the number the radix path's staging buffer has to hold, because round
// 0 appends every element whose 8-bit coarse bin equals the threshold bin
// (xcore1000_topk_v1.cu, radix_topk pass 1 -> step<>) and drops the ones past
// kStagingSize. For uniform(-1,1) scores it is length / 2^k, which crosses the
// 4096-entry cap in the low 60000s. Diagnostic only.
uint32_t threshold_bin_occupancy(const float* row, int len) {
  std::vector<uint32_t> idx(len);
  for (int i = 0; i < len; ++i) idx[i] = static_cast<uint32_t>(i);
  std::nth_element(idx.begin(), idx.begin() + (kTopK - 1), idx.end(),
                   [row](uint32_t a, uint32_t b) { return row[a] > row[b]; });
  const uint32_t thr = fp16_key8(row[idx[kTopK - 1]]);
  uint32_t n = 0;
  for (int i = 0; i < len; ++i) {
    if (fp16_key8(row[i]) == thr) ++n;
  }
  return n;
}

// Max occupancy of the 12-bit FP16 coarse bin this source keys on
// (topk_v1_histogram_4096.cuh:53-60), over the valid prefix of every row.
// Diagnostic only -- it is how the repro shows its tie-free margin.
uint32_t max_coarse_bin_occupancy(const float* row, int len) {
  constexpr int kBits = 12;
  std::vector<uint32_t> hist(size_t{1} << kBits, 0u);
  for (int i = 0; i < len; ++i) {
    const uint32_t f = *reinterpret_cast<const uint32_t*>(row + i);
    const uint32_t sign = f >> 31;
    const uint32_t exp = (f >> 23) & 0xFF;
    const uint32_t mant = f & 0x7FFFFF;
    uint16_t h;
    if (exp == 0xFF) {
      h = static_cast<uint16_t>((sign << 15) | 0x7C00 | (mant ? 0x200 : 0));
    } else {
      const int32_t newexp = static_cast<int32_t>(exp) - 127 + 15;
      if (newexp >= 31) {
        h = static_cast<uint16_t>((sign << 15) | 0x7C00);  // overflow -> inf
      } else if (newexp <= 0) {
        h = static_cast<uint16_t>(sign << 15);  // underflow -> 0 (range is +/-1)
      } else {
        h = static_cast<uint16_t>((sign << 15) | (newexp << 10) | (mant >> 13));
      }
    }
    const uint16_t key =
        (h & 0x8000) ? static_cast<uint16_t>(~h) : static_cast<uint16_t>(h | 0x8000);
    hist[key >> (16 - kBits)]++;
  }
  return *std::max_element(hist.begin(), hist.end());
}

struct Case {
  const char* name;
  int n_rows;
  int n_cols;
  std::vector<int32_t> seq_lens;
  int iters;
  bool expect_naive;
  // Set for shapes this source is known to get wrong. Such a case prints KNOWN
  // instead of FAIL and does not set the exit code -- it is evidence for a
  // documented limitation, not a regression in the extraction. See README
  // "Known large-row defect".
  bool known_defect = false;

  // host + device state
  std::vector<float> h_scores;
  float* d_scores = nullptr;
  int32_t* d_seq = nullptr;
  int32_t* d_out = nullptr;
  int32_t* d_page = nullptr;

  double best_ms = 0.0;
  double gbps = 0.0;
  bool ok = false;
  std::string msg = "ok";
  uint32_t coarse_bin = 0;
  // Worst-row occupancy of the threshold's own 8-bit bin, i.e. how much the
  // radix staging buffer has to hold. Only meaningful for the radix path.
  uint32_t thr_band = 0;
};

// Identity page table: page_to_indices(p, i, 0) == p[i], so a table of
// [0, 1, 2, ...] makes the fused transform the identity and out_indices
// directly comparable to a plain selector.
std::vector<int32_t> identity_page_table(int n_cols) {
  std::vector<int32_t> page(n_cols);
  for (int i = 0; i < n_cols; ++i) page[i] = i;
  return page;
}

void prepare(Case& c) {
  const size_t n_el = static_cast<size_t>(c.n_rows) * c.n_cols;
  c.h_scores.resize(n_el);
  fill_scores(c.h_scores, 0xC0FFEEu + static_cast<uint32_t>(c.n_rows * 31 + c.n_cols));

  CUDA_CHECK(cudaMalloc(&c.d_scores, n_el * sizeof(float)));
  CUDA_CHECK(cudaMalloc(&c.d_seq, c.n_rows * sizeof(int32_t)));
  CUDA_CHECK(cudaMalloc(&c.d_out, static_cast<size_t>(c.n_rows) * kTopK * sizeof(int32_t)));
  // Worst coarse bin over the batch: this is the number the histogram path
  // compares against TopK, and the radix path against its 4096-entry staging
  // buffer, so printing it makes the margin against those limits explicit.
  c.coarse_bin = 0;
  c.thr_band = 0;
  for (int r = 0; r < c.n_rows; ++r) {
    const float* row = c.h_scores.data() + static_cast<size_t>(r) * c.n_cols;
    const uint32_t b = max_coarse_bin_occupancy(row, c.seq_lens[r]);
    if (b > c.coarse_bin) c.coarse_bin = b;
    // The band only matters where the radix path runs and the row is long
    // enough to select from; below TopK the naive arm returns early.
    if (c.seq_lens[r] > kTopK) {
      const uint32_t t = threshold_bin_occupancy(row, c.seq_lens[r]);
      if (t > c.thr_band) c.thr_band = t;
    }
  }
}

void release(Case& c) {
  CUDA_CHECK(cudaFree(c.d_scores));
  CUDA_CHECK(cudaFree(c.d_seq));
  CUDA_CHECK(cudaFree(c.d_out));
  CUDA_CHECK(cudaFree(c.d_page));
  c.d_scores = nullptr;
  c.d_seq = nullptr;
  c.d_out = nullptr;
  c.d_page = nullptr;
  c.h_scores.clear();
  c.h_scores.shrink_to_fit();
}

void stage(Case& c) {
  const size_t n_el = static_cast<size_t>(c.n_rows) * c.n_cols;
  CUDA_CHECK(cudaMemcpy(c.d_scores, c.h_scores.data(), n_el * sizeof(float),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemcpy(c.d_seq, c.seq_lens.data(), c.n_rows * sizeof(int32_t),
                        cudaMemcpyHostToDevice));
  CUDA_CHECK(cudaMemset(c.d_out, 0xFF, static_cast<size_t>(c.n_rows) * kTopK * sizeof(int32_t)));
  if (c.d_page == nullptr) {
    const auto page = identity_page_table(c.n_cols);
    CUDA_CHECK(cudaMalloc(&c.d_page, static_cast<size_t>(c.n_cols) * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(c.d_page, page.data(), page.size() * sizeof(int32_t),
                          cudaMemcpyHostToDevice));
  }
}

void dispatch(const Case& c) {
  mcoplib_topk_transform(c.d_scores, c.d_seq, c.d_out, c.d_page, c.n_rows, c.n_cols, kTopK,
                         /*page_bits=*/0);
}

inline const float* row_ptr(const Case& c, int r) {
  return c.h_scores.data() + static_cast<size_t>(r) * c.n_cols;
}

// One row of a selection arm, checked against the CPU nth_element reference.
//
// The scores are uniform-random fp32, so exact fp32 equality at the kth value
// is a 1-in-2^24 event per pair and the TopK index set is normally unique --
// no "ties are broken arbitrarily" slack is granted. If a row does happen to
// have a repeated kth value, the set is genuinely ambiguous and the mismatch is
// reported with the tie count so it is not mistaken for a kernel bug.
bool verify_row(const float* row, int seq_len, const int32_t* got,
                const std::vector<int32_t>& ref_first, const std::vector<int32_t>& order,
                std::string* msg) {
  *msg = "ok";
  std::vector<int32_t> sel(got, got + kTopK);
  std::vector<int32_t> got_sorted = sel;
  std::sort(got_sorted.begin(), got_sorted.end());
  if (std::adjacent_find(got_sorted.begin(), got_sorted.end()) != got_sorted.end()) {
    *msg = "duplicate indices";
    return false;
  }
  for (int32_t idx : got_sorted) {
    if (idx < 0 || idx >= seq_len) {
      *msg = "index out of range";
      return false;
    }
  }
  std::sort(sel.begin(), sel.end());
  if (sel == ref_first) {
    *msg = "ok";
    return true;
  }

  // The exact sets differ. Accept it only if the difference is genuine
  // tie-break latitude at the kth value; otherwise it is a real bug.
  const float kth = row[order[kTopK - 1]];
  int64_t row_gt = 0, ties = 0;
  for (int i = 0; i < seq_len; ++i) {
    if (row[i] > kth) ++row_gt;
    else if (row[i] == kth) ++ties;
  }
  int64_t got_gt = 0, got_eq = 0;
  for (int32_t idx : sel) {
    if (row[idx] > kth) ++got_gt;
    else if (row[idx] == kth) ++got_eq;
    else {
      *msg = "selected index below the kth value";
      return false;
    }
  }
  if (got_gt != row_gt) {
    char buf[200];
    std::snprintf(buf, sizeof(buf), "selected %lld above kth, reference has %lld",
                  static_cast<long long>(got_gt), static_cast<long long>(row_gt));
    *msg = buf;
    return false;
  }
  if (got_gt + got_eq != kTopK) {
    *msg = "selection does not reach TopK at the kth value";
    return false;
  }
  if (ties <= 1) {
    *msg = "selected set differs from CPU nth_element with a unique kth value";
    return false;
  }
  char buf[200];
  std::snprintf(buf, sizeof(buf), "%lld exact ties at kth (set is ambiguous); count check ok",
                static_cast<long long>(ties));
  *msg = buf;
  return true;
}

// The CPU reference over every row is what bounds the shape grid: 4096 rows x
// 524288 columns is 2.1e9 elements of std::nth_element. `max_rows` caps the
// checked prefix; the caller reports the cap so a reduced check never reads as
// a full one.
void verify(Case& c, int max_rows) {
  CUDA_CHECK(cudaDeviceSynchronize());
  std::vector<int32_t> h_out(static_cast<size_t>(c.n_rows) * kTopK);
  CUDA_CHECK(cudaMemcpy(h_out.data(), c.d_out, h_out.size() * sizeof(int32_t),
                        cudaMemcpyDeviceToHost));

  const int n_check = std::min(c.n_rows, max_rows);
  std::string tie_note;
  for (int r = 0; r < n_check; ++r) {
    const float* row = row_ptr(c, r);
    const int32_t* got = h_out.data() + static_cast<size_t>(r) * kTopK;
    const int32_t sl = c.seq_lens[r];
    if (c.expect_naive || sl <= kTopK) {
      // Per-row naive arm: out[i] = page_to_indices(page, i, bits) for i < len,
      // -1 beyond (topk_v1.cu:98-115). With the identity table and page_bits
      // == 0 that is out[i] == i.
      for (int i = 0; i < kTopK; ++i) {
        const int32_t want = (i < sl) ? i : -1;
        if (got[i] != want) {
          char buf[160];
          std::snprintf(buf, sizeof(buf), "row %d col %d: naive got %d want %d", r, i, got[i],
                        want);
          c.msg = buf;
          c.ok = false;
          return;
        }
      }
      continue;
    }

    // CPU reference: the TopK largest columns of the row's valid prefix.
    std::vector<int32_t> order(sl);
    for (int i = 0; i < sl; ++i) order[i] = i;
    std::nth_element(order.begin(), order.begin() + (kTopK - 1), order.end(),
                     [row](int32_t a, int32_t b) { return row[a] > row[b]; });
    std::vector<int32_t> ref_first(order.begin(), order.begin() + kTopK);
    std::sort(ref_first.begin(), ref_first.end());

    std::string row_msg;
    if (!verify_row(row, sl, got, ref_first, order, &row_msg)) {
      char buf[200];
      std::snprintf(buf, sizeof(buf), "row %d (seq_len %d): %s", r, sl, row_msg.c_str());
      c.msg = buf;
      c.ok = false;
      return;
    }
    // All rows matched the reference exactly except possibly at a repeated kth
    // value; keep the last such note so the tie path is visible in the output.
    if (row_msg != "ok") tie_note = row_msg;
  }
  c.ok = true;
  c.msg = tie_note.empty() ? "ok" : tie_note;
}

double time_once(const Case& c, cudaEvent_t ev0, cudaEvent_t ev1) {
  for (int it = 0; it < c.iters; ++it) dispatch(c);
  CUDA_CHECK(cudaDeviceSynchronize());
  CUDA_CHECK(cudaEventRecord(ev0));
  for (int it = 0; it < c.iters; ++it) dispatch(c);
  CUDA_CHECK(cudaEventRecord(ev1));
  CUDA_CHECK(cudaEventSynchronize(ev1));
  float ms = 0.0f;
  CUDA_CHECK(cudaEventElapsedTime(&ms, ev0, ev1));
  return ms / c.iters;
}

}  // namespace

// Build one case for a requested (n_rows, length, top_k).
//
// `length` is both the number of valid columns and the row stride, so
// n_cols == length and the scores tensor is dense. The reference is a CPU
// nth_element over the whole row, matching what the other reference drivers in
// this tree select over -- no truncated prefix window.
Case make_case(int n_rows, int length) {
  Case c{"shape-driven case", n_rows, length, std::vector<int32_t>(n_rows, length), 4, false};
  // Long rows are the radix staging-cap defect; mark them so grid mode can
  // attach the explanation to a FAIL without changing the verdict token.
  c.known_defect = (length > 32768);
  return c;
}

// Grid mode: one shape per invocation, one comparable number per run.
//
//   main <n_rows> <length> <top_k> [iters]
//
// Timing is the best of kRounds interleaved rounds. The reference is a CPU
// std::nth_element over every row, so only the first kMaxCheckRows rows are
// checked (the kernel still runs over all of them), and the cap is printed --
// a silently reduced row count would look like a normal run at a fraction of
// the work.
int run_grid(int n_rows, int length, int top_k_cfg, int iters) {
  if (top_k_cfg < 1 || top_k_cfg > 4096) {
    std::printf("top_k %d outside contract [1, 4096]\n", top_k_cfg);
    return 2;
  }
  if (length < 1 || n_rows < 1) {
    std::printf("n_rows and length must be positive\n");
    return 2;
  }

  Case c = make_case(n_rows, length);
  c.iters = iters;

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  int smem_optin = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

  prepare(c);
  stage(c);

  // Does the requested top_k actually reach the kernel? mcoplib_topk_transform
  // returns without launching unless top_k == 512 (the kernel's compile-time
  // TopK). Detect that here rather than reporting a number for a shape that was
  // never served.
  const bool launchable = (top_k_cfg == kTopK);
  if (launchable) {
    dispatch(c);
    verify(c, kMaxCheckRows);
  } else {
    c.ok = false;
    c.msg = "top_k != 512: mcoplib_topk_transform returns without launching";
  }

  c.best_ms = 0.0;
  if (launchable) {
    cudaEvent_t ev0, ev1;
    CUDA_CHECK(cudaEventCreate(&ev0));
    CUDA_CHECK(cudaEventCreate(&ev1));
    for (int round = 0; round < kRounds; ++round) {
      stage(c);
      const double ms = time_once(c, ev0, ev1);
      if (c.best_ms == 0.0 || ms < c.best_ms) c.best_ms = ms;
    }
    CUDA_CHECK(cudaEventDestroy(ev0));
    CUDA_CHECK(cudaEventDestroy(ev1));
  }

  const int32_t max_seq = *std::max_element(c.seq_lens.begin(), c.seq_lens.end());
  const double traffic =
      static_cast<double>(n_rows) * (static_cast<double>(length) + top_k_cfg) * 4.0;
  const double gbps = c.best_ms > 0.0 ? traffic / (c.best_ms * 1e-3) / 1e9 : 0.0;

  std::printf("shape           : rows=%d length=%d top_k=%d\n", n_rows, length, top_k_cfg);
  std::printf("dispatch        : %s\n",
              max_seq <= kTopK ? "naive (seq_len <= TopK)"
              : max_seq <= 16384 ? "histogram_4096 (single pass)"
                                 : "radix_256 (two pass)");
  std::printf("radix staging   : %s\n", smem_optin >= 96 * 1024
                                             ? "7168 entries (56KB, 128K-smem chip)"
                                             : "4096 entries (32KB, 64K-smem chip)");
  if (max_seq > kHist4096MaxLen) {
    std::printf("threshold band  : %u elements share the worst row's 512th-value 8-bit FP16\n"
                "                  bin; round 0 stages kStagingSize of them (4096 on a "
                "64K-smem chip).\n"
                "                  The band only decides the outcome when the 512th value "
                "is the top of that\n"
                "                  band -- i.e. fewer than 512 elements sit above it. Above "
                "~1M elements the\n"
                "                  band exceeds 4096 regardless of the draw, so this is a "
                "lower bound on the\n"
                "                  failure length, and the in-range (>4096) fraction is "
                "genuinely draw-dependent.\n", c.thr_band);
  }
  if (n_rows > kMaxCheckRows) {
    std::printf("verify          : capped at %d of %d rows (CPU reference cost)\n",
                kMaxCheckRows, n_rows);
  }
  if (c.best_ms > 0.0) {
    std::printf("time: %.4f ms/iter over %d iters\n", c.best_ms, c.iters);
    std::printf("time: %.6f ms  (best of %d rounds x %d iters, seq_len %d, driver-local)\n",
                c.best_ms, kRounds, c.iters, c.seq_lens.front());
    std::printf("traffic: %.1f MB   bandwidth: %.1f GB/s  (n_rows*(length+top_k)*4)\n",
                traffic / 1e6, gbps);
  }
  // Grid mode always reports PASS/FAIL, never KNOWN: the harness that drives it
  // (ref/crosscheck.py) classifies cells on those two tokens, and a shape this
  // source computes wrongly must read as a failure there, not as a blank.
  std::printf("verify: %s   %s\n", c.ok ? "PASS" : "FAIL", c.msg.c_str());
  if (!c.ok && c.known_defect) {
    std::printf("note: this FAIL is the source kernel's known large-row limit, not an "
                "extraction fault -- see README, 'Known large-row defect'.\n");
  }

  release(c);
  return c.ok ? 0 : 1;
}

int main(int argc, char** argv) {
  if (argc >= 4) {
    const int n_rows = std::atoi(argv[1]);
    const int length = std::atoi(argv[2]);
    const int top_k_cfg = std::atoi(argv[3]);
    const int iters = argc >= 5 ? std::atoi(argv[4]) : 30;
    if (n_rows <= 0 || length <= 0 || iters <= 0) {
      std::fprintf(stderr, "usage: %s <n_rows> <length> <top_k> [iters]\n", argv[0]);
      return 2;
    }
    return run_grid(n_rows, length, top_k_cfg, iters);
  }

  int dev = 0;
  CUDA_CHECK(cudaGetDevice(&dev));
  cudaDeviceProp prop{};
  CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));

  int sm_count = 0, warp_size = 0, smem_block = 0, smem_optin = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&warp_size, cudaDevAttrWarpSize, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_block, cudaDevAttrMaxSharedMemoryPerBlock, dev));
  CUDA_CHECK(cudaDeviceGetAttribute(&smem_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, dev));

  std::printf("device          : %s\n", prop.name);
  std::printf("APs             : %d\n", sm_count);
  std::printf("warp size       : %d\n", warp_size);
  std::printf("smem/block      : %d (optin %d)\n", smem_block, smem_optin);
  constexpr int kLargeSmemThreshold = 96 * 1024;
  std::printf("radix staging   : %s\n",
              smem_optin >= kLargeSmemThreshold ? "7168 entries (56KB, 128K-smem chip)"
                                                : "4096 entries (32KB, 64K-smem chip)");
  std::printf("TopK            : %d\n", kTopK);
  std::printf("histogram_4096 kMaxLen : 16384\n\n");

  // Cases are checked one at a time to keep host memory bounded: n_cols=65536
  // at 8 rows is already 2 MB per case, so holding every case's host + device
  // buffer at once would be pure waste for a repro.
  std::vector<Case> cases;
  cases.push_back({"naive (seq_len 300 <= TopK)", 4096, 512,
                   std::vector<int32_t>(4096, 300), 32, true});
  cases.push_back({"histogram_4096 (16384 = kMaxLen)", 4096, 16384,
                   std::vector<int32_t>(4096, 16384), 4, false});
  // seq_len 16381 with a 16381-wide row puts every row base at
  // r * 65524 bytes, which is not 16-byte aligned, so this hits the predicated
  // per-element load path in histogram_4096 (row_aligned == false,
  // topk_v1_histogram_4096.cuh:209-227).
  cases.push_back({"histogram_4096 (unaligned rows)", 4096, 16381,
                   std::vector<int32_t>(4096, 16381), 4, false});
  cases.push_back({"histogram_4096 (8192, half row)", 4096, 8192,
                   std::vector<int32_t>(4096, 8192), 4, false});
  cases.push_back({"radix_256 (16385, just over)", 1024, 16388,
                   std::vector<int32_t>(1024, 16385), 4, false});
  // Same, but with an unaligned row base, which is what the vec4_prefix
  // prologue in radix_topk exists for (topk_v1.cu:156-165).
  cases.push_back({"radix_256 (unaligned rows)", 1024, 16387,
                   std::vector<int32_t>(1024, 16387), 4, false});
  cases.push_back({"radix_256 (32768, two pass)", 512, 32768,
                   std::vector<int32_t>(512, 32768), 4, false});
  // Known defect: rows long enough that the 8-bit FP16 coarse bin holding the
  // threshold legitimately exceeds the 4096-entry staging buffer. The kernel
  // drops the overflow silently and returns the wrong TopK. Kept in the default
  // run because it is the one shape that makes the limit visible.
  cases.push_back({"radix_256 (131072, past staging)", 256, 131072,
                   std::vector<int32_t>(256, 131072), 4, false, true});
  // Two lengths where the 8-bit FP16 coarse bin holding the row's 512th value
  // exceeds the 4096-entry staging buffer no matter how the scores are drawn
  // (band ~ n/2^k, so the crossover stops depending on the random draw). At
  // n >= ~1M the defect needs no unlucky seed to reproduce.
  cases.push_back({"radix_256 (1048576, band 66k)", 2, 1048576,
                   std::vector<int32_t>(2, 1048576), 4, false, true});
  cases.push_back({"radix_256 (2097152, band 132k)", 1, 2097152,
                   std::vector<int32_t>(1, 2097152), 4, false, true});
  cases.push_back({"mixed batch (max>16384 -> radix)", 8, 65536,
                   std::vector<int32_t>{300, 17000, 32768, 40000, 512, 16384, 16385}, 8,
                   false});

  cudaEvent_t ev0, ev1;
  CUDA_CHECK(cudaEventCreate(&ev0));
  CUDA_CHECK(cudaEventCreate(&ev1));

  std::printf("%-36s %-26s %-5s %8s %10s  %s\n", "case", "shape", "ok", "GB/s", "time", "note");
  std::printf("%s\n", std::string(125, '-').c_str());
  bool all_pass = true;
  int n_known = 0;
  for (auto& c : cases) {
    prepare(c);
    stage(c);
    dispatch(c);
    verify(c, kMaxCheckRows);

    // Interleaved rounds; keep the minimum. The C500 clock is not locked, so a
    // case's absolute time drifts by up to 2x between back-to-back runs and
    // only a best-of is stable enough to quote. Re-staging between rounds keeps
    // first-touch cost from being hidden by warm buffer reuse.
    c.best_ms = 0.0;
    for (int round = 0; round < kRounds; ++round) {
      stage(c);
      const double ms = time_once(c, ev0, ev1);
      if (c.best_ms == 0.0 || ms < c.best_ms) c.best_ms = ms;
    }

    // Minimum traffic: one fp32 read of each row's valid prefix. The naive arm
    // reads no scores at all, and the radix path also re-reads threshold-bin
    // candidates, so these GB/s are a lower bound, not a measured DRAM rate.
    double bytes = 0.0;
    for (int r = 0; r < c.n_rows; ++r) bytes += static_cast<double>(c.seq_lens[r]) * sizeof(float);
    c.gbps = c.expect_naive ? 0.0 : bytes / (c.best_ms * 1e-3) / 1e9;

    int32_t max_seq = *std::max_element(c.seq_lens.begin(), c.seq_lens.end());
    char shape[64];
    std::snprintf(shape, sizeof(shape), "rows=%d n_cols=%d max=%d", c.n_rows, c.n_cols, max_seq);
    const char* verdict = c.ok ? "PASS" : (c.known_defect ? "KNOWN" : "FAIL");
    std::printf("%-36s %-26s %-5s %8.1f %8.1f us  %s\n", c.name, shape, verdict, c.gbps,
                c.best_ms * 1e3, c.msg.c_str());
    std::printf("    best of %d rounds x %d iters; max 12-bit coarse bin %u (TopK %d)", kRounds,
                c.iters, c.coarse_bin, kTopK);
    if (c.n_cols > kHist4096MaxLen) {
      std::printf("; threshold 8-bit band %u vs 4096-entry staging", c.thr_band);
    }
    std::printf("\n");
    if (!c.ok && c.known_defect) ++n_known;
    else all_pass = all_pass && c.ok;
    release(c);
  }
  std::printf("%s\n", std::string(125, '-').c_str());
  std::printf("GB/s is the minimum traffic (one fp32 read of each row prefix);\n");
  std::printf("the radix path moves more (it re-reads threshold-bin candidates).\n");
  if (n_known > 0) {
    std::printf("%d case marked KNOWN: it fails against the CPU reference by design of the\n"
                "source kernel, not by a fault in this extraction. See README, "
                "'Known large-row defect'.\n", n_known);
  }
  std::printf("result: %s\n", all_pass ? "ALL PASS" : "FAILURES PRESENT");
  return all_pass ? 0 : 1;
}
