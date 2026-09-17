// Runnable driver for the DeepSelect fp32 radix TopK reference.
//
//   * builds a device score matrix,
//   * runs the reference selector over it,
//   * checks the answer against a CPU `std::nth_element` reference,
//   * reports time and effective GB/s.
//
// Four cells are always run, because they take different arms of the
// dispatcher (see README.md for the predicates and their line numbers):
//
//   row       B=256 V=16384   k=2048   V < 32768, so `chunked_f32_applies`
//                                      is false: one CTA per row through
//                                      `topk_kernel_radix` -> the fp32 row.
//   split     B=256 V=524288  k=2048   V above the floor and B above
//                                      `kF32ChunksFewBatches`, so the count
//                                      comes from `f32_chunks_large_batch`
//                                      (6 here) and the row is split.
//   ragged    B=8   V=65536   k=512    per-row windows through the `lengths`
//                                      table, some of them shorter than `k`
//                                      so the split's own rows take the
//                                      `length <= topk` shortcut.
//   shortcut  B=2   V=1000    k=2048   `length <= topk` on the row path.
//
// Anything on the command line overrides the shape, so a cell can be re-run:
//   ./ds_topk_ref [n_rows] [n_cols] [top_k] [repeats]
//
// It deliberately does not print the routing decision: the reference's own
// predicates are host functions inside the kernel TU and are not exported.
// Read them in `xcore1000_maca_topk.cu` instead of guessing here -- a driver
// that re-derived them would be a second copy of the policy, which is the one
// thing a reference must not have.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <chrono>
#include <random>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "xcore1000_ds_topk.h"

#define CUDA_CHECK(expr)                                                     \
    do {                                                                     \
        const cudaError_t rc_ = (expr);                                      \
        if (rc_ != cudaSuccess) {                                            \
            std::fprintf(stderr, "%s:%d: %s -> %s\n", __FILE__, __LINE__,    \
                         #expr, cudaGetErrorString(rc_));                    \
            std::exit(1);                                                    \
        }                                                                    \
    } while (0)

namespace {

constexpr int kSmCount = 104;   // queried from the device below, not assumed

// Set by `--dump-prefix`, read by `run_cell`.  A file scope rather than a
// `Cell` field because the dump is a property of the invocation, not of a
// cell: one prefix, four sets of files, named `<prefix>.b<B>.v<V>.k<K>.*`.
std::string dump_prefix;

// Raw little-endian host bytes, no header -- whatever reads them knows the
// shape.  That is the point: the three refs have to agree on *data*, not on a
// container format, and a `.f32`/`.i32` pair is readable from numpy, torch and
// a shell one-liner alike.
template <typename T>
void write_dump(const std::string& path, const std::vector<T>& v) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr) {
        std::fprintf(stderr, "dump: cannot open %s\n", path.c_str());
        std::exit(1);
    }
    if (!v.empty() &&
        std::fwrite(v.data(), sizeof(T), v.size(), f) != v.size()) {
        std::fprintf(stderr, "dump: short write to %s\n", path.c_str());
        std::exit(1);
    }
    std::fclose(f);
}

// May throw `std::bad_alloc` for a shape whose *host* copy does not fit; see
// the note in `run_cell`.
struct Cell {
    int n_rows;
    int n_cols;
    int top_k;
    int repeats;
    int n_timed;
    // When set, a per-row window table is generated: every third row is
    // `top_k / 2` long (so it takes the `length <= topk` shortcut), odd rows
    // are the whole row, even rows the first third of it.
    bool ragged = false;
};

// The CPU reference.  The kernel answers "the `top_k` largest values of the
// row, ties broken arbitrarily", so the check is on the *selected set*: the
// k-th largest value is `kth`, every selected element must be >= `kth`, and
// the number selected strictly above it must equal the number the row has
// strictly above it.  Counting, rather than comparing sorted lists, is what
// makes the check robust to the ties the kernel is allowed to break either
// way.  Duplicate and out-of-range indices are checked too: the kernel writes
// exactly `top_k` distinct slots.
bool verify_row(const std::vector<float>& scores, const std::vector<int32_t>& out,
                const std::vector<int32_t>& lengths, int n_rows, int n_cols,
                int top_k) {
    std::vector<float> row;
    for (int r = 0; r < n_rows; ++r) {
        const float* src = scores.data() + (size_t)r * n_cols;
        const int32_t* o = out.data() + (size_t)r * top_k;
        const int length = lengths.empty() ? n_cols : lengths[r];

        // `-1` is `idx_fill`, written for every slot the row could not fill, so
        // it is expected to repeat; only the real columns have to be distinct.
        std::vector<int32_t> idx(o, o + top_k);
        std::sort(idx.begin(), idx.end());
        const auto first_real =
            std::lower_bound(idx.begin(), idx.end(), (int32_t)0);
        if (std::adjacent_find(first_real, idx.end()) != idx.end()) {
            std::printf("  row %d: duplicate indices\n", r);
            return false;
        }
        for (int i = 0; i < top_k; ++i) {
            if (o[i] < -1 || o[i] >= n_cols) {
                std::printf("  row %d: index %d out of range\n", r, o[i]);
                return false;
            }
        }

        // `length <= topk`: the contract's shortcut.  Slots past the window
        // take `idx_fill`; the rest are the window's indices, ascending.
        if (length <= top_k) {
            for (int i = 0; i < top_k; ++i) {
                const int32_t want = (i < length) ? i : -1;
                if (o[i] != want) {
                    std::printf("  row %d: shortcut slot %d is %d, wanted %d\n",
                                r, i, o[i], want);
                    return false;
                }
            }
            continue;
        }

        row.assign(length, 0.0f);
        std::copy(src, src + length, row.begin());
        std::nth_element(row.begin(), row.begin() + (top_k - 1), row.end(),
                         std::greater<float>());
        const float kth = row[top_k - 1];
        int above_cpu = 0;
        for (int c = 0; c < length; ++c) above_cpu += (src[c] > kth);

        int above = 0, equal = 0;
        for (int i = 0; i < top_k; ++i) {
            const float v = src[o[i]];
            if (v > kth) ++above;
            else if (v == kth) ++equal;
            else {
                std::printf("  row %d: selected %.9g below the row's k-th "
                            "(%.9g)\n", r, v, kth);
                return false;
            }
        }
        if (above != above_cpu || above + equal != top_k) {
            std::printf("  row %d: %d above / %d tied against %d above on the "
                        "row\n", r, above, equal, above_cpu);
            return false;
        }
    }
    return true;
}

int run_cell(const Cell& cell) {
    const int B = cell.n_rows, V = cell.n_cols, K = cell.top_k;
    std::printf("── B=%d V=%d k=%d%s ──\n", B, V, K, cell.ragged ? " (ragged)" : "");

    const size_t n_scores = (size_t)B * V;
    const size_t n_out = (size_t)B * K;
    std::vector<float> host(n_scores);
    std::mt19937 gen(1234);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (size_t i = 0; i < n_scores; ++i) host[i] = dist(gen);

    std::vector<int32_t> host_lengths;
    if (cell.ragged) {
        host_lengths.resize(B);
        for (int r = 0; r < B; ++r) {
            // Every third row is shorter than k (shortcut), the rest are the
            // whole row or a long prefix of it.
            host_lengths[r] = (r % 3 == 2) ? (int)(K / 2) : (r % 2 ? V : V / 3);
        }
    }

    float* d_scores = nullptr;
    int32_t* d_out = nullptr;
    int32_t* d_lengths = nullptr;
    CUDA_CHECK(cudaMalloc(&d_scores, n_scores * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n_out * sizeof(int32_t)));
    CUDA_CHECK(cudaMemcpy(d_scores, host.data(), n_scores * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_out, 0x5A, n_out * sizeof(int32_t)));
    if (cell.ragged) {
        CUDA_CHECK(cudaMalloc(&d_lengths, host_lengths.size() * sizeof(int32_t)));
        CUDA_CHECK(cudaMemcpy(d_lengths, host_lengths.data(),
                              host_lengths.size() * sizeof(int32_t),
                              cudaMemcpyHostToDevice));
    }

    ds_topk(d_scores, d_lengths, d_out, B, V, K, kSmCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<int32_t> got(n_out);
    CUDA_CHECK(cudaMemcpy(got.data(), d_out, n_out * sizeof(int32_t),
                          cudaMemcpyDeviceToHost));

    const bool ok = verify_row(host, got, host_lengths, B, V, K);
    std::printf("  verify: %s\n", ok ? "PASS" : "FAIL");

    // `--dump-prefix Png` names the files after the shape, so one run of the
    // four default cells drops twelve files that cannot collide.
    const std::string tag = dump_prefix.empty()
        ? std::string()
        : dump_prefix + ".b" + std::to_string(B) + ".v" + std::to_string(V) +
              ".k" + std::to_string(K);

    // With `--dump-prefix P`, also write the exact bytes this run selected from
    // and the answer it produced, so a cross-check against another
    // implementation can run on *this* data instead of regenerating it.  The
    // generator is the seeded `mt19937(1234)` / `normal` above, so the files are
    // reproducible from the shape alone; the dump is just a convenience for a
    // process that does not want to re-implement it -- and it removes the
    // "did we both draw the same numbers" question from the comparison.
    if (!tag.empty()) {
        write_dump(tag + ".scores.f32", host);
        write_dump(tag + ".idx.i32", got);
        std::printf("  dump : %s.scores.f32 (%zu B) + %s.idx.i32 (%zu B)\n",
                    tag.c_str(), host.size() * sizeof(float), tag.c_str(),
                    got.size() * sizeof(int32_t));
        if (!host_lengths.empty()) {
            write_dump(tag + ".lengths.i32", host_lengths);
            std::printf("  dump : %s.lengths.i32 (%zu B, per-row window)\n",
                        tag.c_str(), host_lengths.size() * sizeof(int32_t));
        }
    }

    // Timed: back-to-back launches, no sync between them, so the number is the
    // kernel's and not the driver's.  The workspace is allocated once and
    // reused across calls here -- see `xcore1000_maca_topk.cu` -- which is what
    // upstream's `ChunkedScratch` does, so the timed region is the launches and
    // nothing else.  The clock on this box is not locked (persistence off), so
    // take `best` seriously and the drift between repeats as the error bar.
    for (int i = 0; i < cell.n_timed; ++i)
        ds_topk(d_scores, d_lengths, d_out, B, V, K, kSmCount);
    CUDA_CHECK(cudaDeviceSynchronize());

    double best_ms = 1e30;
    for (int rep = 0; rep < cell.repeats; ++rep) {
        const auto t0 = std::chrono::steady_clock::now();
        for (int i = 0; i < cell.n_timed; ++i)
            ds_topk(d_scores, d_lengths, d_out, B, V, K, kSmCount);
        CUDA_CHECK(cudaDeviceSynchronize());
        const auto t1 = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(t1 - t0).count() /
            cell.n_timed;
        if (ms < best_ms) best_ms = ms;
    }

    // One read pass over the score matrix, `K` int32 written per row.  The
    // split reads the row twice (a histogram pass and a collect pass) and the
    // merge reads the candidates once, so this is the *useful* traffic and not
    // what the DRAM actually saw.
    const double bytes =
        (double)n_scores * sizeof(float) + (double)n_out * sizeof(int32_t);
    std::printf("  time : %.4f ms   (best of %d, %d launches each)\n",
                best_ms, cell.repeats, cell.n_timed);
    std::printf("  bw   : %.1f GB/s  (%.1f MB useful: %.1f MB read + %.1f MB "
                "written)\n",
                bytes / (best_ms * 1e-3) / 1e9, bytes / 1e6,
                (double)n_scores * sizeof(float) / 1e6,
                (double)n_out * sizeof(int32_t) / 1e6);

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_out));
    if (d_lengths) CUDA_CHECK(cudaFree(d_lengths));
    return ok ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
    // `--dump-prefix P` (or `-d P`) is the only flag; the rest are positional.
    // Pulled out before the positional parse so that
    // `./ds_topk_ref --dump-prefix /tmp/x 256 16384 2048 5` and
    // `./ds_topk_ref 256 16384 2048 5 --dump-prefix /tmp/x` both work.
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--dump-prefix" || a == "-d") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "%s needs a path prefix\n", a.c_str());
                return 2;
            }
            dump_prefix = argv[++i];
            continue;
        }
        args.push_back(a);
    }

    int dev = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    std::printf("device : %s  sm=%d warp=%d smem/block=%d sharedMemPerMultiprocessor=%d "
                "(driver uses sm_count=%d)\n",
                prop.name, prop.multiProcessorCount, prop.warpSize,
                (int)prop.sharedMemPerBlock, (int)prop.sharedMemPerMultiprocessor,
                kSmCount);
    if (!dump_prefix.empty())
        std::printf("dump   : prefix %s (raw host bytes, no header; tag is "
                    "<prefix>.b<rows>.v<cols>.k<topk>)\n", dump_prefix.c_str());

    auto arg = [&](size_t i) { return std::atoi(args[i].c_str()); };

    int rc = 0;
    if (!args.empty()) {
        Cell cell{256, 524288, 2048, 3, 3};
        if (args.size() > 0) cell.n_rows = arg(0);
        if (args.size() > 1) cell.n_cols = arg(1);
        if (args.size() > 2) cell.top_k = arg(2);
        if (args.size() > 3) cell.repeats = arg(3);
        rc |= run_cell(cell);
        std::printf("\n%s\n", rc == 0 ? "PASS" : "FAIL");
        return rc;
    }

    // V < kF32ChunkedMinVocab (32768): `chunked_f32_applies` is false, so this
    // is the row path -- one CTA per row through `topk_kernel_radix`.
    rc |= run_cell(Cell{256, 16384, 2048, 3, 3});
    // V above the floor and B above kF32ChunksFewBatches, so the count comes
    // from `f32_chunks_large_batch` -- the chunked split.
    rc |= run_cell(Cell{256, 524288, 2048, 3, 3});
    // Per-row windows through the `lengths` table, on the split path (B <= 64,
    // V >= 32768).  Rows whose window is <= k take the shortcut inside the
    // split's own row kernel.
    {
        Cell ragged{8, 65536, 512, 3, 3};
        ragged.ragged = true;
        rc |= run_cell(ragged);
    }
    // `length <= topk` on the row path, which also exercises the shortcut's
    // `idx_fill` slots.
    rc |= run_cell(Cell{2, 1000, 2048, 3, 3});

    std::printf("\n%s\n", rc == 0 ? "ALL PASS" : "FAILURES");
    return rc;
}
