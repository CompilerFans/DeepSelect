// DeepSelect + DeepJIT integration probe: compile and launch the bf16 chunk
// stage1 kernel at two KCHUNK_BLOCK_SIZE variants through DeepJIT, replacing
// an 8-minute AOT arm build with ~30 s of runtime compiles.
//
// Build with ./build.sh (cucc).  Run from the repo root:
//   DEEP_SELECT_REPO=$PWD /tmp/jit_probe
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_runtime.h>

#include <deep_jit/backend/maca/backend.hpp>
#include <deep_jit/runtime/runtime.hpp>

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "cuda: %s at %d\n", cudaGetErrorString(e_), __LINE__); \
    exit(1);} } while (0)

constexpr int kB = 6;
constexpr int kL = 1048576;
constexpr int kTopK = 512;
constexpr int kChunks = 18;

int main() {
    const char* repo_env = std::getenv("DEEP_SELECT_REPO");
    const std::string repo = repo_env ? repo_env : "/root/github/DeepSelect";
    const char* maca = std::getenv("MACA_PATH");
    DJ_HOST_ASSERT(maca != nullptr, "MACA_PATH must be set");
    const std::string maca_path(maca);

    auto jit = deep_jit::create_lazy_jit<deep_jit::MACA>(deep_jit::Config(
        repo, "DSPROBE", "",
        {repo, repo + "/csrc", repo + "/csrc/maca_kernels/common",
         maca_path + "/include", maca_path + "/tools/cu-bridge/include"},
        {}));

    const std::string source =
        "#include \"csrc/maca_kernels/xcore1600/instantiations/"
        "bf16_stage1_k_topk512_nan1.cu\"";

    // device buffers
    uint16_t* scores = nullptr;
    int32_t* lengths = nullptr;
    int32_t* cand_idx = nullptr;
    uint16_t* cand_val = nullptr;
    int32_t* nan_flags = nullptr;
    const size_t scores_bytes = (size_t)kB * kL * 2;
    CK(cudaMalloc(&scores, scores_bytes));
    CK(cudaMalloc(&lengths, kB * 4));
    CK(cudaMalloc(&cand_idx, (size_t)kB * kChunks * kTopK * 4));
    CK(cudaMalloc(&cand_val, (size_t)kB * kChunks * kTopK * 2));
    CK(cudaMalloc(&nan_flags, kB * 4));
    // host-side fill: uniform-random uint16 patterns (any key spread works for
    // the timing; the contract check below reads them back)
    std::vector<uint16_t> h((size_t)kB * kL);
    for (size_t i = 0; i < h.size(); i++) h[i] = (uint16_t)(i * 2654435761u >> 13);
    CK(cudaMemcpy(scores, h.data(), scores_bytes, cudaMemcpyHostToDevice));
    std::vector<int32_t> hl(kB, kL);
    CK(cudaMemcpy(lengths, hl.data(), kB * 4, cudaMemcpyHostToDevice));
    CK(cudaMemset(nan_flags, 0, kB * 4));

    for (int block_size : {512, 1024}) {
        deep_jit::MACA::CompilerOptions opts;
        opts.extra_mxcc_flags = {
            "-DKCHUNK_BLOCK_SIZE=" + std::to_string(block_size),
            "-DDEEP_SELECT_ARCH=1600", "-DDEEP_SELECT_IS_BUILD_ON_MACA",
            "-use-fast-math", "-Xclang", "-fdenormal-fp-math-f32=ieee"};
        auto kernel = jit->compile("stage1", source, opts);

        const size_t smem = 16384 + (size_t)kTopK * 4;
        cudaEvent_t t0, t1;
        CK(cudaEventCreate(&t0)); CK(cudaEventCreate(&t1));
        float best_ms = 1e30f;
        for (int rep = 0; rep < 10; rep++) {
            CK(cudaEventRecord(t0));
            jit->launch(kernel,
                        {.num_smem_bytes = (int)smem,
                         .grid_dim = dim3((unsigned)(kB * kChunks)),
                         .block_dim = dim3((unsigned)block_size)},
                        (const void*)scores, (const void*)lengths,
                        (void*)cand_idx, (void*)cand_val, kB, kL, kChunks,
                        (int64_t)kL, (void*)nan_flags);
            CK(cudaEventRecord(t1)); CK(cudaEventSynchronize(t1));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, t0, t1));
            if (rep > 0 && ms < best_ms) best_ms = ms;
        }
        CK(cudaGetLastError());
        const double gbs = (double)kB * kL * 2 / (best_ms * 1e-3) / 1e9;
        printf("KCHUNK_BLOCK_SIZE=%d: %8.1f us  %7.1f GB/s\n", block_size,
               best_ms * 1000, gbs);

        // contract check on this variant's candidate output: per (row, chunk),
        // min(selected values) >= max(unselected values in the chunk), via a
        // per-chunk selected bitmap (O(L), not O(k*L)).
        std::vector<int32_t> ci((size_t)kB * kChunks * kTopK);
        std::vector<uint16_t> cv((size_t)kB * kChunks * kTopK);
        CK(cudaMemcpy(ci.data(), cand_idx, ci.size() * 4, cudaMemcpyDeviceToHost));
        CK(cudaMemcpy(cv.data(), cand_val, cv.size() * 2, cudaMemcpyDeviceToHost));
        std::vector<uint8_t> sel(kL, 0);
        int bad = 0;
        const int chunk_len = (kL + kChunks - 1) / kChunks + 7 >> 3 << 3;
        for (int b = 0; b < kB && bad == 0; b++) {
            for (int c = 0; c < kChunks && bad == 0; c++) {
                const int cs = c * chunk_len, ce = std::min(kL, cs + chunk_len);
                std::fill(sel.begin() + cs, sel.begin() + ce, (uint8_t)0);
                for (int t = 0; t < kTopK; t++) {
                    const int32_t i = ci[((size_t)b * kChunks + c) * kTopK + t];
                    if (i >= cs && i < ce) sel[i] = 1;
                }
                uint16_t sel_min = 0xFFFF, unsel_max = 0;
                for (int i = cs; i < ce; i++) {
                    if (sel[i]) sel_min = std::min(sel_min, h[(size_t)b * kL + i]);
                    else unsel_max = std::max(unsel_max, h[(size_t)b * kL + i]);
                }
                if (sel_min < unsel_max) bad = 1;
            }
        }
        printf("  contract: %s\n", bad == 0 ? "PASS" : "FAIL");
        if (bad) return 2;
    }
    cudaFree(scores); cudaFree(lengths); cudaFree(cand_idx);
    cudaFree(cand_val); cudaFree(nan_flags);
    return 0;
}
