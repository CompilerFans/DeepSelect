#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// MACA's native bf16.  Every TU includes this header, so `api.cu` and each
// instantiation TU see the same `maca_bfloat16` -- one symbol for
// `TopkSelectConfig<maca_bfloat16, ...>` on both sides.
#include <maca_bfloat16.h>

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 1024; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

// The host repository's selector perf shapes, transcribed from
// `deep_gemm/tests/test_indexer_topk_selector.py::SELECTOR_PERF_SHAPES`
// (test-topk + sglang + dsa), all fp32 and `top_k = 2048`.  Three consumers
// need the same list -- the perf grid, the perf recorder and the correctness
// sample -- so the C++ copy lives here and `tests/test.py` carries the Python
// one; a list repeated in three places is a list that will disagree with
// itself.
//
// Each entry is (n_rows, n_cols, seq_len).  `seq_len` is the window the host
// grid declares, which this repository does not synthesize -- it ranks the
// whole row -- so the two are not comparable at the same shape.
//
// `top_k` is 2048 because that is what the host grid measures and what the
// `deep_gemm` backend serves; a larger `top_k` would leave its column empty.
struct HostSelectorShape {
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t seq_len;
};
inline constexpr HostSelectorShape kHostSelectorPerfShapes[] = {
    {   1,  66551,  66551},   // test-topk-bs1
    {  16,  66551,  66551},   // test-topk-bs16
    { 132,  66551,  66551},   // test-topk-bs132
    { 512,  66551,  66551},   // test-topk-bs512
    {   1, 131072,   2048},   // sglang-bs1-seq2048
    {   1, 131072,   4096},   // sglang-bs1-seq4096
    {   1, 131072,  16384},   // sglang-bs1-seq16384
    {   1, 131072,  65536},   // sglang-bs1-seq65536
    { 132, 131072,   2048},   // sglang-bs132-seq2048
    { 132, 131072,   4096},   // sglang-bs132-seq4096
    { 132, 131072,  16384},   // sglang-bs132-seq16384
    { 132, 131072,  65536},   // sglang-bs132-seq65536
    { 256, 131072,   2048},   // sglang-bs256-seq2048
    { 256, 131072,   4096},   // sglang-bs256-seq4096
    { 256, 131072,  16384},   // sglang-bs256-seq16384
    { 256, 131072,  65536},   // sglang-bs256-seq65536
    {4096, 131072,   2048},   // sglang-bs4096-seq2048
    {4096, 131072,   4096},   // sglang-bs4096-seq4096
    {4096, 131072,  16384},   // sglang-bs4096-seq16384
    {4096, 131072,  65536},   // sglang-bs4096-seq65536
    {   1, 107520, 107520},   // dsa-bs1-seq107520
    {  16,  66551,  66551},   // dsa-bs16-seq66551
    { 132, 107520, 107520},   // dsa-bs132-seq107520
    { 256, 107520, 107520},   // dsa-bs256-seq107520
    {4096, 107520, 107520},   // dsa-bs4096-seq107520
};
inline constexpr uint32_t kHostSelectorPerfTopK = 2048;


static constexpr uint32_t MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23;
static constexpr uint32_t MAX_VOCAB_SIZE = 1u << 23;
static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION);

// Shared memory per SM of the architecture this extension is built for.
//
// One target per build (`--offload-arch`), so this is a compile-time fact, and
// the kernels use it to reject -- at compile time -- a config that could not
// launch.  Upstream's config tables are sized for 227 KiB, so without this
// check a 128 KiB tuple would ride into a 64 KiB build and fail at launch with
// mcErrorInvalidValue instead.
//
// `DEEP_SELECT_NATIVE_ARCH` is the xcore family base, set by `setup.py` from
// the same `CUCC_TARGETS` entry it passes to `--offload-arch`.  The toolchain
// defines `__MACA_ARCH__` from that target, but only in the device pass, and
// these templates are parsed in both.
//
// Mirrors `deep_gemm/utils/arch_config.py`'s `XcoreFamily.shared_memory_bytes`.
// This repository cannot import it, so the two are kept in sync by hand.
#if !defined(DEEP_SELECT_NATIVE_ARCH)
#error "DEEP_SELECT_NATIVE_ARCH is not defined; build through setup.py, which \
sets it per target from CUCC_TARGETS"
#elif DEEP_SELECT_NATIVE_ARCH == 1000
static constexpr uint32_t NATIVE_SHARED_MEMORY_PER_SM_BYTES = 64 * 1024;
#elif DEEP_SELECT_NATIVE_ARCH == 1500 || DEEP_SELECT_NATIVE_ARCH == 1600
static constexpr uint32_t NATIVE_SHARED_MEMORY_PER_SM_BYTES = 128 * 1024;
#else
#error "unknown xcore family in DEEP_SELECT_NATIVE_ARCH; add its shared \
memory capacity here and its row to deep_gemm/utils/arch_config.py"
#endif

// SM count of the architecture this extension is built for, from the same
// `DEEP_SELECT_NATIVE_ARCH`.  Compile-time, not read from the runtime API:
// grid sizing and chunk counts are host-side constants and depend on it.
//
// It matters because a chunked grid is sized in CTAs and one whose count is
// not a multiple of the SM count leaves `ctas mod SM` SMs idle in its last
// wave.  If a future part in one of these families reports a different count,
// this table is where it goes.
#if DEEP_SELECT_NATIVE_ARCH == 1000
static constexpr uint32_t NATIVE_SM_COUNT = 104;
#elif DEEP_SELECT_NATIVE_ARCH == 1500
static constexpr uint32_t NATIVE_SM_COUNT = 28;
#elif DEEP_SELECT_NATIVE_ARCH == 1600
static constexpr uint32_t NATIVE_SM_COUNT = 32;
#endif
static_assert(NATIVE_SM_COUNT > 0, "NATIVE_SM_COUNT must be set per family");

// Work target for the fp32 chunked split's small-batch arm, per family.
//
// The split's small-batch arm fills a machine a short batch leaves empty:
// `batches` row CTAs over one wave's worth of SMs is a fraction of a wave.
// How many chunks that takes is "enough to fill a couple of waves, and no
// more" -- every extra chunk also adds a merge CTA -- and `chunks = largest
// power of two <= K / batches` fits the measured sweep, so the policy is the
// single constant K.  K = 260 is 2.5 x the SM count, rounded from a fitted
// 256 to a form that is visibly a machine property.
//
// The 1500/1600 rows are a scaled reservation, not a measurement -- no such
// hardware was available.  Keeping them as separate constants means a real
// measurement replaces one number rather than changing a formula.  The sweep
// behind K, and the C600/C600U caveat, are in
// `docs/C500-radix-perf-ledger.zh.md`.
#if DEEP_SELECT_NATIVE_ARCH == 1000
static constexpr uint32_t NATIVE_F32_CHUNK_WORK_TARGET = 260;   // measured
#elif DEEP_SELECT_NATIVE_ARCH == 1500
static constexpr uint32_t NATIVE_F32_CHUNK_WORK_TARGET = 70;    // scaled, unmeasured
#elif DEEP_SELECT_NATIVE_ARCH == 1600
static constexpr uint32_t NATIVE_F32_CHUNK_WORK_TARGET = 80;    // scaled, unmeasured
#endif
static_assert(NATIVE_F32_CHUNK_WORK_TARGET > 0,
              "NATIVE_F32_CHUNK_WORK_TARGET must be set per family");


struct TopkSelectArgs {
    uint32_t batch_size;
    uint32_t vocab_size;
    uint32_t topk;

    void* input;
    void* output_value;
    void* output_index;
    int* begin_ptr;
    int* end_ptr;
    int* output_idx_offset;

    // All strides are in number of elements, not bytes
    uint64_t stride_input_batch;
    uint64_t stride_output_value_batch;
    uint64_t stride_output_index_batch;

    bool sorted_value;
    bool sorted_index;
    bool return_value;
    int idx_oob_fill_value;
    float value_oob_fill_value;
    bool abort_when_nan_found;      

    uint64_t shared_memory_size_per_sm;
    cudaStream_t stream;
};
