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

// There is no compile-time architecture selection here, and no
// `DEEP_SELECT_NATIVE_ARCH`.  One extension carries every family's image
// (`setup.py`: one source, one `-offload-arch` list), so a per-family constant
// has no build to be baked into; the kernel that needs one takes it as an
// argument instead.  `deep_select/_arch.py` is where the rows live.
//
// The ported kernels under `csrc/xcore1600/` selected their config tuples
// against `NATIVE_SHARED_MEMORY_PER_SM_BYTES` (64 KiB for family 1000, 128 KiB
// for 1500/1600) and asserted at compile time that their staging fitted.  That
// constant went with the macro: with one extension there is no single capacity
// to assert against.  The port is off the build either way -- see
// `deep_select/_arch.py`'s closing note.


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
