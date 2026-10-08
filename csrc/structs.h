#pragma once

#include <cstdint>

// ── the platform this build is for ──────────────────────────────────────────
//
// Upstream's `csrc/structs.h` dispatches on the same declaration --
// `#if DEEP_SELECT_IS_BUILD_ON_CUDA / #elif DEEP_SELECT_IS_BUILD_ON_ASCEND /
// #else #error` -- and this tree is the MACA row of that dispatch, so the
// declaration is required here for the same reason: the build must say which
// platform it is for, and `setup.py` says it in `nvcc_args` beside
// `-DDEEP_SELECT_ARCH`.  There is no `#else` arm to fall into, so a build that
// forgets the flag is an error naming the flag rather than a silent choice
// between platforms.
//
// Deliberately before the platform headers: without it the first complaint
// would be `maca_bfloat16.h not found`, which names the symptom and not the
// cause.
#if !defined(DEEP_SELECT_IS_BUILD_ON_MACA)
#error "DEEP_SELECT_IS_BUILD_ON_MACA is not set: this source is built for the \
MACA platform.  See setup.py's build_for_maca, which passes it beside \
-offload-arch and -DDEEP_SELECT_ARCH."
#endif

#include <cuda_runtime_api.h>

// MACA's native bf16.  Every TU includes this header, so `api.cu` and each
// instantiation TU see the same `maca_bfloat16` -- one symbol for
// `TopkSelectConfig<maca_bfloat16, ...>` on both sides.
#include <maca_bfloat16.h>

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 1024; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

// ── the per-family build constants ──────────────────────────────────────────
//
// **One artifact per family, and this is the macro that makes it one.**
// `setup.py` compiles each entry of `CUCC_TARGETS` as its own `-offload-arch`
// with `-DDEEP_SELECT_ARCH=<family>` alongside it, so every image carries a
// host half that knows which device it is for.  `_binding.py` loads the
// artifact matching the family the device reports.
//
// This is the form the tree used before `5f5a93c` and it is back for a
// measured reason, not a stylistic one.  The intermediate form -- one `.so`
// with three device images -- cannot carry per-family host constants at all:
// `-offload-arch=a,b,c` is one compilation, host code exists once in it, and
// `__MACA_ARCH__` is defined in the *device* pass only (measured: all three
// targets report it undefined in the host pass).  So a `#if` on the target in
// host code takes the `#else` branch on every architecture, and the constants
// have to travel as arguments instead.
//
// What that bought, and what it cost, so the trade is on the record: the
// arguments form needs no per-family artifact and no family detection at load
// time, but it also means the kernel reads two machine facts per call
// (`cudaDeviceGetAttribute`, 0.330 us) and that the *device* decides which
// numbers it gets rather than the *build*.  One artifact per family inverts
// both: the numbers are compile-time constants again, and the price is three
// compiles per build, three files in the wheel, and a loader that has to name
// the right one.
//
// **Every family must have a row here.**  There is no `#else`: a target added
// to `CUCC_TARGETS` without a row is a build error naming the macro, which is
// the point -- the alternative is an image that silently carries family
// 1000's numbers.
#ifndef DEEP_SELECT_ARCH
#error "DEEP_SELECT_ARCH is not set: this source is built one artifact per \
family, with -DDEEP_SELECT_ARCH=<1000|1500|1600> beside -offload-arch.  See \
setup.py's build_for_maca."
#endif

// **Every row here is a claim about a part, and the AP counts are claims a
// family can have more than one of.**  `ARCH_SMEM_PER_AP_BYTES` is the budget
// `f32_coarse12_applies` tests the 16 KB arena against, and it is the one
// constant in this table with a behavioral consumer.  `ARCH_SM_COUNT` is
// **not** consumed by any rule: `wave_filled_chunks` and
// `f32_chunk_work_target` take the *runtime* `params.sm_count`, deliberately,
// so that a caller handing this entry a tensor on another part gets a grid
// sized for the part in front of it (see the note above `f32_chunk_work_target`
// in `maca_topk.cu`).  The macro is a number for the record and for a reader;
// the grids do not read it.  **That is what lets a two-SKU family have a row
// at all** -- see the C600U note below.
//
// The 1500 and 1600 rows carry the AP counts those families report and the
// 128 KiB every 128 KiB family has.  The 1600 row is the one that cannot be a
// single number -- see below -- and that is exactly why the *performance* half
// of every measured rule is guarded by the device it was measured on
// (`kF32Coarse12MeasuredSmCount`, `is_f32_coarse12_c600u_ap_count`) rather
// than by this macro: geometry compiles in, tuning still has to be earned.
// **The split predicates are the exception and the gap**: `chunked_f32_applies`
// and `chunked_bf16_applies` have no such guard, so a 1600 image does route
// its fp32 split on the C500 chunk counts (`f32_chunks_large_batch` and
// `wave_filled_chunks` do branch on `sm_count`, so the count is adapted, but
// the *band* was never re-fitted off this part).
//
// **The C600U is built two ways -- 28 APs and 32 -- and both are ordinary
// parts, so this row cannot name the family's count and the 1500 row's 28 is
// not a counterexample to it.**  The row read 32 until `2fde057` changed it to
// 28, on a measurement that is true of the part it was taken on
// (`torch.cuda.get_device_properties().multi_processor_count` and
// `cudaDevAttrMultiProcessorCount` through a `cucc`-compiled probe, both 28,
// stable across repeated reads and under load); what that commit's reasoning
// got wrong was the inference that the two numbers "cannot both be right for
// the same silicon generation".  They can: the family ships both.  So the
// number picked here is the one the *ladder* was fitted on, and the gap it
// leaves is closed at the route rather than here -- `maca_topk.cu`'s
// `is_f32_coarse12_c600u_ap_count` accepts both, and says why.
//
// Nothing routes on *this* -- `ARCH_SM_COUNT` has no behavioral consumer (the
// paragraph above is the audit finding that established that) -- so the value
// is a record either way.  **Two consequences a reader should not have to
// discover:**
//
//   - The two `perf_data/MetaX_C600-U/20260915_*` snapshots carry `sm_count
//     32` in their manifests and were taken on another host
//     (`mx-cjxu-ws-cr-20260915131045`; `...-2026092013092900002848` next, with
//     the GPU PCI buses moving `a4/a5` -> `a8/a9`; today's is
//     `...-2026100816195500003270`, buses `63:`), so it is not knowable from
//     the record whether that 32 was the other SKU or another board -- which
//     is the point: the directory named for the device cannot separate them,
//     and that is the failure mode the `device_dir` naming rule exists to
//     prevent (see the note in `scripts/perf_snapshot.py`).  Re-measuring the
//     C600U baseline on this host would write `sm_count 28` into the same
//     directory, and a comparison across it has to read the manifests rather
//     than the path.
//   - `f32_chunks_large_batch` and `f32_chunk_work_target` read the *runtime*
//     `sm_count`, so they already adapt; but the chunk *band* was fitted on
//     104 APs, and `MetaX_C500/baseline` is where the numbers for a re-fit
//     would have to come from.  A C500 baseline's own manifest says
//     `sm_count 104`, so that one is self-consistent.
//
// `ARCH_FAMILY` is what the artifact is *named*, and the loader matches on it:
// `deep_select_maca_xcore<N>.so` is the file, `_binding.family_suffix()` is
// the reader, and this constant is the one place the number is written down on
// the C++ side.  It exists so the pairing can be checked from inside the
// artifact instead of assumed.
#if DEEP_SELECT_ARCH == 1000
static constexpr uint32_t ARCH_FAMILY = 1000;
static constexpr uint32_t ARCH_SM_COUNT = 104;              // C500
static constexpr uint32_t ARCH_SMEM_PER_AP_BYTES = 64 * 1024;
#elif DEEP_SELECT_ARCH == 1500
static constexpr uint32_t ARCH_FAMILY = 1500;
static constexpr uint32_t ARCH_SM_COUNT = 28;               // C600
static constexpr uint32_t ARCH_SMEM_PER_AP_BYTES = 128 * 1024;
#elif DEEP_SELECT_ARCH == 1600
static constexpr uint32_t ARCH_FAMILY = 1600;
static constexpr uint32_t ARCH_SM_COUNT = 28;               // C600U; the family
// also ships at 32 -- see above, and `is_f32_coarse12_c600u_ap_count` for the
// route that must accept both
static constexpr uint32_t ARCH_SMEM_PER_AP_BYTES = 128 * 1024;
#else
#error "unknown xcore family in DEEP_SELECT_ARCH; add its row here (AP count \
and per-AP shared memory) and its entry to CUCC_TARGETS in setup.py"
#endif
static_assert(ARCH_FAMILY == (uint32_t)DEEP_SELECT_ARCH,
              "ARCH_FAMILY is the row's own key: it must equal the macro that "
              "selected the row, or the artifact's name would not name it");
static_assert(ARCH_SM_COUNT > 0, "ARCH_SM_COUNT must be set per family");
static_assert(ARCH_SMEM_PER_AP_BYTES > 0,
              "ARCH_SMEM_PER_AP_BYTES must be set per family");

// The `deep_gemm` selector's perf shapes, transcribed from
// `deep_gemm/tests/test_indexer_topk_selector.py::SELECTOR_PERF_SHAPES`
// (test-topk + sglang + dsa), all fp32 and `top_k = 2048`.
//
// This is a *mirror*, kept by hand: `tests/test.py` carries the Python list
// that the perf grid, the perf recorder and the correctness sample actually
// run, and this copy is the compile-time record of the same shapes.  Edit the
// two together.  Nothing in `csrc/` reads it.
//
// Each entry is (n_rows, n_cols, seq_len).  `seq_len` is the window the
// `deep_gemm` grid declares, which this repository does not synthesize -- it
// ranks the whole row -- so the two are not comparable at the same shape.
//
// `top_k` is 2048 because that is what the `deep_gemm` grid measures and what
// its backend serves; a larger `top_k` would leave its column empty.
struct DeepGemmSelectorShape {
    uint32_t n_rows;
    uint32_t n_cols;
    uint32_t seq_len;
};
inline constexpr DeepGemmSelectorShape kDeepGemmSelectorPerfShapes[] = {
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
inline constexpr uint32_t kDeepGemmSelectorPerfTopK = 2048;


static constexpr uint32_t MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23;
static constexpr uint32_t MAX_VOCAB_SIZE = 1u << 23;
static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION);

// The per-family constants are above, in the `DEEP_SELECT_ARCH` table.  They
// are compile-time constants *because* there is one artifact per family: host
// code exists once per compilation, so the family the image is for is known
// when the image is built, and the loader picks the artifact matching the
// device.  `sm_count` is still an argument (see `ffi_entries.h`) -- not
// because it cannot be compiled in, but because the entry point must keep
// working when it is handed a device that is not the one the artifact was
// built for, and refusing a mismatched grid is a better failure than a grid
// sized for the wrong part.
//
// The ported kernels under `csrc/maca_kernels/xcore1600/` selected their config tuples
// against `NATIVE_SHARED_MEMORY_PER_SM_BYTES` (64 KiB for family 1000, 128 KiB
// for 1500/1600) and asserted at compile time that their staging fitted.  That
// is `ARCH_SMEM_PER_AP_BYTES` now, and the assertion can come back with it:
// the constant is per-family again, so a 1600 image can check its own tuples
// against its own budget.
//
// **The port is off the build, and the rename has not been done.**  Two
// `static_assert`s in that tree still name the old
// `NATIVE_SHARED_MEMORY_PER_SM_BYTES` -- `v3/topk_select.cuh` and
// `v3_fp32/topk_select.cuh`, one each -- and that identifier is defined
// nowhere, so those files would not compile if `csrc/maca_kernels/xcore1600/` went back on
// `SOURCES`.  The rename is part of re-adding the tree, not something a
// reader can assume is already done; see `setup.py`'s note in
// `build_for_maca` for what re-adding takes.


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
