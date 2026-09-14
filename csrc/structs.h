#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// [MACA] bf16 值类型改用 MACA 原生头（原先是靠 cute 顺带引入 cu-bridge 的
//   `cuda_bf16.h`）。放在这里是因为它是全仓的"值类型"汇聚点：每个
//   `xcore1600/*/topk_select.h` 都 include 它，于是 api.cu 与各实例化 TU
//   拿到的是同一个 `maca_bfloat16`，`TopkSelectConfig<maca_bfloat16, ...>` 的
//   模板实体在两侧是同一个符号。
#include <maca_bfloat16.h>

static constexpr uint32_t INPUT_STRIDE_ALIGNMENT_REQUIREMENT = 1024; // In number of bytes
static constexpr uint32_t OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT = 32; // In number of bytes

static constexpr uint32_t MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION = 1u << 23;
static constexpr uint32_t MAX_VOCAB_SIZE = 1u << 23;
static_assert(MAX_VOCAB_SIZE <= MAX_INT_ADDITION_RANGE_BY_FP32_SIMULATION);

// [MACA] Shared memory capacity of the architecture this extension is being
// built for, **per SM**.
//
// One target per build (`--offload-arch`), so this is a compile-time fact, and
// the kernels use it to reject -- at compile time -- any config that could not
// possibly launch on it.  That matters because upstream's config tables are
// sized for 227 KiB: carrying one of those tuples into an xcore1000 build must
// fail here, not at a launch that returns mcErrorInvalidValue on the device.
//
// Per SM and not per block because that is the quantity upstream's own check
// is written against: `smem_size * TARGET_OCCUPANCY <= shared_memory_size_per_sm`,
// i.e. the CTAs of one occupancy group have to be co-resident.  On C500 the
// two figures are equal (`mcDeviceGetAttribute` reports 65536 for both
// MaxSharedMemoryPerBlockOptin and MaxSharedMemoryPerMultiprocessor), so the
// single number covers the per-block launch limit as well.
//
// `DEEP_SELECT_NATIVE_ARCH` carries the xcore family base.  It comes from
// setup.py, which sets it from the same `CUCC_TARGETS` entry it passes to
// `--offload-arch` -- the toolchain defines `__MACA_ARCH__` from that target,
// but only in the device pass, and the templates that check the capacity are
// parsed in both, so the value has to reach the host pass explicitly.
//
// The capacities mirror the xcore family table in the host repository
// (`deep_gemm/utils/arch_config.py`, `XcoreFamily.shared_memory_bytes`); this
// repository is standalone and cannot import it, so the two are kept in sync
// by hand -- a change to either belongs in the same review.
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

// [MACA] SM ("AP") count of the architecture this extension is built for.
//
// Compile-time and per target, from the same `DEEP_SELECT_NATIVE_ARCH` as the
// capacity above -- NOT read from the runtime API, because the quantity is a
// property of the arch the kernel was compiled for and every decision that
// uses it (grid sizing, chunk counts) is a compile-time or host-side constant.
//
// **This is a reservation, not a convenience.**  A chunked grid is sized in
// CTAs, and a grid whose CTA count is not a multiple of the SM count leaves
// `ctas mod SM` SMs idle in its last wave -- the smaller the batch, the larger
// the fraction.  C500's 104 APs are what the split's chunk count was tuned
// against (16 chunks at b6 = 96 CTAs = 92% fill); the same 16 on a 32-SM
// C600U would be 3 full waves (100%), and on a 28-SM C600 3.43 waves (86%).
//
// The counts are the AP/SM counts of the parts in each family:
//   xcore1000  C500      104
//   xcore1500  C600       28
//   xcore1600  C600U      32
// (C600-UL is a C600U part and shares the family's count; if a future part in
// one of these families reports a different count, this table is where it
// goes -- and so is the capacity table above, which has the same shape.)
#if DEEP_SELECT_NATIVE_ARCH == 1000
static constexpr uint32_t NATIVE_SM_COUNT = 104;
#elif DEEP_SELECT_NATIVE_ARCH == 1500
static constexpr uint32_t NATIVE_SM_COUNT = 28;
#elif DEEP_SELECT_NATIVE_ARCH == 1600
static constexpr uint32_t NATIVE_SM_COUNT = 32;
#endif
static_assert(NATIVE_SM_COUNT > 0, "NATIVE_SM_COUNT must be set per family");

// [MACA] Work target for the fp32 chunked split's SMALL-batch arm, per family
// and per build, from the same `DEEP_SELECT_NATIVE_ARCH` as everything above.
//
// The split's small-batch arm exists to fill a machine that a short batch
// leaves empty: `batches` row CTAs over 104 APs is a fraction of a wave, and
// the split multiplies the grid by `chunks + 1`.  How many chunks that takes is
// not a fixed 16 -- it is "enough to fill a couple of waves, and no more",
// because every extra chunk also adds a merge CTA whose work is a fraction of a
// row's.  The measured shape of that (C500, `V = 32768`, one binary with only
// `DEEP_SELECT_F32_CHUNKS` varying, 3 alternating rounds, median):
//
//   batches      c=2     c=4     c=8    c=16    c=32    best
//         6     66.6    58.1    58.6    56.8    66.2    16
//        16     69.0    59.8    58.0    60.1    77.4     8
//        24     69.6    62.1    60.6    76.8    86.9     8
//        32     68.8    61.3    61.3    77.9    93.6    4/8
//        40     70.8    66.9    79.6    91.6   105.8     4
//        48     72.9    65.5    83.0    96.2   114.0     4
//        64     78.6    74.3    89.0   111.5   131.2     4
//        80     82.7    91.2   104.7   126.9   154.6     2
//        96     83.7    96.6   112.7   138.9   173.1     2
//       256    148.9   172.0   220.4   302.5   372.5     2
//
// That is `chunks = largest power of two <= K / batches` almost everywhere, so
// the policy is one constant: **K is how much chunked work one SM should be
// carrying.**  Fitted over all 24 measured points, K = 256 matches 22 of them
// and K = 208 is worse (14.7% worst against 3.6%).
//
// K = 260 is 2.5 x 104, i.e. the fitted 256 rounded into a form that is
// obviously a property of the machine rather than a fitted number.  Measured
// behaviour is identical to K = 256 (same 22 points, same 3.6% worst), and both
// keep the batches they change strictly inside the small-batch tier --
// b24..b64, which is where the fixed 16 was wrong by 11..31%.
//
// **The C600/C600U rows are a scaled reservation, not a measurement.**  No
// hardware for those was available, and the ratio K/SM_COUNT is the one thing
// that is not evidenced here; it is 2.5 on C500.  Scaling it is the only
// option, and the value is deliberately a separate constant per family so a
// real measurement replaces one number instead of a formula.
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
