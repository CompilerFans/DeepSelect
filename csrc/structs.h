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
