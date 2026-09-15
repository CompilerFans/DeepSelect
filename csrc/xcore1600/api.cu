// Host-side topk() AND its tvm-ffi export: this file does not include
// `csrc/ffi/ffi_entries.h`, it carries its own `TVM_FFI_DLL_EXPORT_TYPED_FUNC`
// below.  (This line said the export lived in that header until 2026-09-16,
// which would send a reader changing this signature to a file this TU never
// sees -- the two architecture trees' entries would then diverge with nothing
// to report it, the same failure `ffi_entries.h` itself was carrying.)  The
// kernel template instantiations are in separate files for parallel
// compilation.
//
// [MACA] `.cu`, not upstream's `.cpp`: torch routes a `.cpp` to `$cxx` and a
//   `.cu` to `$nvcc`, and mxcc defines `__MACA__` for a `.cu` and only for a
//   `.cu` -- which `kerutils/common/common.h` keys `KERUTILS_IS_BUILD_ON_CUDA`
//   on.  As a `.cu` this file reaches mxcc like every other source here and
//   needs no separate host flag list.  It is host code either way (no
//   `__global__`), so the device pass emits nothing.
#include "../ffi/ffi_checks.h"
#include "../ffi/ffi_error.h"
#include "../ffi/ffi_tensor.h"

#include <tvm/ffi/extra/c_env_api.h>

#include "dispatch_utils.h"

#include "config.h"
#include "v3/topk_select.h"
#include "v3_fp32/topk_select.h"
#include <cstdlib>
#include <cstdio>
// [MACA] 不能 `#include <format>`：宿主 GCC 11.4 的 libstdc++ 还没有它。见下方
//   `check_dim0_stride`。

namespace deep_select {

void topk(
    const tvm::ffi::TensorView &input,
    int64_t topk,
    const tvm::ffi::Optional<tvm::ffi::TensorView> &end,
    bool sorted_value,
    bool sorted_index,
    const tvm::ffi::Optional<tvm::ffi::TensorView> &output_value,
    const tvm::ffi::TensorView &output_index,
    const tvm::ffi::Optional<tvm::ffi::TensorView> &output_idx_offset,
    int64_t idx_oob_fill_value,
    double value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    namespace ffi = deep_select::ffi;

    DS_HOST_CHECK(input.ndim() == 2, "input must be 2-D, got ", input.ndim());
    const int batch_size = (int)ffi::size(input, 0);
    const int vocab_size = (int)ffi::size(input, 1);
    const bool bfloat16_in = ffi::is_bfloat16(input);

    DS_HOST_CHECK(topk > 0, "topk must > 0");
    DS_HOST_CHECK(!(sorted_value && !return_value), "`return_value` must be enabled when `sorted_value` is True");
    DS_HOST_CHECK(!(sorted_value && sorted_index), "`sorted_value` and `sorted_index` cannot be used at the same time");
    DS_HOST_CHECK(!(sorted_value && bfloat16_in), "`sorted_value` is only supported for float32 input");
    if (return_value) {
        DS_HOST_CHECK(output_value.has_value(), "`output_value` must not be `None` when `return_value` is True");
    }

    DS_CHECK_DEVICE(input);
    if (end.has_value()) DS_CHECK_DEVICE(end.value());
    if (output_value.has_value()) DS_CHECK_DEVICE(output_value.value());
    DS_CHECK_DEVICE(output_index);
    if (output_idx_offset.has_value()) DS_CHECK_DEVICE(output_idx_offset.value());

    DS_CHECK_SHAPE(input, batch_size, vocab_size);
    if (end.has_value()) DS_CHECK_SHAPE(end.value(), batch_size);
    if (output_value.has_value()) DS_CHECK_SHAPE(output_value.value(), batch_size, topk);
    DS_CHECK_SHAPE(output_index, batch_size, topk);
    if (output_idx_offset.has_value()) DS_CHECK_SHAPE(output_idx_offset.value(), batch_size);

    DS_CHECK_DTYPE(input, bfloat16_in ? ffi::kBFloat16 : ffi::kFloat32);
    if (end.has_value()) DS_CHECK_DTYPE(end.value(), ffi::kInt32);
    if (output_value.has_value()) DS_CHECK_DTYPE(output_value.value(), input.dtype());
    DS_CHECK_DTYPE(output_index, output_index.dtype());
    if (output_idx_offset.has_value()) DS_CHECK_DTYPE(output_idx_offset.value(), ffi::kInt32);

    DS_CHECK_LAST_DIM_CONTIGUOUS(input);
    if (end.has_value()) DS_CHECK_CONTIGUOUS(end.value());
    if (output_value.has_value()) DS_CHECK_LAST_DIM_CONTIGUOUS(output_value.value());
    DS_CHECK_LAST_DIM_CONTIGUOUS(output_index);
    if (output_idx_offset.has_value()) DS_CHECK_CONTIGUOUS(output_idx_offset.value());

    auto check_dim0_stride = [&](const char tensor_name[], const tvm::ffi::TensorView &tensor, uint32_t alignment_requirement_bytes) {
        int64_t cur_stride = ffi::stride(tensor, 0);
        uint64_t itemsize = ffi::element_size(tensor);
        // [MACA] snprintf 而非上游的 `std::format`：宿主 GCC 11.4 的 libstdc++ 无
        //   `<format>`（GCC 13 才有），而 fmt 是外层仓的 submodule，本仓不依赖它。
        //   报文文本与 std::format 逐字符相同。
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "%s.stride(0) (currently %lld numbers) must be a multiple of %u Bytes (%llu numbers)",
            tensor_name, (long long)cur_stride,
            (unsigned)alignment_requirement_bytes,
            (unsigned long long)(alignment_requirement_bytes / itemsize));
        DS_HOST_CHECK(cur_stride * itemsize % alignment_requirement_bytes == 0, msg);
    };
    check_dim0_stride("input", input, INPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    check_dim0_stride("output_index", output_index, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    if (output_value.has_value()) {
        check_dim0_stride("value", output_value.value(), OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    }

    // `sharedMemPerBlockOptin` is the figure the kernel's `cudaFuncSetAttribute`
    // needs; `multiProcessorCount` turns a batch into a wave count below.
    const int device_index = ffi::device_index(input);
    int shared_mem_per_block_optin = 0;
    int num_sm = 0;
    DS_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
        &shared_mem_per_block_optin, cudaDevAttrMaxSharedMemoryPerBlockOptin,
        device_index));
    DS_CUDA_RUNTIME_CHECK(cudaDeviceGetAttribute(
        &num_sm, cudaDevAttrMultiProcessorCount, device_index));
    const cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(
        (int32_t)ffi::device_type(input), device_index);
    TopkSelectArgs args = {
        (uint32_t)batch_size,
        (uint32_t)vocab_size,
        (uint32_t)topk,

        ffi::data_ptr(input),
        return_value ? ffi::data_ptr(output_value.value()) : nullptr,
        ffi::data_ptr(output_index),
        // `begin` is rejected by the public interface and is not a parameter here.
        nullptr,
        end.has_value() ? ffi::data_ptr<int>(end.value()) : nullptr,
        output_idx_offset.has_value() ? ffi::data_ptr<int>(output_idx_offset.value()) : nullptr,

        (uint64_t)ffi::stride(input, 0),
        return_value ? (uint64_t)ffi::stride(output_value.value(), 0) : 0,
        (uint64_t)ffi::stride(output_index, 0),

        sorted_value,
        sorted_index,
        return_value,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,

        (uint32_t)shared_mem_per_block_optin,
        stream
    };

    const uint32_t num_waves = ((uint32_t)batch_size + (uint32_t)num_sm - 1) / (uint32_t)num_sm;

    if (bfloat16_in) {
        DS_HOST_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for bfloat16 input");
        DS_HOST_CHECK(topk <= 4096, "topk must be <= 4096");
        // [MACA] 上游此处有一档 cluster 分发（`batch_size <= 6 && vocab_size >= 512K
        //   && topk <= 1024`，cluster=16，CTA 间分摊同一行配合 TMA 多播）。MACA 无
        //   cluster 也无 TMA，该变体连同 `csrc/cuda_kernels/v3_cluster/` 整体删除，
        //   bf16 一律走 normal 路径 —— 只少了"小 batch + 超长行"这一档吞吐，正确性
        //   不受影响。
        {
            INTEGER_TYPE_SWITCH(output_index, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        //   wave == 1 -> 512t / B8192 (occ1)
                        //   otherwise -> 256t / B4096 (occ1)
                        //
                        // [MACA] Upstream's tuples re-derived for MACA's 128 KiB SM (upstream
                        // sized them for 227 KiB).  The table and its arithmetic live in
                        // `scripts/generate_instantiations.py`, which generates exactly the
                        // instantiations these calls resolve against -- the two must be edited
                        // together or the mismatch is a link error, not a runtime one.
                        //
                        // `target_occupancy` is 1 everywhere: two CTAs of the smallest tuple
                        // here are 169984 B, more than any MACA SM has.
                        //
                        // No `topk > 1024` arm: `max_topk` 4096 is impossible on 128 KiB, not
                        // merely tight -- `surviving_topk_pairs` alone is 2*4096*8 = 65536 B and
                        // the extra-pairs region at least (4096 + 4096)*8 = 65536 B, leaving
                        // nothing for the input staging buffer.
                        auto dispatch = [&]<uint32_t MAX_TOPK, uint32_t B2>() {
                            if (num_waves == 1)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<maca_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 512, 1, 8192, B2, 5>>(args);
                            else if constexpr (MAX_TOPK <= 512)
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<maca_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 1, 4096, 4096, 4>>(args);
                            else
                                topk_select_bf16_normal::run_topk_select_kernel<TopkSelectConfig<maca_bfloat16, OutIdxT, false, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, 256, 1, 4096, 4096, 3>>(args);
                        };
                        if (topk <= 512) {
                            dispatch.template operator()<512, 4096>();
                        } else if (topk <= 1024) {
                            dispatch.template operator()<1024, 3584>();
                        } else {
                            DS_HOST_UNREACHABLE(
                                "the ported upstream kernel covers topk <= 1024 on this architecture: "
                                "max_topk 4096 needs 227 KiB of shared memory per SM and no MACA part has "
                                "more than 128 KiB.  Use backend=\"maca_c\", which covers topk up to 4096.");
                        }
                    });
                });
            });
        }
    } else {
        DS_HOST_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for float32 input");
        DS_HOST_CHECK(topk <= 4096, "topk must be <= 4096");

        INTEGER_TYPE_SWITCH(output_index, OutIdxT, [&]() {
            //   topk <= 512         -> 512t / B8192 / B2 2560 / TMA3
            //   topk <= 1024        -> 512t / B8192 / B2 1536 / TMA3
            //
            // [MACA] Same 128 KiB floor and the same edit-together rule as the bf16 arm.
            // B2 is lower than bf16's at the same max_topk because `tma_load_buf` holds B
            // elements of the value type (32768 B vs 16384 B) and fp32 keeps its keys 64
            // bits wide in the surviving and extra pairs regions as well.
            auto dispatch = [&]<bool SORTED_VALUE, bool SORTED_INDEX, bool RETURN_VALUE>() {
                auto launch = [&]<uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t B, uint32_t B2>() {
                    topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float, OutIdxT, SORTED_VALUE, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, NUM_THREADS, 1, B, B2, 3>>(args);
                };
                if (topk <= 512) {
                    launch.template operator()<512, 512, 8192, 2560>();
                } else if (topk <= 1024) {
                    launch.template operator()<1024, 512, 8192, 1536>();
                } else {
                    DS_HOST_UNREACHABLE(
                        "the ported upstream kernel covers topk <= 1024 on this architecture: "
                        "max_topk 4096 needs 227 KiB of shared memory per SM and no MACA part has "
                        "more than 128 KiB.  Use backend=\"maca_c\", which covers topk up to 4096.");
                }
            };
            if (sorted_value) {
                dispatch.template operator()<true, false, true>();
            } else {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        dispatch.template operator()<false, SORTED_INDEX, RETURN_VALUE>();
                    });
                });
            }
        });
    }
}

tvm::ffi::Array<int64_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
            OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

}  // namespace deep_select

// The exported symbols.  Whichever kernel tree backs it, `deep_select_xcore<N>`
// carries these two names.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, deep_select::topk)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_alignment_requirement,
                              deep_select::get_alignment_requirement)
