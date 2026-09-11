// This file contains only the host-side topk() function and pybind11 module.
// Kernel template instantiations are in separate files for parallel compilation.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>

#include "kerutils/supplemental/torch_tensors.h"

#include "dispatch_utils.h"

#include "cuda_kernels/config.h"
#include "cuda_kernels/v3/topk_select.h"
#include "cuda_kernels/v3_fp32/topk_select.h"
#include <cstdlib>
// [MACA] 配合下方 `check_dim0_stride` 里把 `std::format` 换成 `snprintf`：
//   宿主是 GCC 11.4，`<format>` 不存在（libstdc++ 到 GCC 13 才有），
//   所以这里不能用 `#include <format>` 解决。
#include <cstdio>
// [MACA] 原为 `#include "cuda_kernels/v3_cluster/topk_select.h"`。cluster 变体
//   （CTA 协同寻址 + TMA 多播）在 MACA 上整体删除，见下方 bf16 分发的说明。


void topk(
    torch::Tensor &input,
    int topk,
    c10::optional<torch::Tensor> &begin,
    c10::optional<torch::Tensor> &end,
    bool sorted_value,
    bool sorted_index,
    c10::optional<torch::Tensor> &output_value,
    torch::Tensor &output_index,
    c10::optional<torch::Tensor> &output_idx_offset,
    int idx_oob_fill_value,
    float value_oob_fill_value,
    bool return_value,
    bool abort_when_nan_found
) {
    int batch_size = input.size(0);
    int vocab_size = input.size(1);
    at::ScalarType value_t = input.scalar_type();
    at::ScalarType output_index_t = output_index.scalar_type();

    TORCH_CHECK(topk > 0, "topk must > 0");
    TORCH_CHECK(!(sorted_value && !return_value), "`return_value` must be enabled when `sorted_value` is True");
    TORCH_CHECK(!(sorted_value && sorted_index), "`sorted_value` and `sorted_index` cannot be used at the same time");
    // Contract: sorted_value is a 32-bit-value-only feature.
    TORCH_CHECK(!(sorted_value && value_t == at::kBFloat16), "`sorted_value` is only supported for float32 input");
    TORCH_CHECK(!begin.has_value(), "`begin` is not supported currently");
    if (return_value) {
        TORCH_CHECK(output_value.has_value(), "`output_value` must not be `None` when `return_value` is True");
    }
    
    KU_CHECK_DEVICE(input);
    KU_CHECK_DEVICE(begin);
    KU_CHECK_DEVICE(end);
    KU_CHECK_DEVICE(output_value);
    KU_CHECK_DEVICE(output_index);
    KU_CHECK_DEVICE(output_idx_offset);
    
    KU_CHECK_SHAPE(input, batch_size, vocab_size);
    KU_CHECK_SHAPE(begin, batch_size);
    KU_CHECK_SHAPE(end, batch_size);
    KU_CHECK_SHAPE(output_value, batch_size, topk);
    KU_CHECK_SHAPE(output_index, batch_size, topk);
    KU_CHECK_SHAPE(output_idx_offset, batch_size);
    
    KU_CHECK_DTYPE(input, value_t);
    KU_CHECK_DTYPE(begin, at::kInt);
    KU_CHECK_DTYPE(end, at::kInt);
    KU_CHECK_DTYPE(output_value, value_t);
    KU_CHECK_DTYPE(output_index, output_index_t);
    KU_CHECK_DTYPE(output_idx_offset, at::kInt);
    
    KU_CHECK_LAST_DIM_CONTIGUOUS(input);
    KU_CHECK_CONTIGUOUS(begin);
    KU_CHECK_CONTIGUOUS(end);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_value);
    KU_CHECK_LAST_DIM_CONTIGUOUS(output_index);
    KU_CHECK_CONTIGUOUS(output_idx_offset);

    auto check_dim0_stride = [&](const char tensor_name[], torch::Tensor &tensor, uint32_t alignment_requirement_bytes) {
        int64_t cur_stride = tensor.stride(0);
        uint64_t itemsize = tensor.dtype().itemsize();
        // [MACA] 原为 `std::format("{}.stride(0) (currently {} numbers) ...", ...)`。
        //   `std::format` 要 C++20 的 libstdc++（GCC >= 13），本机宿主是 GCC 11.4，
        //   `<format>` 根本不存在（是宿主编译器能力问题，不是传递引入问题）。
        //   改用 snprintf：不引入新依赖（fmt 是外层仓的 submodule，本仓不该依赖它），
        //   且报文文本与 std::format 逐字符相同。
        char msg[256];
        std::snprintf(msg, sizeof(msg),
            "%s.stride(0) (currently %lld numbers) must be a multiple of %u Bytes (%llu numbers)",
            tensor_name, (long long)cur_stride,
            (unsigned)alignment_requirement_bytes,
            (unsigned long long)(alignment_requirement_bytes / itemsize));
        TORCH_CHECK(cur_stride * itemsize % alignment_requirement_bytes == 0, msg);
    };
    check_dim0_stride("input", input, INPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    check_dim0_stride("output_index", output_index, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    if (output_value.has_value()) {
        check_dim0_stride("value", *output_value, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT);
    }

    cudaDeviceProp* device_prop = at::cuda::getDeviceProperties(at::cuda::current_device());
    TORCH_CHECK(device_prop != nullptr);
    TopkSelectArgs args = {
        (uint32_t)batch_size,
        (uint32_t)vocab_size,
        (uint32_t)topk,

        input.data_ptr(),
        ku::get_optional_tensor_ptr<void>(output_value),
        output_index.data_ptr(),
        ku::get_optional_tensor_ptr<int>(begin),
        ku::get_optional_tensor_ptr<int>(end),
        ku::get_optional_tensor_ptr<int>(output_idx_offset),

        (uint64_t)input.stride(0),
        output_value.has_value() ? (uint64_t)output_value->stride(0) : 0,
        (uint64_t)output_index.stride(0),

        sorted_value,
        sorted_index,
        return_value,
        idx_oob_fill_value,
        value_oob_fill_value,
        abort_when_nan_found,

        device_prop->sharedMemPerBlockOptin,
        at::cuda::getCurrentCUDAStream().stream()
    };

    uint32_t num_sm = device_prop->multiProcessorCount;
    uint32_t num_waves = (batch_size + num_sm-1) / num_sm;

    TORCH_CHECK(value_t == at::kBFloat16 || value_t == at::kFloat, "input dtype must be bfloat16 or float32");
    if (value_t == at::kBFloat16) {
        TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for bfloat16 input");
        TORCH_CHECK(topk <= 4096, "topk must be <= 4096");
        // [MACA] 此处原为小 batch 专用的一档 cluster 分发：
        //   `batch_size <= 6 && vocab_size >= 512K && topk <= 1024` →
        //   `topk_select_bf16_cluster`（cluster=16，CTA 间分摊同一行、协同寻址，
        //   配合 TMA 多播把长行的读带宽摊到多个 CTA）。
        //   MACA 无 cluster（无 cluster 维、无分布式共享内存、无 cluster 屏障）也无
        //   TMA，该变体连同 `csrc/cuda_kernels/v3_cluster/` 整体删除；
        //   bf16 一律走 normal 路径。少掉的只是"小 batch + 超长行"这一档的吞吐，
        //   正确性覆盖不受影响（normal 路径对该 shape 本就正确，只是慢）。
        {
            INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
                BOOL_SWITCH(sorted_index, SORTED_INDEX, [&]() {
                    BOOL_SWITCH(return_value, RETURN_VALUE, [&]() {
                        //   wave == 1 -> 512t / B8192 (occ1)
                        //   otherwise -> 256t / B4096 (occ1)
                        //
                        // [MACA] These are upstream's tuples re-derived for the 128 KiB a MACA
                        // SM has, not the 227 KiB upstream sized them for; see
                        // `scripts/generate_instantiations.py`, which is where the table lives
                        // and where the arithmetic is recorded, and which generates exactly the
                        // set of instantiations these five `run_topk_select_kernel` calls
                        // resolve against.  The two must be edited together -- a mismatch is a
                        // link error, not a runtime one.
                        //
                        // `target_occupancy` is 1 everywhere: two CTAs of the smallest tuple
                        // here are 169984 B, more than any MACA SM has.
                        //
                        // There is no `topk > 1024` arm.  `max_topk` 4096 is not merely a tight
                        // fit on 128 KiB, it is impossible: `surviving_topk_pairs` is
                        // 2 * 4096 * 8 = 65536 B and the extra-pairs region is at least
                        // (4096 + 4096) * 8 = 65536 B, so those two members fill the SM with
                        // nothing left for the input staging buffer.  Upstream carries that
                        // tuple because 227 KiB has room; this port cannot, and says so below
                        // rather than launching something that would not fit.
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
                            TORCH_CHECK(false,
                                "the ported upstream kernel covers topk <= 1024 on this architecture: "
                                "max_topk 4096 needs 227 KiB of shared memory per SM and no MACA part has "
                                "more than 128 KiB.  Use backend=\"maca_c\", which covers topk up to 4096.");
                        }
                    });
                });
            });
        }
    } else {
        TORCH_CHECK((uint32_t)vocab_size < MAX_VOCAB_SIZE,
                    "vocab_size must be < 2^23 for float32 input");
        TORCH_CHECK(topk <= 4096, "topk must be <= 4096");

        INTEGER_TYPE_SWITCH(output_index_t, OutIdxT, [&]() {
            //   topk <= 512         -> 512t / B8192 / B2 2560 / TMA3
            //   topk <= 1024        -> 512t / B8192 / B2 1536 / TMA3
            //
            // [MACA] Same 128 KiB floor, and the same edit, as the bf16 arm above; the table
            // and its arithmetic live in `scripts/generate_instantiations.py`.  B2 comes out
            // lower than bf16's at the same max_topk because `tma_load_buf` holds B elements
            // of the value type -- 32768 B here against 16384 B -- and fp32 keeps its keys 64
            // bits wide in the surviving and extra pairs regions too.  The `topk > 1024` arm
            // is gone for the reason given there: max_topk 4096 cannot fit 128 KiB.
            auto dispatch = [&]<bool SORTED_VALUE, bool SORTED_INDEX, bool RETURN_VALUE>() {
                auto launch = [&]<uint32_t MAX_TOPK, uint32_t NUM_THREADS, uint32_t B, uint32_t B2>() {
                    topk_select_fp32::run_topk_select_kernel<TopkSelectConfig<float, OutIdxT, SORTED_VALUE, SORTED_INDEX, RETURN_VALUE, MAX_TOPK, NUM_THREADS, 1, B, B2, 3>>(args);
                };
                if (topk <= 512) {
                    launch.template operator()<512, 512, 8192, 2560>();
                } else if (topk <= 1024) {
                    launch.template operator()<1024, 512, 8192, 1536>();
                } else {
                    TORCH_CHECK(false,
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

std::pair<uint32_t, uint32_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT, OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};

}
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("topk", &topk);
    m.def("get_alignment_requirement", &get_alignment_requirement);
}
