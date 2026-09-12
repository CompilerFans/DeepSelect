// 2026 - Modified for DeepSelect.  The tvm-ffi entry points, one function per
// architecture extension.
//
// This is the torch-free replacement for the pybind11 `topk` entry that used
// to live at the bottom of `maca_topk.cu`.  The signature is all-DLTensor and
// all-positional (this tvm-ffi build has no kwargs or defaults), and the
// output buffers are caller-allocated -- the FFI layer cannot return new
// tensors, and a python-side `torch.empty` costs the same as the torch-side
// one it replaces.
//
// Contract, exactly as the pybind11 entry enforced it (`TORCH_CHECK`, now
// `DS_HOST_ASSERT`):
//
//   input              (b, v)        float32 or bfloat16, stride(1) == 1,
//                                    stride(0) * element_size % 1024 == 0
//   output_index       (b, >= topk)  int32 or int64, stride(1) == 1
//   output_value       (b, >= topk)  same dtype as input, stride(1) == 1
//   end                (b,) contig   int32, optional
//   output_idx_offset  (b,) contig   int32, optional
//
// `begin` and `hint` are not accepted: the public interface rejects them
// before this layer is reached (`deep_select/interface.py`).

#pragma once

#include "ffi_tensor.h"

#include <tvm/ffi/container/array.h>
#include <tvm/ffi/container/tensor.h>
#include <tvm/ffi/optional.h>

#include <cstdint>

namespace deep_select {

// What `deep_select/interface.py` reads to size its output rows.  Two int64s
// in an `Array<int64_t>`: tvm-ffi marshals that as a python list, and `std::pair`
// is not a type it knows how to carry across the boundary (it fails to
// instantiate `TypeTraits`).  The old pybind11 entry returned a C++ `std::pair`
// and pybind converted it; that convenience does not survive the FFI edge.
tvm::ffi::Array<int64_t> get_alignment_requirement();

// One row-wise top-K over a 2-D input.  `idx_oob_fill_value` /
// `value_oob_fill_value` fill the slots a row shorter than `topk` leaves
// over; `abort_when_nan_found` chooses trap-on-NaN or the `0x3F3F3F3F`
// guard in `output_index[row, 0]`.
void topk(const tvm::ffi::TensorView &input, int64_t topk,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &end,
          bool sorted_value, bool sorted_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_value,
          const tvm::ffi::TensorView &output_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_idx_offset,
          int64_t idx_oob_fill_value, double value_oob_fill_value,
          bool return_value, bool abort_when_nan_found);

}  // namespace deep_select

// ── the exported symbols ───────────────────────────────────────────────────
// Each architecture extension compiles this header, so the same two names
// exist in every `deep_select_xcore<N>` artifact -- which is the point: the
// runtime resolves a backend by extension name, and a caller cannot tell
// which kernel tree backed it.
TVM_FFI_DLL_EXPORT_TYPED_FUNC(topk, deep_select::topk)
TVM_FFI_DLL_EXPORT_TYPED_FUNC(get_alignment_requirement,
                              deep_select::get_alignment_requirement)
