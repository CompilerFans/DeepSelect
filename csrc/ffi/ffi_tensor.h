// 2026 - Modified for DeepSelect.  DLTensor edge helpers for the tvm-ffi
// binding layer, modelled on mcDeepGEMM's
// `csrc/utils/ffi_tensor.hpp` (branch `dev_tvm_ffi`, commit
// 3a6e6ba3) and pared to what this tree uses.
//
// The extension is torch-free: tensors cross the FFI boundary as
// `tvm::ffi::TensorView` (the non-owning form the upstream tvm-ffi
// kernel-library guide recommends for kernel parameters), and every method
// call the old pybind11 layer made on `torch::Tensor` is reproduced here
// against the DLTensor payload.
//
// Why this exists at all: the pybind11 entry points linked libtorch/libc10,
// so the extension carried six torch DT_NEEDED entries and its behavior was
// tied to whichever torch built it (the c10_cuda_check_implementation trap).
// A DLPack boundary at TensorView removes that: the .so no longer links
// torch, and the DWARF line number in its error messages -- the only other
// version-dependent thing in it -- goes with it.

#pragma once

#include "ffi_error.h"

#include <tvm/ffi/container/tensor.h>

#include <dlpack/dlpack.h>

#include <cstdint>

namespace deep_select::ffi {

// ── dtype predicates ────────────────────────────────────────────────────────
// `at::kFloat` / `at::kBFloat16` / `at::kInt` / `at::kLong`, in DLPack terms.
constexpr DLDataType kFloat32{kDLFloat, 32, 1};
constexpr DLDataType kBFloat16{kDLBfloat, 16, 1};
constexpr DLDataType kInt32{kDLInt, 32, 1};
constexpr DLDataType kInt64{kDLInt, 64, 1};
constexpr DLDataType kUInt8{kDLUInt, 8, 1};

inline bool same_dtype(const DLDataType &a, const DLDataType &b) {
    return a.code == b.code && a.bits == b.bits && a.lanes == b.lanes;
}

inline bool is_float32(const tvm::ffi::TensorView &t) {
    return same_dtype(t.dtype(), kFloat32);
}
inline bool is_bfloat16(const tvm::ffi::TensorView &t) {
    return same_dtype(t.dtype(), kBFloat16);
}
inline bool is_index_type(const tvm::ffi::TensorView &t) {
    return same_dtype(t.dtype(), kInt32) || same_dtype(t.dtype(), kInt64);
}

// ── extraction helpers, mirroring the Tensor methods the impls used ─────────

inline int64_t size(const tvm::ffi::TensorView &t, int dim) {
    DS_HOST_ASSERT(dim >= 0 && dim < t.ndim());
    return t.size(dim);
}

inline int64_t stride(const tvm::ffi::TensorView &t, int dim) {
    DS_HOST_ASSERT(dim >= 0 && dim < t.ndim());
    return t.stride(dim);
}

inline uint32_t element_size(const tvm::ffi::TensorView &t) {
    // DLPack bits are per element and this tree is bytes-only (no vector
    // dtypes), so the byte size is bits/8 -- what `Tensor::element_size()`
    // returned.
    DS_HOST_ASSERT(t.dtype().lanes == 1);
    return (uint32_t)(t.dtype().bits / 8);
}

inline bool is_contiguous(const tvm::ffi::TensorView &t) {
    return t.IsContiguous();
}

inline int device_index(const tvm::ffi::TensorView &t) {
    return t.device().device_id;
}

inline DLDeviceType device_type(const tvm::ffi::TensorView &t) {
    return t.device().device_type;
}

template <typename T>
inline T *data_ptr(const tvm::ffi::TensorView &t) {
    return reinterpret_cast<T *>(t.data_ptr());
}

inline void *data_ptr(const tvm::ffi::TensorView &t) {
    return t.data_ptr();
}

// The kernel layer indexes the *raw* bytes, sized by dtype alone.  A
// `data_ptr<T>` cast is the same address; this names the intent.
inline const void *const_data_ptr(const tvm::ffi::TensorView &t) {
    return t.data_ptr();
}

}  // namespace deep_select::ffi
