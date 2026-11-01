// The torch-free host-layer checks.
//
// These are the spellings every call site below uses.  The validation
// semantics and messages are what a caller sees, so a rejected call still says
// which tensor and what was wrong.  The `_opt` forms accept a
// `tvm::ffi::Optional` and are a no-op when it holds nothing.

#pragma once

#include "ffi_error.h"
#include "ffi_tensor.h"

#include <tvm/ffi/optional.h>

namespace deep_select::ffi {

// Shape as a compile-time list, so `DS_CHECK_SHAPE(t, b, v)` can compare
// against it without building a torch::IntArrayRef.
inline bool shape_is(const tvm::ffi::TensorView &t, std::initializer_list<int64_t> dims) {
    if ((size_t)t.ndim() != dims.size()) return false;
    size_t i = 0;
    for (int64_t d : dims) {
        if (t.size((int)i) != d) return false;
        ++i;
    }
    return true;
}

}  // namespace deep_select::ffi

// `#tensor` spells the argument, so an error names the operand the caller
// passed rather than a position in the call.
#define DS_CHECK_DEVICE(t)                                                      \
    do {                                                                        \
        DS_HOST_CHECK(t.device().device_type == kDLCUDA,                        \
                      #t " must be on CUDA");                                   \
    } while (0)

#define DS_CHECK_SHAPE(t, ...)                                                  \
    do {                                                                        \
        DS_HOST_CHECK(::deep_select::ffi::shape_is((t), {__VA_ARGS__}),         \
                      #t " must have shape (" #__VA_ARGS__ ")");                \
    } while (0)

#define DS_CHECK_CONTIGUOUS(t)                                                  \
    do {                                                                        \
        DS_HOST_CHECK(::deep_select::ffi::is_contiguous(t),                     \
                      #t " must be contiguous");                                \
    } while (0)

#define DS_CHECK_LAST_DIM_CONTIGUOUS(t)                                         \
    do {                                                                        \
        DS_HOST_CHECK((t).ndim() == 0 || ::deep_select::ffi::size((t), (t).ndim() - 1) == 1 || \
                          ::deep_select::ffi::stride((t), (t).ndim() - 1) == 1, \
                      #t " must have contiguous last dimension");               \
    } while (0)

#define DS_CHECK_DTYPE(t, dl_dtype)                                             \
    do {                                                                        \
        DS_HOST_CHECK(::deep_select::ffi::same_dtype((t).dtype(), (dl_dtype)),  \
                      #t " must have dtype " #dl_dtype);                        \
    } while (0)
