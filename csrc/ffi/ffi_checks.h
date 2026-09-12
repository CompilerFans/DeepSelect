// 2026 - Modified for DeepSelect.  The torch-free replacements for the
// `KU_CHECK_*` family the ported kernel's host layer used
// (`kerutils/supplemental/torch_tensors.h`, which is torch-only).
//
// Each macro keeps its old name and its old message spelling, so the call
// sites in `api.cu` read the same and a rejected call still says which tensor
// and what was wrong.  The `_opt` forms accept a `tvm::ffi::Optional` and are
// a no-op when it holds nothing -- that is what `_check_optional_tensor` meant.

#pragma once

#include "ffi_error.h"
#include "ffi_tensor.h"

#include <tvm/ffi/optional.h>

namespace deep_select::ffi {

// Shape as a compile-time list, so KU_CHECK_SHAPE(t, b, v) can compare
// against it without the torch::IntArrayRef the old macro built.
inline bool shape_is(const tvm::ffi::TensorView &t, std::initializer_list<int64_t> dims) {
    if ((size_t)t.ndim() != dims.size()) return false;
    size_t i = 0;
    for (int64_t d : dims) {
        if (t.size((int)i) != d) return false;
        ++i;
    }
    return true;
}

// Optional forms: an absent optional satisfies the check (that is what
// `_check_optional_tensor` did).
inline bool device_ok(const tvm::ffi::Optional<tvm::ffi::TensorView> &t) {
    return !t.has_value() || t.value().device().device_type == kDLCUDA;
}

}  // namespace deep_select::ffi

// `#tensor` in the old macros spelled the argument; these keep that.
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
