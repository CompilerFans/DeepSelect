/*
 * Adapted from
 * https://github.com/pytorch/pytorch/blob/v2.0.1/aten/src/ATen/Dispatch.h
 *
 * 2026 - Modified for DeepSelect: torch-free.  The switches read DLPack dtypes
 * through `ffi::same_dtype` and their failure arms go through
 * `DS_HOST_UNREACHABLE`.  Those arms are unreachable from `api.cu` today (the
 * entry checks every dtype first) but stay spelled out so a new call site
 * cannot silently fall off the end.
 */
#pragma once

#include "ffi_error.h"
#include "ffi_tensor.h"

#include <cstdint>

#define BOOL_SWITCH(COND, CONST_NAME, ...)      \
  [&] {                                         \
    if (COND) {                                 \
      constexpr static bool CONST_NAME = true;  \
      return __VA_ARGS__();                     \
    } else {                                    \
      constexpr static bool CONST_NAME = false; \
      return __VA_ARGS__();                     \
    }                                           \
  }()

#define INTEGER_TYPE_SWITCH(tensor, INDEX_TYPE, ...)                    \
  [&] {                                                                 \
    if (::deep_select::ffi::same_dtype((tensor).dtype(),                \
                                       ::deep_select::ffi::kInt64)) {   \
      using INDEX_TYPE = int64_t;                                       \
      return __VA_ARGS__();                                             \
    } else if (::deep_select::ffi::same_dtype(                          \
                   (tensor).dtype(), ::deep_select::ffi::kInt32)) {     \
      using INDEX_TYPE = int32_t;                                       \
      return __VA_ARGS__();                                             \
    } else {                                                            \
      DS_HOST_UNREACHABLE("Unsupported integer dtype for `" #tensor     \
                          "` (must be int32 or int64)");                \
    }                                                                   \
  }()

#define FLOATING_TYPE_SWITCH(tensor, FP_TYPE, ...)                      \
  [&] {                                                                 \
    if (::deep_select::ffi::is_float32(tensor)) {                       \
      using FP_TYPE = float;                                            \
      return __VA_ARGS__();                                             \
    } else if (::deep_select::ffi::is_bfloat16(tensor)) {               \
      using FP_TYPE = maca_bfloat16;                                    \
      return __VA_ARGS__();                                             \
    } else {                                                            \
      DS_HOST_UNREACHABLE("Unsupported floating point dtype for `"      \
                          #tensor "` (must be float32 or bfloat16)");   \
    }                                                                   \
  }()

// Kept although nothing reaches it: the cluster variant is deleted on MACA (no
// cluster launch, no distributed shared memory).  A dispatch arm still selecting
// on `cluster_size` finds this rather than an undefined-macro compile error.
#define CLUSTER_SIZE_SWITCH(cluster_size, ...)  \
  [&] {                                         \
    if (cluster_size == 1) {                    \
      constexpr static int CLUSTER_SIZE = 1;    \
      return __VA_ARGS__();                     \
    } else {                                    \
      DS_HOST_UNREACHABLE("Unsupported cluster_size: MACA has no cluster "  \
                          "launch; the cluster variant was deleted");       \
    }                                           \
  }()
