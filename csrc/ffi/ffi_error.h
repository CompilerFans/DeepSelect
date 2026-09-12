// 2026 - Modified for DeepSelect.  Host-side error reporting for the tvm-ffi
// binding layer.
//
// The pybind11 entries used `TORCH_CHECK`, which threw a `c10::Error`.  The
// tvm-ffi boundary wants a plain C++ exception: `tvm::ffi::Error` is what the
// runtime turns into a python exception, and constructing one through the
// `TVMFFIErrorSetRaisedFromCStr` env API is the documented way for a host
// function to report a failure that carries a useful message.
//
// Two levels, mirroring the host repository's `DG_HOST_ASSERT` /
// `DG_CUDA_RUNTIME_CHECK` pair:
//
//   DS_HOST_ASSERT(cond)            a contract violation the caller caused
//   DS_CUDA_RUNTIME_CHECK(cmd)      a CUDA runtime call that returned != success
//   DS_HOST_UNREACHABLE(reason)     a case the dispatch cannot represent
//
// The message carries `__FILE__:__LINE__` deliberately: that is the
// diagnostic the pybind11 build got from c10's `TORCH_CHECK`, and losing it
// would make a rejected call noticeably harder to place.

#pragma once

#include <cuda_runtime.h>
#include <tvm/ffi/error.h>

#include <sstream>
#include <string>

namespace deep_select::ffi {

// `tvm::ffi::Error`'s constructor is (kind, message, backtrace) and it copies
// all three, so temporaries are fine.  "Internal" is the honest kind: these
// are contract violations from the caller's point of view, but tvm-ffi has no
// caller-error kind and inventing one would not change how it surfaces (a
// python exception carrying the message string).
inline void raise(const std::string &message) {
    throw tvm::ffi::Error("Internal", message, "");
}

}  // namespace deep_select::ffi

#define DS_RAISE_AT(file, line, ...)                                            \
    do {                                                                        \
        std::ostringstream _ds_oss;                                             \
        _ds_oss << (file) << ":" << (line) << ": " << __VA_ARGS__;              \
        ::deep_select::ffi::raise(_ds_oss.str());                               \
    } while (0)

// A caller-visible contract violation.  Every use carries the condition's own
// spelling so the message says what was required, not just that something was
// wrong -- `TORCH_CHECK(cond, "msg")` gave both and callers read them.
#define DS_HOST_ASSERT(cond)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            DS_RAISE_AT(__FILE__, __LINE__, "assertion failed: " #cond);        \
        }                                                                       \
    } while (0)

#define DS_HOST_CHECK(cond, ...)                                                \
    do {                                                                        \
        if (!(cond)) {                                                          \
            DS_RAISE_AT(__FILE__, __LINE__, __VA_ARGS__);                       \
        }                                                                       \
    } while (0)

#define DS_HOST_UNREACHABLE(...)                                                \
    DS_RAISE_AT(__FILE__, __LINE__, __VA_ARGS__)

#define DS_CUDA_RUNTIME_CHECK(cmd)                                              \
    do {                                                                        \
        const cudaError_t _ds_err = (cmd);                                      \
        if (_ds_err != cudaSuccess) {                                           \
            DS_RAISE_AT(__FILE__, __LINE__, #cmd " failed: ",                   \
                        cudaGetErrorString(_ds_err));                           \
        }                                                                       \
    } while (0)
