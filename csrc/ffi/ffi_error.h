// 2026 - Modified for DeepSelect.  Host-side error reporting for the tvm-ffi
// binding layer.
//
// The pybind11 entries used `TORCH_CHECK`, which threw a `c10::Error`.  The
// tvm-ffi boundary wants a plain C++ exception: `tvm::ffi::Error` is what the
// runtime turns into a python exception, and constructing one through the
// `TVMFFIErrorSetRaisedFromCStr` env API is the documented way for a host
// function to report a failure that carries a useful message.
//
// Two levels, mirroring `deep_gemm`'s `DG_HOST_ASSERT` /
// `DG_CUDA_RUNTIME_CHECK` pair:
//
//   DS_HOST_ASSERT(cond)            a contract violation the caller caused
//   DS_CUDA_RUNTIME_CHECK(cmd)      a CUDA runtime call that returned != success
//   DS_HOST_UNREACHABLE(reason)     a case the dispatch cannot represent
//
// The message carries `__FILE_NAME__:__LINE__` -- the file, not the path.
// torch passes sources to the compiler as absolute paths, so `__FILE__` would
// put *this build machine's* checkout directory into an error a caller reads;
// mxcc accepts no `-ffile-prefix-map` to strip it (measured: rejected both
// bare and through `-Xclang`), and the basename is what the diagnostic needs
// anyway.

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

// The operands of a check are a comma-separated *list* at most call sites
// (`DS_HOST_CHECK(cond, what, ".stride(1) must be 1")`), so they reach the
// stream through this rather than through chained `<<`.  Chained, they parse
// as a comma *operator*: the whole `<<` chain becomes the discarded left
// operand, so everything past the first argument is dropped from the message
// and `-Wunused-value` fires once per remaining argument.
template <typename... Args>
std::string concat(const Args &...args) {
    std::ostringstream oss;
    (oss << ... << args);
    return oss.str();
}

}  // namespace deep_select::ffi

#define DS_RAISE_AT(file, line, ...)                                            \
    ::deep_select::ffi::raise(::deep_select::ffi::concat(                       \
        (file), ":", (line), ": ", __VA_ARGS__))

// A caller-visible contract violation.  Every use carries the condition's own
// spelling so the message says what was required, not just that something was
// wrong -- `TORCH_CHECK(cond, "msg")` gave both and callers read them.
#define DS_HOST_ASSERT(cond)                                                    \
    do {                                                                        \
        if (!(cond)) {                                                          \
            DS_RAISE_AT(__FILE_NAME__, __LINE__, "assertion failed: " #cond);   \
        }                                                                       \
    } while (0)

#define DS_HOST_CHECK(cond, ...)                                                \
    do {                                                                        \
        if (!(cond)) {                                                          \
            DS_RAISE_AT(__FILE_NAME__, __LINE__, __VA_ARGS__);                  \
        }                                                                       \
    } while (0)

#define DS_HOST_UNREACHABLE(...)                                                \
    DS_RAISE_AT(__FILE_NAME__, __LINE__, __VA_ARGS__)

#define DS_CUDA_RUNTIME_CHECK(cmd)                                              \
    do {                                                                        \
        const cudaError_t _ds_err = (cmd);                                      \
        if (_ds_err != cudaSuccess) {                                           \
            DS_RAISE_AT(__FILE_NAME__, __LINE__, #cmd " failed: ",              \
                        cudaGetErrorString(_ds_err));                           \
        }                                                                       \
    } while (0)
