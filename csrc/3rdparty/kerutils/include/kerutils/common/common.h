#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

// ---------------------------------------------------------------------------
// Platform selection.  Each arm below keys on something its toolchain supplies
// by itself -- an `__has_include` probe or a predefined macro -- and never on a
// macro a build system would have to pass, so a vendored copy of this library
// keeps working in a translation unit that was given no extra flags.
//
// **The MACA test has to come first, and that ordering is load-bearing.**
// `cucc` -- the driver torch uses, and the only compiler this repository
// builds with -- passes `-imacros __macro_mxcc.h` to every translation unit,
// and that header defines both `__NVCC__` and `__CUDACC__`.  They are the
// CUDA-dialect adapter, not a statement about the platform: torch's c10
// headers and this file both branch on them, which is why the adapter supplies
// them.  So on MACA `__CUDACC__` *is* defined, and testing it first would make
// both arms fire.  (They did, and the mutual-exclusion check below is what
// reported it.)  Testing `__MACA__` first says the true thing: the platform is
// MACA and the CUDA spelling is an alias this toolchain synthesizes.
//
// Measured on MACA 3.7.0 with a `cucc -std=c++20 --offload-arch=xcore1000`
// probe, in the pass that emits host code: `__MACA__` and `__MACACC__`
// defined, `__CUDACC__` and `__NVCC__` defined (by the adapter header above),
// `__CUDA_ARCH__` and `__MACA_ARCH__` *not* defined.  `__MACA__` being present
// in the host pass is what this file needs, because `host/host.h` puts
// `launch_kernel` and its neighbours under the platform macro; a key that only
// the device pass defined would leave the host pass with nothing but the
// `#error` below.
// ---------------------------------------------------------------------------

#ifdef __MACA__
#define KERUTILS_IS_BUILD_ON_MACA
#elif defined(__CUDACC__)
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#if __has_include("kernel_operator.h")
#define KERUTILS_IS_BUILD_ON_ASCEND
#endif

// Pairwise, and in upstream's nesting shape so its own CUDA x ASCEND check
// reads unchanged.
#ifdef KERUTILS_IS_BUILD_ON_CUDA
#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#error "KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_ASCEND is defined at the same time!"
#endif
#ifdef KERUTILS_IS_BUILD_ON_MACA
#error "KERUTILS_IS_BUILD_ON_CUDA and KERUTILS_IS_BUILD_ON_MACA is defined at the same time!"
#endif
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#ifdef KERUTILS_IS_BUILD_ON_MACA
#error "KERUTILS_IS_BUILD_ON_ASCEND and KERUTILS_IS_BUILD_ON_MACA is defined at the same time!"
#endif
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#ifndef KERUTILS_IS_BUILD_ON_ASCEND
#ifndef KERUTILS_IS_BUILD_ON_MACA
#error "None of KERUTILS_IS_BUILD_ON_CUDA / KERUTILS_IS_BUILD_ON_ASCEND / KERUTILS_IS_BUILD_ON_MACA is defined!"
#endif
#endif
#endif

namespace kerutils {

// Both are called from **device** code -- the ported kernels under
// `csrc/maca_kernels/xcore1600/` size their tiles with them -- so both carry
// `__host__ __device__`.  That is a deliberate departure from upstream's
// plain `inline constexpr`: the NVCC extension that makes a constexpr function
// callable from device code without the attribute is `--expt-relaxed-constexpr`,
// and mxcc does not guarantee it.  (The CUDA arm is unaffected either way; the
// attribute is a superset.)
template<typename T>
inline __host__ __device__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __host__ __device__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

}
