#pragma once

namespace kerutils {}

#define KU_PRINTLN(fmt, ...) { cute::print(fmt, ##__VA_ARGS__); print("\n"); }

namespace ku = kerutils;

#ifdef __CUDACC__
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

// [MACA] mxcc 编译 CUDA 源时既不定义 __CUDACC__，也没有 __CUDA_ARCH__，
// 所以上面那支不成立。用 __MACA__ 承接：对 kerutils 而言「有设备码要编」
// 这件事与 CUDA 构建同构，宏语义保持不变。
// 用 __MACA__ 而不是 __MACA_ARCH__：mxcc 只在**设备**遍定义 __MACA_ARCH__，
// 而本文件也要在**宿主**遍成立（host/host.h 里 launch_kernel 等整块都在这个
// 宏的 #ifdef 之下，宿主遍不定义它就只剩一个 #error）。本机用编译探针实测：
// mxcc 的宿主遍与设备遍都定义 __MACA__。
// （Hopper/Ampere 专有的 TMA 类型与 sm80/sm90/sm100 内联汇编已在各自文件里
//   直接删除，不需要额外的平台标记。）
#if defined(__MACA__) || defined(__MACA_ARCH__)
#define KERUTILS_IS_BUILD_ON_CUDA
#endif

#ifndef KERUTILS_IS_BUILD_ON_CUDA
#error "KERUTILS_IS_BUILD_ON_CUDA must be defined. It is defined automatically when compiling with NVCC; pass `-DKERUTILS_IS_BUILD_ON_CUDA` when compiling with a host compiler."
#endif
