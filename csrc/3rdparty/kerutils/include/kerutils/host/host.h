#pragma once

#include <cstdio>
#include <exception>
#include <memory>
#include <string>
#include <utility>
#include <sstream>
#include <vector>

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include <cuda_runtime_api.h>
#include <cudaTypedefs.h>
#endif

#include "kerutils/common/common.h"

namespace kerutils {

class KUException final : public std::exception {
    std::string message = {};

public:
    template<typename... Args>
    explicit KUException(const char *name, const char* file, const int line, Args&&... args) {
        std::ostringstream oss;
        
        oss << name << " error (" << file << ":" << line << "): ";
        (oss << ... << args);
        message = oss.str();
    }

    const char *what() const noexcept override {
        return message.c_str();
    }
};

#define THROW_KU_EXCEPTION(name, ...) \
    throw kerutils::KUException(name, __FILE__, __LINE__, __VA_ARGS__)

// This `KU_ASSERT` is triggered no matter if the code is compiled with `-DNDEBUG` or not.
#define KU_ASSERT(cond, ...)                                                                     \
    do {                                                                                         \
        if (not (cond)) {                                                                        \
            char _ku_buf[1024];                                                                  \
            int _ku_len = snprintf(_ku_buf, sizeof(_ku_buf),                                    \
                                   "Assertion `%s` failed (%s:%d)", #cond, __FILE__, __LINE__); \
            __VA_OPT__(                                                                          \
                _ku_len += snprintf(_ku_buf + _ku_len, sizeof(_ku_buf) - _ku_len,              \
                                    ": " __VA_ARGS__);                                          \
            )                                                                                    \
            fprintf(stderr, "%s\n", _ku_buf);                                                    \
            THROW_KU_EXCEPTION("Assertion", _ku_buf);                                            \
        }                                                                                        \
    } while(0)

#ifdef KERUTILS_IS_BUILD_ON_CUDA

#define KU_CUDA_CHECK(call)                                                                                   \
do {                                                                                                          \
    cudaError_t status_ = (call);                                                                             \
    if (status_ != cudaSuccess) {                                                                             \
        char _ku_buf[1024];                                                                                   \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUDA error (%s:%d): %s", __FILE__, __LINE__, cudaGetErrorString(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                    \
        THROW_KU_EXCEPTION("CUDA", _ku_buf);                                                                  \
    }                                                                                                         \
} while(0)

#define KU_CUTLASS_CHECK(call)                                                                                   \
do {                                                                                                             \
    cutlass::Status status_ = (call);                                                                            \
    if (status_ != cutlass::Status::kSuccess) {                                                                 \
        char _ku_buf[1024];                                                                                      \
        snprintf(_ku_buf, sizeof(_ku_buf), "CUTLASS error (%s:%d): %d", __FILE__, __LINE__, static_cast<int>(status_)); \
        fprintf(stderr, "%s\n", _ku_buf);                                                                       \
        THROW_KU_EXCEPTION("CUTLASS", _ku_buf);                                                                 \
    }                                                                                                            \
} while(0)

#define KU_CHECK_KERNEL_LAUNCH() KU_CUDA_CHECK(cudaGetLastError())

template<typename T>
inline __host__ __device__ constexpr T ceil_div(const T &a, const T &b) {
    return (a + b - 1) / b;
}

template<typename T>
inline __host__ __device__ constexpr T ceil(const T &a, const T &b) {
    return (a + b - 1) / b * b;
}

template<typename T, T LOWER_BOUND = 1>
inline __host__ __device__ constexpr T find_next_power_of_2(const T& x) {
    if (x <= LOWER_BOUND)
        return LOWER_BOUND;
    return find_next_power_of_2<T, LOWER_BOUND*2>(x);
}

// Given strides (in number of elements), this function converts their datatype in uint64_t and then multiplies by elem_size
template<typename T>
static inline std::vector<uint64_t> make_stride_helper(const std::vector<T> &strides_in_elems, size_t elem_size) {
    std::vector<uint64_t> res;
    for (auto stride : strides_in_elems) {
        res.push_back(((uint64_t)stride) * elem_size);
    }
    return res;
}

struct KernelLaunchConfig {
    dim3 grid;
    dim3 block;
    size_t dynamic_smem;
    cudaStream_t stream;
    dim3 cluster{1, 1, 1};
    bool use_pdl{false};
    bool cooperative{false};
};

namespace detail {

template <class Arg>
void* kernel_arg_ptr(Arg&& arg) {
    return const_cast<void*>(reinterpret_cast<const void*>(std::addressof(arg)));
}

} // namespace detail

template<typename KernelFunc, typename... Args>
void launch_kernel(const KernelLaunchConfig &cfg, KernelFunc kernel, Args&&... args) {
    if (cfg.dynamic_smem > 0) {
        KU_CUDA_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(cfg.dynamic_smem)));
    }

    // [MACA] 原来这里还有 cooperative / cluster / PDL 三条启动分支，已整组删除：
    //   * cluster —— MACA 无 cluster 支持（JIT/驱动侧直接拒绝 cluster launch），
    //     依赖它的 v3_cluster 变体已整体删除，没有任何调用方会传 cluster != {1,1,1}；
    //   * PDL（programmatic dependent launch）—— MACA 同样不支持；
    //   * cooperative —— cu-bridge 没有 cudaLaunchCooperativeKernel 的对应物。
    //  三者的共同点是「不启用时就退化成最普通的 <<<>>> 启动」，所以这里只留
    //  那一条。
    //  请求了但得不到满足时**只警告、不中断**：启动照常按普通 <<<>>> 走，
    //  同时打印一条说明。静默忽略是最难查的那种失败，而直接抛异常又会把一个
    //  可降级的启动变成硬失败 —— 警告是这两者之间正确的取舍。
    if (cfg.cluster.x != 1 || cfg.cluster.y != 1 || cfg.cluster.z != 1) {
        fprintf(stderr, "[kerutils] warning: cluster launch is not supported on MACA; "
                        "the cluster dimension request is ignored\n");
    }
    if (cfg.use_pdl) {
        fprintf(stderr, "[kerutils] warning: programmatic dependent launch is not supported on MACA; "
                        "the request is ignored\n");
    }
    if (cfg.cooperative) {
        fprintf(stderr, "[kerutils] warning: cooperative launch is not supported on MACA; "
                        "launching without the co-residency guarantee\n");
    }

    kernel<<<cfg.grid, cfg.block, cfg.dynamic_smem, cfg.stream>>>(
        std::forward<Args>(args)...);

    KU_CHECK_KERNEL_LAUNCH();
}

#endif  // KERUTILS_IS_BUILD_ON_CUDA

}   // namespace kerutils