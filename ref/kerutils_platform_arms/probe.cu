// kerutils' platform arms, compiled the way this repository's build compiles
// them.  Run `./run_probe.sh`.
//
// ---------------------------------------------------------------------------
// WHAT THIS PROVES
//
// `csrc/3rdparty/kerutils/` selects one of three platform arms --
// `KERUTILS_IS_BUILD_ON_CUDA`, `_ASCEND`, `_MACA` -- and the MACA one is the
// only arm nothing else here compiles: `device/device.cuh`, `host/host.h` and
// `kerutils.cuh` all reach it, but the sole consumer
// (`csrc/maca_kernels/xcore1600/`) is off `setup.py`'s `SOURCES`, so an
// ordinary build of this tree never instantiates it.  This probe is that
// compilation.
//
// It checks the arm is *self-consistent*, not merely present: that it is the
// one selected and the only one; that the `ceil_div`/`ceil` pair folds in the
// host pass and is callable in a kernel; that the three device primitives
// `maca/common.h` supplies round-trip through shared memory; and that
// `launch_kernel` launches, including on a config that requests a capability
// this platform does not have.
//
// ---------------------------------------------------------------------------
// WHAT IT DOES NOT PROVE
//
// That `csrc/maca_kernels/xcore1600/` compiles.  It does not, and the work
// outstanding there is recorded under "Known holes" in CLAUDE.md.  The subject
// here is the vendored library's platform arm, which is a smaller claim than
// the port that would consume it.

#include <cstdint>
#include <cstdio>

#include <kerutils/kerutils.cuh>

// Four 32-thread groups per block, which is two of MACA's 64-lane waves -- the
// pair of numbers the `canonical_warp_idx_sync` check below turns on.
static constexpr uint32_t kThreads = 128;
// The second block exists only so the warned launch has a grid to launch; both
// blocks run the same arithmetic on the same inputs.
static constexpr uint32_t kGrid  = 2;
static constexpr uint32_t kElems = kThreads * kGrid;

// ---------------------------------------------------------------------------
// 1. Exactly one arm, and it is MACA.
//
// `common/common.h` raises its own `#error` when none of the three fires, so
// the first check below re-tests something already enforced.  It is kept
// because it is the one that names *which* arm fired: without it a MACA build
// that lost `__MACA__` would report "the MACA arm did not fire" from here,
// while a build where some other arm fired would report it from three files
// away.
// ---------------------------------------------------------------------------
#if !defined(KERUTILS_IS_BUILD_ON_MACA)
#error "the MACA arm did not fire: mxcc's __MACA__ was not defined in this pass"
#endif
#if defined(KERUTILS_IS_BUILD_ON_CUDA)
#error "the CUDA arm fired as well: __CUDACC__ is defined, and the two are exclusive"
#endif
#if defined(KERUTILS_IS_BUILD_ON_ASCEND)
#error "the ASCEND arm fired as well: __has_include(\"kernel_operator.h\") found a header"
#endif

// ---------------------------------------------------------------------------
// 2. `ceil_div` / `ceil`, in the HOST pass.
//
// This is a constant fold, and upstream's plain `inline constexpr` would
// satisfy it too.  It is here so the pair is read next to part 3, which is the
// half that needs `__host__ __device__` -- the attribute is on both for that
// reason and not for this one.
// ---------------------------------------------------------------------------
static_assert(ku::ceil_div(10u, 3u) == 4u, "ceil_div(10, 3) must round up to 4");
static_assert(ku::ceil(10u, 3u) == 12u, "ceil(10, 3) must round up to the 12 boundary");

// ---------------------------------------------------------------------------
// 2b. What each compilation pass sees, printed rather than argued.
//
// A `.cu` is preprocessed twice -- once for host code, once for device -- and
// the two see different macro sets.  Getting this table wrong is how arm
// selection goes wrong, and reading it wrong once is already on the record
// here: `__CUDA_ARCH__` reads as "not defined on MACA" until you look in the
// device pass, where cucc defines it as 800.
//
// The two rows that decide the design:
//   * `__MACA__` is defined in **both** passes -- which is why arm selection
//     keys on it and not on `__MACA_ARCH__`, a device-pass-only macro.  A
//     `host/host.h` waiting on the latter would find only its `#error`.
//   * `__CUDACC__` is defined in both, by cucc's `-imacros __macro_mxcc.h`.
//     It is an alias for CUDA-dialect source, not a statement about the
//     platform, so the MACA test has to precede it or both arms fire.
// ---------------------------------------------------------------------------
enum : uint32_t { kMaca = 1u << 0, kMacacc = 1u << 1, kCudacc = 1u << 2, kNvcc = 1u << 3 };

__device__ __forceinline__ uint32_t pass_bits_device() {
    uint32_t m = 0;
#ifdef __MACA__
    m |= kMaca;
#endif
#ifdef __MACACC__
    m |= kMacacc;
#endif
#ifdef __CUDACC__
    m |= kCudacc;
#endif
#ifdef __NVCC__
    m |= kNvcc;
#endif
    return m;
}

static uint32_t pass_bits_host() {
    uint32_t m = 0;
#ifdef __MACA__
    m |= kMaca;
#endif
#ifdef __MACACC__
    m |= kMacacc;
#endif
#ifdef __CUDACC__
    m |= kCudacc;
#endif
#ifdef __NVCC__
    m |= kNvcc;
#endif
    return m;
}

__global__ void pass_report_kernel(uint32_t* out) {
    uint32_t cuda_arch = 0, maca_arch = 0;
#ifdef __CUDA_ARCH__
    cuda_arch = (uint32_t)__CUDA_ARCH__;
#endif
#ifdef __MACA_ARCH__
    maca_arch = (uint32_t)__MACA_ARCH__;
#endif
    out[0] = pass_bits_device();
    out[1] = cuda_arch;
    out[2] = maca_arch;
}

static void print_bits(const char* pass, uint32_t m) {
    std::printf("     %-7s __MACA__=%s __MACACC__=%s __CUDACC__=%s __NVCC__=%s\n", pass,
                (m & kMaca) ? "Y" : "-", (m & kMacacc) ? "Y" : "-",
                (m & kCudacc) ? "Y" : "-", (m & kNvcc) ? "Y" : "-");
}

// Returns the number of failures; the two checked facts are the ones the arm
// selection rests on, not the whole table.
static int report_passes() {
    uint32_t* d = nullptr;
    if (cudaMalloc(&d, 3 * sizeof(uint32_t)) != cudaSuccess) return 1;
    pass_report_kernel<<<1, 1>>>(d);
    if (cudaDeviceSynchronize() != cudaSuccess) return 1;
    uint32_t h[3] = {0u, 0u, 0u};
    if (cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost) != cudaSuccess) return 1;
    cudaFree(d);

    std::printf("ok   what each compilation pass sees\n");
    print_bits("HOST", pass_bits_host());
    print_bits("DEVICE", h[0]);
    std::printf("     ...  DEVICE __CUDA_ARCH__=%u  __MACA_ARCH__=%u\n", h[1], h[2]);

    int bad = 0;
    if (!(h[0] & kMaca)) {
        std::printf("FAIL: __MACA__ is absent in the DEVICE pass -- arm selection would be "
                    "inconsistent between the two passes\n");
        ++bad;
    }
    if (!(pass_bits_host() & kMaca)) {
        std::printf("FAIL: __MACA__ is absent in the HOST pass -- `host/host.h` would find "
                    "only its #error\n");
        ++bad;
    }
    if (h[1] != 0) {
        std::printf("     note: __CUDA_ARCH__=%u in the device pass is cucc's compatibility "
                    "value, not this part;\n"
                    "           it is why `maca/common.h` omits the SM gate block rather than "
                    "relying on it being inert\n", h[1]);
    }
    return bad;
}

// ---------------------------------------------------------------------------
// 3. The device pass.
//
// `maca/common.h` is where these names come from; every one of them is called
// from `csrc/maca_kernels/xcore1600/`.  `ku::trap()` is referenced but not
// reached -- it aborts, so it cannot be part of a passing run, and a call the
// compiler must still typecheck is what this needs from it.
// ---------------------------------------------------------------------------
__global__ void arm_probe_kernel(const uint32_t* in, uint32_t* out, uint32_t n) {
    extern __shared__ uint4 smem[];              // 16 B per thread for the vector store
    auto* smem_halves = reinterpret_cast<uint2*>(smem + gridDim.x * blockDim.x);

    const uint32_t tx = blockIdx.x * blockDim.x + threadIdx.x;
    if (tx >= n) return;
    const uint32_t v = in[tx];

    // the constexpr pair, in the pass that needs the attribute
    const uint32_t rounded_up = ku::ceil_div(v, 7u);
    const uint32_t boundary   = ku::ceil(v, 7u);

    // The 32-thread-group index.  MACA's *wave* is 64 lanes, so its own wave
    // index would be tx/64 and this block's 128 threads would report 0,0,1,1 --
    // the check below wants 0,1,2,3.  That gap is the whole reason this
    // function is not `threadIdx.x / 64u`; see `maca/common.h`.
    const uint32_t group = ku::canonical_warp_idx_sync();

    uint4* mine_vec    = smem + tx;              // 16 bytes, its own slot
    uint2* mine_halves = smem_halves + tx;       // 8 bytes, its own slot
    ku::st_shared(mine_vec, make_uint4(v, v + 1u, v + 2u, v + 3u));
    ku::st_shared(mine_halves, (uint64_t)v, (uint64_t)(v + 1u));

    __syncthreads();

    const uint4 a = *mine_vec;
    const uint2 b = *mine_halves;

    if (v == 0xFFFFFFFFu) {
        ku::trap();                              // unreachable for this driver's input
    }

    out[tx] = rounded_up + boundary + group
            + a.x + a.y + a.z + a.w
            + b.x + b.y;
}

// ceil_div(v,7) + ceil(v,7) + group + (4v+6) + (2v+1)
//
// `group` is `canonical_warp_idx_sync()`, which is **block-local**: it is the
// 32-thread group's index *within its block*, so it says nothing about
// `blockIdx.x` and the second block's slots repeat 0..3 rather than
// continuing at 4.  Modelling that here is also what makes this a
// discriminator rather than an echo: at 128 threads the four groups are
// 0,1,2,3, while MACA's own wave index (`threadIdx.x / 64`) would be 0,0,1,1 --
// so slots 32..95 would disagree with the model if the function returned the
// platform's wave instead of CUDA's warp.
static uint32_t expect(uint32_t tx) {
    const uint32_t v = tx + 1u;
    const uint32_t up = (v + 6u) / 7u;
    const uint32_t group = (tx % kThreads) / 32u;
    return up + up * 7u + group + (4u * v + 6u) + (2u * v + 1u);
}

int main() {
    if (cudaSetDevice(0) != cudaSuccess) {
        std::printf("FAIL: cudaSetDevice(0)\n");
        return 1;
    }

    const int pass_bad = report_passes();
    if (pass_bad) {
        std::printf("FAIL: %d pass-consistency check(s) -- the arm selection is not sound\n", pass_bad);
        return 1;
    }

    uint32_t h_in[kElems], h_out[kElems];
    for (uint32_t i = 0; i < kElems; ++i) { h_in[i] = i + 1u; h_out[i] = 0u; }

    uint32_t *d_in = nullptr, *d_out = nullptr;
    const size_t smem = (size_t)kElems * sizeof(uint4) + (size_t)kElems * sizeof(uint2);
    if (cudaMalloc(&d_in, sizeof(h_in)) != cudaSuccess ||
        cudaMalloc(&d_out, sizeof(h_out)) != cudaSuccess ||
        cudaMemcpy(d_in, h_in, sizeof(h_in), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("FAIL: device allocation / copy\n");
        return 1;
    }

    // ---- 3a. launch_kernel on a default config ------------------------------
    ku::KernelLaunchConfig cfg{dim3(kGrid), dim3(kThreads), smem, (cudaStream_t)0};
    ku::launch_kernel(cfg, arm_probe_kernel, d_in, d_out, kElems);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("FAIL: sync after the default launch: %s\n", cudaGetErrorString(cudaGetLastError()));
        return 1;
    }
    if (cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: copy back\n");
        return 1;
    }

    int bad = 0;
    for (uint32_t i = 0; i < kElems; ++i) {
        if (h_out[i] != expect(i)) {
            if (bad < 4) {
                std::printf("FAIL: out[%u] = %u, want %u\n", i, h_out[i], expect(i));
            }
            ++bad;
        }
    }
    if (bad) {
        std::printf("FAIL: %d of %u slots wrong -- the device primitives are not round-tripping\n",
                    bad, kElems);
        return 1;
    }
    std::printf("ok   launch_kernel + the three device primitives    %u/%u slots\n", kElems, kElems);

    // ---- 3b. a config that asks for what this platform does not have --------
    // The launch must still happen and the request must be reported.  The
    // warning text goes to stderr; this only checks the launch is unaffected.
    for (uint32_t i = 0; i < kElems; ++i) h_out[i] = 0u;
    if (cudaMemcpy(d_out, h_out, sizeof(h_out), cudaMemcpyHostToDevice) != cudaSuccess) {
        std::printf("FAIL: clear\n");
        return 1;
    }

    ku::KernelLaunchConfig warned{dim3(kGrid), dim3(kThreads), smem, (cudaStream_t)0};
    warned.cluster     = dim3(2, 1, 1);
    warned.use_pdl     = true;
    warned.cooperative = true;
    std::printf("     (three warnings are expected around here, one per request;\n"
                "      they go to stderr, so where they land depends on buffering)\n");
    ku::launch_kernel(warned, arm_probe_kernel, d_in, d_out, kElems);
    if (cudaDeviceSynchronize() != cudaSuccess ||
        cudaMemcpy(h_out, d_out, sizeof(h_out), cudaMemcpyDeviceToHost) != cudaSuccess) {
        std::printf("FAIL: the warned launch did not complete\n");
        return 1;
    }
    for (uint32_t i = 0; i < kElems; ++i) {
        if (h_out[i] != expect(i)) {
            std::printf("FAIL: warned launch produced out[%u] = %u, want %u\n", i, h_out[i], expect(i));
            return 1;
        }
    }
    std::printf("ok   launch_kernel with cluster/PDL/cooperative set  %u/%u slots\n", kElems, kElems);

    std::printf("\nPASS: the MACA arm is the one selected, and it is self-consistent.\n");
    return 0;
}
