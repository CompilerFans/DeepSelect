// fp32 filter+stage walk at the sample-route's real shape: one CTA per row,
// float4 walk, preset per-row threshold (the true rank-ceil(1.4k) key, computed
// by the driver). Self-contained; compile on the pod in seconds:
//   cucc -O3 --offload-arch=xcore1600 -shared -fPIC \
//        docs/experiments/20261021_f32_sample_route/filter_stage.cu -o /tmp/fs.so
//
// Modes (what each answers):
//   0 pure read                  -- the ceiling at this geometry (threads x CTA/row)
//   1 per-element shared atomic into a 4096-bin histogram
//     (= coarse12 pass 1, the two-pass baseline the route replaces)
//   2 filter+stage, arena idx32[4096] only (16 KB)
//   3 filter+stage, arena idx32[4096]+key32[4096] (32 KB, the route's real plan)
//   4 filter-only (per-thread count, no survivor atomic/write)
//     -- splits walk+compare from the ~1.1% survivors' stage cost
//   5 mode 2 with two float4 in flight (the MLP arm)
// Widths: THREADS in {512, 640, 1024} -- previews the thread-width sweep's
// walk-side effect without a full-tree build.
#include <cstdio>
#include <cstdint>

__device__ __forceinline__ uint32_t key_of(float v) {
    const uint32_t b = __float_as_uint(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

template <int MODE, int THREADS>
__global__ __launch_bounds__(THREADS) void fs_kernel(
    const float *__restrict__ in, const uint32_t *__restrict__ thresh,
    uint32_t len, uint32_t *__restrict__ out_count) {
    extern __shared__ uint32_t smem[];
    uint32_t *arena_idx = smem;                          // [4096]
    uint32_t *arena_key = smem + 4096;                   // [4096], MODE 3 only
    uint32_t *counter   = smem + (MODE == 3 ? 8192 : 4096);
    uint32_t *overflow  = counter + 1;
    const uint32_t tx = threadIdx.x;
    const float4 *row4 =
        reinterpret_cast<const float4 *>(in + (uint64_t)blockIdx.x * len);
    const uint32_t T = thresh[blockIdx.x];
    for (uint32_t b = tx; b < 4096u; b += THREADS) arena_idx[b] = 0u;
    if (tx == 0) { *counter = 0u; *overflow = 0u; }
    __syncthreads();

    const uint32_t vec_len = len >> 2;
    uint32_t acc = 0u;                                   // DCE guard / per-thread count

#define FS_ELEM(w, elem_index)                                              \
    do {                                                                    \
        const uint32_t key = key_of(w);                                     \
        if (MODE == 0) {                                                    \
            acc ^= key;                                                     \
        } else if (MODE == 1) {                                             \
            atomicAdd(&arena_idx[key >> 20], 1u);                           \
        } else if (MODE == 4) {                                             \
            acc += (key > T) ? 1u : 0u;                                     \
        } else {                                                            \
            if (key > T) {                                                  \
                const uint32_t pos = atomicAdd(counter, 1u);                \
                if (pos < 4096u) {                                          \
                    arena_idx[pos] = (uint32_t)(elem_index);                \
                    if (MODE == 3) arena_key[pos] = key;                    \
                } else {                                                    \
                    atomicAdd(overflow, 1u);                                \
                }                                                           \
            }                                                               \
        }                                                                   \
    } while (0)

    if (MODE == 5) {
        uint32_t vec = tx;
        for (; vec + THREADS < vec_len; vec += 2 * THREADS) {
            const float4 a = __ldg(row4 + vec);
            const float4 b = __ldg(row4 + vec + THREADS);
            FS_ELEM(a.x, vec * 4 + 0); FS_ELEM(a.y, vec * 4 + 1);
            FS_ELEM(a.z, vec * 4 + 2); FS_ELEM(a.w, vec * 4 + 3);
            const uint32_t vb = vec + THREADS;
            FS_ELEM(b.x, vb * 4 + 0); FS_ELEM(b.y, vb * 4 + 1);
            FS_ELEM(b.z, vb * 4 + 2); FS_ELEM(b.w, vb * 4 + 3);
        }
        for (; vec < vec_len; vec += THREADS) {
            const float4 a = __ldg(row4 + vec);
            FS_ELEM(a.x, vec * 4 + 0); FS_ELEM(a.y, vec * 4 + 1);
            FS_ELEM(a.z, vec * 4 + 2); FS_ELEM(a.w, vec * 4 + 3);
        }
    } else {
        for (uint32_t vec = tx; vec < vec_len; vec += THREADS) {
            const float4 a = __ldg(row4 + vec);
            FS_ELEM(a.x, vec * 4 + 0); FS_ELEM(a.y, vec * 4 + 1);
            FS_ELEM(a.z, vec * 4 + 2); FS_ELEM(a.w, vec * 4 + 3);
        }
    }
#undef FS_ELEM

    __syncthreads();
    if (MODE == 0) {
        if (acc == 0xDEADBEEFu && tx == 0) out_count[blockIdx.x] = acc;
    } else if (MODE == 1) {
        if (arena_idx[0] == 0xDEADBEEFu && tx == 0) out_count[blockIdx.x] = 1u;
    } else if (MODE == 4) {
        if (acc) atomicAdd(counter, acc);
        __syncthreads();
        if (tx == 0) out_count[blockIdx.x] = *counter;
    } else {
        if (tx == 0) out_count[blockIdx.x] = *counter;
    }
}

template <int MODE, int THREADS>
static float bench_one(const float *p, const uint32_t *t, uint32_t *oc,
                       int rows, uint32_t len, int iters, int smem) {
    if (smem > 48 * 1024) {
        cudaFuncSetAttribute(fs_kernel<MODE, THREADS>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, smem);
    }
    cudaEvent_t e0, e1;
    cudaEventCreate(&e0);
    cudaEventCreate(&e1);
    for (int r = 0; r < 2; r++)
        fs_kernel<MODE, THREADS><<<rows, THREADS, smem>>>(p, t, len, oc);
    cudaDeviceSynchronize();
    cudaEventRecord(e0);
    for (int r = 0; r < iters; r++)
        fs_kernel<MODE, THREADS><<<rows, THREADS, smem>>>(p, t, len, oc);
    cudaEventRecord(e1);
    cudaEventSynchronize(e1);
    float ms = 0;
    cudaEventElapsedTime(&ms, e0, e1);
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    return ms / iters;
}

template <int THREADS>
static void run_width(const float *p, const uint32_t *t, uint32_t *oc,
                      int rows, uint32_t len, int iters) {
    static const char *names[6] = {
        "pure read (ceiling)      ", "elem-atomic (coarse12 p1) ",
        "filter+stage idx  (16 KB)", "filter+stage idx+key(32KB)",
        "filter-only (no stage)   ", "filter+stage OUT=2 (16 KB)"};
    static const int smems[6] = {4096 * 4 + 8, 4096 * 4 + 8, 4096 * 4 + 8,
                                 8192 * 4 + 8, 4096 * 4 + 8, 4096 * 4 + 8};
    for (int mode = 0; mode < 6; mode++) {
        float ms = 0;
        switch (mode) {
        case 0: ms = bench_one<0, THREADS>(p, t, oc, rows, len, iters, smems[0]); break;
        case 1: ms = bench_one<1, THREADS>(p, t, oc, rows, len, iters, smems[1]); break;
        case 2: ms = bench_one<2, THREADS>(p, t, oc, rows, len, iters, smems[2]); break;
        case 3: ms = bench_one<3, THREADS>(p, t, oc, rows, len, iters, smems[3]); break;
        case 4: ms = bench_one<4, THREADS>(p, t, oc, rows, len, iters, smems[4]); break;
        case 5: ms = bench_one<5, THREADS>(p, t, oc, rows, len, iters, smems[5]); break;
        }
        const double gbs = (double)rows * (double)len * 4.0 / (ms * 1e6);
        printf("THREADS %4d  %s  %8.3f ms  %7.1f GB/s\n", THREADS, names[mode],
               ms, gbs);
    }
    fflush(stdout);
}

extern "C" void probe(const void *p, const void *t, void *oc, int rows,
                      uint32_t len, int iters) {
    const float *in = reinterpret_cast<const float *>(p);
    const uint32_t *th = reinterpret_cast<const uint32_t *>(t);
    uint32_t *counts = reinterpret_cast<uint32_t *>(oc);
    run_width<640>(in, th, counts, rows, len, iters);
    run_width<512>(in, th, counts, rows, len, iters);
    run_width<1024>(in, th, counts, rows, len, iters);
}
