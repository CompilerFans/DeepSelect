// Price the per-CTA latency floor: what does a CTA cost when it does nothing,
// and how does a fixed amount of reading per CTA scale with grid size?
#include <cstdio>
#include <cstdint>
#include <cuda_runtime.h>

__global__ void k_empty() {}
__global__ void k_empty_smem() { extern __shared__ char s[]; if (threadIdx.x == 0) s[0] = 1; }

// CTA b reads the contiguous uint4 window [b*len4, (b+1)*len4).
__global__ void k_read_u4(const uint4* __restrict__ in, uint32_t* out, int len4) {
    uint32_t acc = 0;
    const uint4* p = in + (size_t)blockIdx.x * (size_t)len4;
    for (int i = threadIdx.x; i < len4; i += blockDim.x) {
        uint4 v = __ldg(p + i);
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0xdeadbeefu) out[blockIdx.x] = acc;
}

static void timeit(const char* tag, int iters, void (*launch)(int), int grid) {
    cudaEvent_t a, b;
    cudaEventCreate(&a); cudaEventCreate(&b);
    for (int i = 0; i < 5; i++) launch(i);
    cudaDeviceSynchronize();
    cudaEventRecord(a);
    for (int i = 0; i < iters; i++) launch(i);
    cudaEventRecord(b);
    cudaEventSynchronize(b);
    float ms = 0; cudaEventElapsedTime(&ms, a, b);
    printf("%-34s grid=%-4d  %8.2f us/launch\n", tag, grid, ms * 1000.0f / iters);
}

// ---- launch thunks ----
static uint4* g_buf; static uint32_t* g_out; static int g_len4;
static void l_empty(int)      { k_empty<<<1, 1024>>>(); }
static void l_empty_smem(int) { k_empty_smem<<<1, 1024, 20480>>>(); }
static int g_grid;
static void l_empty_smem_g(int) { k_empty_smem<<<g_grid, 1024, 20480>>>(); }
static void l_read(int)       { k_read_u4<<<g_grid, 1024>>>(g_buf, g_out, g_len4); }

int main() {
    int dev = 0; cudaSetDevice(dev);
    cudaDeviceProp prop; cudaGetDeviceProperties(&prop, dev);
    printf("device=%s sm_count=%d\n\n", prop.name, prop.multiProcessorCount);
    size_t words = (size_t)104 * (1 << 20);        // 104 MB, plenty
    cudaMalloc(&g_buf, words * 16);
    cudaMalloc(&g_out, 4096);
    cudaMemset(g_buf, 1, words * 16);

    timeit("empty <<<1,1024>>>", 2000, l_empty, 1);
    timeit("empty +20KB smem", 2000, l_empty_smem, 1);
    for (int g : {2, 6, 16, 48, 104, 208}) {
        g_grid = g;
        char buf[64]; snprintf(buf, sizeof buf, "empty +20KB smem");
        timeit(buf, 2000, l_empty_smem_g, g);
    }
    printf("\n");
    for (int kb : {16, 128, 512}) {
        g_len4 = kb * 64;                          // KB -> uint4 count
        for (int g : {1, 2, 4, 6, 8, 16, 32, 64, 104, 208}) {
            if ((size_t)g * kb * 1024 > words * 16) break;
            g_grid = g;
            char buf[64]; snprintf(buf, sizeof buf, "read %d KB/CTA", kb);
            timeit(buf, 500, l_read, g);
        }
        printf("\n");
    }
    return 0;
}
