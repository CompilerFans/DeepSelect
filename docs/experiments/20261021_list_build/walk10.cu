// walk10.cu -- the kept-block list build, isolated.  The phase probe puts it at
// 7% of b4096 V1M k512 (52,584 cycles/CTA), against 32,768 scalar uint16 block
// maxima per row and ~1,000 kept blocks (the 6% the V/k = 1024 density keeps).
// Question: how much of that is the load pattern and how much the one
// shared-atomic-per-kept-block, and what do the two obvious fixes buy?
//   A  production form: scalar uint16, `continue`, atomic per kept block
//   A2 same, block_max declared const __restrict__ (aliasing check)
//   B  scalar loads + wave ballot compaction: one atomic per 64-block wave
//   C  uint4 loads (8 blocks/thread) + one atomic per thread
//   D  uint4 loads + ballot over 8 per thread (both at once)
// Each kernel writes nlist and the list itself into a per-row workspace; the
// sums are dumped so the compiler cannot delete the work.
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <cuda_runtime.h>

#define ROWS 4096u
#define VBITS (1u << 20)
#define NBLK (VBITS / 64u)          // 16,384 blocks per row
#define CUT 3840u                   // ~6% of blocks kept
#define CAP (NBLK - (NBLK >> 2))

#define CK(x) do { cudaError_t e = (x); if (e != cudaSuccess) { \
    printf("FAIL %s: %s\n", #x, cudaGetErrorString(e)); exit(1); } } while (0)

// A: the production loop, written the way the kernel has it (a plain pointer).
__global__ void listA(const uint16_t* bm, uint32_t* list, uint32_t* out) {
    const uint32_t tx = threadIdx.x;
    __shared__ uint32_t s_n;
    if (tx == 0) s_n = 0;
    __syncthreads();
    uint32_t* row_list = list + (size_t)blockIdx.x * CAP;
    for (uint32_t b = tx; b < NBLK; b += blockDim.x) {
        if (bm[b] < CUT) continue;
        const uint32_t pos = atomicAdd(&s_n, 1u);
        if (pos < CAP) row_list[pos] = b;
    }
    __syncthreads();
    if (tx == 0) out[blockIdx.x] = s_n;
}

// A2: same loop, `const __restrict__` -- is the aliasing the cost?
__global__ void listA2(const uint16_t* __restrict__ bm, uint32_t* list,
                       uint32_t* out) {
    const uint32_t tx = threadIdx.x;
    __shared__ uint32_t s_n;
    if (tx == 0) s_n = 0;
    __syncthreads();
    uint32_t* row_list = list + (size_t)blockIdx.x * CAP;
    for (uint32_t b = tx; b < NBLK; b += blockDim.x) {
        if (bm[b] < CUT) continue;
        const uint32_t pos = atomicAdd(&s_n, 1u);
        if (pos < CAP) row_list[pos] = b;
    }
    __syncthreads();
    if (tx == 0) out[blockIdx.x] = s_n;
}

// B: one ballot per wave, one atomic per wave that keeps anything.
__global__ void listB(const uint16_t* __restrict__ bm, uint32_t* list,
                      uint32_t* out) {
    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 63u;
    __shared__ uint32_t s_n;
    if (tx == 0) s_n = 0;
    __syncthreads();
    uint32_t* row_list = list + (size_t)blockIdx.x * CAP;
    for (uint32_t b = tx; b < NBLK; b += blockDim.x) {
        const bool kept = bm[b] >= CUT;
        const unsigned long long live = __activemask();
        const unsigned long long m = __ballot_sync(live, (int)kept);
        if (m) {
            uint32_t base = 0;
            if (lane == 0u) base = atomicAdd(&s_n, (uint32_t)__popcll(m));
            base = (uint32_t)__builtin_mxc_readfirstlane((int)base);
            if (kept) {
                const unsigned long long lt = (1ull << lane) - 1ull;
                const uint32_t pos = base + (uint32_t)__popcll(m & lt);
                if (pos < CAP) row_list[pos] = b;
            }
        }
    }
    __syncthreads();
    if (tx == 0) out[blockIdx.x] = s_n;
}

// C: eight blocks per thread per iteration, one atomic per thread.
__global__ void listC(const uint16_t* __restrict__ bm, uint32_t* list,
                      uint32_t* out) {
    const uint32_t tx = threadIdx.x;
    __shared__ uint32_t s_n;
    if (tx == 0) s_n = 0;
    __syncthreads();
    uint32_t* row_list = list + (size_t)blockIdx.x * CAP;
    const uint4* bm4 = reinterpret_cast<const uint4*>(bm);
    const uint32_t n4 = NBLK / 8u;
    for (uint32_t q = tx; q < n4; q += blockDim.x) {
        const uint4 v = __ldg(bm4 + q);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
        uint32_t cnt = 0, pos = 0;
        bool kept[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            kept[i] = h[i] >= CUT;
            cnt += kept[i] ? 1u : 0u;
        }
        if (cnt) pos = atomicAdd(&s_n, cnt);
        uint32_t at = pos;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (kept[i]) {
                if (at < CAP) row_list[at] = q * 8u + (uint32_t)i;
                at++;
            }
        }
    }
    __syncthreads();
    if (tx == 0) out[blockIdx.x] = s_n;
}

// D: uint4 loads, then one ballot per wave per 8-entry group is not available
// (a ballot covers one predicate per lane), so the vector form's compaction is
// per-thread: C already measures it.  D instead keeps the vector load and uses
// the wave ballot on the *thread* predicate, sized by cnt.
__global__ void listD(const uint16_t* __restrict__ bm, uint32_t* list,
                      uint32_t* out) {
    const uint32_t tx = threadIdx.x;
    const uint32_t lane = tx & 63u;
    __shared__ uint32_t s_n;
    if (tx == 0) s_n = 0;
    __syncthreads();
    uint32_t* row_list = list + (size_t)blockIdx.x * CAP;
    const uint4* bm4 = reinterpret_cast<const uint4*>(bm);
    const uint32_t n4 = NBLK / 8u;
    for (uint32_t q = tx; q < n4; q += blockDim.x) {
        const uint4 v = __ldg(bm4 + q);
        const uint16_t* h = reinterpret_cast<const uint16_t*>(&v);
        uint32_t cnt = 0;
        bool kept[8];
        #pragma unroll
        for (int i = 0; i < 8; i++) { kept[i] = h[i] >= CUT; cnt += kept[i]; }
        // One atomic per wave: the wave's total, then a prefix over lanes.
        const unsigned long long live = __activemask();
        const unsigned long long m = __ballot_sync(live, (int)cnt);
        uint32_t base = 0, lead = 0;
        if (lane == 0u) base = atomicAdd(&s_n, (uint32_t)__popcll(m));
        base = (uint32_t)__builtin_mxc_readfirstlane((int)base);
        #pragma unroll
        for (int k = 0; k < 6; k++) {          // 64-lane prefix sum of cnt
            const uint32_t o = (uint32_t)__builtin_mxc_bsm_bpermute(
                (int)(((lane - (1u << k)) & 63u) << 2), (int)cnt);
            lead += (lane >= (1u << k)) ? o : 0u;
        }
        uint32_t at = base + lead;
        #pragma unroll
        for (int i = 0; i < 8; i++) {
            if (kept[i]) {
                if (at < CAP) row_list[at] = q * 8u + (uint32_t)i;
                at++;
            }
        }
    }
    __syncthreads();
    if (tx == 0) out[blockIdx.x] = s_n;
}

static __global__ void fill(uint16_t* p, size_t n, uint32_t seed) {
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t x = seed ^ (uint32_t)(i * 2654435761u);
        x ^= x << 13; x ^= x >> 17; x ^= x << 5;
        p[i] = (uint16_t)(x & 0xFFFu);
    }
}

template <typename F>
static void run(const char* name, F kernel, uint32_t block,
                const uint16_t* bm, uint32_t* list, uint32_t* out,
                int rounds) {
    int occ = 0;
    CK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&occ, kernel, block, 0));
    kernel<<<ROWS, block>>>(bm, list, out);
    CK(cudaDeviceSynchronize());
    double best = 1e30;
    for (int r = 0; r < rounds; r++) {
        cudaEvent_t a, b;
        CK(cudaEventCreate(&a));
        CK(cudaEventCreate(&b));
        CK(cudaEventRecord(a));
        kernel<<<ROWS, block>>>(bm, list, out);
        CK(cudaEventRecord(b));
        CK(cudaEventSynchronize(b));
        float ms = 0;
        CK(cudaEventElapsedTime(&ms, a, b));
        if (ms < best) best = ms;
        CK(cudaEventDestroy(a));
        CK(cudaEventDestroy(b));
    }
    // receipt: total kept blocks, and a checksum over the lists
    uint64_t sum = 0;
    uint32_t* h = (uint32_t*)malloc(ROWS * sizeof(uint32_t));
    CK(cudaMemcpy(h, out, ROWS * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    for (uint32_t i = 0; i < ROWS; i++) sum += h[i];
    free(h);
    printf("%-6s BLOCK %4u  occ %d  %8.3f ms  kept/row %llu\n",
           name, block, occ, best, (unsigned long long)(sum / ROWS));
}

int main(int argc, char** argv) {
    int dev = argc > 1 ? atoi(argv[1]) : 0;
    CK(cudaSetDevice(dev));
    const size_t nb = (size_t)ROWS * NBLK;
    uint16_t* bm = nullptr;
    uint32_t* list = nullptr;
    uint32_t* out = nullptr;
    CK(cudaMalloc(&bm, nb * sizeof(uint16_t)));
    CK(cudaMalloc(&list, (size_t)ROWS * CAP * sizeof(uint32_t)));
    CK(cudaMalloc(&out, ROWS * sizeof(uint32_t)));
    fill<<<512, 256>>>(bm, nb, 12345u);
    CK(cudaDeviceSynchronize());
    const int rounds = 3;
    for (uint32_t block : {1024u, 512u}) {
        run("A", listA, block, bm, list, out, rounds);
        run("A2", listA2, block, bm, list, out, rounds);
        run("B", listB, block, bm, list, out, rounds);
        run("C", listC, block, bm, list, out, rounds);
        run("D", listD, block, bm, list, out, rounds);
    }
    CK(cudaFree(bm));
    CK(cudaFree(list));
    CK(cudaFree(out));
    return 0;
}
