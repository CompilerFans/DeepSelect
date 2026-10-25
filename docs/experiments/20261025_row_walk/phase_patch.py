import sys

p = sys.argv[1]
s = open(p).read()

subs = [
    ("namespace topk_select_radix {\n",
     """#ifdef DS_PHASE_TIMERS
// Scratch instrumentation (not shipped): thread 0 of each CTA accumulates the
// cycles between two barriers into a device-global array.
__device__ unsigned long long ds_phase[8];
#define DS_PH_INIT() unsigned long long ds_t0_ = (threadIdx.x == 0u ? clock64() : 0ull)
#define DS_PH(i) do { if (threadIdx.x == 0u) { const unsigned long long t_ = clock64(); atomicAdd(&ds_phase[i], t_ - ds_t0_); ds_t0_ = t_; } } while (0)
#else
#define DS_PH_INIT() ((void)0)
#define DS_PH(i) ((void)0)
#endif

namespace topk_select_radix {
"""),
    ("    const uint32_t tx = threadIdx.x;\n    uint32_t remain_topk = topk;\n",
     "    const uint32_t tx = threadIdx.x;\n    uint32_t remain_topk = topk;\n"
     "    DS_PH_INIT();\n"),
    ("    for (uint32_t b = tx; b < kCoarse12Bins; b += BLOCK_SIZE) s_wide[b] = 0;\n"
     "    __syncthreads();\n",
     "    for (uint32_t b = tx; b < kCoarse12Bins; b += BLOCK_SIZE) s_wide[b] = 0;\n"
     "    __syncthreads();\n    DS_PH(0);\n"),
    ("    __syncthreads();\n    // The histogram is complete and the CTA's NaN vote is in, on one barrier --\n",
     "    __syncthreads();\n    DS_PH(1);\n"
     "    // The histogram is complete and the CTA's NaN vote is in, on one barrier --\n"),
    ("    scan_threshold(remain_topk);\n\n    // ── the kept-block list",
     "    scan_threshold(remain_topk);\n    DS_PH(2);\n\n    // ── the kept-block list"),
    ("        use_list = list_cap > 0 && nlist <= list_cap;\n    }\n",
     "        use_list = list_cap > 0 && nlist <= list_cap;\n    }\n    DS_PH(3);\n"),
    ("        __syncthreads();\n        scan_threshold(remain_topk);\n    }\n",
     "        __syncthreads();\n        DS_PH(4);\n        scan_threshold(remain_topk);\n"
     "        DS_PH(5);\n    }\n"),
    ("        __syncthreads();\n    }\n\n    {\n        const bool overflow",
     "        __syncthreads();\n        DS_PH(6);\n    }\n\n    {\n        const bool overflow"),
]

for old, new in subs:
    n = s.count(old)
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
