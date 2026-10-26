"""Phase timers for `topk_coarse12_row`.

Anchors are the function's own barriers, each asserted unique:

    0  pass 1: the 4,096-bin histogram, the block summary and the NaN vote
    1  the 256-bin fold, the cumsum and the 12-bit narrow
    2  the kept-block list build
    3  pass 2: the collect over the list (or the dense walk)
    4  the refine: the overflow rescan or the arena's three rounds

    python3 f32phase_patch.py <path-to-a-copy-of-topk_coarse12.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()

subs = [
    ("namespace topk_select_coarse12 {\n",
     """#ifdef DS_PHASE_TIMERS
__device__ unsigned long long ds_f32_phase[8];
#define DS_PH_INIT() unsigned long long ds_t0_ = (threadIdx.x == 0u ? clock64() : 0ull)
#define DS_PH(i) do { if (threadIdx.x == 0u) { const unsigned long long t_ = clock64(); atomicAdd(&ds_f32_phase[i], t_ - ds_t0_); ds_t0_ = t_; } } while (0)
#else
#define DS_PH_INIT() ((void)0)
#define DS_PH(i) ((void)0)
#endif

namespace topk_select_coarse12 {
"""),
    # 0 = pass 1, ends on the barrier that publishes the histogram and the vote.
    ("    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {\n"
     "        if (tid == 0) atomicOr(nan_flag, 1);\n"
     "    }\n"
     "    __syncthreads();\n",
     "    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {\n"
     "        if (tid == 0) atomicOr(nan_flag, 1);\n"
     "    }\n"
     "    __syncthreads();\n    DS_PH(0);\n"),
    # 1 = the fold, the cumsum and the narrow.
    ("        num_input[0] = 0;\n        counter = 0;\n    }\n    __syncthreads();\n",
     "        num_input[0] = 0;\n        counter = 0;\n    }\n    __syncthreads();\n    DS_PH(1);\n"),
    # 2 = the kept-block list.
    ("        nlist = kept_block_count;\n"
     "        use_list = nlist * kListMaxDenominator <= nblk * kListMaxNumerator;\n    }\n",
     "        nlist = kept_block_count;\n"
     "        use_list = nlist * kListMaxDenominator <= nblk * kListMaxNumerator;\n    }\n"
     "    DS_PH(2);\n"),
    # 3 = pass 2, up to the barrier that hands the candidates to the refine.
    ("#undef DG12_COLLECT\n    __syncthreads();\n",
     "#undef DG12_COLLECT\n    __syncthreads();\n    DS_PH(3);\n"),
    # The function's own entry, so 4 is everything the refine costs.
    ("    const int tid = threadIdx.x;\n    const unsigned int u_length = static_cast<unsigned int>(length);\n",
     "    const int tid = threadIdx.x;\n"
     "    const unsigned int u_length = static_cast<unsigned int>(length);\n"
     "    DS_PH_INIT();\n"),
]

for old, new in subs:
    n = s.count(old)
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
