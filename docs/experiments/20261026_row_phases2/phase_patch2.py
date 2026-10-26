"""Insert clock64() phase timers into a copy of `radix_row_bf16.cuh`.

Written for the dataflow that stages during the counting walk (the collect is
gone) and packs the 12-bit bin above the index.  Every anchor is asserted
unique, so a header that has moved underneath reports it instead of patching
the wrong barrier.

Slots 0..5 are timers, 6..7 and 13..14 are counters (they say which arm the
map was taken on and how big its two sets were), 8..11 split the two scans.

    python3 phase_patch2.py <path-to-a-copy-of-radix_row_bf16.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()

subs = [
    # The timers themselves.  Thread 0 of each CTA accumulates the cycles
    # between two anchors into a device-global array.
    ("namespace topk_select_radix {\n",
     """#ifdef DS_PHASE_TIMERS
// Scratch instrumentation (not shipped).
__device__ unsigned long long ds_phase[16];
#define DS_PH_INIT() unsigned long long ds_t0_ = (threadIdx.x == 0u ? clock64() : 0ull)
#define DS_PH(i) do { if (threadIdx.x == 0u) { const unsigned long long t_ = clock64(); atomicAdd(&ds_phase[i], t_ - ds_t0_); ds_t0_ = t_; } } while (0)
#else
#define DS_PH_INIT() ((void)0)
#define DS_PH(i) ((void)0)
#endif

namespace topk_select_radix {
"""),
    # 0 = pass 1 (the summary walk and its tail), ends on the barrier that
    # publishes the histogram.
    ("    const uint32_t tx = threadIdx.x;\n    uint32_t remain_topk = topk;\n",
     "    const uint32_t tx = threadIdx.x;\n    uint32_t remain_topk = topk;\n"
     "    DS_PH_INIT();\n"),
    ("    __syncthreads();\n    // The histogram is complete and the CTA's NaN vote is in, on one barrier --\n",
     "    __syncthreads();\n    DS_PH(0);\n"
     "    // The histogram is complete and the CTA's NaN vote is in, on one barrier --\n"),
    # 1 = scan #1 (the block-count fold, the scan and the narrow).
    ("    if (!one_pass) scan_threshold(remain_topk);\n\n    // ── the kept-block list",
     "    if (!one_pass) scan_threshold(remain_topk);\n    DS_PH(1);\n\n    // ── the kept-block list"),
    # 2 = the kept-block list build.  Slot 6 counts the rows that got a list.
    ("        use_list = list_cap > 0 && nlist <= list_cap;\n    }\n",
     "        use_list = list_cap > 0 && nlist <= list_cap;\n    }\n    DS_PH(2);\n"
     "    if (threadIdx.x == 0u && use_list) atomicAdd(&ds_phase[6], 1ull);\n"),
    # 3 = the staging walk over the list, plus the row's tail.  Slot 7 counts
    # the rows that kept the arena (the rest fell back to the counting walk).
    ("            __syncthreads();\n            num_staged = s_nstage;\n",
     "            __syncthreads();\n            DS_PH(3);\n            num_staged = s_nstage;\n"),
    # The arena's verdict, counted after it is set -- `staged` is still false on
    # the line above.
    ("            staged = num_staged <= SMEM_INPUT_SIZE;\n            packed = staged && fits_packed;\n",
     "            staged = num_staged <= SMEM_INPUT_SIZE;\n            packed = staged && fits_packed;\n"
     "            if (threadIdx.x == 0u && staged) atomicAdd(&ds_phase[7], 1ull);\n"),
    # 13 = the counting arm's own histogram walk over the row, and 14 = its
    # collect.  Neither exists on the staged arm.
    ("            scan_threshold(remain_topk);\n        } else {\n",
     "            scan_threshold(remain_topk);\n            DS_PH(13);\n        } else {\n"),
    # 13/15 as plain sums on the one-pass path: the staged population and the
    # sampled cut, divided by the row count to read as values.
    ("            staged = num_staged >= topk && num_staged <= SMEM_INPUT_SIZE;\n",
     "            if (threadIdx.x == 0u) {\n"
     "                atomicAdd(&ds_phase[13], (unsigned long long)num_staged);\n"
     "                atomicAdd(&ds_phase[15], (unsigned long long)s_high_threshold_bin_id);\n"
     "            }\n"
     "            staged = num_staged >= topk && num_staged <= SMEM_INPUT_SIZE;\n"),
    ("            for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)\n"
     "                collect_one(idx, __ldg(input + idx));\n        }\n",
     "            for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)\n"
     "                collect_one(idx, __ldg(input + idx));\n            DS_PH(14);\n        }\n"),
    # 4 = the staged scan: the arena pass over the landing byte's members, the
    # scan over them and the narrow.
    ("            __syncthreads();\n            narrow(remain_topk, s_histogram_buf[1]);\n",
     "            __syncthreads();\n            DS_PH(11);\n"
     "            narrow(remain_topk, s_histogram_buf[1]);\n            DS_PH(4);\n"),
    # 8 = the 4,096 -> 256 fold, booked only on the arm that takes it, so the
    # slot is scan #1's fold alone.
    ("        __syncthreads();\n\n        exclusive_suffix = 0;\n",
     "        __syncthreads();\n        if (fold) DS_PH(8);\n\n        exclusive_suffix = 0;\n"),
    # 9 = the 256-wide scan, on both calls.
    # The refine's own call to the same scan is not this one: the anchor runs
    # from the inserted fold timer down, which exists only in `scan_head`.
    ("        if (fold) DS_PH(8);\n\n        exclusive_suffix = 0;\n"
     "        inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);\n",
     "        if (fold) DS_PH(8);\n\n        exclusive_suffix = 0;\n"
     "        inclusive_suffix = run_cumsum_warp(s_histogram_buf, tx, exclusive_suffix);\n"
     "        DS_PH(9);\n"),
    # 10 = narrow's trailing barrier, on both calls; 12 = the crossing test and
    # the thread-0 sub-bin loop that precede it.
    ("            s_nlist = 0;\n        }\n        __syncthreads();\n    };\n",
     "            s_nlist = 0;\n        }\n        DS_PH(12);\n        __syncthreads();\n"
     "        DS_PH(10);\n    };\n"),
]

for old, new in subs:
    n = s.count(old)
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
open(p, "w").write(s)
print("patched", p)
