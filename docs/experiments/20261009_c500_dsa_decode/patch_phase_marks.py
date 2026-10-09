#!/usr/bin/env python3
"""Patch a throwaway tree with the in-kernel phase marks Probe `probe_phase_marks.py` reads.

Five `__builtin_mxc_get_time()` marks per thread are carried in registers and
written into `selected[topk-8..topk-4]` at each of the row kernel's exits; the
last mark lands after the refine, which is also the last writer of those
slots.  `topk_kernel_radix` then copies them into the output row, because the
emit it runs right before clobbers the same slots with the answer.

Both inserts are anchored and asserted, so a source drift fails loudly rather
than patching nothing:

    python3 patch_phase_marks.py <tree>        # then: develop.sh in <tree>

The arm is diagnostic: the marks cost registers and the dump costs five shared
loads + a barrier.  Its own kernel time is therefore measured alongside (the
probe prints it) -- 6,151 us against the production 6,144 us on the headline
cell, i.e. within 0.2%, so the phases are read as a split, which is what they
are for.
"""
import sys

TREE = sys.argv[1]
CORE = f"{TREE}/csrc/maca_kernels/xcore1000/radix_core.cuh"
WRAP = f"{TREE}/csrc/maca_kernels/xcore1000/maca_topk.cu"


def sub1(text, old, new, n=1):
    c = text.count(old)
    assert c == n, f"anchor count {c} != {n}: {old[:70]!r}"
    return text.replace(old, new)


src = open(CORE).read()
i = src.index("__device__ __forceinline__ void radix_topk_row_bf16_b(")
j = src.index("__device__ __forceinline__ void radix_topk_row_bf16(", i)
head, body, tail = src[:i], src[i:j], src[j:]

body = sub1(body, "    const uint32_t tx = threadIdx.x;\n", """    const uint32_t tx = threadIdx.x;
    // ---- phase probe (throwaway arm, never lands) ----
    int64_t ds_t_ = __builtin_mxc_get_time();
    uint32_t ds_ph_[5] = {0u, 0u, 0u, 0u, 0u};
    #define PHASE_MARK(k) { const int64_t n_ = __builtin_mxc_get_time(); \\
        ds_ph_[k] = (uint32_t)(n_ - ds_t_); ds_t_ = n_; }
    #define PROBE_DUMP() { if (blockIdx.x < 8u && tx < 5u) \\
        output[topk - 8u + tx] = (int32_t)ds_ph_[tx]; }
""")
body = sub1(body, "    for (uint32_t b = tx; b < kCoarse12Bins; b += BLOCK_SIZE) s_wide[b] = 0;\n    __syncthreads();\n",
                  "    for (uint32_t b = tx; b < kCoarse12Bins; b += BLOCK_SIZE) s_wide[b] = 0;\n    __syncthreads();\n    PHASE_MARK(0);\n")
body = sub1(body, "        *row_nan = __syncthreads_or((int)nan_local) != 0;\n",
                  "        *row_nan = __syncthreads_or((int)nan_local) != 0;\n    PHASE_MARK(1);\n")
body = sub1(body, "        s_wide_above = above;\n    }\n    __syncthreads();\n",
                  "        s_wide_above = above;\n    }\n    __syncthreads();\n    PHASE_MARK(2);\n")
body = sub1(body, "        __syncthreads();\n    }\n\n    {\n        const bool overflow",
                  "        __syncthreads();\n    }\n    PHASE_MARK(3);\n\n    {\n        const bool overflow")
body = sub1(body, "            __syncthreads(); return;\n", "            __syncthreads(); PROBE_DUMP(); return;\n", n=3)
body = sub1(body, "                if (p > 0) output[topk - p] = static_cast<int32_t>(idx);\n            }\n        }\n        __syncthreads();\n    }\n}",
                  "                if (p > 0) output[topk - p] = static_cast<int32_t>(idx);\n            }\n        }\n        __syncthreads();\n        PHASE_MARK(4);\n        PROBE_DUMP();\n    }\n}\n#undef PHASE_MARK\n#undef PROBE_DUMP")
open(CORE, "w").write(head + body + tail)

w = open(WRAP).read()
open(WRAP, "w").write(sub1(w, """            if (RV) out_value_row[i] = __ldg(input_row + src);
        }
        return;
    }
""", """            if (RV) out_value_row[i] = __ldg(input_row + src);
        }
        // ---- phase probe (throwaway arm): carry the row kernel's five phase
        // timestamps out of `selected`, which the emit above just clobbered.
        __syncthreads();
        if (blockIdx.x < 8u && tid < 5u)
            out_index_row[topk - 8u + tid] = (OutIdxT)selected[topk - 8u + tid];
        return;
    }
"""))
print("patched", CORE, "and", WRAP)
