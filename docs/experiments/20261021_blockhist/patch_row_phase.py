"""Insert a clock64 phase probe into radix_topk_row_bf16_b (scratch tree only).

Phases: 0 start, 1 pass-1 walk done, 2 fold+narrow done, 3 kept-block list done,
4 collect (pass 2) done, 5 refine cumsum done, 6 end.  Path code in the 8th
slot: 1 exact-fit filter, 2 overflow rescan, 3 arena remain==0, 4 arena main.

Archived from the run that produced the phase table in this directory's README
§0.  Usage:  python patch_row_phase.py <scratch-tree>

It edits `<scratch-tree>/csrc/maca_kernels/common/radix_row_bf16.cuh` and
applies to the PRE-block-histogram tree (`e168b4e`, the README's "before" arm):
every anchor below is a pre-`095224c` line.  On a later tree `sub()` finds a
different count and exits without writing -- that is the intended check, not a
failure to work around.

Only the tree path was parameterized at archive time; the anchors and the
inserted code are exactly what ran.  The probe writes its deltas into the last
16 slots of each row's output (`output[topk-16 .. topk)`), so the probe tree is
for timing only -- its output is not a correct top-k.
"""
import io
import os
import sys

root = sys.argv[1]
p = os.path.join(root, "csrc/maca_kernels/common/radix_row_bf16.cuh")
src = io.open(p, encoding="utf-8").read()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        sys.exit("anchor %r found %d (want %d)" % (old[:70], n, count))
    src = src.replace(old, new)


MACRO = """#define DS_ROW_EMIT(code)                                                        \\
    do { if (tx == 0) {                                                          \\
        ds_pp[6] = clock64();                                                    \\
        for (int ds_i = 0; ds_i < 7; ds_i++)                                     \\
            output[topk - 16 + ds_i] = (int32_t)(ds_pp[ds_i] - ds_pp[0]);         \\
        output[topk - 9] = (int32_t)(ds_pp[6] - ds_pp[0]);                        \\
        output[topk - 8] = (code); } } while (0)
"""

# the macro, just before the function that uses it
sub("template <uint32_t BLOCK_SIZE, bool kNan = false, bool kSkip = true>\n"
    "__device__ __forceinline__ void radix_topk_row_bf16_b(",
    MACRO +
    "template <uint32_t BLOCK_SIZE, bool kNan = false, bool kSkip = true>\n"
    "__device__ __forceinline__ void radix_topk_row_bf16_b(")

# probe array at the top of the kernel
sub("""    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = topk;
    auto& s_histogram = s_histogram_buf[0];
""",
    """    const uint32_t tx = threadIdx.x;
    uint32_t remain_topk = topk;
    auto& s_histogram = s_histogram_buf[0];
    __shared__ long long ds_pp[8];
    if (tx == 0) {
        for (int ds_i = 0; ds_i < 8; ds_i++) ds_pp[ds_i] = 0;
        ds_pp[0] = clock64();
    }
""")

# 1: end of pass 1
sub("""    __syncthreads();
    // The histogram is complete and the CTA's NaN vote is in, on one barrier --""",
    """    __syncthreads();
    if (tx == 0) ds_pp[1] = clock64();
    // The histogram is complete and the CTA's NaN vote is in, on one barrier --""")

# 2: end of the narrow
sub("""    __syncthreads();

    // ── the kept-block list""",
    """    __syncthreads();
    if (tx == 0) ds_pp[2] = clock64();

    // ── the kept-block list""")

# 3: end of the list build
sub("""        use_list = list_cap > 0 && nlist <= list_cap;
    }
""",
    """        use_list = list_cap > 0 && nlist <= list_cap;
    }
    if (tx == 0) ds_pp[3] = clock64();
""")

# 4: end of the collect
sub("""        for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
            collect_one(idx, __ldg(input + idx));
        __syncthreads();
    }
""",
    """        for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
            collect_one(idx, __ldg(input + idx));
        __syncthreads();
        if (tx == 0) ds_pp[4] = clock64();
    }
""")

# 5: refine cumsum
sub("""        __syncthreads();
        const auto threshold_bin = s_threshold_bin_id;""",
    """        __syncthreads();
        if (tx == 0) ds_pp[5] = clock64();
        const auto threshold_bin = s_threshold_bin_id;""")

# emits
sub("""            __syncthreads(); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;""",
    """            __syncthreads(); DS_ROW_EMIT(1); return;
        }
        __syncthreads();
        if (tx < RADIX + 1) s_histogram[tx] = 0;""")

sub("""                emit_member(idx, __ldg(input + idx));
            __syncthreads(); return;""",
    """                emit_member(idx, __ldg(input + idx));
            __syncthreads(); DS_ROW_EMIT(2); return;""")

sub("""            __syncthreads(); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {""",
    """            __syncthreads(); DS_ROW_EMIT(3); return;
        }
        __syncthreads();
        for (uint32_t i = tx; i < num; i += BLOCK_SIZE) {""")

sub("""        __syncthreads();
    }
}

}  // namespace topk_select_radix""",
    """        __syncthreads();
        DS_ROW_EMIT(4);
    }
}

#undef DS_ROW_EMIT

}  // namespace topk_select_radix""")

io.open(p, "w", encoding="utf-8").write(src)
print("row phase probe applied")
