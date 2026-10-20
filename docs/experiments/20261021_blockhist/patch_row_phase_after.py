"""Insert a clock64 phase probe into radix_topk_row_bf16_b -- the POST-block-hist
tree (`095224c` and later; anchors measured against 43e566a).

Phases: 0 entry, 1 first walk (block maxima + NaN vote) done, 2 first fold+narrow
done, 3 kept-block list done, 4 second element walk + its fold+narrow done,
5 collect (pass 2) done, 6 refine cumsum done, 7 end.  Path code in the ninth
slot: 1 exact-fit filter, 2 overflow rescan, 3 arena remain==0, 4 arena main.

Usage:  python patch_row_phase_after.py <scratch-tree>

It edits `<scratch-tree>/csrc/maca_kernels/common/radix_row_bf16.cuh`.  Every
anchor is exact-count-checked and the script exits without writing if one moved.
The probe writes eight deltas plus the code into `output[topk-16 .. topk)`, so
the probe tree is for timing only -- its output is not a correct top-k.

Not to be confused with the archived pre-block-histogram probe
(`patch_row_phase.py` + `row_phase.py`), which has six phases and anchored on
`e168b4e`; this one splits the first walk from the second, which is the phase
the block histogram created.
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
        ds_pp[7] = clock64();                                                    \\
        for (int ds_i = 0; ds_i < 8; ds_i++)                                     \\
            output[topk - 16 + ds_i] = (int32_t)(ds_pp[ds_i] - ds_pp[0]);         \\
        output[topk - 8] = (code); } } while (0)
"""

sub("template <uint32_t BLOCK_SIZE, bool kNan = false, bool kSkip = true>\n"
    "__device__ __forceinline__ void radix_topk_row_bf16_b(",
    MACRO +
    "template <uint32_t BLOCK_SIZE, bool kNan = false, bool kSkip = true>\n"
    "__device__ __forceinline__ void radix_topk_row_bf16_b(")

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

# 1: end of the first walk and its NaN vote
sub("""    __syncthreads();
    // The histogram is complete and the CTA's NaN vote is in, on one barrier --""",
    """    __syncthreads();
    if (tx == 0) ds_pp[1] = clock64();
    // The histogram is complete and the CTA's NaN vote is in, on one barrier --""")

# 2: end of the first fold + narrow (scan_threshold is barrier-terminated)
sub("""    scan_threshold(remain_topk);

    // ── the kept-block list""",
    """    scan_threshold(remain_topk);
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

# 4: end of the second walk + second fold/narrow.  The block-histogram arm
# ends on the second scan's barrier; the other arms do not, so one barrier
# here makes mark 4 a phase boundary on every path.
sub("""        scan_threshold(remain_topk);
    }

    const uint32_t coarse_threshold = s_high_threshold_bin_id;""",
    """        scan_threshold(remain_topk);
    }
    __syncthreads();
    if (tx == 0) ds_pp[4] = clock64();

    const uint32_t coarse_threshold = s_high_threshold_bin_id;""")

# 5: end of the collect
sub("""        for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
            collect_one(idx, __ldg(input + idx));
        __syncthreads();
    }
""",
    """        for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
            collect_one(idx, __ldg(input + idx));
        __syncthreads();
        if (tx == 0) ds_pp[5] = clock64();
    }
""")

# 6: refine cumsum
sub("""        __syncthreads();
        const auto threshold_bin = s_threshold_bin_id;""",
    """        __syncthreads();
        if (tx == 0) ds_pp[6] = clock64();
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
print("row phase probe (post-block-hist) applied")
