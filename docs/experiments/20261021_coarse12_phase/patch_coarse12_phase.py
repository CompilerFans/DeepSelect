"""Insert a clock64 phase probe into topk_coarse12_row (scratch tree only).

Phases (output[top_k-16 .. top_k-10], deltas from kernel start, clock ticks):
  0 start, 1 pass-1 histogram+NaN done, 2 fold/cumsum/narrow done,
  3 pass-2 collect done, 4 overflow path end, 5 arena path entry, 6 end.
Path code in output[top_k-8]: 1 exact-fit filter, 2 overflow rescan,
3 arena remain==0, 4 arena main.
Data slots: -9 arena round 0, -7 num_input[0] (candidate count),
-6 round at emit, -5 requested_topk, -4 threshold_bin_id (12-bit),
-3 post-narrow remaining topk, -2 counter, -1 arena rounds 1+2.

Usage:  python3 patch_coarse12_phase.py <scratch-tree>

Edits `<scratch-tree>/csrc/maca_kernels/common/topk_coarse12.cuh`.  Every anchor
is asserted by count; on a tree whose text has moved the script exits without
writing rather than patching something else.

The probe writes into the last 16 slots of the row's `output` (the
`coarse12_cols` scratch row), so the probe tree's output is NOT a correct
top-k.  That is safe on the official shape (return_value=False, sorted_value=
False): the contract half's emit writes indices only -- `if (RV)` is a
compile-time false -- and its rerank test is `src < 0`, which the masked
non-negative values never trip.  Do not run the probe tree with return_value
or sorted_value set: `__ldg(input_row + src)` would then dereference a clock.
"""
import io
import os
import sys

root = sys.argv[1]
p = os.path.join(root, "csrc/maca_kernels/common/topk_coarse12.cuh")
src = io.open(p, encoding="utf-8").read()


def sub(old, new, count=1):
    global src
    n = src.count(old)
    if n != count:
        sys.exit("anchor %r found %d (want %d)" % (old[:70], n, count))
    src = src.replace(old, new)


MACRO = """#define DS12_EMIT(code)                                                          \\
    do { if (tid == 0) {                                                         \\
        ds_pp[6] = clock64();                                                    \\
        for (int ds_i = 0; ds_i < 7; ds_i++)                                     \\
            output[requested_topk - 16 + ds_i] =                                 \\
                (int32_t)((uint32_t)(ds_pp[ds_i] - ds_pp[0]) & 0x7FFFFFFFu);     \\
        output[requested_topk - 9] =                                             \\
            (int32_t)((uint32_t)(ds_rr[0] - ds_pp[5]) & 0x7FFFFFFFu);            \\
        output[requested_topk - 8] = (code);                                     \\
        output[requested_topk - 7] = num_input[0];                               \\
        output[requested_topk - 6] = ds_round;                                   \\
        output[requested_topk - 5] = requested_topk;                             \\
        output[requested_topk - 4] = threshold_bin_id;                           \\
        output[requested_topk - 3] = ds_topk0;                                   \\
        output[requested_topk - 2] = counter;                                    \\
        output[requested_topk - 1] =                                             \\
            (int32_t)((uint32_t)(ds_rr[2] - ds_rr[0]) & 0x7FFFFFFFu);            \\
    } } while (0)
"""

# the macro, just before the row entry that uses it
sub("__device__ __forceinline__ bool topk_coarse12_row(",
    MACRO +
    "__device__ __forceinline__ bool topk_coarse12_row(")

# probe state at the top of the row
sub("""    const int tid = threadIdx.x;
    const unsigned int u_length = static_cast<unsigned int>(length);
""",
    """    const int tid = threadIdx.x;
    const unsigned int u_length = static_cast<unsigned int>(length);
    __shared__ long long ds_pp[8];
    __shared__ long long ds_rr[3];
    int ds_round = -1;
    int ds_topk0 = 0;
    if (tid == 0) {
        for (int ds_i = 0; ds_i < 8; ds_i++) ds_pp[ds_i] = 0;
        for (int ds_i = 0; ds_i < 3; ds_i++) ds_rr[ds_i] = 0;
        ds_pp[0] = clock64();
    }
""")

# 1: end of pass 1 (histogram + NaN vote), after its barrier
sub("""    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {
        if (tid == 0) atomicOr(nan_flag, 1);
    }
    __syncthreads();
""",
    """    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {
        if (tid == 0) atomicOr(nan_flag, 1);
    }
    __syncthreads();
    if (tid == 0) ds_pp[1] = clock64();
""")

# 2: end of fold + cumsum + 12-bit narrow
sub("""    __syncthreads();

    const int wide_threshold = threshold_bin_id;
    int topk = requested_topk - threshold_exclusive_count;
""",
    """    __syncthreads();
    if (tid == 0) ds_pp[2] = clock64();

    const int wide_threshold = threshold_bin_id;
    int topk = requested_topk - threshold_exclusive_count;
    ds_topk0 = topk;
""")

# exact-fit filter: emit code 1
sub("""        for (unsigned int index = tid; index < u_length; index += kThreads) {
            if (static_cast<int>(refine_bin(__ldg(input + index)) >> kFirstShift) > wide_threshold) {
                const int position = atomicAdd(&counter, 1);
                output[position] = static_cast<int32_t>(index);
            }
        }
        __syncthreads();
        return true;
    }
""",
    """        for (unsigned int index = tid; index < u_length; index += kThreads) {
            if (static_cast<int>(refine_bin(__ldg(input + index)) >> kFirstShift) > wide_threshold) {
                const int position = atomicAdd(&counter, 1);
                output[position] = static_cast<int32_t>(index);
            }
        }
        __syncthreads();
        DS12_EMIT(1);
        return true;
    }
""")

# 3: end of pass-2 collect
sub("""#undef DG12_COLLECT
    __syncthreads();
""",
    """#undef DG12_COLLECT
    __syncthreads();
    if (tid == 0) ds_pp[3] = clock64();
""")

# overflow: early return inside the round loop
sub("            if (topk == 0 || round == 2) return true;\n",
    "            if (topk == 0 || round == 2) { ds_round = round; DS12_EMIT(2); return true; }\n")

# overflow: end of the block (final return)
sub("""            prefix_bits += radix_bits;
        }
        return true;
    }
""",
    """            prefix_bits += radix_bits;
        }
        DS12_EMIT(2);
        return true;
    }
""")

# 5: arena entry
sub("""    // The arena path: three rounds over the *staged candidates only*, ping-
""",
    """    if (tid == 0) ds_pp[5] = clock64();

    // The arena path: three rounds over the *staged candidates only*, ping-
""")

# arena: remain==0 early return
sub("""            __syncthreads();
            return true;
        }
""",
    """            __syncthreads();
            ds_round = round;
            DS12_EMIT(3);
            return true;
        }
""")

# arena: per-round end marks and the function's final emit
sub("""        __syncthreads();
    }
    return true;
}
""",
    """        __syncthreads();
        if (tid == 0) ds_rr[round] = clock64();
    }
    DS12_EMIT(4);
    return true;
}
""")

io.open(p, "w", encoding="utf-8").write(src)
print("patched", p)
