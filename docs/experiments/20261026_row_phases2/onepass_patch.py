"""Turn the staged arm's list walk into a contiguous full-row staging walk.

Applied *after* `phase_patch2.py`, to a copy of `radix_row_bf16.cuh`, this
answers one question only: what does staging cost when the row is read
contiguously instead of block-by-block through the kept-block list?  That is
the ceiling of a one-pass dataflow, which has no cut to walk to and therefore
no list.

The cut comes from a device global the probe fills with the bin of the
1.15 x topk-th largest element, so the staged set is a superset of the answer
and the downstream (arena histogram, scan, refine, emit) is untouched and
still exact.

    python3 onepass_patch.py <path-to-the-copy-radix_row_bf16.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()

subs = [
    # A device global for the cut, beside the phase counters.
    ("namespace topk_select_radix {\n",
     "#ifdef DS_ONEPASS\n"
     "__device__ unsigned int ds_probe_cut;\n"
     "#define DS_PROBE_CUT ds_probe_cut\n"
     "#else\n"
     "#define DS_PROBE_CUT s_high_threshold_bin_id\n"
     "#endif\n\n"
     "namespace topk_select_radix {\n"),
    # The staged arm takes the probe's cut instead of the block cut.
    ("        if (use_list) {\n            const uint32_t cut = s_high_threshold_bin_id;\n",
     "        if (use_list) {\n            const uint32_t cut = DS_PROBE_CUT;\n"),
    # One contiguous walk over the row's whole blocks replaces the list walk.
    ("            walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, stage_one);\n",
     "#ifdef DS_ONEPASS\n"
     "            for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {\n"
     "                const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));\n"
     "                const maca_bfloat16* hh = reinterpret_cast<const maca_bfloat16*>(&v);\n"
     "                #pragma unroll\n"
     "                for (int i = 0; i < 8; i++) stage_one(idx + (uint32_t)i, hh[i]);\n"
     "            }\n"
     "#else\n"
     "            walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, stage_one);\n"
     "#endif\n"),
]

for old, new in subs:
    n = s.count(old)
    assert n == 1, (n, old[:70])
    s = s.replace(old, new)
open(p, "w").write(s)
print("onepass-patched", p)
