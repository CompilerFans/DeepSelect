"""Replace the staged arm's list walk with a contiguous ADJ4 walk that stages.

Applied after `phase_patch2.py`.  This is the walk a one-pass dataflow would
run: the row is read contiguously in the summary walk's own geometry (four
adjacent blocks per group-iteration, the wide transform, the per-lane max in
hand), and an eight-element vector only descends to the scalar staging path
when its own max key is at or above the cut.

The cut comes from `ds_probe_cut`, which the probe fills with the bin of the
1.15 x topk-th largest element; `with_cut_patch.py` provides it.

    python3 adj4_patch.py <path-to-the-copy-radix_row_bf16.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()

old = "            walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, stage_one);\n"
new = """#ifdef DS_ONEPASS
            // Four adjacent blocks per group-iteration, one `uint4` per lane:
            // the summary walk's geometry, with a staging descent behind the
            // per-lane max it already computes.
            for (uint32_t blk = 8u * (tx >> 4) + (((tx >> 3) & 1u) << 2);
                 ((blk + 4u) << 6) <= vec_len; blk += 4u * (BLOCK_SIZE / 8u)) {
                uint32_t k0 = 0, k1 = 0, k2 = 0, k3 = 0;
                hist_add_bf16_wide<false, false>(s_wide, input, (blk << 6) + lane * 8u, &k0);
                hist_add_bf16_wide<false, false>(s_wide, input, ((blk + 1u) << 6) + lane * 8u, &k1);
                hist_add_bf16_wide<false, false>(s_wide, input, ((blk + 2u) << 6) + lane * 8u, &k2);
                hist_add_bf16_wide<false, false>(s_wide, input, ((blk + 3u) << 6) + lane * 8u, &k3);
                const uint32_t ks[4] = {k0, k1, k2, k3};
                #pragma unroll
                for (int q = 0; q < 4; q++) {
                    if ((ks[q] >> kCoarse12Shift) < cut) continue;
                    const uint32_t base = ((blk + (uint32_t)q) << 6) + lane * 8u;
                    const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + base));
                    const maca_bfloat16* h = reinterpret_cast<const maca_bfloat16*>(&v);
                    #pragma unroll
                    for (int i = 0; i < 8; i++) stage_one(base + (uint32_t)i, h[i]);
                }
            }
#else
            walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, stage_one);
#endif
"""

n = s.count(old)
assert n == 1, (n, old[:70])
s = s.replace(old, new)

# The cut, at file scope, filled by the probe.
old3 = "namespace topk_select_radix {\n"
new3 = ("#ifdef DS_ONEPASS\n"
        "__device__ unsigned int ds_probe_cut;\n"
        "#define DS_PROBE_CUT ds_probe_cut\n"
        "#else\n"
        "#define DS_PROBE_CUT s_high_threshold_bin_id\n"
        "#endif\n\n"
        "namespace topk_select_radix {\n")
n = s.count(old3)
assert n == 1, (n, old3[:70])
s = s.replace(old3, new3)

# ... and the staged arm reads it instead of the block cut.
old4 = ("        if (use_list) {\n"
        "            const uint32_t cut = s_high_threshold_bin_id;\n")
new4 = ("        if (use_list) {\n"
        "            const uint32_t cut = DS_PROBE_CUT;\n")
n = s.count(old4)
assert n == 1, (n, old4[:70])
s = s.replace(old4, new4)

# `lane` is not declared in this scope by the production code; the ADJ4 loop
# needs it.
old2 = "            const auto stage_one = [&](uint32_t idx, maca_bfloat16 raw) {\n"
new2 = ("            const uint32_t lane = tx & 7u;\n"
        "            const auto stage_one = [&](uint32_t idx, maca_bfloat16 raw) {\n")
n = s.count(old2)
assert n == 1, (n, old2[:70])
s = s.replace(old2, new2)

open(p, "w").write(s)
print("adj4-patched", p)
