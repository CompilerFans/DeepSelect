"""Measure a contiguous staging walk at the short rows, where none runs today.

Applied after `phase_patch2.py`, this replaces the counting arm's collect with
a staging body gated on the threshold that arm has just computed -- a one-pass
walk with a perfect cut, which is the ceiling a sampled cut would aim at.  The
downstream is left alone, so the answer is wrong; only slot 14's cost is read.

    python3 stageprobe_patch.py <path-to-a-copy-radix_row_bf16.cuh>
"""
import sys

p = sys.argv[1]
s = open(p).read()

old = """            if (use_list) {
                walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, collect_one);
            } else if (input_aligned) {"""
new = """#ifdef DS_STAGEPROBE
            // One contiguous pass that stages everything at or above the
            // threshold, with the bin beside the index in the arena -- the
            // shape a one-pass dataflow would run, with the cut handed to it.
            {
                // The counting arm's own counters and histogram are live here,
                // so the probe borrows `s_nstage` (the staged arm's, unused on
                // this path) and skips the per-staged-element histogram add --
                // ~3% of the elements, not enough to change the walk's cost.
                if (tx == 0) s_nstage = 0;
                __syncthreads();
                const auto stage_probe = [&](uint32_t idx, maca_bfloat16 raw) {
                    const uint32_t key = refine_bin(raw);
                    const uint32_t bin = key >> kCoarse12Shift;
                    if (bin < threshold_bin) return;
                    const uint32_t pos = atomicAdd(&s_nstage, 1u);
                    if (pos < SMEM_INPUT_SIZE)
                        s_input_flat[pos] = (bin << kStagedIdxBits) | idx;
                };
                if (input_aligned) {
                    for (uint32_t idx = tx * 8; idx < vec_len; idx += BLOCK_SIZE * 8) {
                        const uint4 v = __ldg(reinterpret_cast<const uint4*>(input + idx));
                        const maca_bfloat16* hh = reinterpret_cast<const maca_bfloat16*>(&v);
                        #pragma unroll
                        for (int i = 0; i < 8; i++) stage_probe(idx + (uint32_t)i, hh[i]);
                    }
                } else {
                    for (uint32_t idx = tx; idx < length; idx += BLOCK_SIZE)
                        stage_probe(idx, __ldg(input + idx));
                }
                const uint32_t tail_start = input_aligned ? vec_len : length;
                for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
                    stage_probe(idx, __ldg(input + idx));
            }
            (void)collect_one;
            DS_PH(14);
        }
#else
            if (use_list) {
                walk_kept_blocks<BLOCK_SIZE>(input, block_list, nlist, collect_one);
            } else if (input_aligned) {"""
n = s.count(old)
assert n == 1, (n, old[:70])
s = s.replace(old, new)

old2 = """            for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
                collect_one(idx, __ldg(input + idx));
            DS_PH(14);
        }"""
new2 = """            for (uint32_t idx = tail_start + tx; idx < length; idx += BLOCK_SIZE)
                collect_one(idx, __ldg(input + idx));
            DS_PH(14);
        }
#endif"""
n = s.count(old2)
assert n == 1, (n, old2[:70])
s = s.replace(old2, new2)

open(p, "w").write(s)
print("stageprobe-patched", p)
