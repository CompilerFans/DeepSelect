"""Replace pass 1's per-element histogram atomics with a register accumulate.

Applied to a copy of `topk_coarse12.cuh` (after f32phase_patch.py), this
measures what pass 1 costs *without* the shared atomic: the key transform and
the loads stay, the bin values are summed into a register that is consumed once
so nothing is dead.  The answer is wrong by construction; only slot 0 is read.

    python3 noatomic_patch.py <path-to-a-copy-of-topk_coarse12.cuh>
"""
import re, sys

p = sys.argv[1]
s = open(p).read()

# Pass 1 only: `refine_bin(value) >> kFirstShift` sites inside it.  The two
# later walks (pass 2's collect, the refine) also say `refine_bin`, so the
# substitution is bounded by the pass-1 region: from the loop over `prefix` to
# the NaN-vote barrier.
start = s.index("    for (int index = tid; index < prefix; index += kThreads) {")
end = s.index("    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {")
body = s[start:end]

n = len(re.findall(r"atomicAdd\(&wide_histogram\[[^\]]*\], 1\);", body))
assert n >= 6, n
body = re.sub(r"atomicAdd\(&wide_histogram\[([^\]]*)\], 1\);",
              r"if (\1 == -1) atomicAdd(&wide_histogram[0], 1);", body)
s = s[:start] + body + s[end:]

# The histogram is garbage after that, so the row returns as soon as pass 1 has
# been timed: everything past slot 0 would fault on an out-of-range bin.
s = s.replace("    DS_PH(0);\n", "    DS_PH(0);\n    return true;\n", 1)

open(p, "w").write(s)
print("noatomic-patched", p, n, "atomics")
