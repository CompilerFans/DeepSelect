"""Pass 1 that counts block maxima instead of elements.

The bf16 row's block-histogram arm in fp32 form: one shared atomic per
16-element block (lane 0 of the block's four lanes) instead of one per element,
with the element histogram left to a second walk over the blocks a cut keeps.
The row returns right after pass 1 is timed, so only slot 0 is meaningful.

    python3 blockcount_patch.py <path-to-a-copy-of-topk_coarse12.cuh>
"""
import re, sys

p = sys.argv[1]
s = open(p).read()
start = s.index("    for (int index = tid; index < prefix; index += kThreads) {")
end = s.index("    if (nan_flag != nullptr && __syncthreads_or((int)nan_found) != 0) {")
body = s[start:end]

# Every per-element atomic in the vector walks goes; the two summary sites below
# take one atomic per block instead.
n = len(re.findall(r"atomicAdd\(&wide_histogram\[[abc][0-3]\], 1\);", body))
assert n >= 12, n
body = re.sub(r"\n *atomicAdd\(&wide_histogram\[[abc][0-3]\], 1\);", "", body)

# The summary stores are already gated on lane 0 of the block; count there.
old_a = """                if ((tid & (kBlockLanes - 1)) == 0) {
                    block_max[vec >> 2] = (uint16_t)ma;
                    block_max[(vec + kThreads) >> 2] = (uint16_t)mb;
                }"""
new_a = """                if ((tid & (kBlockLanes - 1)) == 0) {
                    block_max[vec >> 2] = (uint16_t)ma;
                    block_max[(vec + kThreads) >> 2] = (uint16_t)mb;
                    atomicAdd(&wide_histogram[ma], 1);
                    atomicAdd(&wide_histogram[mb], 1);
                }"""
assert body.count(old_a) == 1
body = body.replace(old_a, new_a)

old_c = """            if ((tid & (kBlockLanes - 1)) == 0) block_max[vec >> 2] = (uint16_t)mc;"""
new_c = """            if ((tid & (kBlockLanes - 1)) == 0) {
                block_max[vec >> 2] = (uint16_t)mc;
                atomicAdd(&wide_histogram[mc], 1);
            }"""
assert body.count(old_c) == 2, body.count(old_c)
body = body.replace(old_c, new_c)

s = s[:start] + body + s[end:]
s = s.replace("    DS_PH(0);\n", "    DS_PH(0);\n    return true;\n", 1)
open(p, "w").write(s)
print("blockcount-patched", p, n, "per-element atomics removed")
