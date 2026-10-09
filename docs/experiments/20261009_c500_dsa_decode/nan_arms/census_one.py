#!/usr/bin/env python3
"""Instruction census of one device function in a mxcc -aop -S listing."""
import re
import sys

path, want = sys.argv[1], sys.argv[2]
s = open(path, errors="replace").read()

i = s.find("\n" + want + ":")
if i < 0:
    print(f"{want}: NOT FOUND in {path}")
    sys.exit(1)
j = s.find("\n\t.section", i + 10)
if j < 0:
    j = len(s)
body = s[i:j].split("\n")

ins = [(k, l.strip()) for k, l in enumerate(body)
       if l.strip() and not l.strip().startswith((";", "."))]
print(f"{want.split('topk_kernel_radix')[-1][:40]}")
print(f"  function: {len(ins)} instructions")

OPS = ["ldg_b128", "ldg_b64", "ldg_b32", "ldg_u16", "ldg_", "stg_b32", "stg_",
       "sm_add_u32", "barrier", "sand_b64", "sor_b64", "sicmp", "uicmp",
       "bsm_bperm", "sm_bperm", "shared", "cmp_gt", "cmp_eq"]
txt = " ".join(t for _, t in ins)
print("  " + "  ".join(f"{o}={txt.count(o)}" for o in OPS))

lab = {}
for k, t in ins:
    if t.endswith(":"):
        lab[t[:-1]] = k
loops = []
for k, t in ins:
    m = re.match(r"bra\S*\s+(LBB\d+_\d+)", t)
    if not m or m.group(1) not in lab:
        continue
    start = lab[m.group(1)]
    if start >= k:
        continue
    seg = [x for x in ins if start <= x[0] <= k]
    st = " ".join(x[1] for x in seg)
    loops.append((len(seg), st.count("ldg_"), st.count("sm_add_u32"),
                  st.count("sand_b64") + st.count("sor_b64"),
                  st.count("barrier"), m.group(1)))
loops.sort(reverse=True, key=lambda x: x[0])
for n, nld, nadd, nmask, nbar, name in loops:
    print(f"    loop {name:<12} {n:>5} insns   ldg={nld:<3} sm_add={nadd:<3} "
          f"maskops={nmask:<3} barrier={nbar}")
