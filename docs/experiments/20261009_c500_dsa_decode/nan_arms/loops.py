#!/usr/bin/env python3
"""Loop structure of one device function: every backward branch region,
largest first.  Labels are `.LBB<n>_<m>:` and branch operands carry no dot."""
import re
import sys

path, want = sys.argv[1], sys.argv[2]
s = open(path, errors="replace").read()
i = s.find("\n" + want + ":")
j = s.find("\n\t.section", i + 10)
body = s[i:j if j > 0 else len(s)].split("\n")

LBL = re.compile(r"^\.?(LBB\d+_\d+):")
BRA = re.compile(r"^bra\S*\s+\.?(LBB\d+_\d+)\b")

rows = []                       # (index, text, is_instruction)
for k, l in enumerate(body):
    t = l.strip()
    if not t or t.startswith(";"):
        continue
    m = LBL.match(t)
    rows.append((k, t, m.group(1) if m else None))

at = {lab: idx for idx, (_, _, lab) in enumerate(rows) if lab}
loops = []
for idx, (_, t, _) in enumerate(rows):
    m = BRA.match(t)
    if not m:
        continue
    tgt = m.group(1)
    if tgt not in at or at[tgt] >= idx:
        continue
    seg = [x for x in rows[at[tgt]:idx + 1] if x[2] is None]
    st = " ".join(x[1] for x in seg)
    loops.append((len(seg), tgt, st.count("ldg_"), st.count("sm_add_u32"),
                  st.count("barrier"), st.count("sand_b64")))
loops.sort(reverse=True, key=lambda x: x[0])
print(f"{want.split('topk_kernel_radix')[-1][:44]}  total {len([r for r in rows if r[2] is None])} insns")
for n, tgt, nld, nadd, nbar, nmask in loops[:8]:
    print(f"   loop @{tgt:<10} {n:>5} insns  ldg={nld:<3} sm_add={nadd:<3} "
          f"barrier={nbar:<3} sand={nmask}")
