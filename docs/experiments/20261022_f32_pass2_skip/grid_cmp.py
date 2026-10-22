# Compare two test.py --perf-only logs cell for cell.  Rows are keyed by
# (b, V, k, occurrence), because the grid re-measures some shapes.
import re, sys

def cells(path):
    out, cur = [], None
    for line in open(path):
        m = re.search(r"batch_size=(\d+), vocab_size=(\d+), topk=(\d+)", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)), int(m.group(3)))
            continue
        m = re.match(r"^topk\s*:\s*([0-9.]+) us", line)
        if m and cur:
            out.append((cur, float(m.group(1))))
            cur = None
    return out

a, b = cells(sys.argv[1]), cells(sys.argv[2])
assert len(a) == len(b), (len(a), len(b))
seen, ta, tb = {}, 0.0, 0.0
print("%-24s %10s %10s %8s" % ("cell", "A us", "B us", "delta"))
for ((ka, va), (kb, vb)) in zip(a, b):
    assert ka == kb, (ka, kb)
    ta += va; tb += vb
    n = seen[ka] = seen.get(ka, 0) + 1
    print("b%-5d V%-7d k%-3d #%d %10.1f %10.1f %7.1f%%"
          % (ka[0], ka[1], ka[2], n, va, vb, 100 * (vb - va) / va))
print("%-24s %10.1f %10.1f %7.2f%%  (grid total)" % ("", ta, tb, 100 * (tb - ta) / ta))
