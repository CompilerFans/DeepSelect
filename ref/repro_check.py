#!/usr/bin/env python3
"""Is a driver's output reproducible run to run on the same input?

This started as an attempt to hand two drivers ONE matrix so they could be
compared to each other.  That is not possible through the dumps, and the
attempt is worth recording rather than deleting:

* `--dump-prefix P` makes a driver write the scores it generated **and** the
  indices it selected, then select from the regenerated matrix.  There is no
  flag that makes a driver *read* P and select from it.
* So copying a different matrix onto P changes the file and changes nothing
  about the selection: each driver selects from its own data every time.
  A driver-vs-driver comparison built that way is comparing two different
  matrices and says nothing.  (`tests/bench_dsa_topk.py` is the correct place
  for a cross-backend comparison, because there all three backends are handed
  the same tensor through the same API.)

What the dumps *can* answer is the question below, and it turns out to matter:
a driver is only usable as a reference if re-running it reproduces itself.
Two runs at the same shape must regenerate the same scores (the generator is a
function of the shape) and must produce the same indices.  The first holds by
construction; the second is a real property of the kernel and does not always.

    python3 repro_check.py --shape 6 65536 2048
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys

import torch


def same_selection(a, b, score, length, topk, pad=-1):
    """Per-row value multiset equality -- the rule that actually holds.

    NOT `bench_dsa_topk.py`'s `agrees()`.  That rule ("equal sorted index sets,
    or the differing slots carry equal values") is a slot-wise test, and it is
    unsound on these dumps: at 16384 distinct values in a 16384-long row, 8466
    slots reordering is not tie noise, those are distinct elements.

    What holds is the per-row value multiset, which subsumes both the slot-wise
    question and the tie question and degenerates correctly for `length <= topk`.
    Both sides go through the sort explicitly: the reference for a top-k comes
    back descending while the kernel's selection is unordered, so comparing
    `sort(sel)` against an unsorted `row_desc[:k]` reports a failure on every
    row.  ds's own README flags that footgun; this is the same trap.
    """
    sa = torch.sort(a[:, :topk], dim=-1).values.float()
    sb = torch.sort(b[:, :topk], dim=-1).values.float()
    sa[sa < 0] = pad
    sb[sb < 0] = pad
    if not torch.equal(sa, sb):
        return False
    # ...and the values behind those sets must agree too, so a whole-selection
    # swap that happens to keep the index multiset is still caught.
    ia = a[:, :topk].to(torch.int64).clamp(0, length - 1)
    ib = b[:, :topk].to(torch.int64).clamp(0, length - 1)
    va = torch.gather(score[:, :length], 1, ia)
    vb = torch.gather(score[:, :length], 1, ib)
    return torch.equal(torch.sort(va, dim=-1).values, torch.sort(vb, dim=-1).values)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dg", default="/tmp/dsref_build/deep_gemm/main")
    ap.add_argument("--ds", default="/tmp/dsref_build/ds/main")
    ap.add_argument("--work", default="/tmp/reprocheck")
    ap.add_argument("--shape", nargs=3, type=int, required=True,
                    metavar=("BS", "LEN", "TOPK"))
    ap.add_argument("--dev", default="1")
    args = ap.parse_args()
    bs, length, topk = args.shape

    env = dict(os.environ)
    maca = env.get("MACA_PATH", "/opt/maca")
    env["MACA_PATH"] = maca
    env["LD_LIBRARY_PATH"] = ":".join(
        [os.path.join(maca, p) for p in ("lib", "mxgpu_llvm/lib")]
        + [env.get("LD_LIBRARY_PATH", "")])
    env["CUDA_VISIBLE_DEVICES"] = str(args.dev)

    shutil.rmtree(args.work, ignore_errors=True)
    os.makedirs(args.work, exist_ok=True)

    def scores_of(p):
        for c in (p + ".scores.f32", p + f".b{bs}.v{length}.k{topk}.scores.f32"):
            if os.path.isfile(c):
                return c
        return None

    def idx_of(p):
        for c in (p + ".idx.i32", p + f".b{bs}.v{length}.k{topk}.idx.i32"):
            if os.path.isfile(c):
                return c
        return None

    print(f"# run-to-run reproducibility | shape bs={bs} len={length} k={topk}")
    print(f"# a driver that cannot reproduce itself is not usable as a reference")
    print()
    head = ("impl     scores identical   indices identical  multiset-equal\n"
            "         (regenerated)      (raw slot order)   (by value)")
    print(head)
    print("-" * 64)

    for name, binary in (("deep_gemm", args.dg), ("ds", args.ds)):
        if not os.path.isfile(binary):
            print(f"{name:<8} {'(no binary)':<18}")
            continue
        loaded = []
        for tag in ("a", "b"):
            pref = os.path.join(args.work, f"{name}_{tag}")
            p = subprocess.run([binary, "--dump-prefix", pref, str(bs),
                                str(length), str(topk), "2"],
                               env=env, stdout=subprocess.PIPE,
                               stderr=subprocess.STDOUT, timeout=3600)
            if p.returncode != 0:
                print(f"{name:<8} run {tag} failed rc={p.returncode}: "
                      f"{p.stdout.decode('utf-8','replace').strip().splitlines()[-1][:70]}")
                loaded = None
                break
            loaded.append((
                torch.from_file(scores_of(pref), dtype=torch.float32,
                                size=bs * length).reshape(bs, length),
                torch.from_file(idx_of(pref), dtype=torch.int32,
                                size=bs * topk).reshape(bs, topk)))
        if not loaded:
            continue
        (m1, i1), (m2, i2) = loaded
        same_data = torch.equal(m1, m2)
        same_idx = torch.equal(i1, i2)
        tie_ok = same_selection(i1, i2, m1, length, topk)
        print(f"{name:<8} {str(same_data):<18} {str(same_idx):<18} {str(tie_ok):<14}")
        if same_data and not same_idx:
            n = int((i1 != i2).any(dim=1).sum().item())
            slots = int((i1 != i2).sum().item())
            print(f"{'':<8}   same matrix, {slots} of {bs * topk} slots differ "
                  f"across {n}/{bs} rows; multiset rule: "
                  f"{'the same values, different order' if tie_ok else 'DIFFERENT SELECTIONS'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
