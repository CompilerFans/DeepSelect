#!/usr/bin/env python3
"""The oracle the drivers cannot be: each ref's own output, checked against
`torch.topk` in one python process, tie-tolerantly.

Every driver already checks itself against a CPU `std::nth_element` reference it
computes in C++.  That is necessary and not sufficient: a driver can be
self-consistent and still disagree with the framing the library ops use -- a
different prefix window, an off-by-one at the tie boundary, an index transform
applied where the selector contract has none.  Those are exactly what a
same-process `torch.topk` catches and a self-check cannot, because `torch.topk`
shares no code with any of the three.

The rule is `bench_dsa_topk.py`'s `agrees()`:

    equal sorted index sets, OR the differing slots carry equal values,
    because a tied rank may break either way.

What this can and cannot show:

* **Can**: for a driver that dumps its input (`--dump-prefix`), whether the
  indices it produced are a correct top-k *of the scores it actually read*.
  The dumped matrix is loaded here and `torch.topk` runs on exactly it, so a
  disagreement is the kernel's and not the draw's.
* **Cannot**: compare two drivers to each other element-wise.  deep_gemm fills
  its matrix from its own hash-based generator, ds from `mt19937(1234)`, and
  mcoplib from `mt19937` over `uniform_real`, so each cell has three different
  matrices.  A per-driver verdict says nothing about whether two drivers
  *would* agree on one matrix; that needs a shared input file and only two of
  the three drivers can produce one.

    python3 oracle_check.py --dump /tmp/refdump --prefixes dg,ds --shape 6 16384 2048
"""

from __future__ import annotations

import argparse
import os
import sys

import torch


def agrees(our, ref, score, length, topk, pad=-1):
    """bench_dsa_topk.py's rule, verbatim."""
    a = torch.sort(our[:, :topk], dim=-1).values.cpu()
    b = torch.sort(ref[:, :topk], dim=-1).values.cpu()
    if torch.equal(a, b):
        return True
    for i in range(a.shape[0]):
        sa = set(int(v) for v in a[i].tolist()) - {pad}
        sb = set(int(v) for v in b[i].tolist()) - {pad}
        more, less = sa - sb, sb - sa
        if more or less:
            mv = sorted(score[i, idx].item() for idx in more if 0 <= idx < length)
            lv = sorted(score[i, idx].item() for idx in less if 0 <= idx < length)
            if mv != lv:
                return False
    return True


def load(dump_dir, prefix, bs, length, topk):
    """-> ((scores, indices), "") or (None, reason).

    deep_gemm writes `<P>.scores.f32`; ds writes
    `<P>.b<bs>.v<len>.k<k>.scores.f32`.  Both are raw host bytes, no header.
    """
    cands = [
        (os.path.join(dump_dir, f"{prefix}.scores.f32"),
         os.path.join(dump_dir, f"{prefix}.idx.i32")),
        (os.path.join(dump_dir, f"{prefix}.b{bs}.v{length}.k{topk}.scores.f32"),
         os.path.join(dump_dir, f"{prefix}.b{bs}.v{length}.k{topk}.idx.i32")),
    ]
    for sp, ip in cands:
        if not (os.path.isfile(sp) and os.path.isfile(ip)):
            continue
        scores = torch.from_file(sp, dtype=torch.float32,
                                 size=bs * length).reshape(bs, length)
        idx = torch.from_file(ip, dtype=torch.int32, size=bs * topk).reshape(bs, topk)
        return (scores, idx), ""
    return None, (f"no dump for {prefix!r} "
                  f"(looked for {os.path.basename(cands[0][0])} and "
                  f"{os.path.basename(cands[1][0])})")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dump", default="/tmp/refdump")
    ap.add_argument("--prefixes", default="dg,ds")
    ap.add_argument("--shape", nargs=3, type=int, required=True,
                    metavar=("BS", "LEN", "TOPK"))
    args = ap.parse_args()

    bs, length, topk = args.shape
    prefixes = [p.strip() for p in args.prefixes.split(",") if p.strip()]

    print(f"# oracle : torch.topk vs each driver's own dumped output")
    print(f"# shape  : bs={bs} len={length} k={topk}")
    print(f"# rule   : bench_dsa_topk.py agrees() -- equal sorted index sets, or")
    print(f"#          the differing slots carry equal values (a tie may break")
    print(f"#          either way)")
    print(f"# caveat : each driver dumps the matrix IT generated, so this is a")
    print(f"#          per-driver verdict, not a driver-vs-driver one")
    print()

    head = f"{'impl':<8} {'scores input':<44} {'vs torch.topk':<14} note"
    print(head)
    print("-" * len(head))

    rc = 0
    for prefix in prefixes:
        loaded, why = load(args.dump, prefix, bs, length, topk)
        if loaded is None:
            print(f"{prefix:<8} {'--':<44} {'--':<14} {why}")
            rc = 1
            continue
        drv_scores, drv_idx = loaded
        src = [os.path.basename(p) for p in
               (os.listdir(args.dump))]
        sp = next((n for n in src if n.startswith(prefix) and n.endswith(".scores.f32")), "?")
        ref = torch.topk(drv_scores, topk, dim=-1, sorted=False).indices.to(torch.int32)
        ok = agrees(drv_idx, ref, drv_scores, length, topk)
        print(f"{prefix:<8} {sp:<44} {'ok' if ok else 'WRONG':<14} "
              f"{'indices are a correct top-k of the matrix it read' if ok else 'INDEX SET DISAGREES'}")
        if not ok:
            rc = 1
            n_bad = sum(1 for r in range(bs)
                        if not agrees(drv_idx[r:r + 1], ref[r:r + 1],
                                      drv_scores[r:r + 1], length, topk))
            print(f"{'':<8} {f'{n_bad}/{bs} rows disagree':<44}")

    return rc


if __name__ == "__main__":
    sys.exit(main())
