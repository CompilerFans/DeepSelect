"""Contract probe for the row kernel's staged arm.

The staged arm runs only where the block histogram does -- `topk <= length / 256`
-- so it is the long-row cells, and the official correctness draw is capped at
2^26 elements, which excludes most of them.  This draws its own grid and checks
the contract directly: every emitted index is in range and unique, its value is
the input's, and the multiset of selected values is the multiset of the k
largest (which is a value-level statement, so ties cannot make it ambiguous).

    python stage_probe.py [--backend maca_c] [--device 1]
"""

import argparse
import sys

import torch

import deep_select


def values_of(x, idx):
    return x.gather(1, idx.long())


def check(name, x, k, out_val, out_idx):
    b, v = x.shape
    idx = out_idx.long()
    problems = []
    if idx.shape != (b, k):
        problems.append(f"shape {tuple(idx.shape)} != {(b, k)}")
    if idx.numel() and (idx.min() < 0 or idx.max() >= v):
        problems.append(f"index out of range [{idx.min()}, {idx.max()}]")
    for r in range(b):
        if torch.unique(idx[r]).numel() != k:
            problems.append(f"row {r}: duplicate index")
            break
    sel = values_of(x, idx)
    if out_val is not None:
        if not torch.equal(sel, out_val):
            problems.append("returned values != input[index]")
    ref = torch.topk(x, k, dim=1).values
    # Values, not indices: the contract fixes the selected set up to ties.
    if not torch.equal(torch.sort(sel, dim=1).values,
                       torch.sort(ref, dim=1).values):
        bad = (torch.sort(sel, dim=1).values
               != torch.sort(ref, dim=1).values).sum().item()
        problems.append(f"{bad} selected values differ from the k largest")
    if problems:
        print(f"  FAIL {name}: {'; '.join(problems)}")
        return False
    print(f"  pass {name}")
    return True


def make(kind, b, v, gen):
    if kind == "normal":
        x = torch.randn(b, v, device="cuda", generator=gen)
    elif kind == "ties":
        # Few distinct values, so the threshold bin is crowded and the arena's
        # capacity is what decides the arm.
        x = torch.randint(0, 8, (b, v), device="cuda", generator=gen).float()
    elif kind == "equal":
        # One value everywhere: every element ties at the k-th, the staged
        # population is the whole row, and the arena overflows on purpose.
        x = torch.full((b, v), 3.0, device="cuda")
    elif kind == "dense":
        # The window's tail is a run of equal values straddling the cut.
        x = torch.rand(b, v, device="cuda", generator=gen)
        x[:, v // 2:] = 0.5
    else:
        raise ValueError(kind)
    return x.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="maca_c")
    ap.add_argument("--device", default="1")
    a = ap.parse_args()
    torch.cuda.set_device(int(a.device))
    gen = torch.Generator(device="cuda").manual_seed(20261026)

    shapes = [(1, 131072, 512), (7, 131072, 512), (128, 131072, 512),
              (7, 262144, 512), (7, 262144, 1024), (7, 524288, 1024),
              (1, 1048576, 512), (7, 1048576, 2048), (128, 262144, 512)]
    kinds = ["normal", "ties", "equal", "dense"]
    ok = bad = 0
    for b, v, k in shapes:
        for kind in kinds:
            for idx_dtype in (torch.int32, torch.int64):
                x = make(kind, b, v, gen)
                name = f"{kind:6} b{b} v{v} k{k} {str(idx_dtype).split('.')[-1]}"
                val, idx = deep_select.topk(x, k, backend=a.backend,
                                            indices_type=idx_dtype)
                if check(name, x, k, val, idx):
                    ok += 1
                else:
                    bad += 1
                del x, val, idx
                torch.cuda.empty_cache()
    print(f"{ok} passed, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
