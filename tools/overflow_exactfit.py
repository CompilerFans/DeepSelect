#!/usr/bin/env python3
"""The bf16 row's overflow emit, on crafted exact-fill rows.

CLAUDE.md carried an open question against `overflow_emit_member`
(`csrc/maca_kernels/common/radix_row_bf16.cuh`): its slots all use
`output[atomicAdd(&s_counter, 1u)]`, and the note asked whether slot 0 can be
written twice when the fine-level `remain_topk` lands on 0.  The note's own
instruction was to settle it with a crafted exact-fill row rather than by
argument; this is that row, kept runnable:

    python tools/overflow_exactfit.py
    CUDA_VISIBLE_DEVICES=1 python tools/overflow_exactfit.py --rounds 5
    python tools/overflow_exactfit.py --backend torch   # the control arm

**The construction.**  `refine_bin` maps a positive bf16 to `bits | 0x8000`
(`primitives.cuh`), so the 12-bit coarse bin is `bits >> 4` and the fine
nibble is `bits & 15`.  Every value is chosen by its bf16 bit pattern:

    0x3FC0  1.5          key 0xBFC0   coarse bin 0xBFC
    0x3F8F  1.05859375   key 0xBF8F   bin 0xBF8, fine 15
    0x3F8E  1.0546875    key 0xBF8E   bin 0xBF8, fine 14
    0x3F80  1.0          key 0xBF80   bin 0xBF8, fine 0
    0x3F00  0.5          key 0xBF00   coarse bin 0xBF0

The arena holds 4096 members (`kCoarse12ArenaEntries`), so a threshold bin
with more than that takes the rescan in which `overflow_emit_member` runs:

  A  V=8192, int64: overflow + fine exact fit, coarse complement non-empty.
     12 above the bin; the bin holds 4501 members (500 fine-15, 1 fine-14,
     4000 fine-0).  Coarse: above 12, remain 500; the fine crossing lands at
     14 with 500 strictly above it, so `remain_topk` is 0 and the emit writes
     500 at slots 12..511 -- the complement owns 0..11.  This is the case the
     question needs: an exact fit inside the overflowing arm.
  B  V=8192, int64: overflow + ties.  No coarse complement; 4400 in the bin
     (300 fine-15, 4100 fine-0) -> counter slab [0, 300), tie slab [300, 512).
  C  V=8192, int64: the coarse exact-fit exit with an oversized bin (512
     above; 4501 in the bin) -- returns before the arena is consulted.
  D  V=8192, int64: A's structure with 601 in the bin -- the arena's own
     exact-fit arm, no overflow.
  E  A at V=262144, int32 indices: the block-summary / kept-block-list walk,
     which the dispatcher gates on `index_dtype == 0` (`maca_topk.cu`).  V/k
     is 512, over the 256 density floor, so the rescan walks the list.
  F  B at V=262144, int32 indices.

The longer rows carry the same 4,513-element prefix; the rest of the row is
0.5, which is below the threshold bin and changes none of the counts the
threshold is derived from.

**What is checked** per call: index range, uniqueness, `value_i ==
input[index_i]`, the selected value multiset against `torch.topk`, and the
`-1` / `0x3F3F3F3F` sentinels.  The selected *set* is unique in A, C, D and E
(their boundary is untied) and is expected to vary in B and F, where 212 of
4100 identical values tie at the k-th place -- the contract fixes the value
multiset, not which member of a tie group is taken.

**Rows must be built on the device.**  A row left on the CPU is dereferenced
as a device pointer and reports as `Xnack Error / ATU Fault`, which reads
exactly like the defect this tool asks about and is not one (`CLAUDE.md`'s
environment trap 1).
"""
import argparse
import os
import sys

import provenance

REPO = provenance.REPO

TOPK = 512

# (name, vocab, [(bits, count), ...], out-index dtype)
CASES = [
    ("A short overflow exact-fit", 8192,
     [(0x3FC0, 12), (0x3F8F, 500), (0x3F8E, 1), (0x3F80, 4000)], None),
    ("B short overflow ties", 8192,
     [(0x3F8F, 300), (0x3F80, 4100)], None),
    ("C short coarse-exact", 8192,
     [(0x3FC0, 512), (0x3F8F, 4501)], None),
    ("D short arena exact-fit", 8192,
     [(0x3FC0, 12), (0x3F8F, 500), (0x3F8E, 1), (0x3F80, 100)], None),
    ("E long overflow exact-fit", 262144,
     [(0x3FC0, 12), (0x3F8F, 500), (0x3F8E, 1), (0x3F80, 4000)], "int32"),
    ("F long overflow ties", 262144,
     [(0x3F8F, 300), (0x3F80, 4100)], "int32"),
]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--backend", default="maca_c",
                    help="backend name passed to every call (maca_c tests the "
                         "kernel; torch is the control arm)")
    ap.add_argument("--rounds", type=int, default=5,
                    help="identical calls per case (a racy kernel agrees with "
                         "anything once)")
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES shorthand")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    provenance.use_checkout_if_built()
    import torch
    import deep_select

    dev = torch.device("cuda")
    props = torch.cuda.get_device_properties(torch.cuda.current_device())
    print(f"device {torch.cuda.current_device()}: {props.name}, "
          f"{props.multi_processor_count} AP")
    print(provenance.package_receipt(deep_select))
    print(f"backend={args.backend}  topk={TOPK}  rounds={args.rounds}")

    def bf16(bits):
        return torch.tensor([bits], dtype=torch.int16, device=dev).view(torch.bfloat16)[0]

    lo = bf16(0x3F00)
    failures = 0
    for name, vocab, parts, idx_dtype in CASES:
        row = torch.cat([bf16(bits).repeat(n) for bits, n in parts])
        row = torch.cat([row, lo.repeat(vocab - row.numel())]).contiguous()
        assert row.numel() == vocab, (name, row.numel(), vocab)
        row = row.unsqueeze(0)
        kwargs = {"backend": args.backend}
        if idx_dtype == "int32":
            kwargs["indices_type"] = torch.int32
        try:
            picked = []
            for rep in range(args.rounds):
                val, idx = deep_select.topk(row, TOPK, **kwargs)
                ii = idx[0].long()
                assert tuple(idx.shape) == (1, TOPK), (name, rep, tuple(idx.shape))
                assert int(ii.min()) >= 0 and int(ii.max()) < vocab, (
                    name, rep, int(ii.min()), int(ii.max()))
                assert int(torch.unique(ii).numel()) == TOPK, (
                    name, rep, f"{TOPK - int(torch.unique(ii).numel())} duplicated")
                assert torch.equal(row[0][ii], val[0]), (
                    name, rep, "value_i != input[index_i]")
                ref = torch.sort(torch.topk(row[0], TOPK).values).values
                got = torch.sort(val[0]).values
                assert torch.equal(got, ref), (
                    name, rep, "selected value multiset differs from torch.topk")
                assert not bool((ii == 0x3F3F3F3F).any()), (name, rep, "index sentinel")
                assert not bool((val[0].view(torch.int16) == 0x3F3F3F3F).any()), (
                    name, rep, "value sentinel")
                picked.append(tuple(sorted(int(x) for x in ii)))
            print(f"OK   {name:30s} {args.rounds}/{args.rounds} pass, "
                  f"{len(set(picked))} distinct selected set(s) over the rounds")
        except AssertionError as exc:
            failures += 1
            print(f"FAIL {name}: {exc}")
        except Exception as exc:  # noqa: BLE001 - a crash is a result here
            failures += 1
            print(f"CRASH {name}: {type(exc).__name__}: {exc}")
    print(f"\n{'PASS' if failures == 0 else 'FAIL'}: "
          f"{len(CASES) - failures}/{len(CASES)} cases")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
