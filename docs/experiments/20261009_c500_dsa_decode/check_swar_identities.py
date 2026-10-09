#!/usr/bin/env python3
"""The two SWAR identities in `hist_add_bf16_wide`, against the per-half forms.

The landed pass-1 body flips the ordering key and tests for NaN on both
bf16 halves of a 32-bit word at once.  Both are arithmetic claims about a
packed word, and both are claims about *carry behaviour between the
halves* -- which is the part a spot check cannot see.  So:

  * every one of the 65,536 half patterns, with the other half swept over
    an adversarial set (both zeros, both infinities, both NaN encodings,
    both sign boundaries);
  * the full cross product of that adversarial set (64 pairs);
  * 300,000 random packed words.

Each SWAR result is compared against the per-half definition it replaces
(`bf16_to_uint16` and `is_nan_bits16` of `radix_core.cuh`).

    python3 check_swar_identities.py        # exits non-zero on any mismatch
"""
import random
import sys

# `bf16_to_uint16`: the ordering-key flip, one half at a time.
# `is_nan_bits16`: exponent all-ones with a non-zero payload.
def per_half_key(h):
    return (h ^ 0xFFFF) if (h & 0x8000) else (h ^ 0x8000)


def per_half_nan(h):
    return (h & 0x7F80) == 0x7F80 and (h & 0x007F) != 0


def swar_key(w):
    """Both halves' keys, packed: low half in bits 0..15, high in 16..31."""
    return w ^ ((((w >> 15) & 0x00010001) * 0x7FFF) + 0x80008000)


def swar_nan(w):
    """Bit 15 set iff the low half is NaN; bit 31 iff the high half is."""
    return (((w & 0x7FFF7FFF) | 0x80008000) - 0x7F817F81) & 0x80008000


ADV = [0x0000, 0x0001, 0x007F, 0x7F7F, 0x7F80, 0x7F81, 0x7FFF,
       0x8000, 0x8001, 0x807F, 0xFF7F, 0xFF80, 0xFF81, 0xFFFF]


def check(w, where, bad):
    lo, hi = w & 0xFFFF, w >> 16
    kp = swar_key(w)
    if (kp & 0xFFFF) != per_half_key(lo) or (kp >> 16) != per_half_key(hi):
        bad.append(("key", where, w, hex(kp)))
    z = swar_nan(w)
    if bool(z & 0x8000) != per_half_nan(lo) or bool(z & 0x80000000) != per_half_nan(hi):
        bad.append(("nan", where, w, hex(z)))


def main():
    bad = []
    # (a) every half pattern, the other half adversarial
    for lo in range(0x10000):
        for hi in ADV:
            check((hi << 16) | lo, "lo-exhaustive", bad)
    for hi in range(0x10000):
        for lo in ADV:
            check((hi << 16) | lo, "hi-exhaustive", bad)
    # (b) the adversarial lattice, both halves at once
    for lo in ADV:
        for hi in ADV:
            check((hi << 16) | lo, "lattice", bad)
    # (c) random packed words
    rnd = random.Random(20261009)
    for _ in range(300_000):
        check(rnd.getrandbits(32), "random", bad)

    n = 2 * 0x10000 * len(ADV) + len(ADV) ** 2 + 300_000
    if bad:
        print(f"MISMATCH: {len(bad)} of {n} checks, first five:")
        for b in bad[:5]:
            print("   ", b)
        return 1
    print(f"both identities hold on all {n} checks "
          f"(the two exhaustive sweeps, the {len(ADV)}x{len(ADV)} lattice, "
          f"300k random words)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
