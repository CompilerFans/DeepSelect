---
name: maca-wave64-port
description: Audit and fix a CUDA kernel ported to MetaX MACA that silently produces wrong answers because it assumes CUDA's 32-lane warp. Use when a MACA kernel compiles but mis-selects, when results vary run to run, when a reduction/ballot/shuffle/scan is involved, or before landing any kernel copied from CUDA-era source.
---

# MACA wave-64 port audit

A CUDA kernel ported to MACA usually arrives carrying CUDA's **32-lane** model:
`lane_idx = threadIdx.x % 32`, `NUM_WARPS = NUM_THREADS / 32`, and `0xFFFFFFFF`
masks on every collective. On MACA the wave is **64 lanes**, and none of those
hold. **The failure is silent** — it compiles, it runs, and it returns numbers
that are wrong and change run to run. There is no assert and no diagnostic.

This skill is the procedure for finding and fixing that class of bug, and the
evidence for why each fix is what it is.

## 1. Measure before you reason — always

Do not read the semantics off a header, a comment, or a neighboring file. MACA's
own in-tree comments contradict each other about this, and one of them is what
produced the bug this skill exists for.

```bash
skills/maca-wave64-port/scripts/run_probe.sh          # this device
skills/maca-wave64-port/scripts/run_probe.sh xcore1600 1
```

It compiles `scripts/wave64_probe.cu` through **cu-bridge's cucc** — the same
device compiler a MACA torch extension build goes through — and prints the
measured semantics plus a verdict on the port's own scan helper. Run it first,
on the part you are actually targeting.

### What it reports, on MACA 3.8.1.3 / MetaX C600-U

| thing | measured |
| --- | --- |
| `warpSize` | **64** |
| `__lane_id()` | `threadIdx.x % 64` |
| `__activemask()` | full wave when converged (see §5) |
| `__ballot_sync(0xFFFFFFFF, pred)` | **`__builtin_mxc_sicmp(pred,0,ICMP_NE) & mask`** — one 64-lane compare, then AND |
| `__ballot_sync(0xFFFFFFFF, 1)` | `0x00000000ffffffff` — lanes 32..63 gone |
| `__popc(ballot)` | counts only the low 32 bits → **`__popcll`** |
| `__reduce_add_sync(0xFFFFFFFF, 1)` | **32**, not 64 |
| `__shfl_up_sync(0xFFFFFFFF, …)` | lane 32 cannot see lane 31 |
| `__any_sync(0xFFFFFFFF, …)` | lanes 32..63 ignored |

**The load-bearing fact:** a 32-bit mask is not "group the wave into 32-lane
groups". It is "**throw away lanes 32..63**". Every mask-based collective —
ballot, reduce, any/all, shuffle — implements that rule, and a lane outside the
mask simply gets its own value back. A kernel that runs its half of the work on
lanes 32..63 while masking them out of every cross-lane operation computes a
confident, silent, wrong answer.

## 2. The audit

`references/audit-checklist.md` is the mechanical version: the grep patterns,
the structural quantities that have to be re-derived (not just the masks), and
how to tell a real 32-lane assumption from an innocent `% 32`.

The four shapes that matter:

1. **A 32-bit mask on a collective.** `__ballot_sync(0xFFFFFFFF, …)`,
   `__reduce_*_sync(0xFFFFFFFF, …)`, `__any_sync`/`__all_sync`,
   `__shfl_*_sync(0xFFFFFFFF, …)`. Every one becomes a 64-bit mask.
2. **A lane index derived with `% 32`.** `lane_idx` must be `threadIdx.x % 64`,
   or `__lane_id()`.
3. **A warp count derived with `/ 32`.** `NUM_WARPS` must be
   `NUM_THREADS / 64`. This one is worse than it looks: it silently doubles, so
   every `warp_cnt[NUM_WARPS]` array is the wrong size, `warp_idx` addresses
   the wrong slot, and `lane_idx < NUM_WARPS` predicates admit the wrong lanes.
   **`kerutils`'s `canonical_warp_idx_sync()` is `threadIdx.x / 32u`** and is
   part of this — a port that calls it inherits the bug from a vendored header.
4. **A scan/prefix width.** `for (i = 1; i <= 16; i <<= 1)` is a 32-lane scan;
   `i <= 32` is a 64-lane one. The guard `if (lane_idx >= i)` comes along.

Do not stop at the masks. In a real port the masks are the *symptom*; the
structural quantities (2) and (3) are what make whole data structures the wrong
size, and they are the ones a "just widen the mask" fix leaves broken.

## 3. The reference implementation is in this repository

`csrc/xcore1000/radix_core.cuh` is a MACA kernel written for a 64-lane wave
from the start, and it is the model to copy:

```cpp
#ifdef __MACACC__
static constexpr uint32_t kWarpSize = 64;
#define FULL_MASK 0xFFFFFFFFFFFFFFFFULL
#else
static constexpr uint32_t kWarpSize = 32;
#define FULL_MASK 0xFFFFFFFFu
#endif
```

**Prefer the `#ifdef` pair over a hard-coded 64.** It keeps the file compilable
for a CUDA target, and — more usefully — it makes the assumption visible at
every use site instead of hiding it in a constant.

For a warp-scope reduction, MACA's own builtin is cheaper than the CUB/`__shfl`
spelling and does not carry the 32-lane wrappers:

```cpp
// 64-lane step-down gather; the hardware index is byte-addressed, hence <<2
int n = __builtin_mxc_bsm_bpermute(((lane + delta) & 63) << 2, val);
```

See the repository's `CLAUDE.md`, "MACA warp intrinsics", for the measured
instruction counts that justify this and for the rest of the builtin catalogue.

## 4. Fix order and verification

1. **Make it not wrong before making it fast.** Convert every site from §2 in
   one pass; a partial conversion stays silent-wrong and is harder to localize
   than the original.
2. **Verify with a deterministic, exactly-representable input**, so the
   expected answer can be *written down* rather than computed by the same
   suspect code. An `arange`-derived row is ideal: `0..511` is exact in bf16,
   so the top-k indices are `range(v-1, v-1-k, -1)`.
3. **Run it several times.** A racy kernel's single run agrees with anything.
   The tell for this bug class is *run-to-run variance*: a fixed offset is a
   logic error, a varying one is a lane-width race.
4. **Then** run the project's own gate.

```bash
# the shape of a localizer: no seed, no harness, the answer is typed in
python -c "
import torch, deep_select
b, v, k = 2, 512, 8
x = torch.arange(v, device='cuda', dtype=torch.float32) \
     .unsqueeze(0).repeat(b, 1).to(torch.bfloat16)
print(deep_select.topk(x, k)[1][0].tolist())
# want [511, 510, 509, 508, 507, 506, 505, 504]
"
```

## 5. Traps that look like wave-64 bugs and are not

- **`__activemask()` is convergence-dependent.** Reading it while only part of
  the wave has arrived gives a partial mask — that is the primitive working as
  specified, not a lane-width bug. Pass it only where the wave is converged, or
  use the full mask deliberately.
- **`__match_any_sync` is software-emulated** (a 32-iteration per-bit loop) and
  is not a shortcut past any of this; reach for `__ballot_sync` + `__popcll`
  instead.
- **`__shfl_up_sync(0xFFFFFFFF, …)` behaves correctly *within* the low half.**
  A scan that only ever runs on lanes 0..31 of a 64-lane wave can be right by
  accident. That is why the probe reports a per-lane verdict rather than a
  single pass/fail.
- **`cub::BlockRadixSort` / MACA's CUB** get the width right
  (`CUB_LOG_WARP_THREADS` is 6, i.e. 64), so a block-scope primitive built on
  CUB is not automatically suspect — but `cub::WarpReduce` inherits the
  emulated `__shfl_down_sync` and is slow, so verify rather than assume.
- **A `% 32` on a count is not a lane index.** Distinguish them: the audit
  checklist has the disambiguation.

## 6. Recording the result

A wave-64 fix is a behavior change, so the repository's performance/behavior
discipline applies (`CLAUDE.md`, "Performance-change discipline"): state the
principle, give the before→after numbers, and record the gate. Two things are
specific to this bug class and belong in the record:

- **The measurement that established the semantics** (this skill's probe, and
  its raw output), because the next reader will otherwise re-derive it from the
  same wrong comment.
- **The falsifying test that separates "was already wrong" from "this change
  broke it"** — usually a fixed input run several times, and if a baseline is
  being compared, note that the baseline had to be patched to build at all.
