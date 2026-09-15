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
| `__ballot_sync(0xFFFFFFFF, pred)` | **`__builtin_mxc_sicmp(pred,0,ICMP_NE) & mask`** — one 64-lane compare, then AND. Mask honored |
| `__ballot_sync(0xFFFFFFFF, 1)` | `0x00000000ffffffff` — lanes 32..63 gone |
| `__popc(ballot)` | counts only the low 32 bits → **`__popcll`** |
| `__reduce_add_sync(0xFFFFFFFF, 1)` | **32**, not 64. Mask honored |
| `__any_sync(0xFFFFFFFF, …)` | lanes 32..63 ignored. Mask honored |
| `__shfl_up/down_sync(0xFFFFFFFF, …)` | mask **honored**: lane 32 gets its **own** value back, because lane 31 is outside the mask. Lane 63 with a full mask still reads lane 62 — the wave is 64 wide |

**One rule, uniformly: a lane outside the mask reads its own value back.** So a
32-bit mask means "**throw away lanes 32..63**" — not "group the wave by 32".
Grouping is a separate, opt-in thing: pass the `width` argument
(`__shfl_up_sync(full, v, d, /*width=*/32)`).

**Read a shuffle probe only if the source lane is encoded in the value.**
`v[lane] = 1000 + lane` makes the result unambiguous (999+d = "read lane d-1",
1000+d = "kept my own"). An earlier version of the probe here put a
distinctive value on lane 31 alone and concluded, from the same hardware, that
the mask was *ignored* — it could not distinguish "excluded by the mask" from
"included, but reading a lane whose value is its own id". That conclusion was
wrong and is corrected below; do not reproduce the probe shape that produced
it.

## 2. The audit

`references/audit-checklist.md` is the mechanical version: the grep patterns,
the structural quantities that have to be re-derived (not just the masks), and
how to tell a real 32-lane assumption from an innocent `% 32`.

The four shapes that matter:

1. **A 32-bit mask on a collective.** `__ballot_sync`, `__reduce_*_sync`,
   `__any_sync`/`__all_sync`, `__shfl_*_sync`. Each becomes a 64-bit mask, and
   each changes behavior when it does — including the shuffles, where lane 32
   currently reads itself.
2. **A lane index derived with `% 32`.** `lane_idx` must be `threadIdx.x % 64`,
   or `__lane_id()`.
3. **A warp count derived with `/ 32`.** `NUM_WARPS` must be
   `NUM_THREADS / 64`. This one is worse than it looks: it silently doubles, so
   every `warp_cnt[NUM_WARPS]` array is the wrong size, `warp_idx` addresses
   the wrong slot, and `lane_idx < NUM_WARPS` predicates admit the wrong lanes.
   **`kerutils`'s `canonical_warp_idx_sync()` is `threadIdx.x / 32u`** and is
   part of this — a port that calls it inherits the bug from a vendored header.
4. **A scan/prefix width, in the loop bound *and* the guard.** Both
   `for (i = 1; i <= 16; i <<= 1)` and `if (lane_idx + i < 32)` are 32-lane;
   a 64-lane scan is `i <= 32` with `< 64`, and each needs fixing on its own.

Do not stop at the masks. In a real port the masks are the *symptom*; the
structural quantities (2) and (3) are what make whole data structures the wrong
size, and they are the ones a "just widen the mask" fix leaves broken.

## 3. The mask is a dead end — widen the model, not the literal

It is tempting, once the mask is identified, to think the port's *design* is
fine and only the literal is wrong: `NUM_WARPS = NUM_THREADS / 32`,
`lane_idx = threadIdx.x % 32` and `canonical_warp_idx_sync() = threadIdx.x / 32`
all agree with "logical groups of 32 lanes", so maybe each group just needs its
own mask.

**It does not work, and not because of a bug — because the scheme has no
representation on this hardware.** Logical group `g` owns physical lanes
`32g..32g+31`, and a lane mask is 64 bits wide. It can name the first two such
windows and nothing above them; from group 2 up the mask is not merely wrong, it
is inexpressible (and the shift is UB). Measured, with `x = lane+1` per logical
lane and a wanted in-group sum of 528, "wrong" counting lanes that did not get
528:

| block | logical groups | mask `0xFFFFFFFF` | group mask `0xFFFFFFFF << 32g` |
| --- | --- | --- | --- |
| 64 | 2 | 32/64 wrong | **0/64 wrong** |
| 128 | 4 | 64/128 wrong | 64/128 wrong |
| 256 | 8 | 128/256 wrong | 192/256 wrong |
| 512 | 16 | 256/512 wrong | 448/512 wrong |

The group mask is *perfect at 64 threads and progressively worse above it* —
the tell that the scheme is the problem, not the constant.

`__activemask()` is not the escape either: it is the **physical** wave mask, so
it is `0xffffffffffffffff` at every block size and makes a logical-group
reduction *more* wrong, not less (128/128 at 128 threads). Use it only where you
genuinely want the whole converged physical wave.

**What this means for the fix.** A warp-scope operation can only reach lanes
that share a physical wave, so a design that wants groups of 32 cannot be
implemented with warp-scope primitives on a 64-lane wave — it needs either a
physical 64-lane model or shared-memory exchange instead of cross-lane ops. The
port is written as if cross-lane ops reach a 32-lane group; they do not. So:

```cpp
static constexpr uint32_t kWarpSize = 64;        // was 32
lane_idx  = threadIdx.x % kWarpSize;             // was % 32
NUM_WARPS = NUM_THREADS / kWarpSize;             // was / 32
mask      = 0xFFFFFFFFFFFFFFFFull;               // was 0xFFFFFFFF
```

and *then* re-derive the handful of places that assumed 32 items per lane
(§2, item 4, and the audit checklist's judgment list). Widening the masks
without widening `lane_idx`/`NUM_WARPS` is the worst of both: the masks now
reach lanes the rest of the code believes are in another group.

## 4. The reference implementation is in this repository

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

### CUB works — but the builtin is cheaper, and `readlane` is not a substitute

Three ways to replace a 32-lane scan, all measured on a 64-lane wave
(`/tmp/cubvs.cu` in the session that took it; wrong-lanes out of 64):

| approach | correct? | cost |
| --- | --- | --- |
| `cub::WarpScan<T>` / `WarpReduce<T>` (needs an explicit `TempStorage`) | **yes, 0/64** | 66 / 69 device insns |
| `__builtin_mxc_bsm_bpermute` butterfly, written here | **yes, 0/64** | **54 / 42** |
| `__builtin_mxc_readlane(x, src)` | — | **does not generalize**: every `src` returned the caller's own value |

MACA's CUB gets the width right (`CUB_LOG_WARP_THREADS` is 6 → 64, and the
specializations use `0xffffffffffffffffull` masks, `LaneId()` = `__lane_id()`,
and `__shfl_up/down_sync(…, LOGICAL_WARP_THREADS)`). So `cub::WarpScan`,
`cub::WarpReduce`, `cub::WarpExchange`, `cub::BlockRadixSort` and friends are a
**legitimate and correct** answer — with two caveats:

1. **It is ~1.3–1.6× the instruction count** of the same operation written with
   `bsm_bpermute`, because CUB's shuffles go through the `__shfl_*_sync`
   wrappers' per-call index arithmetic. For a warp-scope primitive in a hot
   loop, write the butterfly:

   ```cpp
   // 64-lane step-down gather; the hardware index is byte-addressed, hence <<2
   int n = __builtin_mxc_bsm_bpermute(((lane + delta) & 63) << 2, val);
   ```

   `bpermute` wraps modulo 64, so a butterfly that would read below lane 0 (or
   above 63) must mask the contribution itself — `if (lane >= d) x += n`.

2. **This CUB generation requires explicit temp storage** —
   `cub::WarpScan<T> ws(temp_storage)`, not a default constructor. A one-liner
   `cub::WarpScan<T> ws;` will not compile.

**Do not reach for `__builtin_mxc_readlane` to replace a shuffle.** It reads one
lane's value by a mode selected from a 16-entry table (`0x150`..`0x15f`, i.e. a
16-lane row), and in this probe every source lane returned the caller's own
value. A per-lane gather is `bsm_bpermute`; a broadcast is
`__builtin_mxc_readfirstlane`.

See the repository's `CLAUDE.md`, "MACA warp intrinsics", for the rest of the
builtin catalogue and the measured instruction counts behind this.

## 5. Fix order and verification

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
# the shape of a localizer: no seed, no harness, the answer is typed in.
# backend="maca_c" is required -- the library default is `torch`, and a bare
# call would localize the reference, which is not what a kernel audit is for.
python -c "
import torch, deep_select
b, v, k = 2, 512, 8
x = torch.arange(v, device='cuda', dtype=torch.float32) \
     .unsqueeze(0).repeat(b, 1).to(torch.bfloat16)
print(deep_select.topk(x, k, backend="maca_c")[1][0].tolist())
# want [511, 510, 509, 508, 507, 506, 505, 504]
"
```

## 6. Traps that look like wave-64 bugs and are not

- **`__activemask()` is convergence-dependent.** Reading it while only part of
  the wave has arrived gives a partial mask — that is the primitive working as
  specified, not a lane-width bug. Pass it only where the wave is converged, or
  use the full mask deliberately.
- **`__match_any_sync` is software-emulated** (a 32-iteration per-bit loop) and
  is not a shortcut past any of this; reach for `__ballot_sync` + `__popcll`
  instead.
- **A scan can be wrong in the loop bound and the guard independently.** Fixing
  the bound (`i <= 16` → `i <= 32`) without the guard (`lane_idx + i < 32` →
  `< 64`), or the reverse, leaves it wrong in a *different* way. Fix both, then
  re-measure per lane — a scan that is right on lanes 0..31 and wrong above is
  the signature of having fixed only one.
- **`cub::BlockRadixSort` / MACA's CUB** get the width right
  (`CUB_LOG_WARP_THREADS` is 6, i.e. 64), so a block-scope primitive built on
  CUB is not automatically suspect — but `cub::WarpReduce` inherits the
  emulated `__shfl_down_sync` and is slow, so verify rather than assume.
- **A `% 32` on a count is not a lane index.** Distinguish them: the audit
  checklist has the disambiguation.

## 7. Recording the result

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

## 8. Worked example: `csrc/xcore1600/` in this repository

The tree this skill was written against. **Not fixed yet** — this is the
measurement and the diagnosis, recorded so the fix starts from evidence.

Symptom, on a MetaX C600-U (reports `sm89` → family 1600):

```
b=2, v=512, k=8, input = arange(0..511) in bf16
want [511, 510, 509, 508, 507, 506, 505, 504]
got  [448, 449, 450, 451, 452, 453, 454, 455]   <- and different every run
```

Eight consecutive runs gave eight distinct wrong index sets. A random row
returns `1.5e+37` for a row whose true max is `3.3`. On the official slice:
4/200. Other shapes trap (`[topk_select] NaN detected` on input with no NaN) or
raise `device-side assert`.

### 8.1 The three masks do not behave the same

Measured per-primitive (`__ballot_sync`, `__reduce_add_sync`, `__any_sync`,
`__shfl_up/down_sync`) on MACA 3.8.1.3. The rule is the same for all of them —
**a lane outside the mask reads its own value back** — and that includes the
shuffles, which is the one worth checking rather than assuming:

| primitive | is the mask honored? |
| --- | --- |
| `__ballot_sync` | **yes** — `sicmp(pred,0,NE) & mask`; lanes outside the mask read 0 |
| `__reduce_*_sync` | **yes** — the inner loop is gated on `mask & (1 << lane)`, and a lane outside gets its own value back. `__reduce_add_sync(0xFFFFFFFF, 1)` = **32** |
| `__any_sync` / `__all_sync` | **yes** |
| `__shfl_up_sync` / `__shfl_down_sync` | **NO** — the implementation (`__clang_maca_device_functions.h:712`) never reads `mask`; it clamps only against `width`, defaulting to `warpSize` = **64** |

The shuffle row is the trap. A 32-bit mask on a shuffle is *cosmetic* at the
default width, so widening it changes nothing — the 32-ness lives in the
**guard**, not the mask:

```
__shfl_up_sync(0xFFFFFFFF, v, 1)   lane 32 -> 777   (reads lane 31: wave-wide)
__shfl_up_sync(0xFFFFFFFF, v, 1)   lane 63 -> 62    (wave-wide)
__shfl_up_sync(full mask, v, 1)    lane 63 -> 62    (same)
__shfl_up_sync(full, v, 1, /*width=*/32)  lane 32 -> 32  (grouped, opt-in)
```

So `utils.cuh`'s scan is wrong for **three independent reasons**, and a fix
that addresses fewer than all three leaves it wrong:

1. the mask `0xFFFFFFFFu` excludes lanes 32..63, so they gather from themselves;
2. the loop bound `i <= 16` is a 5-step Hillis-Steele, correct for 32 lanes;
   64 needs `i <= 32`;
3. the guard `lane_idx + i < 32` (suffix scan) discards everything above lane 31.

### 8.2 Site inventory

| site | current | why it is wrong |
| --- | --- | --- |
| `utils.cuh:20`, `:37` | `for (i = 1; i <= 16; i <<= 1)` | a 5-step scan is 32-lane; 64 needs `i <= 32` |
| `utils.cuh:39` | `if (lane_idx + i < 32)` | guard discards data the shuffle actually delivered |
| `utils.cuh:21`, `:38` | `__shfl_*(0xFFFFFFFFu, …)` | lane 32+ reads its own value instead of its neighbour; the mask is why |
| `utils.cuh:7-15` | the comment | states the **false premise** ("ballot/reduce/shfl group by 32, so a 32-written scan is correct"), then contradicts itself two lines later. Delete it, do not preserve it |
| `common_parts.cuh:312` | `NUM_WARPS = NUM_THREADS / 32` | **doubles** — feeds `:421` |
| `common_parts.cuh:421` | `static_assert(NUM_SEGS_PER_ROUND == NUM_WARPS)`, `:697` `local_seg_idx = warp_idx`, `:696` `SEGS_PER_WARP == 1` | **the reachable uninitialized read.** `NUM_SEGS_PER_ROUND = elements_per_round / 512 = NUM_THREADS*16/512`, which is `NUM_THREADS/32` — right for 32-lane warps, **twice the real warp count** for 64. On a 256-thread config that is 8 segments per round and 4 real warps, so segments 4..7 are never loaded and the consumer reads uninitialized shared memory. Also makes `is_warp_active` (`v3/topk_select.cuh:92`, `v3_fp32:481`) and the permuted position arithmetic wrong by the same factor |
| `common_parts.cuh:121`, `:122` | `NUM_WARPS = NUM_THREADS / 32`, `static_assert(NUM_THREADS % 32 == 0)` | same `/32`; the assert encodes the wrong width |
| `common_parts.cuh:225` | `if (warp_idx < NUM_WARPS/2)` | with the doubled count this is true for **every** real warp, so the `else` branch that loads `input_values` never runs — `sorted_value` sorts an unloaded buffer |
| `common_parts.cuh:488/496/497/581/784/791/1432/1440/1469` | `__reduce_add_sync(0xFFFFFFFF, …)`, `__reduce_or_sync(0xFFFFFFFFu, …)` | mask honored → sums the low half only |
| `common_parts.cuh:1463` | `__ballot_sync(0xFFFFFFFF, …)`, `(1u << lane_idx) - 1u` | two defects: the mask drops the high half, and the 32-bit shift is UB for `lane_idx >= 32` |
| `common_parts.cuh:1464` | `__popc(bit)` | 32-bit; applied to a 64-lane ballot it truncates silently → `__popcll` |
| `common_parts.cuh:684` | `for (c = lane_idx; c < num_chunks; c += 32u)` | **decide, do not edit.** Chunks per segment is 64 (bf16) / 128 (fp32), so the stride must become 64 — but today the loop only covers chunks 0..31 idempotently (two aliasing lanes copy each chunk), which is why it is not the crash site |
| `common_parts.cuh:536`, `:538`, `:566` | `static_assert(NUM_RECONSTRUCT_BUCKETS == 32*8)`, `bucket_counter + lane_idx * 8`, `lane_idx * 8 + j` | the histogram-to-lane mapping is 32-lane-shaped. `NUM_RECONSTRUCT_BUCKETS` is `1 << NUM_RECONSTRUCT_RADIX_BITS` (`:385`) — it comes from the radix config, **not** the lane count, so the fix is 4 buckets per lane over 64 lanes, not a bigger structure |
| `common_parts.cuh:448`, `:490/786/1434` exchange | `warp_cnt[NUM_WARPS]`, `lane_idx < NUM_WARPS` | sized and indexed by the doubled count; the "lane i holds warp i's total" convention is 32-lane. With 64 lanes the totals arrive in both halves, so the exchange has to be re-derived, not resized |
| `v3/topk_select.cuh:53-54`, `v3_fp32/topk_select.cuh:89-90` | `canonical_warp_idx_sync()` + `threadIdx.x % 32` | the lane index; and see the next row |
| `kerutils/.../device/cuda/common.h:89` | `return threadIdx.x / 32u;` | **outside `csrc/xcore1600/` but in the compiled path** (`common_parts.cuh:4` includes it). Returns a warp index twice the hardware's, so `warp_idx` and `lane_idx` name different groupings and no consistent relabeling of one alone can work |

### 8.3 Not to be "fixed"

`bit_utils.cuh:43/149` (`>> 31`, `0x80000000u`) are the fp32 sign bit;
`common_parts.cuh:1397/1459` `static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND < 32)`
is a per-**thread** element count with a per-thread `hit_mask`, not a lane mask;
`:88` `NUM_BYTES_TO_STORE == 32` is a store width; `:306`/`:1081`/`:1476` and
`v3_fp32:79` are pair packing; `__syncthreads_or` is block-wide with no mask.
`cub::BlockRadixSort` is **width-correct** on MACA (CUB's `WARP_THREADS` is 64
and it uses `0xffffffffffffffffull` masks) — slow, but not part of this bug.

### 8.4 Containment

`deep_select/_arch.py`'s `DEEP_SELECT_128KIB_KERNEL` routes a 128 KiB part to
the hand-written 64-lane kernel instead (`xcore1000`), which passes the slice
200/200. Set it to `xcore1600` to work this tree, and re-run the slice when done.
