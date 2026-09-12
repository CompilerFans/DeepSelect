# Mechanical audit checklist

Run this over the tree being ported. The point is to be **exhaustive on the
mechanical pass and careful on the judgment calls** — a missed site stays
silent-wrong, and a false positive costs a wasted fix that may itself
introduce a bug.

## 1. Grep patterns — every hit is a candidate

```bash
# cross-lane collectives
rg -n '__ballot_sync|__reduce_(add|min|max|and|or|xor)_sync|__any_sync|__all_sync|__match_any_sync|__activemask|__shfl_(up|down|xor|idx|sync)' <tree>

# lane / warp derivation
rg -n '__lane_id|threadIdx\.x\s*%\s*32|threadIdx\.x\s*/\s*32|canonical_warp|% *32|/ *32|>> *5' <tree>

# masks and widths
rg -n '0x[Ff]{8}\b|0x[Ff]{8}[Uu]?\b|NUM_WARPS|WARP_SIZE|warpSize|kWarpSize|lane_idx|lane_id|subgroup' <tree>

# scans whose step count encodes a width
rg -n 'i *<= *16|<= *16; *i *<<=|\* *= *2\).*31|< *32|< *16' <tree>
```

Then **read every hit**. The grep is for recall; the table below is for
precision.

## 2. Classify each hit

### Definitely a 32-lane assumption → fix

| pattern | correct form |
| --- | --- |
| `__ballot_sync(0xFFFFFFFF, p)` | `__ballot_sync(0xFFFFFFFFFFFFFFFFULL, p)` |
| `__reduce_add_sync(0xFFFFFFFF, v)` | 64-bit mask, or a `bsm_bpermute` butterfly |
| `__any_sync` / `__all_sync(0xFFFFFFFF, p)` | 64-bit mask |
| `__shfl_*(0xFFFFFFFF, …)` | `0xFFFFFFFFFFFFFFFFULL` |
| `__popc(mask)` where `mask` came from a ballot | `__popcll(mask)` |
| `lane_idx = threadIdx.x % 32` | `% 64`, or `__lane_id()` |
| `NUM_WARPS = NUM_THREADS / 32` | `NUM_THREADS / 64` |
| `static_assert(NUM_WARPS <= 32)` | re-derive; the bound came from the old width |
| `cu::canonical_warp_idx_sync()` (kerutils: `threadIdx.x / 32u`) | `threadIdx.x / 64`, or own it locally |
| scan loop `i = 1; i <= 16; i <<= 1` | `i <= 32` |
| shuffle guard `lane_idx + i < 32` | `< 64` |

### Needs judgment — read the surrounding code

- **`x % 32` where `x` is not a lane id.** A `% 32` on an element count,
  a bucket index, a bit offset, or a hash is not a lane assumption. The tell:
  does the value feed a collective, a shuffle, or a per-lane array index? If
  not, leave it.
- **`1u << lane_idx` with a 32-bit shift.** Correct *only* while
  `lane_idx < 32`. If the enclosing block is being widened to 64 lanes, this
  must become `1ull << lane_idx` (or be proven lane-bounded with a
  `static_assert`). It is easy to widen a mask and leave the shift behind.
- **`for (c = lane_idx; c < n; c += 32u)`** — a lane-strided loop. The stride
  is the wave width, so it becomes `+= 64u` *and* the number of iterations
  halves — but only if the loop's body was written for one item per lane per
  stride. Re-derive the trip count; do not just edit the constant.
- **`lane_idx * K`** where K counts contiguous items per lane (a vector of 8,
  a histogram slice). Combined with a `static_assert(N == 32 * K)`, widening
  the wave changes how many lanes cover the structure. Decide whether the
  per-lane payload shrinks (more lanes, same structure) or the structure grows
  (same lanes-per-item, wider coverage) — they lead to different edits, and
  picking wrong silently corrupts a histogram or a scan.
- **Shared arrays sized by the old warp count.** `warp_cnt[NUM_WARPS]`,
  `__shared__ T buf[32]`, anything indexed by `warp_idx` or `lane_idx`. These
  follow from (3) but are easy to miss because they carry no mask.

### Not a lane assumption — leave alone

- A `0xFFFFFFFF` used as a data mask, a bit-field mask, or a sentinel value.
- `__popc` on a value that is genuinely 32-bit and not a wave mask.
- `threadIdx.x % 32` inside code that has already been proven to run on
  lanes 0..31 only (rare; document why if you rely on it).
- CUB block-scope primitives (`cub::BlockRadixSort`, `BlockScan`,
  `BlockExchange`): MACA's CUB has `CUB_LOG_WARP_THREADS = 6` (64), so these
  are already width-correct. `cub::WarpReduce` is correct but *slow* — it
  inherits the emulated `__shfl_down_sync`.

## 3. Compile-time traps hit while doing the audit

- **`__reduce_add_sync(0xFFFFFFFF, v)` will not compile once widened**, if the
  literal is unsuffixed: MACA ships two overloads, `(uint64_t mask, …)` and
  `(unsigned mask, …)`, and `0xFFFFFFFF` matches both. Spell the type:
  `__reduce_add_sync_impl((uint64_t)0xFFFFFFFFFFFFFFFFull, v)`, or use a
  `uint64_t` variable. (The `unsigned` overload is the 32-bit one; the
  `uint64_t` overload is the 64-lane one.)
- `warpSize` is a run-time-ish value here (64) but constant-folds; do not use
  it to size a `__shared__` array without also `static_assert`-ing it.

## 4. After the mechanical pass

The mechanical pass makes a kernel *compile* and *not obviously wrong*. It does
not prove correctness, because the structural quantities in §2's judgment list
change the shape of data. Verify with:

1. A deterministic, exactly-representable input, run several times (§4 of the
   skill).
2. A pass over every block-scope structure whose size was derived from the
   warp count — histograms, scan buffers, `warp_cnt` arrays, per-warp
   producer/consumer handoffs — asking "is this the right size *and* the right
   number of producers now?"
3. The project's own gate, then the project's own perf gate, because the fix
   changes occupancy: `NUM_WARPS` halving means half as many warps, and any
   `static_assert(NUM_WARPS <= 32)` or `__shared__` sizing that depended on the
   old count is now a different launch configuration.
