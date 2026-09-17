# `ref/ds` — the DeepSelect fp32 radix TopK, extracted and runnable

A minimal, torch-free, ffi-free reference repro of DeepSelect's MACA top-K
operator, taken from
`../csrc/xcore1000/{radix_core.cuh,maca_topk.cu}`. It compiles with the
toolchain in `/tmp/dsprobe/TOOLCHAIN.md` (MACA 3.7.0.36, `cucc`, target
`xcore1000`) and runs on the C500 in this box.

Files:

| file | what it is |
| --- | --- |
| `xcore1000_radix_core.cuh` | a verbatim copy of `csrc/xcore1000/radix_core.cuh` (2464 lines), with a provenance banner on top. Both dataflows — fp32 row and bf16 row — and both chunked splits are still in it. |
| `xcore1000_maca_topk.cu` | upstream `maca_topk.cu` lines 1–1108 (the whole kernel/contract layer) verbatim, then a new `ds_topk` host entry. The tvm-ffi edge is not carried. |
| `xcore1000_ds_topk.h` | the public header: one entry, `ds_topk(...)`. |
| `main.cu` | the driver: device randn scores, the selector, a CPU `nth_element` check, time and GB/s. |
| `xcore1000_maca_topk.o`, `ds_topk_ref` | build products (not needed to read this). |

No file under `../csrc/` was modified.

---

## 1. The compile and run, verbatim

```bash
export MACA_PATH=/opt/maca
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH"
cd /home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect/ref/ds

# per-kernel registers / smem, one block per function:
"$CUDA_HOME/bin/cucc" -O3 -std=c++20 -DNDEBUG -fPIC -use-fast-math \
    -I"$MACA_PATH/include" --offload-arch=xcore1000 -resource-usage \
    -c xcore1000_maca_topk.cu -o xcore1000_maca_topk.o        # exit 0

# whole program, host + device image:
"$CUDA_HOME/bin/cucc" -O2 -std=c++20 --offload-arch=xcore1000 \
    -I"$MACA_PATH/include" main.cu xcore1000_maca_topk.cu -o ds_topk_ref   # exit 0

./ds_topk_ref
```

The whole-program form compiles both sources in one `cucc` invocation. It also
works as the two-step shape `setup.py` uses — one `-c` per source with
`--offload-arch` on each, then a host link (verified: `main.o` + `topk.o` link
and run identically).

Nothing else is needed: no torch, no tvm_ffi, no `../ffi/`, no `structs.h`, no
`-DKSMEM_BYTES`. See §5 for the two includes that were dropped and why.

### Real output

```
device : MetaX C500  sm=104 warp=64 smem/block=65536 sharedMemPerMultiprocessor=65536 (driver uses sm_count=104)
── B=256 V=16384 k=2048 ──
  verify: PASS
  time : 0.1388 ms   (best of 3, 3 launches each)
  bw   : 136.0 GB/s  (18.9 MB useful: 16.8 MB read + 2.1 MB written)
── B=256 V=524288 k=2048 ──
  verify: PASS
  time : 2.0752 ms   (best of 3, 3 launches each)
  bw   : 259.7 GB/s  (539.0 MB useful: 536.9 MB read + 2.1 MB written)
── B=8 V=65536 k=512 (ragged) ──
  verify: PASS
  time : 0.1658 ms   (best of 3, 3 launches each)
  bw   : 12.7 GB/s  (2.1 MB useful: 2.1 MB read + 0.0 MB written)
── B=2 V=1000 k=2048 ──
  verify: PASS
  time : 0.0089 ms   (best of 3, 3 launches each)
  bw   : 2.7 GB/s  (0.0 MB useful: 0.0 MB read + 0.0 MB written)

ALL PASS
```

The four cells are chosen to hit four different arms of the dispatcher — see
§3. The `bw` number is *useful* traffic (one read of the score matrix, `k`
int32 written), not DRAM traffic: the fp32 row walks the window twice plus a
third time when the coarse bin overflows, and the split re-reads its
candidates in the merge, so the machine moves more than this.

**The clock is not locked on this box** (persistence off, per TOOLCHAIN.md), so
the per-cell times drift run to run — and not by a little. Five consecutive
runs of the *same* binary, per cell, in ms:

| cell | best of 5 runs | worst of 5 runs | spread |
| --- | --- | --- | --- |
| `B=256 V=16384` | 0.1357 | 0.5407 | 4.0x |
| `B=256 V=524288` | 2.0826 | 3.1793 | 1.5x |
| `B=8 V=65536` (ragged) | 0.1652 | 0.6939 | 4.2x |
| `B=2 V=1000` | 0.0091 | 0.0605 | 6.6x |

The block above is one of those runs, not a representative one. Read the
*ordering* and the *verdicts*, not the absolute numbers: treat anything inside
that spread as noise, and take the `best of 5` column as the closest thing to
signal this box can produce.

**Verified, not asserted.** `main.cu` checks each row against a CPU
`std::nth_element` reference: the k-th largest value `kth` is computed, every
returned index must address an element `>= kth`, the count strictly above `kth`
must equal the row's own count strictly above, and the real columns must be
distinct (`-1`, the `idx_fill`, is allowed to repeat). The `length <= topk`
shortcut is checked separately, slot for slot, including the `-1` fill — it is
reached twice, once from the row path (`B=2 V=1000`) and once from inside the
split (`ragged`, whose every third row is shorter than k). All four cells print
`PASS`.

---

## 2. The two dataflows

`topk_launch` (ref `xcore1000_maca_topk.cu:1072`, upstream `maca_topk.cu:1037`)
picks one of two paths per call, by dtype and shape:

**The row path — `topk_kernel_radix`** (ref `:398`, upstream `:363`);
one CTA per row, `RowParams` taken by value. It is the contract layer over the
radix selection:

1. `length <= topk` shortcut. The window is no longer than k, so the window is
   the answer: ascending indices, `idx_fill` past the end. With
   `sorted_value`/`sorted_index` off (the only modes this reference
   instantiates) it skips `emit_ordered` (ref `:213`, upstream `:188`)
   entirely and degrades to the plain write loop.
2. NaN detection, on the raw bit pattern (`is_nan_value<float>`, ref `:174`),
   because the order-preserving key encode sends the two signed NaNs to
   opposite ends of the key space. `abort_on_nan` traps; otherwise the row's
   answer becomes the sentinel `0x3F3F3F3F`. `check_nan` is a whole extra pass
   over the row, so a caller that already knows its input is clean turns it off.
   It is on for every cell in `main.cu` (`ds_topk` sets `check_nan = true`), so
   the split path additionally launches `nan_scan_kernel<float>` (ref `:364`,
   upstream `:329`) across the whole grid, and the row kernel reads the flag
   table it leaves behind instead of scanning a 12 MB row with one CTA.
3. Selection: `rk::radix_topk_row_f32` (ref radix core `:532`, upstream `:510`)
   — one histogram pass over the high key byte, one vectorized collect pass
   that refines the threshold bin in shared memory, so **two passes over the
   row whatever the key width**. If the coarse threshold bin is wider than the
   arena, `radix_topk_row_f32_rescan` (ref `:464`, upstream `:442`) re-walks the
   window one key byte per round instead of ranking a truncated candidate set.
4. Emit: `topk` slots, the ordered emit only when asked for.

**The chunked split** — one CTA per (row, chunk), then a merge CTA per row:

- fp32: `topk_f32_chunk_stage1_kernel` (ref radix core `:2271`, upstream
  `:2249`) ranks each chunk's window with the row entry above and writes the
  *row columns* it selected plus their values; `topk_f32_chunk_stage2_kernel`
  (ref `:2331`, upstream `:2309`) ranks the `chunks * topk` candidate values
  and maps the resulting positions back through the columns.
- bf16: `topk_bf16_chunk_stage1/2_kernel` (upstream radix core `:1762`/`:1840`)
  — same shape, but the staging answer is a position *inside the chunk*, so the
  columns are carried to be mapped back.
- `launch_typed_f32_chunked` (ref `:997`, upstream `:962`) memset+launches
  `nan_scan_kernel` (ref `:364`, upstream `:329`) on the same grid, runs the
  split, then runs the row kernel again in its `kPreSelected` arm: it takes the
  merged answer and skips selection, but keeps the whole contract half —
  shortcut, NaN, fills, offsets, ordering. A slot the merge could not fill
  (`-1`) sends that row back through the row dataflow.

This reference instantiates `ValueT = float`, `OutIdxT = int32_t`, and
`rv = si = sv = false`, so exactly two `topk_kernel_radix` specializations are
reachable — `<float, int32_t, 512, 0,0,0, PRE>` for `PRE` in {false, true}, the
row arm and the split's arm. The other fifty-eight in the object file are the
ones the upstream dispatcher compiles; they stay in the TU because the kernel
layer is a verbatim copy.

---

## 3. The routing predicates

The three the task names, with ref line numbers (`xcore1000_maca_topk.cu`) and
upstream ones (`maca_topk.cu`) — the bodies are identical.

### `chunked_f32_applies` — ref `:948`, upstream `:913`

```cpp
if (batches == 0) return false;
if (params.vocab_size < kF32ChunkedMinVocab) return false;   // 32768, ref :914, up :879
if (batches > kF32ChunkedMaxBatches) return false;           //  4096, ref :915, up :880
if (params.topk == 512 || params.topk == 1024) return true;  // measured
return topk_worth_splitting_f32(params, batches);
```

The `topk == 512 || 1024` clause is a *measured* shortcut; the general case is
the model below. The comment above it records that the old clause was a policy
bound (the merge used to compile only those two arms) and that opening it to
`topk <= kF32MaxTopK` bought -71..-82% on long rows and cost up to +26.5% on
short ones at a large batch — which is what `topk_worth_splitting_f32` is for.

### `topk_worth_splitting_f32` — ref `:934`, upstream `:899`

```cpp
if (params.topk <= 0 || params.topk > (int)rk::kF32MaxTopK) return false;  // 4096
if (batches == 0) return false;
if (params.vocab_size <= kF32ChunkedMinVocab) return false;          // V <= 32768
if (params.vocab_size <= 65536 && batches > 256) return false;
if (params.vocab_size <= 65536 && batches == kF32ChunkedMaxBatches) return false;
return true;
```

The split buys parallelism the row kernel did not have, so whether it wins is a
function of row length against the machine and of the batch. The comment above
it is explicit that the two-term cost model (`V < B * 6.0e6` with two chunks)
is only trusted *outside* the measured region, and that inside the sweep the
sweep decides.

### `f32_chunked_chunks` — ref `:872`, upstream `:837`

```cpp
const uint32_t work_target = f32_chunk_work_target(sm_count);   // sm_count * 5 / 2
if (override_n != 0) return override_n;                          // DEEP_SELECT_F32_CHUNKS
if (batches > kF32ChunksFewBatches)                              // 64
    return f32_chunks_large_batch(vocab_size, sm_count);
const int ceiling = f32_chunk_ceiling_32() ? 32 : kF32ChunkCeiling;   // DEEP_SELECT_F32_CHUNK32
const int c = f32_chunks_small_batch(batches, ceiling, work_target);  // largest pow2 <= K/B, >= 2
const int cap = f32_topk_chunk_cap(topk, vocab_size, c);
return (cap != 0 && c > cap) ? cap : c;
```

Two regimes, and they disagree about what sizes the split:

- **small batch (`batches <= 64`)**: the chunk count is a parallelism knob,
  `largest power of two <= work_target / batches`, floor 2, ceiling 16 (32 with
  `DEEP_SELECT_F32_CHUNK32`), then capped at 8 when `c * topk` is more than a
  tenth of the row (`f32_topk_chunk_cap`, ref `:783`, upstream `:748`).
- **large batch (`batches > 64`)**: `f32_chunks_large_batch` (ref `:852`,
  upstream `:817`) — `ceil(V / kF32OverflowChunkLen)` with
  `kF32OverflowChunkLen = 1757 * 58 = 101906` (ref `:850`, upstream `:815`),
  floor 2, **and only on C500** (`sm_count == 104`; otherwise the constant 2).
  The 1757 is `kF32SmemInputSize`, the 58 is the measured `L / bin` ratio of a
  randn row's threshold bin; the rule is "a chunk whose threshold bin fits the
  arena never takes the rescan", and the comment above it records that
  disabling the rescan branch took `c = 1` from 4.18 ms to 0.95 ms at
  `b128 V=524288 k=2048`.

Two related predicates are worth knowing when reading the dispatch:

- the **policy/engine split**: `chunked_f32_applies` decides *whether* to
  split, `rk::f32_chunk_engine_supports` (ref radix core `:2459`, upstream
  `:2437`) decides whether the *engine* can serve that `topk` at all
  (`0 < topk <= 4096`). Keeping them apart is what stops the policy from
  looking more expensive than the engine is; they were one expression until
  plan §21.2.
- the 16-bit gate, `chunked_bf16_applies` (ref `:654`, upstream `:619`): the
  bf16 split is compiled only for `topk == 512 || topk == 1024`, and its chunk
  count is `chunked_chunks(params) = wave_filled_chunks(16, sm_count)` (ref
  `:640`/`:628`, upstream `:605`/`:593`) — the only one of the two that reads
  `params.sm_count` through `RowParams`. This reference never reaches it
  (`value_dtype == 0`), but the predicates are in the TU because the kernel
  layer is verbatim.

### `sm_count`

`RowParams::sm_count` (ref `:95`, upstream `:60`) is the one machine number
every grid-sizing decision reads, supplied by the caller rather than read from
the driver or from an arch macro. Upstream gets it from
`deep_select/_arch.py`'s `SM_COUNT`; `ds_topk` takes it as an argument and
`main.cu` passes 104. `f32_chunks_large_batch` compares it against
`kC500SmCount = 104` (ref `:851`, upstream `:816`) and returns the constant 2
on anything else.

**Read this against the source of its date, not today's.** The two paragraphs
above describe the tree this snapshot was taken from. Upstream has moved twice
since: `_arch.py` was deleted (2026-09-17, `a22f5a0`), and the AP count now
comes from `torch.cuda.get_device_properties(...)` at the call site in
`interface.py`; `kC500SmCount` was renamed `kF32Coarse12MeasuredSmCount` and is
a *provenance* stamp rather than a device test; and the build is one artifact
per family as of 2026-09-18. `ds_topk`'s signature — `sm_count` as an argument
— is the part that did **not** change, which is why this repro still compiles
and still matches.

---

## 4. `kSMEM` and the constants it drives

### Which pass, and which value

`kSMEM` (ref radix core `:81–90`, upstream `:59–67`):

```cpp
#ifndef KSMEM_BYTES
#ifdef __MACACC__
constexpr size_t kSMEM = 16 * 1024;
#else
constexpr size_t kSMEM = 48 * 1024;
#endif
#else
constexpr size_t kSMEM = KSMEM_BYTES;
#endif
```

The upstream comment describes a split between "a device pass" and "a host
preprocessing pass". **Under `cucc`, `__MACACC__` is defined in both passes**,
so this build has no such split and takes the 16 KB arm everywhere. Measured
with a one-file probe that prints the same constant from a `__global__` and
from `main`, compiled on the TOOLCHAIN.md line:

```
DEVICE pass: kF32SmemInputSize=1757 kF32StaticBytes=2328 kSmemInputSize=3514 kSMEM=16384
HOST   pass: kF32SmemInputSize=1757 kF32StaticBytes=2328 kSmemInputSize=3514 kSMEM=16384
```

So: **`kSMEM = 16 KB` for this build**, on both sides. No `-DKSMEM_BYTES` is
passed, and — importantly — the reference *cannot* take a larger one: with
`-DKSMEM_BYTES=32768` the build fails outright,

```
xcore1000_radix_core.cuh:273:15: error: static assertion failed ... 'the 16-bit arena
must not be smaller than the region it aliases'
./xcore1000_radix_core.cuh:273:56: note: expression evaluates to '16384 >= 46824'
```

which is the guarded `static_assert` at ref radix core `:273` (upstream `:251`)
— the 16-bit path's 12-bit histogram aliases the whole dynamic arena, so
`kSmemInputSize * 4` must stay within 4096 entries. 16 KB is not a tuning
choice here, it is the only size the untouched core compiles at. (48 KB is
unreachable for the same reason.)

### The constants

| constant | value | where |
| --- | --- | --- |
| `kBlockSize` | 512 | ref radix core `:69`, upstream `:46` (`-DKBLOCK_SIZE` overrides) |
| `kChunkBlockSize` | 1024 | ref `:74`, upstream `:51` (the 16-bit split's staging width) |
| `kLongRowBlockSize` | 1024 | ref `:244`, upstream `:222` (the 16-bit long-row arm) |
| `kWarpSize` | 64 under `__MACACC__` | ref `:235`, upstream `:213` — this is why every shuffle mask in the file is `0xFFFF…FFFF` and not `0xFFFFFFFF` |
| `kSMEM` | **16384** | ref `:83`, upstream `:61` |
| `kSmemStaticBytes` | 2328 | ref `:251`, upstream `:229` — `2*(256+32)*4` + six words |
| `kSmemInputSize` | **3514** | ref `:255`, upstream `:233` — `(kSMEM - kSmemStaticBytes)/4` |
| `kF32StaticBytes` | 2328 | ref `:434`, upstream `:412` |
| `kF32SmemInputSize` | **1757** | ref `:439`, upstream `:417` — `(kSMEM - kF32StaticBytes)/(2*4)` |
| `kF32RowSmemBytes` | 14056 | ref `:441`, upstream `:419` — `2 * kF32SmemInputSize * 4` |
| `kSmemBudgetBytes` | 49152 | ref maca_topk `:529`, upstream `:494` — `kMaxTopK * (4 + 8)`, the `cudaFuncSetAttribute` ceiling **only**; each launch asks for `radix_smem_bytes(topk, …)` (ref `:542`, upstream `:507`) |

The dynamic shared memory the fp32 row actually requests is
`radix_smem_bytes(topk, sorted=false, wide=false) = kSmemInputSize_or_arena +
sizeof(uint32_t) * topk`. Two things about the distinction:

- **static** shared memory shows up in `-resource-usage` — `s_histogram_buf`,
  `s_counter`, `s_threshold_bin_id`, `s_high_threshold_bin_id`, `s_num_input`
  and (fp32 only) `s_last_remain`, which is 2596 B in the row kernel and 2332 B
  in the chunk stage kernels.
- the **dynamic** tail (`extern __shared__ uint32_t s_input_flat[]`) is what
  the launch requests and does not appear there at all: 14056 + 4·topk bytes,
  so 16104 at k=512 and 22248 at k=2048, both well under the 48 KB ceiling
  (`kSmemBudgetBytes`, the `cudaFuncSetAttribute` ceiling).
  `kF32SmemInputSize = 1757` is exactly the number `kF32OverflowChunkLen`
  multiplies (1757 · 58), which is why the two agree.

### Registers and smem per instantiation (`-resource-usage`)

All sixty `topk_kernel_radix` instantiations, and the fp32-relevant kernels.
`staticMaxWarps/PEU : 8` on every one of them, and `0 bytes stack frame`.

`topk_kernel_radix<float, int32_t, 512, ...>` — the ten fp32 arms (the two
reachable ones in **bold**; the rest are the upstream dispatcher's, compiled
because the kernel layer is verbatim). `SI` = sorted_index, `RV` =
return_value, `SV` = sorted_value, `PRE` = `kPreSelected`:

| `SI RV SV PRE` | MTregs | STregs | static smem |
| --- | --- | --- | --- |
| **`0 0 0 0`** (row path) | **42** | **52** | **2596 B** |
| **`0 0 0 1`** (split, preselected) | **42** | **48** | **2596 B** |
| `0 0 1 0` | 42 | 56 | 2596 B |
| `0 0 1 1` | 42 | 60 | 2596 B |
| `1 0 0 0` | 44 | 52 | 2596 B |
| `1 0 0 1` | 44 | 52 | 2596 B |
| `1 1 0 0` | 44 | 56 | 2596 B |
| `1 1 0 1` | 44 | 60 | 2596 B |
| `1 1 1 0` | 44 | 56 | 2596 B |
| `1 1 1 1` | 44 | 60 | 2596 B |

Over all sixty instantiations (fp32, bf16; 512 and 1024 block widths) the
ranges are MTregs 42–46, STregs 44–74, static smem 2596 B — i.e. the bf16 arms
are the register-heavy ones, which is why the fp32-only cut looks cheap. The
512-wide bf16 row and the peaked arm is 46 MTregs / 74 STregs.

The other kernels the TU emits. **The last two are unreachable from
`ds_topk`** — nothing in the tree calls `launch_topk_f32_k2048_b1024_c500` (the
only entry that would launch them), so they are compiled-in dead weight for this
reference and listed only so the resource table is complete:

| function | MTregs | STregs | static smem |
| --- | --- | --- | --- |
| `rk::topk_f32_chunk_stage1_kernel` | 36 | 32 | 2332 B |
| `rk::topk_f32_chunk_stage2_kernel` | 22 | 40 | 2332 B |
| `deep_select_maca::nan_scan_kernel<float>` | 20 | 32 | 256 B |
| `rk::topk_f32_kernel_k2048_b1024` | 16 | 62 | 3336 B |
| `rk::topk_f32_kernel_k2048_b1024_warp_scan` | 22 | 52 | 2328 B |

For scale, the geometry queried from the driver on this box: 104 APs, wavefront
64, 2048 threads/AP, **131072 registers/AP**, 64 KB shared memory/AP (and per
block), 8 MB L2. At 512 threads/CTA and 42 MTregs a CTA is nowhere near the
register file (6 CTAs' worth) or the thread cap (4); the shared memory is what
bounds residency — 18.7 KB per CTA at k=512, 24.8 KB at k=2048, so three and
two CTAs per AP respectively.

---

## 5. What was cut, and why

Nothing was improved, reordered or "fixed". The kernel layer is upstream lines
1–1108 of `maca_topk.cu` byte for byte, with exactly two edits, both in the
include block:

1. `#include "structs.h"` dropped. It was included for
   `INPUT_STRIDE_ALIGNMENT_REQUIREMENT` / `OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT`,
   which only the tvm-ffi edge read (upstream `:1185`, `:1220–1222`), and it
   dragged in `maca_bfloat16.h`; `radix_core.cuh` includes that itself, which
   is where the 16-bit arms still get it. Keeping it would mean this directory
   depends on `../csrc/structs.h` for nothing.
2. `#include "radix_core.cuh"` → `#include "xcore1000_radix_core.cuh"`.

The file that got *added* is the host entry at the bottom, and the one thing
worth stating plainly is what it is not: it is not a new layer of policy. It
fills `RowParams`, calls the same `chunked_f32_applies` /
`f32_chunked_chunks` / `chunked_f32_workspace_bytes` the dispatcher calls to
size the buffer it passes to `topk_launch`, and keeps upstream's grow-only
scratch cache (`ChunkedScratch`, reproduced with its mutex) so the timed region
is the launches and not `cudaMalloc`. The per-row length table is built only
when the caller passes none, which is what upstream does for an `end`-less
call, and the NaN-flag table is the first `batches * sizeof(int32_t)` bytes of
that workspace, read back where the `kPreSelected` arm reads it.

What was **not** carried is upstream `maca_topk.cu:1112–1425`: the
`deep_select::topk(TensorView, …)` entry, the dtype/shape/stride/device checks,
`TVMFFIEnvGetStream`, and the `../ffi/ffi_entries.h` registration tail. The two
error macros the body uses (`DS_HOST_CHECK`, `DS_CUDA_RUNTIME_CHECK`) are
re-spelled locally with the same names: upstream they raise `tvm::ffi::Error`,
here they print and `std::abort()`. A `void` C entry has no error channel, and a
reference that swallowed a failed `cudaMalloc` would print a GB/s number for a
run that never happened.

`xcore1000_maca_topk.cu` got no edits beyond a banner comment and the two
includes. In particular the 16-bit machinery is still there and still compiles;
it just never runs, because `ds_topk` only ever passes `value_dtype = 0`.

**Which revision of `maca_topk.cu` this is.** The working tree's, not committed
`HEAD`'s. Checked rather than assumed: the ref head matches
`csrc/xcore1000/maca_topk.cu` in exactly three diff hunks — the two include
edits above and trailing blank lines — while 995 lines differ from
`git show HEAD:csrc/xcore1000/maca_topk.cu`. That matters for exactly one thing
here: the `f32_chunks_large_batch` / `kF32OverflowChunkLen` chunk-count rule
(ref `:850–852`) is in this copy, so the split cell runs 6 chunks rather than
HEAD's constant 2. If the ref should track HEAD instead, re-copy lines 1–1108
from `git show HEAD:csrc/xcore1000/maca_topk.cu` and re-check the routing
numbers in §3.

---

## 6. Shortcuts this reference takes

Stated so nobody mistakes the repro for the shipped operator:

- **fp32 + int32 only.** `return_value`, `sorted_index`, `sorted_value`,
  `output_idx_offset` are fixed off, so the ordered emit, the int64 index path,
  the value outputs and the index offsets are compiled but not exercised here.
  `length <= topk` therefore takes the plain ascending loop rather than
  `emit_ordered`.
- **No `end_ptr` from the caller unless you pass `lengths`.** The driver's
  ragged cell does, which is the only cell that reaches the split's own
  shortcut rows. `end_ptr == nullptr` also disables the split entirely
  (`topk_launch` requires it), which is why the entry builds the all-`V` table.
- **Packed rows.** `stride_input_batch` is `n_cols * sizeof(float)`; upstream's
  1024-byte input stride alignment requirement is a tvm-ffi-layer check and is
  not reproduced.
- **The workspace is cached, not freed.** Peak RSS holds the largest shape's
  candidate arena for the life of the process — upstream's behavior, kept
  deliberately; `DEEP_SELECT_NO_SCRATCH_CACHE=1` reproduces the per-call path.
- **The `bw` column is useful traffic**, not DRAM traffic (§1).

---

## 7. Shape coverage, and where the driver — not the kernel — is the bound

**Host memory is the real bound, and it is high but not imaginary.** `run_cell`
materializes the whole score matrix on the host (`std::vector<float>
host(n_scores)`) so it can both run the CPU check and write a dump. Measured, not
estimated — `B=4096 V=524288 k=2048` runs to completion:

```
── B=4096 V=524288 k=2048 ──
  verify: PASS
  time : 29.6338 ms   (best of 1, 3 launches each)
  bw   : 291.0 GB/s  (8623.5 MB useful: 8589.9 MB read + 33.6 MB written)
  Maximum resident set size (kbytes): 8452892      (8.1 GiB)
  Elapsed (wall clock) time: 4:18.17
```

Two things that number teaches. The **kernel** is 29.6 ms of that; the other
4 minutes are the driver's randn fill and CPU `nth_element` check on 4096 rows of
524288 — so wall time here is not a kernel measurement and never was. And 8.1 GiB
resident against 2015 GB of RAM means this shape is nowhere near the ceiling:
the split's workspace adds `B * chunks * k * 8` bytes (384 MB here) and a shape
beyond host memory fails in `std::vector` with `std::bad_alloc` *before any GPU
work* — a driver limit, not a kernel one.

**The coarse-bin overflow is not forced by any default cell.** `kF32OverflowChunkLen
= 1757 * 58` (ref `:850`) exists precisely to keep a *random* chunk's threshold
bin inside the 1757-entry arena, and every cell here uses randn scores. So
`radix_topk_row_f32_rescan` (ref radix core `:464`) — the path §2 step 3 calls
out as "do not remove" — is compiled but not exercised by the default cells.
Forcing it needs a row whose values concentrate into few high-key bins, since
`float_to_uint8` bins by sign+exponent+top-3-mantissa-bits. If the cross-check
wants that arm covered, it has to be a separate cell with a purpose-built
distribution; the randn shapes will not reach it.

**Determinism, which matters for a cross-check.** The *data* is fully
reproducible: `mt19937(1234)` seeded inside `run_cell`, so two invocations write
byte-identical `scores.f32` (verified). The *answer* is not, in slot order:
two runs at `B=6 V=16384 k=2048` differed in 8466 of 12288 slots, with the
**sorted index sets identical** and the **sorted value multisets identical** —
i.e. every slot carried a value that belongs in the answer, arranged
differently. Cause: selection is a histogram pass plus a collect pass, and the
collect emits through `atomicAdd(&s_counter, 1u)`, so slot order follows atomic
scheduling. The *set* is stable when the row's values are distinct; ties are
broken arbitrarily, exactly as the contract allows.

Consequence for `agrees()`: it must run on `--dump-prefix` output rather than on
two independent invocations, and it compares sets/multisets, never slot order.
The per-row value multiset is the strongest form and is what I verified against
numpy for dumped cells. Two rules that look tempting and are both wrong here:

- **Do not** use a tie-tolerant slot rule of the form "differing slots carry
  equal values". With ~16382 distinct values in a 16384-long row, 8466 slots
  reordering is not tie noise — those are *distinct* elements reordering.
- **Do not** compare slot order at all, even against `torch.topk` on the same
  bytes. The kernel's order is unspecified and varies run to run.

The form that holds is per row:

```python
sel = np.take_along_axis(row, idx_row, axis=0)          # what the kernel chose
ref = np.sort(row)[::-1][:K]                            # value frame, ties included
assert np.array_equal(np.sort(sel), np.sort(ref))       # equal value multisets
```

That is strictly stronger than slot-wise agreement and is the framing the
`length <= topk` rows need too, where it degenerates to `sel == 0..len-1` padded
with `idx_fill`.

**Heterogeneous per-row windows are exercised on the split path only.**
`lengths` is passed non-null exactly once, by the ragged cell
(`B=8 V=65536 k=512`, lengths `{21845, 65536, 256, 65536, 21845, 256, 21845,
65536}` — whole rows, long prefixes, and two windows below `k` that take the
shortcut). That is the split path (`B <= 64`, `V >= 32768`). The row path reads
the same `end_ptr[row]`, one CTA per row, so the mechanism is identical — but it
is **not measured** here. State it that way rather than claiming per-row windows
work on both.

**No explicit shape guard beyond the one in `ds_topk`** (§5): `n_rows > 0`,
`n_cols > 0`, `0 < top_k <= kMaxTopK` (4096), `sm_count > 0`. Everything else —
vocabulary size, batch, `top_k` against dtype — is routed rather than refused,
which is what the predicates in §3 are for. `kF32MaxTopK = 4096` (ref radix
core `:2376`) is the engine's bound and is the same number.

---

## 8. `--dump-prefix`, for cross-checking against another implementation

The self-check in §1 is necessary but not sufficient: three implementations can
each be self-consistent and still disagree with the framing the library ops use.
To compare on identical data, the driver writes the exact bytes it selected from
and the answer it produced:

```bash
./ds_topk_ref --dump-prefix /tmp/ds 6 16384 2048 5
#   dump : /tmp/ds.b6.v16384.k2048.scores.f32 (393216 B)
#          /tmp/ds.b6.v16384.k2048.idx.i32    (49152 B)
```

Both files are raw little-endian host bytes with no header, row-major, shape
implied by the tag (`<prefix>.b<rows>.v<cols>.k<topk>`), so one run of the four
default cells drops twelve files that cannot collide. A ragged cell additionally
writes `.lengths.i32`, which is what the comparison needs to bound each row's
window. The flag is accepted before or after the positional args; positional-only
invocations behave exactly as before.

```python
scores = np.fromfile(p + ".scores.f32", dtype=np.float32).reshape(B, V)
idx    = np.fromfile(p + ".idx.i32",    dtype=np.int32  ).reshape(B, K)
```

The data is the driver's own `mt19937(1234)` / `normal(0,1)` draw, so it is
reproducible from the shape alone — the dump removes the "did we both draw the
same numbers" question from the comparison rather than answering it by hand.

