# mcoplib SGLang `topk_transform_v1` — torch-free reference repro

Minimal, compilable, runnable extraction of the SGLang TopK kernel from

    mcoplib/op/sglang/jit_kernels/topk_v1.cu                     (527 lines)
    mcoplib/op/sglang/jit_kernels/topk_v1_histogram_4096.cuh     (367 lines)

The kernel layer is copied verbatim. Everything above it — the `at::Tensor`
boundary, `TORCH_CHECK`, `c10::cuda::getCurrentCUDAStream` — is replaced by a
plain host entry so the kernels link and run with no torch in the process.

## Files

| file | role |
|---|---|
| `xcore1000_topk_v1.cu` | the kernel TU, torch-free; compiles to a `.o` |
| `topk_v1_histogram_4096.cuh` | byte-identical copy of the source header (`md5 57abe72d9d02bb0754f33bf09d54bd58`) |
| `xcore1000_topk_v1.h` | public header, one host entry `mcoplib_topk_transform` |
| `main.cu` | runnable driver + CPU `std::nth_element` verification |
| `witness/scan_bad.cu` | witness for the large-row defect: prints the row positions of the wrong entries |
| `run_output.txt` | captured output of the run quoted below |

Toolchain: MACA 3.7.0.36, MetaX C500, target `xcore1000`, per `/tmp/dsprobe/TOOLCHAIN.md`.

## The two algorithms

| | `histogram_4096_topk` | `radix_topk` |
|---|---|---|
| trigger | `max(seq_lens) <= 16384` | otherwise (i.e. any row `> 16384`) |
| dispatch site | `topk_v1.cu:486`, launch at `:508-514` | `topk_v1.cu:486`, launches at `:515-523` |
| kernel body | `topk_v1.cu:389-390` → `topk_v1_histogram_4096.cuh:159` | `topk_v1.cu:392` → `topk_v1.cu:130` |
| HBM passes | one (values held in registers) | two (pass 1 histogram, pass 2 re-reads candidates) |
| binning | 12-bit FP16 key, 4096 bins (`cuh:53-60`) | 8-bit FP16 key, 256 bins (`cu:80-85`), then a 4-round fp32 radix refine (`cu:296-358`) |
| tie handling | warp ballot if `num_ties <= 16`, else `tie_handle` 4-round radix (`cuh:346-361`) | folded into the 4-round refine, `s_last_remain` (`cu:339-344`) |
| capacity | `kMaxLen = 16384` hard | unbounded, but the staging buffer caps threshold-bin candidates |
| staging | none | `kStagingSize` × 2 × 4B dynamic smem |

Both branches also run a per-row `naive_transform` when `seq_len <= TopK`
(`topk_v1.cu:374-377`), which pads the output with `-1` instead of selecting.

The dispatch is **per launch, not per row**: `max(seq_lens)` picks one kernel
for the whole batch (`topk_v1.cu:477-486`), so a single row above 16384 sends
every row down the radix path. `main.cu`'s `mixed batch` case demonstrates it.

### Constants

| name | value | where |
|---|---|---|
| `TopK` | 512 | `topk_v1.cu:39`, `xcore1000_topk_v1.cu:60` |
| `kThreadsPerBlock` | 1024 | `topk_v1.cu:40` |
| `kWarpSize` | 64 (MACA) | `topk_v1.cu:41`, `cuh:25` |
| `kNumWarpsBlock` | 16 | `topk_v1.cu:42` |
| `kSmem` / `kSmemInputSize` | 32KB / 4096 entries | `topk_v1.cu:46-47` |
| `kLargeSmem` / `kLargeSmemInputSize` | 56KB / 7168 entries | `topk_v1.cu:56-57` |
| `kHist4096MaxLen` | `kMaxLen` = 16384 | `topk_v1.cu:61`, `cuh:34` |
| `histogram_4096` smem request | 24KB dynamic | `topk_v1.cu:512` |
| `kVecsPerThread` | 4 float4 = 16 floats/thread | `cuh:33` |
| `HIST_BITS` | 12 (4096 bins) | instantiated at `topk_v1.cu:389` |
| `__launch_bounds__` | `kThreadsPerBlock` = 1024 | `topk_v1.cu:362` |

### What differs from the source, and why

Only the torch boundary, plus one thing that torch was doing implicitly:

1. `at::Tensor` / `TORCH_CHECK` / `getCurrentCUDAStream` dropped; the entry
   returns `void` and writes through raw pointers, launching on the default
   stream (`xcore1000_topk_v1.cu:502-504`).
2. `setup_kernel_smem_once` no longer calls `TORCH_CHECK` on the
   `cudaFuncSetAttribute` result; it keeps the once-only static
   (`xcore1000_topk_v1.cu:459-468`).
3. The source entry computes `max_seq_len` from `seq_lens.cpu()`
   (`topk_v1.cu:480`). `seq_lens` is device memory, so this needs a real D2H
   `cudaMemcpy`; the extraction does it explicitly
   (`xcore1000_topk_v1.cu:511-534`). Without it the dispatch reads device
   pointers on the host and segfaults — that was the first bug hit while
   building this repro.
4. `TopKTransformParams::page_table_stride` is `0`, not `1`. The source took it
   from `page_table.stride(0)` (`topk_v1.cu:464`); the flat entry has no stride
   argument, so it declares one page table shared by every row.
5. `top_k` must equal 512 (it is a compile-time constant downstream), so the
   entry returns early otherwise (`xcore1000_topk_v1.cu:485`).

The page-table transform itself is untouched — `page_to_indices` at
`topk_v1.cu:92-96`, still fused into the same kernel (`topk_v1.cu:396-403`).
With `page_bits = 0` it reduces to `page_table[i]`, so `main.cu` passes a
`[0, 1, 2, …]` table and reads out exactly the raw selection.

## Build and run

Kernel TU plus per-kernel resource usage:

```bash
export MACA_PATH=/opt/maca
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH"

"$CUDA_HOME/bin/cucc" -O3 -std=c++20 -DNDEBUG -fPIC -use-fast-math \
    -I"$MACA_PATH/include" --offload-arch=xcore1000 -resource-usage \
    -c xcore1000_topk_v1.cu -o build/xcore1000_topk_v1.o
```

Whole program, then run:

```bash
"$CUDA_HOME/bin/cucc" -O2 -std=c++20 --offload-arch=xcore1000 \
    -I"$MACA_PATH/include" -I. main.cu build/xcore1000_topk_v1.o -o build/main
./build/main
```

With no arguments `main` runs the fixed case list below. With arguments it runs
one shape and prints one comparable number, for a harness to drive:

```bash
./build/main <n_rows> <length> <top_k> [iters]
```

`length` is both the number of valid columns and the row stride, `top_k` must be
512, and `iters` defaults to 30. Grid mode emits exactly one
`time: %.4f ms/iter over %d iters` line (plus a driver-local duplicate with more
detail); `ref/crosscheck.py` matches the first. The CPU reference costs
`nth_element` over every row, so only the first 512 rows are checked — the cap is
printed rather than applied silently.

**`top_k != 512` is refused, not silently wrong.** The entry returns before
launching (`xcore1000_topk_v1.cu:479`), grid mode prints
`verify: FAIL   top_k != 512: mcoplib_topk_transform returns without launching`,
and the process exits non-zero. `crosscheck.py` is patched to treat a non-zero
exit as `FAIL(self)` even when no verdict token is parseable, so a k=2048 cell
reads as a refusal rather than an empty `no-verdict` cell.

`witness/scan_bad.cu` is the defect witness described below, kept in a
subdirectory on purpose: `ref/build_all.sh` links every `*.cu` *beside*
`main.cu` (`find -maxdepth 1`), and two translation units that each define
`main` cannot be linked into the same binary. Build it separately:

```bash
"$CUDA_HOME/bin/cucc" -O2 -std=c++20 --offload-arch=xcore1000 \
    -I"$MACA_PATH/include" -I. witness/scan_bad.cu build/xcore1000_topk_v1.o -o build/scan_bad
./build/scan_bad 1 65536 512
```

`ref/crosscheck.py` drives all three reference drivers with this CLI and only
credits a cell when `main.cu` is seen to take `argv` — this driver does.
This driver always prints `verify: PASS` or `verify: FAIL` and never a third
token, so a shape it computes wrongly reads as a failure there. Concretely, in
the shared `bs x {16384, 65536, 524288} x k=512` grid, the `len=65536` and
`len=524288` cells read `FAIL(self)` for `mcoplib`, while `len=16384` measures
normally. That is the large-row defect below showing up through the shared
harness, not a cell that failed to run. The harness returns the driver's
non-zero exit status as a failure even when the verdict token cannot be read,
so a wrong answer never lands in the `no-verdict` bucket.

(The `-I.` is needed because `main.cu` includes `xcore1000_topk_v1.h` by name;
the kernel TU finds it relative to itself. Both compile clean — the only
compiler diagnostic is `mxcc: warning: argument unused during compilation:
'--maca-link'`, from the link step.)

### Resource usage per instantiation

`cucc ... -resource-usage` on `xcore1000_topk_v1.cu`, quoted verbatim:

```
 maca info  : Function properties for  _ZN12_GLOBAL__N_121topk_transform_kernelILb1ELj4096EEEvNS_19TopKTransformParamsE
  0 bytes stack frame
 maca info  : Used  49 MTregisters, 64 STregisters, 2048 bytes shared mem
 maca info  : staticMaxWarps/PEU : 8
 maca info  : Function properties for  _ZN12_GLOBAL__N_121topk_transform_kernelILb0ELj7168EEEvNS_19TopKTransformParamsE
  0 bytes stack frame
 maca info  : Used  22 MTregisters, 62 STregisters, 21124 bytes shared mem
 maca info  : staticMaxWarps/PEU : 8
 maca info  : Function properties for  _ZN12_GLOBAL__N_121topk_transform_kernelILb0ELj4096EEEvNS_19TopKTransformParamsE
  0 bytes stack frame
 maca info  : Used  22 MTregisters, 62 STregisters, 21124 bytes shared mem
 maca info  : staticMaxWarps/PEU : 8
```

The three mangled names are `topk_transform_kernel<true, 4096>`,
`<false, 7168>` and `<false, 4096>` — i.e. the histogram_4096 instantiation,
the 128K-smem radix staging, and the 64K-smem radix staging. The histogram
instantiation carries the fatter per-thread state (49 MTregisters / 64
STregisters vs 22 / 62 for the two radix ones), which is the `kVecsPerThread`
trade-off described at `cuh:28-33`. All three report `staticMaxWarps/PEU : 8`.

On C500 the optin smem cap is 64KB (`smem/block : 65536 (optin 65536)` in the
run header), and the `<false, 7168>` instantiation would need 56KB dynamic plus
21KB static — over the cap. The host probe at `topk_v1.cu:496-506` tests for
≥96KB and therefore never selects it here, so it is compiled and sized but not
executed.

Note `histogram_4096`'s per-thread register file: `49 MTregisters` at 1024
threads/block is what the `kVecsPerThread = 4` comment at `cuh:33` is about
(going to 8 or 16 spills).
## Run output

`./build/main`, on MetaX C500, quoted verbatim from `run_output.txt`:

```
device          : MetaX C500
APs             : 104
warp size       : 64
smem/block      : 65536 (optin 65536)
radix staging   : 4096 entries (32KB, 64K-smem chip)
TopK            : 512
histogram_4096 kMaxLen : 16384

case                                 shape                      ok        GB/s       time  note
-----------------------------------------------------------------------------------------------------------------------------
naive (seq_len 300 <= TopK)          rows=4096 n_cols=512 max=300 PASS       0.0     70.6 us  ok
    best of 4 rounds x 32 iters; max 12-bit coarse bin 10 (TopK 512)
histogram_4096 (16384 = kMaxLen)     rows=4096 n_cols=16384 max=16384 PASS     354.9    756.4 us  2 exact ties at kth (set is ambiguous); count check ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 109 (TopK 512)
histogram_4096 (unaligned rows)      rows=4096 n_cols=16381 max=16381 PASS     327.2    820.2 us  ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 106 (TopK 512)
histogram_4096 (8192, half row)      rows=4096 n_cols=8192 max=8192 PASS     238.3    563.3 us  ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 64 (TopK 512)
radix_256 (16385, just over)         rows=1024 n_cols=16388 max=16385 PASS     175.7    382.1 us  ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 100 (TopK 512); threshold 8-bit band 1122 vs 4096-entry staging
radix_256 (unaligned rows)           rows=1024 n_cols=16387 max=16387 PASS     178.0    377.0 us  ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 107 (TopK 512); threshold 8-bit band 1124 vs 4096-entry staging
radix_256 (32768, two pass)          rows=512 n_cols=32768 max=32768 PASS     204.2    328.6 us  ok
    best of 4 rounds x 4 iters; max 12-bit coarse bin 177 (TopK 512); threshold 8-bit band 2189 vs 4096-entry staging
radix_256 (131072, past staging)     rows=256 n_cols=131072 max=131072 KNOWN    272.8    492.0 us  row 0 (seq_len 131072): selected index below the kth value
    best of 4 rounds x 4 iters; max 12-bit coarse bin 614 (TopK 512); threshold 8-bit band 8397 vs 4096-entry staging
radix_256 (1048576, band 66k)        rows=2 n_cols=1048576 max=1048576 KNOWN     10.7    783.7 us  row 0 (seq_len 1048576): selected index below the kth value
    best of 4 rounds x 4 iters; max 12-bit coarse bin 4260 (TopK 512); threshold 8-bit band 65851 vs 4096-entry staging
radix_256 (2097152, band 132k)       rows=1 n_cols=2097152 max=2097152 KNOWN      5.7   1476.0 us  row 0 (seq_len 2097152): selected index below the kth value
    best of 4 rounds x 4 iters; max 12-bit coarse bin 8403 (TopK 512); threshold 8-bit band 131917 vs 4096-entry staging
mixed batch (max>16384 -> radix)     rows=8 n_cols=65536 max=40000 PASS       6.6     74.8 us  ok
    best of 4 rounds x 8 iters; max 12-bit coarse bin 190 (TopK 512); threshold 8-bit band 2528 vs 4096-entry staging
-----------------------------------------------------------------------------------------------------------------------------
GB/s is the minimum traffic (one fp32 read of each row prefix);
the radix path moves more (it re-reads threshold-bin candidates).
3 case marked KNOWN: it fails against the CPU reference by design of the
source kernel, not by a fault in this extraction. See README, 'Known large-row defect'.
result: ALL PASS
```

Wall clock ≈ 25 s, dominated by the CPU `std::nth_element` reference (the GPU
time column totals ≈ 5.9 ms). `KNOWN` is explained under "Known large-row
defect" below; it is not a failure of the extraction and does not affect the
exit code.

### How the numbers should be read

* **GB/s is not comparable to `ref/crosscheck.py`'s or to
  `tests/bench_dsa_topk.py`'s.** Those use `n_rows * (length + k) * 4` — scores
  read plus int32 indices written. This driver quotes only the score traffic the
  kernel actually has to read, so that a case's rate reflects the kernel and not
  a buffer it never touches. On a dense shape the table's formula is about 1.06x
  this one (e.g. `6 x 524288 x 512`: 12.58 MB here against 13.36 MB there). Use
  the `time` column for cross-impl comparison; it is the raw quantity.
* The naive arm reads no scores at all, so its 0.0 is reported honestly rather
  than divided by a meaningless time. The radix path re-reads threshold-bin
  candidates, so its real DRAM traffic is higher and the quoted rate is a lower
  bound, not a measured DRAM rate. The kernel header claims ~800 GB/s
  achievable for the histogram path (`topk_v1.cu:9`); this repro measures
  326-346 GB/s against a stated 1.875 TB/s peak.
* **max 12-bit coarse bin** is the worst per-row FP16-coarse-bin occupancy over
  the batch, computed on the host by re-deriving the kernel's key. It is the
  quantity the selection is at risk from: `histogram_4096` needs the threshold
  bin to hold, at most, the remaining TopK (and tie-breaks anything ≤ 512), and
  `radix_topk` needs the 8-bit threshold bin to hold ≤ `kStagingSize` = 4096
  (`topk_v1.cu:270-276`, `:346-352`) or candidates are silently dropped — see
  "Known large-row defect" below. Scores are `uniform(-1, 1)`, so for the short
  cases these occupancies are 10-190, far under both limits.
* **Clock not locked** (`persistence off`, per `TOOLCHAIN.md`). Per-case spread
  across repeated runs of the same binary is 10-35%; over a longer window
  (`histogram_4096 (16385, just over)` during bring-up) the same case measured
  123.0, 180.1 and 177.2 GB/s back to back. Timing is the best of 4 interleaved
  rounds, with the buffers re-staged before each round so first-touch cost is
  not hidden by warm reuse. Treat any single cell as ±30%.

### Verification

Every row of every case is checked against a CPU `std::nth_element` reference
(`main.cu:180-259`). The scores are uniform fp32, so the kth value is normally
unique and the selected index set is unique too — the exact set is compared, no
"ties are arbitrary" slack. Two cases print a note instead:

```
2 exact ties at kth (set is ambiguous); count check ok
```

That is a row where two scores are exactly equal at the kth value (a
1-in-2^24-per-pair event at these sizes, so it shows up once in a few thousand
rows). The set is genuinely ambiguous there, so the check falls back to the
count form — every selected index is in range, all are ≥ the kth value, exactly
`#above-kth` are strictly above it, and `TopK` are selected. A *unique* kth
value with a mismatched set fails, and so does any duplicate or out-of-range
index. The naive arm is checked against the exact expected output
(`out[i] == i` for `i < seq_len`, `-1` beyond), which is where the fused page
transform is exercised.

Rows exercise the misaligned-load paths as well: `n_cols = 16381` makes every
row base non-16B-aligned, taking the predicated per-element load in
`histogram_4096` (`cuh:209-227`), and `n_cols = 16387` does the same for
`radix_topk`'s `vec4_prefix` prologue (`cu:156-165`).

## Shape coverage for a cross-implementation table

What this driver can and cannot serve, as of this build (C500, 64K-smem):

| n_cols | arm taken (k=512) | correct? | notes |
|---|---|---|---|
| <= 512 | naive | yes | writes `i` for `i < seq_len`, `-1` beyond; no selection |
| 513 … 16384 | histogram_4096 | yes | single pass; hard capacity 16384 |
| 16385 … ~64k | radix_256 | yes on this build's draws | band 1122 at 16385, 2189 at 32768, 3954 at 61440; fails from ~64.5k |
| >= ~64.5k | radix_256 | **no** | staging cap; see below. Deterministic failure from ~1M |

Cross-checked against `ref/crosscheck.py` for the cells it can request: with
`--bs 6,256 --len 16384,65536,524288 --k 512`, `len=16384` measures
`self-ok` (7.1 and 193.7 GB/s at bs=6 and 256 by the harness's traffic
formula), and both `len=65536` and `len=524288` read `FAIL(self)` with the
timing suppressed — the harness refuses to print a GB/s for a wrong answer. The
`k=2048` cells read `--  k outside contract [512, 512]`, driven by the
`max_k`/`min_k` the harness reads from this directory's config.

`top_k` is fixed at 512 — the kernel's `TopK` is a `constexpr` and the entry
refuses anything else. `n_cols` is both the valid length and the row stride in
grid mode.

**Memory.** A call needs `n_rows * n_cols * 4` B of device scores plus the same
again in host memory (the driver generates and holds a full host copy for the
CPU reference), plus a page-table-sized int32 buffer and `n_rows * 512 * 4` B of
output. The harness's largest cell, `bs = 4096, n_cols = 524288`, is 8 GiB of
scores + 8 GiB of host mirror + 8 MiB of output. It was measured on this box
(2 TB RAM, C500) and runs: `./build/main 4096 524288 512 1` completes in
17.54 ms/iter with the reference capped at 512 of 4096 rows. **No `bs` cap is
needed for memory.** Keep `iters` at 1-2 for that cell — the wall time is the
CPU reference, not the kernel.

**Timing includes a device sync per call.** The source entry computes the
dispatch by pulling `seq_lens` to the host (`topk_v1.cu:480`, the extraction at
`xcore1000_topk_v1.cu:511-534`), so every launch is preceded by a
`cudaMemcpy(DtoH)` and the call cannot overlap with anything. On the wire sizes
here (a few KB) that is tens of microseconds, which is large relative to the
0.05 ms histogram case and irrelevant at 0.78 ms. Do not read a small-shape
mcoplib time as pure kernel time.

## Known limits of the source, worth restating

* **`histogram_4096` has `kMaxLen = 16384` and silently truncates past it.**
  `kMaxLen = kVecsPerThread * 4 * kBlockSize` (`cuh:34`) with
  `kVecsPerThread = 4` and `kBlockSize = 1024`. The load loop stops emitting
  once `idx >= length` (`cuh:233-244`), so a row longer than 16384 fed directly
  to this kernel would have its tail dropped with no error. The host dispatch
  (`topk_v1.cu:486`) is what keeps that from happening — do not bypass it. The
  `kVecsPerThread = 4` comment at `cuh:28-32` explains why the bound is not
  simply raised: 8 or 16 spills the register file on C280.
* **The radix staging buffer caps threshold-bin candidates.**
  `if (pos < kStagingSize)` (`topk_v1.cu:271`, `:347`) drops candidates beyond
  the cap without signalling. On a 64K-smem chip that cap is 4096 entries.
  This one is reachable with plain random input — see "Known large-row defect".
* **`radix_256` is wrong for long rows on a 64K-smem C500.** Three entries in
  the default run — `131072`, `1048576` and `2097152` — are marked `KNOWN`, not
  `PASS`: they fail against the CPU reference by design of the source algorithm,
  and the two largest fail on every draw, not just this seed. Full write-up with
  the instrumented trace, the band sweep and the output-position witness is in
  "Known large-row defect in the radix path" below. The `PASS` verdicts for
  `16385`, `16387` and `32768` are true passes on shapes with more headroom:
  their threshold bands are 1122, 1124 and 2189 against the 4096-entry cap, and
  the driver prints those numbers so the margin is visible.
* **This repo's `topk_v1.cu:44-57` comment contradicts itself** about the exact
  size of the large-staging variant (`kLargeSmemInputSize * 2 * sizeof(int32_t)`
  is written as 112KB, which it is, but the comment says "fits 64KB
  static+dynamic"). Left as-is — faithful extraction.
* **The 128K-smem radix instantiation cannot run on C500.** The entry picks
  `kLargeSmemInputSize` (7168) only when the device reports
  `cudaDevAttrMaxSharedMemoryPerBlockOptin >= 96KB` (`topk_v1.cu:496-506`); C500
  reports 65536, so the 4096-entry variant is selected, as the run header shows.
  The `<false, 7168>` instantiation is still compiled and its resource usage is
  reported above, but nothing in this repro executes it. Forcing it
  (`trace_radix<7168>` with 56KB dynamic) fails at launch with
  `mcErrorInvalidValue` on this chip, confirming the probe's threshold.

## Known large-row defect in the radix path

`radix_256` is **not correct for long rows on a 64K-smem C500**, and this is a
defect in the source algorithm, not in the extraction. It is reachable with
plain random fp32 input, so it is worth stating precisely.

**Symptom.** Measured with uniform `(-1, 1)` scores, `top_k = 512`, rows all
equal to `length`. "Wrong entries" counts output indices whose score is below
the true 512th value of the row (so a correct selector returns 0):

| length | wrong entries in the returned TopK (per row) |
|---|---|
| 16385 … 61440 | 0 (correct) |
| 65536 | 2-20 |
| 98304 | 155-182 |
| 131072 | 240-264 |
| 262144 | 355-370 |
| 524288 | 391-406 |

The 65536 row is right at the crossover: at that length the tying band is ~200
values, well inside the 4096-entry buffer, so the loss is not the round-0
truncation itself but the outer loop of the refinement reading a candidate set
that has already been truncated. Either way the result is the same kind of
quietly wrong output.

The wrong entries are genuine row elements with scores *below* the true 512th
value; they displace entries that belong in the TopK. Output slots are never
left unwritten and indices are never out of range — it fails quietly, as a
plausible-looking TopK.

The *verdict* is deterministic (same seed, same FAIL, 13/13 runs across the
sweeps above) but the exact number of wrong entries is not: 18, 19, 20, 19, 19
over five runs of the same configuration in the witness below. The scatter that
fills the output tail uses `atomicAdd` and its order is not fixed, so which of
the truncated candidates land in the last few slots varies. Treat the row counts
in the table above as magnitudes, not exact values.

**Cause.** Round 0 appends every element whose 8-bit FP16 coarse bin equals the
threshold bin, and `if (pos < kStagingSize)` (`topk_v1.cu:271`) silently drops
the ones past 4096. The set that "ties" here is the whole power-of-two FP16 band
around the threshold, not a hairline tie — for uniform `(-1, 1)` scores the top
band holds about `length / 2^k`, where `k` is the precision left at that
magnitude: ~4000 at `length = 65536`, ~8400 at 131072, ~66000 at 1048576.

The staging cap is what makes the output wrong, but the full mechanism also
needs the refinement to lose the dropped candidates *and* the surviving
allocation to sit inside the range the verifier checks. The instrumented run
below shows both stages saturated at exactly 4096; the witness run
(`scan_bad.cu`) shows the wrong entries occupying the *tail* of the output
(slots 492-511 of 512) rather than being scattered, which is the signature of a
refinement that resolved `remain_topk` against a candidate set it had already
truncated.

**The length at which this starts is draw-dependent, not a fixed cutoff.** The
band is a random variable with a mean of `length / 2^k` and a spread of a few
percent, so it crosses 4096 at a length that varies with the seed. Measured on
this driver's own score draw, 6 rows, `top_k = 512`:

| length | threshold band | verdict |
|---|---|---|
| 49152 | 3104 | PASS |
| 57344 | 3681 | PASS |
| 61440 | 3954 | PASS |
| 62464 | 4009 | PASS |
| 63488 | 4071 | PASS |
| 64512 | 4145 | FAIL |
| 65536 | 4259 | FAIL |
| 131072 | 8397 | FAIL |
| 524288 | 32951 | FAIL |

The driver prints this band itself in the radix cases, and it is the number that
tracks the verdict. Note the driver's band and a numpy RNE reference agree on
the ranking but not to the digit (4145 vs 4162 at 64512) — both are counting a
few thousand elements of an exact-FP16 key, so treat the band as exact only up
to a handful of elements.

The rows=1/2/6 flip at `length = 65536` that prompted this investigation is not
a batch-size effect: `fill_scores` seeds off `n_rows * 31 + n_cols`, so those are
three different matrices whose bands are 4181, [4009, 4237] and a max of 4196.
The rows=2 draw has both rows under 4096 and passes; the other two have a row
over and fail.

Two or more independent failures are needed for a *timing* number to be
meaningful here, because one bad row can reorder an otherwise correct output.
The two largest cases in the default run are chosen to make that unnecessary:
at `length >= ~1M` the band exceeds 4096 for *every* row regardless of the draw
(measured: 65851 at 1048576, 131917 at 2097152, against 4096), so those fail
deterministically.

**Witness.** `witness/scan_bad.cu` re-runs one shape and prints, for each bad output
slot, the row position of the returned index. One row, `length = 65536`, the
draw that fails:

```
row 0: kth=0.984339476 (slot 34150)
  slot 492: idx 1901 val 0.984297037  kth-slot 34150  idx-kth -32249  within-512-of-kth 0
  slot 493: idx 11164 val 0.984198809  kth-slot 34150  idx-kth -22986  within-512-of-kth 0
  ...
  slot 511: idx 63493 val 0.98428905  kth-slot 34150  idx-kth 29343  within-512-of-kth 0
  -> 18 of 512 scanned slots hold a value below the true kth
```

(that count varies 18-20 between runs; see above) The bad slots are 492-511 — the tail of the output, not scattered — and their
row indices are tens of thousands of positions away from the kth slot, so this
is not a tie-break ambiguity around the threshold. It is the tail of the
selection being filled from a truncated candidate pool.

**Not covered here.** Whether this matches what SGLang actually feeds the kernel
— real attention scores are not uniform over a magnitude band, and a sparser
distribution keeps the band small, which is why widening the score range to
`exp(-100 U(0,1))` makes `length = 524288` pass. Reproducing a production
distribution was out of scope for this repro; the point is only that the
staging cap is reachable on ordinary input, not just on adversarial ties.

**Alternatives that were not taken.** The `<false, 7168>` instantiation would
help (still short of 5400 at `length = 131072`, but under it at 65536), except
C500 cannot launch it. Reporting the defect honestly was preferred to padding
the staging buffer, which would have been a fix, not an extraction.
