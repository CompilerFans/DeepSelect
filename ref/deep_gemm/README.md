# `ref/deep_gemm` — a runnable, torch-free extraction of the deep_gemm fp32 indexer TopK

A minimal repro of the fp32 indexer TopK selector that lives in
`mcDeepGEMM/csrc/kernels/fp32_topk.cu`. Both `cucc` invocations below exit 0 and
the driver runs and verifies against a CPU `std::nth_element` reference. No
torch, no pybind11, no tvm_ffi, no `deep_gemm` package import anywhere in the
build.

| file | what it is |
| --- | --- |
| `xcore1000_fp32_topk.cu` | the kernel TU, lines 1..1584 of the original (`namespace detail`) plus a torch-free host entry at the end |
| `xcore1000_fp32_topk.h` | the plain header declaring that entry point |
| `main.cu` | runnable driver: randn scores, run, verify, time |
| `resource_usage.txt` | raw `-resource-usage` output for the 14 instantiations |
| `main`, `xcore1000_fp32_topk.o` | build products, left in place |

## Build

```sh
export MACA_PATH=/opt/maca
export CUDA_HOME="$MACA_PATH/tools/cu-bridge"
export CUCC_PATH="$CUDA_HOME"
export LD_LIBRARY_PATH="$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib:$LD_LIBRARY_PATH"

# 1. the kernel TU -> .o
"$CUDA_HOME/bin/cucc" -O3 -std=c++20 -DNDEBUG -fPIC -use-fast-math \
    -I"$MACA_PATH/include" --offload-arch=xcore1000 \
    -c xcore1000_fp32_topk.cu -o xcore1000_fp32_topk.o
# exit=0 ; xcore1000_fp32_topk.o is 450032 bytes

# 2. the whole program
"$CUDA_HOME/bin/cucc" -O2 -std=c++20 --offload-arch=xcore1000 \
    main.cu xcore1000_fp32_topk.cu -o main
# exit=0 ; main is 513440 bytes
```

Both commands were run verbatim from this directory, on the sources as they
stand now, after the last edit. The only build difference between the two is
that the whole-program link is at `-O2` (matching TOOLCHAIN.md's whole-program
line) and does not pass `-DNDEBUG`; the kernel TU builds standalone at
`-O3 -DNDEBUG -use-fast-math` as instructed.

## Run

`./main [--dump-prefix P] [n_rows] [n_cols] [top_k] [iters] [seq_len]` — `seq_len`
defaults to `n_cols` (scan whole rows); a smaller value exercises the
`length <= top_k` padding arm, and the reference honours it too.

`--dump-prefix P` writes the exact input the kernel selected from, plus the
kernel's answer, and skips the normal verify/timing path:

```
$ ./main --dump-prefix /tmp/dg 6 16384 2048 5
device MetaX C500 | APs 104 | warp 64 | regs/AP 131072 | smem/AP 64 KB | L2 8 MB
shape n_rows=6 n_cols=16384 top_k=2048 seq_len=16384 iters=5
chunks chunk_count=4 (APs=104) | workspace bytes (max over 3..6): 283200
selected policy: Chunks (1)
dump: /tmp/dg.scores.f32 (6 x 16384 fp32) | /tmp/dg.idx.i32 (6 x 2048 int32)
$ ls -la /tmp/dg.*
-rw-r--r-- 1 gfx gfx  49152 /tmp/dg.idx.i32      # 6 * 2048 * 4
-rw-r--r-- 1 gfx gfx 393216 /tmp/dg.scores.f32   # 6 * 16384 * 4
```

`scores.f32` is `n_rows * n_cols` row-major float32, `idx.i32` is
`n_rows * top_k` row-major int32. The generator is the driver's own, so the data
is reproducible; the whole row is written, not just the `seq_len` prefix, so a
consumer can reconstruct `col_eff` itself. Independently checked with numpy:
the sorted multiset `scores[row][idx[row]]` equals `sort(scores[row])[-top_k:]`
for all 6 rows, and every index is in range.

Actual output, unedited:

```
$ ./main 1024 4096 128 50
device MetaX C500 | APs 104 | warp 64 | regs/AP 131072 | smem/AP 64 KB | L2 8 MB
shape n_rows=1024 n_cols=4096 top_k=128 seq_len=4096 iters=50
chunks chunk_count=0 (APs=104) | workspace bytes (max over 3..6): 48332800
selected policy: Single (2)
time: 0.251 ms/iter over 50 iters
read bandwidth: 66.8 GB/s (0.017 GB read per call)
verify: 1024/1024 rows match CPU nth_element reference

$ ./main 1024 65536 2048 50
device MetaX C500 | APs 104 | warp 64 | regs/AP 131072 | smem/AP 64 KB | L2 8 MB
shape n_rows=1024 n_cols=65536 top_k=2048 seq_len=65536 iters=50
chunks chunk_count=0 (APs=104) | workspace bytes (max over 3..6): 48332800
selected policy: Coarse12 (3)
time: 0.578 ms/iter over 50 iters
read bandwidth: 464.3 GB/s (0.268 GB read per call)
verify: 1024/1024 rows match CPU nth_element reference

$ ./main 16 16384 512 50
device MetaX C500 | APs 104 | warp 64 | regs/AP 131072 | smem/AP 64 KB | L2 8 MB
shape n_rows=16 n_cols=16384 top_k=512 seq_len=16384 iters=50
chunks chunk_count=4 (APs=104) | workspace bytes (max over 3..6): 755200
selected policy: Chunks (1)
time: 0.033 ms/iter over 50 iters
read bandwidth: 32.2 GB/s (0.001 GB read per call)
verify: 16/16 rows match CPU nth_element reference
```

One shape per policy, each one the shape that policy is routed to. The clock
is not locked (TOOLCHAIN.md: "persistence off — expect run-to-run drift"), and
that is visible: across four repeats the single-policy shape moved between
0.173 / 0.184 / 0.242 / 0.251 ms and the chunks shape between 0.032 / 0.033 /
0.059 ms. Read bandwidth is `n_rows * n_cols * 4 / time`, so the small-grid rows
(0.001–0.017 GB per call) are launch-latency-bound and their GB/s figure is not
meaningful; only the 0.268 GB/call coarse12 row is in streaming territory, at
424–464 GB/s against the 1843.2 GB/s peak.

## The three policies

| policy | enum | kernel(s) | when `select_topk_policy` picks it |
| --- | --- | --- | --- |
| `Single` | 2 | `topk_single` | the fallback |
| `Chunks` | 1 | `topk_chunks_init` → `topk_chunks_coarse_hist` → `topk_chunks_compact_refine`, `NChunks` CTAs per row | only when `n_rows <= 32` **and** `select_chunk_count` returns nonzero |
| `Coarse12` | 3 | `topk_coarse12` | `n_rows >= 128 && n_cols >= 2049 && top_k >= 256`, or `n_rows >= 1024 && n_cols >= 65536` |

Routing thresholds, `csrc/kernels/fp32_topk.cu` (line numbers in the original
tree, which for everything above line 1584 are identical to this file's):

- `select_topk_policy`, `fp32_topk.cu:1499-1510`
  - `1502` `n_rows <= 32` → `Single` or `Chunks` depending on `chunk_count`
  - `1504` `n_rows >= 128 && n_cols >= 2049 && top_k >= 256` → `Coarse12`
  - `1508` `n_rows >= 1024 && n_cols >= 65536` → `Coarse12`
- `select_chunk_count`, `fp32_topk.cu:1440-1492`
  - `1443` `kMinChunkCount = 3`, `1444` `kMaxChunkCount = 6`
  - `1445` `kMinElementsPerChunk = 4096`; `1449-1450`
    `max_chunks = min(6, n_cols / 4096)`, and `1451` returns 0 (→ `Single`)
    when that is below 3, i.e. `n_cols < 12288`
  - `1462-1466` reject chunks unless they beat `topk_single`'s last-wave
    occupancy; `1468-1475` otherwise take the largest count that still fits one
    wave; `1477-1491` otherwise the smallest count whose tail is at least half a
    wave

The chunk count is also reachable directly, via
`deep_gemm_topk_chunk_count()`. Measured on the 104-AP device through the
driver's diagnostic call (which applies the same `* 2` that
`launch_topk_chunks` does), it takes these values:

- **0** — any `n_cols < 12288`, and `n_rows > 32` at every shape in the table
  below (`Single` wins the policy, and the number is not used)
- **4** — most `n_rows <= 32` shapes, e.g. `3..32 x 16384 x 512`
- **5** — `35 x 32768 x 512`
- **6** — `9 x 24576 x 512`, `32 x 24576 x 512`, `32 x 32768 x 512`

3 was never observed on this device, though the switch instantiates it.
`deep_gemm_topk_policy` reproduces the same answers; both call the unmodified
`select_chunk_count`/`select_topk_policy`.

The doubling `num_sms * 2` is the original's, kept so the extraction does not
silently change which chunk count the kernel layer picks
(`fp32_topk.cu:1553-1554`). Note that the *policy* decision in
`fp32_indexer_topk_impl` (`fp32_topk.cu:1653-1655`) passes the **undoubled** SM
count to the same function. Both call sites exist in the file this was
extracted from, and they can disagree; `launch_topk_chunks` is the one that
actually launches, so that is the one reproduced here. The workspace sizing is
safe either way: `select_chunk_count` returns 0 or 3..6 whatever the SM count,
so a buffer sized for the largest of 3..6 always suffices.

## smem and `__launch_bounds__`

From the kernel layer, `fp32_topk.cu:22-34`:

| constant | value | where |
| --- | --- | --- |
| `kThreads` | 1024 | `:22`, `__launch_bounds__` on `topk_single`, both chunks kernels |
| `kMaxTopK` | 2048 | `:23`, size of `__shared__ int selected_indices[]` in `topk_single` |
| `kCoarseBins` | 1024 | `:24`, coarse histogram of the chunks phase 1 |
| `kFineBins` | 256 | `:25`, fine histogram of the chunks phase 2 |
| `kCandidateCapacity` | 4096 | `:26`, staging depth per buffer |
| `kWarpSize` | 64 | `:27` |
| `kTopKCandidateSmemBytes` | 32768 | `:28`, `2 * 4096 * 4` — the **dynamic** smem request for `topk_single` and `topk_chunks_compact_refine` |
| `kCoarse12Threads` | 640 | `:31`, `__launch_bounds__` on `topk_coarse12` |
| `kCoarse12CoarseBits` | 12 | `:32` |
| `kCoarse12SmemBytes` | 16384 | `:34`, dynamic smem for `topk_coarse12` |

`__launch_bounds__`: `kThreads` = 1024 on `topk_single` (`:912`),
`topk_chunks_coarse_hist` (`:1048`) and `topk_chunks_compact_refine` (`:1167`);
`kCoarse12Threads` = 640 on `topk_coarse12` (`:944`). `topk_chunks_init` (`:992`)
launches with `2 * kWarpSize` = 128 threads and carries no launch bounds.

## Registers / smem per instantiation

`-resource-usage` on the TU as built above; raw output in `resource_usage.txt`.
`MTregisters` / `STregisters` are the compiler's two register classes.

| kernel instantiation | stack | MTreg | STreg | static smem | dynamic smem | maxWarps/PEU |
| --- | --- | --- | --- | --- | --- | --- |
| `topk_single<false>` | 0 | 32 | 78 | 12680 B | 32768 B | 8 |
| `topk_coarse12<false>` | 0 | 32 | 44 | 1052 B | 16384 B | 8 |
| `topk_chunks_init<false, 3>` | 0 | 6 | 22 | 0 | 0 | 8 |
| `topk_chunks_coarse_hist<false, 3>` | 0 | 26 | 44 | 4196 B | 0 | 8 |
| `topk_chunks_compact_refine<false, 3>` | 0 | 40 | 60 | 1052 B | 32768 B | 8 |
| `topk_chunks_init<false, 4>` | 0 | 6 | 22 | 0 | 0 | 8 |
| `topk_chunks_coarse_hist<false, 4>` | 0 | 26 | 44 | 4204 B | 0 | 8 |
| `topk_chunks_compact_refine<false, 4>` | 0 | 34 | 60 | 1052 B | 32768 B | 8 |
| `topk_chunks_init<false, 5>` | 0 | 6 | 22 | 0 | 0 | 8 |
| `topk_chunks_coarse_hist<false, 5>` | 0 | 26 | 44 | 4212 B | 0 | 8 |
| `topk_chunks_compact_refine<false, 5>` | 0 | 38 | 60 | 1052 B | 32768 B | 8 |
| `topk_chunks_init<false, 6>` | 0 | 6 | 22 | 0 | 0 | 8 |
| `topk_chunks_coarse_hist<false, 6>` | 0 | 28 | 44 | 4220 B | 0 | 8 |
| `topk_chunks_compact_refine<false, 6>` | 0 | 38 | 60 | 1052 B | 32768 B | 8 |

14 instantiations: only the `Transform = false` arm of each kernel is
instantiated, which is the arm the plain (non-page-table) selector uses. The
static smem figures are the compiler's own accounting and are odd on their face
— `topk_single` reports 12680 static plus a 32768 B dynamic request, against a
64 KB smem/AP budget — but the kernel runs and the numbers are reported as
measured rather than reconciled with the source-level array sizes.

## What the extraction changed, and what it could not

Everything from line 1 to `}  // namespace detail` is kept verbatim. The
additions are:

1. **Header/includes.** `../apis/fp32_topk.h`, `../utils/device_guard.hpp`,
   `../utils/exception.hpp`, `<ATen/cuda/CUDAContextLight.h>`,
   `<c10/cuda/CUDAStream.h>` are replaced by this directory's header plus
   `cuda_fp16.h`/`cuda_runtime.h`/`cstdint`/`limits`/`cstdio`/`cstdlib`. The
   only two macros the kernel layer needs from `exception.hpp` are inlined at
   the top of the `.cu`: `DG_CUDA_RUNTIME_CHECK` (a `cudaGetErrorString` +
   `abort` tripwire) and the kernels' use of `cudaFuncSetAttribute`. No kernel
   body changed.
2. **`torch::empty` at `:1515`.** `_launch_topk_chunks` took the workspace as a
   local `torch::empty(...)` of `n_rows * sizeof(TopKChunksWorkspace<NChunks>)`
   bytes. It is now a parameter; `deep_gemm_topk_chunks_workspace_bytes()`
   reports the size and `main.cu` `cudaMalloc`s it. This is the caller-owned
   buffer the extraction asked for.
3. **`at::cuda::getDeviceProperties` at `:1553`.** Replaced by
   `deep_gemm_device_sm_count()`, a `cudaDeviceGetAttribute(
   cudaDevAttrMultiProcessorCount, 0)` — same number, no torch cache. The
   `* 2` the original applied to it at the call site is kept, so the chunk
   count this launcher picks is unchanged (see the note in the policy section
   above; the policy path passes the undoubled count).
4. **The chunks workspace type is fixed at `NChunks = 3` in the signature** and
   `reinterpret_cast` to 4/5/6 inside the switch. All four instantiations are
   packed structs with the same leading layout and only the trailing
   `candidate_indices` array differing, so the base pointer is valid for every
   arm and the size does not depend on the type parameter.
5. **Host entry.** `deep_gemm_topk_selector()` reproduces the `Auto` arm of
   `fp32_indexer_topk_impl<false>` (`fp32_topk.cu:1649-1673`).

An external `diff` of the region between `namespace deep_gemm::indexer {` and
`}  // namespace detail` against the original lines 14..1584 shows exactly two
hunks, both in the launchers and both described above (items 2 and 3/4);
everything between them is byte-identical.

Deliberately not extracted, with reasons:

- **`Transform = true`** (`_transform`, `_region_pack` entry points). They need
  a page table, `cu_seqlens_row`, `q_positions` and `region_pack`. Nothing in
  the 1..1584 range forced this — `topk_single`/`topk_coarse12`/
  `topk_chunks_*` all instantiate cleanly at `Transform = false` — so only that
  arm is built, as the repro asked. `transform_index`, `transform_region_pack`,
  `write_trivial_page_indices` and the page-table arms of
  `initialize_row_context` are kept in the source (they are referenced by the
  templates) but are never instantiated.
- **`seq_starts`.** The original has one code path that reads it, inside
  `initialize_row_context`'s non-decode branch; the untransformed selector's
  `Auto` path with `is_decode == true` never reaches it. The entry point passes
  `seq_starts = nullptr` and `is_decode = true`, which is the same thing
  `fp32_indexer_topk_selector` does for a plain `(n_rows, n_cols)` score matrix
  with one `seq_len` per row.
- **`topk_values`.** Supported and working — `deep_gemm_topk_selector` takes an
  optional `out_values` and the kernels gather it (`gather_topk_values`). What
  is *not* done is a host-side `scores.gather`; the original's comment at
  `:1688-1694` explains why that would be wrong.
- **`DeviceGuard`** (`:1624`), the `c10::cuda::getCurrentCUDAStream` (`:1625`)
  and the trailing `cudaGetLastError()` (`:1673`). The first two are torch
  stream/device management and are replaced by an explicit `cudaStream_t`
  parameter; the third is a post-launch check the driver's
  `cudaDeviceSynchronize()` subsumes.

Nothing in the 1..1584 range had to be dropped: the two torch references were
both in the launcher, not in a kernel body, and both were mechanical.

## Output contract (read this before comparing against another implementation)

`out_indices[row, 0..top_k)` holds the column indices, inside the scanned
prefix `[0, seq_lens[row])`, of the highest `top_k` values — **as a multiset,
not as a sorted list**. The three radix passes append a boundary bin's members
in whatever order the CTAs win their atomics, so the rows come back in
arbitrary order. The upstream checker says the same
(`deep_gemm/kernels/indexer/dsa/ref.py:741-745`): "Comparison is value-based
(sorted per row) because top-k is not unique when duplicate values exist:
indices may differ but the selected value multiset must match the reference."

Where the scanned prefix is shorter than `top_k`, the tail is padded with `-1`
indices and `-inf` values, in both the kernel and `main.cu`'s reference. This
is a real, working arm (the `length <= top_k` arms of all three policies); it is
exercised here via `./main 16 16384 512 5 300`:

```
$ ./main 16 16384 512 5 300
device MetaX C500 | APs 104 | warp 64 | regs/AP 131072 | smem/AP 64 KB | L2 8 MB
shape n_rows=16 n_cols=16384 top_k=512 seq_len=300 iters=5
chunks chunk_count=4 (APs=104) | workspace bytes (max over 3..6): 755200
selected policy: Chunks (1)
time: 0.013 ms/iter over 5 iters
read bandwidth: 77.9 GB/s (0.001 GB read per call)
verify: 16/16 rows match CPU nth_element reference
```

(`seq_len = 300 < top_k = 512`, so all 16 rows take the padding arm and the
kernel still scans only 300 columns each — ~0.006–0.015 ms rather than the
0.033 ms the same shape costs at `seq_len = 16384`.)

Timing caveat worth knowing before you trust any number here: the *first*
kernel launch after a build can be hundreds of times slower than steady state.
Immediately after a rebuild this exact command reported 3.083 ms/iter, while six
consecutive re-runs of the same binary gave 0.008–0.015 ms. The driver does one
warmup launch before it starts the timed loop, and that is not enough to absorb
it. Treat any single-sample figure in this README as indicative; the tables
above were taken from repeated runs.

## Shapes that were verified

Every row of every shape below was checked against `std::nth_element`; the
reported `verify: N/N` is the driver's own count.

| shape | policy | chunk count |
| --- | --- | --- |
| `3..8 x 8192 x 512`, seq_len 8192 | Single | 0 (`n_cols < 12288`) |
| `9 x 24576 x 512`, seq_len 24576 | Chunks | 6 |
| `16 x 16384 x 512`, seq_len 16384 | Chunks | 4 |
| `16 x 16384 x 512`, seq_len 300 | Chunks | 4, `length <= top_k` padding arm |
| `24 x 20000 x 512` | Chunks | 4 |
| `27..32 x 16384 x 512` | Chunks | 4 |
| `32 x 24576/32768 x 512` | Chunks | 6 |
| `35 x 32768 x 512` | Single, chunk count 5 | 5 |
| `48 x 40960 x 512` | Single, chunk count 4 | 4 |
| `64 x 8192 x 512` | Single | 0 |
| `128 x 4096 x 256`, seq_len 100 | Coarse12, `length <= top_k` padding arm | n/a |
| `1024 x 4096 x 128` | Single | 0 |
| `1024 x 65536 x 2048` | Coarse12 | n/a |

Every row of that table was re-run against the final binary after the last
source edit, and every row verified. The chunk count column applies only to the
`Chunks` policy; it is `deep_gemm_topk_chunk_count(n_rows, n_cols,
multiProcessorCount)`, printed by the driver, i.e. the value the chunks launcher
actually dispatches on. Across the table it takes the values 0, 4, 5 and 6. The
full sweep of all 30 shapes with `n_rows = 3..32`, `n_cols = 16384`,
`top_k = 512` was also run end to end and verified, all 30 with chunk count 4.

## Shape coverage — what is accepted, what is refused, what is risky

**Refused outright** (the only checks in the driver, `main.cu:168-171`):

```
n_rows > 0 && n_cols > 0 && 0 < top_k <= n_cols
0 <= seq_len <= n_cols
```

Anything else is accepted and run. There is no shape-based policy gate, no
upper bound on `n_rows`, and no upper bound on `n_cols`. `seq_len` is per-row
uniform here (`main.cu` fills `seq_lens` with one value); the kernel itself
takes a per-row `seq_lens` array, the driver just does not expose per-row input.

**`top_k` has no enforced upper bound, but above 2048 it is out of contract.**
This is a property of the extracted kernel, not of the driver:
`kMaxTopK = 2048` sizes a `__shared__ int selected_indices[kMaxTopK]` in
`topk_single` (`fp32_topk.cu:935`) and feeds it `params.top_k`. With
`top_k > 2048` and `length > top_k`, that array is over-read on the device. The
public torch wrapper does not check it either — the only place `kMaxTopK`
appears is `static_assert(kCoarse12CandidateCapacity >= kMaxTopK)` at `:422`,
which is a compile-time capacity check, not a runtime guard. Empirically
`./main 128 8192 4096 3` still reports `verify: 128/128`, but that is luck, not
a guarantee: reading past a shared array is undefined, and what it reads is
whatever else the CTA put there. **Treat `top_k <= 2048` as the contract for
all three policies.** `topk_coarse12` and the chunks kernels do not build a
`kMaxTopK` array, but they share `kCandidateCapacity`-sized staging, so the
same bound is the safe one.

**Memory.** The driver only `cudaMalloc`s; there is no host-side mirror of the
score matrix during the normal run, so the device limit is the real limit.
Measured: `./main 4096 524288 2048 1` — 8.59 GB of scores, the shape asked
about — runs and verifies on this 64 GiB C500:

```
$ ./main 4096 524288 2048 1
shape n_rows=4096 n_cols=524288 top_k=2048 seq_len=524288 iters=1
chunks chunk_count=4 (APs=104) | workspace bytes (max over 3..6): 193331200
selected policy: Coarse12 (3)
time: 12.451 ms/iter over 1 iters
read bandwidth: 689.9 GB/s (8.590 GB read per call)
verify: 4096/4096 rows match CPU nth_element reference
real 0m20.1s
```

Per-shape device allocation is `n_rows * n_cols * 4` (scores) +
`n_rows * 4` (seq_lens) + `2 * n_rows * top_k * 4` (indices + values) +
`n_rows * (56 + 16 * chunk_count) + 8192 * chunk_count` bytes for the chunks
workspace. That last term is the reason a chunks shape is not free: 4096 rows
needs 193 MB, and it grows with `n_rows` regardless of `n_cols`.

**Host RAM.** The verify path streams the matrix back `128` rows at a time, and
`--dump-prefix` uses `256`-row blocks, so peak host usage is
`block_rows * n_cols * 4 + block_rows * top_k * 4` — about 268 MB for the 8.59 GB
shape above, not 8.59 GB. A shape is not refused for being larger than host RAM.

**What I have not tested:** any `n_cols` beyond 524288, and any `top_k` between
2049 and `n_cols` beyond the two spot checks noted above. Neither is refused;
both are outside the contract for the reasons given.

## Not covered

- The `Transform = true` kernels are not built, so page-table correctness is
  untested here. That lives in `deep_gemm/tests/test_indexer_topk_selector.py`
  (`test_selector_candidate_overflow`, `test_indexer_topk_selector_transform`)
  and `test_unit/test_indexer_topk_region_pack.py` against the installed
  package.
- `maca_policy` selection: the entry point always resolves `Auto`. The
  original takes an explicit policy from the master front end; adding a
  parameter for it would be new surface, not extraction.
- Ties: with ties at the top-k boundary the selected index set is
  implementation-defined. This repro compares value multisets, so it does not
  pin that down, and neither does upstream.
