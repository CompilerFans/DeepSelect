# Three-way DSA TopK — reference repros and what they do and do not show

Three independent implementations of the same operation — select the `k` largest
values per row of an fp32 score matrix and write their column indices — each
extracted into a torch-free, standalone repro:

| dir         | what it is | source |
|-------------|------------|--------|
| `deep_gemm/` | `deep_gemm.fp32_indexer_topk_selector`, the csrc's own routing | `mcDeepGEMM/csrc/kernels/fp32_topk.cu` |
| `ds/`        | DeepSelect's fp32 radix row kernel | `DeepSelect/csrc/xcore1000/maca_topk.cu` |
| `mcoplib/`   | mcoplib's `topk_transform_v1` (SGLang port) | `mcoplib/op/sglang/jit_kernels/topk_v1.cu` |

Each is a kernel TU + a public header + a `main.cu` driver + this README's
sibling scripts. Build and run with the toolchain in `/tmp/dsprobe/TOOLCHAIN.md`
and nothing else.

```
./build_all.sh          # compile all three drivers -> $DSREF_BUILD_DIR (default /tmp/dsref_build)
./run_all.sh            # build, then run the grid and print one table
./run_all.sh --dev N    # which device (default 0) — check mx-smi first
```

**Two more directories are here and are *not* in that table.** `build_all.sh`
builds every subdirectory with a `main.cu`, so they compile, but
`crosscheck.py`'s `IMPLS` is the three above and `run_all.sh` prints nothing
about them. They answer different questions:

| dir             | what it is | why it is separate |
|-----------------|------------|--------------------|
| `c500_gate/`     | the dispatch gate, printed | It prints `ARCH_SMEM_PER_AP_BYTES` beside the device's own `cudaDevAttrMaxSharedMemoryPerBlockOptin` and evaluates `ref_c500_gate` over a shape sweep. It measures nothing — it is the place a threshold change is read against the constant it was compiled with. |
| `coarse12_port/` | the *ported* coarse12 kernel, driven directly | `rk::dg12::launch_topk_coarse12` with no gate, no dispatch and no contract half, so the kernel itself can be timed against `deep_gemm/`'s original and its whole `top_k` range exercised. This is the only place the port runs without the facade around it. |

Both are referenced from `maca_topk.cu`'s gate comments; neither is part of the
three-way comparison, and a number from either is not comparable to the table
below (different shapes, different work).

---

## 1. The table

One grid, one traffic formula for every impl and every cell
(`n_rows * (len + k) * 4` bytes — scores read plus int32 indices written),
`--` for anything an impl cannot serve. From a real run on a MetaX C500
(104 APs), `./run_all.sh --dev 1 --iters 20`,
`/tmp/ref_table_final.txt`:

```
impl           bs     len     k |         ms      GB/s | agrees
-------------------------------------------------------------------
deep_gemm       6   16384  2048 |     0.0890       5.0 | ok
ds              6   16384  2048 |     0.0544       8.1 | ok
mcoplib         6   16384  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm       6   65536  2048 |     0.0660      24.6 | ok
ds              6   65536  2048 |     0.1205      13.5 | ok
mcoplib         6   65536  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm       6  524288  2048 |     0.3430      36.8 | ok
ds              6  524288  2048 |     0.4154      30.4 | ok
mcoplib         6  524288  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm     256   16384  2048 |     0.0660     286.0 | ok
ds            256   16384  2048 |     0.1318     143.2 | ok
mcoplib       256   16384  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm     256   65536  2048 |     0.1750     395.5 | ok
ds            256   65536  2048 |     0.4000     173.0 | ok
mcoplib       256   65536  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm     256  524288  2048 |     0.8230     654.9 | ok
ds            256  524288  2048 |     2.0668     260.8 | ok
mcoplib       256  524288  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm    4096   16384  2048 |     0.7780     388.2 | ok
ds           4096   16384  2048 |     1.2352     244.5 | ok
mcoplib      4096   16384  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm    4096   65536  2048 |     2.1960     504.2 | ok
ds           4096   65536  2048 |     3.5926     308.2 | ok
mcoplib      4096   65536  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm    4096  524288  2048 |    12.1890     707.5 | ok
ds           4096  524288  2048 |    29.5702     291.6 | ok
mcoplib      4096  524288  2048 |         --        -- | --       k outside contract [512, 512]
deep_gemm       6  524288   512 |     0.3280      38.4 | ok
ds              6  524288   512 |     0.1648      76.4 | ok
mcoplib         6  524288   512 |         --        -- | FAIL(self)  own verifier reports the output is wrong for this shape
deep_gemm     256  524288   512 |     0.7420     724.3 | ok
ds            256  524288   512 |     1.4872     361.3 | ok
mcoplib       256  524288   512 |         --        -- | FAIL(self)  own verifier reports the output is wrong for this shape
deep_gemm    4096  524288   512 |    10.9870     782.6 | ok
ds           4096  524288   512 |    21.5142     399.7 | ok
mcoplib      4096  524288   512 |         --        -- | FAIL(self)  own verifier reports the output is wrong for this shape
```

The `k=512` cells at `len=16384` are not in the grid, but they are the only
mcoplib cells that are both in contract and correct, so they are worth having
(`crosscheck.py --bs 6,256,4096 --len 16384 --k 512`, `/tmp/supp512.txt`):

```
deep_gemm       6   16384   512 |     0.0770       5.3 | ok
ds              6   16384   512 |     0.0509       8.0 | ok
mcoplib         6   16384   512 |     0.0488       8.3 | ok
deep_gemm     256   16384   512 |     0.0480     360.4 | ok
ds            256   16384   512 |     0.1138     152.0 | ok
mcoplib       256   16384   512 |     0.0869     199.1 | ok
deep_gemm    4096   16384   512 |     0.5460     507.0 | ok
ds           4096   16384   512 |     1.0770     257.0 | ok
mcoplib      4096   16384   512 |     0.7747     357.3 | ok
```

### What the columns mean, and what they do not

* **One traffic formula.** `n_rows*(len+k)*4` for every row. The drivers' own
  bandwidth lines are **ignored** — the three report three different quantities
  (deep_gemm counts writes, mcoplib counts only row prefixes, ds counts both)
  and none of them is the table's. GB/s here is comparable to GB/s from
  `tests/bench_dsa_topk.py`.
* **`ms` is each driver's own timer**, and the three are not cross-calibrated:
  deep_gemm uses `cudaEvent` over `iters` back-to-back launches, ds uses
  `steady_clock` best-of-repeats, mcoplib best-of-interleaved-rounds. Treat a
  row as a per-impl signal, not a cross-impl ranking. Use
  `tests/bench_dsa_topk.py` for the timing comparison — it measures all three
  libraries through one timer.
* **`agrees` is the driver's own check** against a CPU `std::nth_element` top-k.
  It is necessary and not sufficient; §3 is the check that is not the driver's
  own. A cell marked `FAIL(self)` gets no bandwidth number: the timing is real,
  but a GB/s on an answer the driver itself calls wrong invites a comparison
  that should not be made.

### Shape coverage, per impl

| impl | fp32 | `k` range | serves | refuses / wrong |
|------|------|-----------|--------|-----------------|
| deep_gemm | yes | 1..2048 (`kMaxTopK`; the driver only checks `k <= n_cols`) | every grid cell | nothing refuses — but see the `k > 2048` note below |
| ds | yes | 1..4096 (`kMaxTopK`) | every grid cell | on the **split** path only, a non-null `lengths` table with entries above `n_cols` walks past the row; the shortcut-plus-mask kept the one case tested in bounds |
| mcoplib | yes | **exactly 512** (compile-time) | `len <= 40000` only | `len >= 65536`: runs, but **wrong**; `k != 512` is refused outright (`top_k N outside contract [1, 4096]`, rc 2) |

**deep_gemm has no runtime check on `k`, and above 2048 it is out of contract.**
`kMaxTopK = 2048` sizes `__shared__ int selected_indices[kMaxTopK]`
(`fp32_topk.cu:935`), and that array is fed `params.top_k`; the public torch
wrapper does not guard it either (`kMaxTopK` appears only in a compile-time
`static_assert` at `:422`). A probe at `top_k = 4096` printed `128/128` rows
verified — that is luck, not a guarantee, since reading past a shared array is
UB. `crosscheck.py` now stops the deep_gemm column at 2048 for that reason. The
grid never exceeded 2048, so no reported number is affected.

**ds's per-row `lengths` table is exercised on the split path only.** A ragged
cell (B=8, V=65536, k=512, lengths `{21845, 65536, 256, 65536, 21845, 256,
21845, 65536}`) passes, including the two rows whose window is below `k` and take
the shortcut. The row path reads the same `end_ptr[row]`, one CTA per row, so the
mechanism is identical — but it is unmeasured, so it is not claimed.

One thing this ref cannot reach, named rather than implied: every driver cell
uses `randn` (deep_gemm a hash-normal, ds `mt19937(normal)`, mcoplib
`uniform_real`). `kF32OverflowChunkLen = 1757 * 58` exists precisely to keep a
random chunk's threshold bin inside the arena, so ds's
`radix_topk_row_f32_rescan` — the overflow path its own README calls "do not
remove" — is **compiled but not exercised by any cell in this table**. Reaching
it needs a deliberately concentrated distribution, which `randn` is not.

**Neither ds nor deep_gemm promises a sorted or stable index order.** ds's
contract says order is *unspecified and non-monotone* (measured: no row came
back descending on either dataflow; the ordered emit is gated on
`sorted_value`/`sorted_index`, both fixed false here), and `repro_check.py` below
measures the same for deep_gemm. Compare as multisets, never element-wise.

**The two mcoplib ops are not the same op, and the k they serve is opposite.**
The ref driver extracts `topk_transform_v1`, whose `TopK` is the compile-time
constant **512**. The op the library benchmark calls,
`torch.ops.sgl_kernel.fast_topk_transform_fused`, has `TopK = 2048` in
`csrc/elementwise/topk.cu:17` and *requires* a `[B, 2048]` destination. So:

* `bench_dsa_topk.py`'s `mcoplib` column is `--` at `k=512` and real at `k=2048`
  — the opposite of this table's mcoplib rows.
* Neither op can serve the other's `k`, so the two mcoplib columns are not two
  measurements of one thing.

**mcoplib timing carries a device sync per call.** The op's own entry does
`seq_lens.cpu()` (a `cudaMemcpy` DtoH, `topk_v1.cu:480`) before it can pick a
kernel, so every call is serialized against the host. At the 0.05 ms histogram
cells that is a large fraction of the measurement; at 0.78 ms it is noise. It is
the op's behaviour, not the driver's, and it is why a small-shape mcoplib time
should not be read as pure kernel time.

---

## 2. Where the numbers came from, and the two gaps

**Every grid cell was requested, not re-used.** `crosscheck.py` probes each
driver's `main.cu` for `int main(... argv)` before it will report a cell for it.
That check exists because `mcoplib/main.cu` spent most of this session running a
fixed case list with `int main()` — a table built from it would have shown a
grid while measuring eight hard-coded shapes. A driver that cannot be asked for
a shape gets `--` with that reason on the row.

**Gap 1 — mcoplib's radix path is wrong, and it is the library's, not the
extraction's.** The ref driver says so itself (`verify: FAIL ... selected index
below the kth value` for every `len >= 65536`), and it is reproducible from a
cold build:

```
$ ./build/main 6 65536 512 2
dispatch        : radix_256 (two pass)
verify: FAIL   row 0 (seq_len 65536): selected index below the kth value
```

It is **not** an artifact of the extraction. Driving the shipped library op over
the same shape on the same data (`/tmp/mcp_isolate.py`, uniform(-1,1), k=512):

```
topk_transform_v1  bs=6  len=16384 : ok
topk_transform_v1  bs=6  len=32768 : ok
topk_transform_v1  bs=6  len=40000 : ok
topk_transform_v1  bs=6  len=65536 : WRONG
topk_transform_v1  bs=6  len=262144: WRONG
topk_transform_v1  bs=6  len=524288: WRONG
```

The extraction's own self-test covers the radix path at `len` 16385 / 16387 /
32768 / 40000 and passes. The break is between 40000 and 65536, which is inside
the range the self-test does not reach: the staging buffer the radix path reads
its threshold-bin candidates from is sized for the row prefix the self-test
uses. Both the shipped op and the extraction fail identically, at the same
lengths, so this is upstream's limit and the ref reports it rather than hiding
it.

**Gap 2 — mcoplib's driver cannot be asked for a shape without an edit.** It
gained a `main(argc, argv)` at 02:20, partway through this work; every table
above uses it. Before that it was `--` with the raw reason on the row.

---

## 3. The checks that are not the driver's own

### 3a. `torch.topk`, in one process

The rule is `tests/bench_dsa_topk.py`'s `agrees()`: equal sorted index sets, or
the differing slots carry equal values, because a tied rank may break either
way. `oracle_check.py` applies it to each driver's *actual output*, loaded from
the dump the driver wrote (`--dump-prefix`), against `torch.topk` on the very
matrix that driver read. `torch.topk` shares no code with any of the three.

```
$ python3 oracle_check.py --dump /tmp/refdump --prefixes dg-6-65536-2048,ds-6-65536-2048 --shape 6 65536 2048
impl     scores input                                 vs torch.topk  note
-------------------------------------------------------------------------
dg-6-65536-2048   dg-6-65536-2048.scores.f32            ok     indices are a correct top-k of the matrix it read
ds-6-65536-2048   ds-6-65536-2048.b.v65536.k2048.scores.f32  ok  indices are a correct top-k of the matrix it read
```

Run over `(6,65536,2048)`, `(256,65536,2048)`, `(6,524288,2048)`,
`(256,524288,2048)`, `(256,524288,512)`: **all ok**. deep_gemm and ds agree
with `torch.topk` on their own data at every shape tested. mcoplib is absent
because its driver exposes no dump; its incorrect radix output is established
by §2 instead.

Note the division of labour between this and §3b, because they use different
rules for a reason. *This* section compares one driver against `torch.topk` on
the **same** matrix, so the slot-wise `agrees()` shorthand is available and
correct. §3b compares a driver against *itself* across two runs where the only
difference is slot order, and there the slot-wise form is unsound — see §3b.

**What this cannot do** is compare two drivers to each other element-wise. Each
generates its own matrix (deep_gemm from a hash, ds from `mt19937(1234)`,
mcoplib from `mt19937` over uniform), and no driver has a "read this file and
select from it" mode — `--dump-prefix P` *writes* P and selects from the matrix
it just regenerated, so copying a different matrix onto P changes nothing about
the selection. A driver-vs-driver comparison built that way compares two
different matrices. `tests/bench_dsa_topk.py` is where a cross-backend
comparison belongs, and is where it is done.

One contract detail that falls out of the dumps and matters to any caller:
ds forwards a non-null `lengths` table straight through as `end_ptr` **and still
uses the full row stride**, so a table whose entries exceed `n_cols` makes the
selector walk past the row it was given. The single case tested (table = 32×V,
V=16384) stayed in bounds because the `length <= topk` shortcut plus the final
mask contain the writes, but that is containment, not a guarantee — the ref does
not validate the table, and a caller should not pass one it has not bounded.

### 3b. Run-to-run reproducibility

A driver is only usable as a reference if re-running it reproduces itself.
`repro_check.py` runs each twice at one shape:

| shape | impl | scores byte-identical | indices identical | multiset-equal |
|-------|------|----------------------|-------------------|--------------|
| 6×16384 k=2048 | deep_gemm | yes | **no** | *the same set, differently ordered* |
| 6×16384 k=2048 | ds | yes | **no** | *the same set, differently ordered* |
| 256×524288 k=2048 | deep_gemm | yes | **no** | *the same set, differently ordered* |
| 256×524288 k=2048 | ds | yes | **no** | *the same set, differently ordered* |

Both drivers regenerate byte-identical scores, then emit index files that differ
in the overwhelming majority of slots (12085 of 12288 at 6×16384; 501697 of
524288 at 256×524288 k=512). The sorted index **sets** are identical and the
selected **value multisets** equal `torch.topk`'s on the same matrix, so this is
order, not content — the kernels do not reproduce a stable *order*. Any caller
that compares indices element-wise, or that assumes a descending output, is
comparing something the reference does not promise. `bench_dsa_topk.py` already
treats index order as don't-care.

**The cause, for both: the emit order follows atomic scheduling, not the
values.** ds writes through `atomicAdd(&s_counter, 1u)` and its ordered emit is
gated on `sorted_value` / `sorted_index`, which `ds_topk` fixes false;
deep_gemm's radix passes append their boundary bin in arrival order. This is why
the order is both unstable run-to-run *and* non-monotone.

**Use the multiset rule, and sort both sides.** An earlier draft used
`bench_dsa_topk.py`'s `agrees()` shape — "equal sorted index sets, or the
differing slots carry equal values" — as a slot-wise test here, and it is
**unsound on these dumps**: at 16384 distinct values in a 16384-long row, 8466
re-run slots differing is not tie noise, those are distinct elements. The rule
that holds is per-row value multiset,
`sort(selected) == sort(topk_reference)`. The reference itself comes back
descending while the kernel's selection is unordered, so **both** sides must go
through the sort — `sort(sel) == sort(row_desc[:k])` reads as a failure on every
row. `repro_check.py` does the sort on both sides.

This is a harness-construction note, not a defect: the drivers' own verifiers are
count-based and unaffected, and no table cell changes.

---

## 4. The library-level runs (`tests/bench_dsa_topk.py`)

The three *libraries*, through one timer (`kernelkit.bench`), on one machine:
`CUDA_VISIBLE_DEVICES=1`, `PYTHONPATH` including both trees.

`--quick --iters 30`:

```
    bs     len     k |     ds ms  ds GB/s |        dg ms     dg GB/s     dg x
     6   16384  2048 |    0.1022      4.3 |       0.0814         5.4    1.255
     6   65536  2048 |    0.1830      8.9 |       0.1328        12.2    1.378
   256   16384  2048 |    0.2311     81.7 |       0.0925       204.0    2.498
   256   65536  2048 |    0.3538     195.6 |       0.2405       287.7    1.471
  4096   16384  2048 |    3.3819     89.3 |       0.8600       351.2    3.933
  4096   65536  2048 |    3.6309     305.0 |       1.7391       636.7    2.088
```

`--bs 256,4096 --len 16384,65536,524288 --topk 2048,512 --iters 30`:

```
    bs     len     k |     ds ms  ds GB/s |        dg ms     dg GB/s     dg x |   mcoplib ms mcoplib GB/s mcoplib x
   256   16384  2048 |    0.1315    143.5 |       0.0477       395.5    2.756 |       0.1214       155.4    1.083
   256   65536  2048 |    0.3086    224.2 |       0.1428       484.8    2.162 |       0.2680       258.2    1.152
   256  524288  2048 |    1.5645    344.5 |       0.7922       680.3    1.975 |       1.3707       393.2    1.141
  4096   16384  2048 |    1.2659    238.6 |       0.5338       565.8    2.372 |       1.5065       200.5    0.840
  4096   65536  2048 |    3.6525    303.2 |       1.6585       667.6    2.202 |       3.3844       327.2    1.079
  4096  524288  2048 |   22.6043    381.5 |      11.3879       757.2    1.985 |      18.3038       471.1    1.235
   256  524288   512 |    0.9982    538.4 |       0.7479       718.5    1.335 |           --          --       --
  4096  524288   512 |   14.4548    594.8 |      10.7777       797.8    1.341 |           --          --       --
```

(`k=512` at `len` 16384/65536 is in the run and omitted here for width; `ds`
never lost a cell. Quote from `/tmp/bench_grid2.txt`.)

No cell reported `!wrong` against `torch.topk` in either run — the library
implementations are all correct on this harness's `randn` data. That is the
distinction that matters: §2's mcoplib failure is specific to
`topk_transform_v1` on a long row, and the op this benchmark calls
(`fast_topk_transform_fused`, `TopK=2048`) is a different one.

**The `mcoplib` column is `--` at `k=512` for a hard reason**, found while
chasing this: the op the benchmark calls requires a `[B, 2048]` destination and
raises `Expected dst_page_table.size(1) == TopK` for anything else. It is
`TopK=2048` (`csrc/elementwise/topk.cu:17`), the ref extraction is `TopK=512`,
and neither can serve the other's `k`. The `--` there is a contract mismatch,
not a missing measurement.

`bs = 6` cells are absent from this run because the split-k branch is a **hard
error** on the library side, not a harness omission:

```
[MCR][E] mc_platform.cpp:938: Shared memory size error, Kernel name
  _ZN12_GLOBAL__N_128topk_transform_splitk_kernelE..., shared memory size in
  kernel 1044 from user 65536 in device 65536
RuntimeError: topk kernel failed: invalid argument
```

**`B <= 26` is the gate, not the trigger — the trigger is the row *stride*.**
`fast_topk_transform_fused` takes the branch when `B * 4 <= sm_count` **and**
`max_len > TopK` (`csrc/elementwise/topk.cu:712`), where `max_len` is
`score.size(1)` — the enclosing tensor's width, not the per-row `lengths`.
`topk_decode.cu:1154-1163` then sets `G = max(sm_count / B, 8)` and
`chunk = ceil(input_stride / G)`, capped at `kSplitKMaxChunk = 16384`
(`topk_decode.cu:760`), and launches with `chunk * 4` bytes of dynamic smem.

`kSplitKMaxChunk * 4` is **65536 B**, and a C500 AP has **64 KB of shared memory
in total** — the cap leaves no headroom, so `chunk == 16384` cannot launch at
all. `chunk` reaches the cap exactly when the stride is wide enough for `G` to
stop covering it:

| `B` | `G = max(104/B, 8)` | first failing `input_stride` (elements) | `B*4` |
|---|---|---|---|
| 1 | 104 | 1703833 | 4 |
| 2 | 52 | 851917 | 8 |
| 4 | 26 | 425959 | 16 |
| 6 | 17 | 278512 | 24 |
| 8 | 13 | 212980 | 32 |
| 13…26 | 8 (floor) | 131065 | 52…104 |
| ≥ 27 | — | never (gate fails) | ≥ 108 |

So the failing set is **`min(B, stride)` above ~26**, not `B <= 26`, and it is
independent of the row length.

Two refinements, both from re-deriving the arithmetic against
`topk_decode.cu:1145-1175` rather than reading the table:

* **`chunk == 16384` is a sufficient condition, not the necessary one.** The
  actual test is `chunk * 4 + 1044 > 65536`, i.e. `chunk > 16123`
  (`1044` = the kernel's static smem — scalars plus `warp_sum[16]` — from the
  driver's own error message). So the true first-failing stride is **~2% below**
  the table's column:

  | `B` | table (first `chunk == 16384`) | true first failure (`chunk > 16123`) |
  |---|---|---|
  | 1 | 1703833 | 1676793 |
  | 4 | 425959 | 419199 |
  | 6 | 278512 | 274092 |
  | 8 | 212980 | 209600 |
  | 13…26 | 131065 | 128985 |

  Same `B`-dependence, ~2% lower. Stride 524288 is over both for every `B` in
  the failing set, so **the measured table and every conclusion from it are
  unaffected** — only the boundary moves.
* **`kSplitKMaxChunk` itself is not the defect.** It is 16384 *because* the AP
  has 64 KB; the constant and the hardware agree. What has no headroom is the
  pair (`kSplitKMaxChunk * 4` dynamic **+** 1044 static) *versus* 65536 total:

  ```
  topk_decode.cu:1173
    setup_kernel_smem_once<topk_transform_splitk_kernel, kSplitKMaxChunk * sizeof(uint32_t)>();
  ```
  asks for exactly 65536 with no allowance for the static allocation, so the
  pre-flight check and the launch disagree by 1044 bytes and the launch loses.
  Fixing it means sizing the arena from the AP's actual budget, not lowering the
  constant. "`kSplitKMaxChunk` leaves no headroom" and "`kSplitKMaxChunk` is too
  large" are different claims and only the first is true.

**Measured** (`CUDA_VISIBLE_DEVICES=2`, `(bs, stride)` tensors, `lengths` varied
independently):

| stride | `bs` = 4 | 8 | 26 | 27 | 32 | 128 |
|---|---|---|---|---|---|---|
| 2048 | — | OK | OK | OK | OK | OK |
| 524288 | abort | abort | abort | OK | OK | — |

at `length` ∈ {2048, 16384, 65536} — identical verdicts for all three, so the
row length is not a factor. The earlier framing ("`bs <= 26` errors, `bs >= 27`
is fine") came from probing with *tightly sized* tensors, where the stride is
the row length and `chunk` never reaches the cap; it is the stride that decides.

**Consequence for `bench_dsa_topk.py` specifically:** its `bs = 6` cells are
unreachable for `mcoplib` because `SEQ = 524288` puts every call over the
stride threshold — not because `bs = 6` is special. `GRID_BS = [6, 32, 128, …]`
(`tests/bench_dsa_topk.py:64`) straddles the `B` half of the gate deliberately,
but the stride half is what the rows share, so `bs = 6` is the only unreachable
one. It is not a harness bug and not fixable by `bs`: making it measurable needs
a narrower allocation, which the grid's other cells are built around.
`bench_dsa_topk.py` catches the abort per cell and prints `--`, so the failure
is contained, but a reader who takes the `--` for "shape not supported" rather
than "the kernel aborted" will draw the wrong conclusion.

### The chunk-count change did not break the library

The change under test: `csrc/xcore1000/maca_topk.cu`, `f32_chunks_large_batch`
went from the constant `2` to
`ceil(vocab_size / 101906)` clamped at 2, gated on `sm_count == 104`. The gate
artifact is `deep_select/deep_select_maca.so`
`md5 3114a5fd50eaaa90e713621b51f98e94`, built 01:49 from
`csrc/xcore1000/maca_topk.cu` at 00:48 (extension newer than source, so not
stale).  (The artifact name is `deep_select_maca_xcore1000.so` since
2026-09-18, when the build became one `.so` per family; this receipt predates
that and its md5 names the file as it was.)

The change is visible where the rule fires, and the cells the rule decides are
exactly `bs > 64` (`kF32ChunksFewBatches`) and `vocab > 262144`:

* `4096×524288 k=2048` → `4096×524288 k=512` above: 22.60 / 14.45 ms.
  Under the old constant-2 rule this was 90.1 ms at k=2048 (the number the
  change's own comment records), so the split is doing its job.
* `256×524288 k=2048` → `256×524288 k=512`: 1.5645 / 0.9982 ms.
* `len = 16384` and `65536` are below the rule's threshold and are unchanged,
  which is the intended scope.

### The official correctness gate

```
./run_test.sh --test --backend maca_c --sample 200 -rf --results /tmp/ref_gate
```

from `/home/compiler_gfx/tilelang/DeepSelect` on `CUDA_VISIBLE_DEVICES=0`:

```
================================================================
  pass             312
  check_fail         0
  crash              0
  skip              10
  unsupported        0
All 312 cases passed!
```

**0 case(s) selected wrong, 0 crashed**, 10 skipped as out-of-memory (each
needs 12–32 GiB on a 63.6 GiB device and is expected — they are OOM skips, not
failures). Receipt: `/tmp/ref_gate/deepselect_run_20260917_012107.txt`.

The sample does exercise the changed rule rather than merely running beside it:
of the sampled fp32 cases with `bs > 64` and `vocab > 262144`, **21 ran and 5
were OOM-skipped**. So the `f32_chunks_large_batch` path is covered by the gate,
and passes.

---

## 5. Blockers hit, and their state

| # | what | state |
|---|------|-------|
| 1 | `deep_gemm/main.cu` did not compile in the intermediate version I first pulled: `row` redeclared against the existing `int64_t row` parameter, and `verify_block` called but never defined | **fixed before my first successful build** by the impl's owner. A later re-check of the file found the two problems gone and the file md5 changed; the owner is confident the broken revision was never the one on disk at 01:26. I cannot resolve which revision I read, so this is recorded as *a revision that existed and was fixed*, not as a claim about the current file. The current file builds clean with the plain toolchain line. |
| 2 | `mcoplib/main.cu` did not compile: `__float2half_rn` with no `#include <cuda_fp16.h>`. `build_all.sh` falls back to adding `-I$MACA_PATH/include` | **fixed** by the owner — builds on the plain line now |
| 3 | `mcoplib/xcore1000_topk_v1.cu` segfaulted on every call: the extraction dropped the host copy the original does at `topk_v1.cu:478-483`, so `max_seq_len_host` was computed by walking the *device* pointer | **fixed** by the owner |
| 4 | `mcoplib/main.cu` had no shape CLI (`int main()`), so no grid cell could be requested from it | **fixed** by the owner; every table above uses the CLI |
| 5 | `mcoplib`'s radix path returns wrong answers for `len > 65536` | **not a blocker** — reproduced against the shipped library (§2), reported as the ref driver reports it (`FAIL(self)`, no GB/s). The owner's follow-up: it is a *second*, independent defect on top of the staging cap (`atomicAdd`-ordered scatter past a saturated 4096-entry staging buffer), and the crossing is draw-dependent rather than a fixed cutoff — the driver now prints the threshold-band occupancy per radix case so a cell can be read against the cap |
| 6 | `deep_gemm` requires `--dump-prefix` *before* the positional args; `ds` accepts it either side | worked around in the harnesses; noted here because it is a real footgun |
| 7 | Neither driver can be fed a matrix to select from, so no driver-vs-driver comparison is possible | documented in §3a; the library-level comparison is `tests/bench_dsa_topk.py` |
| 8 | deep_gemm and ds do not reproduce their own index *order* run to run | documented in §3b; value multiset is stable, order is not |
| 9 | `bs <= 26` errors outright in `fast_topk_transform_fused` (split-k launch, smem mismatch) on this device | library-side, not mine to fix; named in §4 so the `--` is not misread. **The condition is `min(B, stride) > ~26`, not `B <= 26`** — see §4 |
| 10 | I reported a whole-row index hazard for ds (indices past a 16384-row) on the strength of residual output from a sweep whose input file I had already overwritten | **retracted.** The impl's owner could not reproduce it on six shapes (row and split paths, V up to 524288) with zero out-of-row indices, and showed the hp kernel has no tile/segment base to get wrong — every index is `idx + chunk_begin`. My repro was contaminated and I should not have carried the claim forward from it. |
| 11 | I attributed a "ties broken by whatever the radix passes happen to produce… in descending order" sentence to the ds header | **my error** — that sentence is in `ref/deep_gemm/xcore1000_fp32_topk.h`, not `ref/ds/`. The underlying observation was right (ds's order is not descending) and the owner has since made ds's contract explicit: order is unspecified and non-monotone, compare as multisets. |
