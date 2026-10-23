# CHANGELOG

Notable changes, newest first.  This repository is the C500/C600U `topk`
operator; the performance work and its evidence live in
`docs/C500-radix-perf-ledger.zh.md`, and every number quoted here has a section
there.

## Unreleased

### One dispatch table in one TU: the wheel drops from 527 MB to 63 MB

`kRadixTable` moved from `contract.cuh` -- included by all 90 translation units
per family -- into `maca_topk.cu`, the one TU that dispatches.  Every table
entry takes `&launch_radix_config<Cfg>`, and a TU that merely *sees* the table
instantiates all of its rows in the **device** pass: `extern template` is
honored by the host pass and ignored there, so each of the 89 instantiation TUs
carried all 180 `topk_kernel_radix` kernels in its device bitcode even though
its host object held exactly one `__device_stub__`.

Measured on the pod (xcore1600, MACA 20261017.1436): one instantiation TU's
object **7,626,616 -> 680,720 B** (device bitcode 2,396,272 -> 198,608 B,
device ELF 5,213,480 -> 467,136 B) and the `topk_kernel_radix` symbols in that
bitcode **180 -> 1**.  Artifacts: xcore1000 **458,450,392 -> 67,524,144 B**;
xcore1500 and xcore1600 **590,263,768 -> 71,542,320 B**.  Wheel
**526,830,133 -> 62,998,174 B** (sha256
`ec4b180526c1bf81b1e968d252d8ecacb32b6ad7c3598ad57635de23b2369e61`).  The
three-family build drops from ~25 min to ~6 min at the same time, because the
TUs it compiles are smaller.

Not a kernel change: the artifact still defines 60 `topk_kernel_radix`
kernels, 89 `__device_stub__` and 2 `__tvm_ffi_*` exports, and the wheel passes
the delivery gates recorded in `docs/release-validation.zh.md` §3.

### The delivery gate is sized for a delivery, and its record names the binary

Three pieces of release machinery land together, and `docs/release-validation.zh.md`
is the record they produced: three family artifacts, the wheel's sha256, every
gate's command and result, and the one open engineering issue.

`tools/sample_gate.py` runs the official correctness table under a size cap
(`--max-elems`, 2^26 by default): the cases at or below it are the pool,
`--sample N` draws from that pool, the ones above are **counted, not dropped**,
and each drawn case's `p.seed` is pinned so the command re-runs the same data.
It drives `tests/test.py`'s own `correctness_cases_()` and `run_testcase` -- a
copy would be free to drift -- and it follows `run_bench.sh`'s path rule, so a
checkout with no built extension tests the installed wheel and its header prints
the package and artifact md5. Measured on C600U against the shipped wheel
`d24396d18f44`: **8000/8000, 0 check_fail, 0 crash, 0 skip, 0 unsupported**,
606 s (13.2 cases/s); the first taking, on the checkout's artifact, ran the same
draw in 753 s, and the size-fixed wheel re-ran it in 701 s (11.4 cases/s) --
three readings of one draw, spread by machine state, not by the kernel. The
full 105,138-case table remains the pre-release gate.

`scripts/wheel_smoke.py` is the delivered-artifact acceptance. Run from outside
the repository, it refuses if the answering package is the checkout, prints
`_binding.extension_path` and its md5 (the family-suffixed file the deployment
loads, `d24396d18f44` here), and checks the three production cells -- bf16
Lightning Indexer, fp32 Sampler, an fp32 ragged row -- against the contract,
with the fp32 ones elementwise against `backend="torch"`. 12/12 on the installed
wheel.

`tests/contract_test.py` is the targeted half of the gate (73 checks on
`maca_c`, 70 on `torch`): the `DS_HOST_CHECK` refusals, the `end` /
`output_idx_offset` / ordering / NaN surface, the coarse12 route's shape edges,
and the default backend's resolution -- none of which the official table
constructs.

**The wheel was 527 MB when this entry was written** -- P6 compiles 90 translation
units per family and each pays a ~6-7 MB device include tree (a TU including only
`contract.cuh` is 7.6 MB with zero device stubs; the pre-P6 single-TU object is
4.2 MB for 105 kernels). The diagnosis, the measurements that rule out the
toolchain and kernel duplication, and the candidate fixes are in
`docs/release-validation.zh.md` §5.1; the fix that landed is the entry above.

### The fp32 row's second pass stops reading what it will not use

`topk_coarse12_row` read every element a second time to act on the ~2.6% at or
above the threshold.  Pass 1 now leaves each 16-element block's max bin in a
per-row `uint16` table, the blocks at or above the threshold are compacted into
a per-row `uint32` list, and pass 2 strides the list with four lanes per block
-- 6 B per 16 elements, ~9% of the fp32 input.  The list is used only while it
holds at most three quarters of the blocks (past that the walk reads most of
the row anyway) and a call whose allocation failed keeps the dense walk.

Whole fp32 grid, two alternating rounds: **36,340.1 -> 33,304.0 us (-8.35%)**,
led by `b4096 V129280 k512` (the Sampler cell) 5070.4 -> 4395.3 us
(-13.3%, 418 -> 482 GB/s) and `b4096 V131072 k2048` 5176.8 -> 4784.0 us
(-7.6%, 415 -> 449 GB/s).  The one regression is the densest cell on the grid
(`V66551 k2048`, 45.5% of blocks kept) at +1.9%; the sign follows the density
and the gate cannot remove it, because falling back to the dense walk would pay
the summary *and* the walk.

The first version of this change passed all 30 fp32 perf cells and was still
wrong: a full-mask `__shfl_xor_sync` in the summary's tail iteration named
lanes that were not executing it, dropping one block per row on
`b512 V998123 k2048 sorted_value`.  The loop now splits at the iteration count
every lane runs -- the bulk shuffles on a constant mask, the tail on
`__activemask()` -- which is also 1.4% of the fp32 grid cheaper than guarding
every iteration.  Recorded in `docs/experiments/20261022_f32_pass2_skip/`.

The same row's thread width went 640 -> 512 in the same window: the driver
answers 3 CTA/SM at 640 (1920 threads against the 2048-thread/AP ceiling), and
512 x 4 fills the AP for **3.7-6.2%** on the five fp32 cells the change was
measured on (`docs/experiments/20261022_f32_occupancy/`).  A finer block grain
(G=4, one float4 per block) wins both probe halves and loses +3.0..+4.5% on the
Sampler route in the kernel, so G=16 stays ("The grain, revisited",
`docs/experiments/20261022_f32_pass2_skip/README.md`, `0c89f5d`).

### A NaN vote dies with the call that raised it

The per-row NaN flags used to live inside the process-wide, grow-only scratch,
at an offset derived from the call's geometry (`ChunkedWorkspace::nan_flags`,
`ChunkedF32Workspace::lengths`). A clean `b1 v262144 k512 bf16` call that
landed on the slot a `b6 v1048576` call had filled with a candidate index read
that index back as a vote and wrote the `0x3F3F3F3F` sentinel into a row with
no NaN. The per-call `cudaMemsetAsync` was hiding the aliasing, at the price of
one launch per call.

The flags are now the dispatcher's own `scan_flags` table: allocated beside the
scratch and zeroed once when it grows, never per call. The contract half
(`topk_kernel_radix`) reads the row's slot into shared memory and clears it in
the `tid == 0` prologue, before any return path, so every scanning route (row
path pass 1, both splits' stage 1, coarse12, chunks) can still OR a plain 1.
The split workspaces no longer carry flag regions, a failed split launch clears
the table before the row path takes over, and a mid-route failure in chunks
does the same. `check_nan=False` allocates nothing; `DS_FUSE_NAN_OFF=1` still
selects the pre-fusion scan kernel for A/B.

Measured on C600U: `run_test.sh --all --sample 1000 -rf` reports perf 147/147
and correctness 1131/1131 (18 OOM skips), `GATE_RC=0`. The symptom, the
read-and-clear design and the repro scripts are recorded in
`docs/experiments/20261020_scan_flag_read_clear/README.md`.

### cutlass leaves the build

`csrc/3rdparty/cutlass` was a submodule the MACA build never compiled:
`cutlass/kernel_launch.h` resolves to the toolkit's own `mctlass`, and the
vendored `csrc/3rdparty/kerutils` is the only third-party C++ tree `setup.py`
consumes. The gitlink and its `.gitmodules` entry are gone, so a checkout no
longer needs that clone; `README.md` says what the include resolves to instead.

### The kernel tree's directories tell the truth (P6)

The layout named two things it was not: `csrc/maca_kernels/xcore1000/` was the
tree **every** family compiled, and `csrc/maca_kernels/xcore1600/` was the
ported upstream tree **no** family compiled -- `setup.py` carried a comment
warning that family 1600 and the directory `xcore1600/` were two different
things, and that the name collision was the trap.  The collision is retired by
structure:

* `csrc/maca_kernels/common/` -- the shared implementation (`maca_topk.cu`, the
  contract/routing/radix/topk headers, the existing `primitives.cuh`), compiled
  into every family's artifact.
* `csrc/maca_kernels/xcore<N>/instantiations/` -- family `<N>`'s own generated
  tuple set (89 files per family; the three tables are identical today, and a
  family-specific set is a branch in the generator's `family_instantiations()`).
* `csrc/maca_kernels/legacy/` -- the ported upstream kernels, moved there
  unchanged: still source-only, still unbuilt, and (as before) not compilable
  without the `ARCH_SMEM_PER_AP_BYTES` substitution its two `static_assert`s
  need.

`setup.py`'s `SOURCES_BY_FAMILY` table is gone with the collision:
`sources_for_family(f)` is `COMMON_SOURCES + generated_sources(f)`, and the
generator's `xcore1000_sources()` is `family_sources(family)`.  Dated records
(experiment docs, perf ledgers, prior handovers) keep the paths they were
written against; this entry is the mapping.

### The wheel names the toolkit it was built against

The version string carried the git revision and a build timestamp but nothing
about the MACA toolkit -- which is the one compatibility fact a consumer cannot
check any other way.  The extension resolves `libmcruntime.so` through the rpath
baked in at link time, so a wheel built against one SDK generation fails on
another with `mcErrorInternalDeviceFunction` on **every** call: not a plausible
partial but a total failure, whose symptom reads like a kernel defect.

It is now `<base>+maca<MACA_VERSION>.<git-rev>.<timestamp>`, the same
`+maca<v>` spelling mcDeepGEMM's `setup.py` folds in, so a wheel name reads:

    deep_select-1.0.0+maca3.7.0.36.8674df2.20261008.165823-py3-none-linux_x86_64.whl

`MACA_VERSION` is mcDeepGEMM's own variable name.  `build.sh` and `install.sh`
read it from the toolkit's own `Version.txt` (first `Version:` line), a
caller-set value winning -- the precedence those scripts already give
`CUCC_TARGETS`.  A value that cannot appear in a PEP 440 local version is
dropped rather than handed on, and the version then records the absence, which
is also what a bare `python setup.py bdist_wheel` gets: `maca0.0.0.0`.

### MACA is a platform arm of its own

The tree declared its platform by *claiming another one*. `csrc/structs.h` had
no platform macro at all, and `csrc/3rdparty/kerutils/common/common.h` selected
kerutils' CUDA arm on MACA by aliasing `__MACA__` onto
`KERUTILS_IS_BUILD_ON_CUDA`. That aliasing is now gone: the platform is
declared, and kerutils has a MACA arm beside its CUDA and Ascend ones.

Three places, and only the first two are the same mechanism:

* `setup.py`'s `nvcc_args` passes `-DDEEP_SELECT_IS_BUILD_ON_MACA`, beside the
  per-Extension `-DDEEP_SELECT_ARCH`.
* `csrc/structs.h` refuses a build that names no platform
  (`#if !defined(...) → #error`), so the declaration is checked rather than
  conventional -- the shape upstream uses for its own two platforms.
* kerutils selects its arm from the toolchain, **not** from that flag, because
  a vendored library cannot require a build flag: `__CUDACC__` for CUDA,
  `__has_include("kernel_operator.h")` for Ascend, `__MACA__` for MACA. The
  three are mutually exclusive and each pair has an `#error`.

**`__MACA__` is tested before `__CUDACC__`, and that ordering is load-bearing.**
cucc passes `-imacros __macro_mxcc.h` to every translation unit, and that
adapter header defines `__NVCC__` and `__CUDACC__` so CUDA-dialect source
compiles. On MACA they are defined and false. Testing `__CUDACC__` first makes
both arms fire; the mutual-exclusion `#error` is what reported it, on the first
run of the probe below.

The arms themselves: `device/cuda/common.h` is `upstream/main`'s file
byte-for-byte again, and the MACA adaptations that used to live inside it
(`st_shared`, `trap`, `canonical_warp_idx_sync`, the native `<maca_bfloat16.h>`
include) are now `device/maca/common.h`. `device/device.cuh` and
`kerutils.cuh` name the arm; `host/host.h` shares its body between CUDA and
MACA and branches only inside `launch_kernel`, where the cluster / PDL /
cooperative arms do not exist on MACA and a request for one warns rather than
raising. `ceil_div` / `ceil` moved to `common/common.h` where upstream moved
them, keeping `__host__ __device__` -- their callers are device code, and
upstream's plain `inline constexpr` relies on `--expt-relaxed-constexpr`, which
mxcc does not guarantee.

**The MACA arm carries no cutlass**, which is the platform's constraint rather
than a preference: `/opt/maca/include/cutlass/` does not exist (the toolkit's
cutlass-derived library is `include/mctlass/`), so the `bf16` and
`transac_bar_t` aliases the CUDA arm re-exports from cutlass have nothing to
alias. Nothing here used them.

**Nothing in the build compiles the MACA device arm** -- its only consumer,
`csrc/maca_kernels/xcore1600/`, is off `SOURCES` and `include_dirs` -- so
`ref/kerutils_platform_arms/` is what compiles it, with flags read out of
`setup.py` rather than copied from it. It checks that the arm is the one
selected and the only one, that the constexpr pair folds in the host pass and
evaluates in a kernel, that the three device primitives round-trip 256/256
slots, and that `launch_kernel` launches both on a default config and on one
requesting cluster / PDL / cooperative. It does **not** claim
`csrc/maca_kernels/xcore1600/` compiles; that tree is still unbuilt, as
recorded under Known holes in `CLAUDE.md`.

### The kernel trees moved under a platform directory

`csrc/xcore1000/` → `csrc/maca_kernels/xcore1000/` and
`csrc/xcore1600/` → `csrc/maca_kernels/xcore1600/`, so the tree has the layer
upstream's `csrc/cuda_kernels/` and `csrc/ascend_kernels/` have: platform above,
variant below.  Anything outside the repository that names a source path — a
build wrapper, a doc, a script — needs the new one.

`setup.py`'s `SOURCES` and `scripts/generate_instantiations.py`'s two
instantiation-directory keys moved with it.  The kernel sources stopped naming
`csrc/ffi/` by relative depth (`"../ffi/ffi_error.h"` → `"ffi_error.h"`), which
is why the move did not have to renumber them and why the next one will not
either.

The measured-run records under `docs/` and the release notes below are kept as
written; each record doc carries a dated note saying where the tree went.

## v1.0.0 — 2026-09-19

First tagged release.  `perf_data/MetaX_C500/baseline` is pinned to
`perf_data/MetaX_C500/20260919_101516/`, which is the snapshot this tag's
README's numbers were read from.

### What the operator is

One `topk` entry point (`deep_select.topk`) with three backends — `maca_c` (this
repository's MACA kernels), `torch` (a reference built from torch ops), and
`deep_gemm` (that package's fp32 indexer selector, a soft dependency).  `maca_c`
serves bfloat16 and float32, `topk <= 4096`, `sorted_value` / `sorted_index`,
`end` windows, `output_idx_offset`, the out-of-band fills and the NaN contract;
`deep_gemm` serves a subset and says so with `UnsupportedByBackend` rather than
answering something narrower.

### Performance, as of this tag

`perf_data/MetaX_C500/20260919_101516`, device 2, 107 cells:

| against | result |
|---|---|
| `torch.topk` | **97/97 cells faster**, median **0.236x** (4.2x); slowest cell 0.64x |
| `deep_gemm` (the 12 fp32 cells it serves) | median **1.187x** by the kernel clock |

The kernel clock and the wall clock disagree on the `deep_gemm` column by design,
not by error — see "Two clocks" below.

### Routing (all measured, not inherited from `deep_gemm`'s policy)

| arm | band | ledger |
|---|---|---|
| fp32 chunked split | default fp32 path | §3–§7 |
| fp32 `coarse12` | `V <= 131072` → `batches >= 16`; `V > 131072` → `batches * topk >= 114688` | §12.13 |
| fp32 chunks | `batches <= 2` | §12.16, §12.17 |
| bf16 chunked | the Lightning Indexer grid | §4 |

`deep_gemm`'s own `select_topk_policy` is **not** used as the gate; the two
floors above were measured on C500 and the batch crossing moves with `V`.

### Two clocks (new at this tag)

`scripts/perf_snapshot.py` writes both, and they answer different questions:

* `time(us)` — `kk.bench` under `tests/test.py`'s `"topk"` matching rule.  Quiet
  (0.3–1.2% per cell), and the right clock for asking whether a change to *these*
  kernels helped.
* `wall_us` — one event pair around the whole call, L2 flush outside the span
  (`wall_clock_time`, the shape of deep_gemm's `bench_time`).  Every kernel the
  operator launches plus launch and sync; the only clock that compares two
  *backends* fairly.

They diverge by more than an order of magnitude on the `deep_gemm` column
because the matching rule excludes its NaN scan (a torch expression over the
whole row, ~5.4 ms on b256-v131072) and includes ours (folded into pass 1 of our
coarse12 kernel, ~15 us).  On that cell the kernel clock reads 240 us for
`deep_gemm` and the wall clock 1287 us.  `tools/compare_snapshots.py --clock
wall` compares on the second; the default is unchanged.

`csv_format_version` is 2.  A version-1 snapshot has no `wall_us` and compares
as "not measured" on that clock.

### Correctness

* Official gate, in the pinned snapshot: **120 pass / 0 check_fail / 0 skip**,
  plus **30 pass** on the fp32 grid.
* `tests/cases/chunks_arm_official.py` drives the suite's own `check_result`
  over the chunks arm's band and both decline paths.
* `tools/ab_snapshot.py` for paired A/B across arms; `tools/compare_snapshots.py`
  for snapshot-to-snapshot, with a `maca_c` regression as its exit status.

### C600U

**The C600U non-regression evidence is static only** — there is no C600U in this
container, so nothing here was measured on one.  `csrc/xcore1600/` is otherwise
untouched by the C500 work in this release: the one commit that names it since
the previous baseline (`96a5c7f`, 3 days ago) changed **comments only** in
`api.cu`, correcting a claim about where the tvm-ffi export lives — no code, no
kernel, no signature.  The C600U baseline
(`perf_data/MetaX_C600-U/baseline` → `20260915_081856`) is unchanged, and it
predates that comment fix.

`csrc/xcore1000/` and `csrc/xcore1600/` are separate architecture trees sharing
`csrc/structs.h` and the ffi layer; the arch-specific constants (`kMaxTopK` is
4096 in `maca_topk.cu`, 2048 in the two `dg_*.cuh` ports) are per-tree.

### Known limitations

* `deep_gemm`'s selector can pick an arbitrary subset of a row holding more than
  4096 values in one threshold bucket; `maca_c` does not have that limitation
  (`deep_select/interface.py`'s `topk_deep_gemm` docstring).
* The chunks arm's workspace sizing was wrong until `47f7194` — it dropped the
  batch factor.  Unreachable at the shipped band (`batches <= 2`); see §12.17.11.
* `compare_snapshots.py`'s ±3% tolerance was calibrated on cells above 100 us.
  At 3–34 us, `maca_c`'s own re-run spread reaches 3–5% and the tolerance reads
  as a regression when the binary has not changed (§12.18.6).
