# CHANGELOG

Notable changes, newest first.  This repository is the C500/C600U `topk`
operator; the performance work and its evidence live in
`docs/C500-radix-perf-ledger.zh.md`, and every number quoted here has a section
there.

## Unreleased

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
