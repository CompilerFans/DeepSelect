# DeepSelect

DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) (which is used in DeepSeek V3.2, DeepSeek V4, and DeepSeek V4.1 models) and the sampler. It achieves 2 ~ 20x speedup compared to vanilla `torch.topk`.

## News

- 2026.09.10: We've released a brief analysis of the algorithm and its implementation: [English](docs/DeepSelect-deep-dive.md) | [中文](docs/DeepSelect-deep-dive.zh.md)
- 2026.09.10: We've released DeepSelect v1.0.0

## MACA support

This tree also carries a MACA (MetaX) port of the operator, used by `mcDeepGEMM`.
The kernel tree is split by the shared memory a part has per SM, because that is
what a top-K kernel's staging buffers are sized against:

| tree | parts | kernel |
| --- | --- | --- |
| `csrc/xcore1000/` | C500 (64 KiB per SM) | `maca_topk.cu`, written for MACA |
| `csrc/xcore1600/` | C600, C600U (128 KiB per SM) | the upstream kernels, ported |

Which one a device runs is a property of the device, not a choice — but there
is only one answer today: `setup.py` builds a single extension carrying an image
per architecture it was asked for, and **every family runs
`csrc/xcore1000/maca_topk.cu`**. (See the note under `csrc/xcore1600/` below.)

`csrc/xcore1000/maca_topk.cu` reimplements the operator -- the same public
contract, the same `deep_select.interface.topk` signature -- with portable
primitives only (shuffle, `atomicAdd`, `__syncthreads`, `__syncthreads_or`), as
a radix refine over an order-preserving key of each value.

`csrc/xcore1600/` keeps upstream's algorithm (a threshold-and-compact scan in a
random block order, one global read per element) and replaces its device-side
dependencies: TMA tensor-map loads become cooperative `ldg`, mbarriers a single
buffer with `__syncthreads`, inline PTX MACA builtins. Its config tuples are
re-derived for 128 KiB, since upstream's are sized for an H100's 227 KiB.
`v3_cluster` was deleted rather than ported: MACA has no cluster launch.

> **Nothing builds this tree, on any part.** The port selects wrong on a MACA
> C600U -- an `arange` row of 0..511 with `topk=8` returns indices like
> `[448..455]` where the answer is `[511..504]`, and differently on every run;
> the official slice scored 4/200. So every family builds
> `csrc/xcore1000/maca_topk.cu` instead, which passes 200/200 on a C600U **and
> is 1.5-2.9x faster there** (CLAUDE.md, "Can a C600U run the C500 kernel").
> The port stays in the tree as the reserved implementation, but it is no longer
> built or buildable: reaching it now takes a source change to `setup.py`'s
> `SOURCES` *and* its `include_dirs` (its `kerutils` include was dropped from the
> build with it), not an environment variable, and there is no switch for it
> anywhere in `setup.py`. Everything below in this section describes `csrc/xcore1600/` as
> it stands, port bugs included.

Consequences:

- `topk` above 1024 is served by `maca_topk.cu` only. The ported kernel's tuples
  cover `max_topk` 512 and 1024 -- a 4096 tuple cannot fit 128 KiB, since its
  survivor-pairs and extra-pairs regions alone come to exactly 128 KiB -- so a
  C600 / C600U rejects `topk` in `(1024, 4096]` *when the port is what serves
  it*. With the default routing above, a C600U serves those shapes from
  `maca_topk.cu` like every other part.
- The `vocab_size < 2^23` restriction is enforced by the ported kernel, as
  upstream; it comes from the fp32-simulated census there. It is not enforced by
  `maca_topk.cu`, which ranks integer keys (verified for `float32` at
  `vocab_size = 2^23`, the largest vocab in the official table, which is also
  `(1 << 23) - 1`).
- `sorted_value` is accepted for `torch.bfloat16` by `maca_topk.cu`: the ordering
  comes from the same key, so both the descending order and the value/index
  pairing hold. The ported kernel rejects it, as upstream does (fp32 only), and
  so does `backend="torch"`.
- The `v3_cluster` dispatch arm (`bfloat16`, `batch_size <= 6`,
  `vocab_size >= 512K`, `topk <= 1024`) was deleted with the variant; those
  shapes are served by the general kernel (a cluster-less schedule), not
  rejected.
- `begin` and `hint` are rejected, as upstream.
- Everything else in this document -- the two scenarios, `sorted_index` /
  `return_value` / `end` / `output_idx` / `output_idx_offset` /
  `idx_oob_fill_value` / `value_oob_fill_value`, and the NaN contract below --
  behaves identically on MACA.
- MACA performance is measured by upstream's own grid --
  `PYTHONPATH=. python tests/test.py --perf-only` -- which benches each shape
  with the kernelkit kineto harness and times `torch.topk` beside it. The
  numbers in [Performance](#performance) are still the upstream CUDA kernels'.

## Design philosophy: one operator, four axes, no universal kernel

TopK is the rare operator where **no single kernel is optimal over the shape
space**, not for want of engineering but because two of its optima are opposed.
DeepSelect answers that with specialization; this section is what the
specialization is *for*, and where its limits are.

### The shape space has two regimes, and they want opposite structures

Measured on C500, one row per CTA, fp32, 262144 floats per row
(`docs/C500-to-parity-plan.zh.md` §16; `mt_gap_low.cu`):

| grid (CTAs) | 1 | 6 | 104 | 208 | 312 | 4096 |
| --- | --- | --- | --- | --- | --- | --- |
| GB/s | 12.9 | 64.9 | 851.5 | 1413.6 | 1576.0 | 1647.5 |
| % of the 1650 GB/s read wall | 0.8% | 3.9% | **51.6%** | 85.7% | **95.5%** | 99.8% |

The curve is linear in the CTA count until ~250 and only then bends. **Launch
overhead is not what it shows**: an empty kernel with the same geometry measures
6.4 µs at grid 1 and 5.3 µs at grid 104. At grid 6 the kernel runs 96.9 µs, so
94% of it is the work itself, done on a machine where 6 of 104 APs are busy.

- **Small batch, long rows** (`b6` here) is a *parallelism* problem. No amount of
  tuning a one-row-per-CTA kernel fixes it: the row must be **split** across
  CTAs and merged, which costs a merge kernel and a workspace.
- **Large batch, any rows** is a *per-byte efficiency* problem. Here splitting is
  a loss: the merge CTAs are pure overhead once the grid is already several
  waves deep. Measured at `b256`, opening the fp32 split's floor cost **+13.7%**
  on `b4096-v16384`.

The same inversion appears in the algorithm itself: `deep_gemm`'s single-pass
threshold-and-compact selector reads every element once, and it **wins** at
`vocab_size = 1024` (0.73×, 0.90×, 0.90× — its fixed cost has not amortized yet)
while **losing everywhere long** (up to 10.10× against a two-pass radix that
reads the row twice). One pass is not better; it is better *at some shapes*.

### The four axes, and what each one costs

The specialization budget is finite -- the MACA-C backend JIT-compiles one
kernel per distinct generated source (~2 s each, disk-cached), so a routing
change must re-point between **registered** entries and never add a
shape-keyed instantiation axis. Within that budget:

| axis | why it splits | what it costs |
| --- | --- | --- |
| **Radix rounds** | A 16-bit key resolves in two levels (12 coarse bits + 4 fine = the whole key, so refine never ranks past a fine tie). A 32-bit key cannot: 8 bits leaves 24, and an honest 12-bit coarse layer forces a re-scan of the full row on overflow. | More rounds = more passes over the row; fewer = a wider threshold bin and a bigger candidate arena. The coarse level for fp32 is derived by rounding to fp16 first, so only **256 of 4,096 bins are reachable** -- a precision cost paid at the key, not at the scan. |
| **Coarse screening** | The threshold bin's width decides both the arena's overflow rate and how many candidates refine touches. | Widening the coarse level to cut refinement is *not* free: measured, skipping the whole refine block moved `b4096-v262144` by less than that measurement's own ±0.3% noise band (§13). The refinement is already not the bottleneck. |
| **Splitting** | The only lever that raises parallelism, and the only one this project has measured to move the small-batch regime. | `chunks` merge CTAs on top of the row, and ~3 µs of fixed cost per CTA, so short chunks eat the gain back. Hence a *tiered* policy keyed on batch, not a constant. |
| **Shared-memory staging** | Staging buffers are sized against the per-SM capacity, and a config that does not fit is rejected **at compile time**. | This is why parts are separate builds, and why a 227 KiB H100-sized tuple cannot ride into a 64 KiB part and fail at launch. |

### The balance is per-part, and 128 KiB parts are not solved by scaling

The `NATIVE_*` constants in `csrc/structs.h` are one row per family on purpose.
C500's `NATIVE_F32_CHUNK_WORK_TARGET = 260` is **2.5 × its 104 APs, fitted over
24 measured points**; the C600 (70) and C600U (80) rows are **scaled
reservations, not measurements**, and the doc comment says so. The ratio
`K / SM_COUNT` is precisely the thing the C500 data cannot tell us -- and the
whole chunk policy is that one ratio.

What changes on a 128 KiB part, and in which direction:

- **More staging room per SM** means wider coarse layers and bigger arenas are
  affordable, which shifts the radix-rounds axis toward *fewer rounds*.
- **Fewer APs** (32 vs 104) means the parallelism-starved regime starts at a
  *smaller* batch: `b256` is 2.46 waves on C500 but **8 waves on C600U**, so
  shapes that need splitting on C500 may not need it there -- while the merge
  cost, being per-row, is *relatively larger*.
- The two push the split threshold in opposite directions, which is why the port
  is gated on measurement rather than on the ratio.

**A 128 KiB device therefore owes its own measurement of the same curve**, not a
rescaling of the C500 one. The unvalidated port in `csrc/xcore1600/` is the
other half of that story -- see [MACA support](#maca-support) and CLAUDE.md's
"Known holes".

## Supported Cases

TopK workloads vary widely, and the fastest algorithm & implementation highly depends on the input dtype, `batch_size`, `vocab_size`, and `topk`. This repository only focuses on the following cases:

### Lightning Indexer Scenario

This scenario covers:
- Input dtype: `torch.bfloat16`
- `batch_size`: $1 \sim +\infty$ (both large and small batch sizes are optimized)
- `vocab_size`: $1 \sim +\infty$ (both large and small vocabularies are optimized)
- `topk`: small (must be $\le 4096$; larger values are not supported)

Recommendations:
- Disable `sorted_index` unless the output has to be ordered by index or by value; enabling either one costs performance.
- Set `return_value=False` when the values are not needed. This skips the value output and is faster.

### Sampling Scenario

This scenario covers:
- Input dtype: `torch.float32`
- `batch_size`: $1 \sim +\infty$
- `vocab_size`: around 128K
- `topk`: small (must be $\le 4096$; larger values are not supported)

## Performance

Measured with the benchmark in [`tests/test.py`](tests/test.py)
(`python3 tests/test.py --perf-only`), which reports the ratio against `torch.topk`
on the same input. The metric is effective memory bandwidth: TopK does no
floating-point math, so a FLOP rate would not be meaningful here.

> These figures belong to the upstream CUDA kernels. For the MACA port, run
> `tests/test.py --perf-only` on the target device -- what it measures and how
> the two implementations compare there is in
> [MACA support](#maca-support).

### Lightning Indexer Scenario

bfloat16, `topk = 512`, one subplot per batch size, on a shared 0 - 7 TB/s axis.

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer](assets/perf_bf16.png)

### Sampling Scenario

float32, `vocab_size = 129280`, `topk = 512`.

![DeepSelect vs torch.topk, float32 Sampling](assets/perf_fp32.png)

## Installation

```bash
git clone https://github.com/deepseek-ai/DeepSelect.git
cd DeepSelect
git submodule update --init --recursive
pip install -v .
```

### MACA (MetaX)

The build needs the MACA toolkit (`$MACA_PATH`, default `/opt/maca`) and a
MACA-compatible PyTorch. The `csrc/3rdparty/cutlass` submodule is not referenced
-- `cutlass/kernel_launch.h` resolves to MACA's own `mctlass` -- while the
vendored `csrc/3rdparty/kerutils` is. Each architecture is built as its own
extension, since a config's staging buffers are sized against the shared memory
of the architecture it was compiled for:

```bash
CUCC_TARGETS=xcore1000 python setup.py build_ext --inplace             # C500
CUCC_TARGETS=xcore1600 python setup.py build_ext --inplace             # C600, C600U
CUCC_TARGETS=xcore1000,xcore1600 python setup.py build_ext --inplace   # both
PYTHONPATH=. python tests/test.py --backend maca_c --sample 200        # correctness
```

The scripts wrap those calls, divided by **what they produce**:
`develop.sh` builds **in place** and installs nothing; `build.sh` builds a wheel
for every family; `install.sh` builds the same wheel narrowed to this device and
`pip install`s it. `clean.sh` removes the build artifacts, and `run_test.sh` /
`run_bench.sh` run the suites and record what they measured.

**The runners do not require the checkout's extension.** They ask the package
that will answer -- `_binding.extension_path`, the same lookup `load()` does --
and put the repo on the path only when it has a built extension. A tree that has
been `./develop.sh`ed measures the tree (every existing receipt keeps naming the
same file); a tree that has only been `./install.sh`ed measures the installed
wheel, with no build step. `tests/` stays on the path either way, since the grid
lives in the repo and is not in the wheel.

**The wheel goes to `dist/`, and to `${BUILDROOT}/wheel/` when `BUILDROOT` is
set.** That is the host repository's (`mcDeepGEMM/build.sh`) own variable and
destination, so one packaging step can collect both wheels by pointing a single
`BUILDROOT` at both trees. A `--build-only` flag was this script's earlier
spelling and went with its flag parsing (`9a6105b`); the `BUILDROOT` export
came back on its own (`build.sh`, 2026-09-16) because it is the reference's
behaviour and needs no flag.

It is tagged `py3-none-linux_x86_64` and declares `torch` and `apache-tvm-ffi`
(unpinned -- the MACA torch builds carry metax-suffixed local versions, which a
pin would reject). The tag is honest: the extension is loaded through
`tvm_ffi.load_module` and exports no `PyInit`, so there is no CPython ABI in it
to name, and what the wheel is actually sensitive to is the platform. See
CLAUDE.md's wheel table for what the target machine must still provide.

```bash
./develop.sh                           # in-place build, this device only
./build.sh                             # a wheel, every family (CUCC_TARGETS to narrow)
BUILDROOT=/out ./build.sh              # ...and copy it to /out/wheel/
./install.sh                           # a wheel for this device, pip installed
./clean.sh && ./develop.sh             # full rebuild
./run_test.sh --perf --dtype bf16 -nc  # the performance grid
./run_test.sh --all                    # performance grid + correctness sample
```

`CUCC_TARGETS` is the whole build interface (the same variable and meaning as
mcDeepGEMM's `build.sh`). Unset means `xcore1000,xcore1500,xcore1600` in
`build.sh` -- one target per family, the same *set* of families mcDeepGEMM's
default names, minus its per-part aliases, which `mxcc` rejects outright
(`xcore1008`, `xcore1610`, `xcore1620`) -- and `native`, mxcc's own spelling
for the local part, in `develop.sh` and `install.sh`. The difference is deliberate: a wheel has to
carry every family to be shippable, while a local build only has to carry the
board in front of you. `CUCC_TARGETS=xcore1600 ./develop.sh` is the one-family
form when you want it explicitly.

**The target list is a list of images, not of builds.** One extension,
`deep_select/deep_select_maca*.so`, carries one image per target, because mxcc
takes a comma-separated `-offload-arch` and compiles each into its own image of
the same source (measured: three targets → three images, 11.07 MB against
3.7 MB for one). Each target builds `csrc/xcore1000/maca_topk.cu` -- the
hand-written MACA kernel, for every capacity, 64 KiB and 128 KiB alike.

One consequence is worth stating because it shapes the source: **there is no
per-architecture specialization at compile time.** A family macro would be
correct in one image and a lie in the other two, so the two numbers the kernel
sizes its grids against -- the SM count and the fp32 split's work target --
travel as arguments, sourced from the architecture the device reports through
torch. See CLAUDE.md, "the arch constants are arguments".

The ported kernels under `csrc/xcore1600/` are reserved and unbuilt: they select
wrong on a C600U, and the C500 kernel is both correct there and 1.5-2.9x faster
(see CLAUDE.md, "Can a C600U run the C500 kernel"). Wiring the port back is a
source change to `setup.py`'s `SOURCES` and `include_dirs` together, deliberately
not an environment variable.

Both scripts derive `CUDA_HOME`/`CUDA_PATH`/`CUCC_PATH` from `MACA_PATH`: torch's
`_find_cuda_home()` reads the first two before its `${MACA_PATH}/tools/cu-bridge`
fallback, and `cucc` execs `gomxccbin` out of `CUCC_PATH`, so a stale value in
the caller's environment silently beats `MACA_PATH`.

Device code is compiled by `mxcc`, reached through **cu-bridge's `cucc`**, with
`--offload-arch=xcore<N>` passed per extension, so an extension is for the
architecture it is named after and no other. `setup.py` does not reassign
torch's `CUDA_HOME`: `torch.utils.cpp_extension`'s MACA build already resolves
it to `${MACA_PATH}/tools/cu-bridge`, `_join_cuda_home` substitutes `bin/cucc`
for the absent `bin/nvcc`, and cucc is the whole CUDA-dialect adapter (the
`__macro_mxcc.h` compatibility header, torch's `-gencode` →
`-D__CUDA_ARCH__`, `-lcudart` → `-lmcruntime`, the MACA library include
catalogue). `-gencode` derived from the building machine's device is not
involved: the architecture set comes from `CUCC_TARGETS` through
`-offload-arch`, and the include catalogue is cucc's own, not repeated here.

`-use-fast-math` is passed with FTZ turned back off
(`-Xclang -fdenormal-fp-math-f32=ieee`): the ranking path is integer-only and
indifferent either way, and the flag keeps the fill-value conversion exact for a
denormal `value_oob_fill_value`. `api.cu` is host code, spelled `.cu` so torch
routes it to the device rule rather than to `$cxx`.

`pip install .` does not currently work, for a reason inherited from upstream:
`setup.py` stamps the version with `datetime.now()` (upstream `setup.py:204`,
used at `:214`), and a PEP 517 install runs `setup.py` once for metadata and
again for the wheel -- when the two runs straddle a second boundary the wheel is
rejected as misnamed (`Wheel has unexpected file name`). Build isolation adds a
second, unrelated failure (torch is not in pip's isolated build environment),
which `--no-build-isolation` avoids. `build_ext --inplace` above sidesteps both.

## Usage

```python
import torch
import deep_select

# input: (batch_size, vocab_size), torch.bfloat16 or torch.float32.
# Its row stride must be a multiple of `deep_select.get_stride_requirement()[0]` bytes, and its last dimension must be contiguous.
batch_size, vocab_size, topk = 4, 204800, 1024

x = torch.randn(batch_size, vocab_size, dtype=torch.bfloat16, device="cuda")

values, indices = deep_select.topk(
    x,
    topk,
    sorted_index=True,         # return each row's indices in ascending order
    indices_type=torch.int32,  # torch.int32 or torch.int64
    return_value=True,         # False skips the value output (~10% faster)
    backend="maca_c",          # the MACA kernel; see "backends" below for why
)
# values:  (batch_size, topk) of x.dtype
# indices: (batch_size, topk) of indices_type
```

**`backend="maca_c"` above is not decoration.** The library's default is
`"torch"`, a reference implementation built from torch ops -- picked so that a
call which names no backend cannot reach a kernel defect on a device the kernel
has not been validated on. A caller who wants the kernel asks for it, by name
or with `DS_TOPK_BACKEND=maca_c` for a whole process. What each backend is, and
why the default is where it is, is the next section.

The row stride of the input tensor (`x`) must be aligned to `deep_select.get_stride_requirement()[0]` bytes. For unaligned inputs, padding is necessary.

Both outputs are allocated by the call, and their strides are aligned to `deep_select.get_stride_requirement()[1]` bytes (so they may be non-contiguous). Pass `output_idx=` to write indices into a buffer you own, and that buffer must satisfy the same stride requirement.

For the full signature, see [`deep_select/interface.py`](deep_select/interface.py).

`backend=` picks the implementation, and it names implementations rather than
architectures -- the same vocabulary as `mcDeepGEMM`'s `backend=`.

`"torch"` (**the default**) is a reference implementation of the same contract
built from torch ops: it runs on any device and dtype, so it is usable on a
machine with no MACA kernel built at all, and for differentially checking
results (`tests/test.py --backend torch`). It is the backend to trust
when the question is what the *answer* should be, and it is the default for
that reason: the kernel path is validated against it, so leaving it out of the
default costs no coverage and removes the last route by which a caller who
asked for nothing in particular could reach a kernel defect. The cost is speed
-- it is the slow one. Unlike the kernels it rejects `bfloat16` +
`sorted_value`, matching upstream.

`"maca_c"` is the MACA kernel this device has: the hand-written kernel under
`csrc/xcore1000/`, built as one image per family named in `CUCC_TARGETS` (see
"Build" above). **Every capacity runs that same kernel** -- 64 KiB and 128 KiB
parts alike -- so no architecture name appears at this level, and nothing about
the part a caller happens to be on has to reach the call. It is the production
path and the fast one: ask for it by name, or set `DS_TOPK_BACKEND=maca_c` for
a whole process. An unrecognized value in that variable is ignored rather than
raised, so a typo cannot break every call.

`"deep_gemm"` is that package's own indexer selector
(`deep_gemm.fp32_indexer_topk_selector`), called through its Python API. That
package is imported only when this backend is asked for, so this repository
stays standalone without it. It is much faster than the MACA kernel on long
rows, and it implements a strict subset of the contract -- float32 scores,
`topk <= 2048`, unordered output -- so what it cannot serve raises
`deep_select.UnsupportedByBackend` rather than answering something narrower.
It also carries one known hole, in that kernel rather than in this adapter: a
kernel collects the members of the threshold *coarse* bin (the half-precision
ordered key >> 6, so everything inside one 64-half-ULP bucket -- which is what
a row of near-tied scores fills) before refining, and the chunked kernel
silently drops members past its staging capacity. A row with more than 4096
values in one such bucket can therefore be ranked against an arbitrary subset
of it, varying run to run. That is filed as a strict `xfail` in mcDeepGEMM's
suite,
`deep_gemm/tests/test_indexer_topk_selector.py::test_selector_candidate_overflow`;
`maca_c` has no such hole.

### Call logging

Every public entry point can record the shapes and dtypes it was called with,
and the ones it returned -- which is what makes a shape mismatch readable after
the fact instead of only as a contract rejection. It is off unless `DS_LOG`
names a target:

```bash
DS_LOG=1                  # stderr
DS_LOG=run.log            # that file, appended
DS_LOG=/tmp/logs/         # /tmp/logs/deep_select_<date>_<time>.log
DS_LOG=file               # ./deep_select_<date>_<time>.log
```

```
# ============== deep_select run log ==============
# version : 1.0.0 (git 0742ebc)
# device  : xcore1000 (MetaX C500, 104 SMs)
# env     : DS_LOG=1, DS_TOPK_BACKEND=maca_c
# started : 2026-09-15T23:53:42
# ================================================
[23:53:42.474] topk(input=Tensor[4, 204800] torch.bfloat16 (cuda:0), topk=1024, backend='maca_c') -> (Tensor[4, 1024] torch.bfloat16 (cuda:0), Tensor[4, 1024] torch.int64 (cuda:0)) [38.537 ms]
```

A tensor is recorded as shape/dtype/device and **its values are never read**,
so logging cannot change what a call computes. A call that is rejected is
recorded too, with its reason. A call into one backend through `topk` is one
record, not two. When `DS_LOG` is unset the decorator returns each function
unwrapped, so there is no wrapper on the hot path at all.

Two things to know before using it on a benchmark: every record synchronizes
the device (that is the only way its elapsed time means anything), and the
switch is read when `deep_select` is first imported, so it has to be set before
then.

### Environment variables

**Read by the library** (`deep_select/`), at import or per call:

| variable | default | effect |
| --- | --- | --- |
| `DS_TOPK_BACKEND` | unset → `"torch"` | which implementation a call with no `backend=` runs. Unrecognized values are ignored, not raised |
| `DS_LOG` | unset → off | call log target: `1`/`stderr`, `file`/`log`, a directory, or a path. See "Call logging" |
| `DEEP_SELECT_NO_STREAM_GUARD` | unset | set = launch without installing torch's current stream, i.e. on the legacy default stream. Only for bisecting that handoff; a call made this way races a producer on a side stream |

**Read by the build**:

| variable | default | effect |
| --- | --- | --- |
| `CUCC_TARGETS` | `xcore1000,xcore1500,xcore1600` (`build.sh`); `native` (`develop.sh`, `install.sh`) | which `-offload-arch` images go into the one extension; `native` = the local part, in mxcc's own spelling. An unrecognized target is rejected by `mxcc` |
| `MACA_PATH` | `/opt/maca` | the MACA toolkit root, and the authority for it. All three scripts derive `CUDA_PATH`/`CUDA_HOME`/`CUCC_PATH`/`LD_LIBRARY_PATH` from it, since a stale one of those in the caller's shell silently beats it |
| `BUILDROOT` | unset | `build.sh` only. When set, the wheel is also copied to `${BUILDROOT}/wheel/` — the host repository's own destination, so one packaging step can collect both wheels by pointing a single `BUILDROOT` at both trees |
| `MACA_HOME` | — | toolkit root too, consulted when `MACA_PATH` is unset. `MACA_PATH` wins if both are set |
| `MAX_JOBS` | torch's | ninja's `-j`. Not read by this tree -- `build.sh` passes it through to the build |
| `DEEP_SELECT_MACA_STACK_CHECK` | unset → skip | run the mxcc `--resource-usage` spill gate over `1` (all sources) or a comma-separated list |
| `DEEP_SELECT_MACA_STACK_BASELINE` | `48` | the gate's threshold, in bytes. Per toolchain: 48 is what this one reports for a device function that does not spill |

**Read by the test and benchmark harness**:

| variable | default | effect |
| --- | --- | --- |
| `CUDA_VISIBLE_DEVICES` | the machine's | which device the runs use. Recorded in the perf manifest, and the way to pick a device -- there is no exclusivity gate |
| `DS_RESULTS_DIR` | `results/` | where `run_test.sh` writes its log and receipt |
| `DS_BENCH_DIR` | `perf_data` | `run_bench.sh`'s output root |
| `DS_BENCH_TIMEOUT` | `5400` | per-run timeout, in seconds |

There is no variable that points at the `deep_gemm` source or checkout: what
the backend needs is the *package*, so whether it can serve a run is
`import deep_gemm` plus the entry it calls -- which is what
`deep_select.deep_gemm_available()` answers, and what decides whether the
selector shapes ride along.

### Variable-length rows

`end` sets a per-row upper bound (exclusive). Rows shorter than `topk` are padded with
`value_oob_fill_value` / `idx_oob_fill_value`:

```python
batch_size, vocab_size = 2, 129280   # 129280 is a multiple of 256, so float32 is fine
x = torch.randn(batch_size, vocab_size, dtype=torch.float32, device="cuda")

end = torch.tensor([129280, 100000], dtype=torch.int32, device="cuda")  # (batch_size,)
values, indices = deep_select.topk(x, 1000, end=end, sorted=True,
                                   indices_type=torch.int64)
```

### NaN handling

NaN checking is always on. With the default `abort_when_nan_found=True` the kernel invokes `trap()` and aborts. Rows whose length is `<= topk` are never NaN-checked.

The check is a bit-pattern test, not a comparison against a sentinel key, so it
catches every NaN encoding of either sign, quiet or signaling (upstream:
`set.nan`; MACA: exponent all-ones with a non-zero payload). With
`abort_when_nan_found=False` such a row leaves `0x3F3F3F3F` in
`output_idx[row, 0]` and the rest of that row is undefined, so a NaN row must be
excluded from any value comparison, as the test suites do.

## Testing

The suite is upstream's, landed unmodified under [`tests/`](tests/) --
[`tests/test.py`](tests/test.py) builds a correctness table and a performance
grid and runs both through the same `run_testcase`.  Its checks are the
contract's own: index range, uniqueness, `value_i == input[index_i]`, the
definitional `min(selected) >= max(unselected)`, the NaN guard, and the
orderings.  No reference implementation is computed anywhere, so nothing here
can drift away from the contract it is checking.

Two edits under `tests/kernelkit/` are the whole delta, and both are needed to
run at all on MACA: `platform.py` asks torch whether it can see a device
instead of grepping `lspci` (a MACA part does not enumerate as an NVIDIA 3D
controller, so every MACA host was reported CPU-only and `bench()` refused to
run), and the one PEP 701 f-string at `stress.py:292` (Python 3.12 syntax; this
tree builds against 3.10) is rewritten with the same meaning.

### Performance

The performance grid is small, and it runs the way upstream runs it:

```bash
PYTHONPATH=. python tests/test.py --perf-only              # 95 cases
PYTHONPATH=. python tests/test.py --perf-only -nc          # skip the cooldowns
PYTHONPATH=. python tests/test.py --perf-only --dtype bf16 # 90 of them
```

Every case is checked first and timed second, with
`tests/kernelkit/bench.py`'s kineto harness (L2 flushed): it prints
`topk : <us>, <TB/s>`, times raw `torch.topk` on the same shape right after,
and prints the speedup whenever the two are comparable (`end is None`, no
`output_idx_offset`, `vocab_size >= topk`).  Because the checks run first, a
performance case that selects wrong is reported as a failure rather than as a
time.  On C500 the bf16 half of the grid passes in about a minute.

### Correctness

The correctness table is 105,138 cases and takes hours, so
[`tests/test.py`](tests/test.py) -- upstream's file -- takes `--sample N`,
`--seed S` and `--backend NAME`:

```bash
PYTHONPATH=. python tests/test.py --backend maca_c --sample 200   # 200 of the table
PYTHONPATH=. python tests/test.py                                 # all of it (hours)
```

`--backend` is the one thing the suite cannot express on its own -- its call site
passes no `backend=`. It is now a parameter of `run_testcase`, defaulting to
`maca_c` (the kernel this tree exists to ship), so the other two are reachable
by name:

```bash
PYTHONPATH=. python tests/test.py --backend maca_c   # the kernel (the default)
PYTHONPATH=. python tests/test.py --backend torch    # the reference
PYTHONPATH=/path/to/mcDeepGEMM:. python tests/test.py --backend deep_gemm
```

A pinned backend and a process default are different questions and there is a
flag for each. `--backend maca_c` pins the call, so it says nothing about the
default -- the process default is never consulted. `DS_TOPK_BACKEND=maca_c`
leaves the official call site exactly as written and sets the process default
instead, so what runs is the path a caller who names no backend actually takes.

```bash
DS_TOPK_BACKEND=maca_c PYTHONPATH=. python tests/test.py
```

`--backend deep_gemm` in an environment where the package cannot serve the call
is refused before anything runs, rather than discovered case by case. A backend
that *is* present but serves a narrower range says so through
`UnsupportedByBackend`, and those cases are counted as `unsupported` rather than
failing the run.

**Every case reports a status and the run reaches its summary.** Upstream stops
at the first wrong answer (`sys.exit(1)`), which is the right thing for CI and
tells you nothing about the other 105,137 cases. The statuses are `pass`,
`check_fail` (it selected wrong), `crash` (it raised), `skip`
(`torch.cuda.OutOfMemoryError`) and `unsupported`. A `crash` normally stops the
run -- a poisoned CUDA context makes every later case report the same error --
unless `-rf` is given, which keeps going so the damage can be counted. `-rf` is
therefore what makes a long exploratory run worth starting, and it keeps the
meaning it had upstream (`if not args.run_to_finish: sys.exit(1)`), which was
about continuing rather than stopping:

```bash
./run_test.sh --test --sample 2000          # 200 cases by default; this one 2000
```

For CI the exit status is the contract, and it is unchanged: any `check_fail` or
`crash` exits 1, and a clean run exits 0. What *is* new is which implementation a
bare invocation tests. Upstream's call site is `deep_select.topk(...)` with no
`backend=`, which resolves to the library default (`torch`, the reference); here
`run_testcase` passes `backend="maca_c"` unless told otherwise, so the suite
tests the kernel. Asking for the reference is now explicit:

```bash
PYTHONPATH=. python tests/test.py --backend torch
```

Two things the suite does not cover, both recorded rather than papered over: the
contract rejections (a strided input row, the wrong dtype, `topk` out of range,
an output buffer smaller than `(batch_size, topk)`) -- the official table
asserts on values and has no exception cases -- and `begin` / `hint` /
caller-allocated `output_idx`, which the official call site always passes as
`None`.

A timing run needs the device to itself: `pgrep -f "tests/test.py"` first, as
with any benchmark.

## Citation

```text
@misc{deepselect2026,
    title={DeepSelect: High-Performance TopK Kernels for DeepSeek Sparse Attention and Sampling},
    author={Yi Qian and Shengyu Liu and Yichen Li},
    year={2026},
    publisher = {GitHub},
    howpublished = {\url{https://github.com/deepseek-ai/DeepSelect}},
}
```
