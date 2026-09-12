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

Which one a device runs is a property of the device, not a choice: `setup.py`
builds one extension per architecture, and `deep_select.topk` loads the one its
device has.

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

Consequences:

- `topk` above 1024 is served by `maca_topk.cu` only. The ported kernel's tuples
  cover `max_topk` 512 and 1024 -- a 4096 tuple cannot fit 128 KiB, since its
  survivor-pairs and extra-pairs regions alone come to exactly 128 KiB -- so a
  C600 / C600U rejects `topk` in `(1024, 4096]`.
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
PYTHONPATH=. python scripts/official_slice.py                          # correctness
```

The scripts wrap those calls. `build.sh` builds in place, `install.sh` builds a
wheel and installs it (symlinking the extension back under `deep_select/`, since
every suite here runs with `PYTHONPATH=.` and would otherwise resolve the repo's
copy), `clean.sh` removes the build artifacts, and `run_test.sh` runs the suites
and records what it measured:

```bash
./build.sh                             # this device
./install.sh                           # build a wheel and install it
./clean.sh && ./build.sh               # full rebuild
./run_test.sh --perf --dtype bf16 -nc  # the performance grid
./run_test.sh --all                    # performance grid + correctness sample
```

`CUCC_TARGETS` defaults to `native`, the device the build is running on (the
same variable and meaning as the host repository's `build.sh`), and each target
builds the kernel that fits it -- `csrc/xcore1000/` for a 64 KiB part,
`csrc/xcore1600/` for a 128 KiB one -- producing
`deep_select/deep_select_xcore<N>*.so`.

Device code is compiled by `mxcc` directly, with `--offload-arch=xcore<N>`;
neither cu-bridge's `cucc` wrapper nor a `-gencode` derived from the building
machine's device is involved, so an extension is for the architecture it is
named after and no other. `-use-fast-math` is passed with FTZ turned back off
(`-Xclang -fdenormal-fp-math-f32=ieee`): the ranking path is integer-only and
indifferent either way, and the flag keeps the fill-value conversion exact for a
denormal `value_oob_fill_value`. `api.cpp` is host code and is compiled by
`g++`.

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
)
# values:  (batch_size, topk) of x.dtype
# indices: (batch_size, topk) of indices_type
```

The row stride of the input tensor (`x`) must be aligned to `deep_select.get_stride_requirement()[0]` bytes. For unaligned inputs, padding is necessary.

Both outputs are allocated by the call, and their strides are aligned to `deep_select.get_stride_requirement()[1]` bytes (so they may be non-contiguous). Pass `output_idx=` to write indices into a buffer you own, and that buffer must satisfy the same stride requirement.

For the full signature, see [`deep_select/interface.py`](deep_select/interface.py).

`backend=` picks the implementation, and it names implementations rather than
architectures -- the same vocabulary as the host repository. `"maca_c"` (the
default) is the MACA kernel this device has: the hand-written kernel on a
64 KiB part, the ported one on a 128 KiB part. Which of the two that is, is a
property of the device rather than a choice, so no architecture name appears at
this level (`setup.py` still builds one kernel per architecture, and
`CUCC_TARGETS` still selects which). `"torch"` is a reference implementation of
the same contract built from torch ops: it runs on any device and dtype, so it
is usable on a machine with no MACA kernel built at all, and for differentially
checking results (`scripts/official_slice.py --backend torch`). Unlike the kernels it
rejects `bfloat16` + `sorted_value`, matching upstream.

`"deep_gemm"` is the host repository's own indexer selector
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
of it, varying run to run. That is filed as a strict `xfail` in the host
repository's suite,
`deep_gemm/tests/test_indexer_topk_selector.py::test_selector_candidate_overflow`;
`maca_c` has no such hole.

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
[`scripts/official_slice.py`](scripts/official_slice.py) drives a seeded uniform
sample of 200 of them through the same official `run_testcase`, unchanged
(capped at `batch_size * vocab_size <= 2**28`, which bounds the reference
`torch.topk` without dropping a shape family):

```bash
PYTHONPATH=. python scripts/official_slice.py     # 200/200 passed
```

`--backend` is the one thing the official suite cannot express -- its call site
passes no `backend=` -- so the driver rebinds `deep_select.topk` for the run
rather than editing the official file:

```bash
PYTHONPATH=. python scripts/official_slice.py --backend maca_c   # the default
PYTHONPATH=. python scripts/official_slice.py --backend torch    # the reference
PYTHONPATH=/path/to/mcDeepGEMM:. python scripts/official_slice.py --backend deep_gemm
```

A backend whose service range is narrower than the operator's says so through
`UnsupportedByBackend`, and those cases are counted as `unsupported` in the
summary instead of failing the run.

Two things neither arm covers, both recorded rather than papered over: the
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
