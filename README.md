# DeepSelect

DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) (which is used in DeepSeek V3.2, DeepSeek V4, and DeepSeek V4.1 models) and the sampler. It achieves 2 ~ 20x speedup compared to vanilla `torch.topk`.

## News

- 2026.09.10: We've released a brief analysis of the algorithm and its implementation: [English](docs/DeepSelect-deep-dive.md) | [中文](docs/DeepSelect-deep-dive.zh.md)
- 2026.09.10: We've released DeepSelect v1.0.0

## MACA support

This tree also carries a MACA (MetaX) port of the operator, used by `mcDeepGEMM`.
The upstream CUDA kernels under `csrc/cuda_kernels/{v3,v3_fp32,v3_cluster}` are
**not built** there: they are written against TMA tensor-map loads, mbarriers and
cluster launches, none of which the platform provides. `csrc/maca_topk.cu`
reimplements the same operator -- the same public contract, the same
`deep_select.interface.topk` signature -- with portable primitives only
(shuffle, `atomicAdd`, `__syncthreads`, `__syncthreads_or`), as a radix refine
over an order-preserving key of each value. It is the only translation unit in
the MACA extension, so both scenarios below are served by that one general
kernel.

Consequences:

- The `v3_cluster` dispatch arm (`bfloat16`, `batch_size <= 6`,
  `vocab_size >= 512K`, `topk <= 1024`) has no MACA equivalent and is not
  selected; those shapes fall through to the general kernel. They are served
  (a cluster-less schedule), not rejected.
- The `vocab_size < 2^23` restriction is not enforced on MACA. It exists
  upstream because the CUDA kernels do their census with fp32-simulated integer
  arithmetic; the MACA kernel ranks integer keys and has no such bound (verified
  for `float32` at `vocab_size = 2^23`, case `fp32-vocab-2^23` in
  [`tests/check_maca.py`](tests/check_maca.py)).
- `sorted_value` is accepted for `torch.bfloat16` on MACA: the ordering comes
  from the same key, so both the descending order and the value/index pairing
  hold. Upstream rejects this as fp32-only, and `backend="torch"` keeps that
  restriction.
- `begin` and `hint` are rejected, as upstream.
- Everything else in this document -- the two scenarios, the `topk <= 4096`
  bound, `sorted_index` / `return_value` / `end` / `output_idx` /
  `output_idx_offset` / `idx_oob_fill_value` / `value_oob_fill_value`, and the
  NaN contract below -- behaves identically on MACA.
- MACA performance is **not** measured in this tree; the numbers in
  [Performance](#performance) are the upstream CUDA kernels'.

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

> These figures belong to the upstream CUDA kernels. The MACA port
> (`csrc/maca_topk.cu`) has not been benchmarked here -- see
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

Neither the single submodule (`csrc/3rdparty/cutlass`) nor the vendored
`csrc/3rdparty/kerutils` is referenced by the MACA build, which compiles
`csrc/maca_topk.cu` alone. It needs the MACA toolkit (`$MACA_PATH`, default
`/opt/maca`) and a MACA-compatible PyTorch:

```bash
python setup.py build_ext --inplace     # builds deep_select/deep_select_cuda*.so
PYTHONPATH=. python tests/check_maca.py # correctness, see Testing below
```

`setup.py` compiles the device code with `mxcc --offload-arch=xcore1000` (the
C500 / xcore1000 target) and passes `-ftz=false` on top of `--use_fast_math`
(which would otherwise enable FTZ). The ranking path is integer-only and
indifferent either way; the flag keeps the fill-value conversion exact for a
denormal `value_oob_fill_value`.

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

`backend=` picks the implementation: `"maca_c"` (default) is the MACA-native
kernel described above, `"torch"` is a reference implementation of the same
contract built from torch ops. The reference runs on any device and dtype, so it
is usable on a machine without the MACA kernel and for differentially checking
results; unlike the kernel it rejects `bfloat16` + `sorted_value`, matching
upstream. It is not covered by `tests/check_maca.py`, which compares against
`torch.topk` directly.

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

[`tests/check_maca.py`](tests/check_maca.py) is the runnable suite for the MACA
kernel -- it needs only `deep_select` importable:

```bash
PYTHONPATH=. python tests/check_maca.py            # all cases
PYTHONPATH=. python tests/check_maca.py --quick    # two-case smoke test
PYTHONPATH=. python tests/check_maca.py --group ties
PYTHONPATH=. python tests/check_maca.py --list
```

It covers the option matrix (`sorted_index` x `sorted_value` x `return_value`),
both scenarios, both index dtypes, ragged and short `end` windows,
`output_idx_offset`, non-default out-of-band fills, padded input/output row
strides, tie-heavy inputs, the NaN contract, and the contract rejections. Each
case is compared against `torch.topk` run in a separate process (see the module
docstring for why).

Upstream's [`tests/test.py`](tests/test.py) is not runnable on a Python 3.10
interpreter (its `tests/kernelkit/stress.py` uses PEP 701 f-strings, which
Python 3.12 introduced); the MACA suite above re-derives the coverage that
applies here rather than depending on it.

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
