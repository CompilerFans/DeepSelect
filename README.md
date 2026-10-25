# DeepSelect

DeepSelect is a high performance implementation of the TopK kernel used in DeepSeek Sparse Attention (DSA) (which is used in DeepSeek V3.2, DeepSeek V4, and DeepSeek V4.1 models) and the sampler. It achieves 2 ~ 20x speedup compared to vanilla `torch.topk`.

This checkout is the **MACA implementation** of DeepSelect, for MetaX MACA parts (C500 / C600 / C600U). The CUDA implementation and its build live upstream; this tree builds MACA artifacts only. The public Python signature is upstream's, parameter for parameter, plus `backend=` and `check_nan=` appended after upstream's last argument, so an upstream call site works here unchanged.

## News

- 2026.09.10: We've released a brief analysis of the algorithm and its implementation: [English](docs/DeepSelect-deep-dive.md) | [中文](docs/DeepSelect-deep-dive.zh.md)
- 2026.09.10: We've released DeepSelect v1.0.0

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

### Lightning Indexer Scenario

bfloat16, `topk = 512`, one subplot per batch size, in GB/s on a shared axis.

MetaX C600-A (32 AP):

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer, MetaX C600-A](assets/perf_bf16_maca_MetaX_C600-A.png)

MetaX C600-U (28 AP):

![DeepSelect vs torch.topk, bfloat16 Lightning Indexer, MetaX C600-U](assets/perf_bf16_maca_MetaX_C600-U.png)

### Sampling Scenario

float32, `vocab_size = 129280`, `topk = 512`.

MetaX C600-A (32 AP):

![DeepSelect vs torch.topk, float32 Sampling, MetaX C600-A](assets/perf_fp32_maca_MetaX_C600-A.png)

MetaX C600-U (28 AP):

![DeepSelect vs torch.topk, float32 Sampling, MetaX C600-U](assets/perf_fp32_maca_MetaX_C600-U.png)

## Installation

Requires a MACA toolkit (`MACA_PATH`) and the MACA-compatible PyTorch build.

```bash
# from a checkout of this MACA fork
./install.sh
```

`./install.sh` builds the wheel for this machine's device and installs it. `./build.sh` builds a wheel carrying every MACA family (C500, C600, C600U); `./develop.sh` builds the in-place extension. `pip install .` is not supported here.

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

### Backends

`backend=` selects the implementation: `"torch"` is the default and the differential reference, `"maca_c"` is the MACA kernel. A call with no `backend=` follows `DS_TOPK_BACKEND`, so a caller who wants the kernel either passes `backend="maca_c"` or exports `DS_TOPK_BACKEND=maca_c`. `check_nan=False` turns off the input scan; `abort_when_nan_found` decides what happens once a NaN is found.

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

NaN checking is on by default (`check_nan=True`). With the default `abort_when_nan_found=True` the kernel invokes `trap()` and aborts; with `abort_when_nan_found=False` the row is flagged with `0x3F3F3F3F` in `output_idx[row, 0]` and the rest of the row is undefined. Rows whose length is `<= topk` are never NaN-checked.

## Development

Kernel architecture, the per-family build, routing, testing and the performance-change rules are documented in [CLAUDE.md](CLAUDE.md).

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
