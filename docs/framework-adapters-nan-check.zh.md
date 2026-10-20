# 框架适配入口调研：vLLM / SGLang / DeepInfer 是否打开 NaN 检查

日期：2026-10-20。问题："这些框架接 DeepSelect 时，`check_nan` /
`abort_when_nan_found` 这个参数是否打开？"

## 结论

| 框架 | 是否接 DeepSelect | 入口 | `check_nan` | `abort_when_nan_found` | 实际行为 |
| --- | --- | --- | --- | --- | --- |
| **DeepInfer** | 是（内部生产路径） | `npu_ops/deep_select.py`、`models/v4_modules/index_attn.py`、`model_executor/layers/sampler*.py` | 未传 → 默认 **True** | 未传 → 默认 **True** | 扫描 + 命中即 trap |
| **vLLM** | 是，**可选** decode 后端（上游 CUDA 扩展） | `vllm/model_executor/layers/indexer_topk.py` | 该 op 没有这个参数（上游签名） | 显式传 **False** | 扫描 + `0x3F3F3F3F` 哨兵，不 trap |
| **SGLang** | **否** | `python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py` | — | — | 无 NaN 概念 |

**没有任何生产入口把 NaN 扫描关掉。** 唯一显式设置的是 vLLM，设的是
`abort_when_nan_found=False`（"query 不该因为 NaN 而中断"），不是
`check_nan=False`（"不扫描"）。两者的区别是本仓库
`interface.py` 里两个独立的开关（见 CLAUDE.md "NaN contract"）：vLLM 需要的
语义是"找到也**不要** abort"，而不是"不找"。

## 1. DeepInfer（`deep_select` 的三个调用面）

证据：`mcDeepGemm/docs/deepinfer_interface_dependency_20261020.md` §1.2 / §3.3
（调研归档，2026-10-20）与 `mcDeepGemm/tests/test_deepinfer/test_s6_deep_select.py`
（按真实调用形状写的回归测试）。

| 文件 | 场景 | 调用形状 |
| --- | --- | --- |
| `models/v4_modules/index_attn.py` | DSA indexer 选块（prefill/decode 同形） | `topk(p_sum, 512, sorted=False, end=topk_lens, indices_type=int32, sorted_index=True, output_idx=..., output_idx_offset=ks?, idx_oob_fill_value=-1, return_value=False)` |
| `model_executor/layers/sampler.py:643` | prefill top-logprobs | `topk(logprobs, k, True, indices_type=int32)` |
| `model_executor/layers/sampler_utils.py:68,167` | force-topk 模式 | `topk(logits, force_topk, False, indices_type=int32)` |
| `npu_ops/deep_select.py` | NPU 适配层 | sampler 回退 `torch.topk`；indexer 走真 kernel（MetaX 移植时 indexer **不可**回退） |

三个调用面都**不传** `check_nan`，也**不传** `abort_when_nan_found`，因此取
`deep_select.topk` 的默认值：**扫描打开、命中 trap**。归档文档 §3.3 把这条
写成显式契约："NaN 恒检（abort_when_nan_found 默认 trap）"。

参数形状对本树的意义：indexer 是 bf16、`topk=512`、行宽为 candidate 块的
整数倍；decode 时 batch（q 行数）很小 → 走 bf16 split，扫描融合在 stage 1 的
pass 1 里，不是独立一趟。sampler 是 fp32、`vocab≈128K`、k 小 → 走 fp32
coarse12 路线，扫描同样融合进它的 pass 1（`topk_coarse12.cuh`）。
所以 DeepInfer 在生产里付的是**融合后的**扫描代价，不是一份独立的行扫描。

## 2. vLLM（可选 backend，调上游 C++ op）

文件：`vllm/model_executor/layers/indexer_topk.py`（main，2026-10-20 查阅）。

- **接入方式**：`import vllm._deepselect_C` 注册 `torch.ops.deep_select`；
  仅当 CUDA 且 device capability family 为 100（SM100a/SM103a）时可用。
  它**不在** `_resolve_auto` 里——默认 decode 仍是
  `cooperative_topk` / `persistent_topk` / `top_k_per_row_decode`，
  `backend="deep_select"` 是显式可选。prefill 走
  `ops.top_k_per_row_prefill`（NVIDIA 自带实现），不经过 DeepSelect。
- 额外 gate：`topk <= 4096`、`input.shape[1] < 2**23`、fp32/bf16、最后一维
  连续、行 stride 满足输入对齐要求。
- **调用**（逐参数原样）：

  ```python
  torch.ops.deep_select.topk(
      input, topk,
      None,  # begin is not supported
      end,
      False,  # sorted_value
      False,  # sorted_index
      None,  # output_value
      output_idx,
      None,  # output_idx_offset
      IDX_OOB_FILL_VALUE,
      float("-inf"),  # value_oob_fill_value
      False,  # return_value
      False,  # abort_when_nan_found: dummy/capture passes can feed NaN
      # logits from uninitialized KV cache; the reference topk kernels do
      # not trap on NaN, and trapping would abort CUDA graph capture.
  )
  ```

- **这个 op 没有 `check_nan`**：它是 deepseek-ai/DeepSelect 上游的 pybind
  schema（`csrc/api.cpp` 的 13 个参数，`PYBIND11_MODULE(...){ m.def("topk", &topk); }`），
  上游把 NaN 扫描做进 kernel，`abort_when_nan_found` 只决定命中后的处置。
  vLLM 传 `False` 的理由写在注释里：dummy/capture pass 会喂进未初始化 KV
  cache 的 NaN，参考实现不 trap，trap 会打断 CUDA graph capture。
- **换到本树**：本树的 `deep_select.topk` 签名就是上游的参数顺序再加
  `backend` / `check_nan` 两个尾部关键字，所以这份 13 个位置参数的调用原样
  可用；`check_nan` 不传即默认 True，行为与上游一致（扫描 + 哨兵）。
- **不要把 `check_nan=False` 当成 vLLM 的加速开关**：vLLM 的诉求是"别
  trap"，不是"别扫描"；而 capture/dummy pass 的输入里**确实可能有 NaN**
  （注释自己说的），`check_nan=False` 会把 NaN 当成普通值参与排序，
  可能进入选中集合。正确映射是它现在的写法：`abort_when_nan_found=False`、
  扫描保持打开。

## 3. SGLang（没有 DeepSelect 适配）

文件：`python/sglang/srt/layers/attention/dsa/dsa_topk_backend.py`（main，
2026-10-20 查阅）。`DSATopKBackend` 只有三个成员：

| 成员 | 实现 |
| --- | --- |
| `SGL_KERNEL`（`"sgl-kernel"`） | `fast_topk_v2`；融合变换走 `fast_topk_transform_fused` / `..._ragged_fused`，v2/JIT 走 `topk_transform_{paged,packed,ragged}_v2` |
| `TORCH`（`"torch"`） | `torch.topk(dim=-1)` |
| `FLASHINFER`（`"flashinfer"`） | `flashinfer.top_k`；融合走 `top_k_page_table_transform` / `top_k_ragged_transform` |

没有 `deep_select` / DeepSelect 入口，也没有任何 NaN 相关的参数或检查：
越界位置用 `float("-inf")` 掩掉，选中项若等于 `-inf` 则把索引写成 `-1`。
FlashInfer 那一路额外传 `deterministic`、`tie_break`、`dsa_graph_safe=True`。

## 4. 对本站的含义

1. **生产入口全部保持扫描打开。** 本树把扫描融进每条路线的 pass 1
   （row path、bf16 split stage 1、fp32 split stage 1、coarse12/chunks），
   正是生产路径真正付的成本；`DS_FUSE_NAN_OFF=1` 的独立
   `nan_scan_kernel` 只是 fp32 的 A/B arm，没有生产调用方。
2. **`check_nan=False` 省下的 22–32%（fp32 五个官方 cell，ledger §8.4）
   在生产里拿不到**，除非调用方自己保证输入无 NaN。调研的三个框架里没有
   一个敢做这个断言（vLLM 明确说 capture pass 会有 NaN），所以这条路线的
   收益只属于自证干净的内部调用。
3. **`abort_when_nan_found=False` 是适配层应该显式写的那个值**（vLLM 已经
   写了；DeepInfer 保持默认 trap，是它自己的契约选择），它不改变扫描
   成本，只改变命中后的处置，且不参与选路——与本树的"扫描在 pass 1 内
   融合"没有冲突。
4. **flag 表的 read-and-clear 优化与框架无关**：它省的是每次调用的
   `cudaMemsetAsync` 启动开销（b1 decode cell 的 ~15%），所有入口都受益，
   不改变上面任何一条语义。
