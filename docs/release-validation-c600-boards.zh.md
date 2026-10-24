# C600 双板验证记录 —— MetaX C600-A / MetaX C600-U（2026-10-24）

同一份源码提交在两块 C600 系列板卡上各自构建、各自验证、各自测量一份官方
性能网格：**板卡是两台（32 AP 的 C600-A 与 28 AP 的 C600-U），family 是两个
（xcore1500 / xcore1600），记录是两份**。README 的四张性能图分别来自本文 §4
的两份 CSV；两台 pod 的 baseline 也各自指向本轮记录。

这份记录回答的是"这台机器上的这个二进制跑出了什么"，不是"三族发布验证"；
后者是 `docs/release-validation.zh.md`（P6，2026-10-23）的范围。

---

## 1. 结论

| 项 | MetaX C600-A | MetaX C600-U |
| --- | --- | --- |
| 板卡 / AP | `MetaX C600-A` / 32 | `MetaX C600-U` / 28 |
| family | xcore1500 | xcore1600 |
| artifact | `deep_select_maca_xcore1500.so` | `deep_select_maca_xcore1600.so` |
| artifact md5 | `4cdaee40f496ac7789cb23e9fbb5b808` | `98af089849a051b8ab84ed01b8be0152` |
| 契约 `maca_c` / `torch` | 73 / 70 全过 | 73 / 70 全过 |
| source-artifact smoke | 12/12 | 12/12 |
| 8000 例抽样 | 8000/8000，540 s | 8000/8000，781 s |
| 8 阶段 bench | 8/8 通过；snapshot 147 例 × 2 后端 = 294 行 | 8/8 通过；snapshot 147 例 × 3 后端 = 441 行 |
| `deep_gemm` 列 | **不可用**（包内无 xcore1500 镜像，§5） | 25 pass / 122 unsupported |
| 记录目录 | `perf_data/MetaX_C600-A/20261024_061341` | `perf_data/MetaX_C600-U/20261024_062159` |
| baseline | 本板首份记录 | 本板首份记录（不沿用 `20261021_110409`） |

---

## 2. 被测提交与构建

两台 pod 上的 checkout 都干净，指向同一提交：

| | |
| --- | --- |
| 源码 | `ef57dcc3b12ad9d95983b394e38a97fd1d0005d0` |
| DeepJIT gitlink | `6763222888a1a7f84829e1125882a608fd7a335c`（`third-party/DeepJIT`） |
| Python / torch | 3.10.10 / `2.10.0+metax20261017.1120` |
| MACA | `MACA_PATH=/root/maca/maca-sdk-20261017-1436/opt/maca-20261017`，`Version:20261017.1436` |
| 测量设备 | 每台 `CUDA_VISIBLE_DEVICES=1`；device 0 由本任务自己的 keepalive 占用，不参与测量 |

构建按 family 显式指定，不靠 `native` 猜产物身份：

```bash
# C600-A
CUCC_TARGETS=xcore1500 ./develop.sh
# C600-U
CUCC_TARGETS=xcore1600 ./develop.sh
```

两份构建日志都打印了同一对编译器（`develop.sh` 输出）：

```
deep_select: device compiler is /root/maca/maca-sdk-20261017-1436/opt/maca-20261017/tools/cu-bridge/bin/cucc,
             host compiler is .../tools/cu-bridge/bin/gnu
deep_select: compiling csrc/maca_kernels/common/maca_topk.cu + 89 generated instantiations
             once per family, for xcore1500     # C600-U 上为 xcore1600
```

产物与设备自报名称（`_binding.extension_path`）一致：

| | C600-A | C600-U |
| --- | --- | --- |
| 文件 | `deep_select/deep_select_maca_xcore1500.so` | `deep_select/deep_select_maca_xcore1600.so` |
| 字节 | 71,542,320 | 71,542,320 |
| md5 | `4cdaee40f496ac7789cb23e9fbb5b808` | `98af089849a051b8ab84ed01b8be0152` |

两份 manifest 都记 `deep_select_git_dirty: true`；这是 **记录器自己的输出目录**
造成的：`git status --porcelain` 只看到未跟踪的 `perf_data/<device>/`，源码树在
启动时是干净的（构建前的 `git status --porcelain` 为空）。读数时不要把这一位
当源码改动。

---

## 3. 正确性门

每台设备独立跑，全部在 device 1 上、对 checkout 自己构建的 artifact 跑
（`wheel_smoke.py --allow-checkout` 是 source-artifact 冒烟，**不冒充**安装
wheel 的验收）。

| 门 | 命令 | C600-A | C600-U |
| --- | --- | --- | --- |
| 契约与拒绝 | `python tests/contract_test.py --backend maca_c` | 73 passed / 0 failed | 73 passed / 0 failed |
| 参考后端 | `python tests/contract_test.py --backend torch` | 70 passed / 0 failed | 70 passed / 0 failed |
| source smoke | `PYTHONPATH=. python scripts/wheel_smoke.py --allow-checkout` | 12 passed / 0 failed | 12 passed / 0 failed |
| 8000 例抽样 | `python tools/sample_gate.py --sample 8000 --seed 20261024 --backend maca_c` | pass 8000 / 0 / 0 / 0 / 0，540 s（14.83 例/s） | pass 8000 / 0 / 0 / 0 / 0，781 s（10.25 例/s） |

Source smoke 在每台上都先打印了被测 artifact 与设备：

```
artifact .../deep_select_maca_xcore1500.so md5 4cdaee40f496     # C600-A, 32 APs
artifact .../deep_select_maca_xcore1600.so md5 98af089849a0     # C600-U, 28 APs
```

**抽样口径**：`tools/sample_gate.py` 默认 `--max-elems 2^26`，在 seed
`20261024` 下表的 105,138 例中，≤ 2^26 元素的 67,122 例是可抽池，其余
**38,016 例被计数而没有跑**；抽中的每一例 `p.seed` 被固定，同一条命令重跑是
同一批数据。**这不是 105,138 例全表通过**（§7）。

---

## 4. 官方 perf 网格（8 阶段）

命令（每台串行、单独跑，device 1；`--set-baseline` 在审查记录之后才用，
见 §6）：

```bash
CUDA_VISIBLE_DEVICES=1 ./run_bench.sh --backends maca_c,torch        # C600-A
CUDA_VISIBLE_DEVICES=1 ./run_bench.sh --backends maca_c,torch,deep_gemm  # C600-U
```

`run_bench.sh` 的 8 个阶段是 snapshot、correctness、cases_chunks、cases_graph、
cases_graph_cold、cases_graph_grow、official、official_fp32；每个阶段都是
`tests/test.py`/`tests/cases/` 的原文件，不是副本。**每个 perf 用例先检查、
再计时**，所以 official 的 147 是"147 例全部选对并且计时"，不是"跑了 147 个"。

| | C600-A | C600-U |
| --- | --- | --- |
| 记录 | `perf_data/MetaX_C600-A/20261024_061341` | `perf_data/MetaX_C600-U/20261024_062159` |
| 后端 | `maca_c, torch`（§5） | `maca_c, torch, deep_gemm` |
| snapshot cells / rows | 147 / 294 | 147 / 441 |
| `status_counts` | `maca_c/pass 147`、`torch/pass 147` | `maca_c/pass 147`、`torch/pass 147`、`deep_gemm/pass 25`、`deep_gemm/unsupported 122` |
| correctness | pass 536 / check_fail 0 / crash 0 / skip 13 / unsupported 0 | pass 536 / check_fail 0 / crash 0 / skip 13 / unsupported 0 |
| cases_chunks | ALL PASS | ALL PASS |
| cases_graph / cold / grow | ALL PASS / 4-4 / ALL PASS | ALL PASS / 4-4 / ALL PASS |
| official | 147 / 147 | 147 / 147 |
| official_fp32 | 30 / 30 | 30 / 30 |
| README 图（峰值有效带宽） | bf16 677.2 GB/s，fp32 446.0 GB/s | bf16 758.6 GB/s，fp32 486.5 GB/s |

两台的 `run_header.txt` 都记 `STALE no`，并在启动瞬间拍下 `mx-smi`：被
测量的 device 1 为 0% 占用，device 0 是本任务自己的 keepalive 进程
（`/root/busy.py`），不参与测量。

两处 147/30 的来历相同：`tests/test.py` 在 `deep_gemm` 包可导入时把 25 个
selector 形状并进官方网格（122 + 25 = 147；fp32 5 + 25 = 30）。C600-A 上这
25 个形状由 `maca_c`/`torch` 两列测到，`deep_gemm` 列不存在；C600-U 上三列
齐全（其中 `deep_gemm` 在 122 个官方 cell 上是 `unsupported`，在 25 个
selector cell 上 `pass`）。

correctness 阶段两台的 13 个 `skip` **逐项相同**：10 例
`b4096 × V 418673–1048576 × k 280–4096` 与 3 例 `b512 × V 8388607 ×
k 1024/2048`，全部是 `torch.cuda.OutOfMemoryError`（分配 30.46 GiB 一类的
形状），逐项出现在日志里，不是内核拒绝；没有 `check_fail`、没有 `crash`。

---

## 5. `deep_gemm` 在 C600-A 上不可用（本轮发现）

C600-A 上 `import deep_gemm` 成功、`deep_gemm_available()` 为真（包版本
0.6.0，位于 `/opt/conda/lib/python3.10/site-packages/deep_gemm`），但它的
选择器内核**没有 xcore1500 镜像**，第一次调用即失败：

```
[MCC][E]ErrorNoBinaryForGpu: Couldn't find xcore1500 binary for current device!
         Please compile using option: --offload-arch=xcore1500
deep_/floa b16-v66551-k2048    deep_gemm fail   [CUDA runtime error ...]
```

失败不局限于该后端：`ErrorNoBinaryForGpu` 把 CUDA context 置为
`cudaErrorInvalidDeviceFunction`，随后 `generate_testcase` 在下一个用例上
直接抛错，snapshot 循环在第 125/147 例（`b512-v66551-k2048`）中断：

```
generate_testcase failed for TestParam(batch_size=512, vocab_size=66551, ...):
CUDA error: invalid device function
```

因此 3 后端那轮记录（`perf_data/MetaX_C600-A/20261024_055131`）**无效**：
它的 snapshot 只有 125/147 例（`deep_gemm 3 fail / 122 unsupported`,
`maca_c/torch 125 pass`），不能当记录。C600-A 的正式记录因此显式用
`--backends maca_c,torch`，selector 形状仍由 `maca_c`/`torch` 两列测到，
`deep_gemm` 列缺席是环境事实，不是本仓库的选择。

C600-U 不受影响：同一 deep_gemm 包在该板上有 xcore1600 镜像，25 个 selector
用例全部 `pass`（§4）。

---

## 6. baseline 决策

两台 pod 在本轮之前都没有 `perf_data/<device>/baseline` 符号链接
（`MetaX_C600-A/` 下没有任何旧记录，`MetaX_C600-U/` 有历史记录但无指针）。
本轮在**审查完记录**后各自 repoint 到本板记录：

- `perf_data/<device>/baseline` 是机器本地指针，被 `.gitignore` 忽略，不提交；
- C600-A 这是首份设备本地记录，`run_bench.sh` 打印 "no baseline … skipping
  comparison"，不存在可比基线，也不拿 C600-U 的记录代替；
- C600-U 不沿用旧的 `20261021_110409`：本轮记录来自当前候选提交与当前
  artifact，指针只在 §4 的记录被判定完整、无竞争之后才移动。

repoint 的实际结果（`./run_bench.sh --baseline-dir <stamp>`，只改指针、
不测量）：

```
MetaX_C600-A: baseline (none) -> 20261024_061341
MetaX_C600-U: baseline (none) -> 20261024_062159
```

**被作废的运行（记录在这里，不作为 baseline）**：

| 记录 | 原因 |
| --- | --- |
| `MetaX_C600-A/20261024_055131` | `deep_gemm` 无 xcore1500 镜像，context 被污染，snapshot 截断在 125/147（§5） |
| `MetaX_C600-U/20261024_055341` | snapshot 完整，但 official/official_fp32 的 147/30 例全部 OOM skip：同一台 pod 上误启动了第二个 `run_bench.sh`，两个进程争用 device 1 |
| `MetaX_C600-U/20261024_061214` | 上述第二个进程，snapshot 阶段即被 OOM 截断 |

这三份记录留在 pod 上作证据，不进本仓库；有效的两份是 §4 的
`MetaX_C600-A/20261024_061341` 与 `MetaX_C600-U/20261024_062159`。

---

## 7. 未覆盖范围

- **完整 105,138 例全表没有跑**。本轮是 8000 例的 capped 抽样（§3），
  16–17 h 的全表仍是发布前门。
- **C500 / xcore1000 没有跑**：本会话没有 C500 设备；本轮只验证 C600-A
  （xcore1500）与 C600-U（xcore1600）两块。
- `csrc/maca_kernels/legacy/`（上游移植树）仍未编译、不可达，见
  `CLAUDE.md` 的 Known holes。
- `deep_gemm` 后端只在 C600-U 上被测量；C600-A 的包没有该板镜像（§5）。

---

## 8. 复现

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=/root/maca/maca-sdk-20261017-1436/opt/maca-20261017
export CUDA_VISIBLE_DEVICES=1

cd /root/github/DeepSelect
CUCC_TARGETS=xcore1500 ./develop.sh          # C600-A；C600-U 用 xcore1600
md5sum deep_select/deep_select_maca_xcore*.so

python tests/contract_test.py --backend maca_c
python tests/contract_test.py --backend torch
PYTHONPATH=. python scripts/wheel_smoke.py --allow-checkout
python tools/sample_gate.py --sample 8000 --seed 20261024 --backend maca_c

./run_bench.sh --backends maca_c,torch               # C600-A
./run_bench.sh --backends maca_c,torch,deep_gemm    # C600-U
# 审查记录后：
./run_bench.sh --baseline-dir <stamp>

python3 scripts/plot_perf.py perf_data/MetaX_C600-A/<stamp>
python3 scripts/plot_perf.py perf_data/MetaX_C600-U/<stamp>
```
