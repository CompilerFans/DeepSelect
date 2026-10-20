# NaN 扫描 flag 表：read-and-clear，而不是每次调用 memset（2026-10-20，C600-U）

## 结论

每条路线的逐行 NaN 投票改由 dispatcher 独占的 `scan_flags` 表承载：每行一个
`int32`，只在 scratch 增长时清零一次；契约半（`topk_kernel_radix`）在顶部
`tid == 0` 的块里**读取并清零**本行的槽位，且在任何返回路径之前完成。于是：

- 每次调用的 `cudaMemsetAsync` 删除（C600-U 实测 ~3.9 µs/调用，b1 decode cell
  的 ~15%；C500 ~11 µs，都是发射开销，对行长平）；
- 槽位不可能携带上一次调用的答案——这原本是靠那次 memset 保证的。

## 症状（改前实测）

`repro_flags.py`（本目录，C600-U，改前产物）：

```
b1v262144 cold  oob 0 / 512  sentinel False ...
b6v1048576 dirty oob 0 / 3072 sentinel False ...
b1v262144 warm  oob 1 / 512  sentinel True  first [1061109567, ...]
```

warm 调用（b1 v262144 k512 bf16，输入无 NaN）第一槽就是 `0x3F3F3F3F`。
上面是改前脚本的原始输出；提交的 `repro_flags.py` 把同一序列改成逐项
`OK` / `FAIL`。

## 机制

改前 bf16 split 的 flag 表是 `ws.nan_flags`，fp32 split 是 `ws.lengths`——都在
**同一个进程级、只增不减的 scratch** 里，且偏移是几何相关的表达式。上一个调用
在相同字节留下的是候选索引（非零），契约半把它读成了一次投票。每次调用的
memset 只是把症状盖住，代价是一次发射。

改后：表由 dispatcher 单独 `cudaMalloc`，与任何 workspace 布局无关，分配时清零
一次；契约半读+清。写者仍可以无条件 OR 一个 1。

## 代码

- `csrc/maca_kernels/common/maca_topk.cu`：`needs_scan_table = check_nan`；
  增长时 `cudaMemset(grown, 0, …)`；删除每次调用的 memset。
- `csrc/maca_kernels/common/contract.cuh`：`RowParams::nan_flags` 改为非 const；
  新增 `__shared__ int32_t nan_vote`；顶部读+清；两个处置点读 `nan_vote`。
- `csrc/maca_kernels/common/routing.cuh`：`ChunkedWorkspace::nan_flags` 与
  `ChunkedF32Workspace::lengths` 删除；两个 split 都改传 `params.scan_flags`；
  launch 失败后清表（行路径接管，它不读表）。
- `csrc/maca_kernels/common/topk_chunks.cuh` / `topk_coarse12.cuh`：注释改为
  read-and-clear；chunks 的三次发射中途失败时清表。

## 验证

- `repro_flags.py`（本目录）：bf16/fp32 两套 cold-clean → dirty(NaN) →
  warm-clean → `check_nan=False` → dirty-again 序列；默认扫描与
  `DS_FUSE_NAN_OFF=1` 两种配置均 `RESULT OK`。
- `nan_trap.py`（本目录）：b1 v262144 bf16 含 NaN、`abort_when_nan_found=True`，
  trap 在 `nan_abort_kernel<128>`。
- 改前 gate 报出的三个失败形状单独复跑全过：
  `b47 v682965 k512 fp32 offset`、`b4096 v225467 k1 fp32`、
  `b1 v262144 k512 bf16`。
- `./run_test.sh --all --sample 1000 -rf`（C600-U，产物 md5
  `2651fe3d991dc54912335d9bd922abf7`，提交前工作树）：perf **147/147**，
  correctness **1131/1131**（1149 抽样中 18 个 OOM skip），`check_fail 0`、
  `crash 0`、`unsupported 0`，`GATE_RC=0`。独立复跑同计数（产物 md5
  `7008d9ebd3952d100d91977775a19f63`，收据
  `results/deepselect_run_20261020_155618.txt`）。
- **落地产物复验**（`e168b4e` 的提交树，干净工作区，产物 md5
  `187d7932c7992b237058e3888ca1f41f`）：
  `./run_test.sh --test --sample 200` → 349 抽样中 **343 通过 / 6 OOM
  skip**，`check_fail 0`、`crash 0`；
  `./run_bench.sh --set-baseline` → `BENCH_RC=0`，correctness 阶段 549 抽样
  **537 通过 / 12 OOM skip**，official **147/147**，official_fp32 **30/30**，
  chunks 臂全 PASS，graph capture / replay 全 MATCH，`grep -c FAIL` = 0。
  快照落在 `perf_data/MetaX_C600-U/20261020_170531/`，并成为该设备的
  `baseline`；`manifest.json` 与 `run_header.txt` 的扩展 md5 一致。

## 配对 A/B 与算子级读数

`tools/ab_snapshot.py`，同一 session、两臂逐轮交替，3 轮 × 10 iter，
`--abort-when-nan-found no`（官方形状）：old = `3d47bd7`（md5
`8e189dee1a7147a822dfd061583d2ba6`），new = `e168b4e`（md5
`187d7932c7992b237058e3888ca1f41f`），子进程每轮自报加载的 md5。比值 =
median(new)/median(old)，>1 表示新更慢；明细 `ab_paired.json`（本目录）：

| 路线 | n | median |
| --- | ---: | ---: |
| split | 16 | 1.0250 [1.004, 1.078] |
| coarse12 | 10 | 1.0054 |
| row | 95 | 1.0027 |
| 全部 | 121 | 1.0034 |

**这个比值里看不到被删掉的那次 memset**：`ab_snapshot.py` 的核名过滤器
（`stage`/`nan`/`radix`/`coarse12`）不匹配 `Memset (Device)`。那次发射的
收据直接来自 profiler trace：改前 split 调用带 `Memset (Device)`
4.94–5.32 µs（bf16）/ 4.76 µs（fp32），`mcMemsetAsync` 自身 6.20 µs；
改后产物没有 Memset 行。

算子级（每次调用后同步，200 iter 中位数，两轮交替）：

| cell | old | new | Δ |
| --- | ---: | ---: | ---: |
| b1 v262144 k512 bf16（split） | 120.0 µs | 117.4 µs | −2.6 |
| b1 v129280 k512 fp32（split） | 123.3 µs | 119.2 µs | −4.1 |
| b6 v129280 k512 fp32（split） | 135.5 µs | 129.8 µs | −5.7 |
| b4096 v1048576 k512 bf16（row） | ≈16.68 ms | ≈16.68 ms | ≈0 |

**一处未解释的代价，如实记录**：split 的核级时间在两套仪器下都显示新臂的
stage 内核慢约 2.3 µs/调用，且 e2e 的账正好对上——省 4.9 µs 的 memset、
多花 2.3 µs 的内核、净赚 2.6 µs。stage 内核的源码两臂之间是**注释级**差异
（`tools/comment_gate.py` 判定 `radix_chunked.cuh`、`topk_coarse12.cuh` 仅
文档变化），`setup.py` 相同，`nm -S` 只见 host 侧符号变化；在 kk.bench 下
两种先后顺序给出同一结果，也不是"循环里多了一次小 memset"所致（padding
实验：新 32.33 nopad vs 31.9 pad，旧 29.72/29.77）。**机制未定**。主要
假设是 scratch 布局变了：flag 表不再占据进程级 scratch，`params.scan_flags`
是单独分配，scratch 的字节数与基址随之外移，可以改变 stage 内核候选数组的
L2 集映射。该假设未测。净效果仍是算子级更快。

## 代价与边界

- 读+清是每行一次 store，发生在已经写两个行头（`row_offset`、
  `nan_abort_flags`）的同一个 `tid == 0` 块里；寄存器不变。
- 失败路径的 `cudaMemsetAsync` 只在 launch 未成功时发生，正常路径零成本；
  `check_nan=False` 时整张表不存在。
- 代价是 split 路线的核级时间 +2.5%（A/B），机制未定（见上节），算子级
  仍净赚 2.6–5.7 µs；若查清是 scratch 布局所致，可在不动契约的前提下消除。
- 并发调用共享 scratch 是既有性质（workspace、lengths 表同样共享），本次不改变。
- 只在 C600-U（xcore1600）上实测；改动位于 `common/`，xcore1500 产物同样包含它，
  未在该族上测。
