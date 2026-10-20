# NaN 扫描 flag 表：read-and-clear，而不是每次调用 memset（2026-10-20，C600-U）

## 结论

每条路线的逐行 NaN 投票改由 dispatcher 独占的 `scan_flags` 表承载：每行一个
`int32`，只在 scratch 增长时清零一次；契约半（`topk_kernel_radix`）在顶部
`tid == 0` 的块里**读取并清零**本行的槽位，且在任何返回路径之前完成。于是：

- 每次调用的 `cudaMemsetAsync` 删除（C600-U 实测 ~3.9 µs/调用，b1 decode cell
  的 ~15%；C500 ~11 µs，都是发射开销，对行长平）；
- 槽位不可能携带上一次调用的答案——这原本是靠那次 memset 保证的。

## 症状（改前实测）

`/tmp/repro_flags.py`（C600-U，改前产物）：

```
b1v262144 cold  oob 0 / 512  sentinel False ...
b6v1048576 dirty oob 0 / 3072 sentinel False ...
b1v262144 warm  oob 1 / 512  sentinel True  first [1061109567, ...]
```

warm 调用（b1 v262144 k512 bf16，输入无 NaN）第一槽就是 `0x3F3F3F3F`。

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

- `/tmp/nan_state_test.py` 四模式：`CLEAN_OK` / `STATE_TEST_OK` /
  `ABORT_TEST_OK` / `nan_expect_trap`（trap 在 `nan_abort_kernel<128>`）。
- repro 的 warm 调用 `sentinel False`，`oob 0 / 512`。
- 改前 gate 报出的三个失败形状单独复跑全过：
  `b47 v682965 k512 fp32 offset`、`b4096 v225467 k1 fp32`、
  `b1 v262144 k512 bf16`。
- `./run_test.sh --all --sample 1000 -rf`（C600-U，产物 md5
  `2651fe3d991dc54912335d9bd922abf7`，提交前工作树）：perf **147/147**，
  correctness **1131/1131**（1149 抽样中 18 个 OOM skip），`check_fail 0`、
  `crash 0`、`unsupported 0`，`GATE_RC=0`。

## 代价与边界

- 读+清是每行一次 store，发生在已经写两个行头（`row_offset`、
  `nan_abort_flags`）的同一个 `tid == 0` 块里；寄存器不变。
- 失败路径的 `cudaMemsetAsync` 只在 launch 未成功时发生，正常路径零成本；
  `check_nan=False` 时整张表不存在。
- 并发调用共享 scratch 是既有性质（workspace、lengths 表同样共享），本次不改变。
- 只在 C600-U（xcore1600）上实测；改动位于 `common/`，xcore1500 产物同样包含它，
  未在该族上测。
