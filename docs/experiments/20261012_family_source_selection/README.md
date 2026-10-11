# 三族的 kernel：源码是一份，族形状的路由却挂在 AP 数上（2026-10-12）

一句话：**构建侧**三族同源、且这是测量结论（C600U 上 C500 kernel 比 ported 那棵快
1.5–2.9x，见 CLAUDE.md「Can a C600U run the C500 kernel?」）；但**源码之外**那一层
「族形状路由」不是按族分的 —— `maca_topk.cu` 里三条只在 **28-AP C600U** 上拟合过的
梯子，钥匙是 `sm_count ∈ {28, 32}`，而 `csrc/structs.h:154` 说 family 1500（C600）
就是 28 AP。于是**一台 C600 会拿走全部三条 C600U 梯子，而这个仓库从没测过 C600**
（`perf_data/` 里没有 `MetaX_C600`，`docs/experiments/` 里没有一个 C600 目录）。

本记录给三样东西：**精确的映射**（§2）、**keystone 的证据链与两把候选钥匙的行为差 +
补丁文本**（§3–§4）、以及把「一族一份源码」落成 `setup.py` 里一张数据表的**落地与收据**
（§6，今日输出逐字节不变）。

复现：`python3 probe.py`（读树、不读设备；它把 §1/§2/§3 的表全部从源码里抽出来，
不在这里再抄一遍）。原始输出 `probe_raw.txt`，构建收据 `build_receipts.txt`。

**行号基准**：本记录写作期间 `csrc/structs.h` 有另一个 agent 的并行改动，行号会漂。
文中的行号以 `csrc/structs.h` md5 `494ea89be3b56b7fe8347fa730d8585c`、
`csrc/maca_kernels/xcore1000/maca_topk.cu` md5 `826412f752b4a643ff9c32222fd60fc0`
为准；每个引用都同时给了**符号名**（`is_c600u_ap_count`、`ARCH_SM_COUNT`、
`f32_coarse12_applies` …），照名字找比照行号找稳。

---

## 0. 结论

| # | 事实 | 证据 |
|---|---|---|
| 1 | 三族编译**同一份源码**，且这是测量结论不是偷懒 | `setup.py:43-50`；CLAUDE.md「Can a C600U run the C500 kernel?」的 1.5–2.9x 表 |
| 2 | 族**只**决定编译期常数，不决定源码；也**不**决定路由 | `csrc/structs.h:148-170` 三个常数，其中两个没有行为消费者 |
| 3 | 真正按「机形」分的路由，钥匙是**运行期 AP 数**，不是族 | §2.2 六条，`maca_topk.cu:763` 起 |
| 4 | **keystone**：`is_c600u_ap_count` 的名字声称是「C600U」，实际测的是「28 或 32 AP」；C600 就在这个集合里 | `structs.h:154` + `maca_topk.cu:763` |
| 5 | 一个 **1000 镜像被 28-AP 设备调用**时同样满足它 —— 这是 loader 自己写下的 caveat | `interface.py:196-201`；probe §3 的 `1000 / 28` 行 |
| 6 | 没有任何 C600 测量 | `ls perf_data/`、`ls docs/experiments/` |
| 7 | 文档与代码有一处**互相矛盾**的族归属说法 | `CLAUDE.md:1175` vs `arch_config.py` §3.3 |

**本次不改任何一条族门**（无 C600/C600U 可测，本仓的规矩是「路由改动在它针对的部件上量」）；
交付的是分析与补丁文本（§4）。

---

## 1. 今天：一族一份 artifact，三族一份源码

`setup.py` 现在是一张按族的表（本次新增，§6），三行都指同一个文件：

| family | artifact | 源码 | `ARCH_SM_COUNT` | `ARCH_SMEM_PER_AP_BYTES` |
|---:|---|---|---:|---:|
| 1000 | `deep_select_maca_xcore1000.so` | `csrc/maca_kernels/xcore1000/maca_topk.cu` | 104 | 64 KiB |
| 1500 | `deep_select_maca_xcore1500.so` | 同上 | 28 | 128 KiB |
| 1600 | `deep_select_maca_xcore1600.so` | 同上 | 28 | 128 KiB |

`csrc/maca_kernels/xcore1600/`（ported upstream 那棵）**在盘上、不在任何一行里**，
也不在 `include_dirs` 上 —— probe §1 直接打印这两条。

所以「xcore1500/xcore1600 就是 xcore1000 的源码换个 arch 编」这句话**在构建侧完全准确**。
值得补一句的是**它在什么意义上**准确：不是「没来得及分」，是**量过之后决定不分**
（C600U 上用 C500 kernel 1.5–2.9x，`check_result` 全过，而 ported 那棵五个 cell 全挂）。
要分的那条路（一族一份源码）在本记录落地前**不可表达** —— 现在可表达了，见 §6。

---

## 2. 族形状的东西写在哪：三张表

### 2.1 编译期族常数（`csrc/structs.h`）—— 只有一个有行为消费者

| 常数 | 消费者 | 结果 |
|---|---|---|
| `ARCH_FAMILY` | 只有 `static_assert` 与 artifact 命名；运行期的按族选文件在 **python 侧**（`_binding.family_suffix()`，读设备 sm） | 无路由 |
| `ARCH_SM_COUNT` | **无**（`structs.h:74-80` 自己写明） | 无路由 |
| `ARCH_SMEM_PER_AP_BYTES` | **一处**：`maca_topk.cu:1569` 的 `if constexpr`，`f32_coarse12_applies` 的预算半边 | 决定一条路是否存在 |

预算半边对**今天出货的三族全部为真**：它要求 16384 B/AP
（`max(route arena 16384, 被它顶替的 split 14056)`），而三族的预算是 65536/131072/131072。
probe §3 把这条打印成「families whose budget is short of it: none」——
**它今天是一条死门**，存在的理由是「预算装不下的族上这条路是错而不是慢」。

### 2.2 运行期 AP 数（`params.sm_count`）—— 六条，全部是「机形」问题

| 门 | 位置 | 取 28/32 时 | 否则 | 拟合出处 |
|---|---|---|---|---|
| `is_c600u_ap_count` | `maca_topk.cu:763-765` | — | — | — |
| `radix_block_for` | `:795-802` | 长行且 `batch ≥ kC600UNarrowRowBatches`(256) → **512 线程实例** | 长行 → 1024 线程实例 | 一台 28-AP C600U，`docs/experiments/20261010_c600u_narrow_row/` |
| `chunked_bf16_applies` | `:935-948` | `B² · kC600UChunkedVocabPerBatchSq(6400) ≤ V` | `B ≤ V / kChunkedVocabPerRow(21000)` | 一台 28-AP C600U，`..._c600u_split_crossing/` |
| `f32_coarse12_applies` 性能半边 | `:1578-1592` | C600U 梯子 `9216 / 49152` | `sm == 104` → C500 梯子；否则 **false** | C600U 梯子 2026-09-20（一台 28-AP）；C500 梯子出处戳 `kF32Coarse12MeasuredSmCount = 104` |
| `f32_chunks_applies` | `:1680-1682` | `sm != 104` → **false** | C500 段可入 | 出处戳 104 |
| `f32_chunks_large_batch` | `:1314-1317` | `sm != 104` → 2 | 按 `V` 推 chunk 数 | 出处戳 104 |

**这六条问的都是「调用面前这台机器长什么样」**：一个 wave 给某个块宽几个独立 CTA、
一个 batch 能不能填满一台机器、一条实测出来的交叉属于哪条梯子。AP 数回答这些；
`ARCH_FAMILY` 回答的是**产物**的问题。`csrc/structs.h:88-89` 自己给过这条规则
（「geometry compiles in; tuning still has to be earned」），`ffi_entries.h:44-51`
又给了一次（AP 数「是唯一剩下的、以参数形式传进来的、来自设备的输入」）。

### 2.3 自适应，本身不算族键

| 门 | 位置 | 读什么 |
|---|---|---|
| `wave_filled_chunks` | `:908-915` | 运行期 count，把 `6 × chunks` 对齐成整波 |
| `f32_chunk_work_target` | `:904-906` | `sm_count * 5 / 2`（2.5 是 C500 拟合） |
| `select_chunk_count` | `dg_chunks.cuh:459-461` | 参数化的 `num_sms` |

这三条**没有族分支**，所以「一族换一台机器」在它们身上自动成立；它们的风险是另一个
—— §5.3。

### 2.4 两条只有注释、没有把手的 C500 规则

| 规则 | 位置 | 状态 |
|---|---|---|
| `needs_long_row_bf16` 的 `L >= 60000` | `radix_core.cuh:2641-2650` | 注释自述「The measured crossover is specific to the 16 KiB C500 row variant」，但**没有任何族门或 AP 门**：三族都吃这个 C500 交叉点，C600U 只在 batch ≥ 256 时被 §2.2 那条**覆盖掉**，257 以下照旧走 C500 的交叉 |
| `chunked_f32_applies` 无 `sm` 测试 | `maca_topk.cu:1827-1843` | `csrc/structs.h:100-105` 声称如此，**已在代码核对**：这个函数从头到尾没有 `sm_count`，所以一个 1600 镜像按 **C500 的 chunk band** 走 |

---

## 3. keystone：C600 落在 C600U 的三条梯子上

### 3.1 证据链（四步，每一步都可核对）

1. `deep_gemm/utils/arch_config.py`（`_binding.py:88-89` 称它为「the repository's one
   table for this」）：**1500 = C600 / sm86**，1600 = C600U+N300U / sm87/88/89。
2. `_binding.device_family_suffix()`：`sm == 86 → "_xcore1500"`。所以一台 C600 加载的是
   `deep_select_maca_xcore1500.so`。
3. `csrc/structs.h:152-155`：family 1500 的 `ARCH_SM_COUNT = 28`。
4. `maca_topk.cu:763-765`：`is_c600u_ap_count(sm) = (sm == 28 || sm == 32)`。

⇒ 一台 C600 满足这个谓词，于是 **§2.2 里三条 C600U 梯子全部落在它身上**。
probe §2 把这一步打印成一行：`ARCH_SM_COUNT for family 1500 (C600) = 28` /
`=> a C600 satisfies it: True`。

并且这不是「只影响 1500」：`f32_chunks_applies` 要求 `sm == 104`，所以 C600 **拿不到**
C500 的 fp32 两条路；`f32_coarse12_applies` 在 C600U 臂不中之后撞上
`if (params.sm_count != 104) return false`，于是 C600 的 fp32 **只剩 row 路径**。
probe §3 的 `1500 / 28` 行是这张判决表。

### 3.2 一个镜像被另一族的设备调用时，同样满足它

`interface.py:196-201` 写明：`family_suffix()` 只保证**镜像与它的族匹配**，
**不保证每次调用都在挑它的那台设备上**。probe §3 里 `1000 / 28` 与 `1000 / 32` 两行
就是这种情况：一个 C500 镜像被一台 28-AP 设备调用时，
`radix_block_for` 与 `chunked_bf16_applies` 会走 C600U 臂；而 `f32_coarse12_applies`
**不会** —— 它的预算半边是 `if constexpr`，64 KiB 装不下 16384 B 的 arena，那条路在
C500 镜像里根本不存在。**这不是缺陷，是「两把半边各自独立」那条设计正在生效**：
几何进编译期（按镜像），调优还得出处（按机器）。它同时是 §4 里选钥匙的论据之一。

### 3.3 一处文档矛盾（记录，不修改）

`DeepSelect/CLAUDE.md:1175`：

> `MetaX C600` and `MetaX C600-U` are both xcore1600 but have different clocks, …

与 `arch_config.py`（1500 = C600 / sm86）和 `_binding.py`（sm86 → `_xcore1500`）**冲突**。
那句话是 `perf_data` 目录命名规则的论据，写在这条规则的上一句；按仓里唯一的族表，
C600 是 **xcore1500**。两处必须有一处是错的，而本记录的 §3.1 用的是 `arch_config.py`
（`_binding.py` 指名的那张表）。已报给 setup-family；本记录只记冲突，不改文档。

### 3.4 没有任何 C600 测量

```
ls perf_data/          -> MetaX_C500   MetaX_C600-U          （无 MetaX_C600）
ls docs/experiments/   -> 20261009_c500_dsa_decode
                          20261010_c600u_bperm_summary
                          20261010_c600u_narrow_row
                          20261010_c600u_split_crossing
                          20261010_skim_rate
                          20261012_setup_family_detection   （另一个 agent）
                          20261012_family_source_selection  （本记录）
```

每一条拟合出的梯子，其出处目录都叫 `c600u` 或 `c500`。**C600 一次都没上过台**。

### 3.5 这条结论在源码里的落点（并行发生，非本次）

本次工作期间，同一个结论已由另一个 agent 写进 `csrc/structs.h:91-98`（`git diff` 可见，
`tools/comment_gate.py --worktree` 判定它是 `IDENTICAL CODE`，即纯文档）：

> `is_c600u_ap_count` is an AP-count set, not a family -- it is
> `sm_count == 28 || sm_count == 32`, so it catches the 1500 row's 28-AP C600
> as well -- and the three ladders it admits all carry a C600U name … so a C600
> is served by a set fitted on its sibling.

**但谓词自己的那条注释（`maca_topk.cu:761-762`）仍是旧的**：

> A set, not a count: the C600U ships at 28 APs and at 32, and both are ordinary
> parts.  Every gate fitted on that family takes the set.

它说的是「那个族」，而谓词测的不是族 —— §4.3 的补丁正是改这一处。
两条注释现在一个对、一个不对，**读者会先撞上错的那条**（它就在函数头上）。

---

## 4. 两把候选钥匙的行为差与补丁文本

### 4.1 逐条差

| 谓词 | 钥匙 A（今天）：`sm ∈ {28,32}` | 钥匙 B：`ARCH_FAMILY == 1600` |
|---|---|---|
| 一台 **C600**（1500 / 28 AP）上的 `radix_block_for` | 长行且 batch ≥ 256 → **512 线程** | 长行 → **1024 线程**（C500 交叉点） |
| 一台 C600 上的 `chunked_bf16_applies` | `B²·6400 ≤ V`（C600U 曲线） | `B ≤ V/21000`（C500 直线） |
| 一台 C600 上的 `f32_coarse12_applies` | **C600U 梯子** 9216/49152 | **false**（两条梯子都不认 28） |
| 一台 C600 上的 `f32_chunks_applies` | **false** | **false** |
| 一台 **C600U**（1600 / 28 或 32 AP） | 三条 C600U 臂 | 同上（行为不变） |
| 一台 **C500**（1000 / 104 AP） | C500 全臂 | 同上（行为不变） |

**两把钥匙今天在 C500 与 C600U 上逐位相同**；差别**只出现在 C600 上**，
而三条臂在 C600 上**都没有测量**。

### 4.2 为什么分析偏向「保留 A、修掉它的声称」，而不是换 B

1. **六条门问的都是运行期机形，不是产物身份。** 换 B 等于用「这个 `.so` 是给谁编的」
   回答「这台机器现在长什么样」—— 正是 `structs.h` 与 `interface.py` 各自写过一次、
   说不要做的那件事（§2.2 末）。`is_c600u_ap_count` 的**类**是对的，错的是**名字**。
2. **B 不比 A 保守，只是换了一组外推。** B 把三条 C600U 外推换成：两条 C500 规则
   （块宽交叉、chunk band）**按产物身份**套到一台 28-AP 机器上，外加**两条路直接拒掉**
   （coarse12 / fp32 chunks 都不认 28）。这是「三处外推」换成「三处不同的外推 + 两次拒绝」，
   不是「猜」换成「量」。
3. **C600U 的三条臂只能靠 A 到达**（B 也能到，A∧B 也能到；见下）。真正需要保住的是
   「在 C600U 硬件上能走到这三条臂」—— 这是它们唯一的实测来源。
4. **A∧B 是第三种写法**，值得记一笔：`ARCH_FAMILY == 1600 && (sm ∈ {28,32})`。
   它保住 C600U（1600 镜像 + 28/32 AP → true），把 C600 推走，同时保住「AP 数是这台机器
   的事实」这半条理由。代价是它给 1500 引入一条**新**分支，而这条分支在 C600 上同样没量过 ——
   和 B 的问题一样。

**结论（交给用户裁定，本记录不落地）：保留 A，把它的注释改成它真正在说什么**，
并把「C600 落在三条梯子上且从未测量」写成显式的开放问题（§5）。

### 4.3 补丁文本（建议，**未落地**）

```diff
--- a/csrc/maca_kernels/xcore1000/maca_topk.cu
+++ b/csrc/maca_kernels/xcore1000/maca_topk.cu
@@ -760,6 +760,37 @@
-// **A set, not a count: the C600U ships at 28 APs and at 32, and both are
-// ordinary parts.**  Every gate fitted on that family takes the set.
+// **A machine shape, not a family: this is the pair of AP counts the three
+// C600U-class ladders below were fitted on, and `csrc/structs.h` says family
+// 1500 (a C600) reports 28 -- so a C600 takes them too.**  Every gate that
+// reads this is asking what the machine in front of the call looks like: how
+// many independent CTAs a wave gives each block width, how a batch fills a
+// split, which measured ladder a crossing belongs to.  The device's AP count
+// answers those; `ARCH_FAMILY` answers a question about the *artifact*, and
+// `interface.py` says in as many words that the loader guarantees only that
+// the image matches its family at load time, not that every call runs on the
+// part that picked it.
+//
+// **No C600 has been measured.**  `perf_data/` holds a `MetaX_C500` and a
+// `MetaX_C600-U` and no `MetaX_C600`; every experiment that fitted one of these
+// ladders is named `c600u`.  So a C600 inherits all three by this predicate and
+// nobody has checked that it should -- an open question, not a settled one.  A
+// routing change is measured on the part it targets (CLAUDE.md, "Performance-
+// change discipline") and there is no C600 on this box.
+//
+// The alternative key is not the conservative one.  `ARCH_FAMILY == 1600` would
+// give a C600 the C500 block-width crossover and the C500 chunk band by
+// artifact identity, and deny it coarse12 and the fp32 chunks outright (no
+// ladder names 28 APs): three extrapolations replaced by three different
+// extrapolations plus two route denials.  See
+// `docs/experiments/20261012_family_source_selection/`.
 inline bool is_c600u_ap_count(uint32_t sm_count) {
     return sm_count == 28 || sm_count == 32;
 }
```

若裁定走钥匙 B，补丁是同一处一行（并需要同步 §4.1 里四行的注释）：

```diff
 inline bool is_c600u_ap_count(uint32_t sm_count) {
-    return sm_count == 28 || sm_count == 32;
+    return ARCH_FAMILY == 1600 && (sm_count == 28 || sm_count == 32);
 }
```

---

## 5. 开放问题（需要一台 C600）

**没有一条能在本机做** —— `ls /opt/maca`、四台设备全是 `MetaX C500`、APs=104。
按「Performance-change discipline」，这些是**必须在它针对的部件上量**的：

| # | 问题 | 需要什么 |
|---|---|---|
| 5.1 | `radix_block_for` 的 `kC600UNarrowRowBatches = 256` 在 C600 上是否成立 | 一台 28-AP C600，`20261010_c600u_narrow_row/` 那套 copy-and-point A/B，交叉点重扫 |
| 5.2 | `kC600UChunkedVocabPerBatchSq = 6400`（sqrt 曲线）在 C600 上是否成立 | 同上，`20261010_c600u_split_crossing/` 的协议 |
| 5.3 | `f32_coarse12_applies` 的 C600U 梯子（9216/49152）在 C600 上是否成立 | 同上；注意它的理由「两侧都是一 CTA 一行，所以每条行的代价不随 AP 数变」是关于 AP **数**的，不是关于**部件**的 —— C600 与 C600U 的时钟/带宽可能不同 |
| 5.4 | §2.3 的 `f32_chunk_work_target`（`sm*5/2`）与 `wave_filled_chunks`：`ledger §12.11` 已记「没有测量可以判断它对不对」，**至今仍然没有** | 同上 |
| 5.5 | §2.4 的 `L >= 60000`（自述是 C500 的交叉点）在 128 KiB 家族上是否该另有一处 | 一台 C600U 或 C600；这一条对 C600U **已经**在服役（只是被 §2.2 覆盖了 batch ≥ 256 那一段） |

**还有一个不需要 C600 就能做、但本次没做的**：§3.3 那处文档矛盾（`CLAUDE.md:1175`）
必须二选一改掉，否则下一个读族表的人会以为 C600 是 xcore1600。

---

## 6. 落地：`SOURCES_BY_FAMILY`（布局侧，本次唯一的代码改动）

### 6.1 改了什么

`setup.py` 原来是一条无条件常量：

```python
SOURCES = ["csrc/maca_kernels/xcore1000/maca_topk.cu"]
```

现在是一张族表 + 一个默认值 + 一个取值函数：

```python
SHARED_SOURCES = ["csrc/maca_kernels/xcore1000/maca_topk.cu"]
SOURCES_BY_FAMILY = {
    1000: SHARED_SOURCES,   # C500
    1500: SHARED_SOURCES,   # C600
    1600: SHARED_SOURCES,   # C600U -- the shared kernel, not xcore1600/
}

def sources_for_family(family):
    return list(SOURCES_BY_FAMILY.get(family, SHARED_SOURCES))
```

Extension 侧由 `sources=SOURCES` 改为 `sources=sources_for_family(target_family)`；
`print` 那一行改为从表里推（**输出逐字节不变**，见 §6.2）。docstring 里两处 `SOURCES`
的提法同步改掉。

**为什么是三行显式而不是一行 + 默认**：这张表要回答的是「`deep_select_maca_xcore1600.so`
里装的是哪份源码」，最有用的答案是**那一行**而不是一次推断；默认值留给以后加进
`_family_of_target` 元组的族。**而且 1600 这一行必须显式**，因为它要挡住一个名字陷阱：
`family 1600`（C600U）与目录 `csrc/maca_kernels/xcore1600/`（ported upstream 那棵，
本构建不编）是**两件不同的事**，一个读者几乎一定会把它们当成一件。注释里写死了这一点。

`sources_for_family` 每次返回一个新 list，所以三个 `Extension` 不再共享同一个 list 对象
（今天没有活的缺陷 —— 这个 torch 的 `CUDAExtension` 在非 HIP 路径上只读 `sources`
—— 但共享可变对象是下一次改动的坑，一行 `list(...)` 关掉它）。

### 6.2 收据：今日行为可证明不变

**三次** `rm -rf build && ./develop.sh` —— 改动前、改动后、以及两个 agent 都收工之后
在**最终落定的树**上再跑一次（`build_receipts.txt` §E）。三次都是**同一台机器、同一条
命令、同一个目标列**（`CUCC_TARGETS=xcore1000,xcore1500,xcore1600`，每份日志的第 2 行
各自记着）。三次都由 `develop.sh` 自己 `rm -rf build`，所以不存在增量复用的可能：

| 检查 | 改动前 | 改动后 | 最终落定树 |
|---|---|---|---|
| `deep_select: compiling …` 那一行 | `… xcore1000/maca_topk.cu once per family, for xcore1000,xcore1500,xcore1600` | **逐字节相同** | **逐字节相同** |
| 6 条编译/链接命令行（归一化后 `diff`） | — | **逐字节相同**（3× cucc + 3× gnu） | **与两者都逐字节相同** |
| artifact 数 | 3 | 3 | 3 |
| artifact 字节数 | 6464392 / 6468488 / 6468488 | **相同** | **相同** |
| `nm -S --defined-only … \| grep -c topk_kernel_radix` | 60 / 60 / 60 | **60 / 60 / 60** | **60 / 60 / 60** |
| `readelf -d \| grep -iE 'libtorch\|libc10'` | 空 | 空 | 空 |
| `nm -D \| grep -c __tvm_ffi_` | 2 / 2 / 2 | 2 / 2 / 2 | 2 / 2 / 2 |

第三列的口径要写清楚：setup-family 的另一处改动改的是**接受哪些目标拼写**以及拒绝时
说什么，对一组合法的目标列来说它在命令行里**看不见**。所以「与两者都逐字节相同」
证明的是**落定树对默认目标列编出来的东西与 HEAD 逐字节一致**，不是「另一处改动在它
自己的错误路径上无害」。

**收据之后 `setup.py` 又动过，而改动不改这一列的命令行 —— 但那是推理，不是重测。**
写清楚是因为上面那三列的命令行取自一个更早的文件（本记录只钉住一件事：**本记录
不重跑**，重测由落地那次 commit 自己的收据承担，落地前的最后一次改动后 md5 是
`7a0c2c4b…`）。下面两条是收据之后才进文件的：

| 之后的改动 | 何时才生效 | 为何在「explicit 且无重复」这一列上不可见 |
|---|---|---|
| `native` 被接受、解析为本机族 | 只有拼写**字面是** `native` 时 | `_offload_spelling`（`setup.py:160`）对非 `native` 拼写**原样返回**，`-DDEEP_SELECT_ARCH` 与 `name=` 取的是解析出的族，而 `xcore<N>` 解析出来就是它拼写里那个数 —— 显式列上 `native` 那条路径一步都不执行 |
| 同族去重（两条目落到同一族时合一次） | 只有两条目**解析到同一族**时 | 三个拼写是三个不同的族：没有条目被丢、没有打印、`resolved` 与原列表逐项相同 |

第一条（`native`）由 `docs/experiments/20261012_setup_family_detection/` 记。第二条是
**落地前在本树上抓到的一个缺陷**，记在这里只是因为 §6.2 的读者需要知道文件动过：
去重原本只重写了 `targets`，而 Extension 列表与源码集汇总读的是 `resolved`
（修好后的 `7a0c2c4b…` 上分别是 `setup.py:512` 与 `:517`）—— 于是
`CUCC_TARGETS=native,xcore1000` 会先打印 `building it once`、再构造两个同名同输出
路径的 `Extension`，日志还在同一行写着 `for xcore1000`。
team-lead 在冻结的树上自行修正：`resolved` 改由去重循环产出、`targets` 从它派生。

`.so` 的 md5 **会变**，而这**不是**行为变化的证据。本次顺带把「为什么」测准了
（`build_receipts.txt` §D），因为原来的说法比实际粗：

- **编译是确定性的**：拿构建自己的那条 cucc 命令跑四次（两次改 `-o`、两次原样），
  `maca_topk.o` 四次都是 `c7b9b7b3…`、6614560 B、`cmp -l` 为空。
- **链接不是**：拿构建自己的那条 `gnu` 命令，用**同一个 `.o`** 链两次，
  两个 `.so` md5 不同、字节数相同（6464392）；`cmp -l` 给出 **6 字节**，
  单簇，偏移 `0x623a06..0x623a0b`，周围的字符串是
  `…__init_array_entry | ds_link_probe-ac125e.cpp | __FRAME_END__…` —— 整个差异
  就落在一个生成出来的中间文件名里。
- **那个名字是链接阶段现造的**，后缀 6 位十六进制**每次调用随机**：同一次构建的三个
  artifact 分别是 `maca_topk-d1037b.cpp` / `-b7550c.cpp` / `-221bd9.cpp`；
  而**编译产物 `.o` 里根本没有任何 `.cpp` 字符串**。名字的干来自喂给链接的那个 `.o`
  的文件名（`maca_topk.o` → `maca_topk-<hex>.cpp`；对照实验里链
  `ds_link_probe.o` → `ds_link_probe-ac125e.cpp`）。

所以本次凭据是**命令行逐字节相同**（⇒ 同一个 TU、同一组 flag、同一份 host 代码）、
**artifact 字节数相同**、**符号计数相同**、`readelf` 干净，**不是** md5。

**这条比仓里原有记录更细，值得记一笔**：`CLAUDE.md` 的 Build 节记过同形的 6 字节差异
（`0909cc25…` → `4b29550a…`，`0x3a2af8..0x3a2afd`，落在 `…maca_topk-02b752.cpp` 里），
但没分是哪个阶段。**是链接**。推论：比较**编译产物 `.o`** 比比较链接出来的 `.so`
是更强的凭据 —— 它把唯一那个不确定输入消掉了。

原始材料随本记录入库：`build_before_raw.log` / `build_after_raw.log`（两次
`./develop.sh` 的完整输出，ninja 打的每一条命令行都在里面）、
`cmds_before_raw.txt` / `cmds_after_raw.txt`（从中抽出的 6 条编译/链接命令）、
以及汇总与推导 `build_receipts.txt`。

### 6.3 一个 `.o` 路径、三次编译 —— 三族不会共用陈旧产物

读构建日志的人会看到一个**看起来像缺陷**的东西：三个按族的 Extension 写的是**同一个**
目标文件路径

```
build/temp.linux-x86_64-cpython-310/csrc/maca_kernels/xcore1000/maca_topk.o
```

—— 那里面的 `xcore1000` 是**源码所在的目录**，不是按族的产物目录。所以那个担心是
合理的：1500 会不会链到 1000 的 `.o`？

不会，而且这件事**在同一份日志里就能查**，不用论证：每次构建都打出**三行独立的**
`[1/1] cucc …` 编译命令，靠 flag 区分：

```
[1/1] cucc … -offload-arch=xcore1000 -DDEEP_SELECT_ARCH=1000 … -o …/maca_topk.o
[1/1] cucc … -offload-arch=xcore1500 -DDEEP_SELECT_ARCH=1500 … -o …/maca_topk.o
[1/1] cucc … -offload-arch=xcore1600 -DDEEP_SELECT_ARCH=1600 … -o …/maca_topk.o
```

命令行是 ninja 认定目标身份的一部分，所以 `-offload-arch` / `-DDEEP_SELECT_ARCH`
一变，就往那条共享路径**重编一次**，不会把上一族的产物拿来复用。同一事实的另一端是
三个 `.so` 带着三份不同的设备镜像（以及两个不同的字节数）。
（这条是 setup-family 先注意到的，本记录用上面两份日志独立复核过：
`grep -oE '\-o \S+/maca_topk\.o '` 只剩一个路径，
`grep -oE '\-offload-arch=xcore[0-9]+'` 是三个。）

### 6.4 最终落定树

两个 agent 都收工之后，在落定树上又跑了一整轮 `rm -rf build && ./develop.sh`
（`build_receipts.txt` §E）：**6 条命令行与「改动前」「改动后」两次都逐字节相同**，
三个 artifact 字节数相同，`nm` 符号计数 60/60/60，`readelf` 干净，`__tvm_ffi_` 2/2/2，
rc=0、2m40s。另有一项不需要编译的检查：`setup.py --version` 走完整个目标循环、
三族的源码解析逐族相同、每次返回新 list、未列出的族落回默认、且没有任何一行指向
`xcore1600/`。

### 6.5 本次**没有**落地的

- §4.3 的族门补丁：**未落地**，理由见 §4.2 与 §5（无 C600 可测）。
- `csrc/maca_kernels/xcore1600/` 那棵：**仍然不可达**，`SOURCES_BY_FAMILY` 里没有任何
  一行指向它，`include_dirs` 也没有它的 `kerutils`。这是**有意的**（§1），
  本表把「想给某族它自己的 kernel」变成一次数据编辑，**没有**把那条路打开。

---

## 7. 复现

```bash
cd /home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
python3 docs/experiments/20261012_family_source_selection/probe.py     # §1/§2/§3 的表

# §6.2 的构建收据（~3 min，三族；会删掉 in-place 的 .so 再重建）
rm -rf build && ./develop.sh
nm -S --defined-only deep_select/deep_select_maca_xcore1000.so | grep -c topk_kernel_radix   # 60
readelf -d deep_select/deep_select_maca_xcore1000.so | grep -iE 'libtorch|libc10'            # 空
```

**probe 能证明什么、不能证明什么**：它把「这段代码走哪个分支」从树里读出来并判定
（这是 §2/§3 需要的全部），但它**没有设备、也没有 C600/C600U**，所以对
「哪条臂更快」一个字都不说 —— 那正是 §5 全部留空的原因。

---

## 8. 相关记录

- `docs/experiments/20261012_setup_family_detection/`（另一个 agent）：`CUCC_TARGETS`
  能写什么、mxcc 接哪些拼写、`native` 为什么被拒。**本记录不重复它的结论**，
  只在 §6 用它的前提（一族一份 artifact 的名字由族决定）。
- `docs/experiments/20261012_family_architecture/`（另一个 agent）：同一轮里把
  §3.3 那处文档矛盾与 `chunked_f32_applies` 无族门这两件事在文档面收口（只动注释，
  `comment_gate` 判 `IDENTICAL CODE`）—— §3.5 引的就是它的落点。
- `docs/experiments/20261010_c600u_narrow_row/`、`..._c600u_split_crossing/`：
  §2.2 前两条的拟合协议与原始表 —— §5.1/§5.2 要重跑的正是这两套。
- `docs/C500-radix-perf-ledger.zh.md` §12.11/§12.12：`kF32Coarse12MeasuredSmCount`
  与「族常数回答产物、`sm_count` 回答运行」这条规则被定下来的地方。
- `csrc/structs.h` 的「the per-family build constants」章（基准版 `:35-171`）：族表本身，
  以及它对「哪一半该进编译期」的完整论证。
