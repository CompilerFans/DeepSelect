# C500 行式 radix TopK：交接手册

写给接手的人（很可能是几周后的我）。这份手册只讲**现在**：代码在哪、做到
哪一步、怎么验、哪里会踩坑、下一步有哪些路。历史账见
`C500-radix-perf-ledger.zh.md`。

---

## 0. 仓库身份（先看这一条，否则一定踩坑）

这里是 **`third-party/DeepSelect`**，一个**独立的 git 仓库**（`origin` 指向
自己的 remote，分支 `main`），外层 `mcDeepGEMM` 把它当 submodule 引。

- 目录里做 `git status/log/commit/push` 操作的是 **DeepSelect**。
- 但外层仓库的根**也是**一个 git 工作区，而且外层仓库里还有别的未提交改动
  （sparse MQA 那摊）。**一次 `commit -a` 跑错目录，就会把外层的改动收进
  DeepSelect 的 commit 里**——本次工作真的发生过一次，靠 `git reset --mixed
  HEAD~1` 才收回来。
- 因此：**每条命令都显式写 `git -C <DeepSelect 绝对路径>`，或者每步都先
  `pwd`**。`cd` 不跨 Bash 调用保持（工具会把 cwd 重置回主工作目录）。

---

## 1. 当前状态（2026-10-10）

| 项 | 值 |
|---|---|
| 分支 | `main` |
| 内核 HEAD | `ae26b0d` *the merge ranks with the coarse12 row too, not only stage 1* |
| 文档 HEAD | 本文件所在提交 |
| 工作树 | **干净**（`git -C <DeepSelect> status --porcelain` 为空） |
| 产物 | `deep_select_maca_xcore1000.so`，md5 `bb45b940`（`./develop.sh`，三族全建，`BUILD_RC=0`） |

**两条门**（2026-10-10，device 2，独占）：

- 官方性能表（`tests/test.py --perf-only --dtype bf16 -nc`）：**117/117**
- 正确性样本（`run_test.sh --test`）：**All 339 cases passed**，收据
  `results/deepselect_run_20261010_141501.txt`

`ae26b0d` 之后只有文档落地，内核一行没动，所以这两条门就是当前产物的数值。
**§9 的排序已按 2026-10-10 重排**——先读它。

**decode 小 batch 长行的战役**（2026-10-09/10）已收尾：split 的两个 stage 都换成
coarse12、NaN 扫描并进 stage 1 自己的走法、门按实测重指（`kChunkedVocabPerRow`
= 21000）。十格累计 **861.2 → 692.4 µs**，结果与判定在 ledger §14，过程、探针与
被判死的方向在 `docs/experiments/20261009_c500_dsa_decode/`（§9.12–§9.17）。
它留下的下一步在 §9 第 1 条：**替换数据流，不是调参**。

**C600U 侧的两笔（2026-10-10，`39a85d1` / `ca59be3`）**：`maca_topk.cu` 加了两条
**C600U 专属**的运行期门，都在 `is_c600u_ap_count`（28/32 AP）上开、都在
104-AP 的 C500 上走老分支：长行块宽在 `b ≥ 256` 改走 512 线程实例（**C600U 官方
网格 −4.7%**，短 vocab 档最高 −17.7%）；bf16 split 的交叉按 `b² · 6400 ≤ vocab`
重指（C600U 上它随 sqrt(vocab) 长，不是随 vocab；decode 形状 b=8..24 由慢
1.06–2.2x 转正）。证据、探针与门在 `docs/experiments/20261010_c600u_narrow_row/`
与 `docs/experiments/20261010_c600u_split_crossing/`。
**C500 的行为不变，但源码变了 ⇒ xcore1000 重新构建后 md5 必变**（mxcc 嵌行号）。

---

## 2. 这个算子在干嘛（一句话版）

`topk(x, k)`：每行取 k 个最大值的下标（可选返回值/排序/窗口/NaN 语义）。
C500（xcore1000）上走的是**行式 radix 选择**：一行一个 CTA，两趟读趟行，
先粗层直方图定阈值，再在阈值桶里 refine。

代码位置：

| 文件 | 角色 |
|---|---|
| `csrc/maca_kernels/xcore1000/maca_topk.cu` | **契约层 + 启动层**：`topk` 的对外契约（values / sorted / NaN / 窗口 / 偏移 / 短行）、NaN 扫描、动态 smem 尺寸与布局、kernel 启动 |
| `csrc/maca_kernels/xcore1000/radix_core.cuh` | **选择数据流**（header-only）：`radix_topk_row_bf16_b`（16 位行，生产路径）、`radix_topk_row_f32*`（32 位行）、若干 static-k / register / chunked 变体 |
| `csrc/structs.h` | `TopkSelectArgs` / `RowParams` |
| `scripts/official_slice.py` | 官方大表的切片驱动（正确性门）——**2026-09-16 已删除，能力并入 `tests/test.py`**（见 §4 正确性门） |
| `tests/test.py` | 官方性能表（每个 perf 用例先查正确性再计时） |

**这条数据流当前的耗时归因在 `C500-radix-profile.zh.md`**（2026-09-12，
增量消融）。摘要：`b4096-v16384-k512` 满内核 619.6 µs，其中 pass 1 整段 253.3
（它的 8 个共享原子/uint4 占 **102.5**，是已量出的最大单项）、pass 2 的 walk
132.1、emit 原子 36.8、staging 原子 35.7、refine 块 57.3（含屏障等待）。两个
walk 合计 283 µs，对两趟下界 233.4 = 82% 可达速率。要看"下一步动哪里"先读
那份文档的 §5。

**哪条路径在生产里活着**：`maca_topk.cu` 只启动两条行内核——
`radix_topk_row_bf16_b<BLOCK>`（bf16）与 `radix_topk_row_f32`（fp32）。
`radix_core.cuh` 里的 `radix_topk_row_bf16_k` / `_reg` 这些**不由 maca_topk.cu
启动**，只被 `rk::launch_topk_bf16_chunked`（低 batch 长行的 split/merge）用
到。改 `_b` 之前先确认自己改的是不是生产路径。

---

## 3. 现在的数据流（coarse12 之后）

`radix_topk_row_bf16_b<BLOCK>`，一行一 CTA：

```
块 0    arena 区（= 动态 smem 头部）当 4096 桶直方图用，清 0
趟 1    遍历整行：hist[key >> 4]++          （uint4 一次 8 个元素）
折半    4096 -> 256：s_histogram[b] = Σ 16 个 sub-bin（每个高字节一个 bin）
扫描    run_cumsum_warp 对 256 桶做 reverse 后缀和，定位"高字节"
收窄    thread 0 在该字节的 16 个 sub-bin 里从高到低找跨界，得到 12 位阈值
        （s_wide_above = 严格在阈值之上的元素数）
趟 2    遍历整行：key>>4 > 阈值 -> 直接写 output
                  key>>4 == 阈值 -> 存进 arena + 统计低 4 位（16 桶细直方图）
屏障    趟 2 结束的 __syncthreads()（:1369）——细直方图到此才对所有线程可见
refine  细直方图再做一次 256 宽扫描（只前 16 桶非零），得到 4 位细阈值
         （扫描后还有一道 __syncthreads()，:1383，因为阈值由 thread 0 写）
输出    arena 内：低 4 位 > 细阈值 -> 输出；== -> 用 s_last_remain 补尾
溢出    arena 放不下（s_num_input > 4096）才回退整行重扫一遍
```

**两个设计要点，改之前必须理解**：

1. **直方图与 arena 共用同一段 smem**（`s_wide = s_input_flat`）。直方图在折半
   + 收窄之后已死，arena 才开始写；中间有 `__syncthreads()` 隔开。所以
   `radix_smem_bytes(topk, sorted, wide=true)` 的 lead 是**直方图的 16 KB**，
   不是 arena 的 14,056 B。`maca_topk.cu` 的 `radix_layout` 必须和它用同一个
   flag 算 `selected` 的偏移，否则 arena 会盖住 `selected`。
2. **计数是"桶"的属性，不是"装得下多少"的属性**。趟 2 里细直方图对阈值桶的
   **每一个**成员计数，只有写入 arena 才受容量限制。历史上有过把两者一起夹在
   `pos < SMEM_INPUT_SIZE` 里的写法，代价是溢出时必须整行重扫来修——已删。

**收窄那一步有一处"错了不会报错"的语义**：跨界判据是
`above + c > remain_topk`——**窗口本身**，不是"扣掉上面那个字节之后还剩多少"。
用后者会在字节内有富余时提前一个 sub-bin 收手，窗口就短了，refine 找不到落点，
读到未初始化的共享变量。本次为此吃过一次 memory violation + 379/512 个错槽。

---

## 4. 怎么构建、怎么验

### 构建

```bash
cd /home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
rm -f deep_select/deep_select_xcore1000*.so \
      build/lib.linux-x86_64-cpython-310/deep_select/deep_select_xcore1000*.so
python setup.py build_ext --inplace
```

**`rm` 那一步不能省**：`build_ext --inplace` 看时间戳决定要不要拷贝，目标 `.so`
比 `build/lib` 里的新就**静默跳过**拷贝——你会拿着旧二进制测半天。本仓库的
`.so` 在 `deep_select/` 下且被 gitignore，天然会陈旧。

`.so` 的 md5 对**源码换行都敏感**（含源行信息），可以拿来当"我测的到底是哪份
源码"的凭据：同源码两次构建 md5 相同（本次验证过）。

### 正确性门（官方大表）

```bash
cd /home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
PYTHONPATH=$PWD python tests/test.py --backend maca_c --sample 1000000 -rf
```

> **路径已迁移（2026-09-16）**：本节原来跑 `scripts/official_slice.py
> --backend maca_c --sample 1000000 --shard i/4`，×4 串行。那个驱动已删除，
> 它的 `--backend` / `--seed` / `--sample` 并进了官方 `tests/test.py`；
> `--shard` 没有跟过来。因此现在是**一个进程跑全表**，本手册下方与
> `C500-radix-perf-ledger.zh.md` §5 里那条 `--shard` 命令行是**当时的记录**，
> 保留原样（账本不该被改写成一条没产生过那些数字的命令）。
> 重启能力随之消失：账本 §5 记过两次 `dumped core`，`--shard` 是当时唯一的
> 缓解手段。**长跑再出现 core dump 时，第一件事是把 `--shard` 加回来。**

全表 82,170 例。**耗时以当天实测为准，别照抄本文的数字**：同样这 20,543 例的
一个 shard，本文 §4 记的是 `~340 s`，账本 §5 后期几条记的是 **1,730–1,917 s**，
两者差 **5×**，且都没有记当时的机器状态；全表单进程实测**还没跑过**。按
**数小时**规划。<br>
**不要开 8 个并发**——8 个并发 torch 进程会让 CUB 的 onesweep radix sort 把
设备打挂（实测，且会连坐整机）。串行跑也一样快，因为瓶颈在每例的构造。

### 性能门（官方表）

```bash
PYTHONPATH=$PWD python tests/test.py --perf-only    # 95 个用例，先验后计时
```

单跑一条也可以：`tests/test.py --perf-only --dtype bf16`。

### A/B（同 session、同种子、同口径）

官方性能表的用例**按全局计数器取种子**，两次运行数据不同，所以做 A/B 要自己
钉种子。最省事的模板：用 `tests/lib.py` 的 `TestParam(B, V, K, ...)` 造例、
设 `p.seed`、`lib.generate_testcase(p)`，然后用 `tests/kernelkit` 的
`kk.bench(fn, p.num_runs)` 取 **kernel time**（不是 e2e）。

**基准二进制要可复现**：`git stash` 到要对比的那棵树 → 按 §4 构建 → 存下
`.so` → 再换回来。本次就是这么拿到 `ea8bcb0` / `4cd740a` 两侧的。

---

## 5. 环境陷阱清单（按踩到的顺序）

1. **`torch.set_default_device("cuda")` 必须写**。`tests/lib.py` 的
   `generate_testcase` 不设的话造出来的是 CPU 张量，扩展会解引用主机指针，
   报的是 `Xnack Error / ATU Fault`——看起来像内核有 bug，其实是探针自己的锅。
2. **bench 必须独占 GPU**。跑之前先 `pgrep -af python` 确认没有别的 torch
   进程；与 `test_unit` 并行会产出 16 个假回归。
3. **`cd` 不跨调用保持**，工具会把 cwd 重置回主工作目录。所有路径写绝对
   路径，或者每条命令都 `cd` 一次。
4. **内核对齐要求**：行长 8 的倍数 + 16 B 对齐才走向量路径，否则走标量兜底
   （`bf16x8_is_aligned`）。混合向量+尾部在 1024 线程块上曾有间歇性竞态，
   所以奇长度行统一走一种装载模式。
5. **mxcc 的编译缓存**在 `~/.deep_gemm/cache`（按条目名 + 源码摘要），
   换源码只影响改动的那条，不会全量重编。
6. 官方表的 `NormalFloatDistribution` 数据**不是** bit-pattern 均匀。这一条
   曾经被写错并进了 commit message（`ea8bcb0` 末段），已在 `4cd740a` 里更正：
   官方表自己的数据在 8 位粗层下**会**溢出（b4096/L16384/k512：阈值桶 4,686
   vs arena 3,514，一行 28% 落在同一个粗桶里）。

---

## 6. 历史账（一行一个，细节见 ledger）

| commit | 做了什么 | 官方 b4096-v16384-k512 |
|---|---|---|
| `ea8bcb0` | staging 缓冲按 `topk` 定尺而非 `kMaxTopK`（occupancy 1→2） | 2535.8 → 1588.3 µs |
| `4cd740a` | refine 的直方图按"桶"计数，不再被 arena 夹住；删掉整行 rebuild，emit 趟向量化 | 1588.3 → 1034.3 µs |
| `3f8dfe7` | 粗层 8 → 12 位（直方图 alias 在 arena 上） | 1034.3 → **650.7 µs** |

累计（起点是 `ea8bcb0` 之前那份二进制）：该格 **2533.9 → 650.7 µs（3.89x）**，
`b4096-v262144-k512` **22815.5 → 6471.5 µs（3.53x）**。对 torch 现在是
1.67x ~ 4.54x。

---

## 7. 提交的形状（照 `3f8dfe7` / `4cd740a` 抄）

每一条性能/数据流改动的 commit message 都是这个骨架，缺任一项不算记录：

- 标题一行祈使句，说清**原理**（不是"优化了 X"）；
- 为什么：老做法的代价，带实测数字；
- before→after 表（7 个代表 cell，µs + GB/s 双币种），并注明两个数各是什么口径、
  两侧二进制分别来自哪棵树；
- **roofline 判定**：这格是不是带宽受限——逻辑单趟 GB/s 乘趟数，对 **1,650 GB/s**
  只读墙的占比（不是 1,487：见 `C500-to-parity-plan.zh.md` §7.1 的更正）；
  不是的话写清绑定在什么上；
- 门的结果（95/95 + 82170/82170）；
- 架构边界声明（只动 `csrc/maca_kernels/xcore1000/` ⇒ xcore1600 逐字节不变 ⇒ 不欠 C600U 验证）。

落地顺序：

```bash
D=/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
git -C $D status --porcelain          # 确认只有你想提交的文件
git -C $D commit -a -F /path/to/msg   # 或先 git -C $D add <paths> 再 commit
git -C $D push origin main
```

**不要用裸 `git commit -a`**（见 §0）。

---

## 8. 排障顺序（症状 → 先看哪）

| 症状 | 先查 |
|---|---|
| 结果错，但错得不多（几个 slot、rank 差几个） | 阈值的选择判据；大概率是"扣减口径"用错（§3 末尾那条） |
| `Memory Violation(0x4)` / `ATU Fault` | 先查探针是不是漏了 `set_default_device`（§5.1）；再查阈值是否越界，越界会让共享变量未初始化 |
| 整格变慢、其他格不动 | occupancy 掉了：算 `静态 2328 + 动态` 有没有过 32,768 |
| 换源码后行为没变 | `.so` 陈旧（§4 的 `rm`） |
| 编译期 `undeclared identifier` | `radix_core.cuh` 里常量与 helper 的**先后顺序**（helper 用了后面才定义的常量） |

定位阈值类问题最快的办法：**把阈值塞进输出尾部再读回来**。把 kernel 算出的
`{wide_threshold, above, fine_threshold, remain, num_staged, last_remain}` 在
`blockIdx.x == 0 && tx < 8` 时写进 `output[topk-8+tx]`，再用 torch 在 CPU 侧
算同一行的真值比对——本次就是靠这个一步定位到"选了 0xBFA，真值 0xBF9"。
用完记得删干净（`rg RKPROBE` 应该是 0）。

---

## 9. 下一步的候选（2026-10-10 按**实测**重排）

> **2026-09-16 补一条外部参照（ledger §8）**：fp32 行不是只有"离墙多远"这一个
> 坐标了。在 `deep_gemm` 能服务的 12 个 fp32 shape 上，它的 `topk_coarse12`
> 比本行的选择段快 **~2×（大 batch）/ 4.5–5.5×（小 batch）**（两边都去掉 NaN
> 扫描后测的）。
>
> **归因查过了，不要按"我们读两趟"去改**（这条是写本节时先猜、随后被源码否掉的）：
> `b4096-v131072` 上 `launch_topk_f32_chunked` 对输入行只有**一趟**——
> `topk_f32_chunk_stage1_kernel` 走 chunk（`radix_core.cuh:2404`，读 `scores`），
> `stage2` 只在候选缓冲 `vals`/`cols` 上跑 `radix_topk_row_f32`（`:2489`，不碰行）。
> 我们那多出来的一趟是 **NaN 扫描**（ledger §8 的 A/B：1612 µs，恒定 ~20%）。
> 也就是说：**单趟对单趟，我们 334 GB/s，对方 659 GB/s**（2.147 GB / 6425 µs
> vs / 3260 µs，1,650 GB/s 墙的 20% vs 40%）。差在那**每元素的活**上
> （stage1 的阈值/staging 原子，见 §3 与 profile §5），不在趟数上。
> 动它之前先按 ledger §8"未做的部分"把小 batch 那三格拆开。

排序依据：ledger §12.23/§13/§14 与本仓 profile。**原第 1 条（pass 2 尾活段
归因）与原第 4 条（chunked 的 static-k 行）都已结**，从候选移到下面的"已结"。

**已结（不要再当待办）：**

- **pass 2 的尾活段归因**（原第 1 条）。decode 战役的相位阶梯把它拆完了
  （`RK_CUT` 阶梯，len 8192、高斯、1024 线程）：发射 3.32 → pass 1 +0.84 →
  阈值解析 +1.82 → **collect +5.03** → 细直方图 +1.27 → emit +1.63 µs。
  **没有单一大项可删**；collect 最大，但它的成本既不是原子冲突、也不是原子
  发射次数（见"别做"第 1 条）。
- **chunked 路径里的 static-k 行**（原第 4 条）。两个 stage 都改走 coarse12 了
  （§9.12 的 stage 1、§9.15 的 merge；后者 10/10 格更快、合计 0.804×）。
  `_k` 现在只剩静态 k 行核那两条宏在用。

**开放：**

1. **把 `_b` 的两趟合成一趟——替换数据流。** 这是唯一能动"每次调用"成本的方向：
   实测 `_b` 的开销按**调用**计而不是按 kernel 计（同一个 kernel 里调两次
   = 24.06 µs，发两个 kernel = 25.24，合核只省一次发射的 1.2 µs），所以
   decode split 现在带着 **~30 µs 与形状无关的地板**（merge 19.6–20.0 平、
   契约半 5.2–6.0 平、host 5.3–5.6 平），b=1 上占 55%。§12.24 已按实测否掉过
   上游那套**寄存器常驻**单遍（它的常数前提是编译期 `topk`，与本核的运行期
   `topk` 不同源）——所以这条路要先回答"阈值从哪来"，不是照搬。
2. **pass 1 的每元素共享原子（95–110 µs，§12.23）**：仍是内核里最大的已识别
   单项。原设想的"现成寄存器直方图"是假的（`radix_core.cuh:365` 的
   `hist_add_bf16_reg` 只记到 1.5% 的元素），要做得**新写**一个
   register-then-flush 直方图：每线程 `r_hist[16]`、索引 `(key >> 4) & 15`、
   结尾一次性 flush。风险从"冲突"变成"**依赖链**"（见"别做"第 1 条），先量。
3. **行趟展开 / 加大每 CTA 的 MLP**（原第 3 条）：pass 1 的可见上限只有约
   34 µs，所以这条针对的是 pass 2 的读趟——而 pass 2 现在走清单（§13），
   先量清单版还剩多少。

**别做**（已被实验否决，别重复）：

- **分组原子归并的任何形式**。`P4` 实测 619.6 → 657.6 µs（更慢）；decode 战役
  又量了第三种（warp 聚合 + 前缀 + 每 warp 一次原子，`probe_atomic.cu`）：
  **每个档都更慢**（len 8192：9.0 对 7.0 µs；len 262144：139 对 91）。
  本平台共享原子接近 1 拍，那 10 条 shuffle/popc/索引算术是净增——
  **连"减少原子发射次数"这一形式也关闭了**。
- **把 merge 并行**（`<<<B×m>>>` 分段再小合并）：两段就是两次 `_b` 调用，直接 2×。
  **把两个 stage 合进一个 kernel**（最后到齐的 CTA 做归并）只省 1.2 µs 发射，
  而 per-row 票据计数器每次调用都要清零（memset 7.1 µs），净亏。
- **给 bf16 补 chunk 数路由**：只有 `b=1 × V=1M` 值 ~11%，其余格 16 最优，
  三点拟合不出规则（§9.16）。
- 固定加大 arena（24 KB）换溢出；把按最大 k 预留 smem 的写法请回来——
  都不能在 topk ≥ 2048 保住 occupancy。
- 在 C500 上为 FP8 做性能判断（C500/C600 的 FP8 是软件模拟，性能只在 C600U 上
  测）——本题不涉及，但同仓库别的活涉及。
  测）——本题不涉及，但同仓库别的活涉及。

---

## 10. 名词速查

- **粗层 / coarse level**：直方图保留的 key 高位位数。现在是 12（`kCoarse12Bits`）。
- **细层 / fine**：剩下的低位，现在 4 位（`kCoarse12SubBins = 16`）。
- **arena**：`s_input_flat`，存阈值桶成员下标的地方；容量 `kCoarse12ArenaEntries`
  = 4,096（16 位行）/ `rk::kSmemInputSize` = 3,514（fp32 行与 static-k 行）。
- **overflow**：阈值桶成员数超过 arena 容量，回退整行重扫。
- **occupancy（occ）**：每 AP 常驻 CTA 数。C500 是 104 AP、64 KB smem/AP、
  2,048 线程/AP；`静态 2328 B + 动态请求 < 32,768 B` 才有 2 个 CTA/AP。上限 2 是
  寄存器文件卡住的（不是 smem），所以 smem 相对"便宜"。
