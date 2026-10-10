# 跳过表的寄存器账 —— 落地的形状是被 `--resource-usage` 逼出来的（2026-10-10）

`README.md` 记的是"第二趟能跑多快"，这一篇记的是"把它接进去要花多少寄存器"——
后者才是这次落地真正花时间的地方，也是唯一一个差点把收益吃光的地方。

## 起点：一个没先量就动手的代价

`/tmp/res_usage.sh`（`mxcc --resource-usage -x maca -maca-device-only` 加 build 自己的
flag 列表，去掉 cucc 专有拼写）逐实例对账 base 与改后：

| | base | 改后 |
|---|---|---|
| 全部 40 个 bf16 行实例 | 20–28 MT | **32–36 MT** |

**行核在 BLOCK=1024 下需要 ≤32 MT 才有第二个 CTA/SM**（1024 线程 × 2 CTA × 32 =
65,536 = 每 AP 的寄存器堆）。这不是估计：本仓在 NaN 内联那一役量过 2 CTA/SM → 1 CTA/SM
的代价是 **b4096 V524288 k512 bf16 上 34%（9,310 → 6,134 µs）**，而 skip 的预测收益是
网格 **−13.5%**（`README.md`）。**所以寄存器这一项如果修不回来，整个杠杆是净亏。**
`skip` 门内（bf16、`vocab ≥ 65536`）的 cell 全部走 BLOCK=1024 的实例
（`needs_long_row_bf16`：B ≤ 4096、topk ∈ {512,1024}、L ≥ 60000）。

## 定位：15 个编译期的单变量臂

全部是**只编译不跑**的臂（复制 `csrc/` 到 `/tmp/abl/<臂名>`、只改拷贝、编出
`--resource-usage`），每个约 1 分钟。表里是 `<i1024>`（int32 索引）六个代表实例的
MT 范围；`base` = HEAD。

| 臂 | 改了什么 | MT | 读法 |
|---|---|---|---|
| `base` | — | 22–28 | |
| control | 生产改动（全功能） | **33–36** | +8…+12 |
| `nosummary` | pass 1 的摘要循环整个删掉 | 30–34 | 摘要 ≈ −3 |
| `nocompact` | 清单循环 `if (false)` | 31–34 | 压实 ≈ −2 |
| `numlist` | 三条清单走法 `if (false)` | 33–36 | **走法 = 0** |
| `unroll1`(gather) | 走法内层 `#pragma unroll 2` → 去掉 | 33–36 | 0 |
| `rvhist` | 调用点不给 `max_key` | 30–34 | 逐字 max ≈ −3 |
| `pack` | 改成返回值传（去掉指针参数） | 33–36 | **指针本身 = 0** |
| `noinline` | 摘要整块挪进 `__noinline__` 函数 | **40–42** | 隔离 = 更差（printf 那一课重演） |
| `halfmax` | 每字 max 保留、跨字累加删掉 | 30–32 | **累加 = −3…4** |
| `tree2` / `treem` | 两累加器 / 四值树形归约 | 33–36 | 换形状无效 |
| `bperm` | 3 次 `__shfl_down_sync` → `bsm_bpermute` | 33–36 | shuffle 包装 = 0 |
| `unroll1callee` | SWAR 字循环 `#pragma unroll` → `1` | **30–32** | **unroll = −3…4** |
| `unroll2callee` | 同上 → `2` | **30–32** | 同样够，保留两份并行度 |
| `c1` / `cloop` | 清单循环删掉（保留 use_list） | 30 | 压实 = 4 |
| `cJ` | 清单去掉 `list_cap` 守卫 | 34 | 0 |
| `trim` | `use_list`/`nlist` 挪进 shared | 33–36 | 0 |

**两个非显然的结论**：

1. **真凶是 SWAR 字循环的 `#pragma unroll`，不是它旁边那些**。`halfmax` 与
   `unroll1callee` 都落回 30–32：一旦有跨迭代的累加值活着，四路展开的循环体就把它
   钉住 3–4 个寄存器；`unroll 2` 把峰值拉回来而保留两字并行。`tree2`/`treem`
   说明这与依赖链形状无关（换成树的读数不变）；`bperm`/`pack` 说明它与 shuffle
   包装、指针别名都无关。**判据只有逐臂编译，源头上看不出来。**
2. **后一批读数（`c1`/`cloop`/`cJ`/`trim`）是分配器的整体抖动，不是干净归因。**
   `cloop` 与 `cJ` 差一行却差 4 MT，`noinline` 反向 +8 —— 在这个体量下，改任何
   一处都会让分配器重排整函数。所以**不要拿"再省一个变量"当路线**；要保证预算，
   只能改结构。

## 落地的两笔

1. **`#pragma unroll 2`**（`radix_core.cuh`，`hist_add_bf16_wide` 的字循环，附一句
   注释说明 32 MT 的来源）。
2. **`kSkip` 模板开关**：`radix_topk_row_bf16_b<BLOCK, kNan, kSkip>`，由
   `radix_select_row` 从 `std::is_same<OutIdxT, int32_t>` 传下；host 侧两处门
   （`maca_topk.cu` 的分配与 `p.row_block_*` 赋值）同步加 `index_dtype == 0`。
   **64 位索引的行不带这套**：它的 epilogue 本来就比 32 位行高 2 MT（base 28 vs 26），
   带上去就是 34 > 32。没有新增实例——`kSkip=false` 只是让那些实例把这套编掉。

改后逐实例复核（`--resource-usage`，全部 40 个 bf16 行实例）：

```
worst bf16 row MT: 32        # 之前 base 的 40 个里最高的也是 32 档
>32 的实例: 无
```

## 复现

```bash
# 单臂：复制 csrc、打补丁、编译、读表
bash /tmp/abl/run.sh <臂名> <patch.py>          # 见本文表格左列
# 生产树的同一张表
bash /tmp/res_usage.sh                           # 输出里 Function properties / MTregisters
```

臂脚本不入库（它们是这次定位的过程物）；要复跑，按表里的"改了什么"一列重写即可——
每个臂都只有一处改动。
