# mov_shfl 行内树替换 bpermute 组树:语义探针 + 摘要行走 A/B(2026-10-21,准备中)

状态:**探针与 A/B 已写好,尚未在设备上跑**。起因是平台论断(用户,2026-10-21):
**`__shfl_*_sync` / `bsm_bpermute` 实际生成走 smem 通道的 permute,延迟高;
应改用 16 车道行内规约(`mov_shfl`)+ 跨组 `readlane`**。这与已记录事实链
一致 —— `sm_bperm_b32` 名字里就是 shared-memory,ISA 指南把 `mov_shfl`
(行内混洗,5.1 节)与 `bsm_bpermute`(全 warp 混洗,5.2 节)分成两类 ——
但**速率与指令类别都必须实测**,本文是测量准备。

## 1. 为什么这条可能值钱:同一根管子

hist1a(ADJ4 摘要行走,bf16 大格的 73-80%)当前每 64 元素块执行
**3 条链式 `bsm_bpermute`**(8 车道组树)+ 1 条块最大原子。若 bpermute 走
LDS 通道,则摘要树与块最大原子、以及 NaN 投票的 `__syncthreads_or` 之外的
所有 smem 流量**挤同一根管子**。此前"摘要成本 ~35%"(采样+OUT=2 臂
1400-1630 → 加摘要后 ~950-1000)归因于指令开销;管道争用是更精确的机制,
也给出了一个可分离的检验:把树挪出 LDS 通道,速率应回升。

## 2. 替换形态(已在指南语义上推演验证)

现行:8 车道组 × 每块 3 条 bpermute(链式 d=1,2,4),组首 `(tx&7)==0` 写表。

**行内树**:一个 16 车道 row 覆盖 2×4 个相邻块 —— 低半 row(lane 0-7)取
块 {base+0..3}、高半(lane 8-15)取 {base+4..7},每 lane 每块一个 uint4
局部 max;**一次 d=4,2,1 链式 `mov_shfl`(Row Shift,bc=1 保持原值)同时
归约两半**:d=4 后 lane i = max(orig_i, orig_{i+4});d=2 后 lane 0 =
max(o0,o4,o2,o6)、lane 8 = max(o8,o12,o10,o14);d=1 后 lane 0 = max(o0..o7)、
lane 8 = max(o8..o15)。污染只流入不读取的低位 lane。**每块 1.5 条
ALU 通道混洗** vs 现行每块 3 条 LDS 通道 bpermute —— 指令数减半且换管。
写表谓词不变(`(tx&7)==0` 恰好选中 lane 0 与 lane 8);ADJ4 邻接性保留
(row 的 8 块 = 两个半 row 各自的 4 相邻块)。

**方向待探针裁决**:指南只说 0x111-0x11F = Row Shift Right,未定义方向。
若实际是 lane i 读 lane i−d,则 leader 变成 lane 7/15,写表谓词改
`(tx&7)==7`,链序不变 —— **速率两臂相同,与方向无关**;方向只影响内核
补丁的正确性,不影响 A/B 的判读。

`readlane` 的用武之地是**波宽归约**(把 4 个 row 的 leader 收成标量):
本热路径不需要(结果按组用),候选是冷点 `run_cumsum_warp`(每行 2 次,
71 指令包装形态)与 coarse12 的 `warp_cumsum_histogram<256>` —— 本次不动。

## 3. 探针

### 3a. 语义 + SASS(`tree_probe.cu`,自包含可执行)

```bash
cucc -O3 --offload-arch=xcore1600 tree_probe.cu -o /tmp/tree_probe && /tmp/tree_probe
mxcc -aop -S -maca-device-only tree_probe.cu -o /tmp/tree_probe.s
```

车道表回答:Shift Right/Left 的方向、bc=0/1 的越界行为、bmsk 是否只门控
目的 lane。**SASS 是第一道证伪门:`mov_shfl` 若同样降为 sm_/ds_ 类指令,
整个前提不成立,A/B 不必跑。** 同时看 `readlane` 的指令形态。

### 3b. 摘要行走 A/B(`tree_walk_ab.cu` + `tree_walk_ab_run.py`)

hist1a ADJ4 臂的独立复刻(V=1M bf16,一 CTA 一行,BLOCK 512/1024,
4096 桶 s_wide + block_max 表 + 每轮 4 条 leader 原子,全部两臂一致),
三臂 × NaN 有/无 × 两宽度:

| 臂 | 形态 |
| --- | --- |
| 0 | 现行 bpermute 组树(对照) |
| 1 | mov_shfl 行内树(块映射不同,块数/原子数相同) |
| 2 | 无树(仅局部 max,速率上界;摘要值错误,仅作天花板) |

判读:

- **臂 1 ≈ 臂 2** → 树的代价全部消除,直接排内核补丁(radix_row_bf16.cuh
  8 个 `DS_GROUP_MAX8_TREE` 调用点 + 行配对块映射 + 尾循环)。
- **臂 0 < 臂 1 < 臂 2** → 管道争用只占摘要成本的一部分,按差值评估。
- **臂 1 ≈ 臂 0** → bpermute 不在关键路径(延迟已被 4 个在飞 uint4 隐藏),
  前提虽对也无收益,放弃此方向,记录结案。
- 参照点(20261021_blockhist,同形状隔离微基准):FULL2(逐元素原子)862 →
  ADJ4 1296 GB/s;带 NaN 1070.7;纯读天花板 1400-1630。

### 顺序

`tree_probe` 的车道表 + SASS 先行(分钟级);SASS 过关再跑 A/B(~1 分钟)。
内核补丁排在 A/B 指向之后,且只动 `!kHist` 臂(ADJ4 只在那里),`kHist`
臂与 CUDA `#else` 半不动 —— 与 ADJ4 落地时的架构边界同一形状:
`common/` 改动进所有族的 artifact,只量 C600U,其他族不量。

## 4. 结果

(待设备。要填:车道表、SASS 指令类别、A/B 12 行的 GB/s。)
