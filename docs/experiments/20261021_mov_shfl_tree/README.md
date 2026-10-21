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

**方向已实测(2026-10-21,pod 30752)**:`row_shl:d`(0x10d)= lane i 读
lane i+d —— 归约落在每半 row 的**首** lane(lane 0 / lane 8),与写表谓词
`(tx&7)==0` 天然一致;`row_shr:d`(0x11d)方向相反(leader 在 lane 7/15)。
SASS 门同时通过:`mov_shfl` 降为**一条** `mov_b32 rD, rS row_shl:N rmsk:15
bmsk:15 bc:1`(ALU 管,DPP 形态),`readlane` = `rl_b32 sD, rS, lane`
(SALU),全内核无 sm_/ds_/bsm 指令。边界警告(同一张车道表):越出 row
的读取**不**遵守指南的 bc 承诺(row 顶 lane 返回陈旧跨 lane 值,lane 0
返回垃圾 0x1E00000)—— 只有严格行内读取可信;归约链按构造全部行内,
非 leader 的部分值永远不可读。

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
**(2026-10-21 已跑:见 §2 —— 方向 row_shl = i+d,SASS 为单条 DPP mov,
无 smem 类指令;前提成立。)**

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

### 3c. 原语本体 + 正确性/纯速率(`csrc/maca_kernels/common/warp_reduce.cuh` + `warp_reduce_test.cu`)

按 CUB `WarpReduce` 的语义实现的项目原语(已提交,**未经设备编译验证**):

- `warp_reduce_{max,min,sum}_u32` —— 全波归约,返回**标量**(波内一致,
  无需广播):4 条链式行移位 + 4 条向量 op(d=8,4,2,1,bc=1)把 4 个 row 的
  归约各落到 row leader,再 4 次 `readlane` + 3 次标量 op —— 15 条
  ALU/SALU 指令,零 smem 通道流量。对照:`__shfl_down_sync` 包装链
  ~71 指令(LDS 管)、手写 bpermute 蝶形 42-44(LDS 管)。
- `group8_reduce_{max,min,sum}_u32` —— 8 车道组归约,结果在组首
  `(tx & 7) == DS_GROUP8_LEADER_IN8` —— 即 `DS_GROUP_MAX8_TREE` 的替换
  语义(d=4,2,1,3 条行移位)。hist1a 补丁将以行配对块映射调用它。
- functor 形态(`ReduceMaxU32/MinU32/SumU32`)与 CUDA `#else` 臂
  (32 车道 `__shfl_down_sync` 链)按 `radix_common.cuh` 的双编译惯例保留;
  本树只编 MACA 臂。
- **方向开关**:`-DDS_ROW_SHIFT_LEFT` 一处翻转所有 mode 与 leader 常量
  (探针车道表裁决前这是假设,见 §3a)。

```bash
cucc -O3 --offload-arch=xcore1600 -I<repo-root> \
     docs/experiments/20261021_mov_shfl_tree/warp_reduce_test.cu -o /tmp/wrt
/tmp/wrt    # 语义(8 波 × {max,min,sum,8 组} 对 CPU 参考)+ 三形态纯速率
```

测试同时给出三形态(包装链 / bpermute 蝶形 / 本原语)的延迟链与吞吐
ns/reduce —— 纯归约的数字,与 §3b 的 in-situ A/B 互为印证。语义检查
失败且错位形态指向 row 另一端时,用 `-DDS_ROW_SHIFT_LEFT` 重编,不改
调用点。

### 顺序

`tree_probe` 的车道表 + SASS 先行(分钟级)—— 车道表定方向(错了就给
`warp_reduce.cuh` 和 A/B 都加 `-DDS_ROW_SHIFT_LEFT` 重编),SASS 是前提证伪门;
然后 `warp_reduce_test`(语义 + 三形态纯速率),再跑行走 A/B(~1 分钟)。
内核补丁(hist1a 行配对映射 + `group8_reduce_max_u32` 替换
`DS_GROUP_MAX8_TREE`)排在 A/B 指向之后,且只动 `!kHist` 臂(ADJ4 只在
那里),`kHist` 臂与 CUDA `#else` 半不动 —— 与 ADJ4 落地时的架构边界
同一形状:`common/` 改动进所有族的 artifact,只量 C600U,其他族不量。

## 4. 结果(2026-10-21,pod 30752,C600-U 28 AP,MACA 20261017 工具链,设备 1)

**车道表**:Shift Right(0x11d)= lane i 读 lane i−d;**Shift Left(0x10d)=
lane i 读 lane i+d**(归约落首 lane,即默认臂)。越出 row 的读取不遵守指南的
bc 承诺:row 顶 lane(lane 15/63)返回一个固定的陈旧跨 lane 值(row_top−4),
wave 底 lane 0 返回垃圾 0x1E00000;bc=0 的零填充只在部分边界成立。结论:
**只有严格行内读取可信**,归约链按构造全部行内,非 leader 部分值不可读。
`readlane(v,7)` 与 `readfirstlane` 全 lane 一致,干净。

**SASS 门(通过)**:`mov_shfl` → 一条 `mov_b32 r2, r1 row_shl:1 rmsk:15
bmsk:15 bc:1`(DPP 形态,ALU 管);`readlane` → `rl_b32 s0, r1, 7`(SALU);
`readfirstlane` → `rfl_b32 s0, r1`。全内核 **0 条 sm_/ds_/bsm 指令**。

**纯归约速率**(`warp_reduce_test`,一波,ns/reduce;语义 8 波 ×
{max,min,sum,8 组} 全过):

| 形态 | 延迟链 | 吞吐 |
| --- | ---: | ---: |
| `__shfl_down_sync` 包装链(6 步) | 918.94 | 358.17 |
| `bsm_bpermute` 蝶形(6 步) | 747.50 | 79.67 |
| **mov_shfl+readlane(本原语)** | **62.03** | **39.70** |

延迟 12.1×(对蝶形)/ 14.8×(对包装),吞吐 2.0× / 9.0×。链式 LDS permute
每步 ~125 ns —— smem 管延迟的直接读数。

**行走 A/B**(hist1a ADJ4 复刻,V=1M bf16,1024 行一 CTA 一行;NaN 有/无
在此基准无差异 —— 与在核 16.45% 税不同,谓词替身更便宜):

| 臂 | BLOCK 1024 | BLOCK 512 |
| --- | ---: | ---: |
| bpermute 树(现航) | 1061.2 GB/s | 1037.0 |
| **mov_shfl 行配对树** | **1117.0(+5.3%)** | **1110.6(+7.1%)** |
| 无树上界 | 1110.0 | 992.8 |

行配对臂达到/超过无树上界 —— 树移出 LDS 管之外,每 row 8 相邻块的映射
比组映射(4 块)的局部性更好(无树臂保留组映射,故低于行配对臂)。
**注意**:本 pod 是 MACA 20261017 新工具链,绝对值与旧 3.8.1.3 的记录
(同形状隔离 1070.7 带 NaN / 1295.6 无 NaN)不可直接比,以臂间比值为据。
