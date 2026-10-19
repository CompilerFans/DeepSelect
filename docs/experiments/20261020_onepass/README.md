# 一趟式(不建直方图)行内核:探针与设计(2026-10-19,C600-U/C600-A)

背景:现行两遍结构在 28-32 AP 部件上被共享原子发射率封在 ~780-880 GB/s
(pass 1,见 20261019_c600u_pass1 §1)。上游 DeepSelect 的原始数据流不建
直方图:按块扫行、维护运行阈值 T、元素 > T 才进候选 arena、arena 满到
k+B2 就 compact(把 arena 精确收敛回 top-k 并把 T 提到其第 k 名)。每元素
全局只读一次;原子只在候选上,率 ~k·ln(L/B) ≈ 4K/行,不构成本。

## 探针实测(`onepass_probe.cu`,4096 行 × 1M bf16,生产 walk 几何)

| 臂 | GB/s | 读法 |
|---|---:|---|
| 裸读(仿射块序) | 1176-1184 | 随机块序比顺序读(1650)慢 ~28% —— 见决策 1 |
| 过滤+暂存(阈值预置真值) | **930-932** | 每元素:xform+比较+罕见原子 |
| 完整一趟(含 compact) | 492 | compact 的串行 256-bin 扫描是大头 |

Compact 是唯一的开销源,且两个减法都明确:合并阈值只需**保守**(降到
k-th 所在 bin 的底即可,arena 天然多留 slack)→ 一轮而非两轮;B2 加大 →
compact 频次减半。调好后期望 ~800-880 GB/s 端到端。

**对照现状**:同一 cell 的两遍结构 = 513 GB/s 逻辑 ⇒ **1.6-1.7×(随机块序)
/ 2.2-2.3×(若顺序序可行)**。

## 落地形态(若做)

- 新内核 `radix_topk_row_bf16_onepass`:一 CTA 一行,顺序块序 + arena 上界
  `k + B2 + B_check`(B_check = 一次 CTA 迭代覆盖的元素数,512 线程下 4096)
  保证 arena 永不溢出(upstream 的不变量);k ∈ {512, 1024, 2048} 起步。
- smem:arena (k+B2+B_check) × (key4B + idx4B) + 256-bin hist ≈ k2048 时
  ~68 KB —— 只在 128 KiB 部件成立,族门控 `ARCH_SMEM_PER_AP_BYTES`。
- compact 复用 `radix_common.cuh` 的 256-bin warp cumsum;两轮 8+8 精确 select
  (arm D 探针里的代码就是雏形,把 pick 扫描换并行即可)。
- 合同半完全不动:内核只负责把精确 top-k 的索引写进 `selected[]`,emit /
  NaN 处置 / 有序变体都走现状。NaN 投票照旧折进 walk。
- 回退:sorted/对抗输入下 staging 率不失控(count 上界由 compact 保证),
  无超时风险;真正异常的输入(全 NaN)走 NaN 处置。
- C500:其 pass 1 已在 96% 墙,一趟式只省第二趟 —— 另测,不默认。

## 风险清单

1. compact 的正确性(arena ⊇ 当前 top-k 的不变量)—— upstream 的证明 +
   本仓合同检查。
2. 低频 worst-case:输入恰好升序时 staging 率为 100% 直到首次 compact ——
   被 compact 频次兜住,成本有界。
3. 短行/小 batch 不入门(gate 到行长 ≥ 65536 的带表区)。
