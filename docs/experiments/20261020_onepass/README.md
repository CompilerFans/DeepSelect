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

## 原型 v1 的实测结局(2026-10-20,`prototype_v1.patch`)

按上面设计实现了完整内核(running threshold + smem arena + 两轮 8+8 精确
compact + 寄存器快照防改写竞争)。**正确性全过**(8 个目标 cell + 升序/常数/
NaN 对抗组),途中修掉三个只有就位形态才暴露的 bug:

1. **gather 自改写竞争**:compact 的收集把幸存者写回 arena 前面,无序原子
   位置会覆盖未被读走的候选项 → 每行 ~39% 槽位重复。修法:arena 内容先
   快照到寄存器(`snap[21]`)再收集。
2. **轮间清零竞争**:首轮 pick 后各线程读 `s_hist[pick+1]` 算 prefix_keep,
   与次轮直方图清零无 barrier —— 快线程把慢线程未读的 bin 清零,prefix_keep
   变短、阈值算高、误删候选(printf 可掩盖的时序竞争,只在 ~1% 行上显形)。
   修法:pick 读数后补一个 barrier。
3. 升序对抗输入下 compact 每迭代触发、阈值不减,正确性仍成立。

**性能:全部目标 cell 慢 2.0-2.6 倍,不落地。** 归因(与 §探针的对照):

| 位置 | 估算成本/row(V=1M) | 说明 |
|---|---|---|
| walk(load+xform+比较+罕见原子) | ~80-136 µs | 与探针 930 GB/s 一致 |
| compact ×6 次 | **~144 µs** | **hist 两轮 + gather 都要按 idx 回读 key——每次 compact 20K 次分散 2B DRAM gather** |

探针的 "filter+stage 930 GB/s" 没算 compact 的真实 gather 成本;原型把它
摊开了:**瓶颈不在 walk,在 compact 每轮按索引回 gather 键值**。

## 若继续,修复方向(按预期净收益排)

1. **key 随行(sidx + skey 双数组)**:staging 时顺手写 16-bit key 到旁路数组,
   compact 的 hist 全走 smem,只对幸存者(~k/次)做一次回读。代价 arena×1.5
   smem(k512 ≈ 55-60 KB → 2 CTA/SM),配 OUT=2 后 walk 不掉速;compact 成本
   ~10 倍下降。预期整 cell ~560-750 GB/s(比现状 +8~45%,随 B2 调参)。
2. B2 加大(1024):compact 频次减半,smem 再涨一档,与 1 联用。
3. 当前 gather 是 L2-hot 的(行刚流过 L2)——实测确认 L2 命中率后可省 sidecar。

记录:`check_1p.py` 对原型全绿的 8 cell + 对抗组;compacts 计数与阈值
收敛已与模拟逐行核对。

## 终局:v3 与 tie 质量(2026-10-20,`prototype_v3.patch`)

v2 的根因(sidecar 之外)按修复顺序:

1. **洪水**:T=0 起步时每元素都暂存,单计数器原子风暴(几十 µs/行)。
2. **种子越界直方图**:种子用了 4096 桶写进 288 槽的 cumsum 缓冲,OOB 写坏
   相邻 smem → 种子阈值飘到 56832(真 k-th 49235),全行走回退洪水。
   改 256 桶(kp>>8 / kp>>24)。
3. **采样串行**:采样谓词只命中 group 0 的块号,16K 元素全压 8 车道
   (~180 µs/行)→ 改按组分摊。
4. **walk 仍在 ~270 µs/行**:256 次迭代各 1 个在途 load(延迟链)+ 256 个
   vote barrier(26-51 µs)+ 触发 compact。双 load 合批只值 ~10% —— 真正的大头
   是 vote 与暂存量。
5. **barrier 全拆反而更慢(87.8 GB/s)**:无 vote 时 arena 溢出静默丢候选
   → 每行都走洪水回退。

最后的机制,也是这个方向的终审(实测数字,行 0):

```
count(key > 49152) = 23,462     count(key > 49194) = 4,046     count(key > 49235) = 490
```

**k-th 所在的那一个 bf16 值上挤着 ~1.9 万个并列元素。** 任何"阈值 + 候选集"
的数据流(采样阈值、running threshold、upstream 的 random-block)都必须把
这些 tie 全部装下或全部枚举——arena 装不下(23K > 8960)就只能在触发
compact/回退里反复枚举它们,成本与一趟直方图相同。直方图对 tie 是免费的
(只是计数);这正是两遍 radix 存在的理由。**一趟式在并列稀疏的分布上才有
优势,在 NormalFloat bf16 上结构性不成立。** 若哪天 DSA 线上分布被证并列
稀疏(fp8 e4m3 的 8 位键空间会让同一 k-th 桶更挤,方向相反),再拿出
`prototype_v3.patch` 重估。

> **勘误(2026-10-21)**:上段"~1.9 万并列"是误读。19,416 =
> `count(key>0xC000) − count(key>0xC02A)`,是**桶底之上 42 个相邻 bf16 值**
> 的总人口,不是 k-th 值上的并列组;精确并列组(同形状 numpy 模拟,官方
> randn 分布)为 **24–110**,fp32 恒为 1 —— 见
> `20261021_reference_recon/sim_tie_bf16.py`。**终审结论仍然成立,但成立的
> 理由换成两条实测**:(i) compact 的 6 轮 × 20K 按索引回读(上表,这才是
> 原型慢的主因);(ii) ADJ4 之后两遍结构的摘要行走在核内实测
> ~1002–1034 GB/s,已**快于**过滤+暂存行走的 930 GB/s —— 一趟式连
> "compact 免费"的理想形态也跑不过两遍的第一趟。重估条件相应从"分布被证
> 并列稀疏"改为"行走速率关系翻转"。附带地,fp32 没有并列问题,采样阈值
> 一趟式在 fp32 上开放并已模拟验证(`20261021_reference_recon` §7)。

最终架构(本战役落地):两遍 radix + skip-list + OUT=2 双发装载;采样阈值
(L1)、多路副本/波内分组、函数抽取、一趟式(v1/v3)均实测否决。
