# fp32 coarse12:两遍整行行走，还是原子发射?(2026-10-21,准备中)

状态:**探针已写好并离线校验,尚未在设备上跑**。需要一台 C600U(28/32 AP)。

## 0. 为什么是这条路线

基线 `20261021_032150` 的 fp32 `maca_c` 段:17 个不同形状共 19.96 ms(按形状中位求和),
**14 个形状、时间上占 ~99% 的都走 `f32_coarse12_applies`**(narrow 下限 9216 把所有大格都收进去了)。
网格加权后更集中:`b4096 V131072 k2048`(选择器形状,重复 4 行)每行 5.57 ms,
占 fp32 网格时间的一半以上。三个不走 coarse12 的格都是小 batch(b1/b6,乘积低于下限)。

这个形状的单次流量 = 4096 × 131072 × 4 B = 2.147 GB;按 1650 GB/s 墙,一次整行读是 1.30 ms,
实测 5.57 ms 对应**两遍读也只有 47% 的墙**。所以问题不是"读了几遍",而是两遍各自多慢。

## 1. 先排除溢出重扫(离线,已做)

`topk_coarse12_row` 的溢出条件是 12-bit 粗 bin 的成员数超过 `kCandidateCapacity = 2048`,
一旦超过就要再走最多 3 轮 × 2 次整行重扫(那会直接解释 5.57 ms 的 ~4 遍读)。
官方数据是 `NormalFloatDistribution`,即 `torch.randn` 的 fp32,所以可以在本地直接算:
取第 k 大的有序 key,数同 bin(有序 key 的高 12 位)的元素数。

| V | k | 阈值 bin 成员数(8 行) | 溢出 |
| ---: | ---: | --- | --- |
| 131072 | 2048 | 1336 / 1403 / 1464 (min/中位/max) | 0/8 |
| 131072 | 512 | 386 / 422 / 450 | 0/8 |
| 129280 | 512 | 387 / 414 / 452 | 0/8 |
| 107520 | 2048 | 1129 / 1152 / 1188 | 0/8 |
| 66551 | 2048 | 498 / 624 / 656 | 0/8 |
| 262144 | 512 | 399 / 432 / 464 | 0/8 |
| 524288 | 2048 | 1635 / 1681 / 1702 | 0/8 |

**溢出不成立**,并且 k=2048 的格子离 2048 的容量只剩 ~30% 余量 —— 这是"不能再粗"的实证
(粗一级到 11 bit 就会越界)。于是 5.57 ms 的解释只剩两遍整行本身:

- **pass 1**:每个元素一次 4096 桶共享原子。C600U 上这条走的微基准上限是 **780–880 GB/s**
  (`c600u-target-machine` 记忆,bf16 直方图行走),不是 1650。
- **pass 2**:整行再读一遍做收集,只有极少数元素进原子,应该接近墙。

这与 bf16 行 2026-10-21 落地的 skip-table 结构是同一张诊断:pass 1 的价签在**逐元素原子的发射**,
不在访存。bf16 行的解法(块最大摘要 + 块列表,只对保留块做逐元素直方图)已在该行量到
17.42 → 14.16 ms(V=1M k=1024)并落地。

## 2. 探针

`patch_coarse12_phase.py` 往 `topk_coarse12_row` 里插 `clock64()` 相位标记(锚点按计数断言,
在别的树形上会拒写),`probe_coarse12.py` 通过 `out_idx[:, k-16:k]` 读回 —— 这条路是通的:
coarse12 把答案写进 scratch 列,preselected 契约半再逐列抄进 `out_idx`,而官方形状
(RV/SV/SI 全 false)下它只写索引、不解引用 `input[src]`(`if (RV)` 是编译期假),
重排判定是 `src < 0`,探针值被掩成非负所以不会触发。**探针树只能按官方形状跑**,
带 `return_value`/`sorted_value` 会真的去解引用一个时钟值。

相位:start / pass1 / fold+narrow / collect / overflow-end / arena-in / end,
外加 arena 第 0 轮与第 1+2 轮、路径码(1 精确填充、2 溢出、3 arena 提前返回、4 arena 主路径)、
`num_input[0]`、`threshold_bin_id`、narrow 后剩余 topk、counter。它们落在
`out_idx[:, k-16:k]` 的 16 列里:第 0..6 列是相位增量,第 7 列 arena 第 0 轮、
第 8 列路径码、第 9 列 `num_input[0]`、第 12 列 `threshold_bin_id`、
第 13 列 narrow 后剩余 topk、第 15 列 arena 第 1+2 轮。

锚点已在当前树上离线验证:10 个 `sub()` 全部命中(计数 1),patch 结果见脚本自身。

## 3. 跑法(pod 上)

```bash
cd ~/github/DeepSelect && cp -r . /tmp/probe_tree && cd /tmp/probe_tree
python3 docs/experiments/20261021_coarse12_phase/patch_coarse12_phase.py /tmp/probe_tree
./develop.sh                       # 全量 90 TU,15-25 分钟
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=/tmp/probe_tree:/tmp/probe_tree/tests \
  python3 -u docs/experiments/20261021_coarse12_phase/probe_coarse12.py
```

产物 `develop.sh` 会写回 `/tmp/probe_tree/deep_select/*.so`;探针树与被测树分离,
主树不受影响。

**线程数是单点改动,已验证**:`topk_coarse12.cuh` 里的 `extern template
launch_coarse12_kernel<kThreads>` 和生成 TU(`xcore<N>/instantiations/coarse12_kernel.cu`)
的显式实例化都写**符号** `kThreads` 而不是数字,所以改 `constexpr int kThreads`
一处,声明、实例化、`__launch_bounds__`、`<<<B, kThreads>>>` 全部同步,**不需要重跑
`generate_instantiations.py`**。代价是 `topk_coarse12.cuh` 被 `routing.cuh` /
`radix_common.cuh` / `topk_chunks.cuh` 传递包含,几乎每个 TU 都要重编 —— 一个宽度一次
完整 90-TU 构建,所以这一轮只值得在探针给出相位拆分之后跑。

**同一次 pod 会话里的顺序**(基线目前停在 `20261021_032150`,ADJ4 的
`20261021_063653` 随 pod 30717 丢失且从未提交,见记忆):先干净树 `./develop.sh`
→ `CUDA_VISIBLE_DEVICES=1 ./run_bench.sh --set-baseline`(带 gate,~10 分钟;它同时
产出 `run_header.txt`,没有这个文件的基线目录 `run_bench.sh` 看不见)→ 提交
`perf_data/` 目录并重绘 PNG;然后再复制、打补丁、第二次完整构建、跑探针。构建期间
设备 0 上开 CUDA-graph 回放保活(编译是纯 CPU,util 读 0 会被回收)。

## 4. 结果

(待设备。要填的表:每相位占比、路径码分布、`num_input[0]` 分位。)
