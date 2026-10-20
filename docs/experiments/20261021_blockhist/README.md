# bf16 行内核 pass 1:先数块最大值,再只数留下的元素(2026-10-21)

设备:MetaX C600-U(28 AP,family 1600,MACA 3.8.1.3)。落地提交 `095224c`
(`nan_scan_table` 分支),改动只在
`csrc/maca_kernels/common/radix_row_bf16.cuh`。

产物收据:旧臂 `187d7932c7992b237058e3888ca1f41f`(树 `e168b4e`,干净),
新臂 `fd456758ec6d518de09e2e4d4500eae7`。两棵树的 `diff -rq csrc` 在修掉
一处探针残留后**只剩 `radix_row_bf16.cuh` 一个文件**,所以下表量的是这一个
改动的效果。

探针源码都在本目录:`walk.cu`(线性 walk 形态)、`walk2.cu`(生产摘要 walk
形态)、`walk_run.py`(两者的 ctypes 驱动)、`ab_cells.py`(逐 cell 交替 A/B)。

## 0. 为什么目标是 pass 1:相位实测

在 fp32 上先用 `clock64()` 相位探针量了一遍,随后把同一套探针搬进 bf16 行内核
(`ds_pp[]` 时间戳 + `output[topk-16..topk)` 带出;探针只读时钟,不改数据流,
但会覆盖这 16 个输出槽,所以只在临时树上用)。探针归档在本目录:
`patch_row_phase.py`(插入)+ `row_phase.py`(驱动),命令见 §7。b4096
V=1048576 k=1024(17.5 ms,网格里最大的一格):

| 相位 | 占比 |
| --- | ---: |
| hist(pass 1) | **78.4%** |
| fold + narrow | 1.0% |
| 块列表 | 5.3% |
| collect(pass 2) | 14.0% |
| refine cumsum | 0.4% |
| arena + emit | 0.9% |

短行上 collect 的份额变大(块少,保留比例高,列表覆盖的行更多):

| cell | hist | fold | list | collect | ref | arena |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| b4096 V262144 k1024 | 51.6% | 3.1% | 4.2% | **37.5%** | 0.9% | 2.7% |
| b512 V262144 k512 | 62.6% | 3.7% | 5.2% | 24.8% | 1.6% | 2.1% |

in-situ 速率:8.59 GB / 13.7 ms = **627 GB/s**,即读墙(1545 GB/s)的 41%,
约等于共享原子发射率顶(~780 GB/s 的 80%)。**pass 1 是原子发射率受限,不是带宽。**

## 1. 想省原子,先问元素能不能先合并——不能

每元素一次共享原子,能不能在线程内先聚合?在生产分布(NormalFloat bf16)上直接数:

| 量 | 值 |
| --- | ---: |
| 相邻两个元素落在同一个 12 位桶的比例 | **1.4%** |
| 连续 8 个元素的不同 12 位桶数 | **7.62** |
| 前 10 个桶占一行 | 23.3% |
| 最大的单桶占一行 | 2.8% |

V=1048576 / 262144 / 16384 三档、BLOCK 512/1024 两种形状读数一致(1.3–1.5%,
7.61–7.62)。**全局集中不等于局部重复**:排序/合并最多省 0.38 次原子/8 元素
(4.7%),不值得任何比较开销。这条与 `20261019_c600u_pass1` §5 的"寄存器预聚合
427 vs 678 GB/s"互相印证。

## 2. walk 微基准:两种形态,四个臂

`walk.cu` 是线性形态(每线程每迭代一个 `uint4`);`walk2.cu` 是**生产摘要
walk 的形态**(8-lane 组覆盖一个 64 元素块、两块在途、3 步组内 max 树、
`block_max` 落盘),后者才是能外推的读数。1024 行 × V=1048576,5 次迭代,
弃首轮。臂 = 走法 × 原子策略:

`walk.cu`(线性形态):

| 臂 | BLOCK 1024 | BLOCK 512 |
| --- | ---: | ---: |
| walk-only(无原子、无树) | 1098.8 GB/s | 1120.8 |
| elem-atomic(今天的做法) | 710.7 | 712.9 |
| blockmax-atomic(每 64 元素一次) | 888.5 | 905.2 |
| blockmax-noatom | 988.7 | 998.6 |

`walk2.cu`(生产摘要形态,BLOCK 1024):

| 臂 | 时间 | 带宽 |
| --- | ---: | ---: |
| elem-atomic(今天) | 3.943 ms | **544.7 GB/s** |
| blockmax-atomic | 2.513 ms | **854.6 GB/s** |
| no-atomic(天花板) | 2.745 ms | 782.3 GB/s |

三条读法:

1. **生产形态比线性形态慢**(544.7 vs 710.7),与 in-situ 的 627 GB/s 对得上,
   所以外推用 `walk2.cu` 而不是 `walk.cu`。
2. **块最大值直方图在生产形态里值 1.57×**(544.7 → 854.6),和 in-situ 的
   pass-1 相位(13.7 ms)给出的 8.59 GB ÷ 854.6 GB/s ≈ 10.0 ms 一致。
3. **no-atomic 臂(782.3)反而低于 blockmax-atomic(854.6)**,这是臂的运行顺序
   (它最后跑,时钟/热状态)造成的伪影,不是设计输入——本设计只用 545 和 855
   两个数,没有用到"去掉全部原子能到多少"。记在这里是因为它说明**同一探针内
   跨臂比较要交替或至少重复**,单轮顺序读数不可信(与 `20261019` §0 的
   "弃首轮"是同一类陷阱)。

## 3. 设计:块切割是精确的

pass 1 里现有的摘要 walk 本来就算出了每个 64 元素块的**最大键**。把它也数进
直方图(每块 1 次原子),得到的是"块计数"。取块计数跨过 `topk` 的那个桶 `m`:

```
#blocks(max >= m) >= topk   =>   #elements(key >= m) >= topk   =>   k-th element T >= m
```

于是**所有 ≥ T 的元素都在 max ≥ m 的块里**。用 `m` 建块列表,第二次直方图只走
这些块(`1-(1-k/V)^64` 的行),对每个 ≥ T 的桶,它看到的成员就是该桶的**全部
成员**;fold/cumsum/narrow 在那份计数上再跑一次,阈值、arena 溢出规则、collect、
四条返回路径全部不变。第二次扫描是精确的,不是采样:块列表是"可能含候选"的
超集,而在超集内计数是完整的。

溢出规则的合法性也保住了:arena 的容量判断看的是阈值桶的**真实成员数**,而
第二次直方图已经把阈值桶的全部成员数过了。

## 4. 密度门:什么时候值得走两遍

第二次 walk 的成本是行的一个比例 `f = 1-(1-k/V)^64`,所以路线成本是

```
1/855 + f/545        (块直方图 + 元素直方图)
1/545                (今天的单遍元素直方图)
```

两者相等解出 `f = 1 - 545/855 = 0.363`,即 `V/k ≈ 142`(142.6)。落地门取
`topk <= length / 256`,是盈亏点的两倍余量:

```cpp
constexpr uint32_t kBlockHistMinRowPerSlot = 256;
const bool block_hist = summary_wanted && topk <= length / kBlockHistMinRowPerSlot;
```

**这个成本模型是量级保护,不是预测器**:在 V/k = 512 它预测 1.32×,实测 1.135×;
在门槛 V/k = 256 它预测 1.16×,实测 1.012×。偏差来自模型没算的两项——第二次
walk 走的是散落块,局部性比整行顺序读差;以及该格 collect 已占 37.5%。所以门槛
按实测的"优势消失点"取,模型只用来定数量级。

门以下的形状、非对齐行、`kSkip=false` 走的仍是原来那条单遍元素 walk,逐字节
未变(实测两格 1.002× / 0.991×,即对照)。

## 5. 实测(b4096 bf16,官方形状,两轮交替)

官方形状 = `indices_type=int32`、`return_value=False`、
`abort_when_nan_found=False`。`ab_cells.py`,两棵树每轮交替,每格
`kernelkit.bench(fn, 10)`,核名含 `topk` 的核时间求和:

| cell | before(187d7932) | after(fd456758) | 加速 | GB/s |
| --- | ---: | ---: | ---: | ---: |
| b4096 V1048576 k1024 | 17418.5 µs | 14159.2 µs | **1.230×** | 493 → 607 |
| b4096 V1048576 k512 | 16540.0 | 12753.0 | **1.297×** | 519 → 674 |
| b4096 V524288 k1024 | 9723.0 | 8564.6 | **1.135×** | 442 → 502 |
| b4096 V262144 k1024 | 5891.7 | 5822.9 | 1.012× | 365 → 369 |
| b512 V262144 k512 | 681.3 | 601.6 | **1.132×** | 394 → 446 |
| b4096 V131072 k1024 | 3827.6 | 3821.3 | 1.002×(门以下对照) | — |
| b4096 V65536 k1024 | 2350.8 | 2372.9 | 0.991×(门以下对照) | — |

寄存器:该 TU(`value_bf16_outidx_i32_block_1024_si1_rv0_sv0_pre1.cu`)
`--resource-usage` **44 → 42 MT**,无栈帧溢出,所以内联进来的第二条 walk 不占
占用率。

正确性:`tests/test.py --backend maca_c --sample 200` → 抽 346 格
**346/346 通过**,0 check_fail、0 crash,只有 OOM skip。

## 6. 方法陷阱:扰动直方图的 in-kernel 消融会打崩设备(两次)

在落到微基准之前,先试过直接在核内做消融来给 pass 1 定价,两个臂都废了:

1. **完全去掉 pass-1 原子**(直方图不再反映这一行);
2. **1/64 采样 × 64 放大**(阈值变成估计值)。

两次都是 `Memory Violation(0x4)`。机制很清楚:collect 阶段是
`output[atomicAdd(&s_counter, 1u)] = idx`,**没有边界检查**——它依赖阈值正确
(即"≥ 阈值的元素数 ≤ topk")。阈值一被扰动,`s_counter` 就冲出 `topk` 个 slot,
写进 `selected` 区之外。**消融一个"喂给无界写入的估计量",等于把测量变成故障注入。**
正确的做法是本目录的独立微内核:同样的 walk 形态,原子策略作为模板参数,
不碰生产数据流。

## 7. 复现

```bash
# 0) 相位探针(§0 的表)
cp -a <e168b4e 的干净树> /tmp/ds_phase
python patch_row_phase.py /tmp/ds_phase      # 只认 095224c 之前的锚点,数不对就退出
(cd /tmp/ds_phase && rm -rf build && ./develop.sh)
DS_TESTS=$PWD/tests python row_phase.py /tmp/ds_phase
```

`patch_row_phase.py` 的 11 个锚点全部按"出现次数必须等于 1"校验(原来硬编码
`/root/ds_phase`,归档时只把树路径改成 `argv[1]`);它在今天的树上会因锚点计数
不符直接退出——这是设计,不是坏了。探针只能量时序,那一版输出不是正确的 top-k。

```bash
# 1) walk 微基准(设备机)
export MACA_PATH=/root/maca/maca-sdk-3.8.1.3/opt/maca-3.8.1
export LD_LIBRARY_PATH=$MACA_PATH/lib:$MACA_PATH/mxgpu_llvm/lib
$MACA_PATH/tools/cu-bridge/bin/cucc -shared -fPIC -O3 -std=c++20 -use-fast-math \
     -offload-arch=xcore1600 -o /tmp/walk2.so walk2.cu
DS_TESTS=$PWD/tests python walk_run.py /tmp/walk2.so
```

(`walk_run.py` 归档时参数化了:原来/这次运行用的都是 `/tmp/walk.so` 与
`/tmp/walk2.so` 两个路径,现在是 `argv[1]`;`DS_TESTS` 原来硬编码为 pod 上的
checkout,现在是环境变量。除这两处外,脚本与运行当天相同。)

`walk_run.py` 用的输入是 `lib.TestParam` 的 **1024 行 × V=1048576**
(bf16、seed 0),`probe(ptr, rows, len, iters=5)`;两个探针的
`probe()` 签名相同,只是 `walk2.cu` 多一个 `block_max` 缓冲(内部自行
`cudaMalloc`)。

```bash
# 2) 逐 cell A/B(两棵树交替)
for R in 1 2; do
  for T in /root/github/DeepSelect /root/ds_phase; do
    echo "== round $R $T"; python ab_cells.py $T
  done
done
```

`ab_cells.py` 把 `sys.path` 的第一项指向传入的树,第二项指向 `DS_TESTS`
(默认是 pod 上的 `/root/github/DeepSelect/tests`);它打印的
`lib <8 位 md5>` 是"这一轮到底量了哪个产物"的收据(本次两轮分别是
`fd456758` 和 `187d7932`,与正文一致)。

```bash
# 3) 正确性
PYTHONPATH=<tree>:<repo>/tests python tests/test.py --backend maca_c --sample 200
```

## 8. 边界与欠账

- 改动在 `csrc/maca_kernels/common/`,所以 xcore1500 和 xcore1000 的产物同样
  包含它;只在 C600-U(xcore1600)上构建并实测,**其它两族未构建、未测**。
- **完整 147 格网格与 baseline 重指没有跑**:pod 在重建过程中被回收(纯 CPU
  构建时 GPU 空闲,正是 `c600u-target-machine` 记的那条),所以
  `perf_data/` 里仍是旧 baseline,网格级比较欠着。网格暴露面是 53 个 bf16
  cell(V ≥ 256k),占网格时间 63.5%;上表门以上的五格含其中最大的四个 b4096
  格,门以下两格是对照。
- 成本模型只在 C600-U 上标定;`kBlockHistMinRowPerSlot` 的 256 没有在
  xcore1500 上验证过(该族 AP 数不同,原子顶也不同)。
- **`095224c` 之后的相位拆分还没量**:上面 §0 的表是改动前的。下一次测量的
  探针已经准备好、**未运行**——`patch_row_phase_after.py` + `row_phase_after.py`
  (锚点按 `43e566a` 的树校验过,本地试打过;它把 §0 的 hist 一相拆成
  hist1a / fold1 / list / hist1b / collect / ref-cumsum / last,正好对应块
  直方图造出来的两段)。跑它才能决定下一个杠杆落在第二趟 walk、collect 还是
  fold/scan 上,而不是按 14.16 ms 减去 9.9 ms 的差去猜。
- **`095224c` 的提交信息和 `C600U-algorithm-choice.zh.md` 引用的 868 GB/s
  是反解值,不是读数**:8.59 GB ÷ 9.9 ms,而 9.9 ms 来自"walk2 的 854.6 GB/s
  形态 + 其余相位不变"这一模型与实测 cell 总量 14.16 ms 的对齐(模型偏乐观
  2.6%)。唯一实测的是 14.16 ms 和两个微基准(545 / 855)。`095224c` 已推送,
  历史不改;`C600U-algorithm-choice.zh.md` 与 §8 这条是本记录对它的更正。
