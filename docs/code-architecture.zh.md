# 代码层次设计 —— 接口层 / 平台 / dtype / 实例化四条轴（2026-10-12）

本文件是本仓代码布局的**设计依据**：每个阶段先在这里有位置，再改代码。
目标一句话：**一个文件回答一个问题，形状判据集中在一处，实例化清单是生成出来的文件**。

## 0. 一句话结构

```
deep_select/*.py         Python 层：公开契约 / 后端 / 装载 / 日志
csrc/ffi/ + structs.h    接口层：tvm-ffi ABI + 契约常量 + arch 家族表
csrc/maca_kernels/       平台层
├── common/              跨族共享原语（与族无关的位运算）
└── xcore<N>/            族层：contract / routing / 各条数据流 / 生成出来的 instantiations
```

## 1. 官方参照（实测 `upstream/main`，2026-10-12）

```
csrc/api.cpp                        唯一 host 入口（pybind11 绑定 + 派发）
csrc/cuda_kernels/
├── bit_utils.cuh  utils.cuh  common_parts.cuh  config.h    ← 两棵 dtype 树共享
├── v3/        (bf16)  topk_select.cuh + instantiations/*.cu
├── v3_fp32/   (fp32)  topk_select.cuh + instantiations/*.cu
csrc/ascend_kernels/                第二个平台（config.h / kernel.asc / kernel.h）
csrc/3rdparty/kerutils/             设备抽象库（vendored，平台自判）
deep_select/{__init__,__version__,interface}.py     Python 层 = 只放公开契约
scripts/generate_instantiations.py  setup.py  tests/{test.py, lib.py, kernelkit/}
```

命名空间（同一实测里读的，§3b 照此）：共享件 `topk_select_common`
（`bit_utils.cuh:6`、`common_parts.cuh:16`），bf16 树 `topk_select_bf16_normal`
（`v3/topk_select.cuh:21`），fp32 树 `topk_select_fp32`（`v3_fp32/topk_select.cuh:21`），
`api.cpp` 全局、无命名空间。**平级、每个数据流一个**——这就是官方词汇的全部形状。

**官方的"程序生成模板文件"模型**（`scripts/generate_instantiations.py:100-127`）：

- 每个配置元组生成一个 `.cu`，**文件名就是元组**，例如
  `value_bf16_outidx_i32_sv0_si0_rv0_maxtopk_1024_numthreads_256_occupancy_1_b_4096_b2_4096_tma_3.cu`；
- 文件体只有三样：include 内核头、`using Config = TopkSelectConfig<…>`、一条显式实例化
  `template void run_topk_select_kernel<Config>(…)`；
- 生成器**同时打印 setup.py 要收的源清单**（`generate_instantiations.py:10-11`），所以"编哪些实例"是数据，不是散在
  host 代码里的 switch；
- 元组表（生成器）与调用点（`api.cu` 的 `TopkSelectConfig<…>` 实参）是两半，不一致报**链接错误**。

结论：官方的实例化是**文件清单 + 一张表**；派发是**按表查**。这正是"文件层次清晰、易改"的机械保证——
增删一个实例是一次生成 + 一行表，不是在一万行里找一个 switch 分支。

## 2. 四条轴：官方怎么表达，本项目怎么落地

| 轴 | 官方 | 本项目现状 | 目标 |
|---|---|---|---|
| **Python↔csrc 接口** | pybind11 绑定写在 `api.cpp` 里 | `deep_select/{__init__,interface,_binding,_log}.py` + `csrc/ffi/` 的 tvm-ffi ABI（不链接 torch、按族选 `.so`） | **保留分叉**（ABI 与按族装载是实测选择），把 python 侧三层（契约 / 后端 / 装载）显式分文件 |
| **平台** | `cuda_kernels/` + `ascend_kernels/` | `maca_kernels/`（一族一目录；每族一件产物 `deep_select_maca_xcore<N>.so`；`DEEP_SELECT_ARCH` 编译期 + `sm_count` 运行期） | 保留；族层内部再分 dtype 子层 |
| **dtype** | `v3`（bf16）/ `v3_fp32` 两个目录 | 两种 dtype 混在 `radix_core.cuh` + `maca_topk.cu` | 按 dtype 分文件（`radix_row_bf16` / `radix_row_fp32`），共享件上提 `common/` |
| **实例化** | 生成器 + 每元组一个 `.cu` + 表驱动派发 | 模板 + 单 TU 内 switch（`nm` 数出 60 个 `topk_kernel_radix`） | **采用官方模型**：配置元组模板 + 生成器 + `instantiations/*.cu` + 表驱动 |

`v3` / `v3_fp32` 这两个**目录名不照抄**：它们是上游算法代次的名字，本项目的 C500 数据流不是 v3，用它们会谎报血统。
dtype 子层按**格式**命名（`bf16` / `fp32`）。

## 3. 目标树（本项目）

```
docs/code-architecture.zh.md                本文件（设计依据）
scripts/generate_instantiations.py          生成器：元组表 → .cu 清单 + setup.py 源清单
setup.py                                    SOURCES = 手写 TU + 生成清单
deep_select/
├── __init__.py     __version__.py          导出官方名
├── interface.py                            公开契约 + backend 分派（torch / maca_c / deep_gemm）
├── _binding.py                             按族装载 .so（tvm-ffi）
└── _log.py                                 DS_LOG 调用记录
csrc/
├── structs.h                               契约常量 + arch 家族表（编译期）
├── ffi/                                    tvm-ffi ABI、错误宏、host check
└── maca_kernels/
    ├── common/                             共享实现（每族产物都编它）
    │   ├── primitives.cuh                  跨族共享件：位运算 / NaN 位型 / 键编码 / cumsum（device）
    │   ├── launch.cuh                      跨族共享件：`adopt_dynamic_smem`（host，启动原语）
    │   ├── maca_topk.cu                    唯一手写 TU：include + host 入口 + 按表派发
    │   ├── contract.cuh                    契约半：RowParams / 常量 / 键与 NaN / emit / 布局 / 行核与发射
    │   ├── routing.cuh                     选路：形状 → 数据流（纯函数，一处可读）＋梯子 `select_route`＋每条路线的 `launch_typed_*`（`topk_launch` 只分派）
    │   ├── radix_core.cuh                  数据流入口（伞头：只 include 下面四个；TU 只 include 它）
    │   ├── radix_common.cuh                行族共享件：常数 / smem 尺寸 / 直方图与后缀和 / 分块几何
    │   ├── radix_row_bf16.cuh              bf16 行：radix 两趟 + skip 表 + overflow
    │   ├── radix_row_fp32.cuh              fp32 行：+ 溢出重扫
    │   ├── radix_chunked.cuh               两条 dtype 的分块 split/merge（阶段核 + 发射）
    │   ├── topk_coarse12.cuh               deep_gemm 路线一（数据流：核 + 核启动子；路线在 routing.cuh）
    │   └── topk_chunks.cuh                 deep_gemm 路线二（数据流：核 + 核启动子；路线在 routing.cuh）
    └── xcore<N>/instantiations/*.cu        族层：每族自己的生成物（一个配置元组一个文件；今日三族同表，87 × 3）
```

### 3b. 命名空间（用户 2026-10-12 裁定：一条逻辑、一套词汇，照官方的）

三条血统（我方 radix、deep_gemm 移植、上游 DeepSelect）不再各占一个命名空间。
词汇取自官方实测（§1）：`topk_select_common` 放共享件，`topk_select_<变体>`
每个数据流一个；本项目照此：

```
topk_select_common::    跨族共享原语（原 deep_select_common）：位运算 / NaN 位型 /
                        键编码 / cumsum（primitives.cuh，device），以及启动原语
                        adopt_dynamic_smem（launch.cuh，host）。与官方同名同物——它
                        里面的 bf16_is_nan 本就来自官方的 topk_select_common（bit_utils.cuh）
topk_select_radix::     我方 radix 数据流（原 rk）：两趟行核（bf16 / fp32）+ chunked 分块
topk_select_coarse12::  移植一（原 topk_select_radix::coarse12），与文件 / 核名 topk_coarse12_* 同词干
topk_select_chunks::    移植二（原 topk_select_radix::chunks），与文件 / 核名 topk_chunks_* 同词干
deep_select::           契约半（RowParams / kMaxTopK(4096) / emit / 布局 / 选路）
                        与 tvm-ffi 入口同处一名（原 deep_select_maca 撤销并入；
                        `csrc/ffi/ffi_entries.h` 的入口本就在 `deep_select`）
deep_select::ffi::      平台无关边缘（不变）
```

三条理由，都不是口味：

- **平级不嵌套**：官方没有嵌套命名空间，三个数据流是**兄弟**。这正是"一条逻辑"的
  形态——同一族命名规则，不是一个根下再分亲疏。
- **host 侧不新起名**：官方 `api.cpp` 全局无命名空间；本项目的对应物是契约半，
  它要跨 TU 被数据流与入口引用，并入既有的 `deep_select`（包名 = ABI 名）最省一个名字，
  `using namespace deep_select_maca` 那道桥也随之消失。`_maca` 后缀去掉：平台已由目录
  （`csrc/maca_kernels/`）与编译宏（`-DDEEP_SELECT_IS_BUILD_ON_MACA`）承载。
- **`radix` 只管 radix，且只管数据流**：`rk` 是血统缩写（radix kernel，来自 dsa_topk），
  读者不可懂；把 deep_gemm 移植挂在 radix 名下同样不实——移植不是 radix。移植两路
  与自己的文件、核名同词干（P4 已定），命名空间跟上。

## 4. 规则

1. **命名：判据是"读者可懂"，不是"官方"**（用户 2026-10-12 裁定：「不是 dg 改成 official，
   而是改成合理的、实际有意义的名称」）。名字要说清**做什么**：`refine_bin` /
   `warp_cumsum_histogram<Bins>` / `bf16_is_nan` 说清了操作；`topk_coarse12`（12 位粗层的选择）、
   `topk_chunks`（按 chunk 切分的候选合并）说清了算法。官方名是**重要来源**——调用方与计时器按它接，
   且与文件里的 `__global__` 核名一致——但贴切才用；血统名（`dg12` / `dgchunks`）两样都不占，一律不用。
   公开 API 名（`topk` / `fp8_*` / `bf16_*` …）是 ABI，官方拼写**不动**（2026-09-21 裁定仍有效）。
2. **依赖方向**（`X ← Y` 读作"Y include X"）：`primitives/launch ← 数据流 ← contract ← routing ← TU(入口)`；
   python 侧 `interface → _binding → .so`。P2 落地后实测的两点修正，两条都有理由、不是懒：
   - `contract.cuh` include `radix_core.cuh`——契约半持有行核与它的布局/发射
     （`radix_layout` / `radix_select_row` / `launch_radix_config` / `launch_typed_radix`），
     所以它在数据流**之上**，不在其下。
   - `routing.cuh` include 两个移植头——梯子读它们的 `kSmemBytes` / `kMaxTopK`，且分块发射器
     是模板，定义必须在其实例化点可见。
   数据流文件**不 include routing、也不 include contract**（这条成立，是方向的承重面，必须保持）：
   路线胶水——暂存检查、发射、交回契约半——一律在 `routing.cuh` 的 `launch_typed_*` 里，
   数据流头只有核与它的启动子。两个移植头曾各自带一个 `launch_route`（因此 include 契约半），
   2026-10-25 上移；`adopt_dynamic_smem` 也在同一次搬进 `launch.cuh`，否则移植头为它还得反向
   够进 radix 命名空间。
   - **同一次搬走的还有"选哪个臂"**：16 位 split 的 `topk` switch、chunks 的 chunk 数派发、
     fp32 split 的 stage1→stage2 序列。理由不是分层美学，是**器件码**：mxcc 的 device pass 会编译
     每个 TU 里**所有函数体**（`__host__` / `static` / `extern template` 都不拦），所以一个
     **非模板 host 函数**只要名字里点到某个具体核（`launch_..._k<512>`），就把那个核拖进
     **每一个 include 该头的 TU** 的 device image。实测：只 include `radix_core.cuh` 的探针
     `.mc_fatbin` 540,669 B、带着全部 12 个 chunk 核；删掉 radix_chunked 那四个非模板派发器后
     26,221 B（空基线）。xcore1600 产物因此 58,559,864 → 18,812,240 B，行 TU 的 fatbin
     628,301 → 118,053 B，核符号集与尺寸逐项不变（77/77）。所以：**派发器只能住在
     `maca_topk.cu` 可见的 `routing.cuh`**，数据流头里不放任何"点到具体实例"的非模板函数。
3. **TU 结构**：手写 TU 恒为 1（`common/maca_topk.cu`）；设备核由生成出来的 `xcore<N>/instantiations/*.cu` 提供，
   每个文件一条显式实例化。`setup.py` 的源清单 = 手写清单 + 生成器的 `family_sources(family)`
   （直接调用生成器函数，不解析它的打印输出——清单只有一份，没有第二份可漂移）。
4. **路由即表**：所有形状判据（哪条数据流、几个 chunk、什么块宽）集中在 `routing.cuh`，纯函数、可读、可测；
   数据流文件里不出现形状门。
5. **产物与平台不变**：每族一件产物、`-offload-arch` 与 `-DDEEP_SELECT_ARCH` 一一配对、`sm_count` 运行期读设备——
   这些是既定契约，重构不动它们。
6. **实例化清单可数**：`ls instantiations/*.cu | wc -l` 与 `nm | grep -c <kernel>` 两个数必须对得上；
   路由/阈值改动导致的实例增减必须在提交信息里报出这两个数。

## 5. 阶段（每阶段一次提交，各自带收据）

| 阶段 | 内容 | 收据（§6） | 依赖 |
|---|---|---|---|
| P1 | 删 `radix_core.cuh` 里岛外零引用的静态-k 族（≈700 行）+ 重写描述它的注释 | 全绿 | 已落地 `e16138c` |
| P2 | `maca_topk.cu` 拆出 `contract.cuh` + `routing.cuh`；dg 两路的发射归位数据流文件 | 全绿 | 已落地 `d8470ca` |
| P3 | `radix_core.cuh` 按 dtype 拆成 `radix_common` / `radix_row_bf16` / `radix_row_fp32` / `radix_chunked`；`radix_core.cuh` 留作伞头 | 全绿 | 已落地 `b02d872` |
| P4 | `dg_*` → `topk_coarse12` / `topk_chunks`（文件、namespace、调用点、`ref/` 臂、文档） | 全绿 | 已落地 `2a795b1` |
| N | 命名空间整合（§3b）：`rk`→`topk_select_radix`、`topk_select_radix::coarse12`→`topk_select_coarse12`、`topk_select_radix::chunks`→`topk_select_chunks`、`deep_select_common`→`topk_select_common`、`deep_select_maca`→并入 `deep_select` | 全绿（rename 口径：demangle 名集模改后全等 + size 全等 + 冒烟） | 已落地 `83b9b08` |
| P5 | 生成器 + 配置元组模板 + `instantiations/*.cu` + 表驱动派发 + `setup.py` | **内核符号集不变**（demangle 后配置集逐项相同：89=89、两侧各 0）+ 文件数==核数 + 套件绿（`nm` 全表预期变动：host 派发 328→835/族） | 已落地 `ebb1117` |
| P6 | python 侧分文件：`interface.py`（公开契约）+ `_backends.py`（torch / deep_gemm 参考后端）+ `_binding.py` / `_log.py` | 套件绿 + `inspect.signature` 不变 | 已落地 `559b845` |

**P3 与 P4 可并行，两者唯一的潜在交界是 `maca_topk.cu` 的 include 块**（`:50-52`）：伞头让 P3 完全不碰这个文件，
P4 在其中只改 dg 的两行。**N 排在 P3 之后、P2 之前**：它在 P3 搬完的每一行上改名一次，且必须先于 P2
（P2 的新文件要生在新命名空间里；否则 `using topk_select_common::…` 这类声明会留下两代拼写）。
P2 重写 `maca_topk.cu`（拆出契约半与选路），因此在 P4 与 N 之后。
P5 在 P2/P3 之后（它改的是它们刚整理出来的发射层）；P6 独立。

## 5b. P5 的元组表（实测，2026-10-12，供生成器照抄）

行核族的模板头是 `topk_kernel_radix<ValueT, OutIdxT, BLOCK, SI, RV, SV, kPreSelected>`
（`maca_topk.cu:474`），今天的实例化全部由 `launch_typed_radix`（`:949`）的 if-链枚举出来：

| 轴 | 取值 | 出处 |
|---|---|---|
| `ValueT` | `maca_bfloat16` / `float` | 入口 dtype 检查 |
| `OutIdxT` | `int32_t` / `int64_t` | 入口 out-idx 检查 |
| `BLOCK` | 16 位行 `{topk_select_radix::kBlockSize(512), topk_select_radix::kLongRowBlockSize(1024)}`；fp32 行只有 512 | `maca_topk.cu:964-974`（`if constexpr` 只给 bf16 编宽块臂） |
| `SI,SV,RV` | `{(1,1,1),(1,0,1),(1,0,0),(0,0,1),(0,0,0)}` 五种 | `sorted_value ⇒ return_value`，`launch_typed_radix` 的 `run(T,T,T)` 链 |
| `kPreSelected` | `false` / `true` | 行路 vs 分块路的预选臂（`launch_typed_radix` 的 `preselected`） |

**2 × 2 × (2 或 1) × 5 × 2 = 60**，与产物里 `nm | grep -c topk_kernel_radix` 的 60 对得上——
生成器的第一张表就是它，文件命名照官方把元组写进文件名：

```
instantiations/value_bf16_outidx_i32_block_512_si1_rv1_sv1_pre0.cu
instantiations/value_fp32_outidx_i64_block_512_si0_rv0_sv0_pre1.cu
```

P5 的另一半是**把模板改成配置结构**（`template <typename Cfg> __global__ void topk_kernel_radix(...)`，
`Cfg` 带 `ValueT/OutIdxT/BLOCK/SI/RV/SV/PRE` 七个成员），host 侧按表查
`{dtype, outidx, block, flags, pre} → 发射器`。

**实测把"被显式实例化的单位"从核改成了发射器**：在别的 TU 里对 `__global__` 模板本身做显式实例化，
**链接后会跑到另一个核**——探针（A TU `k<Cfg><<<1,1>>>`、B TU `template __global__ void k<Cfg>(…)`）
的返回值来自另一个模板：mxcc 为每个核发射 `__device_stub__`，launch 解析到的是**发射方 TU 的那个桩**。
所以每个生成文件实例化的是一个 host 函数 `launch_radix_config<Cfg>`，**函数体持有该核唯一的
`<<<>>>`**——设备码与唯一发射点同 TU。同一枚硬币的另一面：头文件里的**非模板** `__global__` 被
两个 TU include 就是 `__device_stub__` 重复定义（链接错误），因此树里每个核都是模板——含原本不是的
五个（split 两个 stage 2、coarse12、`nan_abort_kernel`），各自以自己既有的
块宽/线程数为模板参数（1024 / 512 / 640 / 128）。其余三族（split 两阶段 /
coarse12 / chunks）的元组轴同一步里从各自发射点枚举，逐族一张表、逐元组一个文件：**87**
（row 60 / split 12 / coarse12 1 / chunks 12 / nan 2）。

表的两份副本（`contract.cuh` 的 X-macro 清单、生成器的文件表）由**收据**对齐，不由约定：
第一版 X-macro 把可达的 `(SI,RV,SV)=(1,1,0)` 抄成了不可达的 `(0,0,1)`，产物**照常链接**
（共享库容忍只有派发表引用的符号缺失），是 `nm -D --undefined-only` 里我方名字空间必须为 **0**
这一条把它抓出来的（当时 12 个 `launch_radix_config<…>` 未定义）。而漏抄的元组在**运行期**查不到行，
所以 `launch_typed_radix` 对查空的处置是**报出键值并 `abort`**，不是静默返回——查空意味着本镜像
服务不了这次调用，没有正确输出可给。

## 6. 收据协议（所有阶段共用）

```bash
# 每个阶段自己的收据（便宜）：
CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh        # rc=0
nm -S --defined-only deep_select/deep_select_maca_xcore<N>.so  # 与改前对照（判据见下）
git status --porcelain                                          # 干净（不留残留）

# 整轮收尾（P3/P4/P6… 全部落地 main 之后）跑一次，不按阶段跑：
CUDA_VISIBLE_DEVICES=<独占的一张卡> ./run_test.sh --test       # 339 pass / 0 check_fail / 0 crash
```

- **测试节奏：全量套件按"轮"跑，不按阶段跑**（用户 2026-10-12：「不需要每次修改都全量测试，
  效率太低」）。阶段自身的收据 = 构建 rc=0 + nm 判据 + 该阶段特有的**便宜**检查（python 层：
  `inspect.signature` 与导出表不变；冒烟用 `tests/test.py --sample 50`）。整轮收尾在 main 上跑
  一次 339 格作为共同门。提交信息如实写"全量套件留待轮收尾"，不声称未跑的测试。

- **搬迁 / 改名（码流未变）**：`nm -S --defined-only` 按**名字 → (size, type)** 比，存活符号逐项相同。
  无 size 的符号（`GCC_except_table*`、CRT 桩）按**去数字后缀的基名**比——编号与地址随布局移动，不算差异。
  `.so` 字节**不**声称相同（mxcc 把源行号编进立即数，换文件必然移动）。
- **删除不可达代码**（P1 `e16138c` 实测）：`nm` **不是**逐行相同——被删的 `__global__` 即使在 host 侧
  不可达也被发射进产物（实测 19 条 `__device_stub__topk_bf16_kernel_*`，产物 6,464,392 → 5,979,112 B）。
  判据是 **removals-only**：删除集恰好是被删族、**零新增**、存活符号除编译器发射的 `__mc_module_ctor`
  （登记被删桩的模块构造器，实测 5066 → 4192 B）外 size/type 不变。可达性由"每个被删项都是定义，
  任何活引用都会是编译错误"保证。
- P5 例外：host 派发从 switch 改成表，`nm` 全表会动（328 → 835 sized 符号/族），判据换成
  **设备核配置集不变**：`nm --defined-only` 的 `__device_stub__` 行 demangle 后，行核按**去
  `TopkRadixConfig<…>` 包装**比、六个新模板核按基名比（实测 89=89、两侧各 0 独有、三族一致），
  加上**文件数 == 核数**（89），加上 `nm -D --undefined-only` 里我方名字空间为 **0**
  （只有派发表引用的符号缺失不报链接错——上一条里那次抄错就是这条收据抓的）。
- 触及选路/阈值/数据流的阶段另跑 `./run_test.sh --perf`；纯搬迁不跑。
