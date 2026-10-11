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
    ├── common/primitives.cuh               跨族原语（已落地）
    └── xcore1000/                          族层
        ├── maca_topk.cu                    唯一手写 TU：include + host 入口 + 按表派发
        ├── contract.cuh                    契约半：RowParams / 常量 / 键与 NaN / emit / 布局
        ├── routing.cuh                     选路：形状 → 数据流（纯函数，一处可读）
        ├── radix_row_bf16.cuh              bf16 行：radix 两趟 + skip 表 + overflow
        ├── radix_row_fp32.cuh              fp32 行：+ 溢出重扫 + k2048 变体
        ├── radix_chunked.cuh               两条 dtype 的分块 split/merge（阶段核 + 发射）
        ├── topk_coarse12.cuh               deep_gemm 路线一（官方核名）
        ├── topk_chunks.cuh                 deep_gemm 路线二（官方核名）
        └── instantiations/*.cu             生成物：一个配置元组一个文件
```

## 4. 规则

1. **命名**：官方名优先（`refine_bin` / `warp_cumsum_histogram<Bins>` / `bf16_is_nan` 已落地）；两条 deep_gemm
   路线用官方核名 `topk_coarse12` / `topk_chunks`（文件、namespace、注册名一致）；**代码里不再有 `dg` 前缀**
   （`rk::dg12` → `rk::coarse12`，`rk::dgchunks` → `rk::chunks`）。无官方名的一律用角色名，不用血统名。
2. **依赖方向单向**：`common ← contract ← 数据流 ← routing ← TU(入口)`；python 侧
   `interface → _binding → .so`。数据流文件**不 include routing**，routing 不 include 具体发射实现。
3. **TU 结构**：手写 TU 恒为 1（`maca_topk.cu`）；设备核由生成出来的 `instantiations/*.cu` 提供，
   每个文件一条显式实例化。`setup.py` 的 `SOURCES` = 手写清单 + 生成器打印的清单。
4. **路由即表**：所有形状判据（哪条数据流、几个 chunk、什么块宽）集中在 `routing.cuh`，纯函数、可读、可测；
   数据流文件里不出现形状门。
5. **产物与平台不变**：每族一件产物、`-offload-arch` 与 `-DDEEP_SELECT_ARCH` 一一配对、`sm_count` 运行期读设备——
   这些是既定契约，重构不动它们。
6. **实例化清单可数**：`ls instantiations/*.cu | wc -l` 与 `nm | grep -c <kernel>` 两个数必须对得上；
   路由/阈值改动导致的实例增减必须在提交信息里报出这两个数。

## 5. 阶段（每阶段一次提交，各自带收据）

| 阶段 | 内容 | 收据（§6） | 依赖 |
|---|---|---|---|
| P1 | 删 `radix_core.cuh` 里岛外零引用的静态-k 族（≈700 行）+ 重写描述它的注释 | 全绿 | 已批 |
| P2 | `maca_topk.cu` 拆出 `contract.cuh` + `routing.cuh`；dg 两路的发射归位数据流文件 | 全绿 | P1 |
| P3 | `radix_core.cuh` 按 dtype 拆成 `radix_row_bf16` / `radix_row_fp32` / `radix_chunked` | 全绿 | P1 |
| P4 | `dg_*` → `topk_coarse12` / `topk_chunks`（文件、namespace、调用点、`ref/` 臂、文档） | 全绿 | P1 |
| P5 | 生成器 + 配置元组模板 + `instantiations/*.cu` + 表驱动派发 + `setup.py` | **内核符号集不变** + 文件数==核数 + 套件绿（`nm` 全表预期变动：host 派发改成表） | P2/P3 |
| P6 | python 侧三层分文件（契约 / 后端 / 装载），`__init__` 导出不变 | 套件绿 + `inspect.signature` 不变 | 独立 |

P1/P4 互不冲突（文件集不同），可并行；P2 与 P3 各自独立（文件集不同），可并行；
P5 在 P2/P3 之后（它改的是它们刚整理出来的发射层）。

## 6. 收据协议（所有阶段共用）

```bash
CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh        # rc=0
nm -S --defined-only deep_select/deep_select_maca_xcore<N>.so  # 与改前逐行 diff
CUDA_VISIBLE_DEVICES=<独占的一张卡> ./run_test.sh --test       # 339 pass / 0 check_fail / 0 crash
git status --porcelain                                          # 干净（不留残留）
```

- **逐行相同**是"码流未变"的判据；`.so` 字节**不**声称相同（mxcc 把源行号编进立即数，换文件必然移动）。
- P5 例外：host 派发从 switch 改成表，`nm` 全表会动，判据换成"**设备核符号集不变**"（同 mangled 名同 size）
  加上"文件数 == 核数"。
- 触及选路/阈值/数据流的阶段另跑 `./run_test.sh --perf`；纯搬迁不跑。
