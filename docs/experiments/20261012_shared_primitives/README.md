# 公共原语抽取：NaN / 保序键 / warp 直方图后缀和，三份副本并成一份（2026-10-12）

一句话：`xcore1000/` 的三个数据流——`rk` 的 radix 行（`radix_core.cuh`）、`rk::dg12` 的
coarse12（`dg_coarse12.cuh`）、`rk::dgchunks` 的 chunked merge（`dg_chunks.cuh`）——外加契约半
`maca_topk.cu`，各自带着同一批位运算的一份副本（fp32 NaN 判定 4 份、fp32 保序键 4 份、
warp 直方图后缀和 2 份函数体逐字节相同）。按用户 2026-10-12 的裁定抽到
**`csrc/maca_kernels/common/primitives.cuh`**（族目录之上、平台层之下），函数名按同日裁定
**取官方名**（§0），改后三重构建 rc=0，**三个产物的符号表与改前逐行相同**（345 行、
地址与大小不动、`topk_kernel_radix` 各 60），套件 339 pass / 0 check_fail / 0 crash
（349 格，10 格 OOM skip，rc=0）。

判据是**逐字节相同**：只并真正重复的（用户裁定的范围），并之前先把两份函数体逐行 diff 过。

## 0. 命名 —— 官方名优先（用户 2026-10-12 裁定）

| 抽出的东西 | 现名 | 官方出处 | 本树旧名 |
|---|---|---|---|
| fp32 保序键 | `refine_bin(float)` | deep_gemm `fp32_topk.cu:155`，值形态、函数体逐字节相同 | `ordered_key_f32` / `rk::float_to_uint32` / `maca_topk::key_of<float>` |
| 16 位保序键 | `refine_bin(maca_bfloat16)`（重载） | 无值形态对手；位形态是上游 `topk_select_common::distort<uint16_t>`（`bit_utils.cuh:30`），代数相同 | `ordered_key_bf16` / `rk::bf16_to_uint16` |
| 16 位 NaN 判定 | `bf16_is_nan(uint16_t)` | 上游 `topk_select_common::bf16_is_nan`（`bit_utils.cuh:76`），函数体逐字节相同 | `is_nan_bits16` |
| warp 直方图后缀和 | `warp_cumsum_histogram<kBins>` | deep_gemm `fp32_topk.cu:187`，模板形态；调用点也照官方写 `<kRadix>` / `<kFineBins>` | `warp_cumsum_histogram_256` |
| fp32 NaN 判定 | `is_nan_bits32`（本树名，未改） | **无官方对手**：deep_gemm 的 NaN 检查在 torch 层，上游用 `v != v` | — |
| 文件名 | `common/primitives.cuh`（未改） | **无官方对手**：上游没有"族间共享"这一层的文件 | — |

两处取法是判断，理由记在这里：

- **`refine_bin` 取重载而不是 `distort`。** 官方对同一个变换有两个拼法：deep_gemm 的
  `refine_bin(float)`（值形态）与上游的 `distort<UIntValueT>`（位形态）。本树调用点手里拿的是
  **值**（22 个 bf16 位点、15 个 fp32 位点），走 `distort` 就要每处先自己取位，且要换成上游那份
  三元 select 的函数体——而本树 `__MACACC__` 下的 xor 写法是既有的、**避开发散分支**的选择
  （两者值相同）。故 fp32 用官方原名的原形态，16 位用同名重载，22 个位点的形态一个不动。
- **`block_cumsum_histogram_1024`（`dg_chunks.cuh`）保持原地、保持本树名。** 它就是 deep_gemm
  模板的 **coarse arm**（官方名同为 `warp_cumsum_histogram<kCoarseBins>`），但树上只有这一份、
  **没有第二份副本**，按"只并完全相同者"的范围裁定不并进共享模板（并进去要引入一个按 bin 数
  分派的 `if constexpr`，把两个形状的常数搬进共享层）。要合并就是一次独立改动。

`distort` 本身没有进共享层：**没有调用点**——本树没有一处需要"位进位出"的形态。

## 1. 改前的副本清单（可重跑；下面的名字是**抽取前**的）

```console
$ cd csrc/maca_kernels/xcore1000
$ grep -cE '\bis_nan_bits\b'   radix_core.cuh        # 5（1 定义 + 4 用点）
$ grep -cE '\bis_nan_value\b'  dg_coarse12.cuh dg_chunks.cuh maca_topk.cu
dg_coarse12.cuh:5   dg_chunks.cuh:5   maca_topk.cu:7
$ grep -cE '\bfloat_to_uint32\b' radix_core.cuh       # 15
$ grep -cE '\brefine_bin\b'      dg_coarse12.cuh dg_chunks.cuh   # 13 / 7
$ grep -cE '\bbf16_to_uint16\b'  radix_core.cuh       # 22
```

| 原语 | 副本 | 位点（抽取前） |
|---|---|---|
| NaN 位型判定 fp32 | **4** | `rk::is_nan_bits`(radix_core:150)、`dg12::is_nan_value`(:96)、`dgchunks::is_nan_value`(:106)、`maca_topk::is_nan_value<float>`(:198) |
| NaN 位型判定 bf16 | 2 | `rk::is_nan_bits16`(:156)、`maca_topk::is_nan_value<maca_bfloat16>`(:204) |
| fp32 保序键 | **4** | `rk::float_to_uint32`(:168)、`dg12::refine_bin`(:102)、`dgchunks::refine_bin`(:112)、`maca_topk::key_of<float>`(:162) |
| bf16 保序键 | 2 | `rk::bf16_to_uint16`(:173，`__MACACC__` 下无分支 xor)、`maca_topk::key_of<maca_bfloat16>`(:168，三元式) |
| warp 直方图后缀和 | 2 | `dg12::warp_cumsum_histogram_256`(:126)、`dgchunks::warp_cumsum_histogram_256`(:144) |

两份后缀和的函数体**逐字节相同**，只有常数名（`kRadix` / `kFineBins`，都是 256）与签名折行不同——
把常数名统一后 `diff` 只剩签名行：

```console
$ diff <(awk '…warp_cumsum_histogram_256…' dg_chunks.cuh  | sed 's/kFineBins/kBINS/g') \
       <(awk '…warp_cumsum_histogram_256…' dg_coarse12.cuh | sed 's/kRadix/kBINS/g')
1,2c1
< __device__ __forceinline__ void warp_cumsum_histogram_256(
<     int (&histogram)[kBINS + kHistogramPadding])
---
> __device__ __forceinline__ void warp_cumsum_histogram_256(int (&histogram)[kBINS + kHistogramPadding])
```

`radix_core.cuh:144-149` 当时的注释自己记着这件事：「It is a copy, not a call…**nothing ties
them**」——本轮的收益之一就是把那句删掉。

## 2. 抽了什么、什么**不**抽

抽（`deep_select_common::` 一个定义；名字见 §0）：

| 名字 | 形态 |
|---|---|
| `is_nan_bits32(uint32_t)` / `bf16_is_nan(uint16_t)` | 位型判定（调用者手里是 bits 时用） |
| `is_nan_value<ValueT>`（fp32/bf16 两个特化） | 值形态，供 `row_has_nan` 这类泛型调用 |
| `refine_bin(float) -> uint32_t` / `refine_bin(maca_bfloat16) -> uint16_t` | 保序键，按值类型重载 |
| `warp_cumsum_histogram<kBins>(int (&)[kBins + 1])` | 单 warp、`kBins / 64` 个 bin/lane 的后缀和 |

bf16 键统一到 `radix_core` 那份的 `__MACACC__` xor 写法（避开发散分支，本文件族 22 个位点
原本就是它）；`maca_topk` 的三元式版本删掉——两者**值相同**，符号表显示码流未变（§4）。

`maca_topk.cu` 的 `emit_ordered` 是全树唯一一处**泛型**取键（`ValueT` 两种），重载后拼写为
`~static_cast<uint32_t>(refine_bin(...))`：先宽到 32 位再取反是**语义的一部分**——`~` 落在
`uint16_t` 上会经 `int` 提升并符号扩展进 pack 的高位。注释写在该行。

不抽，因为**不是同一个操作**（不是"暂时没抽"）：

- `rk::float_to_uint8` / `bf16_to_uint8`：8 位粗层，且 fp32 那条要先经 fp16 舍入；
  `dgchunks::coarse_bin`：half 键再 `>>6` 的 10 位粗层。三者形状不同（输出宽度、舍入路径），
  合起来只会造出一个参数化的怪物。
- `rk::run_cumsum_warp`：256 bin 摊在**整个 block** 的 warp 上，带二级扫描与 `[257..264]`
  padding 约定；与共享的（单 warp、4-per-lane）是两种算法。名字像，合并不了。
- `dgchunks::block_cumsum_histogram_1024`：deep_gemm 模板的 coarse arm，无第二份副本（§0）。

## 3. 放置与接入（用户裁定：`maca_kernels` 下）

```
csrc/maca_kernels/
├── common/primitives.cuh     ← 新：族之上、平台之下，每族编同一份
├── xcore1000/{maca_topk.cu, radix_core.cuh, dg_coarse12.cuh, dg_chunks.cuh}
└── xcore1600/…               ← 上游移植，不在构建里
```

- **接入用相对 include**（`#include "../common/primitives.cuh"`），不动 `setup.py` 的
  `include_dirs`：引号 include 按**含该指令的文件所在目录**解析，所以谁 include 这四个头都一样。
  旁证：`ref/coarse12_port/xcore1000_dg_coarse12.cu:14` 直接 include `dg_coarse12.cuh`，
  改后照样编（那是 `ref/` 的对照臂，不属本树的构建）。
- **依赖方向**：本头只吃 `cstdint` / `cuda_runtime.h` / `maca_bfloat16.h`，不碰 kerutils /
  cute / cub / 任何族宏——这是"将来别的树也能消费它"的判据，也是它**不能**和 xcore1600 的
  `common_parts.cuh`（include `<kerutils/kerutils.cuh>` 等）合并的理由：那会把 kerutils 拖进
  共享层，而 kerutils 是那棵树自带的 vendored 库。
- 命名空间用一个新名 `deep_select_common`，四个文件各加 `using` 声明把名字拉进本命名空间，
  调用点保持短名。判据：它不属于 `rk`（radix 核心，是 xcore1000 的命名空间），也不属于
  `deep_select_maca`（契约半）。

## 4. 验证

**构建**（`CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh`）：rc=0，
三产物 6464392 / 6468488 / 6468488 B，`topk_kernel_radix` 各 60（改名前后的两次构建都是）。

**符号表对照**（改前基线取自上一步落地时的产物，改后同机同工具链重建）：

```console
$ nm -S --defined-only deep_select/deep_select_maca_xcore<N>.so > after.txt   # 对三个族各一次
$ diff before.txt after.txt
（空）
```

三个族**都是逐行相同**（345 行；每个符号的地址与大小不动）。改名前后各跑过一次该对照，两次都过。

**行为门**：`./run_test.sh --test` → 339 pass / 0 check_fail / 0 crash / 10 skip（OOM）
= 349 格，rc=0（产物 md5 `0a519889f65057b1…`，receipt
`results/deepselect_run_20261012_013416.txt`）。改名后重跑同样 339 pass / 0
check_fail / 0 crash（receipt `results/deepselect_run_20261012_014555.txt`，产物 md5
`bf6bae90038df000345d855748e3a84f`）。

**不声称的事**：`.so` 的字节并不相同（mxcc 把源行号编进立即数，换文件必然移动行号，本仓
已实测过同类），所以这里**没有**做 md5 对照——证据是符号表逐行相同 + 套件，而不是字节相等。

## 5. 未做 / 后续

- **xcore1600 的公共部分不合**：`common_parts.cuh` / `bit_utils.cuh` / `utils.cuh` 是上游移植
  自带的一层，依赖 kerutils/cute/cub，且整棵树不在 `SOURCES` 也不在 `include_dirs`——
  本树没有任何构建会编译它。真重复的只有 `distort`（位上与保序键等价，且本树无调用点）；
  把它并掉的动作属于"把 xcore1600 接回构建"那一步（那一步按纪律要在 C600U 上测功能与性能），
  本轮只保证本头对它是可消费的（§3 的依赖方向）。
- **`block_cumsum_histogram_1024` 没进共享层**（§0）：无第二份副本，且并进去要引入按 bin 数
  分派的 `if constexpr`。要合就是一次独立改动。
- **`hist_reg_to_smem` 等 dead 代码**不在本轮范围（属 `radix_core.cuh` 自己的清理）。
- 半精度粗键（`float_to_uint8` / `coarse_bin`）按范围裁定不并；要并时的形态是给
  `primitives.cuh` 加一个 `<格式>_coarse_bin`，两处各取所需——那会动两个在产函数体，
  需要单独一次带套件的提交。

## 6. 复现

```bash
cd csrc/maca_kernels/xcore1000
grep -nE '__device__ __forceinline__ (bool|uint32_t|uint16_t|void|template) (is_nan|bf16_is_nan|refine_bin|warp_cumsum)' *.cu *.cuh ../common/primitives.cuh
# 每个共享名字应只在 common/primitives.cuh 出现一次定义
```
