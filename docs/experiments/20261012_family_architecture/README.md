# 家族架构的文档对账：三处文实不符 + 一处按裁定改代码（2026-10-12）

一句话：本轮不动任何行为。三处把**写出去了的、与代码矛盾的文档**改回现在时，每处的
判据都是一条可重跑的命令；第四处（`CUCC_TARGETS` 的默认值）按用户 2026-10-12 的裁定
**反过来**——那段文档是设计面，要改的是脚本，见 §1。改完
`tools/comment_gate.py --worktree` 仍然 `IDENTICAL CODE`（`csrc/structs.h` 只动注释）。

改动落在：`csrc/structs.h`（家族表注释块）。`CLAUDE.md` 的 `CUCC_TARGETS` 相关段落按
§1 的裁定重写（构表 + `native` 的解析语义），不是"改回代码"。**未提交**，留在工作树。

## 1. `CUCC_TARGETS` 的默认值：**裁定**——`develop.sh`/`install.sh` 默认 `native`，`build.sh` 全量

**事实**（可重跑）：三个脚本当前是同一个默认，且 `setup.py` 当前**拒绝** `native`。

```console
$ grep -n 'CUCC_TARGETS:-' *.sh
build.sh:15:export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"
install.sh:18:export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"
develop.sh:19:export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"
```

**裁定（2026-10-12，用户）**：`develop.sh` 与 `install.sh` 默认 `CUCC_TARGETS=native`
（两个都作用于*这台机器*），`build.sh` 默认全量（可装运的 wheel 必须一份 image 服务
每个家族）。于是 `CLAUDE.md` 的「`develop.sh` 与 `install.sh` 说 `native`」是**设计面**，
不是漂移——错的是脚本（`67864ad` 把三处都改成了全量列表），该段引用的
`grep -n 'CUCC_TARGETS:-' *.sh` 在脚本改回后重新成立。同理，`setup.py:25-27` 那句
「`develop.sh` / `install.sh` narrow it to `native`」在裁定下是对的（见 §7）。

**连带（另一路落地）**：`setup.py` 必须从"拒绝 `native`"变成"接受 `native` 并解析为
本机 family"，否则默认值不可执行。解析走加载器已有的 sm→family 映射
（`_binding.device_family_suffix()`），无设备时大声失败并给出显式拼写的替代；
`native` 与该 family 拼写在 `-offload-arch` 上等价（C500 实测 byte-identical）。
记录见 `docs/experiments/20261012_setup_family_detection/`。

**本记录对 `CLAUDE.md` 的处理**：按裁定重写该变量的三段（构表改为 `native` / 全量 /
`native`；"每个拼写原样透传"改为"`native` 是唯一被解析的一条"；家族表一节里
"`setup.py` 拒绝 `native`"改为"构建期解析为本机 family"）。

## 2. `is_c600u_ap_count` 是 AP 数的**集合**，不是家族键 —— 1500 行也满足它

**矛盾**：`csrc/structs.h` 的家族表注释说「每条被测量规则的**性能**半边由它被测的
设备把守（`kF32Coarse12MeasuredSmCount`、`is_c600u_ap_count`）」。这对
`kF32Coarse12MeasuredSmCount` 成立（`sm_count != 104` 即拒，C500 专属），对
`is_c600u_ap_count` **不成立**：

```console
$ sed -n '761,766p' csrc/maca_kernels/xcore1000/maca_topk.cu
// **A set, not a count: the C600U ships at 28 APs and at 32, and both are
// ordinary parts.**  Every gate fitted on that family takes the set.
inline bool is_c600u_ap_count(uint32_t sm_count) {
    return sm_count == 28 || sm_count == 32;
}
```

`28 || 32` 命中的不只是 C600U：**同一文件的 1500 行就是 `ARCH_SM_COUNT = 28`**
（`csrc/structs.h:143`），即 C600 也满足它。被它放行的三条阶梯全部按 C600U 拟合：

| 常量 | 位点 | 拟合对象 |
| --- | --- | --- |
| `kC600UNarrowRowBatches = 256` | `maca_topk.cu:790`（`radix_block_for`，用点 `:799`） | C600U（`:771-790` 的 batch×vocab 表） |
| `kC600UChunkedVocabPerBatchSq = 6400` | `maca_topk.cu:875`（`chunked_bf16_applies`，用点 `:942`） | C600U（`docs/experiments/20261010_c600u_split_crossing/`） |
| `kF32Coarse12C600UNarrowProduct / WideProduct = 9216 / 49152` | `maca_topk.cu:1466-1467`（用点 `:1578`） | C600U（`maca_topk.cu:1387-1443` 自述的 2026-09-20 本机实测；`docs/experiments/` 下无对应目录） |

这棵树里**没有 C600 的测量**：`ls perf_data/` 只有 `MetaX_C500` 与 `MetaX_C600-U`，
`docs/experiments/` 下没有 C600 的目录。所以现状是一句话能说完的：**C600 由一套在
兄弟部件上拟合的数服务。**

**待决（不在本轮改）**：那个谓词该按 **AP 数集合**（现状）还是按**家族**取值 ——
即 C600 该不该吃 C600U 的阶梯。另一路会话在产出分析，记录见
`docs/experiments/20261012_setup_family_detection/` 与
`docs/experiments/20261012_family_source_selection/`。本轮只让注释陈述现状，不改
行为、不预设结论。

**改后**：把「由它被测的设备把守」拆成两半 —— C500 阶梯由 `kF32Coarse12MeasuredSmCount`
（104）把守、C600U 阶梯由 `is_c600u_ap_count` 把守 —— 并新增一段说明该谓词是 AP 数集合、
会连带命中 1500 行的 28-AP C600、三条阶梯都是 C600U 拟合、C600 在本树无测量。
按房子的注释纪律：说「现在是什么」，不写「这是个 bug」，也不写改动史。

## 3. `chunked_f32_applies` 仍无家族把守 —— 该句属实，保留

**核验**：

```console
$ sed -n '1827,1836p' csrc/maca_kernels/xcore1000/maca_topk.cu
inline bool chunked_f32_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0) return false;
    if (params.vocab_size < kF32ChunkedMinVocab) return false;
    if (batches > kF32ChunkedMaxBatches) return false;
    if (params.topk == 512 || params.topk == 1024) return true;   // measured
    return topk_worth_splitting_f32(params, batches);
}
```

整个函数体内没有 `sm_count`/family 子句（对照 `chunked_bf16_applies` 的
`is_c600u_ap_count(params.sm_count)` 臂）。辅助侧也只有 `wave_filled_chunks` 读运行时
`sm_count`（`maca_topk.cu:908`），band 的常数（`kF32ChunkedMinVocab = 32768`、
`kF32ChunkedMaxBatches = 4096`）是定值。故原句
「still has no such guard, so a 1600 image routes its fp32 split on the C500 chunk *band*」
成立，未改。

## 4. `_binding.py` 的 sm 拼写：三方一致，未改

**核验**（`deep_select/_binding.py:88-94` 声称 sm 拼写表是 `deep_gemm/utils/arch_config.py`
的，且 `xcore1000`=sm80、`xcore1500`=sm86、`xcore1600`=sm87/88/89）：

```console
$ sed -n '69,108p' /home/compiler_gfx/tilelang/mcDeepGEMM/deep_gemm/utils/arch_config.py
    Xcore1000Family: sm_spellings = (80,)      # C500
    Xcore1500Family: sm_spellings = (86,)      # C600
    Xcore1600Family: sm_spellings = (87, 88, 89)   # C600U/N300U
```

三处互相一致，方向也一致（`device_family_suffix()`：80→`_xcore1000`、86→`_xcore1500`、
87..89→`_xcore1600`），`csrc/structs.h` 的家族行不涉及 sm 拼写、无矛盾。**保留原句**。

说明（不是矛盾，供读者）：本仓必须能在 `deep_gemm` 缺席时独立导入，所以那张表**不能
被 import**，`_binding.py` 里的映射是刻意的副本；那句话说的是"拼写的定义处"，
不是"运行时读它"。

## 5. `build.sh` 一段里关于「mcDeepGEMM 的清单」的两句不成立（team-lead 更正）

**矛盾**：`CLAUDE.md` 说 `CUCC_TARGETS` 是「mcDeepGEMM 的 shape」，又说别名
「`mxcc` rejects several of those spellings outright」。两句都把外仓的构建当成了读这个
变量的人 —— 它不读。

```console
$ sed -n '21p' /home/compiler_gfx/tilelang/mcDeepGEMM/build.sh
export CUCC_TARGETS="${CUCC_TARGETS:-native,xcore1000,xcore1008,xcore1500,xcore1502,xcore1520,xcore1600,xcore1610,xcore1620}"

$ grep -n 'CUCC\|native' /home/compiler_gfx/tilelang/mcDeepGEMM/setup.py
（无输出）

$ sed -n '77,79p' /home/compiler_gfx/tilelang/mcDeepGEMM/csrc/jit/device_runtime.hpp
    std::string get_offload_target() {
        return DeviceCapability::name(family());
    }
```

外仓 `build.sh:21` 确实导出该变量（前缀 `native,`），但它的 `setup.py` 一个字都不读，
它的 JIT 目标来自**设备**（`offload_target = get_offload_target()`，
`csrc/jit/compiler.hpp:288`），而 `DeviceCapability::name(family())` 只会产出
`xcore1000` / `xcore1500` / `xcore1600`（`csrc/utils/device_capability.hpp:81`）。所以
在那一边**没有任何拼写被解析、映射或拒绝**。

**本仓 `setup.py` 是那张清单唯一被读的地方**，拒绝发生在**解析期**
（`_family_of_target`），根本走不到编译器；而且「未知拼写」与「名字指向表中已存在的
家族」走的是同一条拒绝路径（`setup.py` 正被另一路改，行号会动，按函数名与报错文字找）：

```console
$ grep -n 'def _family_of_target\|names family {family}' setup.py
72:def _family_of_target(target, this_dir):
96:            f"CUCC_TARGETS entry {target!r} names family {family}, which has no "

$ sed -n '94,97p' setup.py
        raise RuntimeError(
            f"CUCC_TARGETS entry {target!r} names family {family}, which has no "
            f"row in csrc/structs.h (AP count and per-AP shared memory); add "
            f"one there before building it"
```

即 `xcore1008` 得到的是「family 1008 在 `csrc/structs.h` 里没有行」—— 而 1008 不是家族，
是 1000 的别名，1000 有行。该报错**误导**，且它是解析期的、不是 `mxcc` 的。

**状态**：`CLAUDE.md` 该段的末句已被换掉（「`mxcc` rejects several of those spellings
outright」→「an entry this build cannot turn into a family is refused at parse time, before
the compiler is reached」）——拒绝发生在本仓 `setup.py` 的解析期（`_family_of_target`），
不在 `mxcc`；该句对"别名应当落到它命名的家族"保持开放，两种结论下都成立。核验过的事实
就是上面的 console 块。段内「两个同族 target 建同一个产物」那半句在 dedupe 落地前仍然
成立；整段的收尾改写与三个脚本的默认值、`setup.py` 对 `native` 的处理，以及
`docs/experiments/20261012_setup_family_detection/` 的实测**同一次提交**落地。

## 6. 收尾核对

```console
$ python3 tools/comment_gate.py --worktree
  not gated        CLAUDE.md  (not a code file)
  IDENTICAL CODE   csrc/structs.h

2 files: 1 documentation-only, 0 differ in code, 1 not gated (not code)
RC=0
```

`csrc/structs.h` 是 `IDENTICAL CODE`：本轮只动注释，没有代码改动。不构建、不跑套件、
不提交 —— 本记录与改动一并留在工作树。

（此后 `setup.py` 因另一路的 `SOURCES_BY_FAMILY` 重构报 `CODE DIFFERS`，属预期；
`csrc/structs.h` 仍 `IDENTICAL CODE`。）

## 7. `setup.py:25-27` 的同一句话：**按裁定是对的**，要改的是代码

```console
$ sed -n '25,27p' setup.py
# What this file builds when `CUCC_TARGETS` says nothing.  `build.sh` sets the
# same list, and that is what a wheel has to carry -- one board's image cannot
# be shipped to another; `develop.sh` / `install.sh` narrow it to `native`.
```

按 §1 的裁定，这句话描述的是设计面：`develop.sh` / `install.sh` **应当** narrow 到
`native`。所以这里没有文档要改——要改的是脚本本身与 `setup.py` 对 `native` 的处理，
归 `setup-family` 一路（记录见 `docs/experiments/20261012_setup_family_detection/`）。
本条只作为"同一处设计在四个文件里有四份副本"的清单：`CLAUDE.md` 的三段、这里的一段。
