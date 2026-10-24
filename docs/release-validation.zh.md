# 发布验证矩阵 —— P6 三族交付（2026-10-23）

一次交付候选的完整验证记录：**交付物是什么、跑了哪些门、每道门在哪个二进制上
跑的、结果是什么、哪些没有跑**。数字都在这里；论证在各自的实验文档里，本文只
引用。

本轮的交付范围与正确性门是用户指定的：**三族都要（xcore1000 / xcore1500 /
xcore1600）**，正确性用**大样本 + 定向用例**，完整 105,138 例全表作为发布前最后
一步（见 §6）。

---

## 1. 结论

| 项 | 结论 |
| --- | --- |
| 三族构建 | `BUILD_RC=0`，三个 `.so` 一次产出，约 6 分钟 |
| wheel 安装 | `pip install --force-reinstall --no-deps` 成功；安装后 `_binding` 解析到 site-packages 内的 `xcore1600`，md5 与 wheel 内一致 |
| xcore1600（C600U） | 大样本 8000/8000、契约 73/73、参考后端 70/70、wheel 冒烟 12/12、官方 perf 网格 147/147 |
| xcore1000 / xcore1500 | **只构建，未运行** —— 本会话没有 C500 / C600 设备（见 §4） |
| 代码状态 | 内核源码自 `9ecdeb0` 起未改；本轮唯一的源码提交是派发表搬家（`kRadixTable` 进 `maca_topk.cu`），它只改产物体积，不改任何内核 |
| 产物体积 | **wheel 526,830,133 → 62,998,174 B**（§5.1），三族 artifact 分别 67.5 / 71.5 / 71.5 MB |

一句话：**xcore1600 这一族已经可以用"大样本 + 定向用例"门禁交付；另外两族是
"构建通过、未实测"，不是一个可以声称已验证的状态。**

---

## 2. 交付物身份

构建方式（测试 pod，2026-10-23）：

```bash
export PATH=/opt/conda/bin:$PATH
export MACA_PATH=<MACA SDK 根目录>      # Version:20261017.1436
export MAX_JOBS=16
./clean.sh && ./build.sh            # CUCC_TARGETS 未设置 = 每族一个目标
```

| | |
| --- | --- |
| wheel | `deep_select-1.0.0+maca20261017.1436.9ecdeb0.20261023.174654-py3-none-linux_x86_64.whl` |
| 大小 | 62,998,174 B（60.1 MiB） |
| sha256 | `ec4b180526c1bf81b1e968d252d8ecacb32b6ad7c3598ad57635de23b2369e61` |
| 版本串 | `<base>+maca<MACA_VERSION>.<git-rev>.<timestamp>`，工具链 `Version:20261017.1436`，源码 `9ecdeb0` |
| python / torch | 3.10.10 / `2.10.0+metax20261017.1120` |

版本串里的 git-rev 是 `9ecdeb0`：构建是在工作树上做的，它相对该提交只多 §5.1
的派发表搬家（`contract.cuh` / `maca_topk.cu` / `generate_instantiations.py`
三处），所以 `perf_data` 的两份 manifest 都记 `deep_select_git_dirty: true`。
搬家本身已单独提交（`2f86b0f`），HEAD 与该源码树一致。

wheel 内的三个 artifact（每族一个文件，`_binding.family_suffix()` 按设备选）：

| family | 文件 | 字节 | md5 |
| --- | --- | --- | --- |
| xcore1000 | `deep_select_maca_xcore1000.so` | 67,524,144 | `8a3da52928e1` |
| xcore1500 | `deep_select_maca_xcore1500.so` | 71,542,320 | `fedfcf34f05f` |
| xcore1600 | `deep_select_maca_xcore1600.so` | 71,542,320 | `14801f45326e` |

安装到 `site-packages` 后，`_binding.extension_path("deep_select_maca")` 在
`CUDA_VISIBLE_DEVICES=1`（C600U）下解析到
`.../site-packages/deep_select/deep_select_maca_xcore1600.so`，md5
`14801f45326e`，与 wheel 内一致 —— 即"交付的那个二进制"就是被测的那个二进制。

内核计数是"这次只是搬表、没动内核"的机械证据（`CLAUDE.md` 的 artifact 收据）：
三个 artifact 都定义 **60** 个 `topk_kernel_radix`、**89** 个
`__device_stub__`、**2** 个 `__tvm_ffi_*` 导出，与修复前相同。

### 2.1 修复前的候选（同一份内核源码）

上一轮的 wheel 是 `...9ecdeb0.20261023.161337`，526,830,133 B，三族 artifact
458,450,392 / 590,263,768 / 590,263,768 B（xcore1600 md5 `d24396d18f44`）。
它跑过一轮完整的 `run_bench.sh`（§3.2 的 `20261023_172210`），是本轮 perf 数字
的来源；体积问题的原因与修正在 §5.1。

---

## 3. 门禁（逐条，含命令与结果）

除构建外，全部在 C600U（`MetaX C600-U`，28 AP，device 1）上、对**安装后的
wheel** 跑（checkout 里没有 in-place `.so`，所以四个 runner 的路径规则都指向
site-packages；`tests/contract_test.py` 与本节的工具共享同一条规则）。

| 门 | 命令 | 被测二进制 | 结果 |
| --- | --- | --- | --- |
| 三族构建 | `/root/build_3fam.sh` = `./clean.sh && ./build.sh` | — | `BUILD_RC=0`，3 个 artifact，约 6 分钟 |
| 交付产物冒烟 | `cd /tmp && python <repo>/scripts/wheel_smoke.py` | 安装 wheel `14801f45326e` | **12 passed / 0 failed** |
| 契约与拒绝 | `tests/contract_test.py --backend maca_c` | 安装 wheel `14801f45326e` | **73 passed / 0 failed** |
| 参考后端 | `tests/contract_test.py --backend torch` | 安装 wheel | **70 passed / 0 failed** |
| 大样本正确性 | `python tools/sample_gate.py --sample 8000 --seed 20261024 --max-elems 67108864 --backend maca_c` | 安装 wheel `14801f45326e` | **pass 8000 / check_fail 0 / crash 0 / skip 0 / unsupported 0**，701 s（11.41 cases/s） |
| 官方 perf 网格 | `./run_bench.sh --backends maca_c,torch,deep_gemm` | 安装 wheel `14801f45326e` | `BENCH_RC=0`，8 阶段全过，`maca_c` 147/147、`official_fp32` 30/30（§3.2）|

### 3.1 大样本的抽样口径

`tools/sample_gate.py` 用的是 `tests/test.py` **自己的**表与检查
（`correctness_cases_()` + `run_testcase`），不是副本；差别只在抽样：

* 表在 seed `20261024` 下是 **105,138 例**，其中 **67,122 例** ≤ 2^26 元素
  （平均 1,020 万元素，最大 6,700 万）；**38,016 例**超过上限，被计数而**没有
  跑**（不是静默丢弃）。
* 均匀抽 8000 例不可行：均值约 1e9 元素的行，harness 的检查要多次触碰输入
  （clone、整张 mask、`isnan`），实测 ~0.14 例/s —— 16 小时，不是一个发布前
  会跑的门。这与完整全表在 C500 上的 16–17 h 投影是同一量级的同一个事实。
* 抽中的每一例的 `p.seed` 被固定，所以同一条命令重跑是同一批数据（官方循环
  对 `seed=-1` 用进程计数器，重跑换数据 —— 对扫表无所谓，对发布记录不行）。
* 同一条命令在本轮的两次实测：修复后的 wheel **701 s（11.41 例/s）**，修复前
  的 wheel **606 s（13.20 例/s）**、在 checkout 的 artifact 上 753 s —— 三个
  数字是同一批 8000 例，差异是机器状态，不是内核（§2 的 kernel 计数相同）。

### 3.2 官方 perf 网格

| run | 二进制 | 结果 |
| --- | --- | --- |
| `perf_data/MetaX_C600-U/20261023_172210` | 修复前 wheel，`d24396d18f44`（完整 `run_bench.sh`，含正确性/用例/两张官方网格） | `maca_c/pass 147`、`torch/pass 147`、`deep_gemm` 25 pass / 122 unsupported；correctness 0 wrong / 0 crash / 13 OOM skip；official 147/147、official_fp32 30/30 |
| `perf_data/MetaX_C600-U/20261023_181216` | 安装 wheel `14801f45326e`（完整 `run_bench.sh`，`BENCH_RC=0`） | `maca_c/pass 147`、`torch/pass 147`、`deep_gemm` 25 pass / 122 unsupported；correctness 536 pass / 0 wrong / 0 crash / 13 OOM skip；official 147/147、official_fp32 30/30；8 个阶段全过 |

每个记录都满足"每个 perf case 先检查、再计时"，所以 147 是正确性通过数，不是
"跑完了 147 个"。两份记录的内核源码是同一份（`9ecdeb0` + 只有派发表搬家），
所以 cell 级时间可以直接对读；`maca_c` 与 `torch` 的逐 cell 比值在 CSV 的
`relative_pct_vs_maca_c` 列。

**两份记录对读，就是"搬家只改体积、不改内核"的时间证据**
（`tools/compare_snapshots.py --base …_172210 …_181216`，134 个可比 cell，
±3% 容差）：

| 后端 | 结论 | 总时间 |
| --- | --- | --- |
| `maca_c` | 0 regressed / 10 improved / 124 noise | 111,035.4 → 111,275.1 us（**+0.22%**）|
| `torch`（对照，改动碰不到） | 0 regressed / 0 improved | 1,569,736.3 → 1,569,882.0 us（**+0.01%**）|
| `deep_gemm`（对照） | 0 regressed / 1 improved | 12,276.9 → 12,336.4 us（+0.48%）|

10 个 "improved" 全是 2.4–12 µs 的小 cell（最大 −8.9%，绝对差 ≤ 0.5 µs），
对照后端平在 +0.01%，所以这是几百字节的噪声而非内核变化；verdict 行是
`OK (maca_c within ±3.0% and no status change)`。

两份记录的 `status_counts` 逐项相同（147 / 147 / 25+122），correctness 的
13 个 skip 都是同一批 `batch 4096 × vocab ~1e6 × topk 2333-4096` 的 OOM
（`--max-elems` 之上的形状，不是内核拒绝）。

---

## 4. 逐族状态

| family | 构建 | 运行验证 | 说明 |
| --- | --- | --- | --- |
| xcore1600 | ✅ | ✅ 全套（§3） | C600U，28 AP，MACA 20261017.1436 |
| xcore1500 | ✅ | ❌ | 本会话无 C600 设备；产物由共同源码 `common/` + 该族自己的 89 个实例化 TU 组成，实例化表三族今天相同 |
| xcore1000 | ✅ | ❌ | 本会话无 C500 设备；同上 |

**这不是"三族已验证"。** 三族共享的那一半（`csrc/maca_kernels/common/`）在
xcore1600 上被完整验证；各族自己的实例化表今天逐字节相同（生成器的
`family_instantiations()` 只有一个分支），但"表相同"不等于"该族的编译产物在
该族设备上跑过"。发布记录必须这样写，CI 上补一次每族官方网格是消掉这一行的
唯一办法。

---

## 5. 已知问题与风险

### 5.1 产物体积：527 MB → 63 MB（已修复）

**现象**（2026-10-23 实测于 pod）：P6 之后每个 family 是 **90 个 TU**
（`common/maca_topk.cu` + 89 个 `xcore<N>/instantiations/*.cu`），每个 TU 独立
编译；wheel 是 **526,830,133 B**，xcore1500/xcore1600 的 artifact 各
**590,263,768 B**。

**根因**：`contract.cuh` 里带着派发表 `kRadixTable`，每个表项取
`&launch_radix_config<Cfg>`。一个 TU 只要**看见**这张表，**device pass** 就会把
它的每一行都实例化 —— `extern template` 只对 host pass 生效，device pass 忽略
它。所以 89 个实例化 TU 虽然 host 目标文件里只有一个 `__device_stub__`，设备
侧 bitcode 里却有全部 **180** 个 `topk_kernel_radix`：单 TU 目标文件
**7,626,616 B**（`-bc` 2,396,272 B + device ELF 5,213,480 B）。

**修复**：把 `kRadixTable`、`RadixEntry`、`radix_launcher_for` 的定义和 X-macro
列表搬进 `maca_topk.cu`（唯一做派发的 TU），`contract.cuh` 只留
`radix_launcher_fn` 与 `radix_launcher_for` 的声明；`launch_typed_radix` 不动。
同一份 TU 重编：**680,720 B**，bitcode 里 `topk_kernel_radix` **180 → 1**。

| | 修复前 | 修复后 |
| --- | --- | --- |
| 单实例化 TU 目标文件 | 7,626,616 B | 680,720 B |
| 该 TU 设备 bitcode / ELF | 2,396,272 / 5,213,480 B | 198,608 / 467,136 B |
| 该 TU 的 `topk_kernel_radix` | 180 | 1 |
| xcore1000 artifact | 458,450,392 B | 67,524,144 B |
| xcore1500 / xcore1600 | 590,263,768 B | 71,542,320 B |
| wheel | 526,830,133 B | 62,998,174 B |
| 三族构建时间 | ~25 min | ~6 min |

**不是内核变化**：60 / 89 / 2 的符号计数与修复前相同（§2），且修复后的 wheel
通过了 §3 的全部门。**不要**把表搬回头文件；判据就是"一个实例化 TU 的 bitcode
里有几个 `topk_kernel_radix`"。

### 5.2 `csrc/maca_kernels/legacy/`（上游移植树）

仍然是**源码在、不编译、不可达**。默认路由下没有任何 family 编译它
（`setup.py` 的 `sources_for_family` 与 `include_dirs` 都排除），所以它不在
本轮交付物里；它作为已知缺口保留，见 `CLAUDE.md` 的 Known holes。

### 5.3 `deep_gemm` 后端

严格子集：float32 only、`topk <= 2048`、无序。它不能服务的一律
`UnsupportedByBackend`。官方网格上 147 个 bf16 cell 全部 `unsupported`，25 个
fp32 selector cell 通过 —— 表格里保留这些行，不让"两个后端处处一致"的假象
出现。

### 5.4 完整全表没有跑

105,138 例单进程在 C500 上投影 16–17 h，本分支没有完整跑过一次（历史记录：
2026-10-09 在 11,275/105,287 处按决定停止，0 check_fail / 0 crash / 0 skip）。
本轮用 8000 例大样本 + 73 条定向用例替代。**发布前建议跑一次**
（`tests/test.py --backend maca_c --sample 1000000 -rf`），并按当天实测重新
预算时间。

### 5.5 契约测试覆盖的边界

`tests/contract_test.py` 补上了官方表不构造的那一块：`DS_HOST_CHECK` 的全部
拒绝路径（跨步行、dtype、`topk` 越界、输出缓冲过小）、`begin`/`hint`、
调用方分配的 `output_idx`、`output_idx_offset`、两种排序、`check_nan` 的两种
处置、以及默认后端解析。它按后端分别计数（maca_c 73 / torch 70），因为两个
后端的拒绝措辞不同 —— 措辞本身也被断言。

---

## 6. 复现

```bash
# 构建（三族）
export MACA_PATH=<toolkit>; export MAX_JOBS=16
./clean.sh && ./build.sh
sha256sum dist/*.whl

# 安装并验证交付物本身（从仓库外，确保回答的是 wheel 而不是 checkout）
pip install dist/*.whl --force-reinstall --no-deps
cd /tmp && python <repo>/scripts/wheel_smoke.py

# 大样本（被测包 = 安装的 wheel；checkout 有 in-place .so 时 = checkout）
cd <repo> && CUDA_VISIBLE_DEVICES=1 python tools/sample_gate.py \
    --sample 8000 --seed 20261024 --max-elems 67108864 --backend maca_c

# 定向契约（被测包同样是"谁回答测谁"：checkout 没有 in-place .so 时测安装的 wheel）
CUDA_VISIBLE_DEVICES=1 python <repo>/tests/contract_test.py --backend maca_c

# 官方网格（安静机器；不要在争用机器上 --set-baseline）
cd <repo> && CUDA_VISIBLE_DEVICES=1 ./run_bench.sh --backends maca_c,torch,deep_gemm
```

---

## 7. 上线前的建议顺序

1. **在 C500 与 C600 各跑一次官方网格**，把 §4 的"未运行"两行消掉。
2. **跑一次完整 105,138 例全表**（16–17 h 量级），作为发布前的最后一道门。
3. 每次交付候选重跑 §3 的五道门（大样本 + 契约 + wheel 冒烟 + 网格），
   并把 `manifest.json` 的 `extension_md5` 记进发布记录。
4. 产物体积的门：一个实例化 TU 的设备 bitcode 里只应有一个
   `topk_kernel_radix`；wheel 应在 60-70 MB 量级，500 MB 量级意味着派发表又
   回到了头文件里（§5.1）。
