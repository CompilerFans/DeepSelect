# DeepSelect + DeepJIT 接入说明(2026-10-20)

## 这是什么

`third-party/DeepJIT`(gitlink `e97994c`,`main` 分支;`54d85dd` 是发布准备
提交——MetaX 版权头、上游同步与注释清理,`1c81a45` 更正 CLAUDE.md 里消费者与
fmt 的描述,`8ccb0a7` 给共享状态加锁并支持多入口选择,`4f9eaa4`/`e97994c`
补充 Python 测试依赖并改用标准 requirements.txt;原 `main` 已备份为
`main-pre-maca-tvm-ffi`)—— 官方
DeepGEMM 的 header-only JIT 层(MACA 后端的下游)。自 `712f2a3` 起它只保留
MACA 后端,并把 pybind11 模块换成 TVM FFI 的 `get_jit()` 句柄;本 harness 走
的是纯 C++ 头(`runtime/` + `backend/maca/`),不受那次绑定层改动影响。`third-party/fmt`(12.1.0,**完整源码树**,与 mcDeepGEMM 同形:
普通文件而非 submodule,checkout 后无需 `git submodule update`)是它的 header-only 依赖。`tools/jit_probe/` 是首个消费者:一个独立 harness,
把仓内内核变体**在运行时编译**(~10-30 秒)并启动,替代"改一个编译期常量
就要全量 AOT 重编一轮(6-8 分钟)"的实验循环。

AOT 产物与其收据纪律(md5、nm 符号闸、实例化计数)完全不动;这是实验环的层。

## 组成

- `tools/jit_probe/build.sh` — harness 构建(cucc,宿主编译;设备侧由 DeepJIT
  自己编)。依赖:elfutils(`libdw-dev`)、torch 头(`TORCH_API_INCLUDE_EXTENSION_H`
  与 `USE_MACA` 两个宏,ATen 头需要)、python 头与链接库。
- `tools/jit_probe/stage1_probe.cu` — 自包含单内核的 JIT feed:直接
  `#include <csrc/maca_kernels/common/radix_row_bf16.cuh>`(纯 device-inline
  模板)并复写 stage1 的 40 行几何,排序走 `radix_topk_row_bf16_b`。它不直接吃
  实例化 TU 的理由是**变量,不是内核数**:探针要改 `KCHUNK_BLOCK_SIZE`,一个
  AOT 表里没有的模板参数;实测 2026-10-25,实例化 TU 的器件镜像里恰好一个内核
  (见末节)。任何想 JIT 的内核都按这个范式写一个 feed,`jit_probe` 直接读它
  (`argv[1]`,缺省 `tools/jit_probe/stage1_probe.cu`)。
- `tools/jit_probe/jit_probe.cc` — DeepJIT 用法样板:`create_lazy_jit<MACA>(
  Config(repo, "DSPROBE", include_dirs))` → `compile(tag, source, opts{extra_mxcc_flags})`
  → `launch(kernel, {.num_smem_bytes, .grid_dim, .block_dim}, args...)`。

## 首个实测:kChunkBlockSize 512 vs 1024(bf16 chunk stage1,合同全过)

| 形状(b, v) | 512 线程 | 1024 线程(现默认) |
|---|---:|---:|
| 1, 262144 | 67.3 µs | **25.9 µs** |
| 1, 1048576 | 124.9 | **37.9** |
| 6, 262144 | 56.8 | **44.3** |
| 6, 1048576 | **75.5** | 80.9 |

结论:**现默认 1024 在 4 格里赢 3 格**(小批/小 chunk 时单 CTA 并行度主导),
只在小批大 chunk(b6-1M,108 CTAs 撞波次边界)输 ~7%。**不翻全局默认。**
动态选块宽(按 B×chunks 与 chunk_size 择宽)是选项,代价是 chunk 内核
实例化翻倍,边际收益不足,不做。数据与复现:
`tools/jit_probe/{build.sh,stage1_probe.cu,jit_probe.cc}`(当时还经
`flatten.py`,2026-10-25 删除,它的输出与源文件只差一行注释,见末节)。

## fmt 改为完整源码树(2026-10-24)

`third-party/fmt` 不再是 submodule,而是与 mcDeepGEMM 同形的完整源码树(128 个
文件,`FMT_VERSION 120100`,`diff -r` 与 mcDeepGEMM 逐字节相同)。checkout 后
无需 `git submodule update --init third-party/fmt`;
`tools/jit_probe/build.sh` 的 `-Ithird-party/fmt/include` 不变。

格式化**只用 fmt,不用 `std::format`**:DeepJIT 的 `utils/format.hpp` 定义
`FMT_HEADER_ONLY` 并 `#include <fmt/format.h>`,因为 mxcc 驱动的是 GCC 11 的
libstdc++,没有 `<format>`。DeepJIT 头文件里 `std::format` 出现 0 次;全树最后
一个使用者是 `tests/test_maca_proj/main.cpp` 那套 harness,2026-10-24 也已改走
同一个 `utils/format.hpp`。两者对 `{:a}` 并不可互换:fmt 打 printf 的 `0x`
前缀,`std::format` 跟 `to_chars`、不带前缀 —— harness 生成十六进制浮点字面量
的模板原先自己写了 `0x`,换成 fmt 后前缀重复、mxcc 拒收,已改为由格式符给出。
DeepSelect 自己的源码同样 0 处使用。

vendored fmt 的重跑收据(两机各自 `build.sh` 重编 + 全新 JIT 缓存,shape
b6 v1048576 k512 chunks18):

| 机器 | 512 线程 | 1024 线程 |
|---|---:|---:|
| MetaX C600-A(32 AP) | 88.8 µs / 141.6 GB/s | 101.4 µs / 124.1 GB/s |
| MetaX C600-U(28 AP) | 74.5 µs / 168.9 GB/s | 82.4 µs / 152.6 GB/s |

两台均 `contract: PASS`、RC=0。`jit_probe.cc` 固定 `-DDEEP_SELECT_ARCH=1600`
(探针用的 stage1 几何在两族都是 128 KiB/AP,该宏只影响编译期容量闸)。

同一轮的两机还各自构建并跑通了 DeepJIT 自带的 `tests/test_maca_proj` harness:
两台均 **52 passed / 1 skipped / 0 failed**,末行 `All MACA DeepJIT tests
passed.`。skip 的是 `cooperative launch`:这一代 MACA 驱动接受 2^20 块的
cooperative 网格并分波执行而不是拒绝,harness 现在按能力跳过而不是判失败。
root 下还有一条 NOTE —— `/proc` 条目的 mtime 改得动,只读 secondary cache
那一半前提不成立,lookup 那半照常跑。

更新到 `1c81a45` 后在 MetaX C600-U(28 AP)上重跑:`tests/test_maca.py` 在
两台设备均可见时 **52 passed / 1 skipped / 0 failed**,`55 cache entries
validated`(只可见一台时是 51/2/0、54 条,差别就是 `multiple devices` 一条);
`tests/test_exception.py` RC=0。集成探针在同一棵树的 device 1 上重跑:
flatten 63 行,`build.sh` 构建通过(`ldd` 把 `libc10.so` 解到 torch 的
`lib/`),`/tmp/jit_probe` 在仓库根目录直接运行(不再需要
`DEEP_SELECT_REPO`);`KCHUNK_BLOCK_SIZE=512` 80.4 µs / 156.5 GB/s、`1024`
90.9 µs / 138.5 GB/s,两组 `contract: PASS`,RC=0。探针时间是接入验证的
收据,不是性能记录(同一 pod 上 device 0 仍有 keepalive)。

更新到 `e97994c` 后(`4f9eaa4`/`e97994c` 只补充 Python 测试依赖并改用标准
requirements.txt,测试代码与 `8ccb0a7` 相同)在 MetaX C600-U(28 AP)上重跑:`tests/test_maca.py` 全绿,`59 cache
entries validated`;`tests/test_exception.py` RC=0。新增覆盖包括 8 线程同时
命中 LazyInit/Device/Parser/MemCache、同一 key 的 4 线程并发编译只执行一次、
以及 `multiple_entry_points.cu` 按 `kernel_name` 选择
`entry_first`/`entry_second` 并分别启动出 21/32。DeepSelect 集成探针本轮
未重跑:JIT 头文件的 ABI 与探针使用的 `create_lazy_jit`/`compile`/`launch`
路径未改。

## 宿主编译器改用 SDK 自带的 mxcc(2026-10-24,DeepJIT `6763222`)

MACA SDK 的 `mxgpu_llvm/bin/` 里没有独立的 `clang`/`clang++`:**`mxcc` 就是它
自带的 clang 驱动**,编宿主机 C++ 和编设备代码一样直接。DeepJIT 的 harness
和自包含检查因此不再依赖 conda 工具链:

- `tests/test_maca_proj/build.sh` 默认 `CXX=$MACA_PATH/mxgpu_llvm/bin/mxcc`;
  `MACA_TOOLCHAIN`/clang++-22/sysroot/`--gcc-toolchain` 那一整支删除
  ——它们只是把系统 libstdc++ 找回来,mxcc 自己就会。`CXX`/`TORCH_ROOT`/
  `FMT_ROOT`/`MACA_PATH` 仍是覆盖点。不传 `-pthread`:mxcc 的选项表里没有,
  这一代 glibc 已把 pthread 并进 libc。
- `tests/test_maca.py` 的 `host_toolchain()`/`gcc_tree()`/`gcc_limits_include()`
  换成 `host_compiler()`(默认 mxcc,`MACA_HOST_CXX`/`CXX` 覆盖);header
  self-containment 用 `mxcc -c -o <obj>` 而不是 `-fsyntax-only`(后者不在
  mxcc 选项表里),`-Werror` 保持同等严格。用例计账改比多重集,skip 落在
  套件中间不再算 mismatch。
- `tests/test_exception.py` 的 libdw 头查找候选只剩 `$MACA_PATH/include`
  与 `$MACA_PATH/mxgpu_llvm/include`,不再提 `MACA_TOOLCHAIN`/hipcc。

收据(两台,无 conda 工具链参与,全新 JIT 缓存):

| 机器 | `python tests/test_maca.py` | `python tests/test_exception.py` |
|---|---|---|
| MetaX C600-A(32 AP) | 52/1/0,`55 cache entries validated` | pass |
| MetaX C600-U(28 AP) | 52/1/0,`55 cache entries validated` | pass |

## JIT feed 直接吃源文件,`flatten.py` 删除(2026-10-25)

`flatten.py` 的两个职责(把引号式 include 摊平、剥掉 `extern template`
声明)在这棵树上都已无事可做,两半都是测出来的:

- **include 形式**:`csrc/maca_kernels/` 全树已是 `<csrc/...>` 角括号带前缀
  (`tools/include_gate.py`:274 个文件 0 违规),探针自己的 feed 也是同一写法。
  `flatten.py` 在这份 feed 上的输出与输入只差它自己加的一行注释(HEAD 的 62 行
  feed 摊平成 63 行),内联器一次都没触发 —— DeepJIT 一直是通过 `include_dirs`
  解析这些 include 的。
- **单内核**:`cucc -aop -S -maca-device-only` 逐 TU 列出器件函数,C600-U
  (`xcore1600`、build 自己的编译参数)上五个实例化组各取代表,每个 TU 恰好
  一个 device 内核:

| TU | 器件内核 | `.mc_fatbin` |
|---|---|---:|
| `value_bf16_outidx_i32_block_1024_si1_rv0_sv0_pre1` | `topk_kernel_radix<...1024,true,...>` | 118,069 B |
| `bf16_stage1_k_topk512_nan0` | `topk_bf16_chunk_stage1_kernel<512u,false>` | 64,389 B |
| `fp32_stage1_nan0` | `topk_f32_chunk_stage1_kernel<false>` | 92,677 B |
| `fp32_stage2` | `topk_f32_chunk_stage2_kernel<512>` | 80,373 B |
| `nan_scan_fp32` | `nan_scan_kernel<float>` | 30,405 B |
| `chunks_coarse_hist_c3` | `topk_chunks_coarse_hist<3>` | 43,573 B |
| `coarse12_kernel` | `topk_coarse12_kernel<512>` | 88,581 B |

  头文件里的 `extern template` 声明没有让声明的整集落进器件镜像:
  `radix_chunked.cuh` 里那些是宿主函数模板,器件 pass 只编译被实例化的那一个。
  对照是行内核表还在头文件里的时代(2026-10-23):每个 TU 的器件镜像带着全部
  180 个 `topk_kernel_radix`,单 TU 7,626,616 B。

因此 `jit_probe` 直接读源文件:`argv[1]` 是 feed 路径,缺省
`tools/jit_probe/stage1_probe.cu`。

端到端收据(MetaX C600-U,device 1,MACA `20261017.1436`,harness 用 `build.sh`
现编,DeepJIT 按 gitlink `e97994c` 的本机副本;这台 pod 原本缺 `build.sh` 依赖的
elfutils 头,本轮以 `apt-get install --no-install-recommends libdw-dev` 补上):

| feed | 512 线程 | 1024 线程 | 合同 |
|---|---:|---:|---|
| 源文件(缺省路径) | 78.1 µs / 161.2 GB/s | 85.2 µs / 147.6 GB/s | PASS / PASS |
| HEAD 的 `flatten.py` 输出 | 78.6 µs / 160.1 GB/s | 83.2 µs / 151.2 GB/s | PASS / PASS |

两次 RC=0,同一台机器同一会话,差值在重复噪声内。`flatten.py` 对同一份 feed 的
输出与输入 `diff` 为空(只多第一行它自己的注释),这是它已无事可做的直接证据;
2026-10-24 那次 `e97994c` 上的记录(`KCHUNK_BLOCK_SIZE=512` 80.4 µs /
156.5 GB/s、`contract: PASS`、flatten 63 行、RC=0)是同一形状的读数。
