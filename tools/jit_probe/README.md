# DeepSelect + DeepJIT 接入说明(2026-10-20)

## 这是什么

`third-party/DeepJIT`(CompilerFans,`712f2a3`,`main` 分支;原 `main` 已备份为
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
- `tools/jit_probe/flatten.py` — 把仓内"引号式 #include"头文件树摊平成单文件。
  DeepJIT 的 include parser 只接受 `<...>` 角括号带前缀 include;且它要求**每个
  编译单元恰好一个 device 内核**,所以摊平时会剥掉 `extern template` 声明
  (多行,剥到 `;` 为止)。
- `tools/jit_probe/stage1_probe.cu` — 自包含单内核的写法范式:**不拉实例化 TU**
  (它的头会带进全部 chunk/f32 launcher,实例化出几十个内核),改为直接
  `#include "radix_row_bf16.cuh"`(纯 device-inline 模板,无 launcher zoo)
  并复写 stage1 的 40 行几何。任何想 JIT 的内核都按这个范式写一个 feed。
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
`tools/jit_probe/{build.sh,flatten.py,stage1_probe.cu,jit_probe.cc}`。

## fmt 改为完整源码树(2026-10-24)

`third-party/fmt` 不再是 submodule,而是与 mcDeepGEMM 同形的完整源码树(128 个
文件,`FMT_VERSION 120100`,`diff -r` 与 mcDeepGEMM 逐字节相同)。checkout 后
无需 `git submodule update --init third-party/fmt`;
`tools/jit_probe/build.sh` 的 `-Ithird-party/fmt/include` 不变。

格式化**只用 fmt,不用 `std::format`**:DeepJIT 的 `utils/format.hpp` 定义
`FMT_HEADER_ONLY` 并 `#include <fmt/format.h>`,因为 mxcc 驱动的是 GCC 11 的
libstdc++,没有 `<format>`。DeepJIT 头文件里 `std::format` 出现 0 次;全树唯一
使用它的是 `tests/test_maca_proj/main.cpp`,那套 harness 需要 clang++-22 +
GCC-15 sysroot,本机不构建也不使用。DeepSelect 自己的源码同样 0 处使用
(`legacy/api.cu` 只有两行注释解释为何用 `snprintf` 而不是 `std::format`)。

vendored fmt 的重跑收据(两机各自 `build.sh` 重编 + 全新 JIT 缓存,shape
b6 v1048576 k512 chunks18):

| 机器 | 512 线程 | 1024 线程 |
|---|---:|---:|
| MetaX C600-A(32 AP) | 88.8 µs / 141.6 GB/s | 101.4 µs / 124.1 GB/s |
| MetaX C600-U(28 AP) | 74.5 µs / 168.9 GB/s | 82.4 µs / 152.6 GB/s |

两台均 `contract: PASS`、RC=0。`jit_probe.cc` 固定 `-DDEEP_SELECT_ARCH=1600`
(探针用的 stage1 几何在两族都是 128 KiB/AP,该宏只影响编译期容量闸)。
