# NaN 检查的五条臂 —— 原始证据

README §8.3–§8.7 引用的每一条测量都在这。臂树本身（各 500 MB 的 checkout）
留在 `/tmp` 并在收尾时删除；**这里是记录**。所有臂都是「拷贝 + 指向」：
臂目录是生产树的拷贝，只改被测的那一处，生产树全程未被改动，收尾
`git status --porcelain` 干净。

臂的定义（字母与本文件、README、提交信息一致）：

| 臂 | 是什么 |
|---|---|
| **A** | 生产 HEAD：`radix_select_row` 两套 body 都实例化，运行时按 `fold_nan` 选 |
| **B** | `fold_nan = false` 字面量：编译器证明 NaN epilogue 恒假、整块删掉（−40%，**任何调用者都到不了的配置**） |
| **C / C′** | `params.topk == 0xFFFFFFFFu` / `volatile bool` —— 都**没**做成双 body（被常量折叠），记录在此是因为它们立了判据：是不是双 body 只能逐符号汇编证，不能靠意图、也不能靠 `.so` 尺寸 |
| **D** | 在生产内核里打印整份 `RowParams`：两次调用**只差 `cn` 一个字段** |
| **E** | `(…) && (length == 0xFFFFFFFFu)`：不透明取 false 的**形态混合** —— 一份循环的载入（`ldg_b128` 仍 3）但 A 的屏障/寄存器/smem，跑出 9,102 µs（= A/False 的 9,098）。**支持占用率解释，不是单变量证明**；单变量那一格是 T |
| **T** | B 的代码 + 把 A 的动态 smem 补回来（+2,588 B，合计 23,356 B）—— 否掉共享内存假说 |
| **S1/S2** | B 的代码 + 前置独立 `nan_scan_kernel`（扫描 grid = batches×1 / ×16） |
| **U1/U2** | 生产 + 出线 printf（U1）/ 加 `__launch_bounds__(BLOCK,2)`（U2，与 A 的 `.s` 逐字节相同 —— mxcc 静默忽略第二参数） |
| **W** | fold 只把 NaN 测试 SWAR 化 |
| **X1** | fold 用键域区间测试 `(uint16)(key-0x7f) > 0xff01` |
| **X2** | X1 + `#pragma unroll 4` |
| **Y1** | W 的 SWAR 测试 + SWAR 键翻转（**已落进生产树**，提交 `8e2a232`） |
| **Z1** | W 去掉 `__syncthreads_or` 投票（诊断用，语义不对） |

## 文件

| 文件 | 是什么 |
|---|---|
| `ab_final.log` | A / B / T / S1 / S2，同一 session 交替、3 轮，`measure_sum.py` 逐内核（含"Σ算子核"与官方口径两列） |
| `ab_fold.log` | A / W / Y1，同上 |
| `ab_family.log` + `ab_family_paired.json` | 家族级 A/B（`tools/ab_snapshot.py`，121 格配对、2 臂 × 3 轮 × 10 iters、device 2）；JSON 的每格是 `{a: [...], b: [...], route_a, route_b}` |
| `resource_usage/<臂>.ru2` | `mxcc --resource-usage` 全量输出；关键行见下 |
| `measure_sum.py` | 逐内核计时 + "Σ算子核"（`bench_topk` 的名字过滤会漏掉 `nan_scan_kernel`，所以必须自己加这一列） |
| `measure_nan.py` | 单格 A/B 计时 |
| `occ_probe.py` | 向驱动问真占用率（`mcOccupancyMaxActiveBlocksPerMultiprocessor`）；符号必须用 `__device_stub__` 那一个 mangled 名 |
| `dump_outputs.py` | NaN 契约的逐行输出（NaN 行 → slot 0 得 `0x3F3F3F3F`；±inf → 不得） |
| `loops.py` / `census_one.py` | 循环体指令计数 / 单核指令普查（读 `-aop -S` 的 `.s`） |
| `attr_probe.py` | 记录下来的原因：`mcFuncGetAttribute` 拿错 id **会 core dump**（不是返回错误） |
| `run_ab.sh` / `run_ab3.sh` | 五条臂与三条臂的驱动脚本（含"等对手进程退出"的前置） |
| `dump_asm.sh` | `mxcc -aop -S -maca-device-only` 的封装 |

## 关键数字（都能在上面文件里核对）

生产实例化 `...I15__maca_bfloat16iLi1024ELb0ELb0ELb0ELb0...` 的寄存器：

| 臂 | MT / ST | smem(静态) | 驱动占用率 |
|---|---|---:|---|
| A | 43 / 70 | 4,924 | 1 CTA/SM |
| B | 27 / 40 | 2,336 | 2 CTA/SM |
| T | 27 / 40 | 2,336（动态补齐到 A 的合计） | 2 CTA/SM |
| S1 | 42 / 50 | 2,336 | 1 CTA/SM |
| W / X1 / X2 / Y1 | **37 / 52** | 4,924 | **1 CTA/SM** |
| Z1 | 37 / 52 | 4,668 | 1 CTA/SM |

`regs_per_multiprocessor = 131072`、`max_threads_per_multi_processor = 2048`
⇒ 1024 线程、2 个 CTA 时每线程 32 个寄存器：27 在预算内，37–43 不在。
⇒ **相对 B 的 +10 个寄存器是"这个 fold 存在"本身的账**，四个改写形态
（W/X1/X2/Y1）一个都没跨过这道档。占用率数字由 `occ_probe.py` 直接问驱动
（`A=1, B=2, T=2, S1=1`），不是手算的。
