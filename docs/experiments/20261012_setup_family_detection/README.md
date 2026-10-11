# setup.py 的 target→family 检测：别名拼写拒绝并说明理由，`native` 按裁定解析（2026-10-12）

一句话：`CUCC_TARGETS` 的每一项都要被翻译成**一个族**（`DEEP_SELECT_ARCH ∈ {1000,1500,1600}`，
契约由 `csrc/structs.h` 的行给出），而 mxcc 的 target ID 空间比这三行宽得多。
实测 probe 回答了两件此前只有说法、没有读数的事：**`xcore1008/xcore1610/xcore1620` mxcc 直接拒绝**
（不是别名）；`xcore1502/xcore1520` mxcc 接受、`__MACA_ARCH__` 分别是 1502/1520，
但**发射码相同不构成同族证据**——本工具链上 `xcore1500` 与 `xcore1600` 编同一个核也是逐字节相同。
口径因此是：**别名拼写一律拒绝**，消息必须说清「编译器接受它 / 本构建叫不出它的族 /
该传什么或该加什么」；**`native` 是唯一被解析的一条**——按用户 2026-10-12 的裁定，
`develop.sh` / `install.sh` 默认它（两条本地路径），`build.sh` 默认全量列表（可装运的 wheel
必须一份 image 服务每个家族），所以 `setup.py` 必须把它解析为**本机 family**并落到那个族的拼写上。
不改 `DEFAULT_TARGETS`、产物命名、`(-offload-arch, -DDEEP_SELECT_ARCH)` 配对。

别名拒绝的理由在第 3 节；「拒绝而不是映射」不是省事，是**没有可用证据**。`native` 不同：
它不需要证据，因为它解析到的是加载器本来就会为本机选中的那一行（§4.5 的落地收据）。

环境：`MACA_PATH=/opt/maca -> /opt/maca-3.7.0`（`/opt/maca` 是符号链接，引用一律以解析后的根为准），
`mxcc version 1.0.0 (d9102a1572)`，本机 `-offload-arch=native` 解析为 xcore1000。

## 1. 输入矩阵（实测，不是读代码推的）

方法：每一项跑一次真的 `python setup.py`（该文件在 **import 期**就解析 `CUCC_TARGETS`
并构造 Extension 列表，所以不带命令的运行已经走完整条判定，最后停在 setuptools 的
`error: no commands supplied`）。因此**这一列 rc 只说明有没有命令**，判定结论看「结果」与消息；
真正 rc=0 的构建收据在第 5 节。原始输出：`targets_raw.txt`。

产物落点：被判定的项按解析出的族表产出 `.so`（第 5 节是真实构建的字据）；
被**拒绝**的项在 import 期就抛异常，**在任何文件被写或被删之前**结束——
所以拒绝一个 `CUCC_TARGETS` 不会动 `deep_select/` 里既有的产物，也不会留下半成品。

| # | `CUCC_TARGETS` | 结果（修前） | 结果（修后） | 消息 |
|---|---|---|---|---|
| a | 未设置 | 三族 | 三族 **+ 一行说明** | 修后多打 `deep_select: CUCC_TARGETS unset or empty; building DEFAULT_TARGETS=…` |
| b | `""`（空串） | 三族 | 三族 + 同一行说明 | 与 (a) 同一条路径：`os.environ.get(...) or DEFAULT_TARGETS` |
| c | `"   "`（纯空白） | **拒绝** | 拒绝（消息改） | `CUCC_TARGETS='   ' is set but names no target; leave it unset or empty for the default list xcore1000,xcore1500,xcore1600` |
| d | `xcore1000` | 一族 | 一族 | — |
| e | `xcore1000,xcore1600` | 两族 | 两族 | — |
| f | `native` | 拒绝 | **接受，解析为本机 family**（本机 = 1000，落到 `-offload-arch=xcore1000`） | `deep_select: native -> xcore1000 (device sm 80)`；无设备时大声失败：`CUCC_TARGETS entry 'native' needs a device to resolve, and this build host reports none (no MACA device visible to torch).  On a card-less build host pass explicit spellings, e.g. CUCC_TARGETS=xcore1000,xcore1500,xcore1600.` |
| g | `xcore1008` | 拒绝 | 拒绝（消息改写） | `…entry 'xcore1008' names architecture 1008, which is not one of the families this build has a row for (xcore1000, xcore1500, xcore1600).  …and a spelling like \`xcore1502\` or \`xcore1520\`, which mxcc does accept, is not evidence of which family it belongs to.  If the part shares one of the families above, pass that spelling; otherwise add the family's row to csrc/structs.h and to the tuple here.` |
| h | `xcore1502` | 拒绝（消息误导） | 拒绝（同上消息） | 修前那句是 `names family 1502, which has no row in csrc/structs.h`——**1502 是 mxcc 接受、且名字上属于 15 族**，所以这句话读起来像「本构建缺一行」，其实是「本构建不该替它认族」 |
| i | `xcore1610` | 拒绝 | 拒绝（同 (g)） | 同 (g)；mxcc 也拒绝 1610 |
| j | `xcore999` | 拒绝 | 拒绝（同 (g)） | 同 (g) |
| k | `xcore1000,xcore1000`（重复） | 建两次 | **折叠成一次**（§1b） | `deep_select: CUCC_TARGETS entries 'xcore1000' and 'xcore1000' both name xcore1000; building it once`。修前是建两遍（ninja 第二次 `no work to do`，多一次链接与一次拷贝，产物集合正确——见第 5 节 V4） |
| l | `1000`（裸数字） | 拒绝 | 拒绝（消息改） | `…entry '1000' does not name an xcore family (expected \`xcore<N>\`).  This build produces one artifact per family and needs the family from the target spelling.  …` |
| m | `XCORE1000`（大写） | 拒绝 | 拒绝（同 (l)） | 同 (l)。mxcc 也拒绝大写（`invalid target ID: XCORE1000`），所以这里拒绝与编译器一致 |
| n | `xcore1000,`（尾逗号） | 一族 | 一族 | 空条目被过滤，符合预期 |
| o | `' xcore1000 , xcore1600 '` | 两族 | 两族 | 每项 `strip()`，符合预期 |

**空值口径（deliverable 2 的第 2 条）**：四个入口同一规则——
`build.sh` 是 `export CUCC_TARGETS="${CUCC_TARGETS:-xcore1000,xcore1500,xcore1600}"`（默认表与
`setup.py` 的 `DEFAULT_TARGETS` 逐字相同），`develop.sh` / `install.sh` 按裁定默认 `native`（§4.5）；
`${VAR:-default}` 在**未设置和空串**两种情况下都替换，`setup.py` 的 `or DEFAULT_TARGETS` 读法相同。
实测这三档：

```
CUCC_TARGETS=     -> [DEFAULT]      # 空串：脚本替换成默认表
（未设置）        -> [DEFAULT]      # 未设置：同上
CUCC_TARGETS='   ' -> [   ]          # 纯空白：原样传给 setup.py，由它拒绝
```

所以「空 = 未设置 = 该入口的默认」是有意的一条规则，不是巧合；
而**纯空白是第三种**：它被设置了、又什么都没命名，属于「设了个噤声的值」，
一律拒绝而不是静默取默认。取默认时现在会打一行说明，让「没人要的产物」是可见的而不是静默的。
**默认值本身在两个入口是两个东西**（2026-10-12 用户裁定，§4.5）：本地两条路径
（`develop.sh:19` / `install.sh:18`）默认 `native`，可装运的一条（`build.sh:15`）与
`setup.py` 的 `DEFAULT_TARGETS` 默认全量族表。

### 1b. `native` 与同族折叠（裁定后新增，全部实测）

`native` 是唯一「含义取决于构建机」的拼写，所以它被**解析**而不是透传；解析走加载器自己的
sm→族 表，结果打印出来（产物携带的是**这台机器**的常数）：

| `CUCC_TARGETS` | 设备 | 结果 | 消息 |
|---|---|---|---|
| `native` | 有（本机 sm 80） | **接受** → xcore1000 | `deep_select: native -> xcore1000 (device sm 80)` |
| `native` | `CUDA_VISIBLE_DEVICES=""` | 拒绝 | `CUCC_TARGETS entry 'native' needs a device to resolve, and this build host reports none (no MACA device visible to torch).  On a card-less build host pass explicit spellings, e.g. CUCC_TARGETS=xcore1000,xcore1500,xcore1600.` |
| `native` | `CUDA_VISIBLE_DEVICES=7`（非法下标） | 拒绝 | 同上（`is_available()` 为 False） |
| `native,xcore1000` | 有 | 接受，**同族折叠成一次** | `deep_select: CUCC_TARGETS entries 'native' and 'xcore1000' both name xcore1000; building it once` |
| `xcore1000,xcore1000` | — | 接受，同样折叠 | `deep_select: CUCC_TARGETS entries 'xcore1000' and 'xcore1000' both name xcore1000; building it once` |
| `native,native` | 有 | 接受，折叠（**解析只一次**：设备查询与那行报告被记忆在 `_NATIVE_FAMILY`，因为列表是逐项解析的，两次 `native` 会在折叠发现它们是同一个目标之前各查一次设备） | 同上两条各一行 |
| `xcore1000,xcore1500,xcore1600` | — | 三族（路径不变） | — |

**同族为什么是「报告并折叠」而不是报错**：两个条目解析到同一族时产物集合**完全一致**，
所以那不是用户错误；必须处理它只因两个 Extension 会取同一个 `name=` 与同一个输出路径——
实测那是把产物建两遍（ninja 按命令行哈希去重编译，代价是一次重链接 + 一次多余拷贝）。
`native,xcore1000` 在本机**随手就能写出来**（脚本默认 `native`，再补一个显式族），
所以折叠比报错更贴意图。`xcore1500,xcore1502` 这一类**到不了**这里——`xcore1502` 在前面就被拒（§3）。

## 2. 拼写探针：mxcc 到底接受哪些、编出来是什么（`probe.sh arch`）

方法：同一个 `.s` 探针两个——极简核（`out[tid]=1`）与 cross-lane 核（`__shfl_down_sync` +
`__ballot_sync` + 共享原子，跨 lane primitive 的 arch 相关 lowering 在这里分叉）——
经 `$MACA_PATH/tools/cu-bridge/bin/cucc --offload-arch=<拼写>` 编译，比 md5。
原始输出：`arch_probe_raw.txt`。

| 拼写 | 编译器 | `__MACA_ARCH__` | 极简核 md5 | cross-lane md5 |
|---|---|---|---|---|
| `xcore1000` | 接受 | 1000 | `34fcb6fb1443` | `f948cda367c3` |
| `xcore1002` | 接受 | 1002 | `8abd23b83d74` | `34b442511a4f` |
| `xcore1004` | 接受 | 1004 | `8abd23b83d74` | `34b442511a4f` |
| `xcore1008` | **拒绝** | — | — | `invalid target ID: xcore1008 (A target ID is a processor name, e.g. 'xcore1000')` |
| `xcore1500` | 接受 | 1500 | `289616f08383` | `1e52e5fce1ea` |
| `xcore1501` | 接受 | 1501 | `62ea524325a6` | `68db80c0b6cb` |
| `xcore1502` | 接受 | **1502** | `289616f08383` | `1e52e5fce1ea` |
| `xcore1504` | 接受 | 1504 | `289616f08383` | `1e52e5fce1ea` |
| `xcore1520` | 接受 | 1520 | `12d14b7861f9` | `14f2a8fc327d` |
| `xcore1521` | 接受 | 1521 | `f7675d30ee9b` | `0fe397865634` |
| `xcore1600` | 接受 | 1600 | `289616f08383` | **`1e52e5fce1ea`** |
| `xcore1610` | **拒绝** | — | — | `invalid target ID: xcore1610 (…)` |
| `xcore1620` | **拒绝** | — | — | `invalid target ID: xcore1620 (…)` |
| `native` | 接受 | 1000 | `34fcb6fb1443` | `f948cda367c3` |

三条读数，前两条直接改写了 CLAUDE.md 里那两句话：

1. **`xcore1008/xcore1610/xcore1620` 不是「别名」，mxcc 根本不认**（`invalid target ID`）。
   CLAUDE.md 那句「这几个都是已在表内的族的别名」对这三个是错的；
   同段那句「mxcc rejects several of those spellings outright」对它们才是对的。
2. **`xcore1502/xcore1520` 是「编译器接受、但架构号不是行键」**：`__MACA_ARCH__` 就是 1502/1520，
   不是 1500。它们落在 15 族**只是命名上的推断**（mc major 15），本机无法证实。
3. **发射码相同不能证明同族**：`xcore1500`、`xcore1502`、`xcore1504` 与 **`xcore1600`**
   在 cross-lane 探针上逐字节相同（同一个 md5）。1500 与 1600 是 `csrc/structs.h` 里**两行**
   （族键不同），却编出同一份代码——于是「md5 相同 ⇒ 同一个族」这条判据在本工具链上是假的，
   换更强的探针也救不回来。
   附带：`native` 与本机 `xcore1000` **两个探针都逐字节相同**——这正是它可被解析的根据：
   它不携带任何本机之外的信息，解析出的就是这台机器该用的那一行。但它**只属于本地路径**
   （`develop.sh` / `install.sh`），可装运的 wheel 必须用显式拼写（`build.sh` 的全量默认）。

## 3. 结论：别名为什么是「拒绝 + 说明」，不是「按名字映射到族」

- **行携带的是常数，不是名字**：`csrc/structs.h` 的行给出 `ARCH_SM_COUNT` 与
  `ARCH_SMEM_PER_AP_BYTES`，`ARCH_FAMILY == DEEP_SELECT_ARCH` 还有 `static_assert`。
  把一个 15xx 的拼写映射到 1500 行，就是在断言那台机器的 AP 数与每 AP 共享内存与 C600 相同——
  **本机没有该硬件，无法证实**；猜错的后果是网格按错误的机器定尺寸，而且是静默的。
- **唯一的现成判据不可用**（第 2 节第 3 条）：发射码相同既不能被用来「证明同族」，
  因为 1500 与 1600 也相同。于是「按测量映射」这条路在本工具链上不成立。
- **拒绝的代价是零**：调用者把 `xcore1502` 换成 `xcore1500` 就行，消息直接告诉他；
  而若那台机器确实需要自己的一行，加行是一处整数与两个常数（`csrc/structs.h`）加一个 tuple 项。
- 因此口径写成**别名拼写一律拒绝**，消息分三类：非 `xcore<N>` 拼写、`native`（解析，不再是拒绝）、
  以及「是 `xcore<N>` 但不是行键」。第三类里明确写出「mxcc 接受这类拼写」这个事实，
  正是为了替换掉修前那句会误导的「has no row in csrc/structs.h」。
- **`native` 不属于上面任何一条**：它不猜任何别的东西，解析结果就是 `_binding.device_family_suffix()`
  加载本机产物时用的那张 sm→family 表所给出的同一行（§4.5）。

## 4. 改了什么（`setup.py`）

1. `_family_of_target`：docstring 说明「只接受行键」这条口径及其理由（含 1500==1600 的读数、
   指向本目录）；两类拒绝各给一条会说清「为什么 / 怎么办」的消息；族表在该函数内提成
   `families = (1000, 1500, 1600)`，消息从它拼出，检查与消息不会漂。
   `native` 在该函数最前面分派给 `_native_family`（§4.5），不再是拒绝。
2. `DEFAULT_TARGETS` 上方注释：改成陈述现在的事实——空与未设置在每个入口是同一件事，
   纯空白不是；取默认时会打印；并写明**为什么默认是全量族表而不是 `native`**
   （打包那条路径可能跑在无卡机器上，且 wheel 要能装到别的机器）。
3. `build_for_maca` 的取值段：`raw_targets` 命名出来；空白消息改为
   `is set but names no target`（原来的 `selects no target` 与「取默认」读起来太近）；
   未设置/空时多打一行 `CUCC_TARGETS unset or empty; building DEFAULT_TARGETS=…`。
4. **不动**：`DEFAULT_TARGETS` 的值、`deep_select_maca_xcore<N>.so` 命名、
   每 Extension 的 `(-offload-arch=<拼写>, -DDEEP_SELECT_ARCH=<族>)` 配对。

### 4.5 `native` 的解析（裁定落地，team-lead 实现）

1. `_native_family(this_dir)`：按路径加载 `deep_select/_binding.py`（**不执行 `deep_select`
   包**——省掉 `__init__` 的连锁导入），调它已有的 `device_family_suffix()`（sm→族的那张表，
   与加载器同一处），把 `_xcore<N>` 转成族号。**没有第二张表**：解析用的就是加载器选产物时
   用的同一张。（设备读数本身经 torch，而 `device_family_suffix()` 内部也要 torch，
   所以构建时它在场——`build_for_maca` 开头就要 `torch.utils.cpp_extension`。）
   拒绝分两种，因为修法不同：**无可见设备** → 给出显式拼写的替代；
   **有设备但 sm 不属任何族** → 要求显式传族。
2. `_family_of_target` 在拼写判定之前先接 `native`；`_offload_spelling(target, family)`
   把 `native` 换成 `xcore<family>` 再交给 `-offload-arch`——**cucc 拿到的是显式拼写，
   不是 `native`**（收据 V6 的 `[1/1] cucc … -offload-arch=xcore1000`）。
3. 解析结果由 `_native_family` 打印一次：`deep_select: native -> xcore<N> (device sm <M>)`
   ——构建日志因此记下**哪一个族**，而不只是记下调用者写了 `native`。每进程只解析一次：
   Extension 列表与源码集摘要各要一次族，两处的答案与那一行报告都属于**构建**而不是某一次调用。
4. `build_for_maca` 的取值段增加**同族折叠**（§1b）：解析后按族去重并打印被折叠的一对，
   `targets` 只剩每族一个——这既是 `native,xcore1000` 的修法，也是两个 Extension
   取同一个 `name=`/输出路径这件事的修法。
5. `develop.sh:19` / `install.sh:18` 默认改 `native`（两条本地路径）；`build.sh:15` 保持全量。
   `CLAUDE.md` 的构表与两段相关散文按裁定重写（`docs/experiments/20261012_family_architecture/` §1 记裁定）。

顺带的两条实测记录（本次**没有**改，因为代价与收益都不成立）：

- **重复目标本身无害**（矩阵 k / V4）：同一 `(源, 命令行)` 在 ninja 里命中命令哈希，
  第二次 `no work to do`，只多一次链接与一次拷贝，产物集合正确——这正是同族折叠（§1b）
  取「报告并折叠」而不是报错的依据：产物不变，要做掉的只是那第二遍。
- **每族重编不会掉进「对象文件复用」陷阱**：三个 Extension 的 `.o` 路径相同
  （`build/temp…/xcore1000/maca_topk.o`），但三次都真的重编了——ninja 按**命令行哈希**
  判定，而 `-offload-arch`/`-DDEEP_SELECT_ARCH` 在命令行里。收据见 `builds_raw.txt` 的三条 `[1/1] cucc …`。

## 5. 验证（真实构建，`builds_raw.txt`）

`MACA_PATH=/opt/maca`，本机（`native` → xcore1000）。

| # | 命令 | rc | 产物 | 耗时 |
|---|---|---|---|---|
| V1 | `rm -rf build && ./develop.sh`（默认三族） | 0 | `_xcore1000.so` `_xcore1500.so` `_xcore1600.so` | real 2m41.9s |
| V2 | `rm -rf build && CUCC_TARGETS=xcore1000 ./develop.sh` | 0 | `_xcore1000.so` | — |
| V3 | `rm -rf build && CUCC_TARGETS=xcore1000,xcore1600 python setup.py build_ext --inplace` | 0 | 本次写 `_xcore1000.so` `_xcore1600.so` | — |
| V4 | 同上，`CUCC_TARGETS=xcore1000,xcore1000` | 0 | 本次写 `_xcore1000.so` **两遍**（第二遍 `ninja: no work to do`，再链接再拷贝一次）；`_xcore1600.so` 是 V3 留下的——`build_ext --inplace` 不清目录，这不是 V4 的产物 | — |
| V1b | 与 V1 相同（收尾恢复默认态） | 0 | 三个 `.so` | real 2m40.7s |
| V5 | `rm -rf build && ./develop.sh`（脚本默认已是 `native`） | 0 | **一个** `_xcore1000.so`（本机 sm 80） | real 1m6.0s |
| V6 | `rm -rf build && CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh`（可装运那条） | 0 | 三个 `.so` | — |

V5 的日志头两行（`native` 的落地收据——解析出的族、以及 cucc 拿到的是显式拼写）：

```
deep_select: native -> xcore1000 (device sm 80)
deep_select: device compiler is /opt/maca/tools/cu-bridge/bin/cucc, host compiler is /opt/maca/tools/cu-bridge/bin/gnu
```

V5 的编译命令行里是 `-offload-arch=xcore1000 -DDEEP_SELECT_ARCH=1000`——
**透传给 cucc 的不是 `native`**，所以产物的 `-offload-arch`、`-DDEEP_SELECT_ARCH` 与文件名是同一个数。
V5/V6 都在无并发构建的情况下跑（每个 `build_ext` 独占 `build/`）。

V1 打印的编译器对（build-change 纪律要求读这两行，不许假设）：

```
deep_select: device compiler is /opt/maca/tools/cu-bridge/bin/cucc, host compiler is /opt/maca/tools/cu-bridge/bin/gnu
```

产物收据（`deep_select/deep_select_maca_xcore1000.so`，V6 的产物，6464392 B，md5 `fa0c7e01c432…`；
`native` 落地前那一版是 6464392 B / md5 `95d8738b32e5…`，大小相同、符号计数相同——
本轮改动只碰 `setup.py` 的取值层，不碰 `csrc/`）：

| 检查 | 结果 |
|---|---|
| `readelf -d … \| grep -iE 'libtorch\|libc10'` | 空 |
| `nm -D … \| awk 'NF>=3{print $3}' \| grep -icE 'c10\|torch\|at::'` | 0 |
| `nm -D … \| grep -c __tvm_ffi_` | 2 |
| `nm -S --defined-only … \| grep -c topk_kernel_radix` | 60（实例化收据；改动只碰 setup.py 的取值层，不碰 `csrc/`，所以它**必须**不动） |
| `readelf -d … \| grep NEEDED` | 与文档一致：`libtvm_ffi` + MACA 六件（`libmcruntime/libToolsExt_cu/libruntime_cu/libmcToolsExt/libmccompiler/libmaca_mathlib_host`）+ libstdc++/libm/libgcc_s/libc |
| `readelf -d … \| grep RUNPATH` | `DT_RUNPATH`（`0x1d`，非 `DT_RPATH`）= `/opt/maca/lib:/opt/maca/mxgpu_llvm/lib` |

语义验证按纪律不在本 lane 做（另一 lane 拥有 `run_test.sh`）；本次改动不触碰
`csrc/`、内核体、路由或产物命名，唯一的行为面是「被拒的输入被拒得更清楚」与多一行打印。

### 5.1 落地收据（`landing_receipts.txt`，提交前最后一次采集）

V1–V6 是逐条排查时采的；提交前在**冻结的 `setup.py`**（md5 `0c8bd16be52c…`，全程未变，
脚本在收尾断言过）上重采了一组完整的三次构建，`landing_receipts.txt` 是原始输出：

| # | 命令 | Extension 数 | 产物 | 关键行 |
|---|---|---|---|---|
| B | `CUCC_TARGETS=native ./develop.sh`（脚本默认） | **1** | 一个 `_xcore1000.so` | `native -> xcore1000 (device sm 80)`；日志里的拼写是 `-offload-arch=xcore1000`（不是 `native`）；`topk_kernel_radix` 60 |
| C | `CUCC_TARGETS=native,xcore1000 ./develop.sh`（同族折叠） | **1** | 一个 `_xcore1000.so` | 折叠行 + `for native`——**折叠修好后 Extension 只建一个**（修前打「building it once」却建两个：折叠落在 `targets` 上，而 Extension 列表读的是 `resolved`） |
| A | `CUCC_TARGETS=xcore1000,xcore1500,xcore1600 ./develop.sh` | 3 | 三个 `.so` | 60 / 60 / 60；`readelf` 无 torch/c10；`__tvm_ffi_` = 2 |

跨构建符号检查：B 的 `xcore1000` 与 A 的 `xcore1000` 的 `nm -S --defined-only` **逐行相同**
（各 345 行）——`native` 解析出的产物与显式拼写的产物是同一份代码。

## 6. 未决 / 不在本 lane

- **每族选择源码**（`SHARED_SOURCES` / `SOURCES_BY_FAMILY`）是另一个 lane 的改动，
  同一份 `setup.py` 里，见 `docs/experiments/20261012_family_source_selection/`。
  本 lane 只碰 `_family_of_target` 与两处取值注释，未碰那张表。
- **要不要支持 `xcore1502/xcore1520`**：需要先有证据说它属于哪一行（真机上的 AP 数／
  每 AP 共享内存，或工具链给出的族映射表）。本机两者都没有，所以本轮拒绝。
  真机可得时，落点是一行 `csrc/structs.h` 行 + `families` tuple 一项，
  **不是**在 `_family_of_target` 里加一张别名表。
- `native` 已按裁定解析（§4.5）。它之后的那道闸（`csrc/structs.h` 的 `#error`）查的是
  `DEEP_SELECT_ARCH` 有没有行——解析把拼写落成行键，所以那道闸照旧只对「族号不是行键」开火。

## 7. 复现

```bash
cd docs/experiments/20261012_setup_family_detection
./probe.sh arch      # 拼写探针，秒级
./probe.sh targets   # 输入矩阵，每项约 20 s（先 import torch）
```

构建类收据（V1–V4）是手工跑的，命令写在第 5 节的表里；`builds_raw.txt` 是它们的输出摘录。
