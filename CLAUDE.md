# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repo identity — read this first

This directory is **its own git repository** (`origin` = `git@github.com:CompilerFans/DeepSelect.git`, `upstream` = deepseek-ai), vendored inside the outer `mcDeepGEMM` checkout as `third-party/DeepSelect`. The outer `mcDeepGEMM` root **is also a git worktree**, usually carrying unrelated uncommitted changes.

- A bare `git commit -a` run from the wrong cwd sweeps the outer repo's changes into a DeepSelect commit. This has actually happened here; it was recovered with `git reset --mixed HEAD~1`.
- Always address this repo explicitly: `git -C /home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect <cmd>`. `cd` does not persist across Bash calls (cwd resets to the primary working directory), so absolute paths everywhere.
- This repo is designed to be standalone: it must import and build with `deep_gemm` absent (see `topk_deep_gemm`'s lazy import in `deep_select/interface.py`). The outer package is a soft dependency, reached only when a caller asks for `backend="deep_gemm"`.

`docs/C500-radix-handover.zh.md` is the current handover manual for the C500 row kernel; `docs/C500-radix-perf-ledger.zh.md` is its measured before→after ledger. Both are authoritative and current as of HEAD — read them before touching `csrc/xcore1000/`.

## What this is

`deep_select.topk` — row-wise top-K, the DSA (DeepSeek Sparse Attention) indexer selector and the sampler. Two supported scenarios: bf16 "Lightning Indexer" (any batch, any vocab, `topk <= 4096`) and fp32 "Sampling" (`vocab_size ~128K`). No floating-point math: TopK's currency is **effective memory bandwidth**, not FLOPs.

Public entry point is `deep_select.topk` (`deep_select/interface.py`). Three backends, which name **implementations, not architectures** (the same vocabulary as the host repo's `backend=`):

- `"torch"` (**the default**) — reference implementation built from torch ops; runs on any device/dtype, including where no kernel is built. Also the differential-check arm.
- `"maca_c"` — the MACA kernel this device has: the **production path** and the fast one. Which kernel that is is a property of the device, not a call-site choice.
- `"deep_gemm"` — the host repo's `deep_gemm.fp32_indexer_topk_selector`, called through its Python API, imported lazily. Implements a **strict subset** of the contract (float32 only, `topk <= 2048`, unordered); what it cannot serve raises `UnsupportedByBackend`, never a narrower answer.

**The default is correctness-first, and that is a deliberate reversal of an
earlier one.** `backend` used to default to `"maca_c"`; it defaults to
`"torch"` now (2026-09-15), because the reference is the arm that does not
depend on a kernel being correct on the device in front of you — so making it
the default costs no coverage (the kernel is validated *against* it) and
removes the last route by which a caller who asked for nothing in particular
could reach a kernel defect.  The cost is speed, and it is real: the `torch`
arm is the slow one.

The default is **not** a string default in the signature — `backend=None`
means "take the process default", resolved per call by `_default_backend()`
from `DS_TOPK_BACKEND` (unrecognized values ignored, so a typo in the
environment cannot break every call).  Read the resolution there rather than
in the signature: `inspect.signature(deep_select.topk)` reports `None`, and a
caller who wants the kernel asks for it by name or exports
`DS_TOPK_BACKEND=maca_c`.

Two arms therefore say different things and both are needed when the default
moves: `scripts/official_slice.py --backend maca_c` **pins** the call (it
rebinds `deep_select.topk`), so it tests the kernel and says *nothing* about
the default; `--default-arm maca_c` leaves the official call site unmodified
and sets `DS_TOPK_BACKEND`, so it tests the path a bare call actually takes.
Passing both is refused rather than silently reported as one.

## Build

```bash
./build.sh                                    # CUCC_TARGETS=xcore1000,xcore1500,xcore1600 (default)
CUCC_TARGETS=native          ./build.sh       # just this device
CUCC_TARGETS=xcore1600       ./build.sh       # one architecture
```

`build.sh` is the entry point; it wraps the same `setup.py` calls shown below.
It mirrors the host `build.sh` in shape but deliberately does **not** run
`bdist_wheel` — see the `pip install .` note below for why `--inplace` is the
working path — and it removes the stale in-place `.so` before building rather
than trusting `build_ext`'s timestamp comparison. It resolves targets through
`deep_select/_arch.py` so the script and `setup.py` cannot disagree, and an
unrecognized target fails the build rather than being skipped. `MACA_PATH`
(default `/opt/maca`) and `MAX_JOBS` are honored.

**There is no argument parsing — the target list is `CUCC_TARGETS` and nothing
else**, which is the host repository's shape. Two things `build.sh` does that
the host one does not, both recorded in its header: it removes the stale
in-place `.so` first, and its default is one target per family this tree names
(`xcore1000,xcore1500,xcore1600`) rather than the host default verbatim — the
host list's `xcore1008`/`xcore1502`/`xcore1520`/`xcore1610`/`xcore1620` are
**aliases of a family already in the list**, and this `setup.py` names the
extension after the *family*, so two targets in one family build the same
extension twice while `mxcc` rejects several of those spellings outright.

```bash
./install.sh                                  # build a wheel and install it
CUCC_TARGETS=xcore1600 ./install.sh           # for another architecture
```

`install.sh` additionally installs into the active environment
(`pip install <wheel> --force-reinstall --no-deps`, matching the host repo's
`install.sh`). One thing it has to do that the host one does not:

- **Build the wheel by a single `bdist_wheel` run, then hand pip the file.**
  `pip install .` cannot work here (see below); building first keeps `setup.py`
  to one invocation. One run cannot straddle the second boundary; the split is
  what `pip install .` creates.

The underlying call, if you need it directly (this bypasses the scripts' env
derivation, so export `MACA_PATH`/`CUDA_PATH`/`CUDA_HOME`/`CUCC_PATH`
yourself — see the build-change discipline section for why):

```bash
CUCC_TARGETS=xcore1000 python setup.py build_ext --inplace             # C500
CUCC_TARGETS=xcore1600 python setup.py build_ext --inplace             # C600, C600U
CUCC_TARGETS=xcore1000,xcore1600 python setup.py build_ext --inplace   # both
```

```bash
./clean.sh                             # remove build artifacts (host clean.sh's default)
./clean.sh --dry-run                   # list what would go, remove nothing
```

Removes `build/ dist/ *.egg-info/ __pycache__/` and friends plus loose `*.pyc`
and every `*.so` in the tree (all build products — there is no vendored binary
here). It deliberately does **not** touch `$HOME/.deep_gemm`, `~/.triton`,
`~/.tilelang` or `~/.metax`, unlike the host script: nothing here writes them, so
that would be deleting another project's cache. `--yes` is accepted and ignored
(it was this script's first spelling, when the default was to remove nothing).
`./clean.sh && ./build.sh` is the supported full-rebuild chain.

`CUCC_TARGETS` defaults to `native` (the device the build runs on); same variable and meaning as the host repo's `build.sh`. One extension **per architecture** — `deep_select/deep_select_xcore<N>.cpython-310-x86_64-linux-gnu.so` — because a config's shared-memory footprint is only valid against the architecture it was sized for.

**The stale `.so` must be deleted before rebuilding** — `build.sh` does this for
you, but a bare `setup.py build_ext --inplace` does not (see the handover §4):

```bash
rm -f deep_select/deep_select_xcore1000*.so \
      build/lib.linux-x86_64-cpython-310/deep_select/deep_select_xcore1000*.so
```

`build_ext --inplace` compares timestamps and **silently skips** the copy when the target is newer than `build/lib`. The `.so` lives under `deep_select/` and is gitignored, so it is stale by default. The `.so` md5 is sensitive to source line endings and is usable as a "which source did I actually measure" receipt — **but it is not a content-addressed hash, and a differing md5 is not by itself evidence that the source or the behavior differs.**

**Two identical-source builds do NOT produce identical md5s** (measured 2026-09-14, this host). `./build.sh` twice in a row on an untouched tree gave `0909cc25557dd7f2a12e82110f22b1f2` then `4b29550adb7137d25f0d47b5864db9f3`. `cmp -l` localizes the difference exactly: **6 bytes at `0x3a2af8..0x3a2afd`**, in the middle of a string that reads `…maca_topk-02b752.cpp\0__FRAME_END__…` — mxcc names the intermediate compilation unit with a **random suffix** (`maca_topk-<6 hex>.cpp`), so the embedded debug/line-table string differs per build while the code is byte-identical. The file size is the same. The upshot: cite the md5 as a receipt for *which artifact* you measured, never as evidence that two artifacts are the same or different code; when the difference matters, `cmp -l` the pair first, and if the only differing bytes are that filename string, the binaries are the same build.

**A comment-only source edit also changes the binary, and it is decidable whether it changed code** (measured 2026-09-14). Adding 36 lines of comment moved 32 more bytes in `.text`; `cmp -l` plus a byte-pattern read says exactly what they are: every one sits on a `be <imm32>` (x86 `mov esi, imm32`) and **every immediate shifted by exactly +36** — i.e. they are embedded source line numbers, not code. So the full 38-byte delta was 32 line numbers + 6 filename-suffix bytes, and the code was identical. When you need to make that claim, do it this way rather than by md5 or by argument:
- `cmp -l <a> <b> | wc -l` — is it a handful of bytes or a lot?
- a handful, clustered → disassemble the byte at each offset; if they are all one immediate-operand opcode and all shift by the same constant, they are symbols, not instructions;
- `nm -S --defined-only <so> | sort` on both, diffed — **if every symbol keeps the same address and size, no function moved**, which is the strongest cheap statement that the build is the same one;
- if the delta is large or the offsets straddle more than a couple of instructions, the binary genuinely differs — then the falsifying test is a fixed `arange` input run several times (below), never a random one and never one run.

Build plumbing worth knowing before editing `setup.py`:

- Device code is compiled by **mxcc, reached through cu-bridge's `cucc`** (`--offload-arch=xcore<N>`). `setup.py` does
  **not** write a shim, and does **not** reassign `cpp_extension.CUDA_HOME`. `torch.utils.cpp_extension` on MACA is
  written *against* cu-bridge: `_find_cuda_home()` lands on `${MACA_PATH}/tools/cu-bridge` (its guess #4), and
  `_join_cuda_home` drives the device compiler as `$CUDA_HOME/bin/nvcc`, falling back to `bin/cucc` when that is absent —
  which is this install's case. `cucc` is then the whole CUDA-dialect adapter: `-imacros __macro_mxcc.h` (turns mxcc's
  `__MACACC__` into `__CUDACC__`/`__NVCC__`, which torch's c10 headers and `kerutils/common/common.h` both branch on),
  `-gencode=...` → `-DNV_ARCH_A100 -Xdevice -D__CUDA_ARCH__=800`, `-lcudart` → `-lmcruntime`, plus the MACA library `-I`
  catalogue. Everything it does not recognize it forwards unchanged to mxcc (`-forward-unknown-to-compiler`), which is how
  the mxcc-dialect flags in `compile_args` reach the compiler. `TORCH_EXTENSION_ENABLE_XC1500_COMPILE` is refused outright
  — it would put a second `--offload-arch` in one extension.
- **Do not reimplement cu-bridge.** An earlier revision of `setup.py` wrote its own `bin/nvcc` wrapper over mxcc and pointed
  `CUDA_HOME` at it. It replicated the header and the `-gencode` translation, and silently dropped the rest — including
  `bin/gnu`, which torch asks *this same `CUDA_HOME`* for unconditionally (`get_wcuda_gnu_path()`, called from
  `build_extensions` at `cpp_extension.py:1225`) and which is the **only** return value of `get_cxx_compiler()` under
  `USE_MACA`. Every build died with `no cu-bridge gnu found`, and the file's `os.environ["CXX"] = "g++"` line was dead
  code: ninja's `$cxx` was cu-bridge's `gnu` either way. The lesson is the shape of the bug, not the instance — a
  synthesized `CUDA_HOME` is a contract with torch's MACA patch, and it is not written down anywhere.
  **If a CUDA_HOME must be synthesized, `bin/gnu` is not optional.** (Recovered with `git reset --mixed`-free edits;
  the fix is commit-sized and the fallback is `./build.sh` from a clean `build/`.)
- Every source in `csrc/` is a `.cu`, so the device compiler is the only compiler the build runs for sources.
  `api.cu` is host *code* (no `__global__`), spelled `.cu` so torch routes it to the device rule rather than to `$cxx`:
  mxcc defines `__MACA__` for a `.cu` and only for a `.cu`, and `kerutils/common/common.h` keys `KERUTILS_IS_BUILD_ON_CUDA`
  on it. Upstream calls the same file `api.cpp`; the extension is the whole difference. A `.cpp` here would need its own
  flag list, its own `-DKERUTILS_IS_BUILD_ON_CUDA`, and a pinned `CXX`. (`mxcc -x maca` is the identical switch and does
  define `__MACA__` for a `.cpp`, but it cannot reach the file: torch picks the rule by extension, and `-x` is not a file
  type it knows.)
- `-use-fast-math` is passed with FTZ turned back off (`-Xclang -fdenormal-fp-math-f32=ieee`) so the one float conversion (`__float2bfloat16` of `value_oob_fill_value`) stays exact for a denormal fill. The ranking path is integer-only and indifferent.
- `pip install .` does **not** work, for an inherited upstream reason: `setup.py` stamps the version with `datetime.now()` and a PEP 517 install runs it twice (metadata then wheel); straddling a second boundary yields `Wheel has unexpected file name`. Build isolation adds a second failure (torch absent from pip's isolated env). Use `build_ext --inplace`.
- `install.sh` reads the built wheel through `python -m zipfile -l`, **not `unzip`**: `unzip` is absent on this host and the
  check failed closed on a wheel that was fine.

MACA toolkit root is `$MACA_HOME` or `$MACA_PATH` or `/opt/maca` (a symlink; it pointed at `maca-3.5.3.17` while
`MACA_PATH` named `maca-sdk-3.8.1.3/opt/maca-3.8.1` — the env var is the authoritative one here, and the two are
different SDK generations on this box).

## Kernel architecture

The kernel tree is split **by per-SM shared memory**, because that is what a top-K kernel's staging buffers are sized against. Which tree a device runs is a property of the device:

| tree | parts | kernel |
| --- | --- | --- |
| `csrc/xcore1000/` | C500 (64 KiB/SM), **and C600 / C600U** | `maca_topk.cu`, hand-written for MACA — **what every family builds** |
| `csrc/xcore1600/` | C600, C600U (128 KiB/SM) | the upstream kernels, ported — **reserved, not built, not reachable** |

**Every family this tree builds compiles `csrc/xcore1000/`, and there is no
switch that says otherwise.** `setup.py`'s source selection is one
unconditional `sources = XCORE1000_SOURCES` line, and `_xcore1600_sources()`
sits beside it **uncalled** as the reserved implementation: it is complete and
compiling, and nothing reaches it. Wiring it back is that one line, deliberately
a source change rather than an environment variable — a switch that could put
the broken kernel back is a switch that can be left on.

The reason is not only that the port selects wrong on a C600U (see Known holes);
it is that the C500 kernel is the **better** artifact there: 1.5–2.9x faster,
and passing `check_result` on every cell the port fails. Nothing in that tree is
C500-specific code, and the capacity gate cannot fire in this direction — see
"Can a C600U run the C500 kernel" below.

`csrc/structs.h` is shared by both. It defines the operator's contract constants — `INPUT_STRIDE_ALIGNMENT_REQUIREMENT` (1024 B), `OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT` (32 B), `MAX_VOCAB_SIZE` (`1 << 23`, from the fp32-simulated census in the ported kernel), `TopkSelectArgs` — and the per-SM capacity constant `NATIVE_SHARED_MEMORY_PER_SM_BYTES`, selected by `-DDEEP_SELECT_NATIVE_ARCH` (set by `setup.py` from the same `CUCC_TARGETS` entry as `--offload-arch`). Both kernels reject a config that cannot fit that capacity **at compile time**, so upstream's 227 KiB-sized tuples cannot ride into a 64 KiB build and fail at launch instead.

### xcore1000 — `maca_topk.cu` + `radix_core.cuh` (the shipping C500 kernel)

The contract layer and launch layer. Every mode shares one contract half — the `length <= topk` shortcut, the NaN bit-pattern check, index offsets, out-of-band fills, the optional ordered emit — and **two** selection dataflows feed it, both radix:

- `radix_topk_row_bf16_b<BLOCK>` (16-bit rows) — the production path for bf16, one row per CTA, two passes over the row.
- `radix_topk_row_f32` (32-bit rows) — same two-pass dataflow, with its own overflow rule (`radix_topk_row_f32_rescan`).

`radix_core.cuh` is header-only and ported from a standalone C500 radix-TopK project (provenance banner at the top of the file records the source commit). Only `_b` and `_f32` are launched by `maca_topk.cu`; the `_k` / `_reg` static-k variants exist only for `rk::launch_topk_bf16_chunked` (the low-batch, long-row split/merge). **Check which one you are editing before assuming it is the production path.**

The current 16-bit dataflow is **coarse12** (see handover §3, ledger §2.3). Understand these three things before changing it:

1. **The coarse histogram and the candidate arena alias the same shared memory** (`s_wide = s_input_flat`). Stage 0 uses the arena region as a 4096-bin histogram; the histogram is dead after the halve/narrow steps, and a `__syncthreads()` separates the two uses. So the lead of `radix_smem_bytes(topk, sorted, wide=true)` is the histogram's 16 KB, not the arena's ~14 KB — and `maca_topk.cu`'s `radix_layout` must compute `selected`'s offset with the same flag, or the arena overwrites it.
2. **Counts are a property of the bin, not of what fits.** In pass 2 the fine histogram counts *every* member of the threshold bin; only the write into the arena is capacity-limited. History: clamping both together under `pos < SMEM_INPUT_SIZE` forced a full-row rescan to repair overflow — deleted in `4cd740a`.
3. **The narrow step's crossing test is `above + c > remain_topk`, i.e. the window itself**, not "how much is left after subtracting the byte above". The latter stops one sub-bin early when the byte has slack, leaving the window short; refine then finds no landing point and reads an uninitialized shared variable. This cost a memory violation and 379/512 wrong slots once. It does not raise — it silently mis-ranks.

Coarse is 12 bits (`kCoarse12Bits`) leaving 4 fine bits, which is all of a bf16 key — so refine never has to rank past the fine ties. Arena capacity is `kCoarse12ArenaEntries` (4096) on the 16-bit row, `rk::kSmemInputSize` (3514) on the fp32 / static-k rows. **Overflow** (threshold bin wider than the arena) falls back to a full-row rescan; the fp32 row adds that rescan rule itself, where upstream silently mis-answers by ranking the truncated subset.

Porting conventions: portable primitives only — `__shfl_down_sync`, `atomicAdd`, `__ldg`, `__syncthreads`, `__syncthreads_or`. No inline asm, no TMA, no mbarrier, no cluster. **The wave width is MACA's 64 lanes, not 32** — the CUDA-era code in this tree assumed 32 and does not hold here.

### The fp32 row (`radix_topk_row_f32`) — what has been measured, so it is not re-litigated

The 32-bit row shares the two-pass dataflow and adds a **full-row rescan** when its
8-bit coarse threshold bin does not fit the arena (see "Overflow" above). Three
things about it were settled by measurement on 2026-09-14 and are recorded in
`docs/C500-to-parity-plan.zh.md` §12–§15 and `docs/C500-radix-perf-ledger.zh.md`
§2.5/§4. Read those before proposing any of the following; each has already been
tried or explained, and the two retractions are as important as the wins:

- **The coarse level cannot be widened or narrowed.** fp32's high-byte coarse bin
  is what keeps refine cheap; taking the raw fp32 high byte instead is *order-
  preserving* (same bin order) but widens the window from 2^13 to 2^23, so every
  row overflows the arena and takes the 4-round rescan — **4.4× slower**, 200/200
  still correct. Narrower (12-bit) makes the coarse histogram itself 4,096 bins
  of per-element atomics. Both directions are closed.
- **`tx * 4 + q` addressing is an anti-pattern worth 1.65× in a microbench and
  ~1% in this kernel.** It gives each load instruction a 64-byte lane stride.
  Coalescing pass 1 was measured at **−0.7% on the fp32 grid** (43,456 → 43,146 µs
  over 36 cells, two alternating rounds) because the real walk carries one shared
  bucket atomic per element at **4 CTAs/SM** (2,596 B static + 14,056 B dynamic of
  64 KiB — measured with `cudaOccupancyMaxActiveBlocksPerMultiprocessor`;
  the "3" this file carried until 2026-09-15 was a hand-computation, see the
  occupancy note below), where the two patterns are 7% apart, not 65%. Pass 2's two walks and
  the rescan still carry the pattern deliberately: their coalesced form needs four
  live `float4` (4 more registers) and the register budget is the real constraint
  here.
- **The shared bucket atomic is closed as a lever, at every attainable
  occupancy.** Measured 2026-09-15 (`/tmp/dsab/ablate_cta2.cu`: the same walk,
  the same 2,600 B static + 14,056 B dynamic, the same `tx*4` pattern, one row
  per CTA, grid = 104 x driver-reported occupancy, three rounds): removing the
  atomic is worth **+0.5% at 3 CTAs/SM and +0.8% at 4** — inside the repeat
  noise — and only becomes real below that (+3.4% at 2, +5.0% at 1), where the
  kernel does not run. A pure-read arm at the same occupancy hits **1,645.5
  GB/s (99.7% of the wall)**, so the whole walk is 2.6-2.8% off the wall at both
  4 and 3 CTAs/SM. That 2.6% is the ceiling on *every* instruction-level idea
  here: 0.5% atomic + 0.4% integer key/add + ~1% everything else. It is not a
  4-5x gap. Do not spend effort on the atomic, on `ldg`->`bsm` (the pure-read
  arm walks the whole load chain and still hits the wall), or on int/float
  emission — none of the three has anywhere to go.

Two general traps this cost, both of which this repo has now paid for twice:

1. **A controlled microbench does not transfer to the kernel.** The 1.65× address
   measurement was right and its ~2× prediction was wrong; §9.2's cost model was
   back-solved from a delta and wrong. In both cases the missing factor was the
   per-element shared atomic and the occupancy it implies.
2. **A subtraction- or isolation-ablation can delete more than it isolates.** A
   variant that removed a shared atomic also removed the write semantics that
   went with it (once leaving an uninitialized shared variable and a device
   memory violation), and a "pass 1 only" variant is only meaningful if the
   removal leaves the *other* pass's cost unchanged.

### Can a C600U run the C500 kernel?  Yes — measured, and it already does

The answer to "why not just build `csrc/xcore1000/` on the 128 KiB parts" is
that the tree **already does**, and it is the better artifact, not a fallback.
Measured on one C600U (device 1, MACA 3.8.1), both artifacts built for the same
extension name, official cases / checks / timing rule:

| cell | `csrc/xcore1000/` | `csrc/xcore1600/` |
| --- | --- | --- |
| 4096 x 16384 k=512 bf16 | **1413.8 us / 94.9 GB/s** | 4146.8 us / 32.4 GB/s — FAIL |
| 4096 x 1024 k=512 bf16 | **477.4 us / 17.6 GB/s** | 997.3 us / 8.4 GB/s — FAIL |
| 512 x 262144 k=512 bf16 | **1793.1 us / 149.7 GB/s** | 2721.2 us / 98.6 GB/s — FAIL |
| 8 x 4096 k=512 fp32 | 16.7 us / 7.9 GB/s | 16.4 us / 8.0 GB/s — FAIL |

`check_result` passes all five cells on `csrc/xcore1000/` and fails all five on
the port. The two fp32 cells are latency-bound and identical; the rest is
1.5–2.9x.

**Why it is feasible, checkable rather than assumed:**

- Nothing in `csrc/xcore1000/` is C500-specific code. No `__MACA_ARCH__` branch
  anywhere in the tree; every cross-lane primitive is already 64-lane
  (`radix_core.cuh`'s `kWarpSize = 64` under `__MACACC__`); the ranking is
  integer key manipulation with no float math to differ per part.
- **Capacity cannot fire in this direction.** `structs.h` picks
  `NATIVE_SHARED_MEMORY_PER_SM_BYTES` (64 KiB for family 1000, 128 KiB for
  1500/1600) from `-DDEEP_SELECT_NATIVE_ARCH`, which `setup.py` passes from the
  target, so a C600U build gets the 128 KiB constant. The compile-time rejection
  exists to stop an *oversized* config; a 64 KiB-sized one in a 128 KiB SM
  cannot trip it.
- **The SM-count-sensitive constants are already parameterized** by
  `NATIVE_SM_COUNT` (104/28/32): `wave_filled_chunks` (the chunked split),
  `NATIVE_F32_CHUNK_WORK_TARGET`, and the grid sizing. The C500 values are the
  measured ones; the 1500/1600 values are arithmetic carried over and **marked
  unmeasured in the source**. That is a performance caveat, and the table above
  bounds it — the correct artifact still wins by the measured margin.
- What is given up is exactly the port's reason for existing: `topk` in
  `(1024, 4096]` and `vocab_size >= 2^23` are the *ported* kernel's limits, and
  `maca_topk.cu` has neither (it has no capacity gate by design — "a 128 KiB SM
  runs it with room to spare"). bf16 `sorted_value`, rejected by the port, is
  accepted here. Neither limit is reached by the official grid.

So the default is not a workaround standing in for a broken kernel: it is the
shipping kernel, on a part it is correct and faster on. There is no override to
reach the port — it is a one-line source change in `setup.py`, and that is the
point: a switch that could put the broken kernel back is a switch that can be
left on.
### xcore1600 — the ported upstream kernels

**This tree does not currently pass on a C600U and is not built at all** — see
Known holes for the measurement, the reproduction, and the 32-lane suspect list.
Everything below describes it as written; treat it as unvalidated. To work on
it, point `setup.py`'s `sources =` line at `_xcore1600_sources()` and re-run
`scripts/official_slice.py --backend maca_c` on a C600U.

**A C600U pass is necessary but not sufficient to wire it back in**: switching
the source tree also changes which kernel serves C600 and C600U production
traffic at every shape — that is a routing change, not just an audit milestone,
and it needs the perf arm too (the C600U measurements say the port also *loses*
on time today).

`api.cu` is the host dispatch + pybind11 module; `v3/` (bf16) and `v3_fp32/` (fp32) each hold `topk_select.cuh` + a generated `instantiations/` directory. `common_parts.cuh`, `bit_utils.cuh`, `utils.cuh`, `config.h`, `dispatch_utils.h` are shared.

Upstream's algorithm is kept (threshold-and-compact scan in a random block order, one global read per element); only its device-side dependencies were replaced: TMA tensor-map loads → cooperative `ldg`, mbarriers → a single buffer with `__syncthreads`, inline PTX → MACA builtins/plain C++. Config tuples were re-derived for 128 KiB (upstream's are sized for an H100's 227 KiB). `v3_cluster` was **deleted**, not ported — MACA has no cluster launch — and those shapes fall through to the general kernel with no dispatch arm.

**The config table has two halves that must be edited together**: `scripts/generate_instantiations.py` (the table, its arithmetic, and the `check_fits_maca` refusal) and the `TopkSelectConfig<...>` call sites in `csrc/xcore1600/api.cu`. A mismatch is a **link error**, not a runtime one. Note also that `deep_select/_arch.py` duplicates the host repo's `deep_gemm/utils/arch_config.py` `XcoreFamily` rows (capacity + family spelling) by hand — it cannot import that package, so a change to either belongs in the same review.

Consequences of the port, all deliberate:

- `topk` in `(1024, 4096]` is **rejected** by the ported kernel: a `max_topk = 4096` tuple is not merely tight on 128 KiB, it is impossible — `surviving_topk_pairs` alone is `2 * 4096 * 8 = 65536` B and the extra-pairs region at least `(4096 + 4096) * 8 = 65536` B. With the default routing a C600/C600U serves those shapes from `maca_topk.cu`, so the operator does not reject them.
- `vocab_size < 2^23` is enforced by the ported kernel only (the fp32-simulated census). `maca_topk.cu` has no such limit; it ranks integer keys.
- `sorted_value` for bf16 is accepted by `maca_topk.cu` (same ordering key, so order and value/index pairing both hold) and rejected by the ported kernel, as upstream, and by `backend="torch"`.

## MACA warp intrinsics — the wave is 64 lanes

**Before auditing or porting any kernel that does cross-lane work, run
`skills/maca-wave64-port/scripts/run_probe.sh`.** It compiles a probe through
the same device compiler the build uses and prints what MACA's collectives
actually do on the part in front of you — the semantics below, measured, not
read off a header. The skill at `skills/maca-wave64-port/` carries the full
audit procedure and its checklist.

Get these right before writing any cross-lane code; every one of them is a
silent-wrong-answer trap rather than a compile error. Facts below are read from
the toolchain's own headers (`$MACA_PATH/mxgpu_llvm/lib/clang/19/include/
__clang_maca_device_functions.h`) and the official builtin guide
(OG-26013-000-F5_V01, shipped with the `maca-mxcc-builtins` skill), and
**confirmed by measurement** on MACA 3.8.1.3 / C600-U (2026-09-12). Follow that
skill (and `maca-kernel-dev-and-opt` for the wider workflow) rather than guessing.

**A 32-bit mask does not mean "group the wave by 32" — it means "discard lanes
32..63".** `__ballot_sync(mask, pred)` lowers to `__builtin_mxc_sicmp(pred, 0,
ICMP_NE) & mask`: one comparison covering all 64 lanes, then a bitwise AND. So
`__ballot_sync(0xFFFFFFFF, 1)` returns `0x00000000ffffffff`, `__reduce_add_sync(
0xFFFFFFFF, 1)` returns **32** (not 64), and `__shfl_up_sync(0xFFFFFFFF, v, 1)`
gives lane 32 **its own value back**. The rule is uniform across ballot,
reduce, any/all *and* the shuffles — a lane outside the mask reads itself — and
a kernel that runs work on lanes 32..63 while masking them out is silently
wrong. Measured: the port's own 32-lane scan is wrong on **32 of 64 lanes**.

**Probe a shuffle by encoding the source lane in the value** (`v[lane] = 1000 +
lane`). A distinctive value on one lane cannot distinguish "excluded by the
mask" from "included, but that lane's own value happens to be what you would
expect" — an earlier probe here did exactly that and read the opposite rule off
the same hardware.

**Wave width is 64, so every mask is 64-bit.** The guide's comparison builtins
(`uicmp`/`sicmp`/`fcmp`) are documented as "返回 warp 内 64-bit 比较结果掩码".
A `0xFFFFFFFF` mask is not a shorthand for "all lanes" here — it names the low
half of the wave only, and the result is a silently half-counted operation, not
a compile error.

**Every "_sync" collective's masking rule is the same, and it is measured:**

| collective | with `0xFFFFFFFF` on a 64-lane wave |
| --- | --- |
| `__ballot_sync(0xFFFFFFFF, 1)` | `0x00000000ffffffff` — lanes 32..63 are not in the mask, so they are not counted |
| `__reduce_add_sync(0xFFFFFFFF, 1)` | **32** — sums the low half only; lanes outside the mask get their own value back |
| `__any_sync` / `__all_sync(0xFFFFFFFF, …)` | same rule; a predicate true only on lane 40 reads **false** everywhere |
| `__shfl_up/down_sync(0xFFFFFFFF, v, 1)` | lane 32 reads **itself**, not lane 31 |

`kerutils`'s `canonical_warp_idx_sync()` (`csrc/3rdparty/kerutils/include/
kerutils/device/cuda/common.h:89`) is `threadIdx.x / 32u` and belongs to this
family: a port that calls it for `warp_idx` gets a warp index twice as large as
the real one, which then mis-sizes every `warp_cnt[NUM_WARPS]` array.

| intrinsic | status on MACA | use |
| --- | --- | --- |
| `__ballot_sync(unsigned long long mask, int pred)` | **native — literally `__builtin_mxc_sicmp(pred,0,ICMP_NE) & mask`** (`:2202`). 64-bit overload only | the correct primitive |
| `__activemask()` | **native — one read of the wave mask** (`__builtin_mxc_read_xmsk`, `:2206`). Free | pass as the mask: correct and adaptive |
| `__any_sync` / `__all_sync` | **native** — the same `sicmp` fused with a ballot-mask compare (`:2190`, `:2195`) | cheaper than ballot + `__popcll` when you only need the predicate |
| `__match_any_sync` | **software-emulated — a 32-iteration per-bit loop of `sicmp`** (`:1648`), i.e. 32× the cost of a ballot | **do not use** |
| `__reduce_add_sync` | **software-emulated — a 6-iteration `bsm_bpermute` loop** (`:200`), and it takes a `uint64_t` mask | prefer a ballot-based reduction; with `0xFFFFFFFF` it sums only the low 32 lanes. **MACA ships two overloads, `(uint64_t mask, …)` and `(unsigned mask, …)`, so an unsuffixed `0xFFFFFFFF` literal is *ambiguous* and will not compile — spell the type (`(uint64_t)0xFFFFFFFFffffffffull`) or use a `uint64_t` variable.** The `unsigned` overload is the 32-lane one |
| `__popc` | **exists** (`__clang_macac_math.h:1155`), but it is 32-bit: `__builtin_popcount` of a 64-lane mask **truncates the wave in half** | **`__popcll`** (`:1159`) for anything derived from a mask |
| `__ffsll` / `__lanemask64_lt` | present | MACA's own 64-lane code uses this family (`maca_coalesced_scan.h:104-118`) |

**The rule that follows from the `__match_any_sync` row: do not reach for
warp-aggregated atomics via match-any.** A per-member `atomicAdd` is usually
cheaper than 32 mask ops per call — and 265 device instructions is what those
32 mask ops actually cost, measured below — and match-any buys nothing that a
`__ballot_sync` + `__popcll` + prefix does not. Measured support for this
direction in `docs/C500-radix-profile.zh.md` §5: a variant that *reduced* atomic
conflicts by 4× came out **38 µs slower**, so the atomics' cost here is
per-atomic issue, not contention.

Correct 64-lane idioms:

```cpp
const unsigned long long live = __activemask();            // 64-bit, one insn
const unsigned long long m    = __ballot_sync(live, pred); // NOT 0xFFFFFFFFu
const unsigned     cnt        = (unsigned)__popcll(m);     // NOT __popc
const int          first      = __ffsll((long long)m) - 1; // -1 when empty
```

### Reach for the MACA builtin, not the CUDA-era spelling

This is an instruction-selection rule, and the emulation is not free. Measured
from the emitted device assembly — `mxcc -aop -S -maca-device-only <file>.cu`,
with this repo's own `compile_args` from `setup.py`, on `xcore1000` — counting
real device instructions in a minimal kernel around each call:

| call | how MACA lowers it | insns |
| --- | --- | --- |
| `__ballot_sync(live, pred)` | **native**: one `cmp_gt_i32` + one `sand_b64` | **14** (whole kernel) |
| `__match_any_sync(mask, v)` | 32 iterations of `sicmp` + `sand_b64`/`sxor_b64`/`sadd_co_i32` | **265** |
| `__reduce_add_sync(mask, v)` | 6 iterations of `sm_bperm_b32` plus ~30 insns of mask/index math | **206** |
| `__shfl_down_sync(…)` ×5, a 5-step warp scan | 5× {`__lane_id()` + index math + `sm_bperm_b32`} | **71** |
| `__builtin_mxc_bsm_bpermute` ×5, the same scan | 5× `sm_bperm_b32`, only the arithmetic you wrote | **44** |

The last two rows are the ones to internalize: **each emits the same 6
`sm_bperm_b32`** — the shuffle instruction is identical, five in the source loop
plus one elsewhere in the function — so the 27-instruction difference is
*entirely* the wrapper's per-call `__lane_id()` and
`(self & (width-1)) + delta >= width` arithmetic, which the compiler does
**not** hoist out of the loop. Read a `__shfl_*_sync` as "one native shuffle
wrapped in ~5 instructions of index math", never as one instruction.

To re-measure rather than take these on faith: `mxcc -aop -S -maca-device-only
f.cu -o f.s`, with the same flags `setup.py`'s `compile_args` sets. `-aop` is
what makes the listing emit at all — the clang spelling `-S` alone is rejected —
and it prints `'-aop' is internal, only for DEBUG usage.` on stderr while still
producing the file, so filter that line or ignore a non-zero-looking rc. Count
real instructions between a function's label and its `endk`, skipping the
metadata blocks. `~/.claude/skills/maca-kernel-doctor`'s companion script,
`~/maca_kernel_doctor/maca_kernel_doctor.py`, wraps the same thing (`--asm`,
`--save-temps`) alongside the register/occupancy checks, and is the better entry
point if you want a diagnosis rather than a listing.

So, when a builtin *is* the operation, call it:

```cpp
// 64-lane step-down gather — the shfl_down_sync wrapper, minus its index math
int n = __builtin_mxc_bsm_bpermute(((lane + delta) & 63) << 2, val);
// NOTE the <<2: the hardware index is byte-addressed (dest[n] = data[index[n]/4 % 64])

__builtin_mxc_mov_shfl(val, mode, row_mask, bank_mask, bc);   // 16-lane row ops
__builtin_mxc_update_shfl(old, src, mode, row_mask, bank_mask, bc);
__builtin_mxc_readfirstlane(v);   // uniform broadcast, no shared memory
__builtin_mxc_readlane(v, lane);  __builtin_mxc_writelane(v, lane, old);
unsigned long long m = __builtin_mxc_sicmp(a, b, MACA_ICMP_SLT);
// ubfe / sbfe / alignbit / mad_wide_i32 / pk_fma_f32 / ldg_*_predicator /
// ldg_*_bsm + barrier_and_wait{1,2,4} — see the guide, §12 has worked patterns
```

`bsm_bpermute` is the general 64-lane primitive under all of it; `mov_shfl` /
`update_shfl` are the 16-lane row operations (mirror, row shift, row rotate,
row broadcast) that a full permute can express but not cheaply. There is no
reduce instruction to call: no `__builtin_mxc_*` gives a warp reduction (the
guide has no reduce entry), and the `__reduce_*_sync` family is emulated on top
of the shuffle. So a reduction is a `bsm_bpermute` butterfly you write yourself
— 42 insns for the 6-step 64-lane sum above, versus 206 for
`__reduce_add_sync`.

**CUB is ported and works here, but it is not a shortcut past this.**
`/opt/maca/include/cub/warp/warp_reduce.cuh`'s `cub::WarpReduce` inherits exactly
the emulated `__shfl_down_sync`
(`cub/warp/specializations/warp_reduce_shfl.cuh:147`), so it measured the same
71 instructions as the hand-written `__shfl_down_sync` scan —
against 42 for the same reduction with the tree's own
`__builtin_mxc_bsm_bpermute` butterfly. Use CUB for what it is good at (the
block-wide primitives — `common_parts.cuh` already builds on
`cub::BlockRadixSort`); for a warp-scope reduce in a hot loop, write the
butterfly. MACA's CUB does at least get the width right: `CUB_LOG_WARP_THREADS`
is hardcoded to **6** (`cub/util_arch.cuh:91-99`), i.e. 64, so the `0xffffffff`
masks upstream's ported kernels carry are the CUDA-era texture, not a CUB
requirement.

Budget for the mask-math too: `__popc` **does** exist (`__clang_macac_math.h:
1155`) but is `__builtin_popcount`, 32-bit — applying it to a 64-lane ballot
truncates the wave in half and under-counts silently. Nothing raises. The
coalesced-group headers under `mxgpu_llvm/lib/clang/19/include/`
(`maca_coalesced_scan.h`, `maca_partition.h`, `maca_cooperative_groups.h`) are
the model, and they use **`__popcll`** (`maca_coalesced_scan.h:104`, `:119`).

**Beware reading upstream's CUDA-era code as a model.** `csrc/xcore1600/` (the
ported upstream kernels) is full of `__ballot_sync(0xFFFFFFFF, …)`,
`__reduce_add_sync(0xFFFFFFFF, …)`, `__shfl_sync(0xFFFFFFFF, …)` and
`threadIdx.x % 32` — with `NUM_WARPS = NUM_THREADS / 32`, i.e. CUDA's 32-lane
model (e.g. `common_parts.cuh:1463`, `v3_fp32/topk_select.cuh:90`). Those sites
are the port's texture, not a pattern to copy, and whether any of them is
reachable on a 64-lane wave needs its own audit — treat them as suspect rather
than as precedent, and write new cross-lane code in the 64-lane form above.

`csrc/xcore1000/radix_core.cuh` already does this correctly and is the model to
follow: `kWarpSize = 64` under `__MACACC__` (`:187`), with `#ifdef` pairs like
`__shfl_down_sync(0xFFFFFFFFFFFFFFFFULL, …)` / `0xFFFFFFFF` for the CUDA build.
When you add a mask here, add it to the 64-lane arm.

One caveat on that file as a *primitive-selection* model: three of its
cross-lane sites are the wrapper form — `:255` in `hist_add_bf16_reg`, and
`:339`/`:358` inside `run_cumsum_warp`, which the shipping `radix_topk_row_bf16_b`
calls **twice per row** (`:1064`, `:1159`), i.e. once per pass. Those two are the
71-vs-44 case above, six shuffle steps each. They are not where the time is —
~96 shuffle ops per CTA against pass 1's one shared atomic *per element* (16,384
of them on the profiled row, the measured 102.5 µs) — so converting them is
cleanup, not a lever.

**`hist_add_bf16_reg` / `hist_reg_to_smem` (`:239`, `:269`) are dead, and they
do not work — do not wire them up.** They read as "a register histogram we could
switch on to kill pass 1's atomics", and the handover twice proposed exactly
that. They are actually a *per-element shuffle transport*, not a histogram: each
of the 8 elements does one `__shfl_sync`, and each shuffle lets only **one** lane
(the bin's owner) increment one register, so a warp records 1 element per
element. The ceiling is therefore `lane == owner`'s hit rate, 1/64 — and
`recv_bin % kBinsPerThread` is not the owner's own slot either (it only lines up
when the bin happens to be `owner*4 + recv_bin%4`; `local` is computed and never
used). Simulated at 64 lanes / 4 bins per thread with element values spread over
0..255 *and* crowded into 16, both give the same answer: **1.5% of elements
recorded**. A correct version is a thing to write, not a thing to enable — see
the handover §9 lever 2 for the shape (`r_hist[16]` per thread, since `s_wide`'s
bin is `key >> 4` and 4 bits is exactly `kCoarse12SubBins` per thread).

Also worth not "fixing" blind: the `owner` index handling at `:253-259` and the
`0xFFFFFFFFFFFFFFFFULL` / `0xFFFFFFFF` `#ifdef` pair, which is what makes the
file build for both compilers.

## Resource audit — registers, spill, shared memory

**Before any performance work on a kernel, audit its resources — and run the
audit before the experiment, not after.** Three questions, all answerable
offline on this machine, and the first two decide whether an optimization can
work at all. The cost is one device compile; the cost of skipping it is an
experiment that was never measurable.

### 1. Registers and spill — `--resource-usage`

`mxcc --resource-usage` prints one `maca info:` block per device function:

```
Function properties for  <mangled name>
  <N> bytes stack frame
Used  <N> MTregisters, <N> STregisters, <N> bytes shared mem
staticMaxWarps/PEU : <N>
```

`tests/kernelkit/build.py`'s `check_maca_stack_frame` parses this;
`DEEP_SELECT_MACA_STACK_CHECK` / `DEEP_SELECT_MACA_STACK_BASELINE` is the gate
(Build section). Two facts to read it with:

- **The stack frame is the whole spill signal.** `local > 0` in the CUDA
  original has no separate counterpart here — what would be local memory is
  counted in the stack frame.
- **48 bytes is the floor on this toolchain, not a spill.** 50 of
  `maca_topk.cu`'s 78 devices report exactly 48; the other 28 report 0. A
  `--stack-baseline` of 48 therefore reports the floor and flags a real spill.

Measured 2026-09-14 on `maca_topk.cu` (xcore1000, the build's own flags): the
launched device functions report a **48-byte stack frame, no spill**, with
**`topk_kernel_radix<…,512,…>` at 44 MT / up to 78 STregisters** and
**`topk_kernel_radix<…,1024,…>` at 46 MT / up to 78 ST**. The `topk_bf16_*_kernel_*`
family (the runtime/static-k/chunked arms) is lighter — 24 MT / 42-52 ST — and
those are only reached by the chunked path (below).

### 2. Occupancy — `~/maca_kernel_doctor`

`~/maca_kernel_doctor/maca_kernel_doctor.py` (the companion of the
`maca-kernel-doctor` skill) wraps the same compile and adds the occupancy model
in `lib/occupancy.py`, which is a pure-Python reimplementation of
`mcOccupancyMaxActiveBlocksPerMultiprocessor`. Run it as:

```bash
~/maca_kernel_doctor/maca_kernel_doctor.py \
  --compile "mxcc <the build's own compile_args> -c <src> -o /tmp/k.o" \
  --kernel "*<pattern>*" --blocksize <N> --no-color
```

**It needs `tvm_ffi`'s and Python's include dirs added**, because `maca_topk.cu`
now includes `csrc/ffi/ffi_checks.h` — append
`-I$(python -c "import tvm_ffi,os;print(os.path.dirname(tvm_ffi.__file__))")/include`
and `-I$(python -c "import sysconfig;print(sysconfig.get_paths()['include'])")`.
Without them the compile dies on `'tvm/ffi/error.h' file not found`, which reads
like a doctor bug and is not one.

**The trap that matters: `--resource-usage` reports *static* shared memory
only.** The row kernel's arena comes from `radix_smem_bytes()` and is attached
with `cudaFuncSetAttribute`, so the compiler prints **2596 bytes** where the
kernel actually holds far more. Passing 2596 to the occupancy model returns
"100%, limiter=waves" — a wrong answer that looks like a clean bill of health.
Compute the real figure and add it by hand:

```
smem = radix_smem_bytes(topk, sorted, wide) + <static 2596>
bf16 wide path, topk=512 : 16384 + 2048 + 2596 = 21028 B
bf16 wide path, topk=1024: 16384 + 4096 + 2596 = 23076 B
```

With the correct number the model answers **75% at topk=512 and 50% at
topk=1024, limiter=shared_memory** on C500's 64 KiB/AP. Registers and waves are
not the limiter — so a register-pressure optimization on this kernel buys
nothing, and that is worth knowing before writing one.

**But do not hand-compute that division — ask the driver.** The AP's 65,536 B is
handed out in fixed shares, so the usable threshold is not `65536/N`. Measured
on this part (`/tmp/dsab/occ_probe.cu`), the driver's answer is 4 CTA/SM up to
**16,388 B total**, 3 up to 21,849, 2 up to 32,600 — i.e. a granularity of
`65536/42 = 1,560.4 B` with the top share left unused (41 x 1,560.4 = 63,976).
The bf16 figures above (3 at topk=512, 2 at topk=1024) are unaffected, but the
**fp32 row's 16,652 B computes to 3.94 and is really 4** — its "3 CTAs/SM" was
wrong for as long as this file carried it, and every downstream argument that
rested on it (the shared atomic's cost, "4 vs 3 mismatch") rested on a
hand-computation over a boundary that is not where it was assumed. Use
`cudaOccupancyMaxActiveBlocksPerMultiprocessor` with the *kernel's own* static
+ dynamic total, and never label an occupancy by hand.

### 3. Wave quantization against the batch — the actual binding constraint

C500: **104 AP × 2048 threads = 212,992 thread-slots**. The row kernel is one
CTA per row, so **batch size alone decides the wave count**:

| batch | CTA-waves @512 | thread-fill @512 | CTA-waves @1024 | thread-fill @1024 |
| --- | --- | --- | --- | --- |
| 104 | 1 | 25% | 1 | 50% |
| 208 | 2 | 50% | 2 | 100% |
| 256 | 3 | 61% | 3 | 123% |
| 512 | 5 | 123% | 5 | 246% |
| 4096 | 40 | 984% | 40 | 1969% |

**At 512 threads the first CTA-wave uses only a quarter of the machine**, and
the measured dependency is the reciprocal: `b4096-v1024-k512` runs at **34.9
GB/s** while `b4096-v16384-k512` (16× the row length, identical row count) runs
at **84.5 GB/s** — same batch, same total DRAM per row, 2.4× the throughput.
The short-row cell is latency-bound at 25% thread-fill; the long-row cell has
enough independent work per thread to hide it.

The `maca-kernel-profiling` and `maca-kernel-optimization` skills are the
follow-through once the audit says which limit you are actually on.

### 5. What the reference implementations say about this kernel

Our C500 tree is a **merge** of two designs, and the reference sources are the
two halves of it (`csrc/xcore1000/radix_core.cuh`'s provenance banner records
the port):

- **Upstream DeepSelect** (`docs/DeepSelect-deep-dive.{md,zh.md}`): scan random
  blocks, filter against a running threshold, compact when the candidate buffer
  grows past `k + B2`. Its guarantee is **each element read exactly once**, in
  contiguous blocks; extra space is `O(k + B + B2)`, small enough for shared
  memory. Expected total elements processed is
  `E[W] <= (1 + k/B2) · L · H_m`, `L = k+B+B2`; with `B, B2 = Θ(k)` the extra
  compute is `O(k log(N/k))`. Realized here in `csrc/xcore1600/`.
- **`dsa_topk`** (`/home/compiler_gfx/dsa_topk`, port commit
  `61ab380c77b81669718bfb11b95a583b0e661001`): the two-pass radix row —
  histogram pass, then collect-and-stage into an arena. Realized here in
  `csrc/xcore1000/radix_core.cuh`. Note `opt/radix_topk.cuh` there is *newer*
  than the port source and **is not what we carry**; read it as a separate
  experiment, not as our ancestry.

This kernel had already converged on upstream's key structural ideas, so the
honest summary is that the references mostly **confirm** it, and the remaining
suggestions are small:

- **Both read the row twice where upstream reads once.** Our two passes are
  already at 80% and 89% of single-pass attainable rate, so the headroom in
  relentless MLP is small; a one-pass form is the structural way to the wall
  and is a much larger change than it looks (`radix_topk_row_bf16_k` is a
  different, lighter dataflow than the shipping `_b` row, and its exact
  coverage has **not** been verified here — check before calling it a drop-in).
- **`radix_topk_row_bf16_b` already has the `remain_topk == 0` exact-fit exit**
  (`csrc/xcore1000/radix_core.cuh:1096`) — it returns after the coarse
  complement, exactly as deep_gemm's `topk_coarse12_impl` does. It is the
  *overflow* path that lacks the corresponding exit, and
  **`overflow_emit_member` (`:987`) may write slot 0 twice**: its slots all use
  `output[atomicAdd(&s_counter, 1u)]`, but `output[0]` is already the coarse
  complement when `remain_topk == 0`. The condition appears reachable
  algebraically (`s_high_threshold_bin_id` is the sub-bin index while
  `threshold_bin` counts the whole bucket, so `high <= threshold_bin` can sit
  just inside the arena capacity). A seeded probe on `b4096-v1024-k512` (three
  values of 100.0 with `topk=512`) returned no `-1` sentinel and no duplicate,
  so **the defect is unconfirmed — do not "fix" it on this evidence**. Test it
  deliberately with a crafted exact-fill row before touching the code.
- **Upstream's integer-add → float-add trick** (for `0 <= x,y <= 2^22`,
  `x+y == __float_as_uint(__uint_as_float(x) + __uint_as_float(y))`, so a
  denormal float add replaces an integer add and frees the integer pipe) is
  aimed at upstream's bottleneck, mask generation. Ours is the shared-atomic
  issue rate — `P4` showed the 102.5 µs is per-atomic issue and not contention
  — so the trick is worth understanding and **not** obviously transferable.
- `dsa_topk`'s newer `opt/radix_topk.cuh` runs at `kSMEM = 16 KiB` on MACA
  against a 1949-line predecessor — the same direction as our
  `kCoarse12HistBytes`/arena aliasing, and worth reading as prior art for the
  occupancy-vs-parallelism tradeoff, not as a patch to apply.

### 4. Choosing an intrinsic — the `maca-mxcc-builtins` skill

When an audit says a kernel is bound on a specific *instruction class*, pick the
intrinsic from the `maca-mxcc-builtins` skill rather than from the CUDA-era
spelling — it carries the official guide (OG-26013-000-F5_V01) with per-builtin
signatures and arch gates, plus the measured instruction counts that say what
each wrapper actually costs. The rules already distilled here (64-lane masks,
`__popcll` not `__popc`, `bsm_bpermute` over `__shfl_*_sync`, no
`__match_any_sync`) are that skill's conclusions; go back to it for anything new
rather than extrapolating from them.

**But read the profile before reaching for it.** The instruction-selection
table above was measured on a *minimal* kernel around each call. In situ the
verdict can invert: `docs/C500-radix-profile.zh.md` records that a variant
which *cut* atomic conflicts 4× came out **38 µs slower**, because the cost was
per-atomic issue, not contention. An intrinsic that is cheaper in isolation is
not thereby cheaper in the kernel.

## NaN contract

The check is **always on**. It is a raw **bit-pattern** test (exponent all-ones with non-zero payload — every NaN encoding, either sign, quiet or signaling), *not* a comparison against a sentinel key: the order-preserving encode maps the two signed NaNs to opposite ends of the key space, so comparing keys catches only the single fp32 encoding `0x7FFFFFFF`. `v != v` is not usable — the build enables `--use_fast_math`. Rows whose visible length is `<= topk` are never NaN-checked.

`abort_when_nan_found=True` (default) calls `trap()` / aborts. With `False`, such a row leaves `0x3F3F3F3F` in `output_idx[row, 0]` and the rest of the row is undefined — a NaN row must be excluded from any value comparison, as the suites do.

## What `backend="torch"` is, and is not, as the validation arm

`"torch"` is the right arm for **adapter and interface** work, and it is not an
oracle for the *ranking*.  Both halves are measured; keep them apart.

- **It is not a bare `torch.topk`.**  `topk_torch` (`interface.py:387-557`) is
  ~140 lines of contract glue over one `torch.topk`: the window mask, the
  re-pick loop for rows whose window holds fewer than `k_eff` values above the
  `-inf` mask, the padding fills, `output_idx_offset`, `sorted_index`'s stable
  re-order, and the NaN sentinel.  **The glue is where the bugs have been** —
  two fixed ones sit in its own comments.  So it is a *reference*, not ground
  truth.
- **A set difference against it is not a defect.**  The contract is
  `min(selected) >= max(unselected)` and says nothing about ties.  Measured with
  `NormalFloatDistribution` (bf16, b 8/64, v 4096/65536, k 512/2048, with and
  without a window, `/tmp/dsab/tie2.py`) both arms pass that contract on every
  row, yet they agree on the selected **set** for only **11/64 and 22/64** of
  rows — because **7 to 44 elements of a row tie at the k-th value**, and which
  of them is selected is unspecified.  Set (or elementwise) agreement is
  therefore **not** a usable gate on bf16; the contract check is.
- **Where it does serve as an oracle: the fp32 windowed/sorted cells.**  Same
  measurement, `(8, 4096, 512, fp32, window, sorted_value)` and the same with
  `sorted_index`: `maca_c` and `torch` agree **elementwise, 100%**.  That is the
  measurable differential across dtypes, and it is the one to use.
- **The cost is real.**  The re-pick loop is a **per-row python loop** over
  `(~in_window).any(...)`.  On bf16 it fires often enough to dominate a timing
  run, which is why the snapshot's `torch` column is a **bare** `torch.topk`
  (`perf_snapshot.time_torch_reference`, matching what `tests/test.py` times)
  and **not** `backend="torch"`.
- **Do not copy it as production glue.**  `torch.arange(vocab_size)` plus a
  full `input.float()` are not valid for the shapes here, and it assumes the
  caller already rejected `sorted and not return_value` — `if sorted and
  return_value` is the only use of `return_value` in that function, so
  `sorted_index=True, return_value=False` can come back with `out_val=None`.

So: use it to validate **adapter behaviour** — window handling, the fills,
`output_idx_offset`, `sorted_index`'s ordering, dtype/index-dtype plumbing, the
NaN sentinel, that no row is left undefined — elementwise on **fp32** cells and
against the contract on **bf16** cells.  Do not gate on `maca_c == torch`.

## Tests

There is **no pytest suite and no `conftest.py` here** (unlike the host repo). The suite is upstream's, landed unmodified under `tests/`, driven by its own `__main__`.

```bash
PYTHONPATH=. python tests/test.py --perf-only              # performance grid, 95 cases
PYTHONPATH=. python tests/test.py --perf-only -nc          # skip the inter-case cooldowns
PYTHONPATH=. python tests/test.py --perf-only --dtype bf16 # 90 of them, ~1 min on C500

PYTHONPATH=. python scripts/official_slice.py              # correctness, 200/200 sampled
PYTHONPATH=. python scripts/official_slice.py                    # the default (torch)
PYTHONPATH=. python scripts/official_slice.py --backend torch    # the reference
PYTHONPATH=/path/to/mcDeepGEMM:. python scripts/official_slice.py --backend deep_gemm
```

- `tests/test.py` builds a correctness table (105,138 cases, hours) and a performance grid, and runs both through the same `run_testcase`. Its checks are the contract's own — index range, uniqueness, `value_i == input[index_i]`, the definitional `min(selected) >= max(unselected)`, the NaN guard, the orderings. **No reference implementation is computed anywhere**, so nothing can drift from the contract it checks. Every perf case is checked first and timed second, so a case that selects wrong is reported as a failure rather than as a time.
- `scripts/official_slice.py` drives a seeded uniform sample of the same table through the same official `run_testcase`, capped at `batch_size * vocab_size <= 2**28`.
- `run_test.sh` is the entry point that wraps both arms, records the extension md5 + the `CUDA_VISIBLE_DEVICES` in force + an `mx-smi` snapshot beside each log, and reports a stale extension rather than refusing to run:

  ```bash
  ./run_test.sh --perf --dtype bf16 -nc # the perf grid, ~1 min on C500
  ./run_test.sh --test                  # correctness sample, 200/200
  ./run_test.sh --all                   # both, perf first
  CUDA_VISIBLE_DEVICES=3 ./run_test.sh --perf   # pick the device with the env
  ```

  It has **no exclusivity gate** on purpose: `pgrep` cannot see device pinning, and `mx-smi` was measured on this box lying both ways (`--show-process` said "no process found" while a job ran; `--show-all-process` put a process holding 4 GB on device 3 under GPUs 0–2). Pick the device with `CUDA_VISIBLE_DEVICES`, run, and read the recorded md5 before comparing two runs.
- `--backend` is the one thing the official suite cannot express (its call site passes no `backend=`), so the driver rebinds `deep_select.topk` for the run rather than editing the official file.
- Two edits under `tests/kernelkit/` are the whole delta from upstream there, both required to run on MACA at all: `platform.py` asks torch whether it can see a device instead of grepping `lspci` (a MACA part does not enumerate as an NVIDIA controller), and one PEP 701 f-string at `stress.py:292` is rewritten for Python 3.10. `tests/lib.py` carries two more of the same kind, about this torch's missing UInt kernels: `torch.randint(...).to(torch.uint16)` and `result.view(uint).copy_(...)` both raise `NotImplementedError: "copy_" not implemented for 'UInt16'` under `torch.set_default_device("cuda")`, which with the default device set is *every* `randint` in the distributions — the harness could not generate a case at all. `Distribution.as_uint` casts on the CPU and moves the result; `Distribution.put_uint_bits` writes through the equal-width **signed** view. Both are bit-exact; neither changes what a case contains.
- Not covered by either arm, recorded rather than papered over: the contract rejections (strided row, wrong dtype, `topk` out of range, undersized output buffer) — the official table asserts on values and has no exception cases — and `begin` / `hint` / caller-allocated `output_idx`, which the official call site always passes as `None`.

### Full-table correctness gate (C500, ~17.5 min)

```bash
for i in 0 1 2 3; do
  PYTHONPATH=$PWD python scripts/official_slice.py --backend maca_c --sample 1000000 --shard $i/4
done
```

82,170 cases, **must be 4 serial shards** (~5.6 min each). **Do not run 8 concurrent** — 8 concurrent torch processes make CUB's onesweep radix sort wedge the device and take the machine down with it (measured). Serial is just as fast; the cost is per-case construction.

### Performance snapshots — `scripts/perf_snapshot.py`, `perf_data/<device>/<stamp>/`

```bash
CUDA_VISIBLE_DEVICES=2 PYTHONPATH=$PWD:$PWD/tests \
  python3 scripts/perf_snapshot.py                       # the official grid
  ... --deep-gemm-axes                                   # + the host repo's fp32 grid
  ... --dry-run                                          # print the plan, measure nothing
  ... --cases-file extra.json                            # add cases, no code change
./run_bench.sh [--full] [--quick] [--set-baseline] [--compare-only] [--list]
```

Writes `perf_data/<device>/<YYYYmmdd_HHMMSS>/` — following the host repository's
`deep_gemm/tests/perf_data/` layout — with `manifest.json` plus
`deepselect_perf.csv`, **one row per (cell, backend)**. Three backends:
`maca_c`, `torch`, and `deep_gemm`. `run_bench.sh` is the orchestrator; it keeps
a `baseline` symlink under `perf_data/<device>/` and compares against it with
`tools/compare_snapshots.py`.

**`<device>` is the device's own name, not the arch family** (`MetaX C500` ->
`MetaX_C500`). The directory has to answer *which board was measured on* —
`MetaX C600` and `MetaX C600-U` are both xcore1600 but have different clocks, a
different read wall and a different SM count, so a family-named directory stacks
two machines silently. The arch family is still recorded **per row** as the
`chip` column (`metax_xcore<N>`) and in the manifest, and a reader filters on
that to find same-ISA records; it is never inferred back out of the directory
name. Both sides derive the name from the same call — `perf_snapshot.py`'s
`device_dir_name()` and `run_bench.sh`'s `device_dir` — so a snapshot and its
`run_bench.sh` wrapper cannot land in different directories. The `--device N`
CLI flag is unrelated: it is shorthand for `CUDA_VISIBLE_DEVICES=N`, the GPU
index, not the directory name.

**The directory is one device's, so a mixed-device tree has one `baseline` per
device.** A record taken on another board is not a comparison for this one;
`perf_data/` is expected to hold several `<device>/` trees, each with its own
`baseline` symlink and its own history.

**The recorder is thin on purpose: the measurement is the harness's.** Every
piece is `tests/test.py`, called rather than re-implemented — `performance_cases()`
for the grid, `check_result` / `check_call_contract` for the checks, `bench_topk`
for the timing rule, `bench_torch_reference` for the reference arm (a bare
`torch.topk`, which is what the official runner times — **not**
`backend="torch"`, whose reference implementation pads, masks and converts
around the same call and launches ~21 kernels where this launches one). The only
axis this adds is `backend`. Verified: same binary, same device, the snapshot
reproduces `tests/test.py --perf-only` cell for cell at a **median 0.25%, max
1.3%** per cell above 100 µs (total across the grid `+0.01%`); the only cells
above 5% are the two ~6 µs short rows, where the absolute difference is ≤ 1.2 µs.
Two consequences worth stating:

- `tests/test.py` must import cleanly with no side effects for this to work.
  **Do not put work at its import time.**
- `run_bench.sh` drives the official grid with `python tests/test.py`, not
  `./tests/test.py` — upstream's file is not executable and a bare path dies at
  rc=126 before running anything.

Facts the CSV records and the traps in reading it:

- **The chip is in the path, the manifest and every row.** A perf record whose
  artifact is not identified is not a record: the manifest carries
  `device_name`, `sm_count`, torch version, *both* repositories' commits, the
  extension's md5 and the status counts.
- **`status` is a column, and it is how a backend declines.** `pass` / `fail` /
  `unsupported`. `deep_gemm` is `unsupported` on **all 95** official cells —
  the grid is bf16 and that backend ranks float32 only — and a table that
  dropped those rows would read as "the two backends agree everywhere" when only
  the fp32 cells were ever compared. `fail` still carries its time when one
  could be taken: a case that selects wrong is a defect whatever it runs at.
  `deep_gemm` numbers only exist at all with `--deep-gemm-axes`.
- **The kernel-name filter is case-insensitive**, here and in `tests/test.py`.
  `torch.topk` is spelled two ways and only one has a lowercase `topk`:
  `at::native::mbtopk::*` (large inputs) and `at::native::gatherTopK_opt`
  (small ones). Measured on C500, an exact-case filter finds **nothing** for
  `torch.topk` at b6-v{256,4096}, b256-v1024, b512-v1024 and everything smaller,
  leaving those cells untimed; case-insensitive finds exactly one kernel on both
  paths, which is what makes `len == 1` the kernel time rather than an e2e span.
- **`relative_pct_vs_maca_c` is the comparison**, defined as this repository's
  own kernel = 100%, so **>100% means that backend is faster than `maca_c`** (it
  is `maca_c_us / that_us * 100`). It is repeated on every row of a cell, so a
  row reads on its own.
- **`bandwidth_pct_of_wall`/`bandwidth(GB/s)` are the operator's own traffic,
  not the kernel's roof.** `Byte(MB)` counts the input read plus the outputs
  written, which is the quantity `tests/test.py:138` prints as TB/s; it is
  recomputed per cell from the shape, so it is comparable across backends. It is
  NOT the pure-read wall: for the official grid a median 19% of the 1,487 GB/s
  C500 wall, because 54 of 108 cells are under 16M elements and the grid is
  weighted by batch rather than bytes (the 17 cells at batch 4096 are 152.6 ms
  of the 200.5 ms total). The same kernel's pass 1 measures 94.9–97.9% of the
  wall on a full-length row.
- **Adding a case is data.** `--cases-file FILE` takes a JSON list of
  `lib.TestParam` fields; only `batch_size`, `vocab_size` and `topk` are
  required, and `dtype` / `out_idx_dtype` take `bf16`/`fp32`/`int32`/`int64`.
  An unknown axis or dtype is an error, not a default.
- **`torch.set_default_device("cuda")` must be set before generating cases**
  (the environment traps below); the script does it first thing, and
  `--cases-file` cases go through `lib.generate_testcase` like every other.

## Testing and benchmarking environment traps

1. **`torch.set_default_device("cuda")` must be set** before generating cases. `tests/lib.py`'s `generate_testcase` otherwise builds CPU tensors, and the extension dereferences a host pointer — reported as `Xnack Error / ATU Fault`, which looks like a kernel bug and is not.
2. **A timing run needs the device to itself, and `mx-smi` cannot tell you whether it has it.** `pgrep -af python` first; running alongside other torch work produces ~16 phantom regressions. Measured 2026-09-15 (`20260915_000332`, device 3, launched with `--device 3`): the artifact was **bit-identical** to the baseline's (`extension_md5 d07e79f595b00dfc2683b941a8484a07`, same torch, same part, clean tree at `dc8525d`), and the comparator still reported **`maca_c` 40 regressed / 12 improved, grid total +12.42%** — **and `torch` moved with it, +1.61%**, against that same binary. Two readings settle it: a code change cannot slow `torch.topk` down, and the large cells sat at **1.00×** (`4096-1048576-512`, `4096-524288-1024`, `4096-262144-*` all exactly 1.00×) while the +12.42% came from a tail of small cells at 1.6–2.2×. `mx-smi` read device 3 at **0% / 859 MiB** at the instant that run started, and 41,947 MiB four seconds later; the run's own `run_header.txt` shows `test_attention`, `expected_verdicts.py` and `test_einsum` live. So a pre-run sampling of `mx-smi` is not a gate — **`pgrep -af python` over the whole box is, and a run whose header shows another torch process is not a measurement, whatever the util column said.**
3. **The official perf table seeds cases from a global counter**, so two runs use different data. A/B comparison must pin the seed yourself: build cases with `tests/lib.py`'s `TestParam`, set `p.seed`, call `lib.generate_testcase(p)`, and time **kernel time** (not e2e) with `tests/kernelkit`'s `kk.bench(fn, p.num_runs)`.
4. **The baseline binary must be reproducible**: `git stash` to the tree under comparison → build → save the `.so` → switch back. That is how the ledger's numbers were obtained.
5. **Alignment**: rows whose length is a multiple of 8 and offset-16 B-aligned take the vector path, otherwise a scalar fallback (`bf16x8_is_aligned`). Mixed vector + tail was intermittently racy on a 1024-thread block, so odd-length rows uniformly use one load mode.
6. **mxcc's compile cache** is `~/.deep_gemm/cache` (keyed by entry name + source digest); a source edit recompiles only the affected entry. Counting `kernel.*` dirs under a fresh `DG_JIT_CACHE_DIR` is how you check a routing change did not grow the compiled-kernel count.
7. `NormalFloatDistribution` (the official table's data) is **not** bit-pattern uniform — many elements per row crowd into one high byte. This is the fact every optimization here is organized around.
8. **A toolkit mismatch at *run* time reports as `mcErrorInvalidDeviceFunction`, and it looks exactly like a kernel defect.** The extension is linked against `libmcruntime.so` by soname, and torch's `-Wl,-rpath` becomes a `DT_RUNPATH`, which `LD_LIBRARY_PATH` **overrides**. Measured 2026-09-15 on this box, the *same* `deep_select_xcore1600*.so`, device 1, one variable changed:

   | `MACA_PATH` (and the `LD_LIBRARY_PATH` it derives) | result |
   | --- | --- |
   | `/opt/maca-3.8.1` — `libmcruntime.so` md5 `2da1af3a95a81929bbf301324fb5fe66` | **202/202** |
   | `/opt/maca` — `libmcruntime.so` md5 `9f7a96f64abc67c6ae52ab0001ea5bc5` | **0/202**, every case `StatGetFunc error` + `mcErrorInvalidDeviceFunction` |

   `/opt/maca` is a symlink that was re-pointed on 2026-09-15 05:10 to `maca-3.5.3.17-20260915`, a different SDK generation **that has no `tools/cu-bridge` at all** — so it is the wrong root for both building *and* running a 3.8.1-built artifact, and it is what the shell's profile exports. The tell that it is not a kernel bug: the count is 0/202 rather than a plausible partial, and *every* cell fails identically including `vocab_size=1`, which launches no meaningful work. Before blaming a kernel for a device-function or device-side-assert failure, print the resolved `libmcruntime.so` path and md5. Both `build.sh` and `install.sh` derive `CUDA_PATH`/`CUDA_HOME`/`CUCC_PATH` from `MACA_PATH` for this reason and `run_test.sh` derives `LD_LIBRARY_PATH` from it, so passing `MACA_PATH=<a working toolkit>` to all three is the whole fix.

## Performance-change discipline

Beyond the host repo's general rules (state the principle and the magnitude; keep rejected experiments out of commits), this repo's rules are in the handover §7 and ledger §7. Every performance or dataflow commit message is this skeleton, and a missing item means it is not a record:

- A one-line imperative title stating the **principle**, not "optimized X".
- Why: the old approach's cost, with measured numbers.
- A before→after table over the representative cells, in **both currencies** — µs **and** GB/s, with the trip count and the % of the read-only wall. Logical GB/s is `B × V × 2 B ÷ kernel time`; the wall is a measured **1,487 GB/s streaming read** on C500 (1,344 GB/s mixed) — do not back it out of the kernel. **The wall is per-part; measure it for the part you are on.** On C600U it is **1,545 GB/s**, measured with a purpose-written `uint4` grid-stride read kernel (`/tmp/readwall.cu` in the session that took it — a torch reduction measures 274 GB/s on the same device and is *not* the wall): 224 blocks → 1,545, 448 → 1,532, 896 → 1,523, 1792 → 1,401. The two numbers being close is a coincidence of these two parts, not a constant.
- A **roofline verdict** for the affected cell: if it is not bandwidth-bound, say what it *is* bound on (currently: per-CTA dependency chain — `load → key transform → compare → shared atomic` — and serialized shared atomics).
- The gate results. Both arms: `95/95` perf (`./run_test.sh --perf`, ~100 s) **and** a correctness arm — `82170/82170` for the full-table 4-shard run on C500 (`~17.5 min`), or `200/200` for the seeded sample (`./run_test.sh --test`, ~63 s on C600U) when the change is being iterated rather than landed. Say which one you ran.
- An architecture-boundary statement: changes confined to `csrc/xcore1000/` leave xcore1600 byte-identical, so **no C600U validation is owed**. Say so explicitly when true. (Byte-identical is still the right claim — but as of this writing xcore1600 is *not itself validated*, so "no C600U validation is owed" is an argument about the byte-identity of the artifact, not a claim that xcore1600 works. See Known holes.)

These cells frequently have **no compute roofline** — the kernel does a few comparisons and one histogram increment per element and has no FLOP — so "both currencies" lands as logical-GB/s × trips versus the read wall plus a per-CTA limiting factor.

**A contended run is not a record, and it must not move the baseline.** The tell is that **every** backend moved against the same bit-identical `.so` (see the environment traps §2 for the measurement). One more is available before spending a run: `run_bench.sh --compare-only` prints the per-cell `relative_pct_vs_maca_c` for the latest run against the baseline for free, and `deep_select_perf.csv` carries the same ratio on every row — a contended grid shows the *reference* arm drifting, which no tree change can cause. Two rules follow:

- **Do not pass `--set-baseline` on a run whose header shows another torch process.** `run_bench.sh` repoints whenever every arm passed (`BENCH_STATUS -eq 0`), which a contended run does — all three arms "pass", they just measure the wrong thing. Repointing then bakes a phantom regression into the baseline that later runs are compared against, and the run_bench warning ("--set-baseline was given, so the baseline WILL move past this. That is a decision") is exactly the decision not to make. Land the run as a directory, cite the ratio evidence, and re-measure on a quiet box.
- **A baseline directory written by `perf_snapshot.py` directly is invisible to `run_bench.sh`.** `latest_result_dir()` requires a `manifest.json` **and** a `run_header.txt`; a baseline produced by the snapshot rather than by `run_bench.sh` has only the former, so `--compare-only` and the automatic comparison both report "no baseline at …" and skip. A repoint under `run_bench.sh` will therefore look like it worked and change nothing about what gets compared. (Both old baselines were deleted on 2026-09-15 — `perf_data/` is empty and awaited a clean re-measure on a quiet device; a `run_bench.sh --set-baseline` run is what is supposed to create the first one, precisely so it has a `run_header.txt` and is not born invisible.)

## Build-change discipline

The build has one failure mode worth a rule, because it cost a day and the bug was one missing file:

**A synthesized `CUDA_HOME` is a contract with torch's MACA patch, and that contract is not written down anywhere.**
A build change here is verified by *building both targets and running them*, never by reasoning about which flags
reached the compiler. Concretely:

1. `rm -rf build && ./build.sh` for every target — `build_ext --inplace` compares timestamps and will silently reuse a
   stale object directory, so an incremental "it built" proves nothing about a toolchain change.
2. The device compiler and the host linker are **printed by `setup.py`** (`deep_select: device compiler is …`). Read
   those two lines; do not assume.
3. `readelf -d` the extension and read the NEEDED list — it is the receipt for which libraries the toolchain actually
   pulled in. The expected set on this install is `libmcruntime.so` (the MACA CUDA runtime, reached because cucc
   translates torch's `-lcudart`), plus `libToolsExt_cu.so` / `libruntime_cu.so` / `libmcToolsExt.so` /
   `libmaca_mathlib_host.so` / `libmccompiler.so`, which come from cucc's link-time `adder` in `conf.json`. Those are
   cucc working, not artifacts to strip. A `libcudart.so` here would mean the translation did *not* happen.
4. `./install.sh` end to end (it is a different code path from `build.sh` — wheel, pip, symlink).
5. When reproducing a baseline for comparison, **patch the baseline until it builds.** An old tree that does not build is
   not evidence about behavior; state in the commit that the `gnu` symlink was added to make it build, and that the md5s
   differ.

When a build change is suspected of changing *behavior* (not just the artifact), the falsifying test is a fixed,
exactly-representable input — `arange`-derived, so the expected indices can be written down — not a random one, and the
comparison must be run **several times**: a racy kernel's single run agrees with anything. Here both the old-shim and the
cu-bridge artifacts were wrong on every run and wrong differently on each, which is what established that the toolchain
change neither introduced nor repaired the defect.

Land it as:

```bash
D=/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
git -C $D status --porcelain          # confirm only the intended files
git -C $D commit -a -F /path/to/msg   # or: git -C $D add <paths> && git -C $D commit -F ...
git -C $D push origin main
```

Never a bare `git commit -a` (see repo identity, top).

**Debugging mis-ranked output**: the fastest localizer is to write the kernel's threshold triple `{wide_threshold, above, fine_threshold, remain, num_staged, last_remain}` into `output[topk-8+tx]` on `blockIdx.x == 0 && tx < 8` and compare it against the same row computed with torch on CPU. That is how the `0xBFA` vs true `0xBFC`-class bug was found in one step. Clean it up afterwards — `rg RKPROBE` must be 0. (This probe is written for `maca_topk.cu`'s threshold triple; `csrc/xcore1600/` has a different dataflow and no equivalent variables.)

Symptom → first look (handover §8): wrong-but-not-much (a few slots, rank off by a few) → the threshold's subtraction convention (the `above + c > remain_topk` rule); `Memory Violation(0x4)` / `ATU Fault` → first the probe's missing `set_default_device`, then an out-of-range threshold leaving a shared variable uninitialized; one cell slow while others unchanged → occupancy, check `static + dynamic` against 32,768; source changed with no behavior change → a stale `.so`; compile-time `undeclared identifier` → constant/helper ordering in `radix_core.cuh`; **wrong indices that change run to run → a lane-width bug** (a 32-bit mask or `/ 32` on a 64-lane wave in `csrc/xcore1600/`), not a threshold-convention one — a fixed offset is the threshold, a varying one is a race.

Reproduce the xcore1600 hole with no seed and no harness, so the expected answer is written down rather than computed.
**`backend="maca_c"` is not optional here** — the process default is `torch`, so a bare call would exercise the
reference and "reproduce" nothing at all.

```bash
CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. python -c "
import torch, deep_select
b, v, k = 2, 512, 8
x = torch.arange(v, device='cuda', dtype=torch.float32).unsqueeze(0).repeat(b, 1).to(torch.bfloat16)
_, ii = deep_select.topk(x, k, backend="maca_c")   # the default is torch now
print(ii[0].tolist())   # want [511,510,509,508,507,506,505,504]
"
```

`arange` matters twice over: the values are exactly representable in bf16 (so the expected indices *are* `range`), and
it removes the seeding that made an earlier reading of this look deterministic when it is not. Run it a few times.

## Known holes (recorded, not hidden)

- **`csrc/xcore1600/` selects wrong on a C600U. This is measured, not suspected, and it is the first thing to
  fix.** It is *contained* — every family builds `csrc/xcore1000/`, unconditionally and with no override
  (see Kernel architecture above), which passes 200/200 on a C600U — and that is not merely
  correct there but **faster**: see "Can a C600U run the C500 kernel", which measures 1.5-2.9x against the port and
  records why the containment is the right answer rather than a fallback. The port is still in the tree and
  compiling; building it is a one-line source change in `setup.py`, which is how the audit is done.
  On a `MetaX C600-U` (reports `sm89` → family 1600, so `backend="maca_c"` resolved *here*, not to `maca_topk.cu`;
  that measurement was taken with the port built, which is no longer what a build produces):
  a monotonic row of `0..511` with `topk=8` returns indices like `[448..455]` where the answer is `[511..504]`, and
  **the wrong answer varies run to run**: eight consecutive invocations of the identical command on identical input
  produced **eight distinct** index sets, all wrong, each a different mix of indices scavenged from the middle of the row.
  (An earlier reading of this as "deterministic" was an artefact of seeding; an `arange` input needs no seed, and then it
  is plainly racy.) A random row returns `1.5e+37` for a row whose true max is `3.3`; `min(selected) >= max(unselected)`
  fails; indices are unique but not the top ones. On other shapes it traps (`[topk_select] NaN detected` on input
  containing **no** NaN — confirmed `isnan(x).sum() == 0`) or raises `device-side assert`. Measured on the official slice:
  **4/200 passed** (`--backend maca_c`) against 93/200 for `--backend torch` at the time. **Both numbers are gone now**:
  the 107 `torch` failures were the `tests/lib.py` UInt `copy_` bug (fixed in `ce68c25`), and the 196 `maca_c` failures
  were this hole — with the routing above, `--backend maca_c` is **200/200** and so is `--backend torch`.
  **It is not a build regression.** Both artifacts are racy, and the old one is *more* so — the pre-cu-bridge build (old
  shim, with a `gnu` symlink patched in so it builds at all) gave **7 distinct** wrong sets in 8 runs, the cu-bridge build
  **8 of 8**. The two `.so` md5s differ, so this is same-source-same-behavior under a toolchain change, not a regression.
  The port was only ever validated by compilation — the README says so, and CLAUDE.md's own "Beware reading upstream's
  CUDA-era code as a model" note predicted exactly this.

  **Re-measured 2026-09-15 on an `arange` row: the wrong answer is DETERMINISTIC, and the "racy" reading was an
  artefact of random input.** Five consecutive runs of `2 x 512 k=8` returned the identical index list. The *set* is
  right (`{504..511}`), the pairing is right (`value == input[idx]`), and the **values** are wrong: the row is
  `[512, 506, 508, 510, 512, 514, 516, 518]` where `input[idx]` is exactly that. 512 is exact in bf16, so this is not
  precision, and it is not a shuffle — it is a systematic value corruption. `min(selected) >= max(unselected)` fails on
  every row of the official slice. Run-to-run variance is a tell for a lane-width bug, but it is not the tell *here*, and
  treating it as one sent the audit at the wrong shape.

  **Diagnosis: a rescan reachable only on a C600U.** The port's config tuples stage more per CTA than a 64 KiB SM can
  hold, so a path that re-reads the row once its staging buffer is exhausted cannot be taken on C500 — which is the only
  part the port was ever run on. Where it can be taken, it drops the high members of a **tied run at the threshold** and
  writes every member back with the *group's low member's* key: five 508s for a group `{508, 510, 512}`, three 504s for
  `{504, 506}`. That is exactly the measured shape, it explains why the index set survives and only the values do not,
  and it is reachable only on the larger part. The 32-lane list below remains a live suspect for the *other* symptoms
  (`device-side assert`, the NaN trap on NaN-free input) — it is no longer the explanation for this one.
  **Prime suspect, already documented:** the port carries CUDA's 32-lane model on a 64-lane wave. `common_parts.cuh:121`
  (`NUM_WARPS = NUM_THREADS / 32`), `:1463` (`__ballot_sync(0xFFFFFFFF, …)`), `:488`/`:496`/`:497`/`:784`/`:791`/`:1432`/
  `:1440`/`:1469` (`__reduce_add_sync(0xFFFFFFFF, …)`), `v3/topk_select.cuh:54` (`threadIdx.x % 32`), and the `0xFFFFFFFF`
  mask in `utils.cuh:7-12`, whose own comment says the mask's validity rests on "**前提是 MACA 的 warp 宽度确为 32**" —
  which it is not. On a 64-lane wave `0xFFFFFFFF` names the low half, so every one of those under-counts silently.
  **It is worse than "wrong": it does not run.** With the port built (the `setup.py` source line pointed at
  `_xcore1600_sources()`) and *correct* inputs
  (`torch.set_default_device("cuda")` set, per the environment traps below), three cells — `b4096-v1024-k512`,
  `b4096-v16384-k512`, `b512-v262144-k512`, all bf16 — all die with `device-side assert` before a single timing is
  taken. So this tree cannot be benchmarked against the rerouted one cell for cell; there is no before to put beside the
  after.
- `backend="deep_gemm"`'s host kernel collects the members of the threshold *coarse* bin (half-precision ordered key `>> 6`) before refining, and the chunked kernel silently drops members past its staging capacity. A row with more than 4096 values in one such bucket gets a top-k of an arbitrary subset, varying run to run. Filed as a strict `xfail` in the host repo: `deep_gemm/tests/test_indexer_topk_selector.py::test_selector_candidate_overflow`. **`maca_c` has no such hole** — but note the xcore1600 hole above is a `maca_c` hole, so this sentence is about the `deep_gemm` backend only.
- The `radix_topk_row_bf16_k` static-k row used by the chunked path still runs the 8-bit coarse level and the 3,514-slot arena; it has not received coarse12.
- The fp32 row is a separate codebase path whose overflow handling is multi-round full-row rescan (up to 8 trips). Same "coarse level too coarse" disease, different cure — a 32-bit key cannot be resolved in two levels the way a 16-bit one can. Retesting fp32 cells is mandatory when touching it.

## Torch ABI / host compiler notes

**The extensions do not link torch.** They are built at the Apache TVM FFI ABI
(`csrc/ffi/`), loaded with `tvm_ffi.load_module` from `deep_select/_binding.py`,
and export two `__tvm_ffi_*` symbols each. The gate is mechanical and is the
reason the migration happened:

```bash
readelf -d deep_select/deep_select_xcore1000*.so | grep -iE 'libtorch|libc10'   # empty
nm -D     deep_select/deep_select_xcore1000*.so | grep -icE 'c10|torch|at::'    # 0
nm -D     deep_select/deep_select_xcore1000*.so | grep -c  '__tvm_ffi_'         # 2
```

The pybind11 build linked six torch libraries, which tied the extension to the
host's torch build (`c10_cuda_check_implementation`) and to a cpython tag its
real interface never used. Neither applies now.

What a caller still needs is stated rather than implied: **torch at the call
site** (tensors cross as DLPack, but a `torch.Tensor` is what has the
`__dlpack__` protocol, and output buffers are allocated with `torch.empty`),
and **`tvm_ffi` to load the artifact at all**. `deep_select/interface.py`
imports torch for its own checks and its `backend="torch"` reference arm -- that
is the harness layer, unchanged, not the ABI.

**A kernel launch must run inside `_binding.launching()`.** The C++ side reads
its stream from `TVMFFIEnvGetStream`, which reports the null handle unless
something installs torch's current stream; the null handle is the legacy default
stream, which does not synchronize with torch's non-blocking side streams. The
one call site is wrapped; `DEEP_SELECT_NO_STREAM_GUARD=1` is the explicit
escape hatch for bisecting it.

**Two things did not decouple and are not claimed to have:**
`no_python_abi_suffix=True` does not take effect through torch's
`BuildExtension`, so the artifact filenames still carry `cpython-310`; and
`get_alignment_requirement` returns `Array<int64_t>` rather than the pybind
build's `std::pair` (tvm-ffi cannot carry `std::pair`), which
`interface.py` normalizes to a tuple.

`csrc/xcore1600/api.cu` is host code but is compiled by mxcc's host pass (clang 19), so `std::format` *is* available there now that the file is a `.cu`. The one check message that needs formatting keeps its `snprintf` anyway: it is the ABI-safe spelling at this boundary (and `<format>` needs GCC 13's libstdc++; the host is GCC 11.4). Do not "fix" it back. Anything including `<cuda_runtime_api.h>` must not depend on cu-bridge's compatibility layer for `__nv_bfloat16`: `csrc/structs.h` includes `<maca_bfloat16.h>` so that `api.cu` and every instantiation TU see the *same* `maca_bfloat16`, and `TopkSelectConfig<maca_bfloat16, ...>`'s template entity is one symbol on both sides.
