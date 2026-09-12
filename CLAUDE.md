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

- `"maca_c"` (default) — the MACA kernel this device has. Which kernel that is is a property of the device, not a call-site choice.
- `"torch"` — reference implementation built from torch ops; runs on any device/dtype, including where no kernel is built. Also the differential-check arm.
- `"deep_gemm"` — the host repo's `deep_gemm.fp32_indexer_topk_selector`, called through its Python API, imported lazily. Implements a **strict subset** of the contract (float32 only, `topk <= 2048`, unordered); what it cannot serve raises `UnsupportedByBackend`, never a narrower answer.

## Build

```bash
./build.sh                             # this device (CUCC_TARGETS=native)
./build.sh --all                       # xcore1000,xcore1500,xcore1600
./build.sh --targets xcore1000,xcore1600
./build.sh --clean                     # rm -rf build first
./build.sh --list                      # print resolved targets, exit
```

`build.sh` is the entry point; it wraps the same `setup.py` calls shown below.
It mirrors the host `build.sh` in shape but deliberately does **not** run
`bdist_wheel` — see the `pip install .` note below for why `--inplace` is the
working path — and it removes the stale in-place `.so` before building rather
than trusting `build_ext`'s timestamp comparison. It resolves targets through
`deep_select/_arch.py` so the script and `setup.py` cannot disagree, and fails
loudly if a requested architecture produced no extension. `MACA_PATH` and
`MAX_JOBS` are honored; `CUCC_TARGETS` from the environment always wins over
the default.

```bash
./install.sh                           # build a wheel and install it
./install.sh --targets xcore1600       # for another architecture
./install.sh --all                     # xcore1000,xcore1500,xcore1600
./install.sh --build-only              # leave the wheel in dist/, install nothing
```

`install.sh` additionally installs into the active environment
(`pip install <wheel> --force-reinstall --no-deps`, matching the host repo's
`install.sh`). Two things it has to do that the host one does not:

- **Build the wheel by a single `bdist_wheel` run, then hand pip the file.**
  `pip install .` cannot work here (see below); building first keeps `setup.py`
  to one invocation.
- **Symlink the installed extension back under `deep_select/` as a last step.**
  Every suite here is run as `PYTHONPATH=. python tests/...` from the repo root,
  which resolves `deep_select` to the *repo* copy first — so an installed wheel
  alone leaves the suites exercising nothing. The link is created after the
  install, from `pip show`'s `Location` (asking `import deep_select` from the
  repo root would report the repo back and produce a symlink onto itself). A
  symlink rather than a copy, so re-running `build.sh` overwrites the target
  instead of racing it. Skip it with `--no-link`.

The underlying call, if you need it directly:

```bash
CUCC_TARGETS=xcore1000 python setup.py build_ext --inplace             # C500
CUCC_TARGETS=xcore1600 python setup.py build_ext --inplace             # C600, C600U
CUCC_TARGETS=xcore1000,xcore1600 python setup.py build_ext --inplace   # both
```

`CUCC_TARGETS` defaults to `native` (the device the build runs on); same variable and meaning as the host repo's `build.sh`. One extension **per architecture** — `deep_select/deep_select_xcore<N>.cpython-310-x86_64-linux-gnu.so` — because a config's shared-memory footprint is only valid against the architecture it was sized for.

**The stale `.so` must be deleted before rebuilding** — `build.sh` does this for
you, but a bare `setup.py build_ext --inplace` does not (see the handover §4):

```bash
rm -f deep_select/deep_select_xcore1000*.so \
      build/lib.linux-x86_64-cpython-310/deep_select/deep_select_xcore1000*.so
```

`build_ext --inplace` compares timestamps and **silently skips** the copy when the target is newer than `build/lib`. The `.so` lives under `deep_select/` and is gitignored, so it is stale by default. The `.so` md5 is sensitive to source line endings and is usable as a "which source did I actually measure" receipt (same source → same md5).

Build plumbing worth knowing before editing `setup.py`:

- Device code is compiled by **mxcc directly** (`--offload-arch=xcore<N>`), not through cu-bridge's `cucc` wrapper and not through any `-gencode` derived from the building machine. `setup.py` writes a `CUDA_HOME/bin/nvcc` shim (`_MXCC_NVCC_WRAPPER`) that strips torch's CUDA-dialect flags and execs mxcc. `TORCH_EXTENSION_ENABLE_XC1500_COMPILE` is refused outright — it would put a second `--offload-arch` in one extension.
- Host code (`csrc/xcore1600/api.cpp`) is compiled by `g++` explicitly: PATH resolves `c++` to cu-bridge's wrapper, which is not a compiler. `-DKERUTILS_IS_BUILD_ON_CUDA` is needed because kerutils defines it itself only for a CUDA compilation.
- `-use-fast-math` is passed with FTZ turned back off (`-Xclang -fdenormal-fp-math-f32=ieee`) so the one float conversion (`__float2bfloat16` of `value_oob_fill_value`) stays exact for a denormal fill. The ranking path is integer-only and indifferent.
- `pip install .` does **not** work, for an inherited upstream reason: `setup.py` stamps the version with `datetime.now()` and a PEP 517 install runs it twice (metadata then wheel); straddling a second boundary yields `Wheel has unexpected file name`. Build isolation adds a second failure (torch absent from pip's isolated env). Use `build_ext --inplace`.

MACA toolkit root is `$MACA_HOME` or `$MACA_PATH` or `/opt/maca` (symlink, currently `maca-3.7.0`).

## Kernel architecture

The kernel tree is split **by per-SM shared memory**, because that is what a top-K kernel's staging buffers are sized against. Which tree a device runs is a property of the device:

| tree | parts | kernel |
| --- | --- | --- |
| `csrc/xcore1000/` | C500 (64 KiB/SM) | `maca_topk.cu`, hand-written for MACA |
| `csrc/xcore1600/` | C600, C600U (128 KiB/SM) | the upstream kernels, ported |

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

### xcore1600 — the ported upstream kernels

`api.cpp` is the host dispatch + pybind11 module; `v3/` (bf16) and `v3_fp32/` (fp32) each hold `topk_select.cuh` + a generated `instantiations/` directory. `common_parts.cuh`, `bit_utils.cuh`, `utils.cuh`, `config.h`, `dispatch_utils.h` are shared.

Upstream's algorithm is kept (threshold-and-compact scan in a random block order, one global read per element); only its device-side dependencies were replaced: TMA tensor-map loads → cooperative `ldg`, mbarriers → a single buffer with `__syncthreads`, inline PTX → MACA builtins/plain C++. Config tuples were re-derived for 128 KiB (upstream's are sized for an H100's 227 KiB). `v3_cluster` was **deleted**, not ported — MACA has no cluster launch — and those shapes fall through to the general kernel with no dispatch arm.

**The config table has two halves that must be edited together**: `scripts/generate_instantiations.py` (the table, its arithmetic, and the `check_fits_maca` refusal) and the `TopkSelectConfig<...>` call sites in `csrc/xcore1600/api.cpp`. A mismatch is a **link error**, not a runtime one. Note also that `deep_select/_arch.py` duplicates the host repo's `deep_gemm/utils/arch_config.py` `XcoreFamily` rows (capacity + family spelling) by hand — it cannot import that package, so a change to either belongs in the same review.

Consequences of the port, all deliberate:

- `topk` in `(1024, 4096]` is **rejected** by the ported kernel: a `max_topk = 4096` tuple is not merely tight on 128 KiB, it is impossible — `surviving_topk_pairs` alone is `2 * 4096 * 8 = 65536` B and the extra-pairs region at least `(4096 + 4096) * 8 = 65536` B. On C600/C600U those shapes are served by `maca_topk.cu`, not rejected by the operator.
- `vocab_size < 2^23` is enforced by the ported kernel only (the fp32-simulated census). `maca_topk.cu` has no such limit; it ranks integer keys.
- `sorted_value` for bf16 is accepted by `maca_topk.cu` (same ordering key, so order and value/index pairing both hold) and rejected by the ported kernel, as upstream, and by `backend="torch"`.

## MACA warp intrinsics — the wave is 64 lanes

Get these right before writing any cross-lane code; every one of them is a
silent-wrong-answer trap rather than a compile error. Facts below are read from
the toolchain's own headers (`$MACA_PATH/mxgpu_llvm/lib/clang/19/include/
__clang_maca_device_functions.h`) and the official builtin guide
(OG-26013-000-F5_V01, shipped with the `maca-mxcc-builtins` skill). Follow that
skill (and `maca-kernel-dev-and-opt` for the wider workflow) rather than guessing.

**Wave width is 64, so every mask is 64-bit.** The guide's comparison builtins
(`uicmp`/`sicmp`/`fcmp`) are documented as "返回 warp 内 64-bit 比较结果掩码".
A `0xFFFFFFFF` mask is not a shorthand for "all lanes" here — it names the low
half of the wave only, and the result is a silently half-counted operation, not
a compile error.

| intrinsic | status on MACA | use |
| --- | --- | --- |
| `__ballot_sync(unsigned long long mask, int pred)` | **native — one `sicmp`** (`:2202`). 64-bit overload only | the correct primitive |
| `__activemask()` | **native — one read of the wave mask** (`__builtin_mxc_read_xmsk`, `:2206`). Free | pass as the mask: correct and adaptive |
| `__match_any_sync` | **software-emulated — a 32-iteration per-bit loop of `sicmp`** (`:1648`), i.e. 32× the cost of a ballot | **do not use** |
| `__reduce_add_sync` | **software-emulated — a 6-iteration `bsm_bpermute` loop** (`:200`), and it takes a `uint64_t` mask | prefer a ballot-based reduction; with `0xFFFFFFFF` it sums only the low 32 lanes |
| `__popc` | **does not exist on MACA** | `__popcll` (64-bit) |
| `__ffsll` / `__lanemask64_lt` | present | MACA's own 64-lane code uses this family (`maca_coalesced_scan.h:104-118`) |

**The rule that follows from the `__match_any_sync` row: do not reach for
warp-aggregated atomics via match-any.** A per-member `atomicAdd` is usually
cheaper than 32 mask ops per call, and match-any buys nothing that a
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

## NaN contract

The check is **always on**. It is a raw **bit-pattern** test (exponent all-ones with non-zero payload — every NaN encoding, either sign, quiet or signaling), *not* a comparison against a sentinel key: the order-preserving encode maps the two signed NaNs to opposite ends of the key space, so comparing keys catches only the single fp32 encoding `0x7FFFFFFF`. `v != v` is not usable — the build enables `--use_fast_math`. Rows whose visible length is `<= topk` are never NaN-checked.

`abort_when_nan_found=True` (default) calls `trap()` / aborts. With `False`, such a row leaves `0x3F3F3F3F` in `output_idx[row, 0]` and the rest of the row is undefined — a NaN row must be excluded from any value comparison, as the suites do.

## Tests

There is **no pytest suite and no `conftest.py` here** (unlike the host repo). The suite is upstream's, landed unmodified under `tests/`, driven by its own `__main__`.

```bash
PYTHONPATH=. python tests/test.py --perf-only              # performance grid, 95 cases
PYTHONPATH=. python tests/test.py --perf-only -nc          # skip the inter-case cooldowns
PYTHONPATH=. python tests/test.py --perf-only --dtype bf16 # 90 of them, ~1 min on C500

PYTHONPATH=. python scripts/official_slice.py              # correctness, 200/200 sampled
PYTHONPATH=. python scripts/official_slice.py --backend maca_c   # the default
PYTHONPATH=. python scripts/official_slice.py --backend torch    # the reference
PYTHONPATH=/path/to/mcDeepGEMM:. python scripts/official_slice.py --backend deep_gemm
```

- `tests/test.py` builds a correctness table (105,138 cases, hours) and a performance grid, and runs both through the same `run_testcase`. Its checks are the contract's own — index range, uniqueness, `value_i == input[index_i]`, the definitional `min(selected) >= max(unselected)`, the NaN guard, the orderings. **No reference implementation is computed anywhere**, so nothing can drift from the contract it checks. Every perf case is checked first and timed second, so a case that selects wrong is reported as a failure rather than as a time.
- `scripts/official_slice.py` drives a seeded uniform sample of the same table through the same official `run_testcase`, capped at `batch_size * vocab_size <= 2**28`.
- `--backend` is the one thing the official suite cannot express (its call site passes no `backend=`), so the driver rebinds `deep_select.topk` for the run rather than editing the official file.
- Two edits under `tests/kernelkit/` are the whole delta from upstream, both required to run on MACA at all: `platform.py` asks torch whether it can see a device instead of grepping `lspci` (a MACA part does not enumerate as an NVIDIA controller), and one PEP 701 f-string at `stress.py:292` is rewritten for Python 3.10.
- Not covered by either arm, recorded rather than papered over: the contract rejections (strided row, wrong dtype, `topk` out of range, undersized output buffer) — the official table asserts on values and has no exception cases — and `begin` / `hint` / caller-allocated `output_idx`, which the official call site always passes as `None`.

### Full-table correctness gate (C500, ~17.5 min)

```bash
for i in 0 1 2 3; do
  PYTHONPATH=$PWD python scripts/official_slice.py --backend maca_c --sample 1000000 --shard $i/4
done
```

82,170 cases, **must be 4 serial shards** (~5.6 min each). **Do not run 8 concurrent** — 8 concurrent torch processes make CUB's onesweep radix sort wedge the device and take the machine down with it (measured). Serial is just as fast; the cost is per-case construction.

## Testing and benchmarking environment traps

1. **`torch.set_default_device("cuda")` must be set** before generating cases. `tests/lib.py`'s `generate_testcase` otherwise builds CPU tensors, and the extension dereferences a host pointer — reported as `Xnack Error / ATU Fault`, which looks like a kernel bug and is not.
2. **A timing run needs the device to itself.** `pgrep -af python` first; running alongside other torch work produces ~16 phantom regressions.
3. **The official perf table seeds cases from a global counter**, so two runs use different data. A/B comparison must pin the seed yourself: build cases with `tests/lib.py`'s `TestParam`, set `p.seed`, call `lib.generate_testcase(p)`, and time **kernel time** (not e2e) with `tests/kernelkit`'s `kk.bench(fn, p.num_runs)`.
4. **The baseline binary must be reproducible**: `git stash` to the tree under comparison → build → save the `.so` → switch back. That is how the ledger's numbers were obtained.
5. **Alignment**: rows whose length is a multiple of 8 and offset-16 B-aligned take the vector path, otherwise a scalar fallback (`bf16x8_is_aligned`). Mixed vector + tail was intermittently racy on a 1024-thread block, so odd-length rows uniformly use one load mode.
6. **mxcc's compile cache** is `~/.deep_gemm/cache` (keyed by entry name + source digest); a source edit recompiles only the affected entry. Counting `kernel.*` dirs under a fresh `DG_JIT_CACHE_DIR` is how you check a routing change did not grow the compiled-kernel count.
7. `NormalFloatDistribution` (the official table's data) is **not** bit-pattern uniform — many elements per row crowd into one high byte. This is the fact every optimization here is organized around.

## Performance-change discipline

Beyond the host repo's general rules (state the principle and the magnitude; keep rejected experiments out of commits), this repo's rules are in the handover §7 and ledger §7. Every performance or dataflow commit message is this skeleton, and a missing item means it is not a record:

- A one-line imperative title stating the **principle**, not "optimized X".
- Why: the old approach's cost, with measured numbers.
- A before→after table over the representative cells, in **both currencies** — µs **and** GB/s, with the trip count and the % of the read-only wall. Logical GB/s is `B × V × 2 B ÷ kernel time`; the wall is a measured **1,487 GB/s streaming read** on C500 (1,344 GB/s mixed) — do not back it out of the kernel.
- A **roofline verdict** for the affected cell: if it is not bandwidth-bound, say what it *is* bound on (currently: per-CTA dependency chain — `load → key transform → compare → shared atomic` — and serialized shared atomics).
- The gate results (`95/95` perf + `82170/82170` correctness).
- An architecture-boundary statement: changes confined to `csrc/xcore1000/` leave xcore1600 byte-identical, so **no C600U validation is owed**. Say so explicitly when true.

These cells frequently have **no compute roofline** — the kernel does a few comparisons and one histogram increment per element and has no FLOP — so "both currencies" lands as logical-GB/s × trips versus the read wall plus a per-CTA limiting factor.

Land it as:

```bash
D=/home/compiler_gfx/tilelang/mcDeepGEMM/third-party/DeepSelect
git -C $D status --porcelain          # confirm only the intended files
git -C $D commit -a -F /path/to/msg   # or: git -C $D add <paths> && git -C $D commit -F ...
git -C $D push origin main
```

Never a bare `git commit -a` (see repo identity, top).

**Debugging mis-ranked output**: the fastest localizer is to write the kernel's threshold triple `{wide_threshold, above, fine_threshold, remain, num_staged, last_remain}` into `output[topk-8+tx]` on `blockIdx.x == 0 && tx < 8` and compare it against the same row computed with torch on CPU. That is how the `0xBFA` vs true `0xBFC`-class bug was found in one step. Clean it up afterwards — `rg RKPROBE` must be 0.

Symptom → first look (handover §8): wrong-but-not-much (a few slots, rank off by a few) → the threshold's subtraction convention (the `above + c > remain_topk` rule); `Memory Violation(0x4)` / `ATU Fault` → first the probe's missing `set_default_device`, then an out-of-range threshold leaving a shared variable uninitialized; one cell slow while others unchanged → occupancy, check `static + dynamic` against 32,768; source changed with no behavior change → a stale `.so`; compile-time `undeclared identifier` → constant/helper ordering in `radix_core.cuh`.

## Known holes (recorded, not hidden)

- `backend="deep_gemm"`'s host kernel collects the members of the threshold *coarse* bin (half-precision ordered key `>> 6`) before refining, and the chunked kernel silently drops members past its staging capacity. A row with more than 4096 values in one such bucket gets a top-k of an arbitrary subset, varying run to run. Filed as a strict `xfail` in the host repo: `deep_gemm/tests/test_indexer_topk_selector.py::test_selector_candidate_overflow`. **`maca_c` has no such hole.**
- The `radix_topk_row_bf16_k` static-k row used by the chunked path still runs the 8-bit coarse level and the 3,514-slot arena; it has not received coarse12.
- The fp32 row is a separate codebase path whose overflow handling is multi-round full-row rescan (up to 8 trips). Same "coarse level too coarse" disease, different cure — a 32-bit key cannot be resolved in two levels the way a 16-bit one can. Retesting fp32 cells is mandatory when touching it.

## Torch ABI / host compiler notes

`csrc/xcore1600/api.cpp` (host) is compiled by `g++` (GCC 11.4 here), which is why `std::format` is unavailable (libstdc++ has `<format>` only from GCC 13) and the one `TORCH_CHECK` message that needs formatting uses `snprintf` instead. Do not "fix" that back. Anything including `<cuda_runtime_api.h>` must not depend on cu-bridge's compatibility layer for `__nv_bfloat16`: `csrc/structs.h` includes `<maca_bfloat16.h>` so that `api.cpp` and every instantiation TU see the *same* `maca_bfloat16`, and `TopkSelectConfig<maca_bfloat16, ...>`'s template entity is one symbol on both sides.
