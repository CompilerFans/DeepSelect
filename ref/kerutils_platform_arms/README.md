# kerutils' platform arms — the MACA arm, compiled

`probe.cu` + `run_probe.sh`. Build and run:

```bash
./run_probe.sh                  # xcore1000
ARCH=xcore1600 ./run_probe.sh   # another family
```

## Why this exists

`csrc/3rdparty/kerutils/` selects exactly one platform arm — `KERUTILS_IS_BUILD_ON_CUDA`,
`KERUTILS_IS_BUILD_ON_ASCEND` or `KERUTILS_IS_BUILD_ON_MACA` — and **the MACA one
is the only arm nothing else in this repository compiles.** `kerutils.cuh`,
`device/device.cuh`, `host/host.h` and the three `device/<platform>/common.h`
files all reach it, but the sole consumer, `csrc/maca_kernels/xcore1600/`, is off
`setup.py`'s `SOURCES` and off `include_dirs`. So a full build of this tree runs
`./develop.sh`, passes every suite, and never instantiates the arm. "It builds"
is not evidence about it, and this probe is the compilation that is.

## What it checks

| | |
| --- | --- |
| the arm is selected, and only it | `#error` if `KERUTILS_IS_BUILD_ON_MACA` is missing, **or if CUDA or ASCEND fired alongside it** |
| what each compilation pass sees | both passes report their macros; the probe fails if `__MACA__` is absent in either, or if the two disagree |
| `ku::ceil_div` / `ku::ceil` in the host pass | constant-folded in two `static_assert`s |
| the same pair in the device pass | evaluated in the kernel — this is the half that needs `__host__ __device__`, since upstream's plain `inline constexpr` needs `--expt-relaxed-constexpr`, which mxcc does not guarantee |
| `ku::st_shared` (both overloads) | 16 bytes stored to shared and read back, per thread |
| `ku::canonical_warp_idx_sync` | the value is checked against `threadIdx.x / 32`, block-local |
| `ku::trap` | referenced, not reached — it aborts |
| `ku::launch_kernel` on a default config | launches and returns the right 256/256 slots |
| `ku::launch_kernel` with `cluster` / `use_pdl` / `cooperative` set | still launches (with three warnings on stderr), returns the same 256/256 slots |

## The two passes, measured

A `.cu` is preprocessed twice, and the two passes see different macro sets.
This is the table the arm selection rests on, reported by the probe on every
run (MACA 3.7.0, `--offload-arch=xcore1000`):

| macro | host pass | device pass | where it comes from |
| --- | --- | --- | --- |
| `__MACA__` | **Y** | **Y** | mxcc |
| `__MACACC__` | **Y** | **Y** | mxcc |
| `__CUDACC__` | **Y** | **Y** | cucc's `-imacros __macro_mxcc.h` |
| `__NVCC__` | **Y** | **Y** | same |
| `__MACA_ARCH__` | – | **Y = 1000** | mxcc, device pass only |
| `__CUDA_ARCH__` | – | **Y = 800** | cucc's `-Xdevice -D__CUDA_ARCH__=800` |

Three consequences, and the third is the one that bites:

1. **`__MACA__`, not `__MACA_ARCH__`, is the arm key.** The latter is a
   device-pass-only macro, so a `host/host.h` waiting on it would find nothing
   but its own `#error`.
2. **`__MACA__` must be tested before `__CUDACC__`.** The latter is defined and
   *false* here — it is the CUDA-dialect adapter — so testing it first makes
   both arms fire. That collision is real: it is what the probe reported on its
   first run.
3. **`__CUDA_ARCH__` is defined, as 800.** So `cuda/common.h`'s
   `KERUTILS_ENABLE_SM80/90/100` gates would *fire* on this platform if that
   header were included, re-enabling the `cuda/sm80/` PTX that does not
   assemble here. What keeps them out is `device/device.cuh`'s include list,
   not a macro test — which is why adding a `cuda/sm*` header to the MACA arm
   is not a port. And it is why **no MACA code may branch on `__CUDA_ARCH__`**:
   the family is `DEEP_SELECT_ARCH` (host, per artifact) or `__MACA_ARCH__`
   (device pass), never that value.

The toolkit reaches the same conclusion by a different route: `cute/config.hpp`
keys `CUTE_DEVICE` on `defined(__MACA_ARCH__) || defined(_NVHPC_CUDA) ||
defined(__clang__)`, using `__clang__` — which mxcc defines in both passes — as
the host-pass fallback for a device-pass-only macro.

The `canonical_warp_idx_sync` row is the one worth reading twice. MACA's wave is
**64 lanes**, so the platform's own wave index is `threadIdx.x / 64` and would
give `0,0,1,1` across this probe's 128-thread block; the ported kernels index
32-thread groups and need `0,1,2,3`. The check distinguishes them, so the row
fails if someone "corrects" the function to the platform's answer.

## What it does *not* check

**That `csrc/maca_kernels/xcore1600/` compiles.** It does not — that tree has its
own outstanding work, recorded under "Known holes" in `CLAUDE.md`. The subject
here is the vendored library's platform arm, which is a strictly smaller claim
than the port that would consume it. Do not read this probe's PASS as a
statement about the port.

## How it compiles

Through cu-bridge's `cucc`, which is the driver `torch` uses — this build has no
`bin/nvcc`, and **nothing calls `mxcc` directly**. `cucc` is a bash script that
resolves `MXCC_PATH` (`$MACA_CLANG_PATH/mxcc`, else `$MACA_PATH/mxgpu_llvm/bin/mxcc`),
appends the MACA include catalogue and the CUDA-dialect adapter
(`-imacros __macro_mxcc.h`), and then invokes mxcc. `cucc --version` reports
`mxcc version 1.0.0`; `BASH_XTRACE=1 ./run_probe.sh` prints the expansion.

Two consequences the probe depends on:

- **`<maca_bfloat16.h>` and `<cuda_runtime_api.h>` resolve only through cucc's
  catalogue.** They are at `$MACA_PATH/include/common/` and
  `$MACA_PATH/tools/cu-bridge/include/`, not at `$MACA_PATH/include/`.
- **cucc defines `__CUDACC__` and `__NVCC__`** for every MACA translation unit,
  via that same adapter header. This is why `common/common.h` tests `__MACA__`
  *before* `__CUDACC__`: they are the CUDA-dialect adapter, not a statement about
  the platform, and testing `__CUDACC__` first makes both arms fire. That
  collision is real — it is what this probe reported on its first run.

**The flags come out of `setup.py`.** `run_probe.sh` parses `nvcc_args` from
`setup.py` with `ast.literal_eval` rather than carrying a copy, so a flag added
to the build is a flag the probe compiles with on the next run. What it adds
beside them — `-offload-arch`, `-DDEEP_SELECT_ARCH` and four `-I` paths — it
prints, so the addition is visible rather than assumed. The failure this avoids
is the one `setup.py` itself records: a hand-copied flag catalogue drifting from
the compiler's.

Artifacts go to `/tmp/dsref_build/kerutils_platform_arms/` (`DSPROBE_OUT` to
move them); nothing is written beside the sources.
