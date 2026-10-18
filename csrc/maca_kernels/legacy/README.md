# `legacy/` -- the ported upstream kernels, kept as source, never built

This tree is the MACA port of upstream's `cuda_kernels` (`v3` for bf16,
`v3_fp32` for fp32, `api.cu` the host dispatch), held out of the build.  It is
here as provenance and as the audit target for the 64-lane-wave question -- it
is the CUDA-era texture CLAUDE.md's "MACA warp intrinsics" section warns
against copying.

**Why it is not what ships.**  Measured on a C600U (CLAUDE.md, "Known holes"
and "Can a C600U run the C500 kernel?"): it selects wrong (a rescan reachable
only on a 128 KiB part mis-handles a tied run at the threshold), and where it
runs at all it is 1.5-2.9x slower than the kernel every family actually
compiles.  What ships is the shared implementation in
`csrc/maca_kernels/common/` plus each family's own tuple set in
`csrc/maca_kernels/xcore<N>/instantiations/`.

**Why it would not compile today.**  Two `static_assert`s --
`v3/topk_select.cuh` and `v3_fp32/topk_select.cuh`, one each -- still name
`NATIVE_SHARED_MEMORY_PER_SM_BYTES`, which is defined nowhere since the
per-family constants became `csrc/structs.h`'s `ARCH_SMEM_PER_AP_BYTES`.  The
rename is part of re-adding the tree, not something a reader can assume done.

**What re-adding it takes** (deliberately not an environment variable -- a
switch that could put this kernel back is a switch that can be left on):

1. `setup.py`: point a family's source list at this tree and put
   `csrc/3rdparty/kerutils/include` back on `include_dirs`.
2. The `NATIVE_SHARED_MEMORY_PER_SM_BYTES` → `ARCH_SMEM_PER_AP_BYTES`
   substitution above.
3. The measured C600U defect fixed and re-validated -- a pass is necessary but
   not sufficient; the routing change also needs the perf measurement.

**Regenerating its instantiations** (the 52 files under
`v3/instantiations/` and `v3_fp32/instantiations/`):

```bash
python3 scripts/generate_instantiations.py csrc/maca_kernels/legacy/v3/instantiations
python3 scripts/generate_instantiations.py csrc/maca_kernels/legacy/v3_fp32/instantiations
```

The generator's `TopkSelectConfigs` tables and its `check_fits_maca` capacity
arithmetic exist for this tree; the build's own instantiation tables are a
different mechanism (the launcher unit, see the generator's docstring).
