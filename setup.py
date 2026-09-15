import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

exec(open("deep_select/__version__.py").read())

XCORE1000_SOURCES = [
    "csrc/xcore1000/maca_topk.cu",
]


def _tvm_ffi_root():
    """Where `tvm_ffi` keeps its headers and shared library.

    Returns `(package_root, lib_subdir)`.  Needed at build time and at run
    time: the extension carries no PyInit, so `_binding.py` loads it through
    `tvm_ffi.load_module`.
    """
    import tvm_ffi

    root = os.path.dirname(os.path.abspath(tvm_ffi.__file__))
    for sub in ("lib", os.path.join("lib64")):
        if os.path.isdir(os.path.join(root, sub)):
            return root, sub
    raise RuntimeError(
        f"tvm_ffi at {root} has no lib/ or lib64/; the extension links "
        f"libtvm_ffi.so and cannot be built without it"
    )


def _maca_root() -> str:
    return os.environ.get("MACA_HOME") or os.environ.get("MACA_PATH") or "/opt/maca"


# The device compiler is cu-bridge's `cucc`, and this file does not reimplement
# it.  torch's MACA build is written *against* cu-bridge: `_find_cuda_home`
# lands on `${MACA_PATH}/tools/cu-bridge`, and `_join_cuda_home` drives the
# device compiler as `$CUDA_HOME/bin/nvcc`, falling back to `bin/cucc` -- this
# install's case.  `cucc` is the whole CUDA-dialect adapter (the macro header
# turning `__MACACC__` into `__CUDACC__`, `-gencode` -> `-D__CUDA_ARCH__=`,
# `-lcudart` -> `-lmcruntime`, the MACA include catalogue) and forwards the rest
# to mxcc unchanged, which is how the mxcc-dialect flags in `compile_args` below
# reach the compiler.
#
# Do not write a shim over mxcc and point `CUDA_HOME` at it: that home must also
# provide `bin/gnu`, which torch asks it for unconditionally
# (`get_wcuda_gnu_path`).  An earlier shim dropped it; every build died with
# `no cu-bridge gnu found`.


def build_for_maca():
    """Build one extension per architecture, each holding that architecture's
    kernel.

    **There is one source tree and it is `csrc/xcore1000/`.**  That tree is the
    hand-written MACA kernel, and it is what a C600 and a C600U run as well:
    measured on a C600U it is 1.5-2.9x faster than the ported kernels under
    `csrc/xcore1600/` and passes `check_result` on every cell the port fails
    (CLAUDE.md, "Can a C600U run the C500 kernel").  **The port is not built at
    all** -- not by a switch, not by an env var, not on one architecture: its
    source and its `kerutils` dependency are off the include paths above, so
    nothing in this file can reach it.  Bringing it back is a source change to
    `sources` AND to `include_dirs` together, which is the point: a switch that
    could put the broken kernel back is a switch that can be left on.

    Each architecture still gets its own extension, `deep_select_xcore<N>`,
    because a config's shared memory footprint is only valid for the
    architecture it was sized for, and that name is exactly how
    `topk(backend="maca_c")` resolves one.  Which get built comes from
    `CUCC_TARGETS`, same variable and meaning as the host repository's
    `build.sh`; unset means `_arch.DEFAULT_TARGETS`, one target per family, so a
    build host needs no MACA card to produce a shippable wheel.  `-offload-arch`
    is passed per extension, so each targets exactly one architecture.

    `CUDA_HOME` is deliberately left alone -- see the cu-bridge note above.
    Every source is a `.cu`, so the device compiler is the only compiler this
    build invokes (torch would route a `.cpp` to `$cxx`, meaning a second flag
    list and a second kerutils mode macro).
    """
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    from deep_select._arch import (CAPACITY_BYTES, family_of_target,
                                   resolve_targets)

    maca_root = _maca_root()
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # The headers and `libtvm_ffi.so` ship inside the `tvm_ffi` python package,
    # so asking that package beats guessing a prefix.  `lib_subdir` is
    # platform-dependent and resolved, because the rpath below points at it.
    tvm_ffi_root, lib_subdir = _tvm_ffi_root()
    targets = resolve_targets(os.environ.get("CUCC_TARGETS"))

    # torch's MACA build appends a second `--offload-arch` of its own when this
    # is set, which would put two in one extension -- a config sized for 128 KiB
    # silently riding into a 64 KiB build.  Refuse rather than build something
    # wrong on one of the two architectures it claims.
    if os.environ.get("TORCH_EXTENSION_ENABLE_XC1500_COMPILE"):
        raise RuntimeError(
            "TORCH_EXTENSION_ENABLE_XC1500_COMPILE makes torch append "
            "--offload-arch=xcore1000/xcore1500 to every source; setup.py "
            "targets one architecture per extension.  Unset it and pass the "
            "architectures to build via CUCC_TARGETS instead."
        )

    # The compiler pair, printed rather than assumed -- see the cu-bridge note
    # above for why `CUDA_HOME` must be the one torch resolved.
    print(f"deep_select: device compiler is "
          f"{os.path.join(cpp_extension.CUDA_HOME, 'bin', 'cucc')}, "
          f"host compiler is {cpp_extension.get_cxx_compiler()}")

    # cucc appends this catalogue to every invocation, so it is part of the
    # flags the kernels were developed against even though torch's ninja file
    # never mentions it.  `soft-link/cutlass` is load-bearing:
    # `cutlass/kernel_launch.h` exists only there, as a symlink to `mctlass/`.
    maca_library_includes = ["mcr", "mcblas", "mcfft", "mcsolver", "mcdnn",
                             "common", "mcsparse", "mcrand", "mckl", "mcsml",
                             "mctx", "thrust/detail"]

    include_dirs = [
        os.path.join(this_dir, "csrc"),
        # `csrc/ffi/` -- the tvm-ffi edge (tensor/error/check helpers).  Both
        # kernel trees include it as `"../ffi/..."`, `dispatch_utils.h` as
        # `"ffi_..."`, so the directory itself is on the path.
        os.path.join(this_dir, "csrc", "ffi"),
        # `csrc/xcore1600/` is NOT on this list and neither is
        # `csrc/3rdparty/kerutils/include`, which only its kernels include
        # (`csrc/ffi/` names kerutils once, in a comment, and includes nothing
        # from it).  Both trees stay in the repo as source; neither is compiled.
        # Re-adding these two paths is the first half of building the port
        # again -- see CLAUDE.md, "the port is not built".
        os.path.join(maca_root, "include"),
        os.path.join(maca_root, "tools", "cu-bridge", "include"),
        os.path.join(maca_root, "tools", "cu-bridge", "include", "soft-link"),
    ] + [os.path.join(maca_root, "include", d) for d in maca_library_includes]

    def compile_args(target):
        # In mxcc's dialect.  cucc forwards what it does not recognize, so
        # these reach mxcc unchanged.
        args = [
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            # torch injects `-fPIC` on the `cxx` side and
            # `--compiler-options '-fPIC'` on the device side, which cucc
            # rewrites to `-Xcompiler -fPIC`; named here anyway so the device
            # pass does not rely on that rewrite -- one duplicate flag.
            "-fPIC",
            "-use-fast-math",
            # `-use-fast-math` implies FTZ, turned back off here so the one
            # float conversion in the kernel -- `__float2bfloat16` of
            # `value_oob_fill_value` -- stays exact for a denormal fill.  The
            # ranking path is pure integer key manipulation and cannot care.
            "-Xclang", "-fdenormal-fp-math-f32=ieee",
            f"-offload-arch={target}",
            # `__MACA_ARCH__` is only defined in the device pass, but the
            # capacity constant lives in `structs.h`, which every TU includes,
            # so it is spelled out on the command line -- the same number the
            # toolchain derives, set from the same loop iteration.
            f"-DDEEP_SELECT_NATIVE_ARCH={family_of_target(target)}",
        ]
        return args + [f"-I{d}" for d in include_dirs]

    ext_modules = []
    for target in targets:
        family = family_of_target(target)
        capacity_kib = CAPACITY_BYTES[family] // 1024
        # One source tree for every family, no branch -- see the docstring.
        sources = XCORE1000_SOURCES
        ext = CUDAExtension(
                # The tvm-ffi artifact has no `PyInit` and is NOT an importable
                # python module -- `_binding.py` loads it through
                # `tvm_ffi.load_module`.  `no_python_abi_suffix` keeps a
                # cpython-310 tag off something the import system never sees.
                name=f"deep_select.deep_select_xcore{family}",
                no_python_abi_suffix=True,
                sources=sources,
                # Every source is a `.cu`, so the `nvcc` list is the only one
                # torch reads, and its contents are mxcc's dialect because
                # `$nvcc` resolves to cucc.
                extra_compile_args={
                    "nvcc": compile_args(target),
                },
                # The MACA catalogue, the tvm-ffi headers the binding edge
                # needs, and torch's own paths.
                include_dirs=include_dirs + [os.path.join(tvm_ffi_root, "include")],
                library_dirs=[os.path.join(tvm_ffi_root, "lib")],
                libraries=["tvm_ffi"],
                extra_link_args=[
                    f"-L{maca_root}/lib",
                    f"-Wl,-rpath,{maca_root}/lib",
                    # `libtvm_ffi.so` lives in the tvm_ffi python package, so
                    # the loader resolves it through that package, not
                    # LD_LIBRARY_PATH.
                    f"-Wl,-rpath,{os.path.join(tvm_ffi_root, lib_subdir)}",
                ],
            )
        # `CUDAExtension`'s constructor auto-appends c10/torch/torch_cuda to
        # `libraries`; strip them so the artifact carries no torch DT_NEEDED
        # entry -- the point of the migration.
        ext.libraries = [
            lib for lib in ext.libraries
            if lib.lower() not in ("c10", "torch", "torch_cpu", "torch_python",
                                   "c10_cuda", "torch_cuda")
        ]
        ext_modules.append(ext)
        print(f"deep_select: building xcore{family} "
              f"({capacity_kib} KiB shared memory) for {target}: "
              f"csrc/xcore1000 (maca_topk.cu)")

    return (ext_modules, BuildExtension.with_options(use_ninja=True))


# --- stack-frame check (the MACA counterpart of upstream's spill gate) ------
#
# Upstream scans every extension with `cuobjdump -res-usage` and fails on a
# kernel that spills; there is no `cuobjdump` on MACA, so mxcc's
# `--resource-usage` stands in, through
# `tests.kernelkit.build.check_maca_stack_frame`.
#
# Opt-in rather than wired into `build_ext`, because the flag recompiles the TU
# it is asked about (~60 s for `maca_topk.cu`).  Over a named subset of sources:
#
#     DEEP_SELECT_MACA_STACK_CHECK=1 ./build.sh
#     DEEP_SELECT_MACA_STACK_CHECK=csrc/xcore1000/maca_topk.cu ./build.sh
#     DEEP_SELECT_MACA_STACK_BASELINE=64 DEEP_SELECT_MACA_STACK_CHECK=1 ./build.sh
#
# The baseline is bytes, per toolchain: on this one, 50 of `maca_topk.cu`'s 78
# devices floor at 48, so 48 is the only value separating the toolchain's floor
# from a real spill (upstream's 8 predates a nonzero floor).
DEFAULT_MACA_STACK_BASELINE = 48


def _maca_stack_check(ext_modules, maca_root):
    """Scan the sources behind a built extension, if the caller asked for it.

    The sources come from the extensions themselves, so the check follows what
    was actually built rather than a list repeated here.
    """
    want = os.environ.get("DEEP_SELECT_MACA_STACK_CHECK")
    if not want or want.lower() in ("0", "no", "false"):
        return

    import sys as _sys

    _sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from tests.kernelkit.build import check_maca_stack_frame

    selected = [] if want.lower() in ("1", "yes", "true") \
        else [s.strip() for s in want.split(",")]
    baseline = int(os.environ.get("DEEP_SELECT_MACA_STACK_BASELINE",
                                  DEFAULT_MACA_STACK_BASELINE))
    mxcc = os.path.join(maca_root, "mxgpu_llvm", "bin", "mxcc")

    root = os.getcwd()
    bad = 0
    for ext in ext_modules:
        # The flags the build already resolved, so the check compiles the way
        # the build did rather than from a second copy of the list.  Torch's own
        # include paths are *not* in there, so the checker adds them itself.
        args = list(ext.extra_compile_args["nvcc"])
        for src in ext.sources:
            if selected:
                rel = os.path.relpath(src, root).replace(os.sep, "/")
                if not any(rel.endswith(s) or s.endswith(rel) for s in selected):
                    continue
            hits = check_maca_stack_frame(src, baseline, mxcc, args, quiet=False)
            bad += len(hits)
    if bad:
        raise RuntimeError(
            f"{bad} MACA kernel(s) exceed the stack baseline of {baseline} B. "
            f"Spilling degrades these kernels; raise "
            f"DEEP_SELECT_MACA_STACK_BASELINE only if the reading is expected, "
            f"and unset DEEP_SELECT_MACA_STACK_CHECK to skip the check.")


try:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    git_rev = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").rstrip()
except Exception:
    # e.g. a source tarball with no `.git`: keep the version a valid PEP 440
    # local version.
    git_rev = "unknown"

datetime_rev = datetime.now().strftime("%Y%m%d.%H%M%S")

ext_modules, build_ext = build_for_maca()

# Before the build, not after: `--resource-usage` compiles the source itself.
# Sources are listed relative to this file, which is the root to resolve them
# against.
_maca_stack_check(ext_modules, _maca_root())

setup(
    name="deep_select",
    version=f"{__version__}+{git_rev}.{datetime_rev}",
    packages=find_packages(include=["deep_select"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
