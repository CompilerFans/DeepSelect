import os
import subprocess
import warnings
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

# torch's MACA build warns at import that `flash_attn` is not installed.  This
# package uses none of it -- the kernels are this repository's own -- so the
# warning is noise in every build log.  Matched by message, not by category:
# torch also warns there when `MACA_PATH` is unset or wrong, and that one is
# worth reading.
warnings.filterwarnings("ignore", message=".*flash_attn.*")

exec(open("deep_select/__version__.py").read())

# One source, one extension.  `csrc/xcore1600/` is the ported upstream kernel:
# it stays in the repo as source and is off the build entirely -- see
# `build_for_maca`.
SOURCES = [
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
# device compiler as `$CUDA_HOME/bin/nvcc`, falling back to `bin/cucc`.  `cucc`
# is the whole CUDA-dialect adapter (the macro header turning `__MACACC__` into
# `__CUDACC__`, `-gencode` -> `-D__CUDA_ARCH__=`, `-lcudart` -> `-lmcruntime`,
# the MACA include catalogue) and forwards the rest to mxcc unchanged, which is
# how the mxcc-dialect flags in `compile_args` below reach the compiler.
#
# A synthesized `CUDA_HOME` must also provide `bin/gnu`, which torch asks it for
# unconditionally (`get_wcuda_gnu_path()`); without it every build dies with
# `no cu-bridge gnu found`.


def build_for_maca():
    """Build the one extension: `SOURCES` compiled for every target.

    **This is a csrc compile and nothing else.**  One `.cu`, one `.so`;
    everything under `deep_select/` is Python and reaches the artifact through
    `tvm_ffi.load_module`.

    `CUCC_TARGETS` becomes one comma-separated `-offload-arch`, which mxcc takes
    as a set of images of the same source in one file.  Unset means
    `_arch.DEFAULT_TARGETS`, one target per family, so a build host needs no
    MACA card to produce a shippable wheel; `native` is the one-image shortcut.
    An unrecognized target fails the build by name.

    Nothing is specialized per architecture at compile time, and nothing can
    be: a family macro would be a lie in two of the three images.  The two
    numbers the kernel sizes its grids against travel as arguments instead.

    `csrc/xcore1600/` is not built: its source is off `SOURCES` and its
    `kerutils` include is off `include_dirs` below.  Re-adding both is what
    building it would take.

    Every source is a `.cu`, so the device compiler is the only compiler this
    build invokes (torch would route a `.cpp` to `$cxx`, meaning a second flag
    list and a second kerutils mode macro).  `CUDA_HOME` is deliberately left
    alone -- see the cu-bridge note above.
    """
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    from deep_select._arch import resolve_targets

    maca_root = _maca_root()
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # The headers and `libtvm_ffi.so` ship inside the `tvm_ffi` python package,
    # so asking that package beats guessing a prefix.  `lib_subdir` is
    # platform-dependent and resolved, because the rpath below points at it.
    tvm_ffi_root, lib_subdir = _tvm_ffi_root()
    targets = resolve_targets(os.environ.get("CUCC_TARGETS"))

    # The compiler pair, printed rather than assumed -- see the cu-bridge note
    # above for why `CUDA_HOME` must be the one torch resolved.
    print(f"deep_select: device compiler is "
          f"{os.path.join(cpp_extension.CUDA_HOME, 'bin', 'cucc')}, "
          f"host compiler is {cpp_extension.get_cxx_compiler()}")

    # Three paths, and the MACA catalogue is not one of them: cucc's own
    # `all/adder` (`$MACA_PATH/tools/cu-bridge/bin/conf.json`) already passes
    # `-I` for `cu-bridge/include`, `include/soft-link` and every `include/mc*`
    # library, plus `-imacros __macro_mxcc.h`.  What it does *not* add is the
    # toolkit's own `include/`, which is where `maca_bfloat16.h` and `cub/` are.
    include_dirs = [
        os.path.join(this_dir, "csrc"),
        # `csrc/ffi/` -- the tvm-ffi edge (tensor/error/check helpers).  Both
        # kernel trees include it as `"../ffi/..."`, `dispatch_utils.h` as
        # `"ffi_..."`, so the directory itself is on the path.
        os.path.join(this_dir, "csrc", "ffi"),
        os.path.join(maca_root, "include"),
        # `csrc/xcore1600/` is NOT on this list and neither is
        # `csrc/3rdparty/kerutils/include`, which only its kernels include
        # (`csrc/ffi/` names kerutils once, in a comment, and includes nothing
        # from it).  Both trees stay in the repo as source; neither is compiled.
        # Re-adding these two paths is the first half of building it again.
    ]

    # In mxcc's dialect.  cucc forwards what it does not recognize, so these
    # reach mxcc unchanged.
    nvcc_args = [
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
        # One comma-separated list: mxcc compiles each architecture into its
        # own image of the same source, in one extension.
        f"-offload-arch={','.join(targets)}",
    ]
    # No `-I` here: `include_dirs=` above/below is what carries them, and torch
    # already turns that into `-I` on the device pass.  Spelling them in both
    # places put every one of them on the command line twice.

    def strip_torch_libs(ext):
        # `CUDAExtension`'s constructor auto-appends c10/torch/torch_cuda, and
        # the artifact must carry no torch DT_NEEDED entry -- the point of the
        # tvm-ffi migration.
        ext.libraries = [
            lib for lib in ext.libraries
            if lib.lower() not in ("c10", "torch", "torch_cpu", "torch_python",
                                   "c10_cuda", "torch_cuda")
        ]
        return ext

    ext_modules = [
        strip_torch_libs(CUDAExtension(
                # The artifact has no `PyInit` and is NOT an importable python
                # module -- `_binding.py` loads it through `tvm_ffi.load_module`.
                # So the cpython tag comes off both names: the `.so` filename
                # through `no_python_abi_suffix` on the **build command** (not
                # on this Extension -- torch reads that flag out of the
                # command's kwargs, so an attribute here does nothing) and the
                # wheel through `_BdistWheel` at the bottom.  The floor is then
                # `python_requires` in `setup()`.
                #
                # Named for the *package*, not for an architecture: there is
                # one of these and it serves every family it carries an image
                # for.  `deep_select_maca` matches the C++ namespace.
                name="deep_select.deep_select_maca",
                sources=SOURCES,
                # Every source is a `.cu`, so the `nvcc` list is the only one
                # torch reads, and its contents are mxcc's dialect because
                # `$nvcc` resolves to cucc.
                extra_compile_args={"nvcc": nvcc_args},
                # The MACA catalogue, the tvm-ffi headers the binding edge
                # needs, and torch's own paths.
                include_dirs=include_dirs + [os.path.join(tvm_ffi_root, "include")],
                library_dirs=[os.path.join(tvm_ffi_root, lib_subdir)],
                libraries=["tvm_ffi"],
                extra_link_args=[
                    # No rpath for `libtvm_ffi.so`, deliberately: it ships
                    # inside the `tvm_ffi` *python package*, so any rpath would
                    # bake this build machine's `site-packages` into the wheel.
                    # The soname is satisfied at run time by `import tvm_ffi`,
                    # which `_binding.load` does.
                    f"-L{maca_root}/lib",
                    f"-Wl,-rpath,{maca_root}/lib",
                    # `DT_RUNPATH`, not `DT_RPATH`: the two differ in *order*
                    # -- RPATH is searched before `LD_LIBRARY_PATH` and RUNPATH
                    # after it -- and that order is what decides whether a
                    # machine with a different toolkit at the same path can
                    # override this one.  With RPATH it cannot, and the failure
                    # is the `mcErrorInvalidDeviceFunction` whose cause looks
                    # like a kernel defect.  cucc emits RUNPATH on its own;
                    # this link does not, so ask for it explicitly.
                    "-Wl,--enable-new-dtags",
                ],
            ))
    ]
    print(f"deep_select: compiling {', '.join(SOURCES)} for {','.join(targets)}")

    # `no_python_abi_suffix` belongs *here*, not on the Extension above: torch's
    # `BuildExtension.__init__` reads it out of its own kwargs
    # (`cpp_extension.py`), and `with_options` is what puts it there.  Set it on
    # the Extension object and nothing reads it -- the `.so` keeps
    # `cpython-310-x86_64-linux-gnu` in its name.
    return (ext_modules,
            BuildExtension.with_options(use_ninja=True, no_python_abi_suffix=True))


# --- stack-frame check ------------------------------------------------------
#
# Reads mxcc's `--resource-usage` over the sources, through
# `tests.kernelkit.build.check_maca_stack_frame`; a device function above the
# baseline is one that spills.
#
# Opt-in rather than wired into `build_ext`, because the flag recompiles the TU
# it is asked about (~60 s for `maca_topk.cu`).  Over a named subset of sources:
#
#     DEEP_SELECT_MACA_STACK_CHECK=1 ./build.sh
#     DEEP_SELECT_MACA_STACK_CHECK=csrc/xcore1000/maca_topk.cu ./build.sh
#     DEEP_SELECT_MACA_STACK_BASELINE=64 DEEP_SELECT_MACA_STACK_CHECK=1 ./build.sh
#
# The baseline is bytes and is per toolchain: 48 is what this one reports for a
# device with no spill, so it is the value that separates a floor from a leak.
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

# --- wheel tag --------------------------------------------------------------
#
# `bdist_wheel` derives the tag from the *build* interpreter, which would stamp
# `cpython-310-cp310-linux_x86_64` on a wheel whose extension has no `PyInit`.
# `py3-none-<plat>` is the honest one: what is version-sensitive here is the
# *platform*, and what is not is the interpreter.
#
# The floor that remains real is `python_requires` below, which is about the
# runtime prerequisites rather than the extension: torch requires >= 3.9 and
# `apache-tvm-ffi` >= 3.8, so this package's floor is the higher of the two.
try:
    # Canonical since setuptools 70.1; the `wheel` package's copy still works
    # but prints a FutureWarning on every command, including `--version`.
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel_base
except ImportError:
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel_base


class _BdistWheel(_bdist_wheel_base):
    def get_tag(self):
        _, _, plat = super().get_tag()
        # Keep the platform `super()` resolved -- `root_is_pure` is False
        # because this distribution has ext_modules, so it is the real one.
        return "py3", "none", plat


setup(
    name="deep_select",
    version=f"{__version__}+{git_rev}.{datetime_rev}",
    packages=find_packages(include=["deep_select"]),
    ext_modules=ext_modules,
    cmdclass={
        "build_ext": build_ext,
        "bdist_wheel": _BdistWheel,
    },
    # Both are imported at run time by `interface.py` / `_binding.py`, and
    # neither is vendored here; torch is also the build's own prerequisite.
    # Unpinned on purpose: the MACA builds of torch are metax-suffixed local
    # versions, and a pin would refuse them.
    install_requires=["torch", "apache-tvm-ffi"],
    python_requires=">=3.9",
    zip_safe=False,
)
