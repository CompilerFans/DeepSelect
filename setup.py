import os
import subprocess
from datetime import datetime
from pathlib import Path

from setuptools import setup, find_packages

exec(open("deep_select/__version__.py").read())

# MACA port: `csrc/maca_topk.cu` is the shipping backend and the only
# translation unit in the extension.
#
# The upstream `csrc/cuda_kernels/v3` (bf16) and `v3_fp32` trees have been
# ported to MACA (TMA -> cooperative `ldg`, mbarrier -> single buffer +
# `__syncthreads`, inline PTX -> MACA builtins) and their explicit
# instantiations compile for xcore1000, but they are NOT wired into this
# extension yet -- the host-side dispatch (`csrc/api.cpp`) still binds through
# pybind11 + libtorch, which the MACA build here does not link.  They are kept
# as the reference implementation of the upstream algorithm.
# `csrc/cuda_kernels/v3_cluster` is deleted: MACA has no cluster launch.
CUDA_SOURCES = [
    "csrc/maca_topk.cu",
]


def _maca_root() -> str:
    return os.environ.get("MACA_HOME") or os.environ.get("MACA_PATH") or "/opt/maca"


def build_for_maca():
    """Build the extension with the MACA toolchain.

    Device code is compiled by `mxcc` (MACA's nvcc equivalent), which torch's
    CUDAExtension already selects because `CUDA_HOME` resolves to
    `$MACA_PATH/tools/cu-bridge` in this environment.  The CUDA-specific
    `-gencode arch=compute_100a,code=sm_100a` flags have no meaning on MACA
    and are replaced by `--offload-arch=xcore1000` (C500 / xcore1000), the
    same target the host repo's `CUCC_TARGETS=native` resolves to.
    """
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CUDA_HOME

    assert CUDA_HOME is not None, "PyTorch must provide a CUDA/MACA toolchain"

    maca_root = _maca_root()
    this_dir = os.path.dirname(os.path.abspath(__file__))

    include_dirs = [
        os.path.join(this_dir, "csrc"),
        os.path.join(maca_root, "include"),
        os.path.join(maca_root, "tools", "cu-bridge", "include"),
    ]

    extra_compile_args = {
        "cxx": [
            "-O3",
            "-std=c++17",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
        ],
        "nvcc": [
            "-O3",
            "-std=c++20",
            "-DNDEBUG",
            "-Wno-deprecated-declarations",
            "--use_fast_math",
            # `--use_fast_math` implies FTZ, so it is turned back off here.
            # The ranking path cannot care either way: it is pure integer
            # manipulation of the key (`__float_as_uint` plus integer ops),
            # and FTZ only governs floating-point arithmetic and conversion
            # results.  Keeping it off keeps the one float conversion in the
            # kernel -- `__float2bfloat16` of `value_oob_fill_value` -- exact
            # for a denormal fill instead of flushing it to zero.
            "--ftz=false",
            "--offload-arch=xcore1000",
        ] + [f"-I{d}" for d in include_dirs],
    }

    ext_modules = [
        CUDAExtension(
            name="deep_select.deep_select_cuda",
            sources=CUDA_SOURCES,
            include_dirs=include_dirs,
            extra_compile_args=extra_compile_args,
            extra_link_args=[
                f"-L{maca_root}/lib",
                f"-Wl,-rpath,{maca_root}/lib",
            ],
        )
    ]

    return (ext_modules, BuildExtension.with_options(use_ninja=True))


try:
    cmd = ["git", "rev-parse", "--short", "HEAD"]
    git_rev = subprocess.check_output(cmd, stderr=subprocess.DEVNULL).decode("ascii").rstrip()
except Exception:
    # e.g. building from a source tarball that has no `.git`. Keep the version
    # a valid PEP 440 local version in that case.
    git_rev = "unknown"

datetime_rev = datetime.now().strftime("%Y%m%d.%H%M%S")

ext_modules, build_ext = build_for_maca()

setup(
    name="deep_select",
    version=f"{__version__}+{git_rev}.{datetime_rev}",
    packages=find_packages(include=["deep_select"]),
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
