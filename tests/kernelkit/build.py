import os
import subprocess
import re
import sys
from typing import List, Optional

from .platform import Platform, requires_platform, get_current_platform

if get_current_platform() == Platform.CUDA:
    from torch.utils.cpp_extension import BuildExtension
else:
    # Don't import `torch.utils.cpp_extension` since it prints "No CUDA runtime is found, using CUDA_HOME='/usr/local/cuda'", which is annoying
    class BuildExtension:
        pass

@requires_platform([Platform.CUDA, Platform.CPU_ONLY])
def check_kernel_reg_spill_in_artifact(artifact_path: str, stack_baseline: int = 0, quiet: bool = False, suppress_checking_env_var: Optional[str] = None) -> List[str]:
    """
    Check the compiled artifact (e.g. a .cubin or .so file) for register spilling.

    Returns a list of kernel names that spill (local memory > 0, or stack > stack_baseline).

    If `quiet` is True, no output is printed.
    If `suppress_checking_env_var` is not None, the check is skipped entirely when
    the environment variable named by `suppress_checking_env_var` is set to "1", "yes", or "true".
    """
    from torch.utils.cpp_extension import CUDA_HOME

    if suppress_checking_env_var is not None and os.environ.get(suppress_checking_env_var, '0').lower() in ['1', 'yes', 'true']:
        return []
    
    if not quiet:
        print(f"Checking register spills in: {artifact_path}")

    cuda_home = CUDA_HOME if CUDA_HOME is not None else '/usr/local/cuda'
    cuobjdump_path = os.path.join(cuda_home, 'bin/cuobjdump')
    if not os.path.exists(cuobjdump_path):
        raise FileNotFoundError(f"cuobjdump not found (looked at {cuobjdump_path})")
    
    try:
        result = subprocess.run(
            [cuobjdump_path, "-res-usage", artifact_path],
            capture_output=True, text=True, timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("cuobjdump timed out during spill checking")
        raise RuntimeError()

    if result.returncode != 0:
        print(f"cuobjdump failed:\n{result.stdout}\n{result.stderr}")
        raise RuntimeError()

    def _parse_cuobjdump(output: str) -> list[tuple[str, int, int, int]]:
        """Parse cuobjdump output, returning [(name, reg, stack, local)] for kernels that spill."""
        spills = []
        current_name = None
        func_re = re.compile(r"^\s*Function\s+(\S+)")
        resource_re = re.compile(
            r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:\d+\s+LOCAL:(\d+)"
        )

        for line in output.splitlines():
            m = func_re.match(line)
            if m:
                current_name = m.group(1)
                continue
            m = resource_re.search(line)
            if m and current_name:
                reg = int(m.group(1))
                stack = int(m.group(2))
                local = int(m.group(3))
                if stack > stack_baseline or local > 0:
                    spills.append((current_name[:-1], reg, stack, local))
                current_name = None

        return spills

    spills = _parse_cuobjdump(result.stdout)
    if not spills:
        if not quiet:
            print("No register spills detected.")
        return []

    if not quiet:
        print(f"Found {len(spills)} kernel(s) with register spilling:\n")
        print(f"{'REG':>6}  {'STACK':>6}  {'LOCAL':>6}  Kernel")
        print("-" * 60)
        for name, reg, stack, local in spills:
            print(f"{reg:>6}  {stack:>6}  {local:>6}  {name}")

        print("\nRegister spilling can significantly degrade kernel performance.")
        print("This is often caused by differences in the compiler version or toolchain.")
        print("If you see this message, you may:")
        print("  - Investigate the cause of the spill")
        if suppress_checking_env_var is not None:
            print(f"  - Or, suppress this check by setting {suppress_checking_env_var}=1")

    return [s[0] for s in spills]

class SpillCheckBuildExtension(BuildExtension):
    @requires_platform(Platform.CUDA)
    def __init__(self, stack_baseline: int = 0, suppress_checking_env_var: Optional[str] = None):
        self.stack_baseline = stack_baseline
        self.suppress_checking_env_var = suppress_checking_env_var
        super().__init__()
    
    def run(self):
        super().run()
        for ext in self.extensions:
            so_path = self.get_ext_fullpath(ext.name)

            spilled_kernels = check_kernel_reg_spill_in_artifact(so_path, self.stack_baseline, False, self.suppress_checking_env_var)
            if len(spilled_kernels) > 0:
                print('Register spilling detected. Build failed!')
                sys.exit(1)
        print('Register spill check passed.')


# --- MACA ------------------------------------------------------------------
#
# [MACA] Upstream's gate above reads `cuobjdump -res-usage` out of
# `$CUDA_HOME/bin`, which does not exist on a MACA install, so
# `SpillCheckBuildExtension` has no MACA counterpart.  mxcc reports the same
# triple itself under `--resource-usage`, one `maca info:` block per kernel:
#
#     Function properties for  <mangled name>
#       <N> bytes stack frame
#     Used  <N> MTregisters, <N> STregisters, <N> bytes shared mem
#     staticMaxWarps/PEU : <N>
#
# so the check transposes rather than disappearing.  Two facts about the
# substitution, both measured on C500 with this tree's own flags:
#
#   - The `local > 0` half of upstream's test has no reported counterpart.  A
#     kernel deliberately made to push a 128-element double array through
#     `#pragma unroll 1` reports a 1032-byte stack frame and *the same 10
#     MTregisters* as a trivial kernel -- whatever would be local memory is
#     counted in the stack frame here, not separately.  So the stack-frame
#     reading is the whole signal, and `--stack-baseline` is the whole knob:
#     upstream's `stack_baseline=8` is not portable, since even a kernel with
#     no spill lands at 48 bytes on this toolchain (all 78 devices in
#     `maca_topk.cu` report either 0 or 48).
#   - `--resource-usage` is a second full device compile of the TU it is asked
#     about: ~60 s for `maca_topk.cu` (one TU, 78 devices), against tens of
#     minutes for a single xcore1600 instantiation.  That is why this is a
#     script rather than a `build_ext` subclass, and why it is opt-in per
#     source file rather than run over the build.

_RESOURCE_USAGE_FUNC = re.compile(r"Function properties for\s+(\S+)")
_RESOURCE_USAGE_FRAME = re.compile(r"(\d+)\s+bytes\s+stack\s+frame")
_RESOURCE_USAGE_REGS = re.compile(
    r"Used\s+(\d+)\s+MTregisters?,\s+(\d+)\s+STregisters?,\s+(\d+)\s+bytes\s+shared\s+mem")


def check_maca_stack_frame(source_path: str, stack_baseline: int,
                           mxcc: Optional[str] = None,
                           compile_args: Optional[List[str]] = None,
                           quiet: bool = False) -> List[tuple]:
    """Kernels in one MACA translation unit whose stack frame exceeds a baseline.

    Returns `[(mangled_name, stack_bytes, mtregisters, shared_bytes)]`, and is
    read exactly as `check_kernel_reg_spill_in_artifact`'s result is: empty
    means nothing spills.  `compile_args` is the device-side argument list the
    build uses -- `setup.py`'s `compile_args(target)` -- and `mxcc` the
    compiler torch drives, `<maca_root>/mxgpu_llvm/bin/mxcc`.  Both are passed
    in rather than reproduced here so this cannot drift from what the build
    actually compiles with.

    The include paths are the exception, and they are added here on purpose:
    torch's `include_paths()` is what the build's own include list is assembled
    from, and it differs with the torch install that is loading this file --
    hard-coding it would be the drift this signature is avoiding.  They are
    appended after the caller's, so a caller's explicit override still wins.
    Python's own headers come with them, because these translation units reach
    `torch/extension.h` and through it `Python.h`.
    """
    if mxcc is None or compile_args is None:
        raise ValueError(
            "check_maca_stack_frame needs the compiler path and the build's own "
            "compile args; see setup.py's use of it")

    import sysconfig
    from torch.utils.cpp_extension import include_paths
    try:
        extra_includes = list(include_paths(device_type="cuda"))
    except TypeError:                     # older torch has no device_type
        extra_includes = list(include_paths())
    extra_includes.append(sysconfig.get_paths()["include"])

    cmd = [mxcc, "-c", source_path, "-o", os.devnull, "--resource-usage",
           *compile_args]
    if not any(a.startswith("-I") and "torch/include" in a
               for a in compile_args):
        cmd += [f"-I{p}" for p in extra_includes]

    if not quiet:
        print(f"Checking MACA stack frames in: {source_path} "
              f"(baseline {stack_baseline} B)")

    result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
    output = result.stdout + result.stderr
    if result.returncode != 0:
        raise RuntimeError(f"mxcc failed on {source_path}:\n{output[-2000:]}")

    spilling = []
    name = None
    for line in output.splitlines():
        m = _RESOURCE_USAGE_FUNC.search(line)
        if m:
            name = m.group(1)
            continue
        if name is None:
            continue
        m = _RESOURCE_USAGE_FRAME.search(line)
        if m and int(m.group(1)) > stack_baseline:
            spilling.append((name, int(m.group(1)), None, None))
            name = None
            continue
        m = _RESOURCE_USAGE_REGS.search(line)
        if m and spilling and spilling[-1][0] == name:
            old = spilling[-1]
            spilling[-1] = (old[0], old[1], int(m.group(1)), int(m.group(3)))
            name = None

    if spilling and not quiet:
        print(f"Found {len(spilling)} kernel(s) over the stack baseline:\n")
        print(f"{'STACK':>7}  {'MTREG':>6}  {'SHARED':>7}  Kernel")
        print("-" * 70)
        for n, stack, reg, shared in spilling:
            print(f"{stack:>7}  {str(reg):>6}  {str(shared):>7}  {n}")
    elif not quiet:
        print("No MACA kernel exceeds the stack baseline.")

    return spilling
        