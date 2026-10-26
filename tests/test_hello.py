"""Say what this machine will actually run, then run one small call.

The companion of mcDeepGEMM's `deep_gemm/tests/test_hello.py`, and it prints
the same first block for the same reason: the device, the interpreter and the
framework are the three things every measurement here is conditional on.
It adds what a top-K operator needs on top of that -- **which artifact this
process will load**, and which toolkit the artifact resolves against -- because
both are silent-wrong-answer traps rather than conveniences:

  - `_binding.extension_path` is the loader's own answer, so the md5 printed
    here is the receipt for the binary a `topk` call would use, not a guess
    from a path under the checkout.  Two measurements are only comparable when
    that number is the same.
  - the MACA toolkit is resolved through the rpath baked at link time.  An
    artifact built against another toolkit generation fails **every** call with
    `mcErrorInvalidDeviceFunction`, which reads exactly like a kernel defect;
    the toolkit version and the resolved `libmcruntime.so` are what tell the
    two apart.

The git revision and build time are **not** printed: they are already on the
package, as `setup.py`'s version string (`1.0.0+maca<toolkit>.<git-rev>.
<timestamp>`), and this prints that string.  Re-deriving them from `.git`
would add a second answer to a question that already has one.

    python tests/test_hello.py                 # device 0
    CUDA_VISIBLE_DEVICES=1 python tests/test_hello.py

Which package answers follows the rule `run_bench.sh`, `tools/sample_gate.py`
and `tests/contract_test.py` use: the checkout when it has a built extension,
the installed wheel otherwise.  Exit status is the verdict -- 0 only when every
check in the smoke passed.
"""
import hashlib
import importlib.metadata
import os
import sys

# The path rule the other runners use.  It has to run before `deep_select` is
# importable, since it decides *which* `deep_select` answers.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if any(name.startswith("deep_select_maca") and name.endswith(".so")
       for name in os.listdir(os.path.join(_REPO, "deep_select"))):
    sys.path.insert(0, _REPO)


def _md5_of(path: str) -> str:
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_line(path: str, prefix: str) -> str:
    try:
        with open(path) as handle:
            for line in handle:
                if line.startswith(prefix):
                    return line[len(prefix):].strip()
    except OSError:
        pass
    return ""


def _loaded_library(basename: str) -> str:
    """The path the loader actually mapped for `basename`, or "".

    The dynamic loader's answer, not the toolkit's: this is the file whose
    build generation decides whether the artifact's device functions exist.
    """
    try:
        with open("/proc/self/maps") as maps:
            for line in maps:
                fields = line.split()
                if len(fields) >= 6 and fields[-1].endswith("/" + basename):
                    return fields[-1]
    except OSError:
        pass
    return ""


def _print_toolchain() -> None:
    maca_home = (os.environ.get("MACA_PATH")
                 or os.environ.get("MACA_HOME") or "/opt/maca")
    print("maca home:", maca_home)
    resolved = os.path.realpath(maca_home)
    # The symlink is re-pointed under you, so the two are separate facts.
    print("maca home resolves to:", resolved)
    version = _first_line(os.path.join(maca_home, "Version.txt"), "Version:")
    print("maca version:", version or "(Version.txt not found)")

    mxcc = os.path.join(maca_home, "mxgpu_llvm", "bin", "mxcc")
    print("mxcc path:", mxcc if os.path.exists(mxcc) else "(not found)")
    cucc = os.path.join(maca_home, "tools", "cu-bridge", "bin", "cucc")
    print("cucc path:", cucc if os.path.exists(cucc) else "(not found)")

    runtime = _loaded_library("libmcruntime.so")
    print("libmcruntime.so:", runtime or "(not loaded)")


def main() -> int:
    print("Python ver:", sys.version)
    try:
        import torch
    except Exception as error:  # any import failure is the answer
        print("PyTorch ver: unavailable --", error)
        print()
        _print_toolchain()
        print()
        print("torch could not be imported, so deep_select cannot be imported")
        print("either (its interface module imports torch).  MACA is a")
        print("prerequisite of torch, not only of this kernel: `libmcruntime.so`")
        print("above is what import torch needs first.")
        return 1

    print("PyTorch ver:", torch.__version__)
    print("device count:", torch.cuda.device_count())
    if not torch.cuda.is_available():
        print("device: no MACA device visible")
        _print_toolchain()
        return 1
    props = torch.cuda.get_device_properties(0)
    print("device name:", torch.cuda.get_device_name(0))
    print("device capability:", torch.cuda.get_device_capability())
    print("multi processor count:", props.multi_processor_count)
    print("total memory (GiB):", props.total_memory / 1024 ** 3)

    try:
        import deep_select
        from deep_select import _binding
    except Exception as error:  # the failure is the report
        print("deep_select import failed --", error)
        _print_toolchain()
        return 1

    print()
    print("deep_select path:", deep_select.__file__)
    try:
        # The wheel's version is where the toolkit generation, the git
        # revision and the build time are recorded; `__version__` is the
        # static base a checkout reports.
        version = importlib.metadata.version("deep_select")
    except importlib.metadata.PackageNotFoundError:
        version = f"{deep_select.__version__} (checkout: no wheel metadata)"
    print("deep_select version:", version)
    print("device family:", _binding.device_family_suffix() or "(unknown)")
    artifact = _binding.extension_path("deep_select_maca")
    print("extension artifact:", artifact or "(not built)")
    if artifact:
        print("extension md5:", _md5_of(artifact))
    print("tvm_ffi version:", _module_version("tvm_ffi", "apache-tvm-ffi"))
    # The resolver is a constant now, so this line is the whole answer: what a
    # call with no `backend=` runs, and what names `backend=` accepts.
    # The list is the dispatcher's, not a copy: `interface._BACKENDS` is what
    # `backend=` actually accepts, so this line cannot go stale against it.
    from deep_select.interface import _BACKENDS
    print(f"topk backends: {', '.join(_BACKENDS)} "
          f"(default: {_default_backend_name()})")
    _print_toolchain()
    try:
        print("stride requirement (input, output) bytes:",
              deep_select.get_stride_requirement())
    except Exception as error:  # an unloadable artifact is an answer
        print("stride requirement: unavailable --", error)

    print()
    # Two cells, because the dtype decides what can be asserted.  `arange` is
    # exactly representable in bf16 **only up to 256**: bf16 has 8 significand
    # bits, so above 256 the spacing is 2 and `arange(512)` is a run of ties
    # (`511` rounds to `512`).  fp32 at 256 therefore has one written-down
    # answer, and bf16 at 1024 -- the production dtype and a legal row stride
    # of 1024 bytes, where 256 bf16 is only 512 -- has only the contract:
    # `min(selected) >= max(unselected)`.
    topk = 8
    cases = ((torch.float32, 256, True), (torch.bfloat16, 1024, False))
    ok = True
    for dtype, vocab, exact in cases:
        x = (torch.arange(vocab, device="cuda", dtype=torch.float32)
             .unsqueeze(0).repeat(2, 1).to(dtype))
        print(f"topk smoke: 2 x {vocab}, topk={topk}, {str(dtype).split('.')[-1]}"
              f" arange ({'exact set' if exact else 'contract only'})")
        for backend in ("torch", "maca_c"):
            try:
                _, idx = deep_select.topk(x, topk, backend=backend)
                torch.cuda.synchronize()
                if exact:
                    passed = sorted(idx[0].tolist()) == list(
                        range(vocab - topk, vocab))
                else:
                    selected = x[0, idx[0]]
                    outside = torch.ones(vocab, dtype=torch.bool, device="cuda")
                    outside[idx[0]] = False
                    passed = bool(selected.min() >= x[0, outside].max())
                shown = f"idx {idx[0].tolist()}"
            except Exception as error:
                shown, passed = f"raised {type(error).__name__}: {error}", False
            ok = ok and passed
            print(f"  {backend:<7}", shown, "ok" if passed else "FAILED")
    return 0 if ok else 1


def _default_backend_name() -> str:
    try:
        from deep_select.interface import _default_backend
        return _default_backend()
    except Exception:
        return "unknown"


def _module_version(module_name: str, dist_name: str) -> str:
    """The module's own `__version__`, else the distribution's.

    The two disagree on this stack: the package is distributed as
    `apache-tvm-ffi` while the import is `tvm_ffi`, so a metadata query on the
    import name reports "not installed" for an installed library.
    """
    try:
        module = __import__(module_name)
        version = getattr(module, "__version__", None)
        if version:
            return version
    except Exception:  # its absence is the answer
        pass
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return "(not installed)"


if __name__ == "__main__":
    sys.exit(main())
