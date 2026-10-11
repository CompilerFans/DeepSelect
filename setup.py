import importlib.util
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

# ── which sources each family builds ────────────────────────────────────────
#
# The kernel *layout* is per-family (`csrc/maca_kernels/<family>/`) and this
# table is the build saying the same thing.  It is the one place a family's
# source set is decided, so both shapes this build has to express are edits
# here and nowhere else:
#
#   - **the default** -- a family with no row of its own is compiled from
#     `SHARED_SOURCES`;
#   - **the exception** -- a family with its own kernel names that kernel in
#     its row.
#
# All three rows name the same file today, and that is the measured answer
# rather than a default of convenience: the C500 kernel is the artifact every
# family ships, 1.5-2.9x faster than the ported upstream tree on a C600U and
# passing the cells that tree fails (CLAUDE.md, "Can a C600U run the C500
# kernel?").
#
# **`family 1600` and the directory `csrc/maca_kernels/xcore1600/` are two
# different things, and the name collision is the trap this table has to
# survive.**  The first is the C600U; the second is the *ported upstream* kernel
# tree, which this build does not compile -- see `build_for_maca`.  A row names
# a family's own source set; no row points at that directory, and pointing one
# there would also take its `kerutils` include back (`include_dirs` below is the
# other half of that change).
_REPO_ROOT = Path(__file__).resolve().parent
GENERATOR = _REPO_ROOT / "scripts" / "generate_instantiations.py"

_generated_sources = None


def generated_sources():
    """The `instantiations/*.cu` files, read from the generator that writes them.

    Every configuration tuple is its own translation unit holding one explicit
    instantiation, so the source list is a *table* -- `scripts/
    generate_instantiations.py`'s -- and this reads it rather than repeating it.
    89 file names transcribed here would be a second copy of a list that already
    exists, and the two would drift in the direction nobody notices: a file the
    build does not compile is a kernel the artifact silently loses.
    """
    global _generated_sources
    if _generated_sources is None:
        spec = importlib.util.spec_from_file_location(
            "_deep_select_generate_instantiations", GENERATOR)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        sources = list(module.xcore1000_sources())
        missing = [s for s in sources if not (_REPO_ROOT / s).is_file()]
        if missing:
            raise SystemExit(
                "deep_select: %d generated source(s) missing (e.g. %s).  They "
                "are committed; regenerate with `python %s`."
                % (len(missing), missing[0],
                   GENERATOR.relative_to(_REPO_ROOT)))
        _generated_sources = sources
    return list(_generated_sources)


# The hand-written half: the tvm-ffi entry point, the contract layer, the
# dispatcher.  The generated half below it is one file per configuration tuple.
SHARED_SOURCES = [
    "csrc/maca_kernels/xcore1000/maca_topk.cu",
] + generated_sources()
SOURCES_BY_FAMILY = {
    1000: SHARED_SOURCES,   # C500
    1500: SHARED_SOURCES,   # C600
    1600: SHARED_SOURCES,   # C600U -- the shared kernel, not xcore1600/
}


def sources_for_family(family):
    """The sources one family's artifact is compiled from.

    Returns a fresh list per call, so the three `Extension`s below never share
    one.  A family with no row takes `SHARED_SOURCES`: one kernel serving every
    family is this tree's standing answer, so a family added to
    `_family_of_target`'s tuple needs no edit here -- giving one its own kernel
    is one row.
    """
    return list(SOURCES_BY_FAMILY.get(family, SHARED_SOURCES))


# What this file builds when `CUCC_TARGETS` says nothing.  "Nothing" is
# **unset or empty**: `${CUCC_TARGETS:-...}` substitutes for an empty value as
# well as an unset one, so a script that sets it treats the two alike, and this
# file's `or` below reads them the same way.
#
# The default is the **family list and not `native`**, because this file is also
# what a packaging build runs: a wheel has to carry every family -- one board's
# image cannot be shipped to another -- and a build host may have no card in it,
# where `native` has no answer.  `build.sh` passes the same list explicitly;
# `develop.sh` and `install.sh` act on this machine and pass `native`, which
# `_native_family` resolves to this host's family.
#
# A *whitespace-only* value is neither unset nor empty.  It was set and it names
# nothing, so `build_for_maca` refuses it rather than reading it as a default --
# and a build that did take the default says so in its first lines, so the
# artifacts nobody asked for are visible rather than merely silent.
DEFAULT_TARGETS = "xcore1000,xcore1500,xcore1600"


# `native`'s answer, held for the build.  The target list is resolved entry by
# entry, so a list naming `native` twice would otherwise query the device and
# print the report twice before the family collapse below notices that the two
# entries are one target.
_NATIVE_FAMILY = {}


def _native_family(this_dir):
    """This build host's xcore family, for the `native` target.

    `native` is the one spelling that is *resolved* rather than passed through:
    it names this machine, and the artifact is named for and carries its
    family's constants, so the number has to be known here.  The map is the
    loader's -- `deep_select/_binding.py`'s `device_family_suffix()`, the same
    function that decides which artifact a run loads -- read out of that file by
    path, so a build does not execute the `deep_select` package (and everything
    its `__init__.py` imports) to answer one question.  The device query under
    it is torch's either way.

    **A host with no visible device cannot resolve it**, and that is refused
    loudly rather than guessed at; a device whose sm is not one of the families
    is refused separately, because the two want different fixes.  `mxcc
    -offload-arch=native` cross-compiles on a card-less build host, but that is
    the one machine where `native` has no answer.  `build.sh` never uses it --
    its default is explicit spellings -- so the shippable path does not depend
    on a device.

    The resolution is printed, because the artifact that comes out of it carries
    *this* host's numbers: a build log that says `native` and nothing else does
    not record which family it built.
    """
    if "family" in _NATIVE_FAMILY:
        return _NATIVE_FAMILY["family"]
    import importlib.util

    import torch

    path = os.path.join(this_dir, "deep_select", "_binding.py")
    spec = importlib.util.spec_from_file_location("_deep_select_binding", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suffix = module.device_family_suffix()
    known = suffix.startswith("_xcore") and suffix[len("_xcore"):].isdigit()
    if not known:
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUCC_TARGETS entry 'native' needs a device to resolve, and "
                "this build host reports none (no MACA device visible to "
                "torch).  On a card-less build host pass explicit spellings, "
                "e.g. CUCC_TARGETS=xcore1000,xcore1500,xcore1600."
            )
        major, minor = torch.cuda.get_device_capability()
        raise RuntimeError(
            f"CUCC_TARGETS entry 'native' found a device at sm "
            f"{major * 10 + minor}, which is not one of the families this "
            f"build has a row for; pass that family's spelling explicitly."
        )
    family = int(suffix[len("_xcore"):])
    if family not in (1000, 1500, 1600):
        raise RuntimeError(
            f"CUCC_TARGETS entry 'native' resolved to xcore{family}, which has "
            f"no row in csrc/structs.h (AP count and per-AP shared memory); add "
            f"one before building for it."
        )
    major, minor = torch.cuda.get_device_capability()
    _NATIVE_FAMILY["family"] = family
    print(f"deep_select: native -> xcore{family} (device sm {major * 10 + minor})")
    return family


def _offload_spelling(target, family):
    """The `-offload-arch` spelling a target is built under.

    The caller's spelling, except for `native`: it resolved to a family, and
    passing that family on is what keeps `-offload-arch`, `-DDEEP_SELECT_ARCH`
    and the artifact's name stating the same number.
    """
    return "xcore%d" % family if target == "native" else target


def _family_of_target(target, this_dir):
    """The xcore family a `-offload-arch` spelling names, as an integer.

    `xcore<N>` is mxcc's own vocabulary, so the family is the suffix -- but it
    is parsed rather than sliced, because a target this file cannot turn into a
    family is a build that must fail here.  `csrc/structs.h` refuses an unknown
    `DEEP_SELECT_ARCH` with its own `#error`; this is the same refusal one step
    earlier, where the message can name the offending `CUCC_TARGETS` entry.

    **A spelling is accepted only if its number is one of this build's
    families**, and that is a decision rather than a shortcut: mxcc's target-ID
    space is wider than these three (it also takes `xcore1002`, `xcore1502`,
    `xcore1520` and others), and this file has nothing to resolve one of those
    to a row.  A row carries the family's AP count and per-AP shared memory, so
    an artifact built from the wrong row sizes its grids for the wrong machine,
    and it does that silently.  Device code does not settle it either: on this
    toolchain `xcore1500` and `xcore1600` compile the same kernel to identical
    instructions, so "same codegen" is not "same family".  A part that needs a
    family of its own gets a row -- the two constants in `csrc/structs.h` --
    rather than a guess from its name.  Measured survey, including which
    spellings mxcc accepts at all:
    `docs/experiments/20261012_setup_family_detection/`.

    **`native` is accepted, and resolved to this host's family**
    (`_native_family` above): it is what `develop.sh` and `install.sh` default
    to, and both act on *this* machine.  It is the one spelling a build must not
    be *shipped* from -- a wheel built on a C500 carries the C500's numbers
    under a name that claims to be whatever target machine lands on it -- which
    is why `build.sh`'s default is the explicit list.  Resolution needs a device
    and refuses loudly without one.  (Measured on this host,
    `-offload-arch=native` is byte-identical to `-offload-arch=xcore1000`, so
    resolving it here agrees with what the compiler would have done.)
    """
    families = (1000, 1500, 1600)
    spelled = ", ".join("xcore%d" % f for f in families)
    if target == "native":
        return _native_family(this_dir)
    prefix = "xcore"
    if not target.startswith(prefix) or not target[len(prefix):].isdigit():
        raise RuntimeError(
            f"CUCC_TARGETS entry {target!r} does not name an xcore family "
            f"(expected `xcore<N>`).  This build produces one artifact per "
            f"family and needs the family from the target spelling.  The "
            f"families this build has rows for are {spelled}."
        )
    family = int(target[len(prefix):])
    if family not in families:
        raise RuntimeError(
            f"CUCC_TARGETS entry {target!r} names architecture {family}, which "
            f"is not one of the families this build has a row for ({spelled}).  "
            f"`csrc/structs.h` holds each family's AP count and per-AP shared "
            f"memory and the artifact is named for the family, so a target this "
            f"build cannot name cannot be built -- and a spelling like "
            f"`xcore1502` or `xcore1520`, which mxcc does accept, is not "
            f"evidence of which family it belongs to.  If the part shares one "
            f"of the families above, pass that spelling; otherwise add the "
            f"family's row to csrc/structs.h and to the tuple here."
        )
    return family


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
    """`MACA_PATH` first, then `MACA_HOME`, then `/opt/maca`.

    That order is the scripts' order too (`build.sh` exports `MACA_PATH` from
    `${MACA_PATH:-${MACA_HOME:-/opt/maca}}`), so a host that sets only
    `MACA_HOME` gets one root here and in every path they derive from it.
    """
    return os.environ.get("MACA_PATH") or os.environ.get("MACA_HOME") or "/opt/maca"


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
    """Build one extension per family: that family's sources, one compile each.

    **This is a csrc compile and nothing else.**  One `.cu`, one `.so` per
    target; everything under `deep_select/` is Python and reaches an artifact
    through `tvm_ffi.load_module`.

    `CUCC_TARGETS` is a list of `xcore<N>` spellings, and **each one is its own
    `-offload-arch` compile** carrying `-DDEEP_SELECT_ARCH=<N>` beside it.  That
    pairing is what makes the host half of an image agree with its device half:
    `csrc/structs.h` reads the macro for the family's AP count and per-AP shared
    memory, and refuses a target with no row.  Unset means `DEFAULT_TARGETS`
    below, one target per family, so a build host needs no MACA card.

    **`native` is accepted and resolved to this host's family** -- it is what
    `develop.sh` and `install.sh` default to, and both act on *this* machine.
    A per-family artifact must not have its constants chosen by the build
    machine, though, so a *wheel* must not be built that way: it would carry
    this host's numbers under a name that claims to be the target machine's.
    That is why `build.sh`'s default is the explicit family list; a narrow
    `CUCC_TARGETS=xcore1000` is still allowed, and produces a wheel that serves
    one family and names it in the error on any other.

    The two numbers the kernel sizes its grids against are compile-time
    constants again (`ARCH_SM_COUNT`, `ARCH_SMEM_PER_AP_BYTES`), which is the
    reason for one artifact per family.  The form this replaced -- one file,
    one comma-separated `-offload-arch`, three device images -- cannot carry
    them: `__MACA_ARCH__` is defined in the device pass only, so a host `#if` on
    it takes the `#else` branch on every architecture, and host code exists once
    in a fat binary regardless.

    `csrc/maca_kernels/xcore1600/` is not built: no row of
    `SOURCES_BY_FAMILY` names it and its `kerutils` include is off
    `include_dirs` below.  Re-adding both is what building it would take.

    **Run `ref/kerutils_platform_arms/run_probe.sh` first.**  The two edits above
    are the whole mechanical change, so they are what a reader looks at -- and a
    build that makes them does not thereby exercise the vendored library's MACA
    arm, which nothing else in the tree compiles.  The arm is where this tree's
    platform selection lives, and it is the one thing that cannot be checked by
    the act of adding the consumer back.

    Every source is a `.cu`, so the device compiler is the only compiler this
    build invokes (torch would route a `.cpp` to `$cxx`, meaning a second flag
    list and a second kerutils mode macro).  `CUDA_HOME` is deliberately left
    alone -- see the cu-bridge note above.
    """
    import torch.utils.cpp_extension as cpp_extension
    from torch.utils.cpp_extension import BuildExtension, CUDAExtension

    maca_root = _maca_root()
    this_dir = os.path.dirname(os.path.abspath(__file__))

    # The headers and `libtvm_ffi.so` ship inside the `tvm_ffi` python package,
    # so asking that package beats guessing a prefix.  `lib_subdir` is
    # platform-dependent and resolved, because the rpath below points at it.
    tvm_ffi_root, lib_subdir = _tvm_ffi_root()

    # Unset or empty means `DEFAULT_TARGETS` -- the scripts' `${VAR:-default}`
    # reads an empty value the same way, so this `or` is that rule and not a
    # coincidence (`DEFAULT_TARGETS` above).  A value that was set and names no
    # target at all is a different thing and is refused: it is the shape a typo
    # arrives in, and taking the default there would answer a question nobody
    # asked.  A build that did fall back says so, because that fallback is the
    # one that produces three artifacts on a bare `python setup.py`.
    raw_targets = os.environ.get("CUCC_TARGETS")
    targets = [t.strip() for t in
               (raw_targets or DEFAULT_TARGETS).split(",") if t.strip()]
    if not targets:
        raise RuntimeError(
            f"CUCC_TARGETS={raw_targets!r} is set but names no target; leave it "
            f"unset or empty for the default list {DEFAULT_TARGETS}")
    if not raw_targets:
        print(f"deep_select: CUCC_TARGETS unset or empty; "
              f"building DEFAULT_TARGETS={DEFAULT_TARGETS}")
    # Resolved once, in order: the family decides the artifact's name, its
    # constants and which sources it takes.  `native` is the only spelling whose
    # name is not itself; it is resolved and reported by `_native_family`.
    #
    # **One artifact per family**, so two entries that resolve to one family are
    # built once.  The case arrives without meaning to -- `native,xcore1000` on
    # the host `native` names, or a doubled entry -- and it is not an error: the
    # artifacts are the same set either way, so it is reported and collapsed
    # rather than refused.  It has to be handled because the two Extensions
    # would otherwise take the same `name=` and the same output path; measured,
    # that builds the artifact twice (ninja dedupes the compile by command hash,
    # so the cost is one relink and one extra copy).
    #
    # The collapse lands in `resolved` itself, and everything downstream reads
    # that one list -- the Extension comprehension, the source summary, and the
    # `targets` this print names, which is derived from it rather than kept
    # beside it.
    seen_family, resolved = {}, []
    for target in targets:
        family = _family_of_target(target, this_dir)
        if family in seen_family:
            print(f"deep_select: CUCC_TARGETS entries {seen_family[family]!r} "
                  f"and {target!r} both name xcore{family}; building it once")
            continue
        seen_family[family] = target
        resolved.append((target, family))
    targets = [target for target, _ in resolved]

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
        # `csrc/ffi/` -- the tvm-ffi edge (tensor/error/check helpers).  Every
        # kernel source includes it as `"ffi_..."`, so the directory itself is
        # on the path; nothing names it by relative depth, which is what keeps
        # these includes independent of where a kernel tree sits.
        os.path.join(this_dir, "csrc", "ffi"),
        os.path.join(maca_root, "include"),
        # `csrc/maca_kernels/xcore1600/` is NOT on this list and neither is
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
        # The platform this build is for, beside the family.  Upstream declares
        # its own the same way (`-DDEEP_SELECT_IS_BUILD_ON_CUDA` on the CUDA
        # extension, `_ASCEND` on the NPU one) and `csrc/structs.h` refuses a
        # build that names no platform, so the declaration is checked rather
        # than conventional.
        #
        # It is deliberately *not* what kerutils keys on.  That library
        # detects its platform from the toolchain -- `__CUDACC__`,
        # `__has_include("kernel_operator.h")`, `__MACA__` -- so it keeps
        # working when vendored into a build that passes no flags, which a
        # vendored library cannot require.
        "-DDEEP_SELECT_IS_BUILD_ON_MACA",
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
    ]
    # `-offload-arch` and `-DDEEP_SELECT_ARCH` are per-Extension, not shared:
    # one artifact per family, each its own compile.  See the loop below.
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

    # **One Extension per family.**  The pair `(-offload-arch=xcore<N>,
    # -DDEEP_SELECT_ARCH=<N>)` is what makes the host half of the image agree
    # with its device half; `csrc/structs.h` turns a target with no row into a
    # build error.  The intermediate form -- one Extension, one comma-separated
    # `-offload-arch` -- cannot carry per-family host constants at all: that is
    # one compilation, host code exists once in it, and `__MACA_ARCH__` is
    # defined in the device pass only, so an `#if` on the target takes the
    # `#else` branch on every architecture.  Reverting to it means the
    # constants travel as arguments again and every reader of this file needs
    # to know why -- see `csrc/structs.h`'s block and CLAUDE.md.
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
                # One artifact per family, named for the family it carries.
                # `deep_select_maca` (the C++ namespace) plus the suffix
                # `_binding.py` globs and the loader resolves -- the device's
                # family has to name the file, because the file's *host* half
                # is what holds that family's constants.
                name=f"deep_select.deep_select_maca_xcore{target_family}",
                # This family's own source set, resolved once per Extension --
                # `sources_for_family` hands back a fresh list, so no two
                # artifacts share one.
                sources=sources_for_family(target_family),
                # Every source is a `.cu`, so the `nvcc` list is the only one
                # torch reads, and its contents are mxcc's dialect because
                # `$nvcc` resolves to cucc.  The two per-family flags are
                # appended here rather than shared, which is the whole point of
                # one Extension per target.
                extra_compile_args={"nvcc": nvcc_args + [
                    # `native` is resolved before it reaches here, so the
                    # spelling passed on is the family it named: `-offload-arch`,
                    # `-DDEEP_SELECT_ARCH` and the artifact's name then state the
                    # same number.
                    f"-offload-arch={_offload_spelling(target, target_family)}",
                    f"-DDEEP_SELECT_ARCH={target_family}",
                ]},
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
        for target, target_family in resolved
    ]
    # One clause per *distinct* source set, read off the table rather than a
    # second copy of it: families that compile the same tree share a clause,
    # and a family with its own kernel gets its own.  The generated
    # instantiations are counted rather than listed -- there are 89 of them and
    # the list is the directory they live in.
    def summarise(sources):
        hand = [s for s in sources if "/instantiations/" not in s]
        generated = len(sources) - len(hand)
        clause = ", ".join(hand)
        return f"{clause} + {generated} generated instantiations" if generated else clause

    distinct = sorted({summarise(sources_for_family(f)) for _, f in resolved})
    print(f"deep_select: compiling {'; '.join(distinct)} once per family, "
          f"for {','.join(targets)}")

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
#     DEEP_SELECT_MACA_STACK_CHECK=csrc/maca_kernels/xcore1000/maca_topk.cu ./build.sh
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


# --- the version string's three parts ---------------------------------------
#
# The MACA toolkit generation, read from the environment the way mcDeepGEMM's
# `setup.py` reads it and folded in the way it folds it (`<version>+maca<v>`).
# `MACA_AI_VERSION` is the same variable's earlier spelling, taken second for
# parity with that reference, whose own comment marks it for removal.
#
# It earns its place in the *name* rather than in a manifest: the extension
# resolves `libmcruntime.so` through the rpath baked in at link time, so a
# wheel built against one toolkit generation dies on another with
# `mcErrorInvalidDeviceFunction` -- 0/202, not a plausible partial -- which
# reads like a kernel defect.  See the run-time trap in CLAUDE.md.  A wheel
# name that does not say which toolkit it was built against is the one name a
# consumer cannot check.
#
# `build.sh` and `install.sh` supply it from `$MACA_PATH` so the default below
# is what a bare `python setup.py bdist_wheel` gets -- and `maca0.0.0.0` says,
# truthfully, that no toolkit version was given.
maca_version = os.environ.get("MACA_VERSION", "0.0.0.0")
if maca_version == "0.0.0.0":
    maca_version = os.environ.get("MACA_AI_VERSION", "0.0.0.0")

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
    version=f"{__version__}+maca{maca_version}.{git_rev}.{datetime_rev}",
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
