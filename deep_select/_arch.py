"""MACA architecture vocabulary shared by the build and the runtime.

A kernel's config is sized against its architecture's shared memory capacity,
so the build names architectures and the runtime picks the matching build;
both go through here so the two cannot disagree about what a device is.

The spellings and capacities duplicate the host repository's
``deep_gemm/utils/arch_config.py`` ``XcoreFamily`` rows by hand -- this
repository is standalone and cannot import it -- so a change to either belongs
in the same review.
"""

from typing import Optional

# Compiler target spelling (`--offload-arch=xcore<N>`) -> family base.  The
# sub-variants (1008, 1502, 1520, 1610, 1620) are members of the family whose
# base they share.
FAMILY_OF_TARGET = {
    "xcore1000": 1000,
    "xcore1008": 1000,
    "xcore1500": 1500,
    "xcore1502": 1500,
    "xcore1520": 1500,
    "xcore1600": 1600,
    "xcore1610": 1600,
    "xcore1620": 1600,
}

# Family base -> per-SM shared memory capacity in bytes: the number a kernel's
# config table is valid against.
CAPACITY_BYTES = {
    1000: 64 * 1024,
    1500: 128 * 1024,
    1600: 128 * 1024,
}

# Family base -> SM ("AP") count of the parts in it.  The same numbers as
# `csrc/structs.h`'s `NATIVE_SM_COUNT`, which is where the kernels read them;
# this copy lets a host-side report name the machine without a device call.
SM_COUNT = {
    1000: 104,   # C500
    1500: 28,    # C600
    1600: 32,    # C600U / C600-UL
}

# The CUDA-compat sm spelling torch reports -> family base.  Which xcore1600
# spelling a part reports depends on the SDK generation, so the family is the
# stable key and the spellings are its aliases.
FAMILY_OF_SM = {
    80: 1000,
    86: 1500,
    87: 1600,
    88: 1600,
    89: 1600,
}


def family_of_target(target: str) -> int:
    """Family base of a ``--offload-arch`` spelling, or raise if unknown.

    An unknown target is an error rather than a default: guessing a capacity is
    the mistake the capacity table exists to prevent.
    """
    try:
        return FAMILY_OF_TARGET[target]
    except KeyError:
        raise ValueError(
            f"unknown MACA target {target!r}; add it to "
            f"deep_select/_arch.py::FAMILY_OF_TARGET (and to the host "
            f"repository's arch table if it is a new family)"
        ) from None


def native_target() -> str:
    """The ``--offload-arch`` spelling of the device in this process.

    Read through torch, which is how the runtime picks a build too -- the same
    answer from the same source, so the wheel and the module it loads cannot
    disagree.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no MACA device visible, so the native target cannot be detected; "
            "build with an explicit CUCC_TARGETS=xcore<N> instead"
        )
    sm = torch.cuda.get_device_capability()[0] * 10 + torch.cuda.get_device_capability()[1]
    try:
        family = FAMILY_OF_SM[sm]
    except KeyError:
        raise RuntimeError(
            f"device reports sm{sm}, which is not a known MACA family; add it to "
            f"deep_select/_arch.py::FAMILY_OF_SM"
        ) from None
    return f"xcore{family}"


def resolve_targets(spec: Optional[str]) -> list:
    """Expand a ``CUCC_TARGETS`` value into concrete target spellings.

    Mirrors the host repository's meaning of the variable (its ``build.sh``
    defaults it to ``native``); ``native`` resolves to the device this build
    runs on.
    """
    if not spec:
        spec = "native"
    targets = []
    for entry in spec.split(","):
        entry = entry.strip()
        if not entry:
            continue
        targets.append(native_target() if entry == "native" else entry)
    if not targets:
        raise ValueError(f"CUCC_TARGETS={spec!r} selects no target")
    for target in targets:
        family_of_target(target)
    return targets


# ── which kernel a family builds -- a build-side note, not this module's ──
#
# **Every family this tree builds runs `csrc/xcore1000/maca_topk.cu`.**  The
# ported upstream kernels under `csrc/xcore1600/` are kept in the tree,
# complete and compiling, as the reserved implementation: they are not
# reachable from a build, and reaching them is a source change in `setup.py`
# rather than an environment variable -- a switch that could put the broken
# kernel back is a switch that can be left on.
#
# The port is not merely unused, it is wrong: it fails `check_result` on every
# cell measured on a C600U and selects wrong on an `arange` row, while
# `maca_topk.cu` passes all of them and is 1.5-2.9x faster besides.  That is
# feasible because nothing in `csrc/xcore1000/` is C500-specific code (no
# `__MACA_ARCH__` branch, 64-lane cross-lane primitives, integer-key ranking
# with no float math to differ per part) and because the capacity gate cannot
# fire upward: a 64 KiB-sized config in a 128 KiB SM cannot trip it while
# `-DDEEP_SELECT_NATIVE_ARCH` is passed from the target.
#
# What a caller gets on a 128 KiB part from `maca_topk.cu` rather than the port:
# `topk` in `(1024, 4096]` is answerable, `vocab_size >= 2^23` is not a limit,
# and bf16 `sorted_value` works.  To work on the port, point `setup.py`'s source
# selection at `_xcore1600_sources()` and expect it to fail; see the handover
# and CLAUDE.md's "Known holes" for the measurement and the diagnosis.
