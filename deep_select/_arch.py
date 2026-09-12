"""MACA architecture vocabulary shared by the build and the runtime.

The ported upstream kernels gate themselves on shared memory capacity, which is
a property of the architecture they are compiled for, so the build needs to
name architectures and the runtime needs to pick the matching build.  Both go
through this module so the two cannot disagree about what a device is.

The spellings and capacities mirror the xcore family rows of the host
repository (``deep_gemm/utils/arch_config.py``, ``XcoreFamily``).  This
repository is standalone and cannot import that package, so the table is
duplicated here and the two are kept in sync by hand; a change to either
belongs in the same review.
"""

from typing import Optional
import os

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

# Family base -> per-SM shared memory capacity in bytes.  This is the number a
# kernel's config table is valid against.
CAPACITY_BYTES = {
    1000: 64 * 1024,
    1500: 128 * 1024,
    1600: 128 * 1024,
}

# The capacity the ported kernel's tuples were re-derived against
# (`scripts/generate_instantiations.py`, which refuses to emit a tuple whose
# `occupancy * shared_memory_bytes()` exceeds it).
XCORE1600_KERNEL_CAPACITY_BYTES = 128 * 1024

# The CUDA-compat sm spelling torch reports -> family base.  Which of the three
# xcore1600 spellings a part reports depends on the SDK generation, so the
# family is the stable key and the spellings are just its aliases.
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
    the exact mistake the capacity table exists to prevent.
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
    answer from the same source, so a wheel built for this machine and the
    module it loads at run time cannot disagree.
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


# ── which kernel a 128 KiB part builds -- switch, and why the default flipped ─
#
# `csrc/xcore1600/` (the ported upstream kernels) selects **wrong** on a MACA
# C600U.  Measured, not suspected: an `arange` row of 0..511 with topk=8 returns
# indices like [448..455] where the answer is [511..504], and it differs on
# every run -- eight runs, eight distinct wrong answers.  A random row returns
# 1.5e+37 for a row whose true max is 3.3.  On the official slice
# (`scripts/official_slice.py --backend maca_c`) it scored 4/200 against
# 200/200 for the hand-written kernel.
#
# The prime suspect is CUDA's 32-lane model on MACA's 64-lane wave (see
# CLAUDE.md, "Known holes").  It is a real kernel bug, not a build one: the
# pre-cu-bridge artifact is wrong too, and wrong differently each run.
#
# So the default for a 128 KiB family is `csrc/xcore1000/maca_topk.cu` -- the
# hand-written kernel CLAUDE.md calls "the shipping C500 kernel", which is
# correct on every slice it is run against and has no capacity gate (a 128 KiB
# SM runs it with room to spare).
#
#     DEEP_SELECT_128KIB_KERNEL=xcore1600     # back to the ported kernel, for
#                                             # debugging the 32-lane audit
#     DEEP_SELECT_128KIB_KERNEL=xcore1000     # the default, spelled out
#
# The switch chooses a *source tree*, never an extension name: the extension is
# still `deep_select_xcore<N>` and a caller cannot tell from the outside which
# one it is.  That is deliberate -- the operator's behavior is the contract, and
# the ported kernel does not currently meet it on this hardware.
#
# Delete this override (and the `DEEP_SELECT_128KIB_KERNEL` branch in
# `kernel_directory`) once the port passes the official slice on a C600U.
DEFAULT_128KIB_KERNEL = "xcore1000"

_KERNEL_CHOICES = ("xcore1000", "xcore1600")


def kernel_for_128kib() -> str:
    """The tree a 128 KiB family builds, from the environment or the default.

    Read at call time rather than cached, so a debugging session can flip it
    between builds without a fresh interpreter.
    """
    choice = os.environ.get("DEEP_SELECT_128KIB_KERNEL", DEFAULT_128KIB_KERNEL)
    if choice not in _KERNEL_CHOICES:
        raise ValueError(
            f"DEEP_SELECT_128KIB_KERNEL={choice!r} is not a kernel tree; "
            f"expected one of {', '.join(_KERNEL_CHOICES)}"
        )
    return choice


def kernel_directory(family: int) -> str:
    """Which kernel tree under `csrc/` serves this family.

    The split is by capacity, not by name: a part with 128 KiB of shared memory
    per SM would run `csrc/xcore1600/`, whose config tuples were re-derived for
    exactly that figure, and the 64 KiB part runs `csrc/xcore1000/`, the
    hand-written MACA kernel, which has no capacity gate to satisfy.

    A 128 KiB family currently defaults to `csrc/xcore1000/` as well -- see the
    note above `DEFAULT_128KIB_KERNEL`, and `DEEP_SELECT_128KIB_KERNEL` to
    build the port instead while it is being debugged.

    The directory name is the same string the build names the extension after
    (`deep_select_xcore<N>`), so one name covers the tree and the module; it is
    also what `topk(backend="maca_c")` resolves to on a device of this family,
    though that resolution names no architecture at the call site.
    """
    if CAPACITY_BYTES.get(family, 0) < XCORE1600_KERNEL_CAPACITY_BYTES:
        return "xcore1000"
    return kernel_for_128kib()


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
