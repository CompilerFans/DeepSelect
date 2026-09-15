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

Which *source tree* a family builds is deliberately **not** part of this
vocabulary.  Every family builds the same one, it is one line in `setup.py`,
and it cannot be reached from an environment variable -- see the note at that
selection site and CLAUDE.md's "Kernel architecture".  A switch that could put
the broken kernel back is a switch that can be left on.
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

# Family base -> per-SM shared memory capacity in bytes.  This is the number a
# kernel's config table is valid against.
CAPACITY_BYTES = {
    1000: 64 * 1024,
    1500: 128 * 1024,
    1600: 128 * 1024,
}

# Family base -> SM ("AP") count of the parts in it.  The same numbers as
# `csrc/structs.h`'s `NATIVE_SM_COUNT`, which is where the kernels read them;
# this copy exists so a host-side report can name the machine without a device
# call, and the two are kept in sync by hand like the capacity table above.
SM_COUNT = {
    1000: 104,   # C500
    1500: 28,    # C600
    1600: 32,    # C600U / C600-UL
}

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
# **Every family this tree builds runs `csrc/xcore1000/maca_topk.cu`** -- the
# hand-written kernel CLAUDE.md calls "the shipping C500 kernel".  The ported
# upstream kernels under `csrc/xcore1600/` are kept in the tree, complete and
# compiling, as the reserved implementation: they are **not reachable from a
# build**, and reaching them is a source change in `setup.py` rather than an
# environment variable.  A switch that could put the broken kernel back is a switch that can
# be left on.
#
# Why the port is not the default, measured on a C600U (MACA 3.8.1, device 1;
# both artifacts built for the same extension name, official cases and checks --
# CLAUDE.md has the full table):
#
#     cell                        csrc/xcore1000        csrc/xcore1600
#     4096 x   16384 k=512 bf16    1413.8 us   94.9 GB/s   4146.8 us  32.4 GB/s  FAIL
#     4096 x    1024 k=512 bf16     477.4 us   17.6 GB/s    997.3 us   8.4 GB/s  FAIL
#     512 x  262144 k=512 bf16    1793.1 us  149.7 GB/s   2721.2 us  98.6 GB/s  FAIL
#       8 x    4096 k=512 fp32      16.7 us    7.9 GB/s     16.4 us   8.0 GB/s  FAIL
#
# `check_result` passes every one of those cells on `csrc/xcore1000/` and fails
# every one on the port.  So this is not a fallback that happens to be correct:
# the shipping kernel is **1.5-2.9x faster on the parts in question as well**.
#
# Two reasons that is possible at all, both checkable rather than assumed:
#
#   * Nothing in `csrc/xcore1000/` is C500-specific code -- no `__MACA_ARCH__`
#     branch anywhere in the tree, every cross-lane primitive already 64-lane
#     (`radix_core.cuh`'s `kWarpSize = 64` under `__MACACC__`), and integer key
#     ranking with no float math to differ per part.
#   * **Capacity cannot fire in this direction.**  `csrc/structs.h` picks
#     `NATIVE_SHARED_MEMORY_PER_SM_BYTES` (64 KiB for family 1000, 128 KiB for
#     1500/1600) from `-DDEEP_SELECT_NATIVE_ARCH`, which `setup.py` passes from
#     the target, so a C600/C600U build gets the 128 KiB constant.  The
#     compile-time gate exists to reject an *oversized* config; a 64 KiB-sized
#     one in a 128 KiB SM cannot trip it.
#
# The one real cost is that the SM-count-sensitive constants (`wave_filled_chunks`
# and `NATIVE_F32_CHUNK_WORK_TARGET`, both keyed on `NATIVE_SM_COUNT`) carry the
# C500 values' arithmetic to 28 and 32 SMs rather than being re-measured there;
# the source marks them as such, and the table above bounds the error.
#
# What a caller gets on a 128 KiB part, from `maca_topk.cu` rather than the port:
# `topk` in `(1024, 4096]` is answerable (the port refuses it), `vocab_size >=
# 2^23` is not a limit (it is the port's fp32-simulated census), and bf16
# `sorted_value` works (the port, as upstream, rejects it).
#
# To work on the port: build it by hand, naming its sources --
#
#     CUCC_TARGETS=xcore1600 python setup.py build_ext --inplace   # after
#     # pointing `setup.py`'s source selection at `_xcore1600_sources()`
#
# -- and expect `scripts/official_slice.py --backend maca_c` to fail on a
# C600U.  It is unvalidated, not merely unused: see CLAUDE.md, "Known holes".
