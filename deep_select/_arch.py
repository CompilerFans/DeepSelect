"""MACA architecture vocabulary shared by the build and the runtime.

Two jobs, and they meet at the family base (1000 / 1500 / 1600):

* the build turns `CUCC_TARGETS` into `--offload-arch` spellings, and
* the runtime turns the device torch sees into the two **machine numbers a
  grid is sized against** -- the SM count and the fp32 split's work target.

Neither number is asked of the driver.  The device reports its architecture
through torch, the architecture names a family, and the family is the key to
the tables below -- one lookup on a fact the caller already has, rather than a
device query per call.

The family rows mirror the host repository's
``deep_gemm/utils/arch_config.py`` ``XcoreFamily`` rows, copied by hand because
this repository does not import it; a change to either belongs in the same
review.
"""

from typing import Optional

# Compiler target spelling (`--offload-arch=xcore<N>`) -> family base.  Only
# what `mxcc` accepts: `xcore1008`, `xcore1610` and `xcore1620` are rejected
# with `invalid target ID`, so they are not here.
FAMILY_OF_TARGET = {
    "xcore1000": 1000,
    "xcore1500": 1500,
    "xcore1502": 1500,
    "xcore1520": 1500,
    "xcore1600": 1600,
}

# What the build does when `CUCC_TARGETS` says nothing: one target per family.
# One source for it, so `build.sh`/`install.sh` do not repeat the literal.
DEFAULT_TARGETS = "xcore1000,xcore1500,xcore1600"

# Family base -> SM ("AP") count of the parts in it.  A chunked grid is sized
# in CTAs of `kBatch * chunks`, and a count that is not a multiple of this
# leaves `ctas mod SM` SMs idle in the last wave, so every grid-sizing
# decision in the kernel is a function of it.  Passed to the kernel, not
# compiled in: one extension serves all three families (setup.py).
SM_COUNT = {
    1000: 104,   # C500
    1500: 28,    # C600
    1600: 32,    # C600U / C600-UL
}

# No work-target row: the fp32 split derives it from `SM_COUNT` alone (see
# `maca_topk.cu`'s `f32_chunk_work_target`).

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

    An unknown target is an error rather than a default: a guessed family
    silently builds images for a part the caller did not name.
    """
    try:
        return FAMILY_OF_TARGET[target]
    except KeyError:
        raise ValueError(
            f"unknown MACA target {target!r}; add it to "
            f"deep_select/_arch.py::FAMILY_OF_TARGET"
        ) from None


def native_family() -> int:
    """Family base of the device in this process, read through torch.

    torch is the only source consulted, for both the build's `native` spelling
    and the runtime's machine numbers -- one answer from one place, so a build
    and the module loaded into a process on that device cannot disagree about
    what the device is.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no MACA device visible, so the native architecture cannot be "
            "detected; the build is unaffected (it names its targets with "
            "CUCC_TARGETS), but a kernel launch is not"
        )
    major, minor = torch.cuda.get_device_capability()
    sm = major * 10 + minor
    try:
        return FAMILY_OF_SM[sm]
    except KeyError:
        raise RuntimeError(
            f"device reports sm{sm}, which is not a known MACA family; add it to "
            f"deep_select/_arch.py::FAMILY_OF_SM"
        ) from None


def native_target() -> str:
    """The ``--offload-arch`` spelling of the device in this process."""
    return f"xcore{native_family()}"


def native_sm_count() -> int:
    """SM count of the device in this process, from its family."""
    return SM_COUNT[native_family()]


def resolve_targets(spec: Optional[str]) -> list:
    """Expand a ``CUCC_TARGETS`` value into concrete target spellings.

    These become the one extension's `-offload-arch` list, so this is the set
    of images in the fat binary, not a set of builds.

    **The default is the whole family list, not this device.**  A build host
    need not have a MACA card in it at all, and a wheel that carries only the
    device it was built on is a wheel that cannot be shipped anywhere else --
    so "no value" means every family this tree names, and ``native`` remains
    available as an *explicit* spelling for a caller who wants a fast local
    build of exactly one image.
    """
    if not spec:
        spec = DEFAULT_TARGETS
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
