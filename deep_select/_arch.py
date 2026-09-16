"""What the runtime reads off the device: the SM count and the family name.

Torch's device properties are the source of both -- `multi_processor_count` is
the SM ("AP") count, and the capability pair is the CUDA-compat sm spelling
every part in this ecosystem already keys on.

**Nothing here is consulted by the build.**  `CUCC_TARGETS` reaches `mxcc` as
its own `-offload-arch` vocabulary, `native` included, so a build host needs no
card.  An earlier revision resolved `native` through this module by asking
torch, which made a card-less cross-build fail with "no MACA device visible" --
a detection step that only existed to translate a spelling the compiler already
understands.

Both functions are properties of a **process**, not of a call: `interface.py`'s
`_sm_count()` caches the first, and the perf record calls the second once.
"""

import functools

# CUDA-compat sm pair -> the family base in the name a record filters on.  The
# key is the stable one, not the product name: how a part spells itself varies
# by SDK generation (`MetaX C600U` / `MetaX C600-UL`), while the sm pair does
# not.  Same vocabulary as `deep_gemm/_common.py`'s `is_xcore*_family`
# predicates and `tests/kernelkit/platform.py`'s copy.
FAMILY_OF_SM = {
    80: "xcore1000",   # C500
    86: "xcore1500",   # C600
    87: "xcore1600",   # C600U / C600-UL
    88: "xcore1600",   # C600U / C600-UL
    89: "xcore1600",   # C600U / C600-UL, legacy SDK
}


def _sm_pair() -> int:
    """This process's device capability as one integer, e.g. 80."""
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no MACA device visible; a kernel launch is not possible here"
        )
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


@functools.lru_cache(maxsize=1)
def native_sm_count() -> int:
    """SM ("AP") count of this process's device.

    A kernel's grids are sized in CTAs against this -- `maca_topk.cu`'s
    `wave_filled_chunks` and `f32_chunk_work_target` -- so it travels to the
    kernel as an argument rather than being compiled in: one extension serves
    every family.

    Raises when no device is visible.  `interface.py` calls this only from
    inside a launch, and `maca_topk.cu` refuses a zero with `DS_HOST_CHECK`
    rather than defaulting it -- a zero sizes every grid to nothing, which is
    an empty answer, not a crash.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no MACA device visible, so the SM count cannot be read; a kernel "
            "launch is not possible here"
        )
    return int(torch.cuda.get_device_properties(
        torch.cuda.current_device()).multi_processor_count)


def native_family() -> str:
    """This process's device family, e.g. ``xcore1000``.

    A label for records and logs, never an input to a decision: what a kernel
    does is decided by the arguments it is handed, and one extension serves
    every family.
    """
    try:
        return FAMILY_OF_SM[_sm_pair()]
    except KeyError:
        raise RuntimeError(
            f"device reports sm{_sm_pair()}, which is not a MACA family this "
            f"tree names; add it to deep_select/_arch.py::FAMILY_OF_SM"
        ) from None
