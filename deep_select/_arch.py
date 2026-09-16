"""The one device fact a launch needs: the SM ("AP") count.

`torch.cuda.get_device_properties().multi_processor_count` is the source, and
this module exists only because that is read in two places (`interface.py`'s
`_sm_count()` and `perf_snapshot.py`) and costs a device query each time.

**Nothing here is consulted by the build.** `CUCC_TARGETS` reaches `mxcc` as its
own `-offload-arch` vocabulary, `native` included, so a build host needs no card.
An earlier revision resolved `native` through this module by asking torch, which
made a card-less cross-build fail with "no MACA device visible" -- a detection
step that only existed to translate a spelling the compiler already understands.

The **device's name is not here either.** It is `torch.cuda.get_device_name()`,
which is what `perf_data/<device>/` is already named after; the `xcore<N>`
spelling was a second answer to the same question, derived through a capability
pair, and the two could disagree -- a part reporting anything but sm80 would
have put one name in the directory and another in the `chip` column.
"""

import functools


@functools.lru_cache(maxsize=1)
def get_device_num_sms() -> int:
    """SM ("AP") count of this process's device.

    A kernel's grids are sized in CTAs against this -- `maca_topk.cu`'s
    `wave_filled_chunks` and `f32_chunk_work_target` -- so it travels to the
    kernel as an argument rather than being compiled in: one extension serves
    every device.

    Raises when no device is visible. `interface.py` calls this only from inside
    a launch, and `maca_topk.cu` refuses a zero with `DS_HOST_CHECK` rather than
    defaulting it -- a zero sizes every grid to nothing, which is an empty
    answer, not a crash.
    """
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "no MACA device visible, so the SM count cannot be read; a kernel "
            "launch is not possible here"
        )
    return int(torch.cuda.get_device_properties(
        torch.cuda.current_device()).multi_processor_count)
