import enum
import functools

import torch


class Platform(enum.Enum):
    CUDA = "CUDA"
    CPU_ONLY = "CPU_ONLY"


@functools.lru_cache(maxsize=1)
def get_current_platform() -> Platform:
    """
    Get the current platform by asking torch whether it can see a device

    This used to grep `lspci` for "3D controller: NVIDIA Corporation Device",
    which answers a narrower question than its callers ask -- "is there an
    NVIDIA part on this host" rather than "is there a device I can launch on"
    -- and answers it wrong for anything that is not NVIDIA.  A MetaX MACA part
    enumerates as `Display controller: Device 9999:4001`, so every MACA host
    reported CPU_ONLY and `bench()` raised `Unknown platform`.  It also threw
    outright where `lspci` is absent, as in a container without pciutils.

    On an NVIDIA host the two agree: a 3D controller in lspci implies a driver
    that can see it, which is what `is_available()` reports.  They can differ
    when the device exists but is hidden from this process
    (`CUDA_VISIBLE_DEVICES=""`), and there torch is the more useful answer --
    the callers of this function launch kernels, and no visible device means
    they cannot.

    Cached because device visibility is a property of the process, fixed at
    first query.
    """
    return Platform.CUDA if torch.cuda.is_available() else Platform.CPU_ONLY


def assert_current_platform(target_platform: Platform | list[Platform]):
    """
    Assert that the current platform matches the expected platform(s).

    Args:
        target_platform: A single Platform or a list of Platforms to check against.

    Raises:
        RuntimeError: If the current platform is not in the target list.
    """
    if isinstance(target_platform, Platform):
        target_platform = [target_platform]
    current_platform = get_current_platform()
    if current_platform not in target_platform:
        raise RuntimeError(
            f"Current platform is {current_platform.value}, but expected {[p.value for p in target_platform]}"
        )


def requires_platform(target_platform: Platform | list[Platform]):
    """
    Decorator that ensures the current platform matches before the function is called.

    Args:
        target_platform: A single Platform or a list of Platforms that are allowed.

    Raises:
        RuntimeError: If the current platform does not match.
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            assert_current_platform(target_platform)
            return func(*args, **kwargs)
        return wrapper
    return decorator


def is_on_cuda_platform() -> bool:
    return get_current_platform() == Platform.CUDA


def is_on_cpu_only_platform() -> bool:
    return get_current_platform() == Platform.CPU_ONLY
