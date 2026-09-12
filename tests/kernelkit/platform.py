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


# [MACA] The architecture checks below.  `get_current_platform` used to grep
# `lspci` for an NVIDIA signature; it now asks torch, which is the port's one
# edit to that function (the reasoning is on it).  These predicates are what a
# caller that wants to *name* the architecture should use instead of reaching
# for `Platform`: the enum answers "can I launch here", and a MACA host and an
# NVIDIA host are both `CUDA` to it, which is correct for `requires_platform`
# and useless for anything that has to branch on the hardware.
#
# The key is the same one the port uses everywhere else -- the sm pair the
# device reports, which is what `deep_select/_arch.py::FAMILY_OF_SM` and the
# C++ side's `DeviceCapability::from_mc_arch` both key on.  A device *name* is
# deliberately not used: the parts spell themselves inconsistently across SDK
# generations (`MetaX C600U`, `MetaX C600-UL`, ...), which is exactly the
# reason `_arch.py` keeps a family base as the stable key and treats the
# spellings as aliases.
#
# This table mirrors `deep_select/_arch.py::FAMILY_OF_SM`, and the two are kept
# in sync by hand: `tests/` is the suite and `deep_select/` is the installed
# package, so this file does not import it (a test helper importing the package
# under test would also make `setup.py`'s own `import tests.kernelkit` circular
# on a tree where the extension is not built yet).  A change to either belongs
# in the same review.
#
# An unrecognized sm answers 0 rather than guessing a family -- the same rule
# `family_of_target` applies to an unknown compiler target, which raises there
# because a build has to stop; a test helper that only classifies should not.
MACA_FAMILY_OF_SM = {
    80: 1000,       # C500
    86: 1500,       # C600
    87: 1600,       # C600U / C600-UL, newest SDK
    88: 1600,       # C600U / C600-UL
    89: 1600,       # C600U / C600-UL, legacy SDK
}


def device_arch() -> int:
    """The current device's sm number, e.g. 89, or 0 when there is no device."""
    if not torch.cuda.is_available():
        return 0
    major, minor = torch.cuda.get_device_capability()
    return major * 10 + minor


def maca_family() -> int:
    """The xcore family base of the current device (1000/1500/1600), or 0.

    0 means either "no device" or "an sm this table does not know" -- use
    `is_maca_device` when the two cases need telling apart.
    """
    return MACA_FAMILY_OF_SM.get(device_arch(), 0)


def is_maca_device() -> bool:
    """Whether the current device is a MACA part this suite knows how to name."""
    return maca_family() != 0


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
