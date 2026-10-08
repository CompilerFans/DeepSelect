#pragma once

#include "kerutils/common/common.h"

#include "host/host.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "device/device.cuh"
#endif

// [MACA] `host/host.h` above is unconditional and this arm needs what is in it
// -- `launch_kernel` and `KernelLaunchConfig` are how the ported kernels under
// `csrc/maca_kernels/xcore1600/` get launched, and that header's body is under
// the platform macro rather than under an unconditional one.  `device/` is the
// half that has to be named per platform, because it is what pulls in a
// platform's headers.
#ifdef KERUTILS_IS_BUILD_ON_MACA
#include "device/device.cuh"
#endif
