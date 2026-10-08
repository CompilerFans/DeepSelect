#pragma once

#include "kerutils/common/common.h"

#ifdef KERUTILS_IS_BUILD_ON_CUDA
#include "cuda/common.h"
#include "cuda/sm80/intrinsics.cuh"
#include "cuda/sm80/helpers.cuh"
#include "cuda/sm90/intrinsics.cuh"
#include "cuda/sm100/intrinsics.cuh"
#endif

// [MACA] A narrower arm than the CUDA one above, and the narrowing is the
// point: the `cuda/sm*` headers it does not list are inline PTX (`cp.async`,
// TMA, UMMA, cluster barriers, `st_async`) written for CUDA's assembler, and
// mxcc takes MACA ISA only -- the 64-bit operand constraints in them do not
// even parse (`invalid constraint`).  What the shared device headers call out
// of that set (`st_shared`, `trap`, `canonical_warp_idx_sync`) is implemented
// in `maca/common.h`, which is the file to extend rather than this list.
//
// **Adding a `cuda/sm*` header here is not a port, and two of the reasons are
// counters to the obvious argument.**  Those headers depend on cutlass, which
// this platform has no include for (the toolkit's copy is `mctlass/`); and
// `cuda/common.h`'s `KERUTILS_ENABLE_SM80/90/100` gates -- which is what would
// pull them in -- test `__CUDA_ARCH__`, which is **defined here as 800**, not
// absent.  So the gates would fire and re-enable exactly the PTX that does not
// assemble.  `__CUDA_ARCH__=800` is a compatibility value cucc passes
// (`-Xdevice -D__CUDA_ARCH__=800`), not a statement about this part: the
// family is `DEEP_SELECT_ARCH` on the host side and `__MACA_ARCH__` in the
// device pass, and **no MACA code may branch on `__CUDA_ARCH__`**.
#ifdef KERUTILS_IS_BUILD_ON_MACA
#include "maca/common.h"
#endif

#ifdef KERUTILS_IS_BUILD_ON_ASCEND
#include "ascend/common.h"
#endif
