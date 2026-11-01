// The host-side primitive every dataflow's launcher shares, one definition.
//
// `primitives.cuh` is the device half of the shared layer: bit patterns, the
// key encode, the warp histogram scan -- all integer-only and indifferent to
// the platform.  This is the host half, and it is the only one: launching a
// kernel that stages in dynamic shared memory needs the driver's opt-in, and
// four dataflows (the radix rows, the split's stages, the coarse12 selector)
// plus the contract half all need it.  It lives in the neutral namespace so a
// dataflow header does not have to reach across into another dataflow's for a
// helper that has nothing to do with that kernel.
//
// The namespace is the neutral one (`topk_select_common`), beside the device
// primitives, and the file boundary is host vs device rather than a taste:
// a device primitive is compiled twice (host and device pass) and must stay
// integer-only, while this one calls the runtime and is compiled once.

#pragma once

#include <cstddef>
#include <cuda_runtime.h>

namespace topk_select_common {

// ── host-side kernel configuration ─────────────────────────────────────────
//
// Every launcher that stages in dynamic shared memory opts its kernel in to
// that much before the first launch.  The opt-in is per function, so it is
// done once and the launcher's `static bool` is what keeps
// `cudaFuncSetAttribute` off the per-call path.
//
// It is not a no-op in the other direction either.  On this platform the
// default ceiling is the whole per-AP shared memory (131072 B on a 128 KiB
// part), so a call *lowers* the function's limit to the value it asks for.
// Each caller therefore asks for the largest block its own kernel is ever
// launched with, which is what makes the value it installs the right one;
// asking for a per-call size instead would fail the next, larger call.
//
// A failed adoption leaves `adopted` false, so the next launch retries rather
// than launching against a ceiling the driver never accepted.  The flag is one
// per kernel (per template instantiation); the operator's scratch already binds
// a process to one device, so a once-per-process flag has the same lifetime as
// the configuration it records.
template <typename KernelT>
inline cudaError_t adopt_dynamic_smem(KernelT kernel, size_t bytes,
                                      bool &adopted) {
    if (adopted) return cudaSuccess;
    const cudaError_t err = cudaFuncSetAttribute(
        (const void *)kernel, cudaFuncAttributeMaxDynamicSharedMemorySize,
        (int)bytes);
    if (err == cudaSuccess) adopted = true;
    return err;
}

}  // namespace topk_select_common
