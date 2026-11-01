// ── which family the device in front of the call is ─────────────────────────
//
// `csrc/structs.h` answers "which family was this artifact built for" from
// `DEEP_SELECT_ARCH`.  This header answers the other question, "which family is
// the device", and routing asks it: the same artifact must select for the part
// it is handed, and a family is a property of the part rather than of the image.
//
// **A family is keyed by the sm pair the part reports**, which is also what the
// loader matches on (`deep_select/_binding.py`'s `device_family_suffix`).  One
// family's part renumbers across toolkit generations, so the family and not the
// number is the anchor.  The rows live here and there, and the two must agree:
// a device the loader recognizes and this header does not would load an
// artifact and then decline every family ladder.
//
// **Identity and AP count are two questions, and neither is derived from the
// other.**  A family can ship at more than one AP count, so `family()` reads
// the sm pair and `ap_count()` reads the AP count; a ladder selects on the
// first and is tuned with the second.
#ifndef FAMILY_CUH
#define FAMILY_CUH

#include <cstdint>

#include <cuda_runtime.h>

namespace deep_select {

// The families this tree builds one artifact per.  `Unknown` is what a device
// whose sm pair has no row answers: every family ladder then declines and the
// generic route answers the call, which is the same arm a device no ladder was
// fitted for takes.
enum class Family { Xcore1000, Xcore1500, Xcore1600, Unknown };

namespace detail {

inline Family family_of_sm(int sm) {
    switch (sm) {
    case 80:
        return Family::Xcore1000;
    case 86:
        return Family::Xcore1500;
    case 87:
    case 88:
    case 89:
        return Family::Xcore1600;
    default:
        return Family::Unknown;
    }
}

}  // namespace detail

// The current device's family, resolved once per process.  The artifact a
// process loads is chosen per family (`_binding.py`), so a process that changes
// device family is already running an image built for another one.
inline Family family() {
    static const Family f = [] {
        int device = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return Family::Unknown;
        int major = 0;
        int minor = 0;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   device) != cudaSuccess)
            return Family::Unknown;
        if (cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                                   device) != cudaSuccess)
            return Family::Unknown;
        return detail::family_of_sm(major * 10 + minor);
    }();
    return f;
}

// The current device's AP count: the input a family ladder is tuned against,
// and nothing else.  Grids are sized from the caller's `sm_count`, which is the
// device the *call* is on, and a family is `family()`'s answer.  A query that
// fails answers 0, which no ladder admits.
inline uint32_t ap_count() {
    static const uint32_t n = [] {
        int device = 0;
        int value = 0;
        if (cudaGetDevice(&device) != cudaSuccess) return 0u;
        if (cudaDeviceGetAttribute(&value, cudaDevAttrMultiProcessorCount,
                                   device) != cudaSuccess)
            return 0u;
        return (uint32_t)value;
    }();
    return n;
}

}  // namespace deep_select

#endif  // FAMILY_CUH
