// The gate, in one TU.
//
// `f32_coarse12_applies` lives inside `maca_topk.cu`'s `detail` namespace and
// needs a `RowParams` to be called, so a driver cannot reach it without the
// whole translation unit.  What it *is* -- and all this file exists to make
// checkable from outside -- is the predicate's own decision table, re-stated
// from the shipping source so a reader can diff the two.
//
// **Three revisions, and the last one is the interesting one.**
//
// It used to compare `sm_count` against the literal 104, which is a fact about
// the *C500*.  It then compared the device's
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` -- passed in as an argument --
// against what the kernel asks for, which is a fact about *shared memory*.
// The two agreed on every device anyone had looked at, and the agreement was
// the entire argument for the first form.
//
// Now the budget is neither a device query nor an argument: it is
// `ARCH_SMEM_PER_AP_BYTES`, a compile-time constant of the family the artifact
// was built for (`csrc/structs.h`), because there is one artifact per family.
// That makes this file's job *smaller* and its claim *stronger*.  Smaller: the
// predicate takes no budget, so it is a function of the shape alone.  Stronger:
// on a family whose constant cannot hold the arena the `if constexpr` is false
// for every shape, and the route does not exist in that image at all -- not a
// runtime test that a device could talk its way past, a property of the build.
//
// So the driver's job changes with it.  It no longer feeds the device's report
// *into* the predicate; it prints the report *beside* the constant the
// predicate used, because the interesting failure is now a mismatch between
// the two -- a C500 image loaded on a part with a smaller arena would route
// with a budget it does not have.  `_binding.family_suffix()` is what keeps
// that from happening, and this driver is where the numbers it decides on are
// visible.
#include <cstdint>
#include <cstddef>

namespace rk {
namespace dg12 {

constexpr int kThreads = 640;
constexpr int kMaxTopK = 2048;
constexpr int kCoarseBits = 12;
constexpr int kCoarseBins = 1 << kCoarseBits;
constexpr size_t kSmemBytes = 16 * 1024;
constexpr int kCandidateCapacity = (int)(kSmemBytes / (2 * sizeof(int32_t)));

}  // namespace dg12

// The split's own dynamic request, from `kF32SmemInputSize` at the default
// `kSMEM`.  Stated rather than derived because this file must not include the
// kernel tree -- see the header comment.
constexpr size_t kF32RowSmemBytesRef = 2 * 1757 * sizeof(uint32_t);   // 14056

}  // namespace rk

// ── the compile-time half ───────────────────────────────────────────────────
//
// These two are `csrc/structs.h`'s family-1000 row, transcribed.  A `-DDEEP_SELECT_ARCH=`
// build is what selects the real ones; this file is built by `ref/build_all.sh`
// with no `-D` at all, so it states the C500 row it is a driver for rather
// than including a header that would `#error` without the macro.
static constexpr uint32_t kArchSmemPerAp = 64 * 1024;   // ARCH_SMEM_PER_AP_BYTES
static constexpr uint32_t kArchSmCount = 104;           // ARCH_SM_COUNT

// C500's `cudaDevAttrMaxSharedMemoryPerBlockOptin`, read off the device by
// `main.cu` and printed there.  Kept as a *separate* constant from the compile-
// time one on purpose: they are the same number by construction, and this file
// is where a reader can see that the two sources agree rather than having to
// trust that they do.
static constexpr uint32_t kC500SmemPerAp = 65536;

// The AP count the ladder was measured on -- a *provenance* constant, distinct
// from the geometry above.  The shipping predicate carries both halves: the
// budget protects a device from the kernel, this protects it from the tuning.
static constexpr uint32_t kF32Coarse12MeasuredSmCount = 104;

// The budget half, as the shipping file writes it now: a comparison of two
// compile-time constants, decided by the compiler.  `kNeeded` is the larger of
// the route's request and the split's, because which one runs is what the
// predicate is deciding.  Exposed so `main.cu` can print it and so the
// `if constexpr`'s truth value is a symbol rather than a claim in a comment.
static constexpr uint32_t kNeeded =
    (uint32_t)(rk::dg12::kSmemBytes > rk::kF32RowSmemBytesRef
                   ? rk::dg12::kSmemBytes
                   : rk::kF32RowSmemBytesRef);
static constexpr bool kBudgetHalf = kArchSmemPerAp >= kNeeded;

// `f32_coarse12_applies`, re-stated.  **No budget argument**, because the
// shipping predicate has none: the budget half is the `if constexpr` above and
// is therefore the same answer for every call in this image.  What is left is
// the performance half, in the order the shipping file writes it, so a diff
// against it is a diff of two predicates and not of two shapes.
extern "C" bool ref_c500_gate(uint32_t sm_count, uint32_t batches,
                              uint32_t vocab_size, uint32_t topk)
{
    if (!kBudgetHalf) return false;
    if (sm_count != kF32Coarse12MeasuredSmCount) return false;
    if (topk > (uint32_t)rk::dg12::kMaxTopK) return false;
    if (batches == 0) return false;
    if (vocab_size < 2048) return false;                 // the original's bound
    if (vocab_size <= 131072) return batches >= 16;      // narrow arm
    return (uint64_t)batches * topk >= 114688;           // wide arm, the product
}

extern "C" int ref_candidate_capacity() { return rk::dg12::kCandidateCapacity; }
extern "C" int ref_smem_bytes() { return (int)rk::dg12::kSmemBytes; }
extern "C" int ref_row_smem_bytes() { return (int)rk::kF32RowSmemBytesRef; }
extern "C" int ref_c500_smem_per_ap() { return (int)kC500SmemPerAp; }
extern "C" int ref_measured_sm_count() { return (int)kF32Coarse12MeasuredSmCount; }
extern "C" int ref_arch_smem_per_ap() { return (int)kArchSmemPerAp; }
extern "C" int ref_arch_sm_count() { return (int)kArchSmCount; }
extern "C" int ref_needed_smem() { return (int)kNeeded; }
extern "C" int ref_budget_half() { return kBudgetHalf ? 1 : 0; }
