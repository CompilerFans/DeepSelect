// The gate, in one TU.
//
// `f32_coarse12_applies` lives inside `maca_topk.cu`'s `detail` namespace and
// needs a `RowParams` to be called, so a driver cannot reach it without the
// whole translation unit.  What it *is* -- and all this file exists to make
// checkable from outside -- is the predicate's own decision table, re-stated
// from the shipping source so a reader can diff the two.
//
// The shape of that table changed on 2026-09-17, and the change is why this
// file has a second revision.  It used to compare `sm_count` against the
// literal 104, which is a fact about the *C500*; it now compares the device's
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` against what the kernel asks for,
// which is a fact about *shared memory*.  The two agreed on every device
// anyone had looked at, and the agreement was the entire argument for the old
// form.  A named family cannot answer a budget question, so the claim this
// file makes is three numbers rather than two: the route's request, the
// split's own request that it replaces, and the C500's budget.
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

// C500's `cudaDevAttrMaxSharedMemoryPerBlockOptin`, read off the device by
// `main.cu` and printed there.  Named here as the value the threshold is
// checked against.
static constexpr uint32_t kC500SmemPerAp = 65536;

// The AP count the ladder was measured on -- a *provenance* constant, distinct
// from the budget above.  The shipping predicate carries both halves: the
// budget protects a device from the kernel, this protects it from the tuning.
static constexpr uint32_t kF32Coarse12MeasuredSmCount = 104;

// `f32_coarse12_applies`, re-stated.  Both halves are here, in the order the
// shipping file writes them, so a diff against it is a diff of two predicates
// and not of two shapes.
extern "C" bool ref_c500_gate(uint32_t smem_per_ap, uint32_t sm_count,
                              uint32_t batches, uint32_t vocab_size,
                              uint32_t topk)
{
    const uint32_t needed =
        (uint32_t)(rk::dg12::kSmemBytes > rk::kF32RowSmemBytesRef
                       ? rk::dg12::kSmemBytes
                       : rk::kF32RowSmemBytesRef);
    if (smem_per_ap < needed) return false;
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
