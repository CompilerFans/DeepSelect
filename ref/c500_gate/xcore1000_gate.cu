// The gate, in one TU.
//
// `f32_coarse12_applies` lives inside `maca_topk.cu`'s `detail` namespace and
// needs a `RowParams` to be called, so a driver cannot reach it without the
// whole translation unit.  What it *is* -- and all this file exists to make
// checkable from outside -- is the two constants the dispatch compares
// `sm_count` against, and the arena bound that makes the comparison an
// equality rather than a threshold.  They are re-stated here rather than
// included for the same reason `ref/c500_gate/main.cu` prints the device's own
// numbers: the claim is "C500 is 104 and the arena that matters is 16 KB", and
// a claim about two numbers is checked by looking at the numbers.
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
}  // namespace rk

// The discriminator the shipping file uses; kept here as the same literal so a
// reader can diff the two lines.
static constexpr uint32_t kC500SmCount = 104;
static constexpr uint32_t kF32Coarse12MinBatches = 128;
static constexpr uint32_t kF32Coarse12MinVocab = 262144;

extern "C" bool ref_c500_gate(uint32_t sm_count, uint32_t batches,
                              uint32_t vocab_size, uint32_t topk)
{
    if (sm_count != kC500SmCount) return false;
    if (topk > (uint32_t)rk::dg12::kMaxTopK) return false;
    if (batches == 0) return false;
    if (batches < kF32Coarse12MinBatches) return false;
    if (vocab_size < kF32Coarse12MinVocab) return false;
    return true;
}

extern "C" int ref_candidate_capacity() { return rk::dg12::kCandidateCapacity; }
extern "C" int ref_smem_bytes() { return (int)rk::dg12::kSmemBytes; }
