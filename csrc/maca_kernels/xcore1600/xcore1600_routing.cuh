// ── the xcore1600 routing ladder ────────────────────────────────────────────
//
// This family's routing floors, and the gates composed from them.
// `common/routing.cuh` reaches them only after `family() == Family::Xcore1600`;
// every other device keeps the generic ladder.
//
// **The AP count is a parameter, not the identity test.**  The family ships at
// more than one AP count, so the floors below are chosen by the device's count;
// that is where a re-fit on a single part lands.
#ifndef XCORE1600_ROUTING_CUH
#define XCORE1600_ROUTING_CUH

#include <cstdint>

#include <csrc/maca_kernels/common/family.cuh>

namespace deep_select {
namespace detail {

// ── the narrow row body on a long row ───────────────────────────────────────
//
// `needs_long_row_bf16` sends every long row to the 1024-thread instantiation,
// and its crossover belongs to the 104-AP part.  On this family the narrow row
// takes over once the batch alone is large: the AP holds 2 x 1024- or
// 4 x 512-thread CTAs, so a full wave gives the narrow row four independent
// CTAs against the wide row's two, while below a wave the wide row's extra
// threads on a lone row win.  The batch at which the two cross climbs with the
// row length, so no one batch crosses every row; 256 is the smallest batch at
// which the narrow row wins **every** cell, and below it the wide row is kept
// because that is where the narrow row's losses are.
inline constexpr uint32_t k_xcore1600_narrow_row_batches = 256;

// ── the 16-bit split's crossing ─────────────────────────────────────────────
//
// The generic crossing is a line through the origin (`vocab / kChunkedVocabPerRow`
// rows), and on this family no line of that shape holds the crossings: they
// grow about half as fast as the rows do.  The square-root form
// `batches^2 * 6400 <= vocab` holds them, and its worst regret is the boundary
// cell itself.  The benchmark grid has no cell in the band the rule moves --
// its batches below 64 are 1 and 6, which both rules send to the split -- so
// this is a routing rule for decode shapes, not a grid number.
inline constexpr uint32_t k_xcore1600_chunked_vocab_per_batch_sq = 6400;

// ── the coarse12 floors ─────────────────────────────────────────────────────
//
// The generic ladder's performance half refuses any device that is not the one
// it was fitted on, so the route never ran here, and these floors are what it
// does when it can.
//
// **Both arms are a product `batches * topk` here, not a batch.**  On the
// generic ladder the narrow arm is a flat `batches >= 16` and the wide one is a
// product; here the narrow crossing moves with `topk` as plainly as the wide one
// does, and a product holds all three arms.  A flat 16 is wrong in *both*
// directions on this part: it declines a band of wins at the long `topk` and it
// enters the loss beside them at the short one.
//
// The two floors are sited on the crossings and are conservative at the low
// end: a product cannot hold a crossing that moves with `topk` there, so each
// declines a band of wins rather than risk the loss beside it.  The narrow
// floor is the `topk = 512` crossing and the wide one the `topk = 2048`
// crossing.
//
// `DEEP_SELECT_F32_COARSE12` is how a probe turns this ladder into a point on a
// single shape, but it is read *after* the family test in
// `common/routing.cuh`, so a probe has to be on a device in this family.
inline constexpr uint64_t k_f32_coarse12_xcore1600_narrow_product = 9216;
inline constexpr uint64_t k_f32_coarse12_xcore1600_wide_product = 49152;

// ── the floors, selected by AP count ────────────────────────────────────────
//
// One value serves both of the family's AP counts: the floors are fitted on the
// 28-AP part and the 32-AP part is served by the same numbers rather than
// declined, because both routes the floors sit between are one CTA per row and
// the per-row costs the crossing is made of do not move with the AP count.
// `ap_count` is where a separate fit on one part lands.
inline uint32_t xcore1600_narrow_row_batches(uint32_t ap_count) {
    (void)ap_count;   // one floor serves both of the family's AP counts
    return k_xcore1600_narrow_row_batches;
}

inline uint32_t xcore1600_chunked_vocab_per_batch_sq(uint32_t ap_count) {
    (void)ap_count;   // one floor serves both of the family's AP counts
    return k_xcore1600_chunked_vocab_per_batch_sq;
}

inline uint64_t xcore1600_coarse12_narrow_product(uint32_t ap_count) {
    (void)ap_count;   // one floor serves both of the family's AP counts
    return k_f32_coarse12_xcore1600_narrow_product;
}

inline uint64_t xcore1600_coarse12_wide_product(uint32_t ap_count) {
    (void)ap_count;   // one floor serves both of the family's AP counts
    return k_f32_coarse12_xcore1600_wide_product;
}

// ── the gates `common/routing.cuh` composes ─────────────────────────────────
//
// Whether the narrow (1024-thread) row body answers a long row at this batch.
inline bool xcore1600_narrow_row_applies(uint32_t ap_count, bool long_row,
                                         uint32_t batches) {
    return long_row && batches >= xcore1600_narrow_row_batches(ap_count);
}

// Whether the 16-bit split's crossing admits this shape.  `batches` is at most
// `kChunkedMaxBatches` here, so the square cannot overflow.
inline bool xcore1600_chunked_bf16_applies(uint32_t ap_count, uint32_t batches,
                                           uint32_t vocab_size) {
    return batches * batches <=
           vocab_size / xcore1600_chunked_vocab_per_batch_sq(ap_count);
}

// Whether the coarse12 floors admit this shape.  `narrow_vocab` is the width
// boundary both ladders split on, which the generic ladder owns.
inline bool xcore1600_coarse12_applies(uint32_t ap_count, uint32_t batches,
                                       uint32_t topk, uint32_t vocab_size,
                                       uint32_t narrow_vocab) {
    const uint64_t product = (uint64_t)batches * topk;
    return product >= (vocab_size <= narrow_vocab
                           ? xcore1600_coarse12_narrow_product(ap_count)
                           : xcore1600_coarse12_wide_product(ap_count));
}

}  // namespace detail
}  // namespace deep_select

#endif  // XCORE1600_ROUTING_CUH
