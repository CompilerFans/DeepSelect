// Routing: every shape decision in one place -- which dataflow answers a call,
// how many chunks a split gets, and which block width the row kernel runs at.
// The predicates here are pure functions of the shape and the device's AP
// count; the ladder that reads them is `select_route` below, whose answer is
// one `Route` value, and each route's launch and emit is one `launch_typed_*`
// here -- the row paths' via `launch_typed_radix` (`contract.cuh`, where the
// emit lives) and the four staged ones via `launch_staged_or_row`.  The
// handoff therefore lives above the dataflows, not inside them: a dataflow
// header is the kernel and its launcher, nothing that knows about the
// contract half.  `topk_launch` switches on the `Route` value and does nothing
// else -- in particular it does not re-decide.
#ifndef ROUTING_CUH
#define ROUTING_CUH

#include <cstdint>
#include <cstdlib>

#include <csrc/structs.h>
#include <csrc/maca_kernels/common/contract.cuh>
#include <csrc/maca_kernels/common/topk_coarse12.cuh>
#include <csrc/maca_kernels/common/topk_chunks.cuh>

namespace deep_select {
namespace detail {

// ── the 16-bit row's skip tables ────────────────────────────────────────────
//
// The row kernel reads its row twice.  The second pass only acts on elements
// whose 12-bit bin is at or above the threshold -- a constant ~600 of them at
// k=512, regardless of the row's length -- so on a long row more than 95% of
// that pass is reading lines it does nothing with.  Pass 1 leaves each
// 64-element block's max bin in a summary; the collect compacts the blocks that
// can hold a candidate into a list and walks that instead.
//
// The gate is the row length, because the summary is 2 bytes per 64 elements
// (1.6% of the row, written and read back) and the kept-block *fraction* is
// what decides whether that pays: measured on C500, 55% kept at 65536 elements
// is already a 1.42x pass -2, 33% at 131072 is 2.0x, and 3.7% at 1048576 is
// 7.0x, while below this floor the fraction climbs towards all-of-them and the
// list is traffic on top of a walk that reads the row anyway.  Both tables are
// sized by `vocab_size`; the kernel's own switch (a density past three quarters
// falls back to walking the row) needs no host input.
//
// Measurements and the density sweep: docs/experiments/20261010_skim_rate/.
constexpr uint32_t kSkipMinVocab = 65536;

// The density half of the same gate: `vocab_size >= kSkipMinRowPerSlot * topk`
// bounds `k/V`, which is what the kept-block fraction `1 - (1 - k/V)^64` follows
// from. See the gate for the two measurements that bracket it.
constexpr uint32_t kSkipMinRowPerSlot = 128;

// The campaign's A/B switch for the tables above, read once per process:
// `DEEP_SELECT_BF16_SKIP=0` answers every call with the row walk, which is what
// a differential check and a paired A/B need -- one artifact, one environment
// variable, no second build.  Unset or non-zero means the tables are built,
// which is the shipping behavior.
inline bool bf16_skip_enabled() {
    static const bool v = [] {
        const char *s = std::getenv("DEEP_SELECT_BF16_SKIP");
        return s == nullptr || s[0] == '\0' || std::atoi(s) != 0;
    }();
    return v;
}

// **A set, not a count: the C600U ships at 28 APs and at 32, and both are
// ordinary parts.**  Every gate fitted on that family takes the set.
inline bool is_c600u_ap_count(uint32_t sm_count) {
    return sm_count == 28 || sm_count == 32;
}

// `needs_long_row_bf16` sends every long row to the 1024-thread instantiation,
// and its crossover is a C500 measurement -- its own comment says so.  On the
// C600U the narrow row takes over once the batch alone is large: the AP holds
// 2 x 1024- or 4 x 512-thread CTAs (128 KiB of registers, 2048 threads), so a
// full wave gives the narrow row four independent CTAs against the wide row's
// two, while below a wave the wide row's extra threads on a lone row win.
//
// Measured 2026-10-10 (the copy-and-point A/B the split's crossing used: each
// width forced in its own tree, three rounds alternating in one session, seeds
// pinned; ratio = narrow / wide, below 1.0 is the narrow row):
//
//     vocab    65536  131072  262144  524288  1048576
//     b 64      0.97   1.03    1.07    1.09    1.10
//     b 128     0.95   0.99    1.02    1.04    1.05
//     b 192     0.91   0.97    1.00    1.01    1.03
//     b 256     0.90   0.93    0.98    1.00    1.00
//     b 4096    0.83   0.87    0.93    0.96    0.97
//
// The batch at which the ratio passes 1.0 climbs with the row length (45 at
// 65536 through ~270 at 1048576), so no one batch crosses every row; 256 is
// the smallest measured batch at which the narrow row wins **every** cell
// (worst 1.004, best 0.826).  Below it the wide row is kept because that is
// where the narrow row's losses are (1.37 at `b1..b6`, 1.10 at `64x1048576`).
inline constexpr uint32_t kC600UNarrowRowBatches = 256;

// The ported dispatcher's long-row regime is the only thing that differs
// between the two block widths (`topk_select_radix::needs_long_row_bf16`); the batch is the
// launch's row count, one row per CTA.
inline int radix_block_for(uint32_t batches, uint32_t vocab_size, uint32_t topk,
                           uint32_t sm_count) {
    const topk_select_radix::TopKConfig cfg{(int)batches, (int)vocab_size, (int)topk};
    const bool long_row = topk_select_radix::needs_long_row_bf16(cfg);
    if (long_row && is_c600u_ap_count(sm_count) &&
        batches >= kC600UNarrowRowBatches) {
        return topk_select_radix::kBlockSize;
    }
    return long_row ? topk_select_radix::kLongRowBlockSize : topk_select_radix::kBlockSize;
}

// ── the chunked split, for rows too long for one CTA to carry ───────────────
//
// One row per CTA leaves the machine idle when the batch is small and the rows
// long: B=6 at L=1M is 6 CTAs for the whole device.  The split ranks each row
// across `chunked_chunks(params)` CTAs and merges.  It wins by 11x at B=6/L=1M and
// loses above ~256 rows, which is the batch bound.  Upstream's own gate
// (`needs_chunked`: L >= 1M) is the same idea with a higher floor; 262144 is
// where the row count is still the problem here.  Measurements in
// `docs/C500-radix-perf-ledger.zh.md`.
constexpr uint32_t kChunkedMaxBatches = 64;
constexpr uint32_t kChunkedMinVocab = 262144;

// Row length the split has to buy before it is worth a batch: the gate below
// admits `vocab_size / kChunkedVocabPerRow` rows.
//
// The split pays three kernels (two stages and the `preselected` row kernel),
// reads the input once where the row kernel walks it twice, and buys CTA
// parallelism for a row the row kernel would otherwise walk alone.  That stops
// paying once the batch alone keeps the machine busy -- and the batch at which
// it does grows with the row length, because the row kernel's cost is flat in
// the batch for as long as it fits one wave (measured: 96.0 -> 99.5 us over
// `b2..b32` at vocab 262144, and 280.4 -> 287.9 over `b12..b64` at 1048576).
//
// Measured crossing on C500, each route forced in its own copy-and-point arm
// so both are reachable at every batch, three rounds alternating in one session
// (`docs/experiments/20261009_c500_dsa_decode/`, `probe_crossing.py`):
//
//     vocab  262144   k512 b=19.4   k1024 b=17.7    (vocab/b 13523, 14803)
//     vocab  524288   k512 b=29.4   k1024 b=29.7    (vocab/b 17808, 17672)
//     vocab 1048576   k512 b=43.5   k1024 b=35.6    (vocab/b 24095, 29416)
//
// The crossing is not proportional to the row length, so no single line
// through the origin passes through all three, and 21000 is the constant with
// the best worst case: the boundary lands at 12 / 24 / 49, and the worst
// measured regret over the grid is 1.146 (vocab 524288, b=26 -- the row kernel
// there costs 14.6% more than the split would have).  20000 -- 13 / 26 / 52 --
// cost 1.249 at `vocab 1048576 k1024 b=52`, where the split runs 385 us
// against the row kernel's 308, and 17000 or below cost 1.47x there instead.
constexpr uint32_t kChunkedVocabPerRow = 21000;

// ── the same crossing on the C600U: the batch grows like sqrt(vocab) ────────
//
// The constant above is a line through the origin, and on the C600U no line of
// that shape holds the three crossings: they are ~6.7 / 9.3 / 12.7 batches at
// 262144 / 524288 / 1048576, growing about half as fast as the rows do, and
// the best line through the origin has a worst measured regret of 1.30 -- it
// admits `1048576 x b16`, where the split is that much slower.  The
// square-root form `batches^2 * 6400 <= vocab` holds all three to 0.7 batch
// (the boundary lands at 6 / 9 / 12) and its worst measured regret is 1.008,
// the boundary cell itself (`1048576 x b12`, k=1024).
//
// Measured 2026-10-10 on one 28-AP C600U, each route forced in its own tree,
// three rounds alternating in one session, seeds pinned; ratio = row / split,
// above 1.0 is the split (`docs/experiments/20261010_c600u_split_crossing/`):
//
//     vocab     262144     524288     1048576
//     b=2       1.57       --         --
//     b=4       1.26/1.32  1.60/1.63  --
//     b=6       1.12/1.17  1.39/1.41  1.83/1.66
//     b=8       0.90/0.94  1.08/1.11  1.38/1.26
//     b=10      0.79/0.83  0.95/0.98  1.20/1.11
//     b=12      0.72/0.76  0.86/0.89  1.09/0.99
//     b=16      0.57/0.61  0.67/0.70  0.84/0.77
//     b=24      0.42/0.46  0.49/0.52  0.61/0.56
//
// (k=512 / k=1024; the two agree to 1.5 batches, so the batch is the axis.)
// The official grid has no cell in the band the rule moves -- its batches
// below 64 are 1 and 6, which both rules send to the split -- so this is a
// routing fix for decode shapes, not a grid number.
constexpr uint32_t kC600UChunkedVocabPerBatchSq = 6400;

// The split is SM-count-sensitive: a grid of `kBatch * chunks` CTAs leaves
// `ctas mod SM` SMs idle unless it is a whole number of waves, and the same 16
// is a different fraction of a wave on a 104-AP C500, a 28-SM C600 and a
// 32-SM C600U.  `params.sm_count` (the caller's, from the device) makes that
// decidable at launch time -- and it is the *device's* number rather than
// `ARCH_SM_COUNT` on purpose, because a caller that hands this entry a tensor
// on another part should get a grid sized for the part in front of it, not for
// the family the artifact was compiled for.
//
// The rule: **keep the measured count where it already fills at least half of
// its last wave, and otherwise round up to a whole number of waves.**  The
// half-wave floor is what keeps C500 exactly where it was measured.
// The fp32 split's work target, derived rather than tabulated.
//
// The split exists to fill a machine a short batch leaves empty, so its size
// is a *machine* property and the only question the sweep had to answer is
// what multiple of the SM count it is.  On C500 that fit is 256 (2.46 x 104
// APs), rounded to 5/2.  A per-family table would be three rows of which one
// is measured; this is the one measured fact, written once.
//
// **It reads the device's count, not `ARCH_SM_COUNT`, and that is deliberate.**
// The artifact is per-family now, so the macro *could* be used here, and it
// would be wrong to: this is the one rule whose whole purpose is to fill the
// machine in front of the call, and a 1600 image reached by a caller with a
// 1000 tensor would then size the grid for 32 APs on a 104-AP part.  The
// family constants answer questions about the *artifact*; this answers a
// question about the *run*.
inline uint32_t f32_chunk_work_target(uint32_t sm_count) {
    return sm_count * 5 / 2;
}

inline int wave_filled_chunks(int base_chunks, int sm_count) {
    constexpr int kBatch = 6;                      // the split's gate is small
    const int kSms = sm_count;
    const int ctas = kBatch * base_chunks;
    const int last_wave = ctas % kSms;
    if (last_wave == 0 || last_wave * 2 >= kSms) return base_chunks;
    return ((ctas + kSms - 1) / kSms) * kSms / kBatch;
}

// [MACA] SM-count-sensitive.  C500's 104 APs are what 16 was measured against
// (b6 x 16 = 96 CTAs = 92% of the last wave, and the chunk sweep shows 16-32
// flat there); C600 and C600U round it by `wave_filled_chunks` -- the latter at
// whichever of its two AP counts the device reports.
inline int chunked_chunks(const RowParams &params) {
    return wave_filled_chunks(16, (int)params.sm_count);
}

struct ChunkedWorkspace {
    int32_t *candidate_indices;
    maca_bfloat16 *candidate_values;
    int32_t *merged;
};

// The split is 16-bit only.  `topk` 512 and 1024 are the static-k arms of the
// merge; 2048 is carried by the runtime-k arms (`launch_typed_chunked`'s
// default branch, already compiled) and is admitted on the 128 KiB parts only.
// Measured on a 28-AP C600U, the DSA decode cells `b1-v{262144,524288,1048576}
// -k2048` improve 24-60% over the row path and the k512 controls are unchanged
// (`docs/experiments/20261018_bf16_split_k2048/`).  C500 keeps the row path
// there: no C500 measurement stands behind the extension, and the per-chunk
// ranking at topk=2048 costs shared memory the 64 KiB part has less of.
inline bool chunked_bf16_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0 || batches > kChunkedMaxBatches) return false;
    if (params.vocab_size < kChunkedMinVocab) return false;
    if (params.topk == 2048) {
        if (!is_c600u_ap_count(params.sm_count)) return false;
    } else if (params.topk != 512 && params.topk != 1024) {
        return false;
    }
    // The C600U's crossing has its own shape -- see the table on
    // `kC600UChunkedVocabPerBatchSq`.  `batches` is <= 64 here, so the square
    // cannot overflow.
    if (is_c600u_ap_count(params.sm_count)) {
        return batches * batches <=
               params.vocab_size / kC600UChunkedVocabPerBatchSq;
    }
    // `kChunkedVocabPerRow` is exact and overflow-free as a division; the
    // product would need 64-bit arithmetic to say the same thing.
    return batches <= params.vocab_size / kChunkedVocabPerRow;
}

// The split's three regions, in the order `launch_typed_chunked` takes
// them: candidate columns, candidate values, merged answer.  One definition
// for the size and for the carve below -- a layout spelled twice is a layout
// that can disagree with itself, and the merge reads `candidate_values` with
// the vectorized row path, which needs that region 16-byte aligned
// (`bf16x8_is_aligned`; the scalar fallback covers it otherwise).  The seam is
// `chunks * topk * 4` bytes per row, a multiple of 2048 for every `topk` this
// split admits (512 and up), so an aligned arena keeps it aligned.
inline size_t chunked_values_offset(uint32_t batches, uint32_t topk,
                                    uint32_t chunks) {
    return (size_t)batches * chunks * topk * sizeof(int32_t);
}

inline size_t chunked_merged_offset(uint32_t batches, uint32_t topk,
                                    uint32_t chunks) {
    return chunked_values_offset(batches, topk, chunks) +
           (size_t)batches * chunks * topk * sizeof(maca_bfloat16);
}

inline size_t chunked_workspace_bytes(uint32_t batches, uint32_t topk,
                                      uint32_t chunks) {
    return chunked_merged_offset(batches, topk, chunks) +
           (size_t)batches * topk * sizeof(int32_t);
}

inline ChunkedWorkspace chunked_workspace(void *base, uint32_t batches,
                                          uint32_t topk, uint32_t chunks) {
    ChunkedWorkspace ws{};
    ws.candidate_indices = (int32_t *)base;
    ws.candidate_values = (maca_bfloat16 *)((char *)base +
        chunked_values_offset(batches, topk, chunks));
    ws.merged = (int32_t *)((char *)base +
        chunked_merged_offset(batches, topk, chunks));
    return ws;
}

// ── the fp32 split ──────────────────────────────────────────────────────────
//
// Same shape as the 16-bit split and the same gate, so the scratch and the
// `end_ptr` route are one code path.  The chunk count is a function of the
// batch because the two row regimes disagree: long rows prefer fewer chunks
// monotonically (hence the constant 2 above 64 rows), short rows want
// `largest power of two <= K / batches`, where K is the caller's
// `f32_chunk_work_target(params.sm_count)`.  Sweeps and caveats:
// `docs/C500-radix-perf-ledger.zh.md`.
//
// At an SM count other than the one measured, the large-batch arm's merge CTA
// per row becomes a much larger share of the waves.  The fix the sweep points
// at is dropping that CTA (`chunks + 1` -> `chunks`) -- a change to the merge
// launch, not a constant, so it is not made on arithmetic alone.
namespace {
// The work target is `f32_chunk_work_target(sm_count)` above -- `sm_count * 5
// / 2`, the 2.5 being the only fit ever measured (on C500).  The SM count is
// read where the split is sized, not cached in a module-level constant.
//
// `DEEP_SELECT_F32_CHUNK32` raises the ceiling from 16 to 32 for the b6/V=1M
// cell the V-sweep flagged (parity plan §17.3).  A ceiling, not a count, so
// b32/b64 -- already at their optimum per the sweep -- are unaffected.
inline bool f32_chunk_ceiling_32() {
    static const bool on = [] {
        const char *v = std::getenv("DEEP_SELECT_F32_CHUNK32");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}
inline constexpr int kF32ChunkCeiling = 16;
inline int f32_chunks_for(uint32_t batches, int ceiling, uint32_t work_target) {
    const uint32_t b = batches == 0 ? 1u : batches;
    int c = 2;
    while (c < ceiling && (uint64_t)(c * 2) * b <= work_target) c *= 2;
    return c;
}

// **The chunk count's own rule cannot see `topk`, and above 512 it is the
// wrong axis.**  Stage 2's merge reads `num_chunks * topk` candidates per row
// -- `candidate_stride` in `topk_select_radix::launch_f32_chunk_stage2` -- and
// `f32_chunks_for` sizes itself from the batch alone.  The batch rule is
// already at the optimum for `topk <= 512`; above that the merge weights the
// count and a `topk`-aware cap is what §8.6 added.  Small-batch tier only --
// above `kF32ChunksFewBatches` the count is the large-batch rule and never
// reaches this.
//
// §8.6's version of that cap pinned the count at 8 and was measured over a
// sweep that is now superseded: it was wrong at both ends.  What this one is
// bounded by is the **chunk length**, and every length measured wants a count
// that is a multiple of two -- so the count is written as the product of two
// independent doubled quantities rather than as one number.
//
// `kF32ChunkSplitCeiling` (32) bounds the second one and is a *smem* bound, not
// a fit.  The arena is `kF32SmemInputSize` = 1757 slots and the 8-bit coarse
// level's threshold bin is not `L / 256` wide but about `L / 58` -- the suffix
// count reaches `topk` several bins below the mean, so it stops in the wider
// tail of the distribution (ratio flat at 57.9 / 56.8 / 58.2 for `L` in
// {87384, 65536, 131072}).  A bin that does not fit falls through to
// `radix_topk_row_f32_rescan`, which re-walks the whole window once per key
// byte -- five passes instead of two.  So `L / 58 <= 1757`, i.e. **L <= 101906**.
// The largest power of two under that is 65536.  That is the whole constant:
// two geometry facts, no measured value.
//
// **There used to be a `topk`-aware cap here and it was wrong in both
// directions.**  It read `c_default * topk * 10 <= vocab` and returned 8 when
// that failed.  Measured, interleaved repeats, `kk.bench`, contract checked,
// b1 and b6, `k = 2048`, medians of 2-3 rounds (ledger §10.9):
//
//     V        library ran    optimum   delta     -> V/c     optimum's chunk
//     98304    8              8            0%       12288
//     114688   8              4        -17.2%       28672
//     131072   8              4        -17.2%       32768
//     196608   8              6         -7.6%       32768
//     229376   8              4         -8.4%       57344
//     262144   8              6         -4.1%       43690
//     327680   16             6        -31.7%       54613
//     393216   16             6        -24.7%       65536
//     458752   16             6        -26.3%       76459
//     524288   16             8        -18.8%       65536
//
// The cap fired far too little (it exempted V >= 327680 and left the batch
// rule's 16, worth -18.8% .. -31.7%) and, where it did fire, pinned the count
// at 8 -- which is above the optimum at every V below 131072 on this ladder.
// The 8 was not a measurement of this tier's optimum; it was the merge weight
// `topk` puts on it, and that is a different axis.  **This is not an artefact
// of the fused NaN scan**: the `DS_FUSE_NAN_OFF` cross (`ab_fuse_chunks.py`,
// b1/b6 at V = 65536, 129280) puts both arms' optimum at the same count, so
// §8.6's sweep was already pointing low on its own.
//
// **What the optimum actually is: where stage 1 and stage 2 balance.**  They
// move in opposite directions in `c` -- stage 1 falls (`V / c` per CTA, more
// CTAs to hide the latency) and stage 2 rises (`num_chunks * topk` candidates
// read per row, and it is one CTA per row either way).  Setting the two
// derivatives against each other gives `c = theta * sqrt(V / topk)`, and the
// ladder above is that curve with `theta = 1/2`:
//
//     V        0.5*sqrt(V/2048)   nearest even   measured optimum
//     98304    3.46               4              8   (4 is 6.4% off)
//     114688   3.74               4              4
//     131072   4.00               4              4
//     196608   4.90               4              6   (1.4% off)
//     229376   5.29               6              4   (2.3% off)
//     262144   5.66               6              6
//     327680   6.32               6              6
//     393216   6.93               6              6
//     458752   7.48               8              6   (1.0% off)
//     524288   8.00               8              8
//
// Worst cell 6.4%, mean under 1.5%, and the two structural instincts are both
// in it: the count grows as `sqrt(V)` (more chunks only pay while stage 1's
// per-CTA latency is what is hiding) and falls as `1/sqrt(topk)` (a bigger
// merge is more expensive per candidate read).
//
// `theta = 1/2` is the one fitted number and it is fitted on `topk = 2048`
// only -- the `topk` axis of the curve is *predicted*, not measured, so the
// formula is applied only where `topk > 512` (the same gate §8.6 used) and the
// result is a cap: the caller takes `min` with the batch rule, so a cell the
// formula would raise past its measured count is unaffected.
//
// **`kF32ChunkCurveFloor` is a discontinuity in the data, not in the model.**
// The curve predicts 4 at both `V = 98304` and `V = 114688`, but the measured
// optimum steps 8 -> 4 across that interval (`0.0972` against `0.1034` at the
// low end; `0.1438` against `0.1550` at the high end, and the stage split flips
// sign between them -- s1's slope dominates at 98304, s2's at 114688). Above
// the floor the curve is used; below it the count is the measured 8, which is
// what both cells below want. Between those two V there is no measurement, so
// the boundary is placed on the measured one.
//
// `work_target` is the batch rule's own cap (`c * b <= 2.5 * sm_count`): the
// split exists to fill a machine a short batch leaves empty, so at a large
// batch it has nothing to buy and must not be forced up.
inline constexpr uint32_t kF32ChunkSplitCeiling = 32;
inline constexpr uint32_t kF32ChunkCurveFloor = 114688;
// The curve is fitted on `k = 2048` only, and it is applied on that axis only.
// For `512 < topk < 2048` §8.6's measurements still stand and still disagree
// with the curve: at `V = 262144, k = 1024` the batch rule's own 16 measures
// 99.5 us against 138.4 us for the curve's 8, so extrapolating down the `topk`
// axis would replace a measured optimum with a predicted one.
inline constexpr uint32_t kF32ChunkCurveTopK = 2048;

inline uint32_t f32_isqrt(uint64_t v) {
    uint32_t r = 0;
    for (uint32_t bit = 1u << 30; bit != 0; bit >>= 1) {
        if ((uint64_t)(r + bit) * (r + bit) <= v) r += bit;
    }
    return r;
}

// Round a half-integer up to the next even value: the ladder's optima are all
// even (they come in `c` and `c/2` pairs of a 2-way split), and an odd count
// makes `chunk_size = ceil(L/c)` uneven, which the chunk geometry's 8-element
// rounding then amplifies.
inline int f32_chunk_round_even(uint32_t c) {
    if (c < 2u) return 2;
    return (c & 1u) ? (int)(c + 1u) : (int)c;
}

inline int f32_topk_chunk_cap(uint32_t topk, uint32_t vocab, uint32_t c_default,
                              uint32_t work_target) {
    if (topk <= 512) return 0;   // 0 = no cap; keeps `DEEP_SELECT_F32_CHUNK32` in force
    if (vocab >= kF32ChunkCurveFloor && topk == kF32ChunkCurveTopK) {
        // `c = 0.5 * sqrt(V / topk)`, in integers: `c = isqrt(V / (4 * topk))`.
        // Nothing else: the merge-share guard below is *not* allowed to exempt
        // a cell here, and that is the point of splitting the two cases.
        int c = f32_chunk_round_even(f32_isqrt((uint64_t)vocab / (4ull * topk)));
        if ((uint64_t)c > work_target) c = f32_chunk_round_even(work_target);
        if (c > (int)kF32ChunkSplitCeiling) c = (int)kF32ChunkSplitCeiling;
        return c;
    }
    // §8.6's rule for everything the curve does not cover.  The guard is
    // `c_default * topk * 10 <= vocab` -- the merge reading under a tenth of the
    // row is already at the batch rule's own count, so capping it is a pure
    // loss.  **The curve above had to be split out of this because the guard
    // fires exactly at the cells the ladder says the curve should decide**: at
    // `k = 2048` the product is `8 * 2048 * 10 = 163840` (cap in force) or
    // `16 * 2048 * 10 = 327680` (exempt), so every `V >= 327680` took the
    // `return 0` arm and ran the batch rule's 16.  That is why `rule_ab.py`
    // read `V = 393216 / 524288` at `-0.2% / +0.2%` while the ladder's optimum
    // there is worth `-24.7% / -18.8%` (in force 0.2859 / 0.2959 against
    // 0.2152 / 0.2414 at `c = 6 / 8`).
    if ((uint64_t)c_default * topk * 10 <= vocab) return 0;
    return 8;   // §8.6's measured cap for `512 < topk < 2048`
}

// ── the chunk count above `kF32ChunksFewBatches` ────────────────────────────
//
// This used to be the constant 2.  What bounds it is the chunk *length*, and
// the bound is a shared-memory one.
//
// `radix_topk_row_f32` stages the members of the coarse threshold bin into an
// arena of `kF32SmemInputSize` slots.  A bin that does not fit falls through to
// `radix_topk_row_f32_rescan`, which re-walks the whole window once per key byte
// with no early exit -- five passes over the chunk instead of two.  Measured
// directly: on `b128 V=524288 k=2048`, disabling that branch (the `if (false
// && ...)` probe) takes `c = 1` from 4.18 ms to 0.95 ms, and `c = 4` from
// 3.80 ms to 0.96 ms, while `c = 6` is unchanged at ~1.05 ms -- i.e. the whole
// cliff is the rescan, and it is gone once a chunk clears the arena.
//
// So the question is only: for a row of `V` split `c` ways, does a chunk of
// `V / c` keep its threshold bin under the arena?  The 8-bit coarse level has
// 256 bins, and on a randn row the threshold bin is not `L / 256` wide but
// about `L / 58` -- the suffix count reaches `topk` several bins below the
// mean, so the bin it stops at is the wider tail of the distribution.  The
// ratio is flat at 57.9 / 56.8 / 58.2 for `L` in {87384, 65536, 131072}, which
// is exactly the range this rule decides in (measured with numpy over seeded
// randn draws, the same distribution `tests/lib.py` generates).
//
//     overflow when  L / 58 > kF32SmemInputSize  =>  L > kF32OverflowChunkLen
//
// Both factors are device geometry (`kF32SmemInputSize` from `kSMEM`, `kRadix`)
// except the 58, which is the measurement.  The count is then the smallest `c`
// that clears it:
//
//     V = 131072 -> 1.29 -> 2      (unchanged; the sweep is flat here)
//     V = 262144 -> 2.57 -> 3      (measured optimum 3)
//     V = 524288 -> 5.14 -> 6      (measured optimum 6)
//
// and both measured optima land where the rule says they do, with no fitted
// boundary: `c = 5` at `V = 524288` would put every chunk 3% over the arena
// (104858 against 101906) and the merge pays for it, 5.77 ms against 0.22 ms.
//
// Measured on C500, `k = 2048`, `kk.bench`, contract checked, 3-5 repeats,
// min-of-medians:
//
//     V=524288  b256 7.26 -> 2.05 ms   b4096 90.1 -> 29.6 ms   (c=2 -> 6)
//     V=262144  b256 3.81 -> 1.12 ms   b4096 46.5 -> 15.1 ms   (c=2 -> 3)
//
// `V < 262144` is untouched, which is where the sweep has the whole gate
// losing to the row path anyway (`b4096-v65536` is flat at 3.62 ms for every
// `c`); that is the batch bound at `topk_worth_splitting_f32`, not this.
//
// **Provenance, not a device test.**  Both inputs to the rule are device
// geometry -- the arena and the AP count the grid is sized against -- and the
// two fitted numbers below were read off one machine.  C600/C600U disagree on
// both (28/32 APs against 104, and a different smem budget), so they keep
// `c = 2`.  `__MACA_ARCH__` looks like the obvious discriminator and is the
// wrong one: it is defined only in the *device* pass, and this is a host
// function, so every architecture would take the `#else` branch.
//
// This is the same shape as the coarse12 gate's provenance half: a constant
// naming *where a number was measured*, not what the device is.  It reads
// `sm_count` because that is the only machine identity this file has -- unlike
// the coarse12 gate, there is no budget question here to answer instead, since
// the rule's own first factor already carries the arena.
//
// The first factor is not written down: it is `kF32SmemInputSize`, the arena
// the kernel actually has, and transcribing its value here is what let the
// small-batch branch above go without it for so long.  Deriving it means a
// `-DKSMEM_BYTES=` build moves this bound with the arena instead of leaving it
// behind -- the two numbers agreed at 1757 only because nothing had changed
// `kSMEM`.  The second factor, 58, is the one measured quantity here and is
// the only one that stays a literal.
inline constexpr uint32_t kF32OverflowChunkLen = topk_select_radix::kF32SmemInputSize * 58u;
// The 58 was measured against the default arena, and `f32_chunks_large_batch`
// applies this bound to *every* `topk` -- so a build that moves `kSMEM` would
// carry the old ratio into a new arena without anything noticing.
//
// **The guard below cannot fire in any build this tree has.**  `KSMEM_BYTES`
// is undefined by default (`radix_common.cuh`'s `#ifndef KSMEM_BYTES` picks 16 KB
// for `__MACACC__` and 48 KB otherwise), and no entry in `setup.py`'s
// `build_for_maca`, no `CUCC_TARGETS` value and no script in this repository
// passes `-DKSMEM_BYTES=`.  So it is a trap for a build nobody runs, and the
// default build -- the one that ships -- has the assert compiled out.  It is
// kept because the day someone does pass the flag is exactly the day the 58
// stops being the measurement, and a silent wrong bound is worse than a
// compile error.  Making it fire would take an `#else` branch asserting the
// default `1757`, which is the same check written twice; the honest statement
// is that the default is unguarded and `radix_common.cuh`'s `kSMEM` is the one
// place to read what the arena is.
//
// **A `-DKSMEM_BYTES=` build cannot get that far anyway**, which is a stronger
// statement than "nobody passes it": `radix_common.cuh`'s
// `kCoarse12ArenaEntries` assert fails for both 32 KB and 48 KB, so the only
// value that compiles
// is the 16 KB default.  This assert is therefore unreachable twice over --
// once because nothing sets the macro, and once because setting it to anything
// else does not build.  Measured on this tree with `-fsyntax-only`, both
// passes: `16384` compiles, `32768` fails `'16384 >= 30440'`, `49152` fails
// `'16384 >= 46824'`.
#ifdef KSMEM_BYTES
static_assert(topk_select_radix::kF32SmemInputSize == 1757,
              "KSMEM_BYTES moved the arena, so kF32OverflowChunkLen's 58 is no "
              "longer the measured value: re-derive it against the new "
              "kF32SmemInputSize (the ladder is in the ledger, SS12.5) before "
              "shipping this build.");
#endif
// The AP count the coarse12 ladder was measured on, and the same kind of
// constant as the `sm_count` test just above -- a number naming a
// measurement's provenance, not a claim about what a device is.  There used to
// be a `kC500SmCount` here as well, for the coarse12 gate's device half; that
// half is now the budget comparison in `f32_coarse12_applies`, and the only
// question left that `sm_count` can answer is "was this the machine the
// numbers came from".  One constant, one job.
inline constexpr uint32_t kF32Coarse12MeasuredSmCount = 104;
inline int f32_chunks_large_batch(uint32_t vocab_size, uint32_t sm_count) {
    // Same shape as the coarse12 gate's provenance half, and the same reason:
    // the `58` below was read off this device.
    if (sm_count != kF32Coarse12MeasuredSmCount) return 2;
    // `c = 2` is the floor: `launch_f32_chunk_stage1` rejects
    // `num_chunks <= 1`, so fewer than two is not a split at all.
    const int c = (int)((vocab_size + kF32OverflowChunkLen - 1) / kF32OverflowChunkLen);
    return c < 2 ? 2 : c;
}

// The batch above which the chunk count stops being a parallelism knob: 64 is
// the split's own small-batch ceiling, a property of the gate rather than a
// fitted value.  At 6 rows the machine is empty and chunks are the only CTAs
// there are; at 256 rows the grid already covers the machine twice over.
inline constexpr uint32_t kF32ChunksFewBatches = 64;

// Where the coarse12 route takes over.  The batch floor is not one number,
// because the measured crossing is not one number: it moves with the width.
// A row walk is only worth the machine when there are enough CTAs to fill it,
// and how many that is depends on how much work each CTA has -- which is `V`.
// The ladder below is `k = 2048`, paired medians, `B/A < 1` meaning the route
// won; the ledger's §12 has the full table.
//
//      V        b16     b24     b32     b48     b64     b96    b128
//      8192    0.821   0.801   1.015   0.763   0.724     -      -
//     32768    0.630     -       -       -       -       -      -
//     65536    0.897     -       -       -       -       -      -
//    131072    0.910   0.908   0.839     -       -       -      -
//    196608    1.029     -     0.925   0.891   0.796     -      -
//    262144    1.182   1.092   1.048   1.042   0.915   0.813  0.672
//    393216    1.407     -     1.255   1.184   1.031     -      -
//    524288    1.525   1.469   1.319   0.277   0.253   0.840    -
//
// Two regimes fall out of it, and one constant cannot hold both.  At or below
// `V = 131072` the route is already ahead at `b16` and never measured behind.
// `196608` is the first width where `b16` loses (1.029), so `131072` is the
// boundary: everything at or under it takes the 16 floor, everything above
// takes 64.  The two widths between them are unmeasured and are deliberately
// left on the wide side, which is the conservative direction.
//
// Above the boundary the crossing sits between `b48` and `b64` at `262144` and
// above `b64` at `393216`, so 64 is the one value that is a win at every wide
// cell measured except `393216` (1.031 -- the flat part of the crossing, +3%).
// `524288` wants a floor near 36, but that cell is the split's own cliff --
// `b32` is 1.319 and `b40` is 0.277 against a baseline whose stage 1 jumps
// 0.22 ms to 1.68 ms between them -- so it is not a crossing to site a
// constant on, and the 64 floor takes `b64` there at -74.7%.
// The wide-width floor is a product, not a count, because the crossing moves
// with `topk` as well as with the width.  Measured crossing batch, `V = 262144`,
// paired medians (`B/A < 1` = the route won):
//
//     topk    crossing    batches * topk
//     2048    ~56         114688
//     1024    ~112        114688
//     512     ~95         48640   (and ~190 at V = 524288 -> 97280)
//
// The first two agree to the digit; `topk = 512` crosses earlier than the
// product would put it, so this floor is **conservative there by design** --
// it declines `b128` at `V = 262144 k = 512` (-12.2%, a win left on the table)
// in exchange for never entering the two cells where the route loses badly at
// that `topk` (`b64 V = 524288` is +55.7%, `b128 V = 524288` is +4.1%,
// `b64 V = 262144` is +17.6% -- all three excluded).
inline constexpr uint64_t kF32Coarse12WorkTarget = 114688;
inline constexpr uint32_t kF32Coarse12MinBatchesNarrow = 16;
inline constexpr uint32_t kF32Coarse12NarrowVocab = 131072;

// ── C600U's ladder: the same two arms, re-sited on the 28-AP part ────────────
//
// The C500 ladder above is a measurement on 104 APs, and on a C600U it is not
// merely conservative -- it is **unreachable**, because the performance half of
// `f32_coarse12_applies` refuses any device that is not the one it was measured
// on.  So the route has never run here, and what follows is what it does when
// it can.  Measured 2026-09-20 on this part (28 AP, 128 KiB/AP), the arms
// interleaved per round over the official data and timing rule, verdict =
// median of per-round ratios, `B/A < 1` meaning the route won:
//
//     narrow (V <= 131072)      k=512        k=1024       k=2048
//       b1                                   1.07         0.95
//       b2                                                0.88
//       b4                                                0.91
//       b8                     1.42          0.95         0.90
//       b9                                   1.07
//       b14                                  1.05
//       b15                                  0.96
//       b16                    1.08          0.96         0.77
//       b17                    1.08
//       b18                    0.84
//       b24                    0.83                        0.79
//       b32                    0.83                        0.75
//
//     wide (V > 131072)         k=512        k=1024       k=2048
//       b16                    1.46         1.23          1.11 (V=393216)
//       b18                                               1.00
//       b20                    1.11          --           0.90
//       b24                    1.03         0.16          0.89
//       b32                    1.02         0.16          0.88
//       b40                    1.01          --            --
//       b48                    0.90         0.15          0.78
//
// **Both arms are a product `batches * topk` here, not a batch.**  On C500 the
// narrow arm is a flat `batches >= 16` and the wide one is a product; here the
// narrow crossing moves with `topk` as plainly as the wide one does -- `k=2048`
// crosses at `b1`, `k=1024` at `b9`, `k=512` at `b18` -- and a product holds
// all three.  The C500 narrow arm's flat 16 is wrong in *both* directions on
// this part: it declines `b5..b15` at `k=2048` (0.88-0.95, every one a win)
// and it enters `b16..b17` at `k=512` (1.08, a loss).
//
// The two floors are sited on the measured crossings and are **conservative in
// exactly the way the C500 ladder's `topk = 512` arm is** -- a product cannot
// hold a crossing that moves with `topk` at the low end, so each declines a
// band of wins rather than risk the loss beside it:
//
//   * narrow, `9216`: the binding cell is `k = 512` at `b16..b17` (+8%), so the
//     floor sits at `b18` there.  `k = 1024` crosses at `b9` and the product
//     puts it at `b9` -- exact.  `k = 2048` crosses at `b1` and the product puts
//     it at `b5`, declining `b1..b4` (0.88-0.95).
//   * wide, `49152`: the binding cell is `k = 2048` at `b16` (+11% at
//     `V = 393216`, +21% at `524288`, +22% at `163840`), so the floor sits at
//     `b24` there.  `k = 512` crosses near `b40` (`V = 262144`) and the product
//     puts it at `b96`; the band between is a win at every cell measured except
//     `96 x 524288` (+7%, isolated -- 80 is 0.83 and 112 is 0.98).
//
// **The `k = 1024` `V = 262144` `b24..b48` column is not a crossing.**  It
// reads 0.15-0.16 because the *default* route falls into its rescan there
// (`c = 2` gives a 131072-element chunk against the 101906 bound, and the arena
// floor in `f32_chunked_chunks` is guarded to `topk == 2048`), which is the
// chunk curve's own hole rather than a property of this route.  Those cells are
// why the floor is not raised past `b24` at that `topk`; the healthy cell at
// the same `topk` is `b16` at 1.23, which is why it is not sited there either.
//
// What it is worth on the production shapes, official data, default route =
// 1.0: `4096 x 129280 k=512` 8789 -> 5456 us (0.62), `768 x 129280 k=512`
// 1702 -> 1076 (0.63), `512 x 66551 k=2048` 1138 -> 717 (0.63), `132 x 107520
// k=2048` 584 -> 391 (0.67).
// The ladder below was fitted on the 28-AP C600U -- the set
// `is_c600u_ap_count` names -- so the 32-AP arm is served by *extrapolation*,
// and this is where that says so.
//
// Two reasons it is extrapolated rather than declined.  The routes the floors
// sit between are **both one CTA per row**, so the per-row costs the crossing
// is made of do not move with the AP count -- and the step here is 1.14x,
// against the 3.7x (104 -> 28) this ladder was already re-sited across.  And
// declining is not the conservative arm: a 32-AP part then takes the row path
// on every shape the route serves, which on this host measures **1.61x** at
// `4096 x 129280 k=512` float32.  Measured 2026-10-08 on one binary by
// varying this entry's own `sm_count` argument at a fixed 28 -- 8832 us with
// `DEEP_SELECT_F32_COARSE12=0`, 5484 with `=1` -- so the two arms differ in
// the route and in nothing else this file decides.
//
// `DEEP_SELECT_F32_COARSE12` is how a 32-AP machine would turn the
// extrapolation into a measurement, but it is read *below* the AP test, so a
// probe has to be on a part in this set already.
inline constexpr uint64_t kF32Coarse12C600UNarrowProduct = 9216;
inline constexpr uint64_t kF32Coarse12C600UWideProduct = 49152;


// The width floor is the one value in this gate that is not a crossing at all.
// The port wins at every width the split is legal at -- `V = 4096` is -29.1%,
// `8192` is -45.9%, `16384` is -53.4%, and the `(V, k)` sweeps above that are
// all wins -- so there is nothing to site it on.  What sets it instead is the
// original's own bound: `select_topk_policy` never routes to coarse12 below
// `n_cols = 2049`.  Below that the split is the only arm anyone has measured,
// so the gate matches the original rather than inventing a floor.  (2048, not
// 2049: the facade's own stride guard already rejects anything that is not a
// multiple of 256 floats, and 2048 is the largest legal width under 2049.)
inline constexpr uint32_t kF32Coarse12MinVocab = 2048;

// ── the deep_gemm coarse12 route ────────────────────────────────────────────
//
// `topk_coarse12.cuh` carries `topk_coarse12`, extracted whole from
// `mcDeepGEMM/csrc/kernels/fp32_topk.cu`.  It answers one row in one CTA and
// writes *positions in the row*, which is exactly what the contract half below
// consumes, so it replaces the whole split for the shapes it serves rather than
// sitting beside it.
//
// **Budget-gated, and for once the reason is not a measurement.**
// The kernel's 16 KB arena holds a 4096-bin coarse histogram first and the
// candidate array second, so it needs 16 KB of dynamic shared memory.  That is
// `kF32RowSmemBytes`-sized on C500 and is not on a device with a smaller
// budget -- and unlike every other C500-only rule in this file, a smaller
// budget does not make it slower, it makes it **wrong**.
//
// **Two questions, and this predicate used to answer both with one number.**
//
// *Does the route fit?*  A budget question, and it is answered against
// `ARCH_SMEM_PER_AP_BYTES` -- this artifact's own family's budget, a
// compile-time constant (`csrc/structs.h`).  The predicate used to read
// `sm_count != 104`, which is a fact about the C500 that has nothing to do
// with shared memory; the two agreed on every device anyone had looked at, and
// that agreement was the whole argument for the old form.  In between it read
// `cudaDevAttrMaxSharedMemoryPerBlockOptin` at the entry point, which is the
// same number obtained one call later and from the wrong place: the artifact
// already knows which family it was built for, so asking the driver at launch
// time was a second source for a fact the build had.
//
// *Is the route faster here?*  A measurement, and this one is not adaptive --
// the crossing in the tables above is a C500 ladder, and applying it to a
// machine nobody measured would be a guess dressed as a rule.  So it is
// guarded by the device it was measured on, named for what it is.  **Since
// 2026-09-20 that is two devices and two ladders**, the 104-AP C500 one above
// and the C600U one beside it; the second exists because the route had never
// run on a C600U at all, so "unmeasured" was declining shapes it wins by up to
// 0.62x.  The C600U set carries both of that family's AP counts -- the 32-AP
// arm is an extrapolation and says so at its own definition.
//
// Splitting them is the point, and it still is under the macro: the budget
// half is now a *compile-time* property of the family, while the performance
// half stays a runtime property of the machine in front of the call.  A family
// 1600 image carries the 128 KiB budget without carrying the C500 ladder --
// the two halves move independently, which is the whole reason they are two
// halves.  An artifact whose `ARCH_SMEM_PER_AP_BYTES` is too small for the
// arena takes the `#if` and never routes at all, and that is a *build* fact
// now: it cannot be talked out of at runtime by a device that reports more.
//
// Why it is worth having at all: on the shapes deep_gemm routes here it is
// 2.0-2.3x faster than the row kernel (ledger §12).  The two measured
// differences are 12 coarse bits against 8 and one 16 KB arena against 14 KB
// plus a ping-pong refine buffer; the block size is the original's 512 as well
// (it was 640 until 2026-10-22, which left 128 of the AP's 2048 thread slots
// unused -- see `docs/experiments/20261022_f32_occupancy/`).
// `DEEP_SELECT_F32_COARSE12`: `1` forces the route onto every shape that is
// otherwise *legal* for it, `0` denies it, unset takes the ladder.  In the same
// shape as `DEEP_SELECT_F32_CHUNKS` below, and for the same reason: the ladder
// is a set of measured crossings, and re-siting one means measuring the route on
// the shapes the current floor declines -- which is otherwise a source edit and a
// three-artifact rebuild per probe.  It replaces the floor and **nothing else**:
// the arena budget, the `topk` bound and the width bound are still enforced, so
// a forced route is a legal route.  It does not change any default, and it
// cannot reach a family the ladder was not measured on -- the `sm_count` test
// above it runs first.  (It is what sited the C600U ladder: the knob is read
// *after* the AP-count test, so a probe needs that test relaxed to reach the
// arms at all -- see the note on `is_c600u_ap_count`.)
inline int f32_coarse12_override() {
    static const int v = [] {
        const char *s = std::getenv("DEEP_SELECT_F32_COARSE12");
        if (s == nullptr || s[0] == '\0') return -1;
        return std::atoi(s) != 0 ? 1 : 0;
    }();
    return v;
}

inline bool f32_coarse12_applies(const RowParams &params, uint32_t batches) {
    // ── the budget half: this family's compile-time budget, and the request ──
    // `kSmemBytes` is the route's whole request (histogram over candidates);
    // `kF32RowSmemBytes` is what the *split* it replaces needs.  Both have to
    // fit, because which one runs is what this predicate is deciding.
    //
    // `if constexpr`, not a runtime test: on a family whose budget cannot hold
    // the arena the answer is false for every shape, and the build says so
    // rather than every call re-deriving it.  It also keeps the comparison
    // honest -- an unsigned constant against an unsigned constant is decided
    // by the compiler, so a family that fails this half is a compile-time
    // fact visible in the generated code rather than a branch.
    constexpr uint32_t kNeeded =
        topk_select_coarse12::kSmemBytes > topk_select_radix::kF32RowSmemBytes
            ? (uint32_t)topk_select_coarse12::kSmemBytes
            : (uint32_t)topk_select_radix::kF32RowSmemBytes;
    if constexpr (ARCH_SMEM_PER_AP_BYTES < kNeeded) {
        return false;
    } else {
        // ── the performance half: where the ladder below was measured ──
        // The C600U is the exception to "a device the ladder was not measured
        // on": the ladder *was* re-measured on that family (2026-09-20, the
        // table above), so the test is which of the two ladders applies rather
        // than whether any does.  A part that is neither takes the row path, as
        // before.
        if (is_c600u_ap_count(params.sm_count)) {
            if (params.topk > (uint32_t)topk_select_coarse12::kMaxTopK) return false;
            if (batches == 0) return false;
            if (params.vocab_size < kF32Coarse12MinVocab) return false;
            if (f32_coarse12_override() >= 0) return f32_coarse12_override() != 0;
            const uint64_t product = (uint64_t)batches * params.topk;
            if (params.vocab_size <= kF32Coarse12NarrowVocab)
                return product >= kF32Coarse12C600UNarrowProduct;
            return product >= kF32Coarse12C600UWideProduct;
        }
        if (params.sm_count != kF32Coarse12MeasuredSmCount) return false;
        // The arena's bound, not a fitted one: `topk_coarse12_row` serves at
        // most what the candidate half can hold, and there is no overflow path
        // in the *staged* refine that can make a larger answer correct.
        if (params.topk > (uint32_t)topk_select_coarse12::kMaxTopK) return false;
        if (batches == 0) return false;
        // The original's own width bound (`n_cols >= 2049`).  It matters more
        // now than it did: the narrow arm below is `return batches >= 16` with
        // no other `V` test, so without this line a 1024-wide row would route,
        // which is below every width anyone has measured on either arm.
        if (params.vocab_size < kF32Coarse12MinVocab) return false;
        // The two arms below are the only part of this predicate that is a
        // *measured floor* rather than a bound -- everything above is a limit
        // the route would be wrong or impossible outside of.  So this is where
        // the A/B knob goes, and it is the whole of what it replaces.
        if (f32_coarse12_override() >= 0) return f32_coarse12_override() != 0;
        // Route on the shape deep_gemm itself routes on
        // (`select_topk_policy`), with the floor taken from the measured
        // crossing on this device.  The narrow-width arm is why there are two
        // arms and not one: at or below `V = 131072` the route is ahead at
        // `b16` and never measured behind, so the width decides which floor
        // applies, and only the wide one is a function of `topk`.  The tables
        // above have both.
        if (params.vocab_size <= kF32Coarse12NarrowVocab)
            return batches >= kF32Coarse12MinBatchesNarrow;
        return (uint64_t)batches * params.topk >= kF32Coarse12WorkTarget;
    }
}

// ── the deep_gemm chunks route: the `b <= 2` band ───────────────────────────
//
// `topk_chunks.cuh` carries `topk_chunks`, deep_gemm's third kernel family, in the
// same shape as `topk_coarse12.cuh` carries the second.  It is here because the
// two arms this file already has are both the wrong shape at one or two rows,
// and that is measured rather than argued (ledger §12.16.6, interleaved five
// rounds on device 2, `k = 2048`):
//
//     b     V        split    deep_gemm    deep_gemm/split
//     1     66551     90.8      53.0        0.584x
//     2     66551     93.4      54.1        0.580x
//     4     66551     93.8     360.8        3.844x
//     1    107520    174.8      60.6        0.347x
//     1    131072    135.0      62.7        0.464x
//
// **The cliff is deep_gemm's, at `b = 4`.**  Its own `topk_chunks_compact_refine`
// goes from 24.4 us at one row to 310.7 us at sixteen -- 12.7x per row -- while
// the CTA count only goes from 6 to 96 on a 104-AP part.  So this is not a
// fallback to reach for whenever we are behind: it is a `b <= 2` special case,
// and the batch bound below is sited on the measured step, not on a guess.
//
// Why it wins there: deep_gemm walks the row a second time and writes every
// element **above** the threshold bin straight to the output in that same walk
// (its `guaranteed` half), staging only the threshold bin's members.  The split
// instead materializes `num_chunks * topk` candidates and merges them, and at
// `k = 2048, V = 66551` the elements above the threshold bin are already almost
// the whole answer -- which is why the sweep over `DEEP_SELECT_F32_CHUNKS`
// (2..32, ledger §12.16.6) moves the split by 46% and never reaches this.
//
// **Not `constexpr`-gated, and it does not need to be**: the arena here is
// **device** memory in the per-row workspace (`TopKChunksWorkspace::candidate_indices`,
// `topk_chunks.cuh`), not a shared buffer -- all three launches pass `0` dynamic
// shared memory -- and every family this repository builds has at least 64 KB
// per AP (`csrc/structs.h`), so the budget half of `f32_coarse12_applies` would
// be satisfied by all three.  The one question is whether the machine in front
// of the call is one this was measured on, which is the same `sm_count` test
// the coarse12 tuning half uses and for the same reason.
inline constexpr uint32_t kF32ChunksMaxBatches = 2;
inline constexpr uint32_t kF32ChunksMinVocab = 2048;

// `DEEP_SELECT_F32_CHUNKS_ROUTE`: `1` forces this arm on, `0` denies it, unset
// takes the band.  Same shape and same reason as `DEEP_SELECT_F32_COARSE12` --
// the band is a measured step and re-siting it means measuring the arm on the
// batches it declines.  It replaces the band and nothing else.
//
// **"And nothing else" is a property of where `f32_chunks_applies` calls it,
// not of this function.**  Every bound above the band still applies: the AP
// count the ladder was measured on, `kMaxTopK`, the width floor, and
// `batches <= kF32ChunksMaxBatches`.  In particular the AP-count test comes
// first, so this knob **cannot reach a C600U at either of its AP counts** --
// that family takes the row path at every batch, whatever the variable is set
// to.  `DEEP_SELECT_F32_CHUNKS=n`
// is the knob for a different chunk *count*; this one only decides whether the
// arm is entered at all.
inline int f32_chunks_route_override() {
    static const int v = [] {
        const char *s = std::getenv("DEEP_SELECT_F32_CHUNKS_ROUTE");
        if (s == nullptr || s[0] == '\0') return -1;
        return std::atoi(s) != 0 ? 1 : 0;
    }();
    return v;
}

inline bool f32_chunks_applies(const RowParams &params, uint32_t batches) {
    // ── the limits: outside these the route is wrong or impossible ──
    if (params.sm_count != kF32Coarse12MeasuredSmCount) return false;
    if (params.topk > (uint32_t)topk_select_chunks::kMaxTopK) return false;
    // The original's own width bound, as on the coarse12 arm: `2048`, not
    // `2049`, because the facade already rejects a stride that is not a
    // multiple of 256 floats.
    if (params.vocab_size < kF32ChunksMinVocab) return false;
    if (batches == 0 || batches > kF32ChunksMaxBatches) return false;
    // ── the measured band, and the A/B knob over it ──
    // **The override replaces the band and only the band**, which is why it
    // sits below the bounds rather than among them.  `batches <= 2` here is a
    // *limit*, not the band: the chunks kernel is one CTA per row and its
    // scratch sizing (`chunks_workspace_bytes`) is `batches *` one row's
    // workspace, so a batch past this is a route that was never sized for.
    // What the override moves is the step between "the row path" and "the
    // chunks arm" at the batch the arm was measured on, and the coarse12 arm
    // orders the same two pieces the same way (`f32_coarse12_applies` puts its
    // `batches == 0` test above its `f32_coarse12_override`).
    //
    // **The AP-count test above is a limit for the same reason, and it is why
    // the knob cannot reach a C600U**: `f32_chunks_route_override()` is read
    // only from here, so `DEEP_SELECT_F32_CHUNKS_ROUTE=1` changes nothing on
    // that family, at either of its AP counts.  See `structs.h`'s note on the
    // family table.
    if (f32_chunks_route_override() >= 0) return f32_chunks_route_override() != 0;
    return true;
}
}  // namespace

// `topk` and `vocab_size` are parameters here, not read off `params`, because
// every caller that sizes the workspace must get the same answer as the one
// that launches: a workspace sized for one chunk count and a grid launched with
// another is a write past the arena, and the gate at `select_route` compares
// the two.
int f32_chunked_chunks(uint32_t batches, uint32_t topk, uint32_t vocab_size,
                       uint32_t sm_count) {
    const uint32_t work_target = f32_chunk_work_target(sm_count);
    static const int override_n = [] {
        const char *v = std::getenv("DEEP_SELECT_F32_CHUNKS");
        if (v != nullptr && v[0] != '\0') {
            const int d = std::atoi(v);
            return (d >= 2 && d <= 256) ? d : 0;
        }
        return 0;
    }();
    if (override_n != 0) return override_n;   // A/B knob; does not change default
    if (batches > kF32ChunksFewBatches)
        return f32_chunks_large_batch(vocab_size, sm_count);
    const int ceiling = f32_chunk_ceiling_32() ? 32 : kF32ChunkCeiling;
    const int c = f32_chunks_for(batches, ceiling, work_target);
    const int cap = f32_topk_chunk_cap(topk, vocab_size, (uint32_t)c, work_target);
    int n = (cap != 0 && c > cap) ? cap : c;
    // **The arena floor, which this branch never had.**  `f32_chunks_large_batch`
    // above is derived from exactly one thing -- the largest chunk whose
    // threshold bin still fits `kF32SmemInputSize` -- but the small-batch
    // branch below `kF32ChunksFewBatches` was sized from the batch alone and
    // never learned it.  The omission is not a slope, it is a cliff, and it
    // lands precisely on the branch boundary: at `V = 524288` the batch rule
    // gives `c = 4` for `33 <= batches <= 64`, which is a 131072-element chunk
    // against the `kF32OverflowChunkLen` bound, so `radix_topk_row_f32_rescan`
    // runs five passes where it should run two.  Measured, split alone,
    // `k = 2048`: `b32` is 0.3732 ms (c = 8, 65536/chunk, no rescan) and `b33`
    // is 1.77 ms (c = 4); `b65` takes the branch above, gets 6, and is fine
    // again.  So the two branches disagreed about the same row at their own
    // seam.  With the floor, split alone: `b40` **-77.6%**, `b64` **-72.5%**,
    // and the two controls that have no overflow (`b32`, `b40 V = 262144`) at
    // -0.1% / -0.3%.
    //
    // Applied *after* the cap and as a floor, not a replacement: the batch rule
    // and the curve still decide the count wherever they are already above the
    // bound.
    //
    // **The `topk` guard is the measurement; the branch above is geometry.**
    // `f32_chunks_large_batch` applies the same arena bound to *every* `topk`,
    // because that is the only way it can be a derivation rather than a table.
    // Here the bound has to be checked against a measurement instead: the 58 is
    // a `k = 2048` value, and at `k = 512` the threshold bin is narrower and
    // `c = 4` never overflowed -- so raising it there is free in the arena and
    // the merge pays for it, measured `b64 V = 524288 k = 512` at **+14.3%**
    // against -0.1% / -0.3% on the controls.  So this arm fires where its
    // constant was derived, and `k = 1024` keeps the `k = 2048` verdict for
    // want of a measurement rather than for want of a reason.
    if (topk == kF32ChunkCurveTopK) {
        const uint32_t arena =
            (vocab_size + kF32OverflowChunkLen - 1) / kF32OverflowChunkLen;
        if ((uint32_t)n < arena) n = (int)arena;
    }
    return n;
}

// The batch bound has no single value: the split costs a merge over `batches`
// CTAs on an otherwise idle machine and saves the row kernel's per-CTA time, so
// which way it goes is measured per shape.  The floor was 262144 and the chunk
// count above is why it could drop to the small-batch tier's value -- with the
// count now a constant 2 above 64 rows the merge is 3 CTAs per row at every
// shape, so the cost stops scaling with the gate.  Measured over the eight
// cells the change reaches: -12.7% total, with the two neutral cells being the
// contract (b6 was already open, b4096-v262144 already served).
//
// **The band `32768 <= V < 65536` is what that unification newly serves**, and
// it was measured before being opened: six controls at `V >= 65536`, which the
// change cannot reach, hold to |delta| <= 0.3%.  Every cell in the band wins,
// -5.6% to -27.7%, the magnitude falling as the batch rises -- the same curve
// every other result here shows.  Full tables in
// `docs/C500-radix-perf-ledger.zh.md`.
//
// There used to be four constants here, a small-batch tier (`V >= 65536`,
// `batches <= 64`) and a large-batch one (`V >= 262144`, later 65536,
// `batches <= 4096`).  Lowering the large tier's floor to 32768 makes the small
// branch unreachable (it would need `V < 32768` and `V >= 65536` at once), so
// the predicate is one condition and the two constants below are what is left.
// A constant that cannot select anything is a knob a reader will try to turn.
constexpr uint32_t kF32ChunkedMinVocab = 32768;
constexpr uint32_t kF32ChunkedMaxBatches = 4096;

// **When is a split worth its own two extra stages?**  The row kernel already
// runs one CTA per row, so a split adds a scan and a merge over
// `batches * (chunks + 1)` CTAs and buys exactly one thing: parallelism the row
// kernel did not have.  That is a function of the row length against the
// machine, not of `topk` -- the gain rises with `vocab` and falls with `batches`,
// and large-batch short rows are the corner where the split has nothing to buy.
//
// The model: the split's own stages cost `B * (chunks + 1) * fixed` for a fixed
// per-CTA cost, and a row costs `V * c` at a per-element rate `c`; with
// chunks = 2 that makes rows win below `V < B * 6.0e6`.  A two-term model is
// not trustworthy near the corner it was fitted at, so it is **coupled to the
// measured region**: inside the sweep the sweep decides, and only outside it
// does the model speak.  The sweep's verdicts, and the one cell it leaves
// ambiguous (`b768-v65536`, which the model sends to the split and the gate
// follows), are in `docs/C500-radix-perf-ledger.zh.md` -- recorded there rather
// than smoothed over.

inline bool topk_worth_splitting_f32(const RowParams &params,
                                     uint32_t batches) {
    if (params.topk <= 0 || params.topk > (int)topk_select_radix::kF32MaxTopK) return false;
    if (batches == 0) return false;
    // V=32768 rows, and every row of V<=65536 at a batch past the 768 tier.
    if (params.vocab_size <= kF32ChunkedMinVocab) return false;
    if (params.vocab_size <= 65536 && batches > 256) return false;
    return true;
}

// The split is fp32-only (the caller's `value_dtype == 0`) and its merge only
// compiles the k=512/1024 arms, which is the whole gate.
inline bool chunked_f32_applies(const RowParams &params, uint32_t batches) {
    if (batches == 0) return false;
    // `end_ptr` absent means the row really is `vocab_size` long, so the cap is
    // the vocab either way -- and it is the only length the host knows without
    // a device read.
    if (params.vocab_size < kF32ChunkedMinVocab) return false;
    if (batches > kF32ChunkedMaxBatches) return false;
    // The topk clause that used to sit here (`== 512 || == 1024`) was a *policy*
    // bound, not an instantiation one -- the fp32 chunk engine is dynamic-k
    // (`topk_select_radix::f32_chunk_engine_supports`) -- so it is replaced by a measured one
    // rather than by a wider constant.  Plan §21.3 opened it to
    // `topk <= kF32MaxTopK` and got -71..-82% on the long rows but up to +26.5%
    // back on short ones at a large batch; the cause is the shape of what a
    // split buys.
    if (params.topk == 512 || params.topk == 1024) return true;   // measured
    return topk_worth_splitting_f32(params, batches);
}

inline size_t chunked_f32_workspace_bytes(uint32_t batches, uint32_t topk,
                                          int chunks) {
    return topk_select_radix::chunked_f32_workspace_bytes(batches, topk, (uint32_t)chunks);
}

// The workspace's layout.  `topk_select_radix::chunked_f32_cols` / `_vals_offset` / `_out` are
// the *definition* of it -- the split's own region order and the 16-byte pad
// `radix_topk_row_f32`'s unguarded `float4` walkers need -- and this function
// is the one place that composes them: the pad is exactly the kind of thing a
// second copy of the arithmetic would miss, and the launch takes the three
// pointers rather than re-deriving them from the arena base.  The region in
// front used to be the flag table; it lives in the dispatcher's dedicated
// `scan_flags` now, so the arena base is the `cudaMalloc`'d workspace and
// `cols` is its first region, which is what `chunked_f32_cols` documents.
struct ChunkedF32Workspace {
    int32_t *cols;
    float *vals;
    int32_t *merged;
};

inline ChunkedF32Workspace chunked_f32_workspace(void *base, uint32_t batches,
                                                 uint32_t topk, int chunks) {
    ChunkedF32Workspace ws{};
    const uint32_t b = batches, k = (uint32_t)topk, c = (uint32_t)chunks;
    ws.cols = topk_select_radix::chunked_f32_cols(base, b, k, c);
    ws.vals = topk_select_radix::chunked_f32_vals(ws.cols, b, k, c);
    ws.merged = topk_select_radix::chunked_f32_out(base, b, k, c);
    return ws;
}

// Split, merge, then run the row kernel's contract half over the merged
// answer.  `params` must satisfy `chunked_f32_applies`, workspace included.
//
// The row table is passed through as `end_ptr` rather than being consumed here
// -- the split's own kernels take it as `lengths` -- which is what keeps the
// dispatch below (`params.end_ptr != nullptr`) meaningful.
template <typename OutIdxT>
void launch_typed_f32_chunked(const RowParams &params, uint32_t batches,
                              cudaStream_t stream, bool sorted_index,
                              bool sorted_value, bool return_value, int block,
                              void *workspace) {
    const int chunks = f32_chunked_chunks(batches, params.topk,
                                          params.vocab_size, params.sm_count);
    const ChunkedF32Workspace ws = chunked_f32_workspace(
        workspace, batches, params.topk, chunks);
    // The NaN table is the dispatcher's dedicated `scan_flags`, not a region of
    // this workspace -- the workspace is one grow-only scratch shared across
    // split geometries and dtypes, so a slot read at a geometry-dependent
    // offset is whatever the previous call wrote there.  The table is zeroed
    // once when it grows and raised per row by whichever walk found the bit
    // pattern: stage 1's own pass 1 with the scan fused, `nan_scan_kernel`
    // otherwise.  The contract half reads and clears each row's slot in the
    // same breath, so there is no per-call memset here; only a failure after a
    // partial launch can leave a raised slot behind, and that is what the
    // handoff's clear covers.  Both the scan and that clear go when `check_nan`
    // is off: the row kernel reads this table only under that same flag.
    int32_t *const flags = params.check_nan ? params.scan_flags : nullptr;
    const bool fuse_scan = params.check_nan && topk_select_radix::f32_stage1_fuses_scan();
    if (params.check_nan && !fuse_scan) {
        launch_nan_scan<float>(
            params.input, params.end_ptr,
            (int64_t)(params.stride_input_batch / sizeof(float)),
            params.vocab_size, (uint32_t)chunks, flags, batches, stream);
    }
    // Stage 1 then stage 2, named here rather than in `radix_chunked.cuh` for
    // the reason `launch_typed_chunked` records: a non-template function body
    // in a header every kernel TU includes is compiled into every one of those
    // TUs' device images, and these two names are concrete specializations.
    //
    // The three arenas are **arguments, not derived here**.  The caller already
    // owns the workspace and places the split's regions inside it, so it is the
    // one that knows the layout; re-deriving it in the dataflow was a second
    // copy of the same arithmetic, and the two agreed only because they were
    // written the same way.  `chunked_f32_cols` / `chunked_f32_vals` /
    // `chunked_f32_out` remain as the definition of the layout, and the caller
    // uses them.  The per-row NaN table is **not** a region of this workspace
    // -- it is the dispatcher's own `scan_flags` allocation, passed separately,
    // because a geometry-dependent offset into shared scratch is not a flag
    // slot.
    cudaError_t rc = fuse_scan
        ? topk_select_radix::launch_f32_chunk_stage1<true>(
              (const float *)params.input, params.end_ptr, ws.cols, ws.vals,
              (int)batches, (int)params.vocab_size, (int)params.topk, chunks,
              stream,
              // The row stride is in bytes at this layer and in elements there.
              (int64_t)(params.stride_input_batch / sizeof(float)), flags)
        : topk_select_radix::launch_f32_chunk_stage1<false>(
              (const float *)params.input, params.end_ptr, ws.cols, ws.vals,
              (int)batches, (int)params.vocab_size, (int)params.topk, chunks,
              stream,
              (int64_t)(params.stride_input_batch / sizeof(float)), flags);
    if (rc == cudaSuccess) {
        rc = topk_select_radix::launch_f32_chunk_stage2<
            (int)topk_select_radix::kF32ChunkBlocks>(
            ws.vals, ws.cols, ws.merged, (int)batches, (int)params.topk, chunks,
            stream);
    }
    // A failed launch answers through the row path, which reads no flag table;
    // the handoff clears anything a partially launched split raised.
    launch_staged_or_row<float, OutIdxT>(params, batches, stream, sorted_index,
                                         sorted_value, return_value, block,
                                         rc == cudaSuccess, ws.merged,
                                         /*clear_flags_on_failure=*/true);
}

// Split, merge, then run the row kernel's contract half over the merged
// answer.  `params` must satisfy `chunked_bf16_applies`, workspace included.
template <typename OutIdxT>
void launch_typed_chunked(const RowParams &params, uint32_t batches,
                          cudaStream_t stream, bool sorted_index,
                          bool sorted_value, bool return_value, int block,
                          void *workspace) {
    const int chunks = chunked_chunks(params);
    const ChunkedWorkspace ws =
        chunked_workspace(workspace, batches, params.topk, (uint32_t)chunks);
    // The flags table is the dispatcher's dedicated `scan_flags`, not a region
    // of this workspace -- the rule and the trap are the fp32 split's note
    // above; this arm differs only in having no separate scan kernel, because
    // its stage 1 always fuses the scan.  Stage 1 raises a row's slot and the
    // contract half reads and clears it; only a failure after a partial launch
    // can leave a raised slot behind, and that clear follows the launch.
    int32_t *const flags = params.check_nan ? params.scan_flags : nullptr;
    const maca_bfloat16 *const scores = (const maca_bfloat16 *)params.input;
    // The row stride is in bytes at this layer and in elements there.
    const int64_t score_stride =
        (int64_t)(params.stride_input_batch / sizeof(maca_bfloat16));
    // The `topk` switch is here rather than in `radix_chunked.cuh` for the
    // reason `launch_typed_f32_chunked` records: a non-template function body
    // in a header every kernel TU includes is compiled into every one of those
    // TUs' device images, and every arm below names a concrete kernel.  The
    // gate is the dataflow's own -- `chunked_bf16_applies` admits only 512 and
    // 1024, so the static-k arms are the ones a real call takes and the
    // runtime-`topk` pair is the arm for a shape the gate has yet to admit.
    cudaError_t rc = cudaErrorInvalidValue;
    const int topk = (int)params.topk;
    if (topk <= topk_select_radix::kMaxTopK && chunks > 1) {
        switch (topk) {
        case 512:
            rc = topk_select_radix::launch_topk_bf16_chunked_k<512>(
                scores, params.end_ptr, ws.merged, ws.candidate_indices,
                ws.candidate_values, (int)batches, (int)params.vocab_size,
                chunks, stream, score_stride, params.check_nan, flags);
            break;
        case 1024:
            rc = topk_select_radix::launch_topk_bf16_chunked_k<1024>(
                scores, params.end_ptr, ws.merged, ws.candidate_indices,
                ws.candidate_values, (int)batches, (int)params.vocab_size,
                chunks, stream, score_stride, params.check_nan, flags);
            break;
        default:
            rc = params.check_nan
                ? topk_select_radix::launch_bf16_chunk_stage1_runtime<true>(
                      scores, params.end_ptr, ws.candidate_indices,
                      ws.candidate_values, (int)batches, (int)params.vocab_size,
                      topk, chunks, stream, score_stride, flags)
                : topk_select_radix::launch_bf16_chunk_stage1_runtime<false>(
                      scores, params.end_ptr, ws.candidate_indices,
                      ws.candidate_values, (int)batches, (int)params.vocab_size,
                      topk, chunks, stream, score_stride, flags);
            if (rc == cudaSuccess) {
                rc = topk_select_radix::launch_bf16_chunk_stage2_runtime<
                    topk_select_radix::kChunkBlockSize>(
                    ws.candidate_values, ws.candidate_indices, ws.merged,
                    (int)batches, chunks, topk, stream);
            }
            break;
        }
    }
    // The row path that answers a failed split reads no flag table, so the
    // handoff clears a slot a partial launch raised rather than leaving it for
    // the next call to mis-read.
    launch_staged_or_row<maca_bfloat16, OutIdxT>(
        params, batches, stream, sorted_index, sorted_value, return_value,
        block, rc == cudaSuccess, ws.merged, /*clear_flags_on_failure=*/true);
}

// The coarse12 route's launch and emit.  The kernel answers in row *columns*,
// which are int32 by nature, so it cannot write the caller's buffer directly;
// the columns are staged and the contract half widens them the way it widens
// the split's answer.  The gate that admits this route is
// `f32_coarse12_applies` above; what is here is everything after the
// decision.
template <typename OutIdxT>
void launch_typed_coarse12(const RowParams &params, uint32_t batches,
                           cudaStream_t stream, bool sorted_index,
                           bool sorted_value, bool return_value, int block) {
    int32_t *const staging = params.coarse12_cols;
    bool launched = false;
    if (staging != nullptr) {
        // No split, no workspace of its own, no NaN table of its own: this
        // kernel walks each row itself and answers in row columns, so the
        // contract half below is the only other thing that touches the row
        // -- and it is a pass over the *answer*, not over the input.
        // `params.end_ptr` is the row table the dispatcher already
        // populated with the scratch's all-`vocab_size` entry when the
        // caller passed no `end`, which is the same table the split reads.
        // `params.vocab_size` -- not the row stride -- is the width to
        // scan.  They agree for a full-width tensor and differ the moment
        // the caller hands over a column slice of a wider matrix, which is
        // exactly what the benchmark harness does (`s[:, :L]`): the kernel
        // would then walk past the live prefix and rank columns that are
        // outside the tensor the caller asked about.
        launched = topk_select_coarse12::launch_topk_coarse12(
            (const float *)params.input, params.end_ptr, staging,
            params.scan_flags, (int)batches, (int)params.topk,
            (int64_t)(params.stride_input_batch / sizeof(float)),
            (int)params.vocab_size, params.coarse12_block_max,
            params.coarse12_block_list, stream) == cudaSuccess;
    }
    // Hand the answer to the contract half the way the split does, so its
    // per-row `-1` check decides row by row: a row this kernel could not
    // serve, or one whose window is no longer than `topk`, comes back with an
    // empty slot and is re-ranked by the row path.  A launch that did not
    // happen leaves the buffer untouched and answers that way too -- as does
    // an allocation that could not be made.
    //
    // One kernel, so a failed launch means nothing ran: there is no partial
    // work whose scan flag could have survived, which is why this route
    // clears nothing (`clear_flags_on_failure = false`).
    launch_staged_or_row<float, OutIdxT>(
        params, batches, stream, sorted_index, sorted_value, return_value, block,
        launched, staging, /*clear_flags_on_failure=*/false);
}

// The chunks route's launch and emit, on the same handoff as the coarse12
// route: the kernels answer in row columns, the contract half reads them out
// of the staging scratch, and a row they could not serve leaves an empty slot
// there for the contract half to re-rank on the row path.  The gate that
// admits this route is `f32_chunks_applies` above.
template <typename OutIdxT>
void launch_typed_chunks(const RowParams &params, uint32_t batches,
                         cudaStream_t stream, bool sorted_index,
                         bool sorted_value, bool return_value, int block) {
    int32_t *const staging = params.coarse12_cols;
    bool launched = false;
    // The chunk-count dispatch is here rather than in `topk_chunks.cuh` for the
    // reason `launch_typed_f32_chunked` records -- the `NChunks` cases name
    // concrete kernels, and a non-template body over them lands in every
    // including TU's device image.
    const int topk = (int)params.topk;
    cudaError_t rc = cudaErrorInvalidValue;  // 0 chunks: the shape declines
    if (staging != nullptr && params.chunks_workspace != nullptr &&
        topk <= topk_select_chunks::kMaxTopK) {
        const int chunks = topk_select_chunks::select_chunk_count(
            (int)batches, (int64_t)params.vocab_size, params.sm_count);
        topk_select_chunks::dispatch_chunk_count(
            chunks, [&]<int N>() {
                rc = topk_select_chunks::launch_topk_chunks_impl<N>(
                    (const float *)params.input, params.end_ptr, staging,
                    params.scan_flags, params.chunks_workspace, (int)batches,
                    topk,
                    (int64_t)(params.stride_input_batch / sizeof(float)),
                    (int)params.vocab_size, stream);
            });
        launched = rc == cudaSuccess;
    }
    // This route is three launches, and the histogram one raises the scan
    // table.  A later stage's launch failing leaves that kernel's work in
    // flight while the row path answers this call -- and the row path reads
    // no flag table, so anything raised would sit there until the next call
    // that uses this table read a stale 1 as its own answer.  Cleared by the
    // handoff for the same reason the splits clear theirs after a failed
    // launch; unreachable through the gate, and cheap enough not to depend on
    // that.
    launch_staged_or_row<float, OutIdxT>(
        params, batches, stream, sorted_index, sorted_value, return_value, block,
        launched, staging, /*clear_flags_on_failure=*/true);
}

// Every route below is templated on the index width, and `index_dtype` is the
// dispatcher's encoding of it -- 0 is `int32_t`, anything else `int64_t`, the
// same encoding `maca_topk.cu` derives from `output_index.dtype()`.  One place
// turns the int into a type, so a route cannot carry the pair and give its two
// arms different arguments: the call is written once, against `OutIdxT`.
template <typename RouteFn>
inline void dispatch_index_width(int index_dtype, RouteFn &&route) {
    if (index_dtype == 0) {
        route.template operator()<int32_t>();
    } else {
        route.template operator()<int64_t>();
    }
}

// ── the decision: which route answers a call ────────────────────────────────
//
// The predicates above are pure functions of the shape and the device; this is
// the one place that reads them, in order, and its answer is a value rather
// than six booleans re-tested at the launch.  The dispatcher (`maca_topk.cu`)
// sizes its scratch from the same `RouteGates`, so the route and the buffers it
// may read are one decision: a buffer cannot be sized for a route the launch
// does not take, which is the class of bug the grow-only scratch makes quiet
// rather than loud.
enum class Route {
    kRowBf16,
    kSplitBf16,
    kF32Chunks,
    kF32Coarse12,
    kSplitF32,
    kRowF32,
};

// The four shape gates, evaluated once per call, by the dispatcher, before it
// allocates.  Each is the *policy* half only: whether this call can supply the
// workspace a split needs is `select_route`'s question, because it depends on
// what the dispatcher was able to allocate.
struct RouteGates {
    bool bf16_split;
    bool f32_split;
    bool f32_chunks;
    bool f32_coarse12;
};

inline RouteGates route_gates(const RowParams &params, uint32_t batches,
                              int value_dtype) {
    return RouteGates{
        value_dtype == 1 && chunked_bf16_applies(params, batches),
        value_dtype == 0 && chunked_f32_applies(params, batches),
        value_dtype == 0 && f32_chunks_applies(params, batches),
        value_dtype == 0 && f32_coarse12_applies(params, batches),
    };
}

// What answers this call.  `workspace`/`bytes` are the split's scratch as this
// call has it; the guards are re-tested rather than assumed from the gate,
// because a null or undersized workspace must answer through the row path
// rather than launch a split that would write past it.
//
// The order of the returns is the decision, not an accident of where the arms
// landed -- see the `b <= 2` note below for the one pair that could otherwise
// both claim a cell.
inline Route select_route(const RouteGates &gates, const RowParams &params,
                          uint32_t batches, int value_dtype,
                          const void *workspace, size_t bytes) {
    // The host entry admits exactly float32 (0) and bfloat16 (1).
    if (value_dtype != 0) {
        // Long rows at a small batch go through the split; the caller sized the
        // workspace for exactly this gate, and the split reads 16-bit input.
        if (gates.bf16_split && workspace != nullptr &&
            params.end_ptr != nullptr &&
            bytes >= chunked_workspace_bytes(batches, params.topk,
                                             (uint32_t)chunked_chunks(params)))
            return Route::kSplitBf16;
        return Route::kRowBf16;
    }
    // ── the `b <= 2` arm: deep_gemm's chunks kernel ────────────────────────
    // Same handoff as the coarse12 arm below it -- answer in row columns, staged
    // through the scratch, contract half reads the per-row `-1` -- because it is
    // the same class of kernel: one CTA per row, walking the row itself.  Tested
    // *before* coarse12 so the two cannot both claim a cell; the batch bands are
    // disjoint by construction (`<= 2` here, `>= 16` or a 114688 product there),
    // but the order makes that a fact about the code rather than about two
    // constants that have to keep agreeing.
    if (gates.f32_chunks) return Route::kF32Chunks;
    // ── the coarse12 arm: fp32, answered in row columns ─────────────────────
    // Both index widths are served, and that is what makes it reachable at
    // all: `deep_select.topk`'s default `indices_type` is `torch.int64`.  The
    // kernel answers in row *columns*, which are int32 by nature, so a 64-bit
    // call stages them through the dispatcher's scratch and the contract half
    // does the widening it already does for the split.  Writing them straight
    // into `output_index` would be wrong for int64 (`int32_t*` arithmetic over
    // an int64 buffer), so the scratch is not an optimization here, it is what
    // makes the arm legal.
    if (gates.f32_coarse12) return Route::kF32Coarse12;
    // The fp32 split, for rows too long for one CTA at a batch too small for
    // the row path.  The engine's own bound is kept separate from the policy
    // predicate so the two cannot be confused (they were one expression until
    // §21.2); the row path serves anything the engine cannot.
    if (gates.f32_split && workspace != nullptr && params.end_ptr != nullptr &&
        bytes >= chunked_f32_workspace_bytes(
                     batches, params.topk,
                     f32_chunked_chunks(batches, params.topk,
                                        params.vocab_size,
                                        params.sm_count)) &&
        topk_select_radix::f32_chunk_engine_supports((int)params.topk))
        return Route::kSplitF32;
    return Route::kRowF32;
}

}  // namespace detail

// The one launch entry.  `route` is `select_route`'s answer, computed by the
// dispatcher from the gates it sized the scratch with -- so this function does
// not re-decide anything, it only resolves the last axis that is still a value:
// the output index width, which is a template argument on every route and is
// therefore turned into a type once, below.
void topk_launch(const RowParams &params, int64_t batches, void *stream,
                 detail::Route route, int index_dtype, bool sorted_index,
                 bool sorted_value, bool return_value, void *chunked_workspace) {
    auto *cuda_stream = reinterpret_cast<cudaStream_t>(stream);
    const uint32_t n = (uint32_t)batches;
    const bool rv = return_value || sorted_value;
    // Both dtypes run the ported radix dataflow; only the block width differs,
    // and the fp32 row has a single width.
    const bool bf16 = route == detail::Route::kRowBf16 ||
                      route == detail::Route::kSplitBf16;
    const int block =
        bf16 ? detail::radix_block_for(n, params.vocab_size, params.topk,
                                       params.sm_count)
             : (int)topk_select_radix::kBlockSize;
    detail::dispatch_index_width(index_dtype, [&]<typename OutIdxT>() {
        switch (route) {
            case detail::Route::kSplitBf16:
                detail::launch_typed_chunked<OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block, chunked_workspace);
                break;
            case detail::Route::kF32Chunks:
                detail::launch_typed_chunks<OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block);
                break;
            case detail::Route::kF32Coarse12:
                detail::launch_typed_coarse12<OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block);
                break;
            case detail::Route::kSplitF32:
                detail::launch_typed_f32_chunked<OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block, chunked_workspace);
                break;
            case detail::Route::kRowF32:
                detail::launch_typed_radix<float, OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block);
                break;
            case detail::Route::kRowBf16:
                detail::launch_typed_radix<maca_bfloat16, OutIdxT>(
                    params, n, cuda_stream, sorted_index, sorted_value, rv,
                    block);
                break;
        }
    });
}

}  // namespace deep_select

#endif  // ROUTING_CUH
