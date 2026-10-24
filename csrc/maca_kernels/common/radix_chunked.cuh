#ifndef RADIX_CHUNKED_CUH
#define RADIX_CHUNKED_CUH

#include <cstdlib>           // std::getenv, for the fused scan's A/B knob

#include <csrc/maca_kernels/common/radix_common.cuh>
#include <csrc/maca_kernels/common/radix_row_bf16.cuh>
#include <csrc/maca_kernels/common/radix_row_fp32.cuh>

namespace topk_select_radix {

// One body, two arms.  `TOPK == 0` is the runtime-`topk` arm -- `topk` arrives
// as an argument instead of a template parameter -- and an explicit non-zero
// `TOPK` is the static-`k` arm the 512/1024 decode cells take, where
// `topk_arg` folds away at compile time.  The two are one function so a change
// to the chunk's walk cannot land in one and miss the other; the launchers
// below keep the tuple-per-kernel split the build needs.
//
// `kNan` is the caller's `check_nan`: with it on this kernel is *also* the NaN
// scan and `flags` is the table it raises, in place of `nan_scan_kernel`.  The
// two share the grid by construction (`nan_scan_kernel`'s comment), so the only
// thing that changes is which kernel reads the chunk's bytes.
template <uint32_t TOPK, bool kNan = false>
__global__ void topk_bf16_chunk_stage1_kernel(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* candidate_indices,
    maca_bfloat16* candidate_values, int64_t score_stride, int topk_arg, int B,
    int num_chunks, int chunk_size, int32_t* flags = nullptr)
{
    constexpr int TOPK_I = static_cast<int>(TOPK);
    const int topk = TOPK_I != 0 ? TOPK_I : topk_arg;
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;

    const int32_t length = lengths[bid];
    const int start = chunk * chunk_size;
    const int chunk_len = (start < length) ? min(chunk_size, static_cast<int>(length - start)) : 0;
    constexpr int BLOCK_SIZE = kChunkBlockSize;
    const int candidate_stride = num_chunks * topk;
    int32_t* chunk_indices = candidate_indices + bid * candidate_stride + chunk * topk;
    maca_bfloat16* chunk_values = candidate_values + bid * candidate_stride + chunk * topk;
    const maca_bfloat16* row = scores + bid * score_stride;
    const maca_bfloat16 invalid_value = __float2bfloat16(-65504.0f);

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
        return;
    }

    if (chunk_len <= topk) {
        // The window is shorter than `topk`, so the copy already touches every
        // byte the scan would read: the vote rides it, as in the fp32 split.
        bool nan_local = false;
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            if (i < chunk_len) {
                const int32_t idx = start + i;
                const maca_bfloat16 raw = __ldg(row + idx);
                chunk_indices[i] = idx;
                chunk_values[i] = raw;
                if (kNan) nan_local |= bf16_is_nan(__bfloat16_as_ushort(raw));
            } else {
                chunk_indices[i] = -1;
                chunk_values[i] = invalid_value;
            }
        }
        if (kNan && flags != nullptr && __syncthreads_or((int)nan_local) != 0
            && threadIdx.x == 0)
            atomicOr(flags + bid, 1);
        return;
    }

    // The chunk is ranked by the coarse12 row: this walk is over the input's
    // bytes, and the 12-bit coarse level is what makes that walk a streaming
    // one.  Measured on C500, device 2
    // (`docs/experiments/20261009_c500_dsa_decode/README.md`): the same chunk
    // runs at 382 GB/s on the 12-bit level against 256 on the 8-bit one
    // (§9.12), and `b6 x 262144 k512` goes from 103.9 to 70.5 us e2e against
    // the row kernel's 97.7.  Both stages rank with it, and both therefore
    // take `kChunkSmem`.  `kSkip=false`: these rows carry no skip tables (the
    // candidates live in the global arrays), so the summary is compiled out.
    bool nan_row = false;
    radix_topk_row_bf16_b<kChunkBlockSize, kNan, false>(
        row + start, chunk_indices, chunk_len, (uint32_t)topk,
        kNan ? &nan_row : nullptr);
    if (kNan && flags != nullptr && nan_row && threadIdx.x == 0)
        atomicOr(flags + bid, 1);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t local_idx = chunk_indices[i];
        if (local_idx >= 0 && local_idx < chunk_len) {
            const int32_t idx = start + local_idx;
            chunk_indices[i] = idx;
            chunk_values[i] = __ldg(row + idx);
        } else {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
    }
}

// Stage 2 of both arms, on the same `TOPK == 0` sentinel: `kBlock` is the
// block width (both launchers pass `kChunkBlockSize`, and it is a template
// parameter because the launcher's tuple spells it), a non-zero `TOPK` folds
// `topk_arg` away, and `kSkip=false` for the same reason as stage 1.
template <int kBlock, uint32_t TOPK = 0u>
__global__ void topk_bf16_chunk_stage2_kernel(
    const maca_bfloat16* candidate_values, const int32_t* candidate_indices,
    int32_t* indices, int candidate_stride, int topk_arg, int B)
{
    constexpr int TOPK_I = static_cast<int>(TOPK);
    const int topk = TOPK_I != 0 ? TOPK_I : topk_arg;
    const int bid = blockIdx.x;
    if (bid >= B) return;

    const maca_bfloat16* values = candidate_values + bid * candidate_stride;
    const int32_t* candidates = candidate_indices + bid * candidate_stride;
    int32_t* out = indices + bid * topk;
    constexpr int BLOCK_SIZE = kBlock;

    // The merge's input is `chunks * topk` candidate values rather than the
    // row's bytes, and the coarse12 row is the faster one over it too:
    // measured at 10/10 of the split's cells, 0.804x summed, where the same
    // swap on stage 1 was 0.832x (`README.md` §9.15 and §9.12).  The arena is
    // `kCoarse12ArenaEntries` on a `chunks * topk` row, not the row kernel's
    // longer one.
    radix_topk_row_bf16_b<kBlock, false, false>(values, out, candidate_stride, (uint32_t)topk);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t candidate_pos = out[i];
        out[i] = (candidate_pos >= 0 && candidate_pos < candidate_stride)
               ? candidates[candidate_pos]
               : -1;
    }
}

// Stage 1 of the static-k arm.  **One launcher, one kernel**: a tuple is one
// file (`scripts/generate_instantiations.py`), so each explicit instantiation
// there wants a launcher that owns exactly one `<<<>>>` -- which is also what
// puts the kernel's device code and the only launch that reaches it in the same
// translation unit.
//
// `kNan` is the caller's `check_nan`: with it on this kernel is also the NaN
// scan and `nan_flags` is the table it raises.  It is a template parameter here
// rather than the composing function's runtime branch so the two arms are two
// instantiations, which is what the two generated files are.
template <uint32_t TOPK, bool kNan>
inline cudaError_t launch_bf16_chunk_stage1_k(
    const maca_bfloat16* scores, const int32_t* lengths,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int num_chunks, cudaStream_t stream, int64_t score_stride,
    int32_t* nan_flags)
{
    if (TOPK > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    // Both stages run the coarse12 row, whose arena *is* the 16 KB coarse
    // histogram it aliases, followed by the `topk` staging the core writes.
    constexpr size_t kChunkSmem =
        kCoarse12HistBytes + sizeof(uint32_t) * (size_t)TOPK;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_bf16_chunk_stage1_kernel<TOPK, kNan>, kChunkSmem, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;

    const int chunk_size = chunk_size_for(L, num_chunks);
    // Deviation 3: the row stride can be wider than the window (`L`), which is
    // how a caller with padded input rows reaches this path; upstream passes
    // `L` and so can only split packed rows.
    const int64_t stride = score_stride > 0 ? score_stride : (int64_t)L;
    topk_bf16_chunk_stage1_kernel<TOPK, kNan>
        <<<B * num_chunks, kChunkBlockSize, kChunkSmem, stream>>>(
            scores, lengths, candidate_indices, candidate_values, stride,
            static_cast<int>(TOPK), B, num_chunks, chunk_size,
            kNan ? nan_flags : nullptr);
    return cudaGetLastError();
}

extern template cudaError_t launch_bf16_chunk_stage1_k<512u, true>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, cudaStream_t, int64_t, int32_t*);
extern template cudaError_t launch_bf16_chunk_stage1_k<512u, false>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, cudaStream_t, int64_t, int32_t*);
extern template cudaError_t launch_bf16_chunk_stage1_k<1024u, true>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, cudaStream_t, int64_t, int32_t*);
extern template cudaError_t launch_bf16_chunk_stage1_k<1024u, false>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, cudaStream_t, int64_t, int32_t*);

// Stage 2 of the static-k arm: the merge over `num_chunks * TOPK` candidates.
template <uint32_t TOPK>
inline cudaError_t launch_bf16_chunk_stage2_k(
    const maca_bfloat16* candidate_values, const int32_t* candidate_indices,
    int32_t* indices, int B, int num_chunks, cudaStream_t stream)
{
    if (TOPK > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    constexpr size_t kChunkSmem =
        kCoarse12HistBytes + sizeof(uint32_t) * (size_t)TOPK;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_bf16_chunk_stage2_kernel<kChunkBlockSize, TOPK>, kChunkSmem,
        smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;

    const int candidate_stride = num_chunks * static_cast<int>(TOPK);
    topk_bf16_chunk_stage2_kernel<kChunkBlockSize, TOPK>
        <<<B, kChunkBlockSize, kChunkSmem, stream>>>(
            candidate_values, candidate_indices, indices, candidate_stride,
            static_cast<int>(TOPK), B);
    return cudaGetLastError();
}

extern template cudaError_t launch_bf16_chunk_stage2_k<512u>(
    const maca_bfloat16*, const int32_t*, int32_t*, int, int, cudaStream_t);
extern template cudaError_t launch_bf16_chunk_stage2_k<1024u>(
    const maca_bfloat16*, const int32_t*, int32_t*, int, int, cudaStream_t);

// The static-k arm's two stages, in order.  `want_nan` picks which of stage 1's
// two instantiations runs; the launch itself belongs to the two functions
// above, so this one compiles no device code.
template <uint32_t TOPK>
inline cudaError_t launch_topk_bf16_chunked_k(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int num_chunks, cudaStream_t stream,
    int64_t score_stride = 0, bool want_nan = false,
    int32_t* nan_flags = nullptr)
{
    if (TOPK > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    cudaError_t err = want_nan
        ? launch_bf16_chunk_stage1_k<TOPK, true>(
              scores, lengths, candidate_indices, candidate_values, B, L,
              num_chunks, stream, score_stride, nan_flags)
        : launch_bf16_chunk_stage1_k<TOPK, false>(
              scores, lengths, candidate_indices, candidate_values, B, L,
              num_chunks, stream, score_stride, nan_flags);
    if (err != cudaSuccess) return err;
    return launch_bf16_chunk_stage2_k<TOPK>(candidate_values, candidate_indices,
                                            indices, B, num_chunks, stream);
}

// Stage 1 of the runtime-`topk` arm -- the `TOPK == 0` instantiation of the
// kernel above, which is why it is one instantiation per `kNan` instead of per
// (`TOPK`, `kNan`).
template <bool kNan>
inline cudaError_t launch_bf16_chunk_stage1_runtime(
    const maca_bfloat16* scores, const int32_t* lengths,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride, int32_t* nan_flags)
{
    if (topk > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_bf16_chunk_stage1_kernel<0u, kNan>, kSMEM, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;

    const int chunk_size = chunk_size_for(L, num_chunks);
    const int64_t stride = score_stride > 0 ? score_stride : (int64_t)L;
    topk_bf16_chunk_stage1_kernel<0u, kNan>
        <<<B * num_chunks, kChunkBlockSize, kSMEM, stream>>>(
            scores, lengths, candidate_indices, candidate_values, stride,
            topk, B, num_chunks, chunk_size, kNan ? nan_flags : nullptr);
    return cudaGetLastError();
}

extern template cudaError_t launch_bf16_chunk_stage1_runtime<true>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, int, cudaStream_t, int64_t, int32_t*);
extern template cudaError_t launch_bf16_chunk_stage1_runtime<false>(
    const maca_bfloat16*, const int32_t*, int32_t*, maca_bfloat16*, int, int,
    int, int, cudaStream_t, int64_t, int32_t*);

template <int kBlock>
inline cudaError_t launch_bf16_chunk_stage2_runtime(
    const maca_bfloat16* candidate_values, const int32_t* candidate_indices,
    int32_t* indices, int B, int num_chunks, int topk, cudaStream_t stream)
{
    if (topk > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_bf16_chunk_stage2_kernel<kBlock>, kSMEM, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;
    topk_bf16_chunk_stage2_kernel<kBlock><<<B, kBlock, kSMEM, stream>>>(
        candidate_values, candidate_indices, indices, num_chunks * topk, topk, B);
    return cudaGetLastError();
}

extern template cudaError_t launch_bf16_chunk_stage2_runtime<kChunkBlockSize>(
    const maca_bfloat16*, const int32_t*, int32_t*, int, int, int, cudaStream_t);

inline cudaError_t launch_topk_bf16_chunked(
    const maca_bfloat16* scores, const int32_t* lengths, int32_t* indices,
    int32_t* candidate_indices, maca_bfloat16* candidate_values,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride = 0, bool want_nan = false,
    int32_t* nan_flags = nullptr)
{
    if (topk > kMaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    switch (topk) {
    case 512:
        return launch_topk_bf16_chunked_k<512>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream, score_stride, want_nan, nan_flags);
    case 1024:
        return launch_topk_bf16_chunked_k<1024>(
            scores, lengths, indices, candidate_indices, candidate_values,
            B, L, num_chunks, stream, score_stride, want_nan, nan_flags);
    default:
        break;
    }

    cudaError_t err = want_nan
        ? launch_bf16_chunk_stage1_runtime<true>(
              scores, lengths, candidate_indices, candidate_values, B, L, topk,
              num_chunks, stream, score_stride, nan_flags)
        : launch_bf16_chunk_stage1_runtime<false>(
              scores, lengths, candidate_indices, candidate_values, B, L, topk,
              num_chunks, stream, score_stride, nan_flags);
    if (err != cudaSuccess) return err;
    return launch_bf16_chunk_stage2_runtime<kChunkBlockSize>(
        candidate_values, candidate_indices, indices, B, num_chunks, topk, stream);
}

// ============================================================
// 块宽选择 — `maca_topk.cu` 的 `radix_block_for` 是唯一消费者
// ============================================================
//
// `TopKConfig` is the triple that decision reads, and `needs_long_row_bf16`
// is the regime test itself; the block widths, the launch's dynamic smem and
// the chunked split's workspace are `maca_topk.cu`'s (`radix_smem_bytes`,
// `chunked_workspace_bytes`).
//
struct TopKConfig {
    int B;
    int L;
    int topk;
};

inline bool needs_long_row_bf16(const TopKConfig& cfg)
{
    // The measured crossover is specific to the 16 KiB C500 row variant. A
    // larger arena, a short batch, or an unsupported k keeps the 512 path so
    // compact-fit and small-row latency do not regress.
    if (kBlockSize != 512 || kSMEM > 16 * 1024 || cfg.B <= 0 || cfg.B > 4096 ||
        cfg.L < 60000)
        return false;
    return cfg.topk == 512 || cfg.topk == 1024;
}

// stage 1's answer is a *value*, so the merge can rank `chunks * topk` of them
// and map positions back.  A slot stage 1 could not fill takes `-inf`, which
// cannot appear in a real input: `is_nan_value` gates NaNs out before any
// selection runs, so no element is NaN and -inf is unreachable from the row.
//
// `kNan` is the caller's `check_nan`: with it on, this kernel is *also* the NaN
// scan, and `flags` -- the table `nan_scan_kernel` used to raise -- is its
// output.  The two share the grid by construction (`nan_scan_kernel`'s comment
// says so), so which kernel reads the bytes is the only thing that changes;
// the answer is the same predicate over the same window.
template <bool kNan = false>
__global__ __launch_bounds__(kBlockSize) void topk_f32_chunk_stage1_kernel(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int64_t score_stride, int topk, int B, int num_chunks, int chunk_size,
    int candidate_stride, int32_t *flags = nullptr)
{
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;

    const int32_t length = lengths[bid];
    const int start = chunk * chunk_size;
    const int chunk_len = (start < length)
                              ? min(chunk_size, static_cast<int>(length - start))
                              : 0;
    constexpr int BLOCK_SIZE = kBlockSize;
    int32_t *chunk_cols = cols + bid * candidate_stride + chunk * topk;
    float *chunk_vals = vals + bid * candidate_stride + chunk * topk;
    const float *row = scores + (int64_t)bid * score_stride;

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            chunk_cols[i] = -1;
            chunk_vals[i] = -__builtin_huge_valf();
        }
        return;
    }

    if (chunk_len <= topk) {
        // This window is short enough that the kernel already touches every one
        // of its bytes, so the NaN test rides the copy that was happening
        // anyway -- there is no second read to save here, only a kernel.
        bool nan_local = false;
        for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
            if (i < chunk_len) {
                const float v = __ldg(row + start + i);
                chunk_cols[i] = start + i;
                chunk_vals[i] = v;
                if (kNan) nan_local |= is_nan_bits32(__float_as_uint(v));
            } else {
                chunk_cols[i] = -1;
                chunk_vals[i] = -__builtin_huge_valf();
            }
        }
        if (kNan && flags != nullptr && __syncthreads_or((int)nan_local) != 0
            && threadIdx.x == 0)
            atomicOr(flags + bid, 1);
        return;
    }

    // The row entry, on this chunk's window, answering in row columns.  Its
    // `-1` sentinel is a column index that cannot be real, so it becomes the
    // same empty-slot marker the two arms above write.
    bool nan_row = false;
    radix_topk_row_f32<kNan>(row + start, chunk_cols, (uint32_t)chunk_len,
                             (uint32_t)topk, (uint32_t)start,
                             kNan ? &nan_row : nullptr);
    if (kNan && flags != nullptr && nan_row && threadIdx.x == 0)
        atomicOr(flags + bid, 1);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t col = chunk_cols[i];
        if (col >= start && col < start + chunk_len) {
            chunk_vals[i] = __ldg(row + col);
        } else {
            chunk_cols[i] = -1;
            chunk_vals[i] = -__builtin_huge_valf();
        }
    }
}

// The merge.  It ranks the candidate *values* -- so its answer is a position
// into the candidate arrays, which is what lets `cols` be mapped back -- and
// rewrites `topk` columns.
template <int kBlock>
__global__ __launch_bounds__(kBlock) void topk_f32_chunk_stage2_kernel(
    const float *vals, const int32_t *cols, int32_t *out, int topk, int B,
    int candidate_stride)
{
    const int bid = blockIdx.x;
    if (bid >= B) return;

    const float *row_vals = vals + bid * candidate_stride;
    const int32_t *row_cols = cols + bid * candidate_stride;
    int32_t *row_out = out + (int64_t)bid * topk;
    constexpr int BLOCK_SIZE = kBlock;

    radix_topk_row_f32(row_vals, row_out, (uint32_t)candidate_stride,
                       (uint32_t)topk);
    for (int i = threadIdx.x; i < topk; i += BLOCK_SIZE) {
        const int32_t pos = row_out[i];
        row_out[i] = (pos >= 0 && pos < candidate_stride) ? row_cols[pos] : -1;
    }
}

// ── the fp32 chunked split ──────────────────────────────────────────────────
//
// The 16-bit split above ranks each chunk with `radix_topk_row_bf16_b`, whose
// output is a position *inside the chunk*, so its merge ranks `chunks * topk`
// candidate values and maps positions back.  The fp32 row entry writes row
// columns instead (its `chunk_begin` is the chunk's column offset), so the
// split carries the columns alongside the values:
//
//   cols[batches * chunks * topk]  int32  the row columns stage 1 selected
//   (pad to a 16-byte boundary)            -- see `chunked_f32_vals_offset`
//   vals[batches * chunks * topk]  float  their values, -inf for an empty slot
//   out [batches * topk]           int32  the merge's answer, one column each
//
// Stage 2 ranks `vals` and maps the resulting position through `cols`.
// Everything else -- the grid, the `-1` sentinel, the contract half's `rerank`
// fallback -- is the 16-bit split's.
//
// The chunk count, the work target behind it and the reason a short batch is
// split at all are the caller's (`maca_topk.cu`'s `f32_chunk_work_target`).
constexpr size_t kF32ChunkBlocks = kBlockSize;   // the launch width, named once
// The array bound the engine is built to, and the only thing this header needs
// to know about the caller's `topk`: `chunked_f32_workspace_bytes` takes the
// real value as an argument, so the constant is here to bound the `uint32_t`
// arithmetic in this file and to be the `<=` in `f32_chunk_engine_supports`.
// It is a *topk* bound, not a chunk-count one.
constexpr uint32_t kF32MaxTopK = 4096u;

inline size_t chunked_f32_workspace_bytes(uint32_t batches, uint32_t topk,
                                          uint32_t chunks);

// The three regions of the fp32 split's arena, in order, as the *caller* must
// carve them (`chunked_f32_workspace_bytes` is the total):
//
//   cols[candidates]  int32  the row columns stage 1 selected
//   (pad to 16)              -- see `chunked_f32_vals_offset`
//   vals[candidates]  float  their values, -inf for an empty slot
//   out [candidates]  int32  the merge's answer, one column per row
//
// `out` is `candidates` wide here rather than `batches * topk` because that is
// the slot the caller actually has: `merged` sits directly after `vals`, and
// `candidates = batches * chunks * topk >= batches * topk` since `chunks >= 2`.
// The merge only writes `batches * topk` of it.
//
// `chunked_f32_cols` is the identity on the arena base; it exists so the
// offset that *does* need explaining has somewhere to point.
inline int32_t *chunked_f32_cols(void *base, uint32_t batches, uint32_t topk,
                                 uint32_t chunks) {
    (void)batches; (void)topk; (void)chunks;
    return (int32_t *)base;
}

// `vals` must start on a 16-byte boundary.  `radix_topk_row_f32` -- which
// stage 2 hands it to -- walks it with `__ldg(reinterpret_cast<const float4*>)`
// and has **no** alignment guard (`hist_add_f32`, pass 2's two loops); the
// 16-bit lane guards the same hazard (`bf16x8_is_aligned`), this one never
// learned to.  The arena base is a `cudaMalloc`, so it is aligned, and `cols`
// starts there (`chunked_f32_cols`); what can move `vals` off a 16-byte
// boundary is the size of the `cols` region, `batches * chunks * topk` int32.
//
// `chunks` is even by construction (`f32_chunked_chunks` returns a power of two
// or `f32_chunk_round_even`'s result) and `topk >= 512` on every arm the split
// is measured on, so `chunks * topk` is even and the offset is a multiple of 8
// bytes -- the padding is `0` or `8`, and the answer is determined by
// `batches & 1u`.  Written as a round-up so it stays correct if `chunks` ever
// becomes odd.
//
// Without it `vals` sits at `batches * chunks * topk * 4` bytes: 8 mod 16
// whenever `batches` is odd, which is every `b1-*` fp32 cell in the perf grid
// (`b1-v66551-k2048`, `b1-v131072-k2048`, `b1-v107520-k2048`).  The flag table
// that used to sit in front of the arena moved `cols` too, but that prefix is
// the dispatcher's own allocation now; the region size alone decides the pad.
// MACA tolerates the misaligned load, which is exactly why the old gap
// survived: it is UB that works, not a visible mis-computation.
inline size_t chunked_f32_vals_offset(uint32_t batches, uint32_t topk,
                                      uint32_t chunks) {
    const size_t bytes = (size_t)batches * chunks * topk * sizeof(int32_t);
    return (bytes + 15u) & ~(size_t)15u;
}

inline float *chunked_f32_vals(void *base, uint32_t batches, uint32_t topk,
                              uint32_t chunks) {
    return (float *)((char *)base + chunked_f32_vals_offset(batches, topk, chunks));
}

inline size_t chunked_f32_workspace_bytes(uint32_t batches, uint32_t topk,
                                          uint32_t chunks) {
    const size_t candidates = (size_t)batches * chunks * topk;
    // cols (padded) + vals + the merge's answer.  The last term is slack rather
    // than a separate region: `chunked_f32_workspace` places `merged` directly
    // after `vals`, and `candidates >= batches * topk` because `chunks >= 2`,
    // so the answer always fits in the `candidates`-wide slot -- this keeps the
    // bound it has always had rather than tightening it.
    return chunked_f32_vals_offset(batches, topk, chunks) +
           candidates * sizeof(float) + candidates * sizeof(int32_t) +
           (size_t)batches * topk * sizeof(int32_t);
}

inline int32_t *chunked_f32_out(void *base, uint32_t batches, uint32_t topk,
                                uint32_t chunks) {
    return (int32_t *)(chunked_f32_vals(base, batches, topk, chunks) +
                       (size_t)batches * chunks * topk);
}

// One CTA per (batch, chunk).  The chunk geometry comes from `chunk_size_for`
// -- one definition shared with `nan_scan_kernel` -- so the scan answers for
// exactly the window this split stages.  With `want_nan` the scan is not a
// separate launch but *absorbed*: this launch is the only one that walks the
// row, and `nan_flags` comes back from it.
// Stage 1 of the fp32 split.  **One launcher, one kernel** -- the same rule as
// the 16-bit stages above: the generated TU for this tuple is where the device
// code and its only launch both live.
template <bool kNan>
inline cudaError_t launch_f32_chunk_stage1(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride, int32_t *nan_flags)
{
    if (topk > (int)kF32MaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_f32_chunk_stage1_kernel<kNan>, kSMEM, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;
    const int chunk_size = chunk_size_for(L, num_chunks);
    const int candidate_stride = num_chunks * topk;
    topk_f32_chunk_stage1_kernel<kNan>
        <<<B * num_chunks, (int)kF32ChunkBlocks, kSMEM, stream>>>(
            scores, lengths, cols, vals, score_stride, topk, B, num_chunks,
            chunk_size, candidate_stride, kNan ? nan_flags : nullptr);
    return cudaGetLastError();
}

extern template cudaError_t launch_f32_chunk_stage1<true>(
    const float *, const int32_t *, int32_t *, float *, int, int, int, int,
    cudaStream_t, int64_t, int32_t *);
extern template cudaError_t launch_f32_chunk_stage1<false>(
    const float *, const int32_t *, int32_t *, float *, int, int, int, int,
    cudaStream_t, int64_t, int32_t *);

template <int kBlock>
inline cudaError_t launch_f32_chunk_stage2(
    const float *vals, const int32_t *cols, int32_t *out,
    int B, int topk, int num_chunks, cudaStream_t stream)
{
    if (topk > (int)kF32MaxTopK || num_chunks <= 1) return cudaErrorInvalidValue;
    static bool smem_adopted = false;
    const cudaError_t smem_err = adopt_dynamic_smem(
        topk_f32_chunk_stage2_kernel<kBlock>, kSMEM, smem_adopted);
    if (smem_err != cudaSuccess) return smem_err;
    const int candidate_stride = num_chunks * topk;
    topk_f32_chunk_stage2_kernel<kBlock><<<B, (int)kBlock, kSMEM, stream>>>(
        vals, cols, out, topk, B, candidate_stride);
    return cudaGetLastError();
}

extern template cudaError_t launch_f32_chunk_stage2<(int)kF32ChunkBlocks>(
    const float *, const int32_t *, int32_t *, int, int, int, cudaStream_t);

// stage 1, then stage 2 -- the split's entry point for the fp32 row.
inline cudaError_t launch_topk_f32_chunks_stage1(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride, bool want_nan = false, int32_t *nan_flags = nullptr)
{
    return want_nan ? launch_f32_chunk_stage1<true>(
                          scores, lengths, cols, vals, B, L, topk, num_chunks,
                          stream, score_stride, nan_flags)
                    : launch_f32_chunk_stage1<false>(
                          scores, lengths, cols, vals, B, L, topk, num_chunks,
                          stream, score_stride, nan_flags);
}

inline cudaError_t launch_topk_f32_chunks_stage2(
    const float *vals, const int32_t *cols, int32_t *out,
    int B, int topk, int num_chunks, cudaStream_t stream)
{
    return launch_f32_chunk_stage2<(int)kF32ChunkBlocks>(
        vals, cols, out, B, topk, num_chunks, stream);
}

// stage 1, then stage 2.  `cols_base` is the start of the candidate arena the
// split carved out (`chunked_f32_cols`); `out` must have room for `topk`
// columns per row.  They are separate arguments because `chunked_f32_vals` may
// pad the columns region by 8 bytes to keep `vals` 16-byte aligned, so the
// caller composes the layout once and hands its regions over.
//
// This is the *engine* bound, not the policy one: `chunked_f32_applies` (in
// `maca_topk.cu`) decides whether to split and calls this to find out whether
// the split can serve the topk.  Keeping them separate is what stops the policy
// from looking more expensive than the engine actually is.
inline bool f32_chunk_engine_supports(int topk) {
    // The fp32 split is dynamic-k: stage1/stage2 are templated on `kBlockSize`
    // alone and take `topk` as an argument (`radix_topk_row_f32`'s `topk` is a
    // runtime parameter too), so the engine's bound is the array limit, not a
    // switch.  The `512 || 1024` that used to sit in `chunked_f32_applies` was
    // a *policy* bound from where the split was measured, not an instantiation
    // one -- which is what plan §21.2 says and this makes explicit.
    return topk > 0 && topk <= (int)kF32MaxTopK;
}

// Whether stage 1 carries the contract's NaN scan in its own first pass.  The
// knob is the point: `DS_FUSE_NAN_OFF` puts the separate `nan_scan_kernel` back
// on the same binary, so the two arms differ in one boolean and nothing else --
// same chunk count, same grid, same stage 2.  Read once per process; this is a
// per-call-invariant decision, not a per-call one.
inline bool f32_stage1_fuses_scan() {
    static const bool on = [] {
        const char *v = std::getenv("DS_FUSE_NAN_OFF");
        return v == nullptr || v[0] == '\0' || v[0] == '0';
    }();
    return on;
}

inline cudaError_t launch_topk_f32_chunked(
    const float *scores, const int32_t *lengths, int32_t *cols, float *vals,
    int32_t *out, int B, int L, int topk, int num_chunks, cudaStream_t stream,
    int64_t score_stride, bool want_nan = false, int32_t *nan_flags = nullptr)
{
    // The three arenas are **arguments, not derived here**.  The caller already
    // owns the workspace and places the split's regions inside it, so it is the
    // one that knows the layout; re-deriving it here was a second copy of the
    // same arithmetic, and the two agreed only because they were written the
    // same way.  `chunked_f32_cols` / `chunked_f32_vals` / `chunked_f32_out`
    // remain as the definition of the layout, and the caller uses them.  The
    // per-row NaN table is **not** a region of this workspace -- it is the
    // dispatcher's own `scan_flags` allocation, passed separately, because a
    // geometry-dependent offset into shared scratch is not a flag slot.
    cudaError_t err = launch_topk_f32_chunks_stage1(
        scores, lengths, cols, vals, B, L, topk, num_chunks, stream, score_stride,
        want_nan, nan_flags);
    if (err != cudaSuccess) return err;
    return launch_topk_f32_chunks_stage2(
        vals, cols, out, B, topk, num_chunks, stream);
}

}  // namespace topk_select_radix

#endif  // RADIX_CHUNKED_CUH
