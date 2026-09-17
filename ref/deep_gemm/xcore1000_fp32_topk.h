// Torch-free public interface to the deep_gemm fp32 indexer TopK selection
// kernel (MetaX/MXMACA, target xcore1000).
//
// This is a faithful extraction of the kernel layer of
//   mcDeepGEMM/csrc/kernels/fp32_topk.cu  (lines 1..1584, `namespace detail`)
// with the torch/pybind host wrapper (lines 1585+) dropped and the one
// `torch::empty` inside `_launch_topk_chunks` replaced by a caller-owned
// workspace pointer.
//
// Two things the original host wrapper decided are NOT reproduced here, because
// both need the page-table plumbing of the `Transform = true` entry points:
//   * `seq_starts` (a per-sequence row offset into `scores`)
//   * `topk_values` (a second output holding the gathered scores)
// The untransformed selector is exposed, i.e. `Transform = false`.

#pragma once

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace deep_gemm::indexer {

// Which of the three policies `select_topk_policy` picks for a shape, natively
// (no torch).  Returns the `TopKPolicy` enumerator:
//   0 = Auto (never returned), 1 = Chunks, 2 = Single, 3 = Coarse12.
//
// `num_sms` is the device's AP/SM count, as
// `at::cuda::getDeviceProperties(...)->multiProcessorCount` reports it.  It is
// read only when `n_rows <= 32` (see `resolve_chunk_count`), so pass the real
// number if that can happen and 0 otherwise.
int32_t deep_gemm_topk_policy(
    int64_t n_rows, int64_t n_cols, int32_t top_k, int num_sms);

// Which chunk count the chunks policy will resolve to, i.e.
// `select_chunk_count(n_rows, n_cols, num_sms * 2)` -- twice the AP count,
// which is what `launch_topk_chunks` passes (the policy decision above passes
// the undoubled count; `select_chunk_count` ignores it except as a tail-wave
// threshold, and either way the answer is 0 or 3..6).  Same value
// `deep_gemm_topk_selector` computes internally; 0 means "fall back to
// topk_single".
int32_t deep_gemm_topk_chunk_count(int64_t n_rows, int64_t n_cols, int num_sms);

// Bytes required for `chunks_workspace` when the chunks policy resolves to
// `chunk_count` CTAs per row (valid: 3..6, the range `select_chunk_count`
// returns).  Returns 0 for any other count.
size_t deep_gemm_topk_chunks_workspace_bytes(int64_t n_rows, int chunk_count);

// Top-k over the last `n_cols` columns of `scores`, one row per `scores` row.
//
//   scores          device pointer, (n_rows, n_cols) fp32, row-major, contiguous
//   seq_lens        device pointer, n_rows int32.  Per row the selector scans
//                   only `scores[row, 0..seq_lens[row])`, i.e. the row's live
//                   prefix; pass `n_cols` everywhere to scan whole rows.
//   out_indices     device pointer, (n_rows, top_k) int32, column indices into
//                   `scores` (NOT into the scanned prefix -- no `seq_starts`
//                   here, so the row begins at column 0).  Where the live
//                   prefix is shorter than `top_k` the tail is filled with -1.
//   out_values      device pointer, (n_rows, top_k) fp32, or null.  When given,
//                   receives the scores gathered at `out_indices` (-inf where
//                   the index is -1), computed by the kernel itself.
//   chunks_workspace device pointer, or null when the policy is not Chunks.
//                   Size from `deep_gemm_topk_chunks_workspace_bytes`.  It is
//                   NOT zeroed by the caller -- the first kernel of the chunks
//                   sequence initializes it.
//   stream          CUDA stream to launch on; may be null for the default.
//
// All three policies write `out_indices[row, 0..top_k)` with the highest
// `top_k` values of the scanned prefix in descending order; ties are broken by
// whatever the kernel's radix passes happen to produce, so compare results as
// multisets, not element-wise.
void deep_gemm_topk_selector(
    const float *scores, const int32_t *seq_lens, int32_t *out_indices,
    float *out_values, int32_t *chunks_workspace, int64_t n_rows, int64_t n_cols,
    int32_t top_k, cudaStream_t stream);

}  // namespace deep_gemm::indexer
