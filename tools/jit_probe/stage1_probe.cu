// DeepJIT feed: the bf16 chunk stage1 kernel as ONE self-contained kernel.
// Written against the device-inline row core (`radix_topk_row_bf16_b`) rather
// than the instantiation TU because its point is to vary `KCHUNK_BLOCK_SIZE`,
// a template parameter the AOT set does not carry -- not because a TU would
// drag in more than one kernel (measured 2026-10-25: one TU, one device
// kernel, in every instantiation group).
//
// The body mirrors `topk_bf16_chunk_stage1_kernel<TOPK, kNan>` at `TOPK != 0`
// in `csrc/maca_kernels/common/radix_chunked.cuh`; keep it in step with that
// function (the chunk geometry, the rank, the candidate write, the NaN flag).
#include <csrc/maca_kernels/common/radix_row_bf16.cuh>

#ifndef STAGE1_TOPK
#define STAGE1_TOPK 512
#endif
constexpr int kStage1TopK = STAGE1_TOPK;

using namespace topk_select_radix;

extern "C" __global__ void stage1_probe(
    const maca_bfloat16* scores, const int32_t* lengths,
    int32_t* candidate_indices, maca_bfloat16* candidate_values, int B, int L,
    int num_chunks, int64_t score_stride, int32_t* nan_flags)
{
    const int global_bid = blockIdx.x;
    const int bid = global_bid / num_chunks;
    const int chunk = global_bid - bid * num_chunks;
    if (bid >= B) return;
    const int length = lengths ? lengths[bid] : L;
    const int chunk_size = ((length + num_chunks - 1) / num_chunks + 7) / 8 * 8;
    const int start = chunk * chunk_size;
    const int chunk_len = length - start < chunk_size ? length - start : chunk_size;

    const maca_bfloat16* row = scores + (int64_t)bid * score_stride;
    int32_t* chunk_indices = candidate_indices + (int64_t)global_bid * kStage1TopK;
    maca_bfloat16* chunk_values = candidate_values + (int64_t)global_bid * kStage1TopK;
    const maca_bfloat16 invalid_value = __float2bfloat16(-65504.0f);

    if (chunk_len <= 0) {
        for (int i = threadIdx.x; i < kStage1TopK; i += blockDim.x) {
            chunk_indices[i] = -1;
            chunk_values[i] = invalid_value;
        }
        return;
    }

    bool nan_row = false;
    radix_topk_row_bf16_b<kChunkBlockSize, true, false>(
        row + start, chunk_indices, (uint32_t)chunk_len, (uint32_t)kStage1TopK,
        &nan_row);
    if (nan_row && threadIdx.x == 0 && nan_flags != nullptr)
        atomicOr(nan_flags + bid, 1);
    for (int i = threadIdx.x; i < kStage1TopK; i += blockDim.x) {
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
