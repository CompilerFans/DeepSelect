// DeepJIT feed: the bf16 chunk stage1 kernel as ONE self-contained kernel.
//
// The body is the production one: it calls `topk_bf16_chunk_stage1_body`, the
// device-inline function `topk_bf16_chunk_stage1_kernel` wraps in the AOT
// build, so the probe measures the shipping dataflow rather than a copy of it.
// What a feed adds is the width -- `KCHUNK_BLOCK_SIZE` is a template parameter
// the AOT instantiation set does not vary -- and that is the whole reason this
// file exists (measured 2026-10-25: an instantiation TU carries exactly one
// device kernel, so kernel count is not one).
#include <csrc/maca_kernels/common/radix_chunked.cuh>

#ifndef STAGE1_TOPK
#define STAGE1_TOPK 512
#endif

using namespace topk_select_radix;

extern "C" __global__ void stage1_probe(
    const maca_bfloat16* scores, const int32_t* lengths,
    int32_t* candidate_indices, maca_bfloat16* candidate_values, int B, int L,
    int num_chunks, int64_t score_stride, int32_t* nan_flags)
{
    // kNan=true is the production default (`check_nan` on), so the probe keeps
    // paying what a shipping call pays.
    topk_bf16_chunk_stage1_body<kChunkBlockSize, (uint32_t)STAGE1_TOPK, true>(
        scores, lengths, candidate_indices, candidate_values, score_stride,
        STAGE1_TOPK, B, num_chunks, chunk_size_for(L, num_chunks), nan_flags);
}
