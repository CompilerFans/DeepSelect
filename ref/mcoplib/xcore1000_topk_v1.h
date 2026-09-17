// Plain public header for the torch-free extraction of mcoplib's SGLang
// TopK kernel (mcoplib/op/sglang/jit_kernels/topk_v1.cu).
//
// The torch entry point `topk_transform_v1_interface` is replaced by
// `mcoplib_topk_transform`. The dispatch logic it performed is preserved:
//
//   * per-row `seq_len <= TopK (512)`      -> naive_transform (pad with -1)
//   * `max(seq_lens) <= 16384`             -> histogram_4096 (single pass)
//   * otherwise                            -> radix_topk (two pass)
//   * plus the 64K/128K-smem staging-size branch inside the radix path
//
// The interface is deliberately unchanged from topk_v1.cu's data model: the
// page-table index transform stays fused into the same kernel, exactly as it is
// in the source (page_to_indices, topk_v1.cu:92-96 and 399).
//
// page_bits notes
//   The source entry derives page_bits from a power-of-two page_size
//   (topk_v1.cu:452-455). Here it is passed directly.
//   page_bits = 0 gives an identity transform, because page_to_indices with
//   mask == 0 reduces to page_table[i >> 0] << 0, i.e. page_table[i]. Pass a
//   table of [0, 1, 2, ...] to make out_indices equal the raw selected indices,
//   which is what a plain TopK selector returns.
//
// kMaxLen caveat
//   The histogram_4096 path is a fixed-capacity kernel: kMaxLen = 16384
//   (kVecsPerThread 4 * 4 floats * kBlockSize 1024, topk_v1_histogram_4096.cuh:33-34).
//   This entry dispatches on the batch maximum, so it is safe. If the dispatch
//   is bypassed and a row longer than 16384 is fed to the histogram kernel, the
//   tail is silently dropped: the load loop stops emitting once
//   idx >= length and the element count past 16384 is never histogrammed.

#ifndef XCORE1000_TOPK_V1_H_
#define XCORE1000_TOPK_V1_H_

#include <cstdint>

#ifdef __cplusplus
extern "C" {
#endif

// Select the top `top_k` scoring columns of each row of `scores`, then apply the
// paged-attention index transform to the selected indices.
//
//   scores       [n_rows, n_cols] fp32, row stride == n_cols
//   seq_lens     [n_rows] int32; the valid prefix length of each row
//   out_indices  [n_rows, top_k] int32, contiguous; receives
//                page_table[sel >> page_bits] << page_bits | (sel & mask),
//                or -1 in the padded slots when seq_lens[r] <= top_k
//   page_table   one table shared by all rows, laid out as unique page numbers
//                (see page_bits); not per-row strided
//   n_rows       batch size; one thread block per row
//   n_cols       row stride of `scores` (physical columns, >= max(seq_lens))
//   top_k        must be 512 -- this is a compile-time constant in the kernel
//   page_bits    log2(page_size); 0 means an identity transform
//
// Only the default stream is used; synchronise on it before reading the output.
void mcoplib_topk_transform(const float* scores,
                            const int32_t* seq_lens,
                            int32_t* out_indices,
                            const int32_t* page_table,
                            int n_rows,
                            int n_cols,
                            int top_k,
                            int page_bits);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XCORE1000_TOPK_V1_H_
