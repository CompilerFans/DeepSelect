// The reference's public edge -- one host entry, no torch, no tvm_ffi.
//
// Upstream's edge is `deep_select::topk(tvm::ffi::TensorView, ...)` in
// `csrc/xcore1000/maca_topk.cu` (lines 1112-1425).  Everything it did that the
// dataflow does not need is gone here: the dtype / shape / stride / device
// checks against `TensorView`s, the process-wide grow-only `cudaMalloc` scratch
// cache under a mutex, and the stream taken from the FFI environment.  What is
// left is the part the dataflow actually consumes -- the row pointers, the
// per-row window, the SM count every grid-sizing decision reads, and a
// workspace the caller owns.
//
// This is a C entry (`extern "C"`), because the point of a reference is that
// any host can call it.  It is defined in `xcore1000_maca_topk.cu`.

#ifndef XCORE1000_DS_TOPK_H
#define XCORE1000_DS_TOPK_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Row-wise top-k over an fp32 score matrix.
//
//   scores    n_rows * n_cols floats, row-major, row stride == n_cols.
//             The reference is packed-row only, so `stride_input_batch` is
//             derived here rather than passed.  Upstream's 1024-byte input
//             stride alignment requirement is the tvm-ffi layer's; a caller
//             that wants padded rows changes `stride_input_batch` in
//             `xcore1000_maca_topk.cu`'s entry and nothing else.
//   lengths   optional per-row window, `n_rows` entries, device memory,
//             `int32_t`.  A non-null table is passed straight through as
//             `RowParams::end_ptr`; null means every row is the whole of
//             `n_cols`, which is the table upstream builds when its caller
//             passes no `end`.  The table is *not* const because upstream's
//             `end` is an ordinary input tensor; the kernels only read it.
//             Heterogeneous windows are served on both dataflows, and a row
//             whose window is `<= top_k` takes the shortcut on either one.
//             Rows whose window is shorter than `n_cols` are still handed the
//             full row stride, so what bounds a window is `[0, n_cols]`.
//   out       n_rows * top_k int32, row-major, row stride == top_k.  Each row
//             receives exactly `top_k` distinct column indices into that row --
//             all of the row's largest ones, and as many of the next-largest
//             ties as are needed to fill the window.  A slot that cannot be
//             filled takes `-1` (`idx_fill`), which is what the `length <=
//             top_k` shortcut emits past the end of a short row.
//
//             ORDER IS NOT PART OF THIS CONTRACT.  The output is a *set*, and
//             the slots are in an unspecified, non-monotone order -- measured,
//             not assumed: across `B` in {6,8}, `V` in {16384, 65536, 200000}
//             and `k` in {512, 2048}, on both dataflows, no row came back
//             descending.  The reason is step 4 of the README's §2: the ordered
//             emit only runs when `sorted_value` / `sorted_index` is set, and
//             this entry fixes both false, so selection writes through
//             `atomicAdd(&s_counter, 1u)` and slot order follows atomic
//             scheduling.  (The `length <= top_k` shortcut is the exception and
//             is ascending by construction: slot i is column i.)  Compare
//             results as multisets, never element-wise -- see README §7.
//   n_rows    rows, one CTA each (or one CTA per (row, chunk) on the split).
//   n_cols    row length, i.e. `RowParams::vocab_size`.
//   top_k     k, 0 < top_k <= 4096 (`deep_select_maca::kMaxTopK`).
//   sm_count  access processors on the device this call runs on, from the
//             caller's architecture table (upstream: `deep_select/_arch.py`).
//             104 on this box.  The only value rejected is zero -- it sizes
//             every grid below to nothing, and upstream's `topk()` refuses it.
//
// The two dataflows are selected inside, by the same predicates the shipped
// dispatcher uses: `detail::chunked_f32_applies` / `topk_worth_splitting_f32`
// pick the chunked split for long rows at a small-ish batch, and the row path
// (`topk_kernel_radix`) answers everything else.  See README.md.
//
// Only fp32 with int32 indices is instantiated here.  Upstream also serves
// bf16 and int64 indices from the same contract layer by widening two
// template arguments; the reference narrows them to `ValueT = float`,
// `OutIdxT = int32_t`, and `return_value` / `sorted_index` / `sorted_value`
// are all false, so `output_value`, `output_idx_offset`, `idx_fill` and
// `value_fill` are fixed at their upstream defaults.
void ds_topk(const float* scores, const int32_t* lengths, int32_t* out,
             int n_rows, int n_cols, int top_k, int sm_count);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // XCORE1000_DS_TOPK_H
