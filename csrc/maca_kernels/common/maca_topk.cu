// MACA-native row-wise top-K for DeepSelect -- the shipping backend.
//
// Implements the `topk` contract (`deep_select.interface.topk`) on primitives
// that exist on MACA.  One contract layer is shared by every mode: the
// `length <= topk` shortcut, the NaN bit-pattern check, the index offsets, the
// out-of-band fills, and the optional ordered emit.  One selection dataflow
// feeds it for both dtypes -- the ported radix core in `radix_core.cuh` (see
// its provenance banner): one histogram pass over the high key byte plus one
// vectorized collect pass that refines the threshold bin in shared memory, so
// two passes over the row whatever the key width.  The 16-bit row walks a bf16
// key, the 32-bit row an fp32 one; the dtype picks the row entry, not the
// dataflow.
//
// Two things to know before editing:
//
//   * Portable primitives only: `__shfl_down_sync`, `atomicAdd`, `__ldg`,
//     `__syncthreads`, `__syncthreads_or`.  No inline asm, no TMA, no mbarrier,
//     no cluster.
//   * The wave is 64 lanes, not 32.  Any mask or `/ 32` inherited from the
//     CUDA-era kernels is suspect -- see `csrc/maca_kernels/legacy/` for what
//     that looks like when it is wrong.
//
// The upstream `v3` (bf16) and `v3_fp32` trees are ported under `csrc/maca_kernels/legacy/`
// and are NOT built; `v3_cluster` is deleted (MACA has no cluster launch), so
// the shapes it served fall through to this kernel -- the same path every other
// shape takes, with `topk <= 1024 <= 4096` and no vocabulary bound, so they are
// served, only without the cluster-specific scheduling.
//
// This file is the shared half of every family's artifact: family <N> compiles
// `common/` plus its own tuple set in `csrc/maca_kernels/xcore<N>/instantiations/`.

#include <cuda_runtime.h>
// [MACA] 数据类型取 MACA 原生头，不经过 cu-bridge 的 <cuda_bf16.h> 兼容层
// （那里只是 `typedef maca_bfloat16 __nv_bfloat16;`）。本文件其余部分只用
// maca_bfloat16 / maca_bfloat162 这两个原生名字。
#include <maca_bfloat16.h>

#include <cstdint>
#include <type_traits>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <vector>

#include <csrc/structs.h>
// The NaN predicates and the order-preserving key encodes, shared with the
// dataflow headers below.
#include <csrc/maca_kernels/common/primitives.cuh>
// The ported C500 dataflow (see the provenance banner in that file).  It is
// included here so that this translation unit -- the one `setup.py` builds --
// is what proves it compiles under the mxcc/cu-bridge toolchain, and so that
// the bf16 selection path in `contract.cuh` is a header-only dependency.
#include <csrc/maca_kernels/common/radix_core.cuh>
// The contract half, the two ported routes -- each of which includes the
// contract half it hands its answer to -- and the routing that picks between
// them.  The ladder itself is `routing.cuh`'s `topk_launch`; what is left
// below is the scratch the ladder is handed and the tvm-ffi entry.
#include <csrc/maca_kernels/common/contract.cuh>
#include <csrc/maca_kernels/common/topk_coarse12.cuh>
#include <csrc/maca_kernels/common/topk_chunks.cuh>
#include <csrc/maca_kernels/common/routing.cuh>

// ── the tvm-ffi entry points ────────────────────────────────────────────────
//
// Torch-free: this TU no longer includes <torch/extension.h> or
// <ATen/cuda/CUDAContext.h>, so the extension carries no torch DT_NEEDED
// entry and its behavior is not tied to the host's torch build.  The contract
// checks that used to be `TORCH_CHECK` are `DS_HOST_CHECK`, the tensors are
// `tvm::ffi::TensorView`, and the stream comes from the FFI environment.
#include <csrc/ffi/ffi_error.h>
#include <csrc/ffi/ffi_tensor.h>

#include <tvm/ffi/extra/c_env_api.h>

#include <utility>
#include <vector>

namespace deep_select {

namespace dsf = deep_select::ffi;

// ── the chunked path's scratch, held across calls ───────────────────────────
//
// `cudaMalloc`/`cudaFree` on this runtime cost ~70-100 us per *pair* at this
// size class, and the cost is non-monotonic in size (a 12 MB allocation is
// nearly free; 300 KB is the worst case).  The split needs this workspace plus
// a row-length table on every call, so a fresh pair each time put a ~265 us
// host floor in front of a ~96 us kernel -- `e2e = max(host, device)`.
//
// So they are cached process-wide and **grown only**.  Growing rather than
// keying by shape is the design: a caller alternating between two batches would
// realloc on every switch under a shape-keyed cache, which is the cost this
// exists to remove.
//
// Three properties make reuse safe rather than merely fast:
//
//   * `chunked_workspace(base, batches, topk)` walks forward from `base` by the
//     *geometry of this call*, not by the capacity, so a buffer larger than this
//     call needs is correct by construction.
//   * the split's kernels write every slot they read back -- stage 1 fills each
//     candidate slot it is asked for, and the contract half reads and clears
//     each row's NaN vote -- so nothing is inherited from the previous call.
//   * the lengths table is keyed by `(batches, vocab_size)` -- both arguments of
//     this call -- and a key it has not seen gets its own buffer.  Its content
//     is `[vocab_size] * batches`, a pure function of that pair, so the pair
//     *is* the content; see `lengths_table`.
//
// The high-water mark is **not** bounded by one gate, and the number below is
// the one to check against a memory budget rather than the smallest one.
// `chunked_bf16_applies` (`batches <= 64`, `topk in {512, 1024}`) does bound
// the bf16 split at ~6.3 MB, but it does not govern the fp32 one: the fp32
// split is admitted by `chunked_f32_applies` (`batches <= 4096`,
// `vocab_size >= 32768`, `topk <= kF32MaxTopK`), which is a far wider gate, and
// the two branches below share this one `workspace` field.  The fp32 split also
// carries no `sm_count` test -- unlike `f32_coarse12_applies` and
// `f32_chunks_applies`, whose provenance halves are `kF32Coarse12MeasuredSmCount`
// -- so a C600U image takes this same branch with its own chunk count.
//
// The footprint is `batches * chunks * topk * 8` bytes for the candidates plus
// `batches * topk * 4` for the merged answer.  On the perf grid that peaks at
// `b4096 V131072 k2048` = **160 MiB** (held for the life of the process); at
// the gate's own edge it is `b4096 k4096` with a large `V`, which is gigabytes
// but also needs a tensor that does not fit on this box.  Plus `batches * 4`
// bytes per distinct `(batches, vocab_size)` for the row tables, which at
// `b4096` is 16 KB per shape visited.
//
// A bounded, one-time footprint in exchange for removing a per-call host cost
// larger than the kernel it fronts -- 265 us of host floor in front of a 96 us
// kernel.  Freeing instead of holding gives most of that back: free-then-malloc
// *should* be cheap, and measured on this runtime it is not.  A caller that
// cannot afford the peak has `DEEP_SELECT_NO_SCRATCH_CACHE=1`, which allocates
// per call and keeps only `retired` alive.

struct ChunkedScratch {
    std::mutex mu;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    // The row tables, one per distinct `(batches, vocab_size)` this process has
    // asked for, each holding `[vocab_size] * batches`.
    //
    // **A table is never rewritten for a different key.**  The policy this
    // replaced was one buffer plus an epoch, refilled whenever the pair moved
    // -- correct only for a caller that runs one shape at a time.  A caller
    // that has recorded the pointer (a CUDA graph) keeps
    // reading the address it captured, so rewriting that address with another
    // shape's lengths silently re-scopes the row the graph ranks: measured, a
    // `b132 V66551` capture replayed after a `b16 V131072` call ranked a
    // 131072-wide window against a 66551-wide row and 131 of 132 rows differed.
    // Retiring the old buffer does not help -- nothing was freed, the content
    // was overwritten -- so the fix is to stop reusing the address at all.
    //
    // The footprint is `batches * 4` bytes per distinct shape and the split's
    // own gate bounds `batches` at 64, so this is a few hundred bytes for every
    // shape a process visits, held for its life.
    struct LengthsEntry {
        int64_t batches;
        int64_t vocab;
        int32_t *table;
    };
    std::vector<LengthsEntry> lengths_tables;

    // The table for `(batches, vocab)`, allocated and filled on first use.  The
    // caller holds `mu` (`ScratchGuard`), which is what makes the vector safe
    // without a lock of its own.  `stream` orders the fill against the kernel
    // that will read it -- the same ordering the single-buffer version had.
    int32_t *lengths_table(int64_t batches, int64_t vocab, cudaStream_t stream) {
        for (LengthsEntry &e : lengths_tables) {
            if (e.batches == batches && e.vocab == vocab) return e.table;
        }
        int32_t *table = nullptr;
        DS_CUDA_RUNTIME_CHECK(
            cudaMalloc(&table, (size_t)batches * sizeof(int32_t)));
        std::vector<int32_t> host_lengths((size_t)batches, (int32_t)vocab);
        DS_CUDA_RUNTIME_CHECK(cudaMemcpyAsync(table, host_lengths.data(),
                                              (size_t)batches * sizeof(int32_t),
                                              cudaMemcpyHostToDevice, stream));
        lengths_tables.push_back(LengthsEntry{batches, vocab, table});
        return table;
    }
    // One `int32_t` per row, zeroed once when it grows (each call's contract
    // half clears whatever it raised).  This is the table
    // `topk_kernel_radix`'s preselected arm reads to decide whether the
    // row is poisoned -- and it is *read on every preselected path*, while
    // only the two splits used to own one (each embedded in its own workspace
    // layout).  The coarse12 route answers a row without ranking it, so it can
    // hand back a row a scan would have rejected, and it had no table to read:
    // `nan_flags` stayed null and the first `check_nan` read faulted.  This
    // buffer is that table for every route.
    int32_t *scan_flags = nullptr;
    size_t scan_flags_count = 0;
    // `RowParams::nan_abort_flags`, for the calls that can abort.  Grown like
    // the rest and never shrunk; what it is *not* is zeroed per call -- the
    // kernel writes every row's entry, so no call can read another call's
    // answer.
    int32_t *nan_abort_flags = nullptr;
    size_t nan_abort_flags_count = 0;
    // The coarse12 route's answer buffer: `n_rows * topk` int32 columns, which
    // are the row *positions* the kernel ranks.  It is separate from
    // `output_index` because the public entry's default `indices_type` is
    // int64, and a kernel writing int32 through the caller's pointer would
    // interleave.  Grown like the rest and never shrunk.
    int32_t *coarse12_cols = nullptr;
    size_t coarse12_cols_count = 0;
    // `RowParams::row_block_max` / `row_block_list`, one allocation each and
    // grown like the rest.  Both count in *blocks* (64 elements), so one entry
    // of `block_max_count` is one uint16 and one of `block_list_count` is one
    // uint32; they are always grown together but kept as two buffers because
    // they are two different widths.
    uint16_t *block_max = nullptr;
    size_t block_max_count = 0;
    uint32_t *block_list = nullptr;
    size_t block_list_count = 0;
    // The coarse12 route's own skip tables, one uint16 (the block's max bin)
    // and one uint32 (its index in the compacted list) per 16-element block of
    // every row.  Separate from the pair above because the block is 16 elements
    // here against 64 there -- see `RowParams::coarse12_block_max`.  Grown like
    // the rest and never shrunk.
    uint16_t *coarse12_bmax = nullptr;
    size_t coarse12_bmax_count = 0;
    uint32_t *coarse12_blist = nullptr;
    size_t coarse12_blist_count = 0;
    // The chunks route's per-row workspaces.  Separate from `coarse12_cols`
    // because the two have different shapes and different lifetimes: this one
    // is `n_rows * sizeof(TopKChunksWorkspace<NChunks>)` of opaque bytes, that
    // one is `n_rows * topk` int32 columns of answer.  One byte buffer serves
    // every `NChunks` instantiation -- the launch knows which one it is, and
    // the layout is identical for all of them.
    void *chunks_workspace = nullptr;
    size_t chunks_workspace_bytes = 0;
    // Every buffer this allocator has **grown past**, held for the life of the
    // process rather than freed.  See `scratch_retire` for why.
    std::vector<void *> retired;
};

// Grow-only scratch, and "grow" means "allocate and *retire* the old one", not
// "allocate and free it".
//
// `cudaFree` here hands the page back to the driver, and a CUDA graph captured
// against the smaller shape recorded the raw address of the buffer it was
// launched against -- `ChunkedScratch` is process-wide and its pointers are
// handed to the kernel layer directly, so the address a graph froze is the
// address the driver is now free to hand to the next allocation of that size.
// Every replay after a growth reads whatever the allocator put there.  Measured
// (`tests/cases/graph_capture.py --grow-probe`, three runs): capture at
// `b132 V66551`, grow with `b4096 V131072`, replay -> **~262k of 270336 indices
// differ**, deterministically.  A replay that happens to match means the page
// was not reused, not that this is safe.
//
// So the peak is held.  The cost is bounded by the number of distinct sizes one
// process grows through -- one or two in every workload measured -- and it buys
// the operator being capturable at all, which is the price a caller integrating
// it into a decode loop actually cares about.  This is not a leak in the
// sense that matters: it is a high-water mark, and it is the same policy the
// cache already applies to a shape that never shrinks.
//
// The mutex is held by the caller (`ScratchGuard`), which is what makes the
// push safe without a lock of its own.
inline void scratch_retire(std::vector<void *> &retired, void *buffer) {
    if (buffer != nullptr) retired.push_back(buffer);
}

ChunkedScratch &chunked_scratch() {
    static ChunkedScratch s;
    return s;
}

tvm::ffi::Array<int64_t> get_alignment_requirement() {
    return {INPUT_STRIDE_ALIGNMENT_REQUIREMENT,
            OUTPUT_STRIDE_ALIGNMENT_REQUIREMENT};
}

void topk(const tvm::ffi::TensorView &input, int64_t topk,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &end,
          bool sorted_value, bool sorted_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_value,
          const tvm::ffi::TensorView &output_index,
          const tvm::ffi::Optional<tvm::ffi::TensorView> &output_idx_offset,
          int64_t idx_oob_fill_value, double value_oob_fill_value,
          bool return_value, bool abort_when_nan_found, bool check_nan,
          // AP count of the device this call runs on, from the caller's
          // `get_device_properties`.  The last argument rather than one near
          // `topk` because it is the only one the caller derives from the
          // *device* rather than from the problem -- and the only device fact
          // that stays an argument now that the family constants are
          // compile-time (`csrc/structs.h`).  The shared-memory budget is
          // `ARCH_SMEM_PER_AP_BYTES`; the AP count cannot be, because the grid
          // has to be sized for the part in front of the call and not for the
          // one the image was built for.
          int64_t sm_count) {
    DS_HOST_CHECK(input.ndim() == 2, "input must be 2-D, got ", input.ndim());
    const int64_t batches = dsf::size(input, 0);
    const int64_t vocab_size = dsf::size(input, 1);
    const bool float32 = dsf::is_float32(input);
    const bool bfloat16 = dsf::is_bfloat16(input);

    DS_HOST_CHECK(topk > 0, "topk must > 0");
    DS_HOST_CHECK(topk <= (int64_t)kMaxTopK, "topk must be <= ", kMaxTopK);
    DS_HOST_CHECK(!(sorted_value && !return_value),
                  "`return_value` must be enabled when `sorted_value` is True");
    DS_HOST_CHECK(!(sorted_value && sorted_index),
                  "`sorted_value` and `sorted_index` cannot be used at the same time");
    DS_HOST_CHECK(float32 || bfloat16,
                  "input dtype must be float32 or bfloat16");
    DS_HOST_CHECK(dsf::is_index_type(output_index),
                  "output_index dtype must be int32 or int64");
    DS_HOST_CHECK(dsf::stride(input, 1) == 1, "input.stride(1) must be 1");
    DS_HOST_CHECK(dsf::stride(input, 0) * (int64_t)dsf::element_size(input) %
                          (int64_t)INPUT_STRIDE_ALIGNMENT_REQUIREMENT == 0,
                  "input.stride(0) must be a multiple of ",
                  INPUT_STRIDE_ALIGNMENT_REQUIREMENT, " bytes");

    // Every output row is addressed as `row * stride(0) + column`, so a
    // last-dimension stride other than 1 (or a row that is too short) writes
    // outside the columns the caller owns.  Upstream rejects both
    // (api.cu KU_CHECK_LAST_DIM_CONTIGUOUS / KU_CHECK_SHAPE) -- without the
    // check the result is silently scrambled, so refuse instead.
    auto check_out_tensor = [&](const char what[],
                                const tvm::ffi::TensorView &t) {
        DS_HOST_CHECK(t.device().device_id == input.device().device_id,
                      what, " must be on the same device as `input`");
        DS_HOST_CHECK(dsf::stride(t, 1) == 1, what, ".stride(1) must be 1");
        DS_HOST_CHECK(dsf::size(t, 0) == batches && dsf::size(t, 1) >= topk,
                      what, " must be at least (batch_size, topk) = (",
                      batches, ", ", topk, ")");
    };
    check_out_tensor("output_index", output_index);
    if (return_value) {
        DS_HOST_CHECK(output_value.has_value(),
                      "`output_value` must not be `None` when `return_value` is True");
        const tvm::ffi::TensorView &ov = output_value.value();
        DS_HOST_CHECK(dsf::same_dtype(ov.dtype(), input.dtype()),
                      "output_value dtype must match input dtype");
        check_out_tensor("output_value", ov);
    }
    // The per-row tables are read as `table[row]`, so `stride(0)` must be 1 and
    // the tensor must be on the device (a wrong-device pointer faults on read).
    auto check_row_table = [&](const char what[],
                               const tvm::ffi::TensorView &t) {
        DS_HOST_CHECK(t.device().device_id == input.device().device_id,
                      what, " must be on the same device as `input`");
        DS_HOST_CHECK(t.ndim() == 1 && dsf::size(t, 0) == batches &&
                          dsf::stride(t, 0) == 1,
                      what, " must be a contiguous tensor of `batch_size` entries");
    };
    if (end.has_value()) check_row_table("end", end.value());
    if (output_idx_offset.has_value()) {
        check_row_table("output_idx_offset", output_idx_offset.value());
    }

    const tvm::ffi::TensorView &ov =
        return_value ? output_value.value() : output_index;

    RowParams p{};
    p.input = dsf::const_data_ptr(input);
    p.output_value = return_value ? dsf::data_ptr(ov) : nullptr;
    p.output_index = dsf::data_ptr(output_index);
    p.end_ptr = end.has_value() ? dsf::data_ptr<int32_t>(end.value()) : nullptr;
    p.idx_offset_ptr = output_idx_offset.has_value()
                           ? dsf::data_ptr<int32_t>(output_idx_offset.value())
                           : nullptr;
    // The kernel offsets rows in *bytes*; `stride` counts elements.
    p.stride_input_batch = (uint64_t)dsf::stride(input, 0) * dsf::element_size(input);
    p.stride_output_value_batch =
        return_value ? (uint64_t)dsf::stride(ov, 0) * dsf::element_size(ov) : 0;
    p.stride_output_index_batch =
        (uint64_t)dsf::stride(output_index, 0) * dsf::element_size(output_index);
    p.vocab_size = (uint32_t)vocab_size;
    p.topk = (uint32_t)topk;
    // A zero here sizes every grid below to nothing -- an empty answer, not a
    // crash -- and the caller cannot fail to know it (it reads the number off
    // the device).  Refuse rather than round it to something plausible.
    DS_HOST_CHECK(sm_count > 0, "sm_count must be > 0 (got ", sm_count, ")");
    p.sm_count = (uint32_t)sm_count;
    // The *other* machine number -- the shared-memory budget -- is not read
    // here and not passed here: it is `ARCH_SMEM_PER_AP_BYTES`, a compile-time
    // constant of the family this artifact was built for (`csrc/structs.h`).
    // It used to be a `cudaDeviceGetAttribute` on this line, which is the same
    // number obtained a call later and from a place that did not need to be
    // asked: the artifact already knows its family, `_binding.py` loads the
    // artifact matching the device, and the only way the two can disagree is a
    // call that crossed devices -- which this file would rather refuse than
    // silently re-tune for.
    p.idx_fill = (int32_t)idx_oob_fill_value;
    p.value_fill = (float)value_oob_fill_value;
    p.check_nan = check_nan;

    const int value_dtype = float32 ? 0 : 1;
    const int index_dtype = dsf::same_dtype(output_index.dtype(), dsf::kInt32) ? 0 : 1;
    // Both tables below are allocated and handed over on this one answer, so the
    // two sites cannot drift apart: an allocated-but-unpassed table is a walk
    // that reads a summary nothing wrote.
    //
    // `value_dtype == 1` is bf16 (`float32 ? 0 : 1` above) -- the round is
    // compiled into the *16-bit* row's instantiation only, so this is the only
    // dtype that has a reader.  The floor is a row length; the second comparison
    // is a *density*, and it is the one that decides the sign of the change: a
    // 64-element block is kept when any element in it is at or above the
    // threshold, so the walk keeps `1 - (1 - k/V)^64` of the blocks.  Measured
    // on the official grid (docs/experiments/20261010_skim_rate/): `k/V = 1/128`
    // (39% kept) is a wash, `k/V = 1/64` (64% kept) costs +11..+16% -- the list
    // is traffic on top of a walk that then reads most of the row anyway.
    // The row path's skip tables.  Allocated for the calls whose shape passes
    // `skip_tables` and whose row instantiation carries the round.  The
    // 32-bit-index row always carries it; the 64-bit row carries it only where
    // the register file pays for the summary body -- `contract.cuh`'s
    // `kSkipRow` is the device half of this predicate and must agree with it:
    // an allocated-but-unread table is a walk that reads a summary nothing
    // wrote, and a read-but-unallocated one is a fault.
    const bool skip_tables =
        value_dtype == 1 &&
        (index_dtype == 0 || ARCH_REGS_PER_AP >= 131072) &&
        p.vocab_size >= detail::kSkipMinVocab &&
        (uint64_t)p.vocab_size >=
            (uint64_t)p.topk * detail::kSkipMinRowPerSlot &&
        detail::bf16_skip_enabled();
    // The FFI environment holds torch's current stream while the python
    // facade is inside `tvm_ffi.use_torch_stream()` (deep_select/_binding.py).
    // Outside it TVMFFIEnvGetStream reports the null handle, which is the
    // legacy default stream -- the same thing the torch build launched on
    // when no stream was set, so this is not a behavior change.
    const cudaStream_t stream = (cudaStream_t)TVMFFIEnvGetStream(
        (int32_t)dsf::device_type(input), dsf::device_index(input));

    // Both buffers are built HERE rather than taken from the caller.  The
    // lengths table is an internal detail of the split's stage 1 -- `end`
    // absent already means the whole row -- so keeping it here leaves the
    // public entry a pure DLTensor boundary.  The workspace is raw cudaMalloc
    // rather than a torch tensor, because a torch tensor here is exactly the
    // coupling the tvm-ffi migration removes; it is synchronous on the calling
    // thread, so it is ordered before the launches on any stream, and cached
    // across calls -- see `ChunkedScratch`.
    static const bool kCacheScratch = [] {
        const char *v = std::getenv("DEEP_SELECT_NO_SCRATCH_CACHE");
        return !(v != nullptr && v[0] == '1' && v[1] == '\0');
    }();

    // The cache is process-wide while the kernels here have no thread-safety
    // contract of their own, so the scratch carries its own lock rather than
    // assuming the caller serializes.  Held across the launches, because the
    // buffers are not done being read when this function returns.
    //
    // **The lock is taken on both branches, `kCacheScratch` or not.**  The
    // flag selects the *allocation policy* -- borrow from the cache versus
    // `cudaMalloc` per call -- and not whether the shared state is touched:
    // `coarse12_cols`, `chunks_workspace`, `scan_flags` and `retired` are
    // mutated below on either branch, and the `!kCacheScratch` branch is
    // exactly the one that allocates and retires most.  Passing `nullptr` here
    // when the flag was set left those four unguarded while the comment above
    // claimed otherwise.
    struct ScratchGuard {
        std::mutex *m;
        explicit ScratchGuard(std::mutex *mu) : m(mu) { m->lock(); }
        ~ScratchGuard() { m->unlock(); }
    };
    ChunkedScratch &scratch = chunked_scratch();
    ScratchGuard scratch_guard(&scratch.mu);
    bool scratch_borrowed = false;

    int32_t *lengths = nullptr;
    void *workspace = nullptr;
    size_t workspace_bytes = 0;
    // One scratch arm per split.  The two differ in dtype and in the workspace
    // layout they derive from it, so they are separate branches over a shared
    // allocator/table front half; the fp32 arm takes the same `p.end_ptr` route
    // (its split requires `end_ptr != nullptr`).
    const bool bf16_split =
        value_dtype == 1 && detail::chunked_bf16_applies(p, (uint32_t)batches);
    const bool f32_split =
        value_dtype == 0 && detail::chunked_f32_applies(p, (uint32_t)batches);
    // The per-row scan table.  Every route that answers a row **without**
    // ranking it reads this table in the contract half: the coarse12 arm, the
    // chunks arm, and both splits (they hand `params.scan_flags` to the
    // contract as its `nan_flags`).  It is provided for the whole `check_nan`
    // class rather than for the routes a shape predicate can predict, because
    // the route is chosen inside `topk_launch` -- and providing it costs one
    // allocation, not a memset: the table is zeroed once when it grows, and
    // each call's contract half reads and clears the slots that call could
    // have raised.  A plain row run under `check_nan` scans the row itself,
    // never reads this, and pays nothing for the class.
    //
    // **The splits must not raise into their own workspace.**  That region is
    // a geometry-dependent offset inside one grow-only scratch shared by
    // every split shape and both dtypes, so the slot a call reads is whatever
    // the previous call wrote at that offset -- a candidate index, not a flag
    // (measured: a b1 v262144 bf16 cell after a b6 v1048576 call returns the
    // NaN sentinel on a clean row).  This table is the one region that is only
    // ever flags.
    const bool needs_scan_table = check_nan;
    // The coarse12 answer buffer, allocated by the same rule: a call the route
    // can reach gets one, so `topk_launch` can stage the columns it ranks
    // whatever the caller's index width is.  Allocated up front rather than
    // inside the dispatch because the dispatch is the kernel layer and this is
    // a host allocation, and because a failure here has to fall back to the row
    // path rather than fault.  The `b <= 2` chunks arm shares it -- same kind of
    // answer (row columns), same contract half, and the two gates are disjoint.
    const bool needs_coarse12_cols =
        value_dtype == 0 && (detail::f32_coarse12_applies(p, (uint32_t)batches) ||
                             detail::f32_chunks_applies(p, (uint32_t)batches));
    if (needs_coarse12_cols) {
        const size_t need_cols = (size_t)batches * (size_t)p.topk;
        if (need_cols > scratch.coarse12_cols_count) {
            int32_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(
                cudaMalloc(&grown, need_cols * sizeof(int32_t)));
            scratch_retire(scratch.retired, scratch.coarse12_cols);
            scratch.coarse12_cols = grown;
            scratch.coarse12_cols_count = need_cols;
        }
        // The block tables ride on the same gate as the answer buffer: the
        // route that reads them is a subset of the calls this branch already
        // serves, and a call whose allocation fails still gets a correct answer
        // (the kernel walks densely when they are null), so there is nothing to
        // fall back to and nothing to check here.
        const size_t blocks_per_row =
            ((size_t)p.vocab_size + (size_t)topk_select_coarse12::kBlockElems - 1)
            / (size_t)topk_select_coarse12::kBlockElems;
        const size_t need_bmax = (size_t)batches * blocks_per_row;
        if (need_bmax > scratch.coarse12_bmax_count) {
            uint16_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(
                cudaMalloc(&grown, need_bmax * sizeof(uint16_t)));
            scratch_retire(scratch.retired, scratch.coarse12_bmax);
            scratch.coarse12_bmax = grown;
            scratch.coarse12_bmax_count = need_bmax;
        }
        if (need_bmax > scratch.coarse12_blist_count) {
            uint32_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(
                cudaMalloc(&grown, need_bmax * sizeof(uint32_t)));
            scratch_retire(scratch.retired, scratch.coarse12_blist);
            scratch.coarse12_blist = grown;
            scratch.coarse12_blist_count = need_bmax;
        }
    }
    // The row path's skip tables.  Allocated for the calls whose shape passes
    // `skip_tables` and whose row instantiation carries the round -- the
    // 32-bit-index row everywhere, the 64-bit row on the 128 KiB-regfile
    // parts (`kSkipRow` in `contract.cuh`); the routes that do not walk rows
    // never read them.  Sized by `vocab_size`, the table's own row length, so
    // a row with a shorter window still finds its own entry.
    if (skip_tables) {
        const size_t need_blocks = (size_t)batches * (((size_t)p.vocab_size + 63) / 64);
        if (need_blocks > scratch.block_max_count) {
            uint16_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(
                &grown, need_blocks * sizeof(uint16_t)));
            scratch_retire(scratch.retired, scratch.block_max);
            scratch.block_max = grown;
            scratch.block_max_count = need_blocks;
        }
        if (need_blocks > scratch.block_list_count) {
            uint32_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(
                &grown, need_blocks * sizeof(uint32_t)));
            scratch_retire(scratch.retired, scratch.block_list);
            scratch.block_list = grown;
            scratch.block_list_count = need_blocks;
        }
    }
    // The chunks route's workspace, on the same rule: sized for the row count
    // this call has, allocated once and grown, and its absence is a fallback to
    // the row path rather than a fault.  The count itself comes from
    // `select_chunk_count`, which needs the device's SM count -- read here
    // rather than passed in, because this is the sizing half and the launch
    // half has to agree with it exactly.
    if (value_dtype == 0 && detail::f32_chunks_applies(p, (uint32_t)batches)) {
        const size_t need_ws = topk_select_chunks::chunks_workspace_bytes(
            (uint32_t)batches, p.vocab_size, p.sm_count);
        if (need_ws > scratch.chunks_workspace_bytes) {
            void *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_ws));
            scratch_retire(scratch.retired, scratch.chunks_workspace);
            scratch.chunks_workspace = grown;
            scratch.chunks_workspace_bytes = need_ws;
        }
    }
    if (bf16_split || f32_split) {
        const int f32_chunks =
            detail::f32_chunked_chunks((uint32_t)batches, p.topk,
                                       p.vocab_size, p.sm_count);
        const size_t need_lengths = (size_t)batches * sizeof(int32_t);
        const size_t need_workspace =
            f32_split
                ? detail::chunked_f32_workspace_bytes((uint32_t)batches,
                                                      (uint32_t)topk, f32_chunks)
                : detail::chunked_workspace_bytes(
                      (uint32_t)batches, (uint32_t)topk,
                      (uint32_t)detail::chunked_chunks(p));

        if (!kCacheScratch) {
            if (!end.has_value()) {
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&lengths, need_lengths));
                std::vector<int32_t> host_lengths((size_t)batches,
                                                  (int32_t)vocab_size);
                DS_CUDA_RUNTIME_CHECK(cudaMemcpy(lengths, host_lengths.data(),
                                                 need_lengths,
                                                 cudaMemcpyHostToDevice));
                p.end_ptr = lengths;
            }
            DS_CUDA_RUNTIME_CHECK(cudaMalloc(&workspace, need_workspace));
            workspace_bytes = need_workspace;
        } else {
            // Grow-only.  A `cudaFree` here would be correct but would give back
            // exactly the cost this buffer exists to avoid, so the peak is held
            // -- and the old buffer is *retired* rather than freed, because a
            // captured graph may have recorded it (`scratch_retire`).
            if (need_workspace > scratch.workspace_bytes) {
                void *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, need_workspace));
                scratch_retire(scratch.retired, scratch.workspace);
                scratch.workspace = grown;
                scratch.workspace_bytes = need_workspace;
            }
            if (!end.has_value()) {
                // The row table, keyed by the *pair* that is its content.
                //
                // It used to be one buffer plus an epoch: refill when
                // `(batches, vocab_size)` differs from what was last written.
                // That is correct for a caller that runs one shape at a time
                // and wrong for a caller that has *recorded a pointer* -- a CUDA
                // graph.  Capture freezes `end_ptr` at the address this call
                // handed out; a later call at another shape then rewrote that
                // same address in place (measured: `b132 V66551` captured, a
                // `b16 V131072` call in between, replay -> row 0 ranked a
                // 131072-wide window against a 66551-wide row, 131 of 132 rows
                // differ).  `scratch_retire` does not help here: nothing was
                // freed, the content was overwritten.
                //
                // So the table is per-key and never reused across keys.  The
                // footprint is `batches * 4` bytes per distinct shape, and the
                // gate that admits the split bounds `batches` at 64 -- a few
                // hundred bytes for every shape a process visits.
                p.end_ptr = scratch.lengths_table(batches, (int64_t)vocab_size, stream);
            }
            workspace = scratch.workspace;
            workspace_bytes = need_workspace;
            scratch_borrowed = true;
        }
    }

    // The per-row scan table, owned here rather than by the route that reads
    // it.  Every `check_nan` call gets one (`needs_scan_table` above): which
    // route answers is decided inside `topk_launch`, after this point.
    //
    // **Zeroed when it is allocated, never per call.**  What makes that safe is
    // the contract half's invariant, not this buffer's: `topk_kernel_radix`
    // reads each row's slot and clears it in the same breath, before any arm
    // can return, so every slot a call could have raised is zero again by the
    // time that call's kernel completes (see `RowParams::nan_flags`).  The
    // fresh buffer is therefore the only place a stale 1 could come from, and
    // this one `cudaMemset` — once, when the scratch grows — replaces an
    // allocation-bound `cudaMemsetAsync` on every call (~4 us on a C600U, ~15%
    // of a b1 decode cell, measured).  A caller that ever supplies its own
    // table owns zeroing it.
    if (needs_scan_table) {
        if (p.scan_flags == nullptr) {
            if ((size_t)batches > scratch.scan_flags_count) {
                int32_t *grown = nullptr;
                DS_CUDA_RUNTIME_CHECK(cudaMalloc(&grown, (size_t)batches * sizeof(int32_t)));
                DS_CUDA_RUNTIME_CHECK(cudaMemset(grown, 0, (size_t)batches * sizeof(int32_t)));
                scratch_retire(scratch.retired, scratch.scan_flags);
                scratch.scan_flags = grown;
                scratch.scan_flags_count = (size_t)batches;
            }
            p.scan_flags = scratch.scan_flags;
        }
    }
    // The abort table: same grow-only pattern, opposite zeroing policy.  The
    // contract half writes a slot for every row before it can return, so a
    // memset here would buy nothing and cost the call ~11 us (see the field's
    // note).  It is allocated for the whole `abort_on_nan` class rather than
    // for the route that will need it, because the route is chosen inside
    // `topk_launch` and every route it can choose answers through the same
    // contract half.
    if (check_nan && abort_when_nan_found) {
        if ((size_t)batches > scratch.nan_abort_flags_count) {
            int32_t *grown = nullptr;
            DS_CUDA_RUNTIME_CHECK(
                cudaMalloc(&grown, (size_t)batches * sizeof(int32_t)));
            scratch_retire(scratch.retired, scratch.nan_abort_flags);
            scratch.nan_abort_flags = grown;
            scratch.nan_abort_flags_count = (size_t)batches;
        }
        p.nan_abort_flags = scratch.nan_abort_flags;
    }
    if (needs_coarse12_cols) {
        p.coarse12_cols = scratch.coarse12_cols;
        p.coarse12_block_max = scratch.coarse12_bmax;
        p.coarse12_block_list = scratch.coarse12_blist;
    }
    if (skip_tables) {
        p.row_block_max = scratch.block_max;
        p.row_block_list = scratch.block_list;
    }
    if (value_dtype == 0 && detail::f32_chunks_applies(p, (uint32_t)batches))
        p.chunks_workspace = scratch.chunks_workspace;

    // The cached branch hands out the scratch's pointers directly, so the guard
    // must not free them -- it owns only what this call allocated itself (the
    // uncached branch, and nothing at all once `DEEP_SELECT_NO_SCRATCH_CACHE` is
    // off).
    struct FreeIfSet {
        void *p;
        ~FreeIfSet() { if (p) cudaFree(p); }
    } free_lengths{scratch_borrowed ? nullptr : (void *)lengths},
        free_workspace{scratch_borrowed ? nullptr : workspace};

    topk_launch(p, batches, (void *)stream, value_dtype, index_dtype,
                sorted_index, sorted_value, return_value, workspace,
                workspace_bytes);
    // Last, because it reads the table the contract half fills.  Launched only
    // for the calls that can abort, so a `check_nan` call that answers with the
    // sentinel pays nothing for it.
    if (check_nan && abort_when_nan_found && p.nan_abort_flags != nullptr &&
        batches > 0) {
        const uint32_t n = (uint32_t)batches;
        launch_nan_abort<kNanAbortBlock>(p.nan_abort_flags, n, stream);
    }
    DS_CUDA_RUNTIME_CHECK(cudaGetLastError());
}

}  // namespace deep_select

#include <csrc/ffi/ffi_entries.h>

