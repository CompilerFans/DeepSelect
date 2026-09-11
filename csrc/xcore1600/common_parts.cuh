#pragma once

#include <cub/cub.cuh>
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "utils.cuh"
#include "config.h"
#include "bit_utils.cuh"

// [MACA] 原头部还有两行 Hopper 专有依赖，已拆除：
//   #include <cutlass/arch/barrier.h>        —— MACA 上无此文件（mctlass/arch 下没有 barrier.h）
//   #include <cute/arch/copy_sm90_tma.hpp>   —— TMA，MACA 没有
// 以及 `using transac_bar_t = kerutils::transac_bar_t;` —— 该别名已在
// kerutils/device/cuda/common.h 中删除，依赖它的预取流水线在本文件中整段拆除。
//
// 原先 CUTE_ALIGNAS / cute::cast_smem_ptr_to_uint 是靠 cutlass/cutlass.h 顺带
// 带进来的（它同时给出 `#define cutlass mctlass` 的别名）。这里只按名字取用，
// 不再整包引入 cutlass，改为直接包含这两个符号所在的、mctlass 里真实存在的头。
#include <cute/container/alignment.hpp>   // CUTE_ALIGNAS
#include <cute/arch/util.hpp>             // cute::cast_smem_ptr_to_uint

namespace topk_select_common {

// [MACA] 原为 `#ifdef KERUTILS_ENABLE_SM100 -> 1 #else -> 0`。MACA 没有 256 位
// 访存指令，恒为 0；保留这个宏名，是因为 st_global 等处按它做宽度分派，
// 它是一个真实的配置轴而不是可以删掉的残留。
#define IS_LDG_STG_256_AVAILABLE 0

// [MACA] 原实现按 IS_LDG_STG_256_AVAILABLE（仅 SM100 为 1）在两组指令间选：
//   LDG.256/STG.256（sm100+）与 LDG.128/STG.128（sm90 及以下），
//   两者都是带 L1/L2 cache hint 的内联 PTX（.nc / L1::no_allocate / L2::256B）。
//   MACA 汇编器只认 MACA ISA，也没有这些 cache hint 轴，故整组换成等宽的普通
//   向量访存：128 位 = 16 字节，与上游在 sm90 及以下所选宽度一致，
//   因此 NUM_BYTES_PER_GMEM_* 保持不变，下游的切分逻辑不受影响。
//   （cache hint 在 MACA 上无对应物；流式/驱逐策略交由驱动默认。）
static constexpr uint32_t NUM_BYTES_PER_GMEM_LOAD = 128 / 8;
static constexpr uint32_t NUM_BYTES_PER_GMEM_STORE = 128 / 8;
#define LOAD_FROM_GMEM(src_ptr, dst_reg)  (*(uint4 *)(dst_reg) = *(const uint4 *)(src_ptr))
#define STORE_TO_GMEM(dst_ptr, src_reg)  (*(uint4 *)(dst_ptr) = *(const uint4 *)(src_reg))

static constexpr uint32_t NUM_BYTES_PER_SMEM_LOAD = 16;
static constexpr uint32_t NUM_BYTES_PER_SMEM_STORE = 16;


template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void ld_shared(ValueT res[NUM_VALUES], const ValueT* ptr) {
    constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
    static_assert(NUM_BYTES_TO_LOAD == 16 || NUM_BYTES_TO_LOAD == 8 || NUM_BYTES_TO_LOAD == 4 || NUM_BYTES_TO_LOAD == 2);
    uint32_t addr = cute::cast_smem_ptr_to_uint(ptr);
    // [MACA] 原实现走 kerutils::ld_shared 与内联 PTX
    //   (ld.weak.shared::cta.b64/b32/b16)。MACA 汇编器只认 MACA ISA 不认 PTX,
    //   且 xcore1000 不支持 __int128;这里统一改成对 shared 指针的直接按字节搬运,
    //   语义逐位等价(同一块 shared 内存、同样的字节数)。addr 保留供其他分支引用。
    (void)addr;
    if constexpr (NUM_BYTES_TO_LOAD == 16) {
        memcpy(res, ptr, 16);
    } else if constexpr (NUM_BYTES_TO_LOAD == 8) {
        memcpy(res, ptr, 8);
    } else if constexpr (NUM_BYTES_TO_LOAD == 4) {
        memcpy(res, ptr, 4);
    } else if constexpr (NUM_BYTES_TO_LOAD == 2) {
        memcpy(res, ptr, 2);
    } else {
        __builtin_unreachable();
    }
}

template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void ld_shared_with_loop(ValueT res[NUM_VALUES], const ValueT* ptr) {
    constexpr uint32_t NUM_BYTES_TO_LOAD = NUM_VALUES * sizeof(ValueT);
    if constexpr (NUM_BYTES_TO_LOAD <= 16) {
        ld_shared<NUM_VALUES>(res, ptr);
    } else {
        static constexpr uint32_t NUM_LOAD_VALUES_THIS_ROUND = 16 / sizeof(ValueT);
        static_assert(16 % sizeof(ValueT) == 0);
        ld_shared<NUM_LOAD_VALUES_THIS_ROUND>(res, ptr);
        ld_shared_with_loop<NUM_VALUES-NUM_LOAD_VALUES_THIS_ROUND>(res+NUM_LOAD_VALUES_THIS_ROUND, ptr+NUM_LOAD_VALUES_THIS_ROUND);
    }
}

template<uint32_t NUM_VALUES, typename ValueT>
__device__ __forceinline__
void st_global(ValueT* ptr, ValueT src[NUM_VALUES]) {
    constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(ValueT);
    static_assert((IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) || NUM_BYTES_TO_STORE == 16 || NUM_BYTES_TO_STORE == 8 || NUM_BYTES_TO_STORE == 4 || NUM_BYTES_TO_STORE == 2);
    if constexpr (IS_LDG_STG_256_AVAILABLE && NUM_BYTES_TO_STORE == 32) {
        KU_STG_256(ptr, src, "no_allocate", "evict_first");
    } else if constexpr (NUM_BYTES_TO_STORE == 16) {
        // [MACA] 原为 KU_STG_128(ptr, src, "no_allocate", "")——一条带
        // L1::no_allocate cache hint 的内联 PTX st.global.v4。MACA 汇编器只认
        // MACA ISA，且没有 cache hint 轴；此处用与 STORE_TO_GMEM 一致的普通
        // 16 字节向量存储，宽度与对齐要求不变。
        STORE_TO_GMEM(ptr, src);
    } else if constexpr (NUM_BYTES_TO_STORE == 8) {
        *(uint64_t*)ptr = *(uint64_t*)src;
    } else if constexpr (NUM_BYTES_TO_STORE == 4) {
        *(uint32_t*)ptr = *(uint32_t*)src;
    } else if constexpr (NUM_BYTES_TO_STORE == 2) {
        *(uint16_t*)ptr = *(uint16_t*)src;
    } else {
        __builtin_unreachable();
    }
}

template<
    typename Config,
    uint32_t MAX_TOPK,
    uint32_t NUM_THREADS,
    uint32_t TARGET_OCCUPANCY
>
struct EpilogueRunner {
    using ValueT = typename Config::ValueT;
    using OutIdxT = typename Config::OutIdxT;
    static_assert(cute::is_same_v<ValueT, float> || cute::is_same_v<ValueT, maca_bfloat16>);
    static_assert(cute::is_same_v<OutIdxT, int32_t> || cute::is_same_v<OutIdxT, int64_t>);
    using UIntValueT = cute::conditional_t<cute::is_same_v<ValueT, float>, uint32_t, uint16_t>;

    static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
    static_assert(NUM_THREADS % 32 == 0);
    static_assert(NUM_WARPS % 2 == 0);

    static constexpr uint32_t NUM_VALUES_PER_LOAD_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(ValueT);
    static constexpr uint32_t NUM_VALUES_PER_THREAD_FOR_SORT = MAX_TOPK / NUM_THREADS;
    static_assert(MAX_TOPK % NUM_THREADS == 0);
    static constexpr uint32_t NUM_REGS_PER_THREAD = (65536 / (NUM_THREADS * TARGET_OCCUPANCY)) / 8 * 8;
    using BlockRadixSortT = cub::BlockRadixSort<
        UIntValueT,
        NUM_THREADS,
        NUM_VALUES_PER_THREAD_FOR_SORT,
        uint32_t,
        4,
        !(NUM_REGS_PER_THREAD <= 80 && NUM_VALUES_PER_THREAD_FOR_SORT >= 16)  // MemoizeOuterScan. Setting this to `false` can reduce register pressure. We use a simple heuristic here
    >;
    // Gated on `sorted_value`: the sort only runs in the `sorted_value` branch, so this keeps the
    // temp storage out of the smem plan of `sorted_value == false` kernels.
    using BlockRadixSortTempStorageT = cute::conditional_t<Config::sorted_value, typename BlockRadixSortT::TempStorage, char>;

    template<bool IS_SHORTCUT>  // "shortcut" means `end_vocab_idx` <= `topk`
    static __device__ __forceinline__ void topk_select_epilogue(
        typename Config::ValueT* smem_value_buf,    // [MAX_TOPK]
        uint32_t* smem_index_buf,   // [MAX_TOPK]
        const TopkSelectArgs &args,
        uint32_t batch_idx,
        uint32_t end_vocab_idx,
        uint32_t warp_idx,
        BlockRadixSortTempStorageT &radix_sort_temp_storage
    ) {
        // General epilogue for topk select
        // This epilogue has two modes: shortcut mode and non-shortcut mode.
        // In shortcut mode, it
        //  - Generates indices
        //  - Load values from global memory, if `Config::return_value` is `true`
        //  - Perform sorting, if `Config::sorted_value` is `true`
        //  - Save indices & values to global memory
        // In non-shortcut mode, it
        //  - Assume end_vocab_idx > topk holds, i.e. there are enough values for topk selection
        //  - Assume topk indices locate at smem_index_buf[:topk] and topk values locate at smem_value_buf[:topk] (an extra `__syncthreads()` might be necessary)
        //  - Perform sorting, if `Config::sorted_value` is `true`
        //  - Save indices & values to global memory (values are un-distorted if `UNDISTORT_VALUES` is `true`)

        ValueT* input_values = Config::return_value ? (ValueT*)args.input + (uint64_t)batch_idx * args.stride_input_batch : nullptr;
        OutIdxT* result_indices = (OutIdxT*)args.output_index + (uint64_t)batch_idx * args.stride_output_index_batch;
        ValueT* result_values = Config::return_value ? (ValueT*)args.output_value + (uint64_t)batch_idx * args.stride_output_value_batch : nullptr;
        int32_t output_idx_offset = args.output_idx_offset != nullptr ? __ldg(args.output_idx_offset + batch_idx) : 0;
        ValueT oob_fill_value = Config::return_value && IS_SHORTCUT ? (ValueT)args.value_oob_fill_value : (ValueT)0.0f;

        if constexpr (!Config::sorted_value) {
            if constexpr (IS_SHORTCUT) {
                // Saves indices 0...end_vocab_idx (plus `output_idx_offset`) followed by (args.topk-end_vocab_idx) `args.idx_oob_fill_value` into `result_indices`
                constexpr uint32_t NUM_OUTPUT_IDXS_PER_STORE = NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT);
                #pragma unroll 2
                for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDXS_PER_STORE; i < args.topk; i += NUM_THREADS * NUM_OUTPUT_IDXS_PER_STORE) {
                    OutIdxT out[NUM_OUTPUT_IDXS_PER_STORE];
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < NUM_OUTPUT_IDXS_PER_STORE; ++j)
                        out[j] = i+j < end_vocab_idx ? (OutIdxT)(i+j) + output_idx_offset : args.idx_oob_fill_value;
                    STORE_TO_GMEM(result_indices + i, out);
                }

                // Save values into `result_values`, if `RETURN_VALUE` is `True`
                if constexpr (Config::return_value) {
                    #pragma unroll 2
                    for (uint32_t i = threadIdx.x * NUM_VALUES_PER_LOAD_STORE; i < args.topk; i += NUM_THREADS * NUM_VALUES_PER_LOAD_STORE) {
                        ValueT values[NUM_VALUES_PER_LOAD_STORE];
                        if (i < end_vocab_idx) {
                            LOAD_FROM_GMEM(input_values + i, values);
                        }
                        CUTE_UNROLL
                        for (uint32_t j = 0; j < NUM_VALUES_PER_LOAD_STORE; ++j)
                            values[j] = i+j < end_vocab_idx ? values[j] : oob_fill_value;
                        STORE_TO_GMEM(result_values + i, values);
                    }
                }
            } else {
                // Load & cast & save indices
                constexpr uint32_t NUM_OUTPUT_IDX_PER_ROUND = cute::min(NUM_BYTES_PER_SMEM_STORE / sizeof(uint32_t), NUM_BYTES_PER_GMEM_STORE / sizeof(OutIdxT));
                #pragma unroll 2
                for (uint32_t i = threadIdx.x * NUM_OUTPUT_IDX_PER_ROUND; i < args.topk; i += NUM_THREADS * NUM_OUTPUT_IDX_PER_ROUND) {
                    uint32_t indices_u32[NUM_OUTPUT_IDX_PER_ROUND];
                    OutIdxT out[NUM_OUTPUT_IDX_PER_ROUND];
                    ld_shared<NUM_OUTPUT_IDX_PER_ROUND>(indices_u32, smem_index_buf + i);
                    CUTE_UNROLL
                    for (uint32_t j = 0; j < NUM_OUTPUT_IDX_PER_ROUND; ++j)
                        out[j] = (OutIdxT)indices_u32[j] + output_idx_offset;
                    st_global<NUM_OUTPUT_IDX_PER_ROUND>(result_indices + i, out);
                }

                if constexpr (Config::return_value) {
                    // Load & store values
                    constexpr uint32_t NUM_VALUES_PER_ROUND = cute::min(NUM_BYTES_PER_SMEM_STORE, NUM_BYTES_PER_GMEM_STORE) / sizeof(ValueT);
                    #pragma unroll 2
                    for (uint32_t i = threadIdx.x * NUM_VALUES_PER_ROUND; i < args.topk; i += NUM_THREADS * NUM_VALUES_PER_ROUND) {
                        UIntValueT values[NUM_VALUES_PER_ROUND];
                        ld_shared<NUM_VALUES_PER_ROUND>(values, (UIntValueT*)(smem_value_buf + i));
                        st_global<NUM_VALUES_PER_ROUND>((UIntValueT*)(result_values + i), values);
                    }
                }
            }
        } else {
            // Load indices & values into shared memory
            if constexpr (IS_SHORTCUT) {
                if (warp_idx < NUM_WARPS/2) {
                    #pragma unroll 8
                    for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS/2) {
                        smem_index_buf[i] = i < end_vocab_idx ? i : args.idx_oob_fill_value - output_idx_offset;
                    }
                } else {
                    constexpr uint32_t NUM_VALUES_PER_LDG128 = NUM_BYTES_PER_SMEM_LOAD / sizeof(ValueT);
                    #pragma unroll 2
                    for (uint32_t i = (threadIdx.x-NUM_THREADS/2) * NUM_VALUES_PER_LDG128; i < end_vocab_idx; i += (NUM_THREADS/2) * NUM_VALUES_PER_LDG128) {
                        ValueT values[NUM_VALUES_PER_LDG128];
                        // [MACA] 原为 KU_LDG_128(..., ".nc", "no_allocate", "256B")
                        // 加 st_shared(ptr, __int128)。两者都换成 MACA 可用形式：
                        // 宏走本文件顶部的 LOAD_FROM_GMEM，宽 128 位不变；
                        // st_shared 收 uint4（MACA 无 __int128）。
                        LOAD_FROM_GMEM(input_values + i, values);
                        ku::st_shared(smem_value_buf + i, *reinterpret_cast<const uint4 *>(values));
                    }
                }
                __syncthreads();
            }

            // Sort
            UIntValueT local_values[NUM_VALUES_PER_THREAD_FOR_SORT];
            uint32_t local_indices[NUM_VALUES_PER_THREAD_FOR_SORT];
            uint32_t thread_offset = threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT;
            ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(local_values, (UIntValueT*)smem_value_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
            ld_shared_with_loop<NUM_VALUES_PER_THREAD_FOR_SORT>(local_indices, smem_index_buf + threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT);
            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                local_values[i] = distort(local_values[i]);
            if (IS_SHORTCUT || args.topk < MAX_TOPK) {
                uint32_t limit = IS_SHORTCUT ? end_vocab_idx : args.topk;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_values[i] = thread_offset + i < limit ? local_values[i] : 0;
            }
            __syncthreads();    // To allow `radix_sort_temp_storage` overlap with `smem_value_buf` and `smem_index_buf`
            BlockRadixSortT(radix_sort_temp_storage).SortDescending(local_values, local_indices);

            // Store directly to gmem (no smem buffering)
            // For common cases, our MAX_TOPK <= 1024 and NUM_THREADS >= 256, so we have NUM_VALUES_PER_THREAD_FOR_SORT <= 8, which means each thread only need to call STG once. So we don't use SMEM to buffer here
            if (thread_offset < args.topk) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_values[i] = !IS_SHORTCUT || threadIdx.x * NUM_VALUES_PER_THREAD_FOR_SORT + i < end_vocab_idx ? un_distort(local_values[i]) : *(UIntValueT*)&oob_fill_value;
                OutIdxT local_indices_new[NUM_VALUES_PER_THREAD_FOR_SORT];
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_VALUES_PER_THREAD_FOR_SORT; ++i)
                    local_indices_new[i] = (OutIdxT)((int32_t)local_indices[i] + output_idx_offset);
                auto store = [&]<uint32_t NUM_VALUES, typename T>(T* dst, T src[NUM_VALUES]) {
                    constexpr uint32_t NUM_BYTES_TO_STORE = NUM_VALUES * sizeof(T);
                    if constexpr (NUM_BYTES_TO_STORE <= NUM_BYTES_PER_GMEM_STORE) {
                        // Don't need to do OOB check because the output array is at least padded to the maximum width of global store.
                        st_global<NUM_VALUES>(dst, src);
                    } else {
                        static_assert(NUM_BYTES_TO_STORE % NUM_BYTES_PER_GMEM_STORE == 0);
                        CUTE_UNROLL
                        for (uint32_t i = 0; i < NUM_BYTES_TO_STORE; i += NUM_BYTES_PER_GMEM_STORE) {
                            uint32_t elem_offset = i / sizeof(T);
                            if (thread_offset + elem_offset < args.topk)
                                STORE_TO_GMEM(dst+elem_offset, src+elem_offset);
                        }
                    }
                };
                store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>((UIntValueT*)result_values + thread_offset, local_values);
                store.template operator()<NUM_VALUES_PER_THREAD_FOR_SORT>(result_indices + thread_offset, local_indices_new);
            }
        }
    }
};

/*
// TODO Give a brief introduction to the algorithm here
*/
template<typename Config>
class TopkSelectKernelBase {
public:
    using ValueT = Config::ValueT;
    using OutIdxT = Config::OutIdxT;
    static_assert(std::is_same_v<ValueT, maca_bfloat16> || std::is_same_v<ValueT, float>);
    
    static constexpr uint32_t PLACEHOLDER_U32 = std::is_same_v<ValueT, maca_bfloat16> ? 0xff80 : 0xff800000; // -INF
    static constexpr uint64_t PLACEHOLDER_B64 = std::is_same_v<ValueT, maca_bfloat16> ? 0xff80ff80ff80ff80 : 0xff800000ff800000;
    static constexpr uint64_t PLACEHOLDER_PAIR = (uint64_t)PLACEHOLDER_U32 << 32;
    
    static constexpr uint32_t TARGET_OCCUPANCY = Config::target_occupancy;
    static constexpr uint32_t NUM_THREADS = Config::num_threads;
    static constexpr uint32_t NUM_WARPS = NUM_THREADS / 32;
    static constexpr uint32_t MAX_TOPK = Config::max_topk;
    static_assert(NUM_THREADS == 256 || NUM_THREADS == 512);

    static constexpr uint32_t NUM_ELEMS_PER_128b = 16 / sizeof(ValueT); 
    static constexpr uint32_t NUM_UINT32_PER_128b = 4;
    static constexpr uint32_t NUM_ELEMS_PER_ROUND = Config::elements_per_round;
    static constexpr uint32_t NUM_ELEMS_PER_THREAD_PER_ROUND = NUM_ELEMS_PER_ROUND / NUM_THREADS;   // TODO Unify "values", "elements", and "elems" to "elems"
    static constexpr uint32_t NUM_128b_PER_THREAD_PER_ROUND = NUM_ELEMS_PER_THREAD_PER_ROUND / NUM_ELEMS_PER_128b;
    static_assert(NUM_ELEMS_PER_ROUND % (NUM_ELEMS_PER_128b * NUM_THREADS) == 0);
    static constexpr uint32_t ELEMS_PER_THREAD_PER_ROUND_MASK = NUM_ELEMS_PER_THREAD_PER_ROUND - 1;
    // NUM_128b_PER_THREAD_PER_ROUND is the canonical name; no separate NUM_UNITS alias is kept.

    static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND == 16); 

    // Each segment/window contains this many elements. It is the granularity used by both the
    // pseudo-random visit order and the sorted-index epilogue.
    static constexpr uint32_t NUM_ELEMS_PER_SEG = Config::elements_per_segment; // shuffle rule
    static_assert(NUM_ELEMS_PER_SEG == 512);
    // The last NUM_TAIL_ELEMS elements of a row are always visited in their original order.
    // 4096 keeps the tail small enough to fit alongside the init-window perm segments.
    static constexpr uint32_t NUM_TAIL_ELEMS = 4096;  
    static constexpr uint32_t NUM_TAIL_SEGS = NUM_TAIL_ELEMS / NUM_ELEMS_PER_SEG; 
    // The init window is 32 KiB of input data
    static constexpr uint32_t NUM_ELEMS_IN_INIT_WINDOW = 32768 / sizeof(ValueT); 
    static constexpr uint32_t NUM_INIT_ROUNDS_MAX = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_ROUND;
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_ROUND == 0);
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % NUM_ELEMS_PER_SEG == 0);

    static constexpr uint32_t NUM_SEGS_PER_ROUND = NUM_ELEMS_PER_ROUND / NUM_ELEMS_PER_SEG;
    static_assert(NUM_ELEMS_PER_ROUND % NUM_ELEMS_PER_SEG == 0);
    static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_128b == 0);
    static constexpr uint32_t NUM_128b_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_128b;
    static_assert(NUM_THREADS % NUM_128b_PER_SEG == 0); 
    static_assert(NUM_128b_PER_SEG % 32 == 0); // every lane in a warp should in the same seg
    
    static constexpr uint32_t NUM_ISSUE_WARPS = NUM_SEGS_PER_ROUND < 4 ? NUM_SEGS_PER_ROUND : 4;
    static constexpr uint32_t NUM_SEGS_PER_ISSUE_WARP = NUM_SEGS_PER_ROUND / NUM_ISSUE_WARPS;
    static_assert(NUM_SEGS_PER_ROUND % NUM_ISSUE_WARPS == 0);
    static_assert(NUM_TAIL_SEGS % NUM_ISSUE_WARPS == 0);   
    static_assert(NUM_WARPS >= NUM_ISSUE_WARPS);

    // [MACA] 无流水线，单缓冲。原为 `NUM_TMA_LOAD_BUFS = Config::tma_buffer_depth`
    // 与 `TMA_PREFETCH_DEPTH = NUM_TMA_LOAD_BUFS - 1`（>=2 缓冲 + 跨轮预取）。
    // Config::tma_buffer_depth 仍保留在配置里（上游戏法与调参记录的一部分），
    // 但 MACA 路径不再用它开多缓冲。
    static constexpr uint32_t NUM_TMA_LOAD_BUFS = 1;
    static constexpr uint32_t TMA_PREFETCH_DEPTH = 0;
    static_assert(NUM_TMA_LOAD_BUFS == 1);

    static constexpr uint32_t RECONSTRUCT_THRESHOLD = Config::reconstruct_threshold;
    static_assert(RECONSTRUCT_THRESHOLD >= MAX_TOPK);

    // Parameters of the pseudo-random permutation over permuted segments:
    //   permuted_id = (linear_id * (PERM_MUL_PRIME % perm_len) + PERM_ADD_BASE) % perm_len.
    static constexpr uint32_t PERM_ADD_BASE = 0x22262226u;
    static constexpr uint32_t PERM_MUL_PRIME = 0xB559EB75u;

    static constexpr uint32_t NUM_SEGS_IN_INIT_WINDOW = NUM_ELEMS_IN_INIT_WINDOW / NUM_ELEMS_PER_SEG;
    // Number of permuted segments included in the init window after reserving room for the tail.
    static constexpr uint32_t NUM_PERM_SEGS_IN_INIT_WINDOW =
        (NUM_ELEMS_IN_INIT_WINDOW - NUM_TAIL_ELEMS) / NUM_ELEMS_PER_SEG;
    static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW % NUM_TAIL_SEGS == 0);
    static_assert(NUM_PERM_SEGS_IN_INIT_WINDOW > 0);
    static constexpr uint32_t NUM_UINT32_IN_INIT_WINDOW_PER_THREAD = NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT) / sizeof(uint32_t) / NUM_THREADS;
    static constexpr uint32_t NUM_128b_INIT_PER_THREAD = NUM_UINT32_IN_INIT_WINDOW_PER_THREAD / NUM_UINT32_PER_128b;
    static_assert(NUM_ELEMS_IN_INIT_WINDOW % (sizeof(uint32_t) / sizeof(ValueT) * NUM_THREADS) == 0);
    static_assert(NUM_UINT32_IN_INIT_WINDOW_PER_THREAD % (16 * sizeof(ValueT) / sizeof(uint32_t)) == 0);
    static_assert(2 * NUM_UINT32_IN_INIT_WINDOW_PER_THREAD <= 256);

    static constexpr uint32_t NUM_EXTRA_SLOTS = RECONSTRUCT_THRESHOLD + NUM_ELEMS_PER_ROUND;

    static constexpr uint32_t NUM_RECONSTRUCT_RADIX_BITS = 8;
    static constexpr uint32_t NUM_RECONSTRUCT_BUCKETS = 1u << NUM_RECONSTRUCT_RADIX_BITS;
    // Each row also holds one sink slot (index NUM_RECONSTRUCT_BUCKETS) that the LSB histogram sends the
    // out-of-bucket elements to; the +4 rounds a row up to a multiple of 16 bytes so that both rows stay
    // 16B-aligned, since the reader relies on for its 128-bit shared-memory loads.
    static constexpr uint32_t NUM_RECONSTRUCT_BUCKET_SLOTS = NUM_RECONSTRUCT_BUCKETS + 4;
    static_assert(sizeof(uint32_t) * NUM_RECONSTRUCT_BUCKET_SLOTS % 16 == 0);
    static_assert(MAX_TOPK + NUM_EXTRA_SLOTS <= 0xFFFF);
    static_assert(NUM_EXTRA_SLOTS % 2 == 0);

    static constexpr uint32_t NUM_RECONSTRUCT_UNITS_MAX = (MAX_TOPK + NUM_EXTRA_SLOTS) / 2;

    using EpilogueT = EpilogueRunner<Config, MAX_TOPK, NUM_THREADS, TARGET_OCCUPANCY>;
    static_assert(sizeof(typename EpilogueT::BlockRadixSortTempStorageT) <= NUM_EXTRA_SLOTS * sizeof(uint64_t));

    static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) >= NUM_ELEMS_IN_INIT_WINDOW * sizeof(ValueT),
                "the init window must fit in the extra pairs region");
    static_assert(NUM_EXTRA_SLOTS * sizeof(uint64_t) / sizeof(uint32_t) >= MAX_VOCAB_SIZE / NUM_ELEMS_PER_SEG,
                "the index sort's window delta table must fit in the extra pairs region");
    // array `incoming_topk_pairs` will be used as 
    //     1. tma buffer during init phase
    //     2. extra array during main phase
    //     3. windows_delta counter during epilogue phase
    static_assert(2 * MAX_TOPK * sizeof(uint64_t) % 1024 == 0);

    static constexpr uint32_t SWIZZLE_SHIFT = 3;
    static constexpr uint32_t SWIZZLE_MASK = 7 * (16 / sizeof(ValueT));

    static constexpr uint32_t NUM_BYTES_PER_TMA_ROW = 128;  // Limit the innermost box dim of TMA to 1) prevent OOB 2) swizzling has limitations on the innermost box dim
    static_assert(NUM_BYTES_PER_TMA_ROW <= INPUT_STRIDE_ALIGNMENT_REQUIREMENT); // To prevent OOB
    static constexpr uint32_t NUM_ELEMS_PER_TMA_ROW = NUM_BYTES_PER_TMA_ROW / sizeof(ValueT);
    static_assert(NUM_ELEMS_PER_SEG % NUM_ELEMS_PER_TMA_ROW == 0);
    static constexpr uint32_t NUM_TMA_ROWS_PER_SEG = NUM_ELEMS_PER_SEG / NUM_ELEMS_PER_TMA_ROW;
    static_assert(SWIZZLE_MASK == ((1u << SWIZZLE_SHIFT) - 1) * NUM_ELEMS_PER_128b);
    static_assert(NUM_ELEMS_PER_TMA_ROW == (1u << SWIZZLE_SHIFT) * NUM_ELEMS_PER_128b);
    static_assert(NUM_TMA_ROWS_PER_SEG % (1u << SWIZZLE_SHIFT) == 0);

    static_assert(NUM_SEGS_PER_ROUND == NUM_WARPS);
    static_assert(NUM_SEGS_PER_ROUND >= NUM_TAIL_SEGS);
    static_assert(NUM_ELEMS_PER_SEG % (Config::elements_per_round/NUM_WARPS) == 0);
    
    // [MACA] 原为 TMA 张量映射：
    //   struct TmaParams { CUtensorMap tensor_map; };
    //   static CUtensorMap make_topk_tensor_map(const TopkSelectArgs &args) { ... }
    //   它把 (vocab, batch) 拆成 3 维盒、声明 128B swizzle 与 L2 promotion，
    //   供 SM90_TMA_LOAD_3D::copy 寻址。MACA 没有 TMA 也没有 CUtensorMap，
    //   装载改为在 copy_one_seg 里用 args.input + batch_idx*stride_input_batch
    //   直接算全局地址（段内连续，无需张量映射），故整块删除。
    // 原 make_topk_tensor_map 内断言「行首对齐足以放下一个 TMA 行盒」；
    // 装载改直连后仍需要这个对齐前提（段内 16B 块首地址对齐），保留。
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT >= NUM_ELEMS_PER_TMA_ROW * sizeof(ValueT));

    struct SharedMemoryPlanBase {
        // bf16: pair(64b) = unused(16b) | value(16b) | index(32b)
        // fp32: pair(64b) = value(32b) | index(32b)
        // candidate[0]/candidate[1] are the A/B buffers swapped on each reconstruct;
        // incoming holds the pairs scanned since the last reconstruct.
        CUTE_ALIGNAS(1024) uint64_t surviving_topk_pairs[2][MAX_TOPK];
        CUTE_ALIGNAS(1024) uint64_t incoming_topk_pairs[NUM_EXTRA_SLOTS];
        // [MACA] 单缓冲。原为 tma_load_buf[NUM_TMA_LOAD_BUFS] 加
        // tma_load_full_bar / init_full_bar 两组事务屏障，配合跨轮预取。
        // MACA 没有 TMA/mbarrier 也没有生产者-消费者 warp specialization，
        // 流水线整段拆除：全线程装载本轮 → __syncthreads() → 全线程消费。
        CUTE_ALIGNAS(1024) ValueT tma_load_buf[NUM_ELEMS_PER_ROUND];
        uint32_t warp_cnt[NUM_WARPS];
        uint32_t reconstruct_pivot_bucket;
        uint32_t reconstruct_num_should_select;
        CUTE_ALIGNAS(16) uint32_t reconstruct_bucket_counter[2][NUM_RECONSTRUCT_BUCKET_SLOTS];
    };

    // Map a linear visit index to the actual permuted segment id.
    static __device__ __forceinline__
    uint32_t get_permuted_seg_idx(uint32_t linear_id, uint32_t perm_len, uint32_t perm_mul) {
        return ((linear_id % perm_len) * perm_mul + PERM_ADD_BASE) % perm_len;
    }

    static __device__ __forceinline__ 
    void advance_perm_state(uint32_t &state, uint32_t stride, uint32_t perm_len) {
        state += stride;
        if (state >= perm_len) state -= perm_len;
    }

    static __device__ __forceinline__ 
    uint32_t sw_msk(uint32_t e) {
        return (e >> SWIZZLE_SHIFT) & SWIZZLE_MASK;
    }
    static __device__ __forceinline__ 
    uint32_t sw_elem(uint32_t e) {
        return e ^ sw_msk(e);
    }
    static __device__ __forceinline__ 
    uint32_t sw_b128(uint32_t u) {
        return u ^ ((u >> 3) & (7));
    }

    struct EqGtPrefix { uint32_t start_pos_in_collector; uint32_t eq_quota; };
    // Given local > pivot and == pivot counts, compute this thread's output prefix (start position in the final output buffer) and how many equal
    // elements it may still append (eq_quota).
    // Should be called by all threads
    // cnt_gt and cnt_eq must be < 65536.
    static __device__ __forceinline__
    EqGtPrefix compute_equal_quota_and_prefix(uint32_t cnt_gt, uint32_t cnt_eq, uint32_t topk, uint32_t warp_idx, uint32_t lane_idx, uint32_t *warp_cnt) {
        static_assert(NUM_WARPS <= 32);
        uint32_t cnt_packed = (cnt_gt << 16) | cnt_eq;
        uint32_t warp_total_packed = __reduce_add_sync(0xFFFFFFFF, cnt_packed);
        if (lane_idx == 0) {
            warp_cnt[warp_idx] = warp_total_packed;
        }
        __syncthreads();
        
        uint32_t off_lane_packed = warp_level_exclusive_prefix_sum(cnt_packed, lane_idx);
        uint32_t stored_warp_packed = lane_idx < NUM_WARPS ? warp_cnt[lane_idx] : 0u;
        uint32_t off_warp_packed = __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_packed : 0u);
        uint32_t total_packed = __reduce_add_sync(0xFFFFFFFF, stored_warp_packed);
        uint32_t num_total_ge = total_packed >> 16;
        uint32_t num_total_eq_quota = topk - num_total_ge;
        uint32_t off_packed = off_warp_packed + off_lane_packed;
        uint32_t num_gt_before = off_packed >> 16;
        uint32_t num_eq_before = off_packed & 0xFFFFu;
        uint32_t eq_quota = num_total_eq_quota > num_eq_before ? num_total_eq_quota - num_eq_before : 0u;
        uint32_t num_elems_should_select_before = num_gt_before + min(num_eq_before, num_total_eq_quota);
        return {num_elems_should_select_before, eq_quota};
    }

    // Load this thread's slice from the swizzled smem round buffer into registers.
    template<uint32_t NUM_128b>
    static __device__ __forceinline__
    void load_swizzled_slice(
        ValueT dst[NUM_128b * NUM_ELEMS_PER_128b],
        const ValueT *smem_buf,
        uint32_t smem_slice_elem_offset, // element offset into smem_buf where this thread's slice starts
        uint32_t unit_swizzle_mask,      // swizzle mask applied to each 128-bit unit inside the slice
        uint32_t first_128b_unit_in_slice = 0 // which 128-bit unit of this slice to start loading from
    ) {
        CUTE_UNROLL
        for (uint32_t u = 0; u < NUM_128b; ++u) {
            ld_shared<NUM_ELEMS_PER_128b>(
                dst + u * NUM_ELEMS_PER_128b,
                smem_buf + smem_slice_elem_offset +
                    (((first_128b_unit_in_slice + u) * NUM_ELEMS_PER_128b) ^ unit_swizzle_mask));
        }
    }

    // Find the bucket id that the topk-th element resides in, and get how many elements we should select from that bucket
    // Save result to `smem.reconstruct_pivot_bucket` and `smem.reconstruct_num_should_select`
    // If `CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET`, return whether we should select all elements in that bucket.
    // Should be called by a whole warp
    template<bool CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET>
    static __device__ __forceinline__
    bool find_pivot_in_histogram(SharedMemoryPlanBase &smem, const uint32_t *bucket_counter, uint32_t topk, uint32_t lane_idx) {
        bool should_select_whole_bucket = false;
        // Each lane loads 8 counters
        static_assert(NUM_RECONSTRUCT_BUCKETS == 32 * 8);
        uint32_t counts[8];
        const uint32_t *bucket_ptr = bucket_counter + lane_idx * 8;
        ld_shared<4>(counts, bucket_ptr);
        ld_shared<4>(counts + 4, bucket_ptr + 4);

        uint32_t local_sum = 0;
        CUTE_UNROLL
        for (uint32_t j = 0; j < 8; ++j)
            local_sum += counts[j];

        // Get the inclusive suffix sum of the histogram
        uint32_t suffix_count[9];
        suffix_count[8] = warp_level_exclusive_suffix_sum(local_sum, lane_idx);
        CUTE_UNROLL
        for (int32_t j = 7; j >= 0; --j)
            suffix_count[j] = suffix_count[j + 1] + counts[j];

        if (suffix_count[8] < topk && topk <= suffix_count[0]) {
            uint32_t j = 0;
            CUTE_UNROLL
            for (uint32_t k = 1; k < 8; ++k)
                j += (uint32_t)(suffix_count[k] >= topk);
            
            // A depth-3 SEL tree picking suffix_count[j+1] by the bits of j
            // j is the bucket that contains the top-k element
            uint32_t b0 = (j & 1) ? suffix_count[2] : suffix_count[1], b1 = (j & 1) ? suffix_count[4] : suffix_count[3];
            uint32_t b2 = (j & 1) ? suffix_count[6] : suffix_count[5], b3 = (j & 1) ? suffix_count[8] : suffix_count[7];
            b0 = (j & 2) ? b1 : b0;  b2 = (j & 2) ? b3 : b2;
            uint32_t suffix_count_j_plus_1 = (j & 4) ? b2 : b0;   // = s[j+1]
            smem.reconstruct_pivot_bucket = lane_idx * 8 + j;
            smem.reconstruct_num_should_select = topk - suffix_count_j_plus_1;

            if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
                // topk == suffix_count[j] means the whole pivot bucket is selected
                uint32_t d0 = (j & 1) ? suffix_count[1] : suffix_count[0], d1 = (j & 1) ? suffix_count[3] : suffix_count[2];
                uint32_t d2 = (j & 1) ? suffix_count[5] : suffix_count[4], d3 = (j & 1) ? suffix_count[7] : suffix_count[6];
                d0 = (j & 2) ? d1 : d0;  d2 = (j & 2) ? d3 : d2;
                uint32_t suffix_count_j = (j & 4) ? d2 : d0;    // = suffix_count[j]
                should_select_whole_bucket = (topk == suffix_count_j);
            }
        }

        if constexpr (CHECK_IF_SHOULD_SELECT_WHOLE_BUCKET) {
            // The owning lane holds the result; reduce so all lanes in the warp see the same answer.
            return __reduce_or_sync(0xFFFFFFFFu, should_select_whole_bucket);
        }
        return false;
    }

    // Clean every bucket counter using 16-byte shared-memory writes.
    // Need to be called by all threads.
    static __device__ __forceinline__ 
    void clear_reconstruct_histograms(SharedMemoryPlanBase &smem, uint32_t thread_idx) {
        static_assert((2 * NUM_RECONSTRUCT_BUCKET_SLOTS) % 4 == 0);
        // [MACA] 原用 __int128_t 清零（16 字节/次）。MACA 不支持 __int128
        // （'Int128 or UInt128 is not supported on this target'），换成同样
        // 16 字节的 uint4；下标的 /4 是「16 字节单元数」，两种写法一致。
        const uint4 zero = make_uint4(0, 0, 0, 0);
        for (uint32_t i = thread_idx; i < (2 * NUM_RECONSTRUCT_BUCKET_SLOTS) / 4; i += NUM_THREADS) {
            reinterpret_cast<uint4*>(smem.reconstruct_bucket_counter)[i] = zero;
        }
    }

    template<typename ExtraBarInitF>
    static __device__ __forceinline__
    void init_shared_memory(SharedMemoryPlanBase &smem, uint32_t warp_idx, ExtraBarInitF &&init_extra_bar_func) {
        // Pad unused elements in `surviving_topk_pairs` as `-INF` to avoid being selected
        static_assert((2 * MAX_TOPK) % NUM_THREADS == 0);
        CUTE_UNROLL
        for (uint32_t i = 0; i < MAX_TOPK / NUM_THREADS; ++i) {
            smem.surviving_topk_pairs[0][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
            smem.surviving_topk_pairs[1][i * NUM_THREADS + threadIdx.x] = PLACEHOLDER_PAIR;
        }

        clear_reconstruct_histograms(smem, threadIdx.x);

        // [MACA] 原为单线程 elect 后初始化 tma_load_full_bar / init_full_bar 两组
        // mbarrier，并用 cutlass::arch::fence_barrier_init() 发布。
        // 屏障已随流水线拆除，这里只剩一个 __syncthreads() 让初始化对全 block 可见。
        (void)init_extra_bar_func;   // 原供 cluster 变体追加屏障，MACA 无 cluster 变体
        __syncthreads();
    }

    static __device__ __forceinline__
    void init_shared_memory(SharedMemoryPlanBase &smem, uint32_t warp_idx) {
        init_shared_memory(smem, warp_idx, [] {});
    }

    // Issue TMA loads for one round
    // Use IS_INIT to control whether is the init round.
    // For init round: copy the "tail" [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), if HAVE_TAIL is true, and also the first NUM_SEGS_PER_ROUND - NUM_TAIL_SEGS segments according to the perm generator
    // For non-init round: copy the next NUM_SEGS_PER_ROUND segs according to the perm generator
    // 装载一轮。MACA 版：无 TMA、无 mbarrier、无生产者-消费者 warp specialization。
    //   * 全部 warp 参与，每个 warp 负责本轮内序号 ≡ warp_idx (mod NUM_WARPS) 的段
    //     （NUM_SEGS_PER_ROUND == NUM_WARPS，故每人恰好一段）
    //   * 段内 32 条 lane 按 16B 块协作 ldg 到 shared，目标地址不 swizzle
    //   * 不返回任何完成标志：调用方紧随一个 __syncthreads()，按内建文档
    //     「使用 __syncthreads() 保证完成时可省略返回值」，它同时是最省的正确做法
    //   * 置换索引直接由线性位置算出，不再维护跨轮状态机
    template<bool IS_INIT, bool HAVE_TAIL>
    static __device__ __forceinline__
    void load_round_for_round(SharedMemoryPlanBase &smem, const TopkSelectArgs &args,
                        uint32_t batch_idx, uint32_t end_vocab_idx,
                        uint32_t num_perm_segs, uint32_t round_idx,
                        uint32_t local_start_seg_idx,
                        uint32_t warp_idx, uint32_t lane_idx
    ) {
        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t perm_mul = PERM_MUL_PRIME % perm_len;
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;

        static_assert(IS_INIT || !HAVE_TAIL);
        static_assert(NUM_SEGS_PER_ROUND % NUM_WARPS == 0);
        static_assert(NUM_SEGS_PER_ROUND >= NUM_TAIL_SEGS);
        constexpr uint32_t SEGS_PER_WARP = NUM_SEGS_PER_ROUND / NUM_WARPS;

        ValueT *dst_base;
        if constexpr (IS_INIT) {
            // init 轮借用 incoming_topk_pairs 作缓冲（上游已断言其容量足够）
            dst_base = (ValueT*)smem.incoming_topk_pairs + round_idx * NUM_ELEMS_PER_ROUND;
        } else {
            dst_base = smem.tma_load_buf;
        }

        constexpr uint32_t NUM_CHUNKS_PER_SEG = NUM_ELEMS_PER_SEG * sizeof(ValueT) / 16;
        constexpr uint32_t CHUNK_ELEMS = 16 / sizeof(ValueT);
        static_assert(CHUNK_ELEMS == NUM_ELEMS_PER_128b);
        static_assert(NUM_ELEMS_PER_ROUND % CHUNK_ELEMS == 0);

        // 消费端的 swizzle 原点是它自己的缓冲区起点：main 轮是 tma_load_buf 起点，
        // init 轮是 init_buf 起点。IS_INIT 时 dst_base 已经偏移了 round_idx 个轮长度，
        // 这里把这段偏移补回去，保证产/消两侧算的是同一个 sw_b128 下标。
        // 若不做这一步，init 轮的 XOR 掩码会整体错位 —— 静默的错误结果。
        const uint32_t dst_origin_elems = IS_INIT ? round_idx * NUM_ELEMS_PER_ROUND : 0u;

        const ValueT *gmem_row = (const ValueT *)args.input + (size_t)batch_idx * args.stride_input_batch;
        auto copy_one_seg = [&](uint32_t dst_seg_idx, uint32_t global_seg_idx, uint32_t num_valid_elems) {
            const ValueT *gmem_seg = gmem_row + (size_t)global_seg_idx * NUM_ELEMS_PER_SEG;
            // 目标地址必须按消费端的 128B 行内 XOR 布局写（load_swizzled_slice 用
            // smem_read_offset / chunk_swizzle_mask 读同一套映射）：源元素块 u 落到
            // 目标块 sw_b128(u)。sw_b128 是自逆的，且 sw_msk 在 16B 块内恒为常量，
            // 所以"整块搬"与逐元素搬等价。
            uint32_t dst_elems = dst_seg_idx * NUM_ELEMS_PER_SEG;
            uint32_t dst_unit_base = (dst_origin_elems + dst_elems) / NUM_ELEMS_PER_128b;
            // TMA 会对越界元素补零、`mask` 又不支持部分搬运，所以这里自己界定边界：
            // 尾段通常只装到 end_vocab_idx 为止，最后一块可能不满 16B。
            uint32_t num_chunks = min(NUM_CHUNKS_PER_SEG, ku::ceil_div(num_valid_elems, CHUNK_ELEMS));
            for (uint32_t c = lane_idx; c < num_chunks; c += 32u) {
                uint32_t dst_off = sw_b128(dst_unit_base + c) * NUM_ELEMS_PER_128b - dst_origin_elems;
                __builtin_mxc_ldg_b128_bsm(dst_base + dst_off,
                                           (void *)(gmem_seg + c * CHUNK_ELEMS),
                                           0, (size_t)-1, false, false, false, false);
            }
        };

        // 本轮各 warp 负责本地段 `warp_idx`（NUM_SEGS_PER_ROUND == NUM_WARPS，故一人一段）。
        // 本轮的本地段分两区：前 NUM_TAIL_SEGS_THIS_ROUND 段整块预留给尾段（只有 init 轮带尾段），
        // 其余按置换序装载。预留段中不含有效元素的位置不写，由 fill_padded_tail_segments 填 PLACEHOLDER。
        constexpr uint32_t NUM_TAIL_SEGS_THIS_ROUND = HAVE_TAIL ? NUM_TAIL_SEGS : 0;
        static_assert(SEGS_PER_WARP == 1);
        uint32_t local_seg_idx = warp_idx;

        if (local_seg_idx < NUM_TAIL_SEGS_THIS_ROUND) {
            uint32_t num_tail_segs = ku::ceil_div(end_vocab_idx - num_perm_elems, (uint32_t)NUM_ELEMS_PER_SEG);
            if (local_seg_idx < num_tail_segs) {
                uint32_t seg_elem_base = (num_perm_segs + local_seg_idx) * NUM_ELEMS_PER_SEG;
                copy_one_seg(local_seg_idx, num_perm_segs + local_seg_idx, end_vocab_idx - seg_elem_base);
            }
        } else if (IS_INIT && num_perm_segs == 0) {
            // 不置换时按序整段搬入（尾段即全部数据，没有置换区，故不减 NUM_TAIL_SEGS）
            uint32_t num_global_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
            uint32_t global_seg_idx = round_idx * NUM_SEGS_PER_ROUND + local_seg_idx;
            if (global_seg_idx < num_global_segs) {
                uint32_t seg_elem_base = global_seg_idx * NUM_ELEMS_PER_SEG;
                copy_one_seg(local_seg_idx, global_seg_idx, end_vocab_idx - seg_elem_base);
            }
        } else {
            // 置换区线性位置 = 全局起点 + 本轮起点 + 本地段号 - 尾段占位。
            // 必须与消费端 scan_segs 里 `linear_segment_start` 的推导逐项一致。
            uint32_t linear_pos = local_start_seg_idx + round_idx * NUM_SEGS_PER_ROUND
                                + local_seg_idx - NUM_TAIL_SEGS;
            // 置换段按构造全部落在 end_vocab_idx 以下，必为整段
            copy_one_seg(local_seg_idx, get_permuted_seg_idx(linear_pos, perm_len, perm_mul), NUM_ELEMS_PER_SEG);
        }
    }

    // Fill the invalid (padded) tail segments with PLACEHOLDER (-INF).
    // TMA only loads the real tail elements; the unused padded tail region must still be initialized as -INF
    static __device__ __forceinline__ void fill_padded_tail_segments(ValueT *init_buf, uint32_t num_tail_elems) {
        uint32_t tail_covered = ku::ceil(num_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG);
        // [MACA] 原为 constexpr __int128_t FILLER_128b = (PLACEHOLDER_B64 << 64) | PLACEHOLDER_B64
        // 再 st_shared(ptr, FILLER_128b)。MACA 无 __int128，改由双 64 位重载写入
        // 同样的 16 字节（低半字在前，与原构造顺序一致）。
        for (uint32_t i = tail_covered + threadIdx.x * NUM_ELEMS_PER_128b; i < NUM_TAIL_ELEMS; i += NUM_THREADS * NUM_ELEMS_PER_128b) {
            ku::st_shared(init_buf + i, PLACEHOLDER_B64, PLACEHOLDER_B64);
        }
    }

    // Stage the final top-k candidates from the survivor buffer into a contiguous smem layout
    // and run the epilogue.
    //
    // When SORT_BY_INDEX is true, each 512-element window is visited as a whole in the permuted
    // order, so the survivors are already sorted inside each window. Only the window groups need
    // to be reordered by their window id
    //
    // Input cannot contain NaN
    template<bool SORT_BY_INDEX>
    static __device__ __forceinline__
    void stage_output_and_epilogue(SharedMemoryPlanBase &smem, typename EpilogueT::BlockRadixSortTempStorageT &sort_temp_storage,
                                   const TopkSelectArgs &args, uint32_t batch_idx, uint32_t end_vocab_idx,
                                   uint32_t survivor_buf_idx, uint32_t warp_idx, uint32_t lane_idx) {

        uint32_t staging_buf_idx = survivor_buf_idx ^ 1;
        bool is_already_sorted_in_index = end_vocab_idx <= NUM_ELEMS_IN_INIT_WINDOW;
        uint32_t* window_scatter_delta = (uint32_t*)smem.incoming_topk_pairs;
        ValueT* smem_value_buf = (ValueT*)smem.surviving_topk_pairs[staging_buf_idx];
        uint32_t* smem_index_buf = (uint32_t*)(smem.surviving_topk_pairs[staging_buf_idx] + MAX_TOPK * (uint32_t)sizeof(ValueT) / 8);
        
        // Index-sort shuffle: each window's selected candidates are already sorted and contiguous;
        // only the whole window groups are in visit order, so reorder them by their window id.
        if constexpr (SORT_BY_INDEX) if (!is_already_sorted_in_index) {
            uint32_t num_windows = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
            auto get_window_id = [&](uint32_t offset) -> uint32_t {
                return (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][offset] / NUM_ELEMS_PER_SEG;
            };

            // Reset all window counters.
            for (uint32_t i = threadIdx.x; i < num_windows; i += NUM_THREADS) {
                window_scatter_delta[i] = 0;
            }
            __syncthreads();

            // Count how many selected candidates fall into each window.
            for (uint32_t i = threadIdx.x; i < args.topk; i += NUM_THREADS) {
                atomicAdd_block(window_scatter_delta + get_window_id(i), 1u);
            }
            __syncthreads();

            // Convert the per-window counts into exclusive prefix sums (starting offsets).
            uint32_t windows_per_thread = ku::ceil_div(num_windows, (uint32_t)NUM_THREADS);
            uint32_t w_lo = min(threadIdx.x * windows_per_thread, num_windows);
            uint32_t w_hi = min(w_lo + windows_per_thread, num_windows);
            uint32_t my_cnt = 0;
            for (uint32_t i = w_lo; i < w_hi; ++i) {
                // TODO Performance can be optimized by using wider reads (pay attn to alignment issues!)
                my_cnt += window_scatter_delta[i];
            }
            uint32_t warp_total = __reduce_add_sync(0xFFFFFFFF, my_cnt);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total;
            }
            __syncthreads();

            static_assert(NUM_WARPS <= 32);
            uint32_t cur_prefix_sum = __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? smem.warp_cnt[lane_idx] : 0u) + warp_level_exclusive_prefix_sum(my_cnt, lane_idx);
            for (uint32_t i = w_lo; i < w_hi; ++i) {
                uint32_t cnt = window_scatter_delta[i];
                window_scatter_delta[i] = cur_prefix_sum;
                cur_prefix_sum += cnt;
            }
            __syncthreads();

            // Subtract window_delta[i] by the index of the first element in this window around all select elements
            // So that later we can use "index + window_delta[i]" to choose the index
            for (uint32_t i = threadIdx.x+1; i < args.topk; i += NUM_THREADS) { // +1 to skip the first element
                uint32_t idx = (uint32_t)smem.surviving_topk_pairs[survivor_buf_idx][i];
                uint32_t window_idx = idx / NUM_ELEMS_PER_SEG;
                uint32_t prev_elem_window_id = get_window_id(i - 1u);
                if (window_idx != prev_elem_window_id) {
                    // i is the first element in its window
                    window_scatter_delta[window_idx] -= i;
                }
            }

            __syncthreads();
        }

        CUTE_UNROLL
        for (uint32_t i = threadIdx.x; i < MAX_TOPK; i += NUM_THREADS) {
            uint64_t pair = smem.surviving_topk_pairs[survivor_buf_idx][i];
            uint32_t idx = (uint32_t)pair;
            ValueT val;
            if constexpr (sizeof(ValueT) == 2) {
                val = __ushort_as_bfloat16((uint16_t)(pair >> 32));
            } else {
                val = __uint_as_float((uint32_t)(pair >> 32));
            }
            if (SORT_BY_INDEX && !is_already_sorted_in_index) {
                if (i < args.topk) {
                    uint32_t dst_pos = (i + window_scatter_delta[idx / NUM_ELEMS_PER_SEG]) & (MAX_TOPK - 1);
                    smem_value_buf[dst_pos] = val;
                    smem_index_buf[dst_pos] = idx;
                    continue;
                }
            }
            smem_value_buf[i] = val;
            smem_index_buf[i] = idx;
        }

        __syncthreads();

        EpilogueT::template topk_select_epilogue<false>(
            smem_value_buf,
            smem_index_buf,
            args,
            batch_idx, end_vocab_idx, warp_idx,
            sort_temp_storage
        );
    }

    // Take the appropriate action when we see a NaN (i.e. `trap()` or write 0x3f3f3f3f)
    static __device__ __forceinline__
    void take_action_when_have_nan(const TopkSelectArgs &args, uint32_t batch_idx) {
        if (args.abort_when_nan_found) {
            if (threadIdx.x == 0) {
                printf("[topk_select] NaN detected. Calling `trap;` which will result in \"unspecified launch failure\"\n");
            }
            __syncthreads();
            ku::trap();
        } else {
            if (threadIdx.x == 0) {
                *((OutIdxT*)args.output_index + batch_idx * args.stride_output_index_batch) = 0x3F3F3F3F;
            }
        }
    }
};

template<typename Config>
class TopkSelectKernelBF16Base : public TopkSelectKernelBase<Config> {
    using Base = TopkSelectKernelBase<Config>;
public:
    using ValueT = typename Base::ValueT;
    using OutIdxT = typename Base::OutIdxT;
    using SharedMemoryPlanBase = typename Base::SharedMemoryPlanBase;
    using EpilogueT = typename Base::EpilogueT;
    using Base::NUM_THREADS;
    using Base::NUM_WARPS;
    using Base::MAX_TOPK;
    using Base::NUM_ELEMS_PER_128b;
    using Base::NUM_UINT32_PER_128b;
    using Base::NUM_ELEMS_PER_ROUND;
    using Base::NUM_ELEMS_PER_THREAD_PER_ROUND;
    using Base::ELEMS_PER_THREAD_PER_ROUND_MASK;
    using Base::NUM_128b_PER_THREAD_PER_ROUND;
    using Base::NUM_ELEMS_PER_SEG;
    using Base::NUM_TAIL_ELEMS;
    using Base::NUM_TAIL_SEGS;
    using Base::NUM_ELEMS_IN_INIT_WINDOW;
    using Base::NUM_SEGS_IN_INIT_WINDOW;
    using Base::NUM_INIT_ROUNDS_MAX;
    using Base::NUM_UINT32_IN_INIT_WINDOW_PER_THREAD;
    using Base::NUM_128b_INIT_PER_THREAD;
    using Base::NUM_SEGS_PER_ROUND;
    using Base::NUM_SEGS_PER_ISSUE_WARP;
    using Base::NUM_ISSUE_WARPS;
    using Base::NUM_TMA_LOAD_BUFS;
    using Base::TMA_PREFETCH_DEPTH;
    using Base::RECONSTRUCT_THRESHOLD;
    using Base::PERM_ADD_BASE;
    using Base::PERM_MUL_PRIME;
    using Base::NUM_EXTRA_SLOTS;
    using Base::PLACEHOLDER_PAIR;
    using Base::NUM_RECONSTRUCT_BUCKETS;
    using Base::NUM_RECONSTRUCT_UNITS_MAX;

    static_assert(std::is_same_v<ValueT, maca_bfloat16>);
    static_assert(!Config::sorted_value);
    static constexpr uint32_t NUM_UINT32_RECONSTRUCT_PER_THREAD =
        ((NUM_RECONSTRUCT_UNITS_MAX + NUM_THREADS - 1) / NUM_THREADS) | 1u; // padding
    static_assert(2 * NUM_UINT32_RECONSTRUCT_PER_THREAD <= 256);
    static constexpr uint16_t PLACEHOLDER = 0xff80; // -INF
    static constexpr uint32_t NEG_INF_X2_BITS = 0xFF80FF80u;

    // Histogram pass 1 (MSB radix): for each distorted bf16x2 value, retrieve the upper 8 bits and performs atomicAdd on the bucket counter
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    void histogram_radix_msb(uint32_t *bucket_counter, const maca_bfloat162 (&packed_values)[N], uint32_t num_packed_values) {
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            uint32_t raw = bf16x2_to_u32(packed_values[i]);
            uint32_t distorted;
            distort_x2<uint16_t>((uint16_t*)&distorted, (const uint16_t*)&raw);
            // [MACA] 原为内联 PTX：bfe.u32 取两个半字各高 8 位，mad.lo.u32 由桶号
            //   算桶地址（*4），red.shared.add.u32 做共享内存原子加。
            //   MACA 没有 bfe/mad/red 的 asm 拼写，但语义就是「取位段 + 指针寻址
            //   + atomicAdd」，直接写普通 C++：shared 上的 atomicAdd 是 MACA 支持的
            //   （已探针实测），编译器自行选指令。
            uint32_t bucket_idx_0 = (distorted >> 8) & 0xFFu;
            uint32_t bucket_idx_1 = (distorted >> 24) & 0xFFu;
            atomicAdd(bucket_counter + bucket_idx_0, 1u);
            atomicAdd(bucket_counter + bucket_idx_1, 1u);
        }
    }

    // Histogram pass 2 (LSB radix refinement): among values whose distorted MSB equals pivot_hi8, retrieve the lower 8 bits and performs atomicAdd on the bucket counter
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    // `pivot_hi8` is the highest 8 bit of the DISTORTED pivot
    template<bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    void histogram_radix_lsb_for_pivot_msb(uint32_t *bucket_counter, uint32_t pivot_hi8, const maca_bfloat162 (&packed_values)[N], uint32_t num_packed_values) {
        // Every elements in the pivot bucket must have the same sign, so we can use this information to optimize instead of
        bool bucket_negative = pivot_hi8 < 0x80;
        uint32_t orig_pivot_hi8 = bucket_negative ? 0xFF - pivot_hi8 : pivot_hi8 - 0x80;    // Un-distort
        uint32_t raw_hi8_x2 = (orig_pivot_hi8 << 16) | orig_pivot_hi8;
        uint32_t low_byte_xor = bucket_negative ? 0xFFFFFFFFu : 0u;
        // Elements outside the bucket are counted in this row's sink slot (index NUM_RECONSTRUCT_BUCKETS) and never read.
        uint32_t scratch_bucket_idx_x2 = 0x01000100;    // 2x 256
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            // [MACA] 原为一段内联 PTX，逐条等价改写如下（每条右侧注释即原指令语义）：
            uint32_t raw = bf16x2_to_u32(packed_values[i]);
            // prmt.b32 hi, raw, 0, 0x5341 —— 取两半各自的**高字节**放到 0x00hh 形态
            uint32_t hi = ((raw >> 8) & 0x000000FFu) | ((raw >> 16) & 0x00FF0000u);
            // set.eq —— 高字节等于 pivot 的半字给 0xFFFF。
            //   这里比的是**抽出来的高字节**（0x00hh 形态），不是 bf16 值，所以要的是
            //   逐字节相等，`bf16x2_eq_mask` 的按位比较正是它；不能用 census 那个
            //   浮点版（0x00hh 会被当成极小的 denormal 去比大小）。
            uint32_t sel = bf16x2_eq_mask(hi, raw_hi8_x2);
            // lop3.b32 lo, raw, low_byte_xor, 0x00ff00ff, 0x28 —— lo = (raw ^ xor) & 0x00ff00ff
            uint32_t lo = (raw ^ low_byte_xor) & 0x00ff00ffu;
            // lop3.b32 t, sel, lo, scratch, 0xca —— t = sel ? lo : scratch
            uint32_t t = (sel & lo) | (~sel & scratch_bucket_idx_x2);
            // and/shr 拆两半 + mad.lo.u32 算桶地址（*4）+ red.shared.add 原子加
            atomicAdd(bucket_counter + (t & 0xFFFFu), 1u);
            atomicAdd(bucket_counter + (t >> 16), 1u);
        }
    }

    // Count the number of elements that are 1) > pivot 2) = pivot 3) NaN in the given array
    struct CensusCounts { uint32_t cnt_gt; uint32_t cnt_eq; uint32_t cnt_nan; };
    // `num_packed_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    CensusCounts get_census_counts(const maca_bfloat162 (&values)[N], uint32_t num_packed_values, uint32_t pivot_value_x2_bits) {
        static_assert(2 * N <= 256);    // Since we're going to use bf16 for accumulation
        // [MACA] 原为三条内联 PTX（set.gt/.eq/.nan.bf16x2.bf16x2 + add.rn.bf16x2）。
        //   语义要点：set 指令的**目标是 .bf16x2**，所以比较结果按 1.0bf16 / 0.0bf16
        //   给出（不是掩码 —— 掩码形态只在目标为 .s32 时出现，例如上面的
        //   histogram_radix_lsb_for_pivot_msb 里的 set.eq.s32.bf16x2）；再逐半字
        //   用 bf16 加法累加计数。
        //   这里改为逐半字判定 + float32 累加：2*N <= 256，计数始终是小整数，
        //   float32 与 bf16 的累加在这些值上都是**精确**的，所以结果逐位相同，
        //   而代码不再依赖 bf16 加法的舍入行为。
        //   .gt / .eq 用的是**有符号**16 位比较（NaN 由 .nan 单独统计，不参与 gt）。
        float cnt_gt_f = 0.0f, cnt_eq_f = 0.0f, cnt_nan_f = 0.0f;
        CUTE_UNROLL
        for (uint32_t i = 0; i < N; i++) {
            if (i % NUM_PACKED_VALUES_ALIGNMENT == 0 && i == num_packed_values) break;
            uint32_t raw = bf16x2_to_u32(values[i]);
            uint32_t gt_mask = bf16x2_gt_mask_float(raw, pivot_value_x2_bits);
            uint32_t eq_mask = bf16x2_eq_mask_float(raw, pivot_value_x2_bits);
            cnt_gt_f  += (float)((gt_mask & 0xFFFFu) ? 1 : 0) + (float)((gt_mask >> 16) ? 1 : 0);
            cnt_eq_f  += (float)((eq_mask & 0xFFFFu) ? 1 : 0) + (float)((eq_mask >> 16) ? 1 : 0);
            cnt_nan_f += (float)(bf16_is_nan((uint16_t)raw) ? 1 : 0)
                       + (float)(bf16_is_nan((uint16_t)(raw >> 16)) ? 1 : 0);
        }
        uint32_t cnt_gt = (uint32_t)cnt_gt_f;
        uint32_t cnt_eq = (uint32_t)cnt_eq_f;
        // [MACA] `cnt_nan` is a flag, not a count -- it is only ever consumed as
        // `cnt_nan != 0` by the two callers.  Upstream reaches the same value
        // with `min.NaN` folded into the census and one `set.nan` afterwards,
        // which MACA has no instruction for; counting and collapsing here is
        // the equivalent.
        uint32_t cnt_nan = cnt_nan_f > 0.0f ? 1u : 0u;
        return {cnt_gt, cnt_eq, cnt_nan};
    }

    // 判定一个 bf16 半字是否入选；等于 pivot 时按配额放行并消耗一份配额。
    // 语义严格对应原先内联 PTX 的三条谓词逻辑：
    //   setp.gt.bf16x2 g  —— **浮点**比较，NaN 参与时为假
    //   setp.eq.bf16x2 e + setp.ne.and.u32 t, quota, 0, e  —— 浮点等值且配额还有
    //   or.pred s = g | t
    // 注意 `gt` 命中的元素**不**消耗配额（原 PTX 只在 t 为真时 sub）。
    //
    // [MACA] 原为按位型做 int16 比较。逐半字的**等值**恰好与浮点等值同解（除 NaN
    //   与 ±0），但**大小**不同解：+NaN 的位型作 int16 是最大的正数，于是
    //   "+NaN > 任何正数" 为真，而浮点 `gt` 为假 —— 这会让排在 pivot 之后的 NaN
    //   抢占配额。故一律经 `bf16_bits_to_float` 走真浮点比较。
    static __device__ __forceinline__
    bool select_one_half(uint16_t value_half, uint32_t pivot_value_x2_bits, uint32_t &eq_quota) {
        uint16_t pivot_half = (uint16_t)(pivot_value_x2_bits & 0xFFFFu);
        float value_f = bf16_bits_to_float(value_half);
        float pivot_f = bf16_bits_to_float(pivot_half);
        if (value_f > pivot_f) return true;
        if (value_f == pivot_f && eq_quota != 0) {
            eq_quota -= 1;
            return true;
        }
        return false;
    }

    // Decide whether or not to accept an element in the old survivor buffer
    // If taken, update relevant status (`dst_ptr` and `eq_quota`)
    //
    // [MACA] 形参由「shared 窗口地址(uint32)」改为真指针：原先的地址算术依赖
    //   cute::cast_smem_ptr_to_uint 的返回值能被当作通用指针再解引用
    //   （调用点确实这么做了），这在 MACA 上并不成立；改用指针后
    //   ld/st 就是普通的 shared 访存，MACA 直接支持。
    // 逐对处理：pair0 = {index0, value0}，pair1 = {index1, value1}（各 8 字节）。
    // 元素 1 的配额判定发生在元素 0 递减**之后**，这个顺序必须保持。
    template<bool USE_CLUSTER_ADDRESSING>
    static __device__ __forceinline__
    void copy_selected_pairs_to_survivor(
        uint64_t *&dst,     // 新 survivor 缓冲的写指针（按 8 字节对递增）
        uint32_t &eq_quota, // 本线程的 EQ 配额，可能减小
        maca_bfloat162 packed_values,
        uint32_t pivot_value_x2_bits,
        const uint64_t *src // {index, value} 对，共 2 个
    ) {
        uint32_t values_raw = bf16x2_to_u32(packed_values);
        if (select_one_half((uint16_t)values_raw, pivot_value_x2_bits, eq_quota)) {
            dst[0] = src[0];
            dst += 1;
        }
        if (select_one_half((uint16_t)(values_raw >> 16), pivot_value_x2_bits, eq_quota)) {
            dst[0] = src[1];
            dst += 1;
        }
        // [MACA] 原 PTX 的非 cluster 分支用 `add.f32 %0, %0, 0f00000008` 把地址加 8
        //   ——上游刻意借浮点管线做整数加法以免与比较/位运算争整数管线。C++ 里
        //   没有这个约束，直接指针自增即可（微观调度不再由我们控制，语义不变）。
    }

    // The src-in-register version of `copy_selected_pairs_to_survivor`
    //
    // 与上者的差别只在数据来源：元素 0 的值字取 values_raw 整体（其低半字是
    // 有效值），元素 1 的值字取 `values_raw >> 16`（干净的高半字），索引由调用方
    // 以寄存器给出。对齐到 pair 布局的 64 位形式：低 32 位 = index，高 32 位 = value。
    template<bool USE_CLUSTER_ADDRESSING>
    static __device__ __forceinline__
    void append_selected_pairs_from_registers(
        uint64_t *&dst,
        uint32_t &eq_quota,
        maca_bfloat162 packed_values,
        uint32_t pivot_value_x2_bits,
        uint32_t index0,
        uint32_t index1
    ) {
        uint32_t values_raw = bf16x2_to_u32(packed_values);
        uint32_t val_word1 = values_raw >> 16;   // element 1 low, clean high
        if (select_one_half((uint16_t)values_raw, pivot_value_x2_bits, eq_quota)) {
            dst[0] = ((uint64_t)values_raw << 32) | (uint64_t)index0;
            dst += 1;
        }
        if (select_one_half((uint16_t)val_word1, pivot_value_x2_bits, eq_quota)) {
            dst[0] = ((uint64_t)val_word1 << 32) | (uint64_t)index1;
            dst += 1;
        }
    }

    // Get the pivot via a two-stage histogram and get PivotAndQuota
    // The MSB histogram should be ready and all threads should be able to see it (e.g. via a `__syncthreads()`) before we call the function
    // We don't absorb MSB histogram's logic inside since sometimes we'd like to overlap TMA load and MSB histogram's modification
    struct PivotAndQuota {
        uint32_t pivot_value_x2_bits;
        uint32_t start_pos_in_collector;
        uint32_t eq_quota;
        uint32_t cnt_nan;
    };
    // `num_values` should be guaranteed to be aligned by `NUM_PACKED_VALUES_ALIGNMENT`
    template<bool USE_CLUSTER_ADDRESSING, uint32_t N, uint32_t NUM_PACKED_VALUES_ALIGNMENT = 1>
    static __device__ __forceinline__
    PivotAndQuota compute_pivot_and_quota(uint32_t topk, const maca_bfloat162 (&values)[N],
                                       uint32_t num_values, uint32_t warp_idx, uint32_t lane_idx, SharedMemoryPlanBase &smem) {
        if (warp_idx == 0) {
            Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[0], topk, lane_idx);
        }
        __syncthreads();
        uint32_t pivot_hi8 = smem.reconstruct_pivot_bucket;
        uint32_t num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;

        histogram_radix_lsb_for_pivot_msb<USE_CLUSTER_ADDRESSING, N, NUM_PACKED_VALUES_ALIGNMENT>(smem.reconstruct_bucket_counter[1], pivot_hi8, values, num_values);
        __syncthreads();

        if (warp_idx == 0) {
            Base::template find_pivot_in_histogram<false>(smem, smem.reconstruct_bucket_counter[1], num_elem_should_select_in_pivot_bucket, lane_idx);
        }
        __syncthreads();
        uint32_t pivot_lo8 = smem.reconstruct_pivot_bucket;
        uint32_t pivot_distorted = (pivot_hi8 << 8) | pivot_lo8;
        uint16_t pivot_value = un_distort((uint16_t)pivot_distorted);
        uint32_t pivot_value_x2_bits = ((uint32_t)pivot_value << 16) | pivot_value;

        auto census = get_census_counts<N, NUM_PACKED_VALUES_ALIGNMENT>(values, num_values, pivot_value_x2_bits);

        static_assert(NUM_WARPS <= NUM_RECONSTRUCT_BUCKETS);
        auto eqgt = Base::compute_equal_quota_and_prefix(census.cnt_gt, census.cnt_eq, topk, warp_idx, lane_idx, smem.reconstruct_bucket_counter[0]);
        return {pivot_value_x2_bits, eqgt.start_pos_in_collector, eqgt.eq_quota, census.cnt_nan};
    }

    // Main part of the normal kernel and phase A of the cluster kernel.
    // The input is divided into:
    //  - a "perm" prefix [0, num_perm_segs * NUM_ELEMS_PER_SEG), visited in pseudo-random order;
    //  - a "tail" suffix [num_perm_segs * NUM_ELEMS_PER_SEG, end_vocab_idx), visited sequentially.
    // The init phase loads the tail plus the first perm segments into one init window, selects
    // the initial top-k from it, then the main loop continues over the remaining perm segments.
    //
    // Returns the number of real selected pairs in the final survivor buffer; this is normally
    // `topk`, but can be smaller in a cluster CTA whose local range has fewer real elements.
    template<bool USE_CLUSTER_ADDRESSING, typename IsWarpActiveF>
    static __device__ __forceinline__
    uint32_t scan_segs(const TopkSelectArgs &args, SharedMemoryPlanBase &smem,
                       uint32_t batch_idx, uint32_t end_vocab_idx, uint32_t topk,
                       uint32_t warp_idx, uint32_t lane_idx,
                       uint32_t num_perm_segs,                  // The number of segments to be permuted, globally
                       uint32_t local_start_seg_idx,            // The index of the first "local" segment ("local" means "belong to this CTA"). 0 if not the cluster-based implementation
                       uint32_t num_local_perm_segs,            // The number of permuted segments to be processed, locally
                       uint32_t num_local_tail_elems_padded,    // num_local_tail_elems, padded to NUM_TAIL_ELEMS if num_perm_segs > 0, otherwise padded to NUM_ELEMS_PER_SEG
                       uint32_t num_local_tail_elems,           // The number of tail elems locally
                       uint32_t &survivor_buf_idx,
                       bool &have_nan,
                       IsWarpActiveF &&is_warp_active_f) {
        uint32_t num_local_elems_padded = num_local_tail_elems_padded + num_local_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_local_rounds = ku::ceil_div(num_local_elems_padded, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_init_rounds = min(num_local_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
        uint32_t num_local_tail_segs_padded = num_local_tail_elems_padded / NUM_ELEMS_PER_SEG;
        uint32_t num_main_rounds = num_local_rounds - num_init_rounds;

        // Permutation generation arguments
        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t perm_mul = PERM_MUL_PRIME % perm_len;
        (void)NUM_ISSUE_WARPS;

        // [MACA] 原为 TMA 发射侧跨轮维护的置换状态（next_tma_permuted_segment /
        // tma_permuted_segment_stride）——只有 elect 出来的那一条 lane 有效。
        // ldg 装载是全线程协作的，每轮直接由线性位置算出段号（load_round_for_round），
        // 不再需要这份跨轮状态，故删除。

        uint32_t threshold_x2_bits = NEG_INF_X2_BITS;
        uint32_t num_incomers = 0;
        uint32_t num_survivors = 0;

        // Init phase
        if (num_local_elems_padded != 0) {
            // 装载 init window（尾段 + 头若干置换段），落在 init_buf[0: NUM_ELEMS_IN_INIT_WINDOW]。
            // MACA 无 TMA / mbarrier / 预取流水线：全线程协作 ldg，轮次之间无依赖
            // （各 init 轮写 incoming_topk_pairs 的不同区段，互不覆盖），故一次
            // __syncthreads() 即完成发布，随后才开始消费。
            ValueT *init_buf = (ValueT*)smem.incoming_topk_pairs;
            if (num_local_tail_elems_padded != 0) {
                Base::template load_round_for_round<true, true>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    0, local_start_seg_idx, warp_idx, lane_idx);
            } else {
                Base::template load_round_for_round<true, false>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    0, local_start_seg_idx, warp_idx, lane_idx);
            }
            CUTE_UNROLL
            for (uint32_t i = 1; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i >= num_init_rounds) break;
                Base::template load_round_for_round<true, false>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    i, local_start_seg_idx, warp_idx, lane_idx);
            }
            __syncthreads();

            if (num_perm_segs != 0 && num_local_tail_elems_padded != 0) {
                Base::fill_padded_tail_segments(init_buf, num_local_tail_elems);
                __syncthreads();
            }

            // Decide how many 128-bit elements each thread processes.
            uint32_t num_elems_in_init_window_padded = num_local_tail_elems_padded + min(num_local_perm_segs, (uint32_t)NUM_SEGS_IN_INIT_WINDOW - num_local_tail_segs_padded) * NUM_ELEMS_PER_SEG;  // Init window size, padded to NUM_ELEMS_PER_SEG
            uint32_t num_128b_in_init_window_padded = num_elems_in_init_window_padded / NUM_ELEMS_PER_128b;
            uint32_t cnt_floor = num_128b_in_init_window_padded / NUM_THREADS;
            uint32_t cnt_rem = num_128b_in_init_window_padded % NUM_THREADS;
            uint32_t my_elem_start_idx = (threadIdx.x * cnt_floor + min(threadIdx.x, cnt_rem)) * NUM_ELEMS_PER_128b;
            uint32_t num_my_elems = NUM_ELEMS_PER_128b * cnt_floor + (threadIdx.x < cnt_rem ? NUM_ELEMS_PER_128b : 0u);
            uint32_t tail_padding_elems = num_local_tail_elems_padded - num_local_tail_elems;
            uint32_t num_real_init_elems = num_elems_in_init_window_padded - tail_padding_elems;
            num_survivors = min(topk, num_real_init_elems);

            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i == num_init_rounds) break;
                // [MACA] 原为 `smem.init_full_bar[i].wait(0)`（等该轮 TMA 事务完成）。
                // 装载已改为前面那次全线程协作 ldg + 一次 __syncthreads()，
                // 所有 init 轮在此处都已可见，无需逐轮等待。

                // Fill the incomplete segment with PLACEHOLDER (-INF)
                if (num_local_tail_elems % NUM_ELEMS_PER_SEG != 0 && i == num_local_tail_elems / NUM_ELEMS_PER_ROUND) {
                    uint32_t box_end = ku::ceil_div(num_local_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG) * NUM_ELEMS_PER_SEG;
                    for (uint32_t e = num_local_tail_elems + threadIdx.x; e < box_end; e += NUM_THREADS) {
                        init_buf[Base::sw_elem(e)] = __ushort_as_bfloat16(PLACEHOLDER);
                    }
                    __syncthreads();
                }
                maca_bfloat162 my_values[4 * NUM_128b_PER_THREAD_PER_ROUND];
                uint32_t num_my_values = 0;
                CUTE_UNROLL
                for (uint32_t k = 0; k < NUM_128b_PER_THREAD_PER_ROUND; ++k) {
                    uint32_t pos = threadIdx.x + (i * NUM_128b_PER_THREAD_PER_ROUND + k) * NUM_THREADS;
                    if (pos < num_128b_in_init_window_padded) {
                        ld_shared<4>(my_values + k * 4, reinterpret_cast<const maca_bfloat162*>(init_buf + Base::sw_b128(pos) * NUM_ELEMS_PER_128b));
                        num_my_values = 4 * (k + 1);
                    }
                }
                histogram_radix_msb<4 * NUM_128b_PER_THREAD_PER_ROUND, 4>(smem.reconstruct_bucket_counter[0], my_values, num_my_values);
            }

            maca_bfloat162 init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
                if (j * NUM_ELEMS_PER_128b == num_my_elems) break;
                uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
                ld_shared<4>(init_values + j * NUM_UINT32_PER_128b, reinterpret_cast<const maca_bfloat162*>(init_buf + Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
            }

            __syncthreads();    // publish the round-1 histogram

            static_assert(NUM_ELEMS_IN_INIT_WINDOW <= 0xFFFF);
            // The CTA's init-window slice may contain fewer real elements than K (e.g. a cluster
            // rank whose visit range mostly overlaps the padded tail). In that case the pivot
            // selection must degrade to "select all real elements": cap the K used in the pivot /
            // quota computation at the real count
            uint32_t effective_topk = min(topk, num_real_init_elems);
            auto [pivot_value_x2_bits, start_pos_in_collector, eq_quota, cnt_nan] = 
                compute_pivot_and_quota<USE_CLUSTER_ADDRESSING, NUM_UINT32_IN_INIT_WINDOW_PER_THREAD, 4>(effective_topk, init_values, num_my_elems / 2, warp_idx, lane_idx, smem);
            // The init census walks the thread's whole slice of the window, i.e. every element of the init
            // window exactly once, so NaNs inside the window are caught here.
            have_nan |= cnt_nan != 0;

            {
                uint64_t *out = smem.surviving_topk_pairs[0] + start_pos_in_collector;
                uint32_t s0 = local_start_seg_idx + (max(my_elem_start_idx, num_local_tail_elems_padded) - num_local_tail_elems_padded) / NUM_ELEMS_PER_SEG;
                uint32_t perm_state = Base::get_permuted_seg_idx(s0, perm_len, perm_mul);
                uint32_t unit_base = 0;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i++) {
                    // num_my_elems is a multiple of NUM_ELEMS_PER_128b (= 8), so i*2 == num_my_elems
                    // only at i % 4 == 0; the extra condition lets the compiler unroll by 4.
                    if (i % 4 == 0 && i*2 == num_my_elems) break;
                    if (i % 4 == 0) {
                        uint32_t g_u = my_elem_start_idx + i / 4 * NUM_ELEMS_PER_128b;
                        if (g_u < num_local_tail_elems_padded) {
                            unit_base = num_perm_segs * NUM_ELEMS_PER_SEG + g_u;
                        } else {
                            unit_base = perm_state * NUM_ELEMS_PER_SEG + g_u % NUM_ELEMS_PER_SEG;
                            if (g_u % NUM_ELEMS_PER_SEG == NUM_ELEMS_PER_SEG - NUM_ELEMS_PER_128b) {
                                Base::advance_perm_state(perm_state, perm_mul, perm_len);
                            }
                        }
                    }
                    // [MACA] 原为「把整数索引伪装成 fp32 次正规数再用浮点加法求
                    //   index = base + offset」——上游刻意走浮点管线避开整数管线竞争，
                    //   因为两个次正规数相加是精确的，结果与整数加法逐位相同。
                    //   C++ 里直接用整数：语义等价，且不必依赖次正规数的舍入前提。
                    uint32_t index0 = unit_base + (i % 4) * 2;
                    uint32_t index1 = index0 + 1;
                    maca_bfloat162 cur_value = init_values[i];      // two bf16 payloads packed as bf16x2
                    append_selected_pairs_from_registers<USE_CLUSTER_ADDRESSING>(out, eq_quota, cur_value, pivot_value_x2_bits, index0, index1);
                }
            }

            threshold_x2_bits = pivot_value_x2_bits;
        }
        __syncthreads();
    

        uint32_t linear_segment_start =
            local_start_seg_idx
            + num_init_rounds * NUM_SEGS_PER_ROUND
            - num_local_tail_segs_padded
            + warp_idx;
        uint32_t current_permuted_segment = Base::get_permuted_seg_idx(linear_segment_start, perm_len, perm_mul);
        uint32_t permuted_segment_stride_per_round = NUM_SEGS_PER_ROUND % perm_len * perm_mul % perm_len;

        // Re-select the top-k elements from the smem.surviving_topk_pairs[survivor_buf_idx][:num_survivors] + smem.incoming_topk_pairs[:num_incomers],
        // Write the selected pairs into the other survivor buffer, and return the new pivot
        auto reconstruct = [&](uint32_t cur_extra_len) -> uint32_t {
            uint32_t tid = threadIdx.x;
            Base::clear_reconstruct_histograms(smem, tid);

            uint32_t padded_extra_len = ku::ceil(cur_extra_len, 2u);    // Since we process elements in pairs (uint32_t)
            if (warp_idx == 0 && (cur_extra_len & 1u)) {
                smem.incoming_topk_pairs[cur_extra_len] = PLACEHOLDER_PAIR;
            }
            __syncthreads();

            // units [0, MAX_TOPK/2): the current survivor buffer
            // The rest: incomers
            uint32_t num_uint32 = (MAX_TOPK + padded_extra_len) / 2;
            uint32_t num_uint32_per_thread = ku::ceil_div(num_uint32, (uint32_t)NUM_THREADS) | 1u;   // force odd => bank-conflict-free smem gathers
            uint32_t unit_base = tid * num_uint32_per_thread;
            uint32_t num_my_units = unit_base < num_uint32 ? min(num_uint32_per_thread, num_uint32 - unit_base) : 0u;

            // [MACA] 原返回 shared 窗口地址(uint32)，调用方再把它当指针解引用；
            //   MACA 上这个往返不成立，直接返回真指针。
            auto unit_to_pair_addr = [&](uint32_t offset) -> const uint64_t * {
                return offset < MAX_TOPK / 2
                     ? smem.surviving_topk_pairs[survivor_buf_idx] + 2 * offset
                     : smem.incoming_topk_pairs + (2 * offset - MAX_TOPK);
            };

            maca_bfloat162 values[NUM_UINT32_RECONSTRUCT_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
                if (m == num_my_units) break;
                uint32_t pair2[4];
                ld_shared<4>(pair2, (const uint32_t*)unit_to_pair_addr(unit_base + m));
                // One b128 unit = 2 pairs {index0, value0, index1, value1}; pick the low halves of the
                // two value words (the bf16 payloads) and pack them into one bf16x2 word.
                values[m] = u32_to_bf16x2(__byte_perm(pair2[1], pair2[3], 0x5410));
            }

            histogram_radix_msb(smem.reconstruct_bucket_counter[0], values, num_my_units);
            __syncthreads();

            auto [pivot_value_x2_bits, out_prefix, eq_quota, cnt_nan] = 
                compute_pivot_and_quota<USE_CLUSTER_ADDRESSING>(topk, values, num_my_units, warp_idx, lane_idx, smem);
            // Every NaN that the main loop collected is part of the buffer this census just walked (the hit
            // test is `.gtu`, so NaN always becomes an incomer).
            have_nan |= cnt_nan != 0;

            { // write back
                uint64_t *out = smem.surviving_topk_pairs[survivor_buf_idx ^ 1] + out_prefix;
                CUTE_UNROLL
                for (uint32_t m = 0; m < NUM_UINT32_RECONSTRUCT_PER_THREAD; m++) {
                    if (m == num_my_units) break;
                    copy_selected_pairs_to_survivor<USE_CLUSTER_ADDRESSING>(out, eq_quota, values[m], pivot_value_x2_bits, unit_to_pair_addr(unit_base + m));
                }
            }

            survivor_buf_idx ^= 1;
            return pivot_value_x2_bits;
        };

        uint32_t logical_elem_offset = threadIdx.x * NUM_ELEMS_PER_THREAD_PER_ROUND;
        uint32_t offset_in_segment = logical_elem_offset % NUM_ELEMS_PER_SEG;

        uint32_t swizzle_mask = Base::sw_msk(logical_elem_offset);
        uint32_t chunk_swizzle_mask = swizzle_mask & ELEMS_PER_THREAD_PER_ROUND_MASK;
        uint32_t smem_read_offset = logical_elem_offset ^ (swizzle_mask & ~ELEMS_PER_THREAD_PER_ROUND_MASK);

        // Track the TMA buffer index and its phase incrementally instead of dividing every round.
        // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
        // row's true one and the append rate (hence the store path) stays high for the whole scan:
        // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than the
        // extra appends cost, so they keep the plain threshold.
        uint32_t reconstruct_trigger = num_main_rounds > 16 ? RECONSTRUCT_THRESHOLD / 4 : RECONSTRUCT_THRESHOLD;

        for (uint32_t main_round_idx = 0; main_round_idx < num_main_rounds; ++main_round_idx) {
            // [MACA] 原为「elect 一条 lane 预取 main_round_idx + TMA_PREFETCH_DEPTH，
            // 本轮消费等 tma_load_full_bar」。单缓冲 + 无 TMA：改成
            // 「全线程协作装载本轮 → __syncthreads() → 全线程消费本轮」。
            // 轮索引用绝对编号（init 轮在前），置换线性位置的推导才与消费端一致。
            Base::template load_round_for_round<false, false>(
                smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                num_init_rounds + main_round_idx, local_start_seg_idx, warp_idx, lane_idx);
            __syncthreads();

            bool is_warp_active = is_warp_active_f(main_round_idx);

            const ValueT *buf = smem.tma_load_buf;
            // One bit per element of this thread's slice, in element order.
            // Hits are rare, so appending through the set bits of this mask is faster than testing + storing every element again.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32);        // hit_mask is a uint32_t
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND % 4 == 0);     // the loop below packs 4 elements per nibble
            uint32_t hit_mask = 0;
            if (is_warp_active) {
                ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
                Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(values, buf, smem_read_offset, chunk_swizzle_mask);
                // [MACA] 原为一段内联 PTX：两条 set.gtu.s32.bf16x2 做**无符号**16 位比较
                //   （"NaN 一律接受"——两个符号的 NaN 在位型上都大于任何有限值），
                //   prmt 取每个半字的高字节拼成 ind，再用 dp4a 以带符号字节权重
                //   {0xFF,-1},{0xFE,-2},{0xFC,-4},{0xF8,-8} 把 4 个 0x00/0xFF 字节
                //   压成 4 位 nibble（每命中一位贡献 1/2/4/8），最后 mad 把 nibble
                //   接到 hit_mask 低位并整体左移 4。
                //   这里逐条等价改写：dp4a 在 MACA 上不可用，但它在此处只是"把 4 个
                //   0x00/0xFF 字节各取 1 bit 打包"，直接用移位/或即可，不需要乘法。
                CUTE_UNROLL
                for (int32_t j = NUM_ELEMS_PER_THREAD_PER_ROUND - 4; j >= 0; j -= 4) {
                    // [MACA] 原为一条内联 PTX：两条 `set.gtu.s32.bf16x2` 判命中、
                    //   `prmt` 抽位、`dp4a` 打包成 nibble、`mad.lo.u32` 接到 hit_mask。
                    //   这里逐条等价改写（dp4a 在 MACA 上不可用，但它在此处只是"把 4 个
                    //   0x00/0xFF 字节各取 1 bit 打包"，用移位/或即可）。
                    //   `.gtu` = greater-than-unordered（浮点语义）：NaN 一律算命中，
                    //   于是每个 NaN 都会落进 incoming buffer，由 census 报出来 ——
                    //   这是 0f03b68 之后 NaN 检测的唯一入口，见 `bf16x2_gtu_mask_float`。
                    uint32_t gt0 = bf16x2_gtu_mask_float(*(const uint32_t*)(values + j),     threshold_x2_bits);
                    uint32_t gt1 = bf16x2_gtu_mask_float(*(const uint32_t*)(values + j + 2), threshold_x2_bits);
                    // 逐半字命中 → 0xFFFF；取每个半字的 1 个 bit 组成 nibble
                    uint32_t nib = ((gt0 & 0xFFFFu) ? 1u : 0u)
                                 | ((gt0 >> 16)     ? 2u : 0u)
                                 | ((gt1 & 0xFFFFu) ? 4u : 0u)
                                 | ((gt1 >> 16)     ? 8u : 0u);
                    hit_mask = hit_mask * 16u + nib;
                }
            }
            uint32_t num_new_incomers = __popc(hit_mask);

            uint32_t warp_total_hits = __reduce_add_sync(0xFFFFFFFF, num_new_incomers);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total_hits;
            }
            __syncthreads();
            
            static_assert(NUM_WARPS <= 32);
            uint32_t stored_warp_hits = lane_idx < NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
            uint32_t num_total_hits_in_this_round = __reduce_add_sync(0xFFFFFFFF, stored_warp_hits);
            
            uint32_t seg_elem_base = current_permuted_segment * NUM_ELEMS_PER_SEG + offset_in_segment;

            // Element e of this thread's slice lives at smem_read_offset + (e ^ chunk_swizzle_mask)
            // [MACA] 原先把该地址经 cast_smem_ptr_to_uint 变成整数再做 ld.shared/st.shared；
            //   改成真指针 + 普通 shared 访存。
            const ValueT *elem_base = buf + smem_read_offset;

            // Start the first hit's smem load before the prefix scan / count exchange below, so that its
            // latency (and the barrier wait) overlaps with them instead of delaying the first store.
            uint32_t first_hit_e = hit_mask != 0 ? __ffs(hit_mask) - 1u : 0u;
            uint32_t first_hit_val = 0;
            if (is_warp_active && warp_total_hits != 0) {
                first_hit_val = *(const uint16_t*)(elem_base + (first_hit_e ^ chunk_swizzle_mask));
            }

            // Lane prefix via one ballot per bit of the (<= 16) hit count: the ballots are independent,
            // so this is much shorter on the critical path than a 5-step shuffle scan.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND < 32);     // one ballot per bit of num_new_incomers
            uint32_t lane_prefix = 0;
            CUTE_UNROLL
            for (uint32_t k = 0; k < 5; ++k) {
                uint32_t bit = __ballot_sync(0xFFFFFFFF, (num_new_incomers >> k) & 1u) & ((1u << lane_idx) - 1u);
                lane_prefix += (uint32_t)__popc(bit) << k;
            }

            uint32_t dst_slot = 
                num_incomers +
                __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_hits : 0u) +
                lane_prefix;

            if (is_warp_active && warp_total_hits != 0) {
                // pair(64b) 布局：低 32 位 = index，高 32 位 = value
                uint64_t *dst_ptr = smem.incoming_topk_pairs + dst_slot;
                if (hit_mask != 0) {
                    *dst_ptr++ = ((uint64_t)first_hit_val << 32) | (uint64_t)(seg_elem_base + first_hit_e);
                }
                uint32_t mask = hit_mask & (hit_mask - 1u);
                while (mask != 0) {
                    uint32_t e = __ffs(mask) - 1u;
                    mask &= mask - 1u;
                    uint32_t val_word = *(const uint16_t*)(elem_base + (e ^ chunk_swizzle_mask));
                    *dst_ptr++ = ((uint64_t)val_word << 32) | (uint64_t)(seg_elem_base + e);
                }
            }
            Base::advance_perm_state(current_permuted_segment, permuted_segment_stride_per_round, perm_len);

            num_incomers += num_total_hits_in_this_round;
            // No warp may run ahead into the next round: the load at the top of the next
            // iteration reuses the single smem buffer this round is reading, and warp_cnt
            // must not be overwritten while others read it.
            __syncthreads();

            // On long rows the init window is a tiny fraction of the row, so its pivot stays far above the
            // row's true one and the append rate (hence the store path) stays high for the whole scan:
            // reconstruct early to tighten the pivot. Short rows would pay more for the reconstruct than
            // the extra appends cost, so they keep the plain threshold.
            if (num_incomers >= reconstruct_trigger && main_round_idx + 1 < num_main_rounds) {
                threshold_x2_bits = reconstruct(num_incomers);
                num_incomers = 0;
            }
        }

        if (num_incomers > 0) {
            reconstruct(num_incomers);
            num_incomers = 0;
        }

        return num_survivors;
    }
};

} // namespace topk_select_common
