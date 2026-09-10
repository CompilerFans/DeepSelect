/*
TopK select kernel fp32 variant. 

Usage caveats:
  - vocab_size must be < 2^23.
  - Do NOT assume that the kernel will select a prefix index on ties.
*/

#pragma once

#include "topk_select.h"

#include <cutlass/kernel_launch.h>
// [MACA] 原为 <cute/arch/copy_sm90_tma.hpp>：MACA 无 TMA（无 tensor map、无
//   mbarrier、无 elect 发射流水线）。数据装载已改为全线程协作的
//   `Base::load_round_for_round`（见 common_parts.cuh），此处不再需要该头。
#include <kerutils/kerutils.cuh>

#include "structs.h"
#include "cuda_kernels/utils.cuh"
#include "cuda_kernels/common_parts.cuh"

namespace topk_select_fp32 {

template<typename Config>
class TopkSelectKernelFP32 : public topk_select_common::TopkSelectKernelBase<Config> {
    using Base = topk_select_common::TopkSelectKernelBase<Config>;
public:
    using ValueT = typename Base::ValueT;
    using OutIdxT = typename Base::OutIdxT;
    // [MACA] 原为 `using TmaParams = typename Base::TmaParams;`（CUtensorMap + 屏障）。
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
    using Base::NUM_PERM_SEGS_IN_INIT_WINDOW;
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
    using Base::PLACEHOLDER_PAIR;
    using Base::NUM_RECONSTRUCT_BUCKETS;

    static constexpr uint32_t PLACEHOLDER = 0xff800000; // -INF
    static constexpr uint32_t NEG_INF_BITS = 0xFF800000;
    static_assert(MAX_TOPK == 512 || MAX_TOPK == 1024 || MAX_TOPK == 4096);
    static constexpr bool HAS_PARTIAL_ROUNDS = NUM_SEGS_PER_ROUND > NUM_TAIL_SEGS;

    struct SharedMemoryPlanFP32 : SharedMemoryPlanBase {};
    
    // [MACA] 原为「把 shared 地址当整数收进来、`__cvta_shared_to_generic` 转回指针再写」，
    //   且用 `__float_as_uint(__uint_as_float(dst_ptr) + __uint_as_float(8u))` 以浮点加法
    //   代替整数加法（上游 :206-207 的说明：整数加法与比较/位运算争用同一发射端口，
    //   浮点加法走另一条流水线；取值 < 2^23 时 fp32 可精确表示该整数）。
    //   MACA 上这既是不可移植的假设（`__cvta_shared_to_generic` 与 PTX 绑定），
    //   其收益也从未在 MACA 上测过，故按 common_parts.cuh 的统一做法改为真指针 + 整数加法。
    static __device__ __forceinline__
    void append_pair_at(uint64_t *&dst, uint32_t index, uint32_t val_bits) {
        // pair(64b) 布局：低 32 位 = index，高 32 位 = value
        *dst++ = ((uint64_t)val_bits << 32) | (uint64_t)index;
    }

    static __device__ __forceinline__ void topk_select_kernel_devfunc(const TopkSelectArgs &args) {
        uint32_t batch_idx = blockIdx.x;
        uint32_t end_vocab_idx = args.end_ptr == nullptr ? args.vocab_size : __ldg(args.end_ptr + batch_idx);

        extern __shared__ CUTE_ALIGNAS(1024) char wksp_buf[];
        SharedMemoryPlanFP32 &smem = *reinterpret_cast<SharedMemoryPlanFP32*>(wksp_buf);

        uint32_t warp_idx = ku::canonical_warp_idx_sync();
        uint32_t lane_idx = threadIdx.x % 32;

        if (end_vocab_idx <= args.topk) {
            EpilogueT::template topk_select_epilogue<true>(
                (ValueT*)smem.surviving_topk_pairs[0],
                (uint32_t*)(smem.surviving_topk_pairs[1]),
                args,
                batch_idx, end_vocab_idx, warp_idx,
                *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs)
            );
            return;
        }

        uint32_t threshold_bits = NEG_INF_BITS;
        uint32_t num_incomers = 0;
        uint32_t survivor_buf_idx = 0;  // current candidate buffer (A/B, swapped by reconstruct)
        bool have_nan = false;

        Base::init_shared_memory(smem, warp_idx);

        uint32_t num_input_segs = ku::ceil_div(end_vocab_idx, (uint32_t)NUM_ELEMS_PER_SEG);
        uint32_t num_perm_segs = num_input_segs <= NUM_SEGS_IN_INIT_WINDOW ? 0u : (num_input_segs - 1) / NUM_TAIL_SEGS * NUM_TAIL_SEGS;
        uint32_t num_perm_elems = num_perm_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_local_tail_elems_padded = num_perm_segs != 0 ? (uint32_t)NUM_TAIL_ELEMS : num_input_segs * NUM_ELEMS_PER_SEG;
        uint32_t num_rounds = ku::ceil_div(num_local_tail_elems_padded + num_perm_elems, (uint32_t)NUM_ELEMS_PER_ROUND);
        uint32_t num_init_rounds = min(num_rounds, (uint32_t)NUM_INIT_ROUNDS_MAX);
        ValueT *init_buf = (ValueT*)smem.incoming_topk_pairs;   // aliased

        uint32_t perm_len = max(num_perm_segs, 1u);
        uint32_t perm_mul = PERM_MUL_PRIME % perm_len;
        static_assert(NUM_SEGS_PER_ROUND == NUM_SEGS_PER_ISSUE_WARP * NUM_ISSUE_WARPS);
        __shared__ bool should_select_whole_bucket_shared;

        auto find_pivot_in_histogram = [&](uint32_t topk, const uint32_t *bucket_counter) {
            bool should_select_whole_bucket =
                Base::template find_pivot_in_histogram<true>(smem, bucket_counter, topk, lane_idx);
            if (threadIdx.x == 0) {
                should_select_whole_bucket_shared = should_select_whole_bucket;
            }
        };
        // First radix pass: histogram the most significant byte of each distorted
        // fp32 value. Used to locate the top-k bucket prefix.
        uint32_t *hist_base = smem.reconstruct_bucket_counter[0];
        // [MACA] 原为一条内联 PTX：`mad.lo.u32` 直接算出桶地址、`red.shared.add.u32` 原地加一
        //   （无返回值，省一次地址加法）。MACA 汇编器不认 PTX；这里等价写成
        //   `atomicAdd(桶指针, 1)`，桶指针的缩放交给编译器。
        auto histogram_radix_msb_one = [&](uint32_t value_bits) {
            atomicAdd(hist_base + (topk_select_common::distort(value_bits) >> 24), 1u);
        };
        auto histogram_radix_msb = [&]<uint32_t N>(const uint32_t (&vals)[N], uint32_t m_end) {
            CUTE_UNROLL
            for (uint32_t i = 0; i < N; i++) {
                if (i == m_end) break;
                histogram_radix_msb_one(vals[i]);
            }
        };
        struct PivotAndQuota {
            uint32_t pivot_value_bits;     
            uint32_t start_pos_in_collector;            
            uint32_t eq_quota;              
            uint32_t cnt_nan;
        };

        auto compute_pivot_and_quota = [&](uint32_t topk, const auto &for_each_value) -> PivotAndQuota {
            if (warp_idx == 0) {
                find_pivot_in_histogram(topk, smem.reconstruct_bucket_counter[0]);
            }
            __syncthreads();
            // `pivot_prefix` tracks the pivot's leading bits in the *undistorted* (raw) domain.
            // Inside a round every element that survived the previous round must has the pivot's sign, so the distortion rule (to xor what) is the same.
            // So we can optimize scanning & picking & distorting into "comparing the raw leading bits" and "xor-ing `neg_mask`"
            uint32_t pivot_prefix_dist = smem.reconstruct_pivot_bucket;    // distorted top 8 bits
            bool pivot_negative = pivot_prefix_dist < 0x80u;               // distorted >= 0x80 <=> positive
            uint32_t neg_mask = pivot_negative ? 0xFFu : 0u;
            uint32_t pivot_prefix = pivot_prefix_dist ^ (pivot_negative ? 0xFFu : 0x80u);
            uint32_t num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;
            bool should_select_whole_bucket = should_select_whole_bucket_shared;      // whether round 1 is already a clean boundary
            uint32_t rounds_done = 1;

            // Early exit: once topk lands exactly on a bucket's lower edge, the pivot is that bucket's floor.
            CUTE_UNROLL
            for (uint32_t r = 1; r < 4; ++r) {
                if (should_select_whole_bucket) break;
                uint32_t *dst = smem.reconstruct_bucket_counter[r & 1];
                uint32_t *clr = smem.reconstruct_bucket_counter[(r & 1) ^ 1];
                for_each_value([&](uint32_t value_bits) {
                    // Elements whose prefix differs from the pivot still add, but to this warp's own sink slot. This let PTXAS generate ATOMS.INC instructions which is faster
                    bool match = (value_bits >> (32 - 8 * r)) == pivot_prefix;
                    // [MACA] 原为 `bfe.u32`（一条指令取出下一个原始字节）+ `mad.lo.u32` /
                    //   `red.shared.add.u32`。MACA 汇编器不认 PTX；等价写成移位掩码 + atomicAdd。
                    uint32_t bucket_raw = (value_bits >> (24 - 8 * r)) & 0xFFu;
                    uint32_t bucket = match ? (bucket_raw ^ neg_mask) : NUM_RECONSTRUCT_BUCKETS;
                    atomicAdd(dst + bucket, 1u);
                });
                constexpr uint32_t NUM_BUCKET_CLEAR_128b = NUM_RECONSTRUCT_BUCKETS / 4;
                static_assert(NUM_THREADS >= NUM_BUCKET_CLEAR_128b);
                if (threadIdx.x < NUM_BUCKET_CLEAR_128b) {
                    // [MACA] 原用 __int128_t 清零（16 字节）。MACA 无 __int128，改 uint4。
                    reinterpret_cast<uint4*>(clr)[threadIdx.x] = make_uint4(0, 0, 0, 0);
                }
                __syncthreads();
                if (warp_idx == 0) {
                    find_pivot_in_histogram(num_elem_should_select_in_pivot_bucket, dst);
                }
                __syncthreads();
                // The histogram is keyed by distorted bytes, so undo the sign map on the bucket id.
                pivot_prefix = (pivot_prefix << 8) | (smem.reconstruct_pivot_bucket ^ neg_mask);
                num_elem_should_select_in_pivot_bucket = smem.reconstruct_num_should_select;
                should_select_whole_bucket = should_select_whole_bucket_shared;
                rounds_done++;
            }
            // Take the lowest element in the selected bucket
            // Be careful that `pivot` may be negative, and we may have to fill the lower bits with 1
            uint32_t pivot_low_shift = (4 - rounds_done) * 8;
            uint32_t pivot_value_bits = pivot_prefix << pivot_low_shift;
            if (pivot_negative) {
                pivot_value_bits |= (1u << pivot_low_shift) - 1u;
            }
            float pivot_value = __uint_as_float(pivot_value_bits);

            // set.xx.f32.f32 yields 1.0/0.0, so the counts accumulate on the FP pipe instead of the
            // (busiest) integer pipe; the counts stay exact because FP32 can represent every integer within 0 ~ 2**23
            // [MACA] 原为三条 `set.gt/.eq/.nan.f32.f32` + `add.f32` 的内联 PTX。MACA 汇编器
            //   不认 PTX；这里等价写成三目表达式——语义逐条对齐：
            //     * set.gt  = 有序的 (v > pivot)，NaN 参与时为假
            //     * set.eq  = 有序的 (v == pivot)，NaN 参与时为假
            //     * set.nan = 任一操作数为 NaN（此处两侧同为 v，即 isnan(v)）
            //   浮点累加器保留（计数精确：0 ~ 2^23 内的整数 fp32 可精确表示），
            //   比较结果由编译器生成谓词 + 选择，仍是"1.0/0.0 加到 fp32 累加器"的形状。
            float gt_accum = 0.0f, eq_accum = 0.0f, nan_accum = 0.0f;
            for_each_value([&](uint32_t value_bits) {
                float v = __uint_as_float(value_bits);
                gt_accum  += (v >  pivot_value) ? 1.0f : 0.0f;
                eq_accum  += (v == pivot_value) ? 1.0f : 0.0f;
                nan_accum += (v != v)           ? 1.0f : 0.0f;
            });
            uint32_t cnt_gt = (uint32_t)gt_accum;
            uint32_t cnt_eq = (uint32_t)eq_accum;
            // [MACA] 上游此处为 `set.nan.f32.f32 nan_flag, nan_accum, nan_accum`：
            //   它在循环里用 `min.NaN` 折叠、循环外一次 `set.nan` 判定，所以
            //   `nan_accum` 只需是一个"沾到过 NaN"的累加器。MACA 无这两条指令，
            //   上面按逐元素 `(v != v)` 累加，语义等价，这里同样收敛成 0/1 标志
            //   （消费方一律按 `cnt_nan != 0` 使用）。
            uint32_t cnt_nan = nan_accum > 0.0f ? 1u : 0u;

            static_assert(NUM_WARPS <= NUM_RECONSTRUCT_BUCKETS);
            uint32_t *eq_pass_warp_cnt = smem.reconstruct_bucket_counter[0];
            auto eqgt = Base::compute_equal_quota_and_prefix(cnt_gt, cnt_eq, topk, warp_idx, lane_idx, eq_pass_warp_cnt);
            return {pivot_value_bits, eqgt.start_pos_in_collector, eqgt.eq_quota, cnt_nan};
        };

        auto reconstruct = [&](uint32_t cur_extra_len) -> uint32_t {
            uint32_t tid = threadIdx.x;
            Base::clear_reconstruct_histograms(smem, tid);

            uint32_t padded_extra_len = (cur_extra_len + 1u) & ~1u;
            if (warp_idx == 0 && (cur_extra_len & 1u)) {
                smem.incoming_topk_pairs[cur_extra_len] = PLACEHOLDER_PAIR;
            }
            __syncthreads();

            uint32_t num_128b = (MAX_TOPK + padded_extra_len) / 2;
            uint32_t num_128b_per_thread = ((num_128b + NUM_THREADS - 1) / NUM_THREADS) | 1u;   // force odd => bank-conflict-free smem gathers
            uint32_t b128_base = tid * num_128b_per_thread;
            uint32_t num_my_128b = b128_base < num_128b ? min(num_128b_per_thread, num_128b - b128_base) : 0u;

            // [MACA] 原返回 shared 地址整数（`cute::cast_smem_ptr_to_uint`），调用方再转回
            //   指针；改成直接返回真指针（common_parts.cuh 的统一做法）。
            auto b128_to_pair_addr = [&](uint32_t u) -> const uint64_t * {
                // The first MAX_TOPK/2 b128 words live in the current candidate buffer, the rest in the incoming region
                return u < MAX_TOPK / 2
                     ? smem.surviving_topk_pairs[survivor_buf_idx] + 2 * u
                     : smem.incoming_topk_pairs + (2 * u - MAX_TOPK);
            };

            // One b128 word is 16 bytes = 2 pairs {index0, value0, index1, value1};
            constexpr uint32_t NUM_128b_MAX = (MAX_TOPK + RECONSTRUCT_THRESHOLD + NUM_ELEMS_PER_ROUND) / 2;
            constexpr uint32_t NUM_128b_PER_THREAD = ((NUM_128b_MAX + NUM_THREADS - 1) / NUM_THREADS) | 1u;   // | 1u since we may | 1u when generating `num_128b_per_thread`
            auto for_each_value = [&](const auto &fn) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_128b_PER_THREAD; i++) {
                    if (i == num_my_128b) break;
                    uint32_t pair2[4];
                    topk_select_common::ld_shared<4>(pair2, (const uint32_t*)b128_to_pair_addr(b128_base + i));
                    fn(pair2[1]);
                    fn(pair2[3]);
                }
            };

            for_each_value(histogram_radix_msb_one);
            __syncthreads();

            uint32_t topk = args.topk;
            auto [pivot_value_bits, start_pos_in_collector, eq_quota, cnt_nan] = compute_pivot_and_quota(topk, for_each_value);
            // Every NaN the main loop collected is part of the buffer this census just walked (the hit test
            // is `.gtu`, so NaN always becomes an incomer); the CTA-wide OR happens at the end.
            have_nan |= cnt_nan != 0;

            {
                uint64_t *out = smem.surviving_topk_pairs[survivor_buf_idx ^ 1] + start_pos_in_collector;
                float pivot_value = __uint_as_float(pivot_value_bits);
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_128b_PER_THREAD; i++) {
                    if (i == num_my_128b) break;
                    uint32_t pair2[4];
                    topk_select_common::ld_shared<4>(pair2, (const uint32_t*)b128_to_pair_addr(b128_base + i));
                    float v0 = __uint_as_float(pair2[1]);
                    float v1 = __uint_as_float(pair2[3]);
                    bool t0 = v0 == pivot_value && eq_quota != 0;
                    eq_quota -= t0;
                    if (v0 > pivot_value || t0) { append_pair_at(out, pair2[0], pair2[1]); }
                    bool t1 = v1 == pivot_value && eq_quota != 0;
                    eq_quota -= t1;
                    if (v1 > pivot_value || t1) { append_pair_at(out, pair2[2], pair2[3]); }
                }
            }

            survivor_buf_idx ^= 1;
            return pivot_value_bits;
        };

        // [MACA] 原为 TMA 发射侧：由 `cute::elect_one_sync()` 选出一条 lane，用
        //   `issue_tma_loads_for_round` 把 init 各轮与 TMA_PREFETCH_DEPTH 个 main 轮
        //   一次性发出去，消费侧逐轮等 `init_full_bar` / `tma_load_full_bar`。
        //   MACA 无 TMA / mbarrier / 跨轮预取，且置换状态机（next_tma_permuted_segment /
        //   tma_permuted_segment_stride）原先只有那条 elect lane 有效。装载改为
        //   全线程协作的 ldg（`Base::load_round_for_round`），轮次之间无依赖，
        //   该状态机与预取深度整体删除。
        (void)NUM_ISSUE_WARPS;
        (void)TMA_PREFETCH_DEPTH;

        { // INIT
            uint32_t num_elems_in_init_window_padded = num_local_tail_elems_padded + min(num_perm_segs, (uint32_t)NUM_PERM_SEGS_IN_INIT_WINDOW) * NUM_ELEMS_PER_SEG;
            uint32_t num_128b_in_init_window_padded = num_elems_in_init_window_padded / NUM_ELEMS_PER_128b;
            uint32_t cnt_floor = num_128b_in_init_window_padded / NUM_THREADS;
            uint32_t cnt_rem = num_128b_in_init_window_padded % NUM_THREADS;
            uint32_t my_elem_start_idx = (threadIdx.x * cnt_floor + min(threadIdx.x, cnt_rem)) * NUM_ELEMS_PER_128b;
            uint32_t num_my_elems = NUM_UINT32_PER_128b * cnt_floor + (threadIdx.x < cnt_rem ? NUM_UINT32_PER_128b : 0u);
            uint32_t num_local_tail_elems = end_vocab_idx - num_perm_elems;

            // [MACA] 装载 init window（尾段 + 头若干置换段）。全线程协作 ldg，
            //   init 各轮写 incoming_topk_pairs 的不同区段、互不覆盖，故一次
            //   __syncthreads() 即完成发布，随后才开始消费。
            //   `local_start_seg_idx` 取 0：本变体非 cluster，本地段号即全局段号。
            if (num_local_tail_elems_padded != 0) {
                Base::template load_round_for_round<true, true>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    0, 0u, warp_idx, lane_idx);
            } else {
                Base::template load_round_for_round<true, false>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    0, 0u, warp_idx, lane_idx);
            }
            CUTE_UNROLL
            for (uint32_t i = 1; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i >= num_init_rounds) break;
                Base::template load_round_for_round<true, false>(
                    smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                    i, 0u, warp_idx, lane_idx);
            }
            __syncthreads();

            if (num_perm_segs != 0 && num_local_tail_elems_padded != 0) {
                Base::fill_padded_tail_segments(init_buf, end_vocab_idx - num_perm_elems);
                __syncthreads();
            }

            CUTE_UNROLL
            for (uint32_t i = 0; i < NUM_INIT_ROUNDS_MAX; i++) {
                if (i == num_init_rounds) break;
                // [MACA] 原为 `smem.init_full_bar[i].wait(0)`（等该轮 TMA 事务完成）。
                //   装载已改为前面那次全线程协作 ldg + 一次 __syncthreads()，
                //   所有 init 轮在此处都已可见，无需逐轮等待。
                if (num_local_tail_elems % NUM_ELEMS_PER_SEG != 0 && i == num_local_tail_elems / NUM_ELEMS_PER_ROUND) {
                    uint32_t box_end = ku::ceil_div(num_local_tail_elems, (uint32_t)NUM_ELEMS_PER_SEG) * NUM_ELEMS_PER_SEG;
                    for (uint32_t e = num_local_tail_elems + threadIdx.x; e < box_end; e += NUM_THREADS) {
                        init_buf[Base::sw_elem(e)] = __uint_as_float(PLACEHOLDER);
                    }
                    __syncthreads();
                }
                uint32_t my_values[4 * NUM_128b_PER_THREAD_PER_ROUND];
                uint32_t num_my_values = 0;
                CUTE_UNROLL
                for (uint32_t k = 0; k < NUM_128b_PER_THREAD_PER_ROUND; ++k) {
                    uint32_t u = threadIdx.x + (i * NUM_128b_PER_THREAD_PER_ROUND + k) * NUM_THREADS;
                    if (u < num_128b_in_init_window_padded) {
                        topk_select_common::ld_shared<4>(my_values + k * 4, (const uint32_t*)(init_buf + Base::sw_b128(u) * NUM_ELEMS_PER_128b));
                        num_my_values = 4 * (k + 1);
                    }
                }
                histogram_radix_msb(my_values, num_my_values);
            }

            uint32_t init_values[NUM_UINT32_IN_INIT_WINDOW_PER_THREAD];
            CUTE_UNROLL
            for (uint32_t j = 0; j < NUM_128b_INIT_PER_THREAD; ++j) {
                if (j * NUM_UINT32_PER_128b == num_my_elems) break;
                uint32_t u0 = my_elem_start_idx / NUM_ELEMS_PER_128b + j;
                topk_select_common::ld_shared<4>(init_values + j * NUM_UINT32_PER_128b, (const uint32_t*)(init_buf + Base::sw_b128(u0) * NUM_ELEMS_PER_128b));
            }

            __syncthreads();    // publish the round-1 histogram; also the last read of init_buf

            static_assert(NUM_ELEMS_IN_INIT_WINDOW <= 0xFFFF);
            
            auto for_each_init_value = [&](const auto &fn) {
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i++) {
                    if (i == num_my_elems) break;
                    fn(init_values[i]);
                }
            };
            auto [pivot_value_bits, start_pos_in_collector, eq_quota, cnt_nan] = compute_pivot_and_quota(args.topk, for_each_init_value);
            // The init census walks the thread's whole slice of the window, i.e. every element of the init
            // window exactly once, so NaNs inside the window are caught here.
            have_nan |= cnt_nan != 0;

            {
                uint64_t *out = smem.surviving_topk_pairs[0] + start_pos_in_collector;
                float pivot_value = __uint_as_float(pivot_value_bits);
                uint32_t s0 = (max(my_elem_start_idx, num_local_tail_elems_padded) - num_local_tail_elems_padded) / NUM_ELEMS_PER_SEG;
                uint32_t perm_state = Base::get_permuted_seg_idx(s0, perm_len, perm_mul);
                uint32_t index_base = 0;
                CUTE_UNROLL
                for (uint32_t i = 0; i < NUM_UINT32_IN_INIT_WINDOW_PER_THREAD; i += 2) {
                    if (i == num_my_elems) break;
                    if (i % NUM_ELEMS_PER_128b == 0) {
                        uint32_t g_u = my_elem_start_idx + i;      // fp32: word index == element index
                        if (g_u < num_local_tail_elems_padded) {
                            index_base = num_perm_elems + g_u;
                        } else {
                            index_base = perm_state * NUM_ELEMS_PER_SEG + g_u % NUM_ELEMS_PER_SEG;
                            if (g_u % NUM_ELEMS_PER_SEG == NUM_ELEMS_PER_SEG - NUM_ELEMS_PER_128b) {
                                Base::advance_perm_state(perm_state, perm_mul, perm_len);
                            }
                        }
                    }
                    // [MACA] 原为「把整数索引伪装成 fp32 次正规数再用浮点加法求
                    //   index = base + element offset」——两个次正规数相加是精确的，
                    //   结果与整数加法逐位相同，但那是上游为躲开整数管线竞争而依赖的
                    //   浮点语义。C++ 里直接用整数：语义等价，且不依赖次正规数舍入前提。
                    uint32_t index0 = index_base + (i % NUM_ELEMS_PER_128b);
                    uint32_t index1 = index0 + 1;
                    float v0 = __uint_as_float(init_values[i]);
                    float v1 = __uint_as_float(init_values[i + 1]);
                    bool t0 = v0 == pivot_value && eq_quota != 0;
                    eq_quota -= t0;
                    if (v0 > pivot_value || t0) { append_pair_at(out, index0, init_values[i]); }
                    // the quota is re-tested after element 0's decrement
                    bool t1 = v1 == pivot_value && eq_quota != 0;
                    eq_quota -= t1;
                    if (v1 > pivot_value || t1) { append_pair_at(out, index1, init_values[i + 1]); }
                }
            }

            threshold_bits = pivot_value_bits;
            __syncthreads();
        }

        uint32_t linear_segment_start =
            num_init_rounds * NUM_SEGS_PER_ROUND
            - NUM_TAIL_SEGS
            + warp_idx;
        uint32_t current_permuted_segment = Base::get_permuted_seg_idx(linear_segment_start, perm_len, perm_mul);
        uint32_t permuted_segment_stride_per_round = NUM_SEGS_PER_ROUND % perm_len * perm_mul % perm_len;

        uint32_t num_main_rounds = num_rounds - num_init_rounds;
        // [MACA] 原为跨轮维护的 TMA 缓冲下标与相位（tma_buf_idx / tma_buf_phase）。
        //   单缓冲 + 无 TMA，这两项与 NUM_TMA_LOAD_BUFS 的翻转一起删除。
        for (uint32_t main_round_idx = 0; main_round_idx < num_main_rounds; ++main_round_idx) {
            // This thread's contiguous chunk starts at logical_elem_offset in this round.
            uint32_t logical_elem_offset = threadIdx.x * NUM_ELEMS_PER_THREAD_PER_ROUND;
            // Offset inside the 512-element segment; used later to rebuild original indices.
            uint32_t offset_in_segment = logical_elem_offset % NUM_ELEMS_PER_SEG;
            // TMA stores the round data in a swizzled layout. swizzle_mask describes how the
            // logical element offset is mapped to the actual shared-memory offset.
            uint32_t swizzle_mask = Base::sw_msk(logical_elem_offset);
            uint32_t chunk_swizzle_mask = swizzle_mask & ELEMS_PER_THREAD_PER_ROUND_MASK;
            uint32_t smem_read_offset = logical_elem_offset ^ (swizzle_mask & ~ELEMS_PER_THREAD_PER_ROUND_MASK);

            // [MACA] 原为「elect 一条 lane 预取 main_round_idx + TMA_PREFETCH_DEPTH 轮，
            //   本轮消费等 tma_load_full_bar」。单缓冲 + 无 TMA：改成
            //   「全线程协作装载本轮 → __syncthreads() → 全线程消费本轮」。
            //   轮索引用绝对编号（init 轮在前），置换线性位置的推导才与消费端一致。
            Base::template load_round_for_round<false, false>(
                smem, args, batch_idx, end_vocab_idx, num_perm_segs,
                num_init_rounds + main_round_idx, 0u, warp_idx, lane_idx);
            __syncthreads();

            bool is_warp_active = true;
            if constexpr (HAS_PARTIAL_ROUNDS) {
                is_warp_active = (num_init_rounds + main_round_idx) * NUM_SEGS_PER_ROUND + warp_idx < NUM_TAIL_SEGS + num_perm_segs;
            }

            const ValueT *buf = smem.tma_load_buf;
            // One bit per element of this thread's slice, in element order.
            // Hits are rare, so appending through the set bits of this mask is faster than testing + storing every element again.
            static_assert(NUM_ELEMS_PER_THREAD_PER_ROUND <= 32);        // hit_mask is a uint32_t
            uint32_t hit_mask = 0;
            if (is_warp_active) {
                ValueT values[NUM_ELEMS_PER_THREAD_PER_ROUND];
                Base::template load_swizzled_slice<NUM_128b_PER_THREAD_PER_ROUND>(values, buf, smem_read_offset, chunk_swizzle_mask);

                float threshold = __uint_as_float(threshold_bits);
                // [MACA] 原为一条内联 PTX：`setp.gtu.f32`（greater-than-or-**unordered**，
                //   任一操作数为 NaN 即真）+ 谓词化的 `or.b32` 置位。
                //   `.gtu` 就是"NaN 一律接受"：命中的 NaN 交给后面 have_nan 检查兜底。
                //   MACA 汇编器不认 PTX；等价写成 C++ —— 由于 threshold 永远不是 NaN，
                //   "无序"只剩 isnan(value) 一支，但两支都保留以免语义依赖该前提。
                CUTE_UNROLL
                for (uint32_t j = 0; j < NUM_ELEMS_PER_THREAD_PER_ROUND; ++j) {
                    float v = (float)values[j];
                    bool hit = (v > threshold) || (v != v) || (threshold != threshold);
                    if (hit) { hit_mask |= (1u << j); }
                }
            }
            uint32_t num_new_incomers = __popc(hit_mask);

            uint32_t warp_total_hits = __reduce_add_sync(0xFFFFFFFF, num_new_incomers);
            if (lane_idx == 0) {
                smem.warp_cnt[warp_idx] = warp_total_hits;
            }
            __syncthreads();

            uint32_t seg_elem_base = current_permuted_segment * NUM_ELEMS_PER_SEG + offset_in_segment;
            // Element e of this thread's slice lives at smem_read_offset + (e ^ chunk_swizzle_mask)
            // [MACA] 原先把该地址经 cast_smem_ptr_to_uint 变成整数再做 ld.shared/st.shared；
            //   改成真指针 + 普通 shared 访存（common_parts.cuh 的统一做法）。
            const ValueT *elem_base = buf + smem_read_offset;

            // Start the first hit's smem load before the count exchange below, so that its latency
            // (and the barrier wait) overlaps with them instead of delaying the first store.
            uint32_t first_hit_e = hit_mask != 0 ? __ffs(hit_mask) - 1u : 0u;
            uint32_t first_hit_val = 0;
            if (is_warp_active && warp_total_hits != 0) {
                first_hit_val = *(const uint32_t*)(elem_base + (first_hit_e ^ chunk_swizzle_mask));
            }

            static_assert(NUM_WARPS <= 32);
            uint32_t stored_warp_hits = lane_idx < NUM_WARPS ? smem.warp_cnt[lane_idx] : 0u;
            uint32_t num_total_hits_in_this_round = __reduce_add_sync(0xFFFFFFFF, stored_warp_hits);

            uint32_t dst_slot =
                num_incomers +
                __reduce_add_sync(0xFFFFFFFF, lane_idx < warp_idx ? stored_warp_hits : 0u) +
                warp_level_exclusive_prefix_sum(num_new_incomers, lane_idx);

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
                    uint32_t val_word = *(const uint32_t*)(elem_base + (e ^ chunk_swizzle_mask));
                    *dst_ptr++ = ((uint64_t)val_word << 32) | (uint64_t)(seg_elem_base + e);
                }
            }
            Base::advance_perm_state(current_permuted_segment, permuted_segment_stride_per_round, perm_len);

            num_incomers += num_total_hits_in_this_round;
            // No warp may run ahead into the next round: the load at the top of the next
            // iteration reuses the single smem buffer this round is reading, and warp_cnt
            // must not be overwritten while others read it.
            __syncthreads();

            if (num_incomers >= RECONSTRUCT_THRESHOLD) {
                threshold_bits = reconstruct(num_incomers);
                num_incomers = 0;
            }
        }

        if (num_incomers > 0) {
            reconstruct(num_incomers);
            num_incomers = 0;
        }

        have_nan = __syncthreads_or(have_nan) != 0;
        if (have_nan) {
            Base::take_action_when_have_nan(args, batch_idx);
            return;
        }

        Base::template stage_output_and_epilogue<Config::sorted_index>(
            smem, *reinterpret_cast<typename EpilogueT::BlockRadixSortTempStorageT*>(smem.incoming_topk_pairs), args,
            batch_idx, end_vocab_idx, survivor_buf_idx, warp_idx, lane_idx);

    }

};


// [MACA] 原为 `__grid_constant__ const TopkSelectArgs args, __grid_constant__ const typename
//   Kernel::TmaParams tma_params`。MACA 无 `__grid_constant__`（该限定符是 sm70+ 的
//   kernarg 常量化提示，去掉后按值传入的 kernarg 语义不变）；tensor map 参数随 TMA 一起删除。
template<typename Kernel>
__launch_bounds__(Kernel::NUM_THREADS, Kernel::TARGET_OCCUPANCY, 1)
__global__ void topk_kernel(const TopkSelectArgs args) {
    Kernel::topk_select_kernel_devfunc(args);
}

template<typename Config>
void run_topk_select_kernel(const TopkSelectArgs &args) {
    KU_ASSERT(args.sorted_value == Config::sorted_value, "Dispatch failure");
    KU_ASSERT(args.sorted_index == Config::sorted_index, "Dispatch failure");
    KU_ASSERT(args.return_value == Config::return_value, "Dispatch failure");
    static_assert(cute::is_same_v<typename Config::ValueT, float>);
    KU_ASSERT(args.vocab_size < MAX_VOCAB_SIZE, "`vocab_size` is too big");

    using Kernel = TopkSelectKernelFP32<Config>;
    KU_ASSERT(args.topk <= Kernel::MAX_TOPK, "topk is too large. Maximum allowed: %d\n", Kernel::MAX_TOPK);
    static_assert(INPUT_STRIDE_ALIGNMENT_REQUIREMENT % 16 == 0);

    auto kernel = topk_kernel<Kernel>;
    constexpr size_t smem_size = sizeof(typename Kernel::SharedMemoryPlanFP32);
    KU_ASSERT(smem_size * Kernel::TARGET_OCCUPANCY <= args.shared_memory_size_per_sm);
    KU_CUDA_CHECK(cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));

    KU_ASSERT(args.stride_input_batch % 4 == 0, "stride_input_batch must be 16B-aligned");
    // [MACA] 原在此构造 TmaParams（CUtensorMap）；装载已改为 ldg，无 tensor map。

    ku::launch_kernel(ku::KernelLaunchConfig {
        dim3(args.batch_size),
        dim3(Kernel::NUM_THREADS),
        smem_size,
        args.stream
    }, kernel, args);
}

}   // topk_select_fp32
