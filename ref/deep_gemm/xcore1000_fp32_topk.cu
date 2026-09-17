// Copyright (c) 2026 MetaX Integrated Circuits (Shanghai) Co., Ltd. All rights reserved.

#include "xcore1000_fp32_topk.h"

// The two preprocessor facts the kernel layer actually needs from the original
// `../utils/exception.hpp`: an unchecked runtime-status tripwire, and the
// kernel-name/user-string list builder pybind used.  Inlining them here keeps
// the extraction torch-free and header-free without changing any code below.
#include <cstdio>
#include <cstdlib>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

void deep_gemm_abort(const char *file, int line, const char *what)
{
  std::fprintf(stderr, "DG runtime error (%s:%d): %s\n", file, line, what);
  std::abort();
}

#define DG_CUDA_RUNTIME_CHECK(cmd)                                                       \
  do {                                                                                   \
    const cudaError_t _dg_status = (cmd);                                                \
    if (_dg_status != cudaSuccess)                                                       \
      deep_gemm_abort(__FILE__, __LINE__, cudaGetErrorString(_dg_status));               \
  } while (0)

// `at::cuda::getDeviceProperties(device)->multiProcessorCount` without the
// torch cache.  Used both by the policy decision and, in the chunks launcher,
// by `select_chunk_count`.
static int deep_gemm_device_sm_count()
{
  int num_sms = 0;
  DG_CUDA_RUNTIME_CHECK(
      cudaDeviceGetAttribute(&num_sms, cudaDevAttrMultiProcessorCount, 0));
  return num_sms;
}

namespace deep_gemm::indexer {

namespace detail {

constexpr int kHistogramPadding = 1;
constexpr float kNegativeInfinity = -std::numeric_limits<float>::infinity();

// Single/Chunks parameters
constexpr int kThreads = 1024;
constexpr int kMaxTopK = 2048;
constexpr int kCoarseBins = 1024;
constexpr int kFineBins = 256;
constexpr int kCandidateCapacity = 4096;
constexpr int kWarpSize = 64;
constexpr size_t kTopKCandidateSmemBytes = 2 * kCandidateCapacity * sizeof(int32_t);

// Coarse 12 policy parameters
constexpr int kCoarse12Threads = 640;
constexpr int kCoarse12CoarseBits = 12;
constexpr int kCoarse12CoarseBins = 1 << kCoarse12CoarseBits;
constexpr size_t kCoarse12SmemBytes = 16 * 1024;

struct TransformInfo {
  const int32_t *page_table_row;
  bool region_pack;
  int32_t q_position;
};

template <int ChunkCount>
struct TopKChunksWorkspace {
  TransformInfo trans_info;
  int row_begin;
  int row_length;
  int coarse[ChunkCount][kCoarseBins];
  int fine[ChunkCount][kFineBins];
  int guaranteed_bases[ChunkCount];
  int boundary_bases[ChunkCount];
  int threshold;
  int guaranteed_count;
  int boundary_take;
  int candidate_count;
  int arrival;
  alignas(16) int candidate_indices[kCandidateCapacity];
};

struct TopKChunksShared {
  int histogram[kFineBins + kHistogramPadding];
  int guaranteed_counter;
  int boundary_counter;
  int candidate_counter;
  int threshold;
  int last_remain;
  int is_last;
};

struct TopKParams {
  const float *scores;
  const int32_t *seq_lens;
  const int32_t *seq_starts;
  const int32_t *page_table;
  const int32_t *cu_seqlens_row;
  const int32_t *q_positions;
  int32_t *output;
  float *output_values;
  int64_t n_rows;
  int64_t n_cols;
  int64_t page_table_stride;
  int32_t page_batch_size;
  int32_t next_n;
  int32_t top_k;
  bool is_decode;
  bool lookup_page_row;
  bool region_pack;
};

__device__ __forceinline__ float
gather_topk_value(const float *__restrict__ input_row, int index)
{
  return index < 0 ? kNegativeInfinity : input_row[index];
}

__device__ __forceinline__ void gather_topk_values(
    const float *__restrict__ input_row, const int32_t *__restrict__ indices,
    float *__restrict__ values, int top_k)
{
  if (values == nullptr) return;

  const int tid = threadIdx.x;
  const uintptr_t address = reinterpret_cast<uintptr_t>(values);
  const int prefix_unclamped = static_cast<int>(
      (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
  const int prefix =
      (address & (alignof(float4) - 1)) == 0 ? 0 : min(top_k, prefix_unclamped);
  const int vector_length = (top_k - prefix) / 4;
  const int tail = prefix + vector_length * 4;

  for (int pos = tid; pos < prefix; pos += blockDim.x)
    values[pos] = gather_topk_value(input_row, indices[pos]);

  auto *vector_values = reinterpret_cast<float4 *>(values + prefix);
  for (int vec = tid; vec < vector_length; vec += blockDim.x) {
    const int pos = prefix + vec * 4;
    const int4 selected = __ldg(reinterpret_cast<const int4 *>(indices + pos));
    vector_values[vec] = make_float4(
        gather_topk_value(input_row, selected.x),
        gather_topk_value(input_row, selected.y),
        gather_topk_value(input_row, selected.z),
        gather_topk_value(input_row, selected.w));
  }

  for (int pos = tail + tid; pos < top_k; pos += blockDim.x)
    values[pos] = gather_topk_value(input_row, indices[pos]);
}

__device__ __forceinline__ uint32_t refine_bin(float value)
{
  const uint32_t bits = __float_as_uint(value);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ int coarse_bin(float value)
{
  const __half half_value = __float2half_rn(value);
  const uint16_t bits = __half_as_ushort(half_value);
  const uint16_t key = (bits & 0x8000u) ? static_cast<uint16_t>(~bits)
                                        : static_cast<uint16_t>(bits | 0x8000u);
  return static_cast<int>(key >> 6);
}

__device__ __forceinline__ void
coarse_bins_pair(float first, float second, int& first_bin, int& second_bin)
{
  const __half2 packed = __floats2half2_rn(first, second);
  const uint16_t first_bits = __half_as_ushort(__low2half(packed));
  const uint16_t second_bits = __half_as_ushort(__high2half(packed));
  const uint16_t first_key = (first_bits & 0x8000u)
                               ? static_cast<uint16_t>(~first_bits)
                               : static_cast<uint16_t>(first_bits | 0x8000u);
  const uint16_t second_key = (second_bits & 0x8000u)
                                ? static_cast<uint16_t>(~second_bits)
                                : static_cast<uint16_t>(second_bits | 0x8000u);
  first_bin = static_cast<int>(first_key >> 6);
  second_bin = static_cast<int>(second_key >> 6);
}

template <int Bins, int Padding = kHistogramPadding>
__device__ __forceinline__ void warp_cumsum_histogram(int (&histogram)[Bins + Padding])
{
  static_assert(Bins % kWarpSize == 0);
  static_assert(Bins == kFineBins || Bins == kCoarseBins);
  constexpr uint64_t kWarpMask = 0xffffffffffffffffULL;
  const int tid = threadIdx.x;

  if constexpr (Bins == kFineBins) {
    constexpr int kItemsPerLane = Bins / kWarpSize;
    if (tid < kWarpSize) {
      const int lane = tid;
      int values[kItemsPerLane];
#pragma unroll
      for (int item = 0; item < kItemsPerLane; ++item) {
        values[item] = histogram[lane * kItemsPerLane + item];
      }
#pragma unroll
      for (int item = kItemsPerLane - 2; item >= 0; --item) {
        values[item] += values[item + 1];
      }

      const int lane_total = values[0];
      int warp_suffix = lane_total;
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_down_sync(kWarpMask, warp_suffix, offset, kWarpSize);
        if (lane + offset < kWarpSize) warp_suffix += other;
      }
      const int lane_offset = warp_suffix - lane_total;

#pragma unroll
      for (int item = 0; item < kItemsPerLane; ++item) {
        histogram[lane * kItemsPerLane + item] = values[item] + lane_offset;
      }
    }
  }
  else {
    constexpr int kWarps = Bins / kWarpSize;
    static_assert(kWarps <= kWarpSize);
    __shared__ int warp_totals[kWarps];

    const int lane = tid % kWarpSize;
    const int warp = tid / kWarpSize;
    int warp_suffix = histogram[tid];
#pragma unroll
    for (int offset = 1; offset < kWarpSize; offset <<= 1) {
      const int other = __shfl_down_sync(kWarpMask, warp_suffix, offset, kWarpSize);
      if (lane + offset < kWarpSize) warp_suffix += other;
    }
    if (lane == 0) warp_totals[warp] = warp_suffix;
    __syncthreads();

    if (warp == 0) {
      const int warp_total = lane < kWarps ? warp_totals[lane] : 0;
      int block_suffix = warp_total;
#pragma unroll
      for (int offset = 1; offset < kWarpSize; offset <<= 1) {
        const int other = __shfl_down_sync(kWarpMask, block_suffix, offset, kWarpSize);
        if (lane + offset < kWarpSize) block_suffix += other;
      }
      if (lane < kWarps) warp_totals[lane] = block_suffix - warp_total;
    }
    __syncthreads();

    histogram[tid] = warp_suffix + warp_totals[warp];
  }
  __syncthreads();
}

__device__ __forceinline__ int32_t
transform_region_pack(const int32_t *block_table_row, int logical_region, int q_position)
{
  constexpr int kRegionSize = 8;
  constexpr int kRegionsPerPage = 2;
  constexpr int kValidShift = 24;
  const int table_idx = logical_region / kRegionsPerPage;
  const int region_offset = logical_region - table_idx * kRegionsPerPage;
  const int physical_region =
      __ldg(block_table_row + table_idx) * kRegionsPerPage + region_offset;
  const int valid =
      max(0, min(kRegionSize, q_position + 1 - logical_region * kRegionSize));
  return physical_region | (valid << kValidShift);
}

template <bool Transform>
__device__ __forceinline__ int transform_index(const TransformInfo& info, int idx)
{
  if constexpr (Transform) {
    if (info.region_pack)
      return transform_region_pack(info.page_table_row, idx, info.q_position);
    else
      return __ldg(info.page_table_row + idx);
  }
  else {
    return idx;
  }
}

__device__ __forceinline__ void write_trivial_page_indices(
    const int32_t *__restrict__ page_table_row, int32_t *__restrict__ output, int length,
    int top_k)
{
  const int tid = threadIdx.x;
  const uintptr_t address = reinterpret_cast<uintptr_t>(output);
  const int prefix_unclamped = static_cast<int>(
      (alignof(int4) - (address & (alignof(int4) - 1))) / sizeof(int32_t));
  const int prefix =
      (address & (alignof(int4) - 1)) == 0 ? 0 : min(top_k, prefix_unclamped);
  const int vector_length = (top_k - prefix) / 4;
  const int tail = prefix + vector_length * 4;

  for (int pos = tid; pos < prefix; pos += blockDim.x)
    output[pos] = pos < length ? __ldg(page_table_row + pos) : -1;

  auto *vector_output = reinterpret_cast<int4 *>(output + prefix);
  const int4 neg_one = make_int4(-1, -1, -1, -1);
  for (int vec = tid; vec < vector_length; vec += blockDim.x) {
    const int pos = prefix + vec * 4;
    int4 selected;
    if (pos + 3 < length) {
      selected = make_int4(
          __ldg(page_table_row + pos), __ldg(page_table_row + pos + 1),
          __ldg(page_table_row + pos + 2), __ldg(page_table_row + pos + 3));
    }
    else if (pos >= length) {
      selected = neg_one;
    }
    else {
      selected = make_int4(
          pos < length ? __ldg(page_table_row + pos) : -1,
          pos + 1 < length ? __ldg(page_table_row + pos + 1) : -1,
          pos + 2 < length ? __ldg(page_table_row + pos + 2) : -1,
          pos + 3 < length ? __ldg(page_table_row + pos + 3) : -1);
    }
    __stcg(vector_output + vec, selected);
  }

  for (int pos = tail + tid; pos < top_k; pos += blockDim.x)
    output[pos] = pos < length ? __ldg(page_table_row + pos) : -1;
}

template <bool Transform>
__device__ __forceinline__ void write_trivial_topk_indices(
    const TransformInfo& trans_info, int32_t *__restrict__ output, int length, int top_k)
{
  if constexpr (Transform) {
    if (!trans_info.region_pack) {
      write_trivial_page_indices(trans_info.page_table_row, output, length, top_k);
      return;
    }
  }
  const int tid = threadIdx.x;
  const uintptr_t address = reinterpret_cast<uintptr_t>(output);
  const int prefix_unclamped = static_cast<int>(
      (alignof(int4) - (address & (alignof(int4) - 1))) / sizeof(int32_t));
  const int prefix =
      (address & (alignof(int4) - 1)) == 0 ? 0 : min(top_k, prefix_unclamped);
  const int vector_length = (top_k - prefix) / 4;
  const int tail = prefix + vector_length * 4;

  for (int pos = tid; pos < prefix; pos += blockDim.x)
    output[pos] = pos < length ? transform_index<Transform>(trans_info, pos) : -1;

  auto *vector_output = reinterpret_cast<int4 *>(output + prefix);
  const int4 neg_one = make_int4(-1, -1, -1, -1);
  for (int vec = tid; vec < vector_length; vec += blockDim.x) {
    const int pos = prefix + vec * 4;
    int4 selected;
    if (pos + 3 < length) {
      selected = make_int4(
          transform_index<Transform>(trans_info, pos),
          transform_index<Transform>(trans_info, pos + 1),
          transform_index<Transform>(trans_info, pos + 2),
          transform_index<Transform>(trans_info, pos + 3));
    }
    else if (pos >= length) {
      selected = neg_one;
    }
    else {
      selected = make_int4(
          pos < length ? transform_index<Transform>(trans_info, pos) : -1,
          pos + 1 < length ? transform_index<Transform>(trans_info, pos + 1) : -1,
          pos + 2 < length ? transform_index<Transform>(trans_info, pos + 2) : -1,
          pos + 3 < length ? transform_index<Transform>(trans_info, pos + 3) : -1);
    }
    __stcg(vector_output + vec, selected);
  }

  for (int pos = tail + tid; pos < top_k; pos += blockDim.x)
    output[pos] = pos < length ? transform_index<Transform>(trans_info, pos) : -1;
}

__device__ __forceinline__ void
get_row_begin_length(const TopKParams& params, int row, int& begin, int& length)
{
  const int group = row / params.next_n;
  const int row_in_group = row - group * params.next_n;
  begin = params.seq_starts == nullptr ? 0 : params.seq_starts[group];
  length = params.seq_lens[group] - params.next_n + row_in_group + 1;
}

template <bool Transform>
__device__ __forceinline__ TransformInfo
initialize_row_context(const TopKParams& params, int row, int& begin, int& length)
{
  if (params.is_decode) {
    begin = 0;
    length = __ldg(params.seq_lens + row);
    if constexpr (!Transform) {
      return {nullptr, false, 0};
    }
    else {
      const int32_t *table =
          params.page_table + static_cast<int64_t>(row) * params.page_table_stride;
      return {
          table, params.region_pack,
          params.q_positions == nullptr ? 0 : __ldg(params.q_positions + row)};
    }
  }

  get_row_begin_length(params, row, begin, length);
  if constexpr (!Transform) {
    return {nullptr, false, 0};
  }
  else {
    if (params.lookup_page_row) {
      __shared__ int page_row;
      for (int batch = threadIdx.x; batch < params.page_batch_size; batch += blockDim.x) {
        const int start = __ldg(params.cu_seqlens_row + batch);
        const int end = __ldg(params.cu_seqlens_row + batch + 1);
        if (row >= start && row < end) {
          page_row = batch;
        }
      }
      __syncthreads();
      return {
          params.page_table + static_cast<int64_t>(page_row) * params.page_table_stride,
          params.region_pack,
          params.q_positions == nullptr ? 0 : __ldg(params.q_positions + row)};
    }
    else {
      return {
          params.page_table + static_cast<int64_t>(row) * params.page_table_stride,
          params.region_pack,
          params.q_positions == nullptr ? 0 : __ldg(params.q_positions + row)};
    }
  }
}

template <bool Transform>
__device__ __forceinline__ void topk_coarse12_impl(
    const float *__restrict__ input, int32_t *__restrict__ output, int length,
    int requested_topk, const TransformInfo& trans_info)
{
  constexpr int kRadix = 256;
  constexpr int kFirstShift = 32 - kCoarse12CoarseBits;
  constexpr int kWideGroupSize = 1 << (kCoarse12CoarseBits - 8);
  constexpr int kRefineShift0 = kFirstShift - 8;
  constexpr int kRefineShift1 = kFirstShift - 16;
  constexpr int kFinalRadixBits = kRefineShift1;
  constexpr int kFinalRadixMask = (1 << kFinalRadixBits) - 1;
  constexpr int kCoarse12CandidateCapacity = kCoarse12SmemBytes / (2 * sizeof(int32_t));
  static_assert(kCoarse12CandidateCapacity >= kMaxTopK);

  __shared__ int histogram[kRadix + kHistogramPadding];
  __shared__ int counter;
  __shared__ int threshold_bin_id;
  __shared__ int threshold_exclusive_count;
  __shared__ int num_input[2];
  __shared__ int last_remain;

  extern __shared__ int shared_arena[];
  int *wide_histogram = shared_arena;
  int *candidate_indices = shared_arena;
  const int tid = threadIdx.x;
  const unsigned int u_length = static_cast<unsigned int>(length);

  const auto address = reinterpret_cast<uintptr_t>(input);
  const int prefix_unclamped = static_cast<int>(
      (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
  const int prefix =
      (address & (alignof(float4) - 1)) == 0 ? 0 : min(length, prefix_unclamped);
  const int vector_length = (length - prefix) / 4;
  const int tail = prefix + vector_length * 4;
  const float4 *vector_input = reinterpret_cast<const float4 *>(input + prefix);

  for (int bin = tid; bin < kCoarse12CoarseBins; bin += kCoarse12Threads)
    wide_histogram[bin] = 0;
  __syncthreads();

  for (int index = tid; index < prefix; index += kCoarse12Threads)
    atomicAdd(&wide_histogram[refine_bin(__ldg(input + index)) >> kFirstShift], 1);
  for (int vec = tid; vec < vector_length; vec += kCoarse12Threads) {
    const float4 values = __ldg(vector_input + vec);
    atomicAdd(&wide_histogram[refine_bin(values.x) >> kFirstShift], 1);
    atomicAdd(&wide_histogram[refine_bin(values.y) >> kFirstShift], 1);
    atomicAdd(&wide_histogram[refine_bin(values.z) >> kFirstShift], 1);
    atomicAdd(&wide_histogram[refine_bin(values.w) >> kFirstShift], 1);
  }
  for (int index = tail + tid; index < length; index += kCoarse12Threads)
    atomicAdd(&wide_histogram[refine_bin(__ldg(input + index)) >> kFirstShift], 1);
  __syncthreads();

  if (tid < kRadix) {
    int count = 0;
#pragma unroll
    for (int sub_bin = 0; sub_bin < kWideGroupSize; ++sub_bin)
      count += wide_histogram[tid * kWideGroupSize + sub_bin];
    histogram[tid] = count;
  }
  else if (tid == kRadix) {
    histogram[tid] = 0;
  }
  __syncthreads();

  warp_cumsum_histogram<kRadix>(histogram);
  if (tid < kRadix && histogram[tid] > requested_topk
      && histogram[tid + 1] <= requested_topk) {
    threshold_bin_id = tid;
    threshold_exclusive_count = histogram[tid + 1];
  }
  __syncthreads();

  if (tid == 0) {
    const int high8_bin = threshold_bin_id;
    const int high8_exclusive = threshold_exclusive_count;
    const int remain = requested_topk - high8_exclusive;
    int sub_exclusive = 0;
    for (int sub_bin = kWideGroupSize - 1; sub_bin >= 0; --sub_bin) {
      const int count = wide_histogram[high8_bin * kWideGroupSize + sub_bin];
      if (sub_exclusive + count > remain) {
        threshold_bin_id = high8_bin * kWideGroupSize + sub_bin;
        threshold_exclusive_count = high8_exclusive + sub_exclusive;
        break;
      }
      sub_exclusive += count;
    }
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  const int wide_threshold = threshold_bin_id;
  int topk = requested_topk - threshold_exclusive_count;
  if (topk == 0) {
    for (unsigned int index = tid; index < u_length; index += kCoarse12Threads) {
      if (static_cast<int>(refine_bin(__ldg(input + index)) >> kFirstShift)
          > wide_threshold) {
        const int position = atomicAdd(&counter, 1);
        output[position] = transform_index<Transform>(trans_info, index);
      }
    }
    __syncthreads();
    return;
  }

  if (tid < kRadix + 1) histogram[tid] = 0;
  __syncthreads();

  // Keep this explicitly expanded: on MetaX it is consistently 1.7%-2.2%
  // faster than the equivalent capturing lambda on large-batch shapes.
#define DEEP_GEMM_COLLECT_COARSE12_VALUE(value, index)                                   \
  do {                                                                                   \
    const uint32_t key = refine_bin(value);                                              \
    const int bin = key >> kFirstShift;                                                  \
    if (bin > wide_threshold) {                                                          \
      const int position = atomicAdd(&counter, 1);                                       \
      output[position] = transform_index<Transform>(trans_info, index);                  \
    }                                                                                    \
    else if (bin == wide_threshold) {                                                    \
      const int position = atomicAdd(&num_input[0], 1);                                  \
      if (position < kCoarse12CandidateCapacity) {                                       \
        candidate_indices[position] = index;                                             \
        atomicAdd(&histogram[(key >> kRefineShift0) & 0xffu], 1);                        \
      }                                                                                  \
    }                                                                                    \
  } while (0)

  for (int index = tid; index < prefix; index += kCoarse12Threads)
    DEEP_GEMM_COLLECT_COARSE12_VALUE(__ldg(input + index), index);
  for (int vec = tid; vec < vector_length; vec += kCoarse12Threads) {
    const float4 values = __ldg(vector_input + vec);
    const int index = prefix + vec * 4;
    DEEP_GEMM_COLLECT_COARSE12_VALUE(values.x, index);
    DEEP_GEMM_COLLECT_COARSE12_VALUE(values.y, index + 1);
    DEEP_GEMM_COLLECT_COARSE12_VALUE(values.z, index + 2);
    DEEP_GEMM_COLLECT_COARSE12_VALUE(values.w, index + 3);
  }
  for (int index = tail + tid; index < length; index += kCoarse12Threads)
    DEEP_GEMM_COLLECT_COARSE12_VALUE(__ldg(input + index), index);
#undef DEEP_GEMM_COLLECT_COARSE12_VALUE
  __syncthreads();

  if (num_input[0] > kCoarse12CandidateCapacity) {
    int selected_prefix = wide_threshold;
    int prefix_bits = kCoarse12CoarseBits;
#pragma unroll 3
    for (int round = 0; round < 3; ++round) {
      if (tid < kRadix + 1) histogram[tid] = 0;
      __syncthreads();

      const int radix_bits = round == 2 ? kFinalRadixBits : 8;
      const int offset = round == 0 ? kRefineShift0 : (round == 1 ? kRefineShift1 : 0);
      const int mask = (1 << radix_bits) - 1;
      for (unsigned int index = tid; index < u_length; index += kCoarse12Threads) {
        const uint32_t key = refine_bin(__ldg(input + index));
        if (static_cast<int>(key >> (32 - prefix_bits)) != selected_prefix) continue;
        atomicAdd(&histogram[(key >> offset) & mask], 1);
      }
      __syncthreads();

      warp_cumsum_histogram<kRadix>(histogram);
      if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
        threshold_bin_id = tid;
        threshold_exclusive_count = histogram[tid + 1];
        last_remain = topk - histogram[tid + 1];
      }
      __syncthreads();

      const int threshold_bin = threshold_bin_id;
      topk -= threshold_exclusive_count;
      for (unsigned int index = tid; index < u_length; index += kCoarse12Threads) {
        const uint32_t key = refine_bin(__ldg(input + index));
        if (static_cast<int>(key >> (32 - prefix_bits)) != selected_prefix) continue;
        const int bin = (key >> offset) & mask;
        if (bin > threshold_bin) {
          const int position = atomicAdd(&counter, 1);
          output[position] = transform_index<Transform>(trans_info, index);
        }
        else if (round == 2 && bin == threshold_bin) {
          const int position = atomicAdd(&last_remain, -1);
          if (position > 0)
            output[requested_topk - position] =
                transform_index<Transform>(trans_info, index);
        }
      }
      __syncthreads();
      if (topk == 0 || round == 2) return;
      selected_prefix = (selected_prefix << radix_bits) | threshold_bin;
      prefix_bits += radix_bits;
    }
  }

#pragma unroll 3
  for (int round = 0; round < 3; ++round) {
    const int read = round & 1;
    const int current_offset = read * kCoarse12CandidateCapacity;
    const int next_offset = (read ^ 1) * kCoarse12CandidateCapacity;
    const int candidate_count = num_input[read];

    warp_cumsum_histogram<kRadix>(histogram);
    if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
      threshold_bin_id = tid;
      threshold_exclusive_count = histogram[tid + 1];
      num_input[read ^ 1] = 0;
      last_remain = topk - histogram[tid + 1];
    }
    __syncthreads();

    const int threshold_bin = threshold_bin_id;
    topk -= threshold_exclusive_count;
    const int offset = round == 0 ? kRefineShift0 : (round == 1 ? kRefineShift1 : 0);
    const int mask = round == 2 ? kFinalRadixMask : 0xff;
    if (topk == 0) {
      for (int i = tid; i < candidate_count; i += kCoarse12Threads) {
        const int index = candidate_indices[current_offset + i];
        const int bin = (refine_bin(__ldg(input + index)) >> offset) & mask;
        if (bin > threshold_bin) {
          const int position = atomicAdd(&counter, 1);
          output[position] = transform_index<Transform>(trans_info, index);
        }
      }
      __syncthreads();
      return;
    }

    if (tid < kRadix + 1) histogram[tid] = 0;
    __syncthreads();
    for (int i = tid; i < candidate_count; i += kCoarse12Threads) {
      const int index = candidate_indices[current_offset + i];
      const uint32_t key = refine_bin(__ldg(input + index));
      const int bin = (key >> offset) & mask;
      if (bin > threshold_bin) {
        const int pos = atomicAdd(&counter, 1);
        output[pos] = transform_index<Transform>(trans_info, index);
      }
      else if (bin == threshold_bin) {
        if (round == 2) {
          const int pos = atomicAdd(&last_remain, -1);
          if (pos > 0)
            output[requested_topk - pos] = transform_index<Transform>(trans_info, index);
        }
        else {
          const int position = atomicAdd(&num_input[read ^ 1], 1);
          if (position < kCoarse12CandidateCapacity) {
            candidate_indices[next_offset + position] = index;
            const int next_offset_bits = round == 0 ? kRefineShift1 : 0;
            const int next_mask = round == 0 ? 0xff : kFinalRadixMask;
            atomicAdd(&histogram[(key >> next_offset_bits) & next_mask], 1);
          }
        }
      }
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void topk_single_impl(
    const float *__restrict__ input, int32_t *__restrict__ indices, int length,
    int requested_topk)
{
  // Faithful port of indexer_topk::maca::fast_topk_cuda_tl.  The only
  // algorithmic parameter made dynamic is TopK; page transform and profiling
  // are intentionally absent.
  int topk = requested_topk;
  constexpr int kRadix = 1024;
  constexpr int kSmemInputSize = kCandidateCapacity;

  __shared__ int histogram[kRadix + kHistogramPadding];
  alignas(128) __shared__ int counter;
  alignas(128) __shared__ int threshold_bin_id;
  alignas(128) __shared__ int num_input[2];

  extern __shared__ int staged_indices[][kSmemInputSize];

  const int tid = threadIdx.x;
  const auto input_address = reinterpret_cast<uintptr_t>(input);
  const int vec4_prefix_unclamped = static_cast<int>(
      (alignof(float4) - (input_address & (alignof(float4) - 1))) / sizeof(float));
  const int vec4_prefix = (input_address & (alignof(float4) - 1)) == 0
                            ? 0
                            : min(length, vec4_prefix_unclamped);
  const auto *input_vec4 = reinterpret_cast<const float4 *>(input + vec4_prefix);
  const int vec4_length = (length - vec4_prefix) / 4;
  const int vec4_tail = vec4_prefix + vec4_length * 4;

  for (int i = tid; i < kRadix + 1; i += kThreads) {
    histogram[i] = 0;
  }
  __syncthreads();

  for (int i = tid; i < vec4_prefix; i += kThreads) {
    atomicAdd(&histogram[coarse_bin(input[i])], 1);
  }
  for (int vec_index = tid; vec_index < vec4_length; vec_index += kThreads) {
    const float4 values = input_vec4[vec_index];
    atomicAdd(&histogram[coarse_bin(values.x)], 1);
    atomicAdd(&histogram[coarse_bin(values.y)], 1);
    atomicAdd(&histogram[coarse_bin(values.z)], 1);
    atomicAdd(&histogram[coarse_bin(values.w)], 1);
  }
  for (int i = vec4_tail + tid; i < length; i += kThreads) {
    atomicAdd(&histogram[coarse_bin(input[i])], 1);
  }
  __syncthreads();

  warp_cumsum_histogram<kRadix>(histogram);
  if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
    threshold_bin_id = tid;
    num_input[0] = 0;
    counter = 0;
  }
  __syncthreads();

  int threshold_bin = threshold_bin_id;
  // The coarse threshold bin, kept for the rows that overflow the staging
  // buffer: `threshold_bin` itself is the refinement's, and moves every round.
  const int coarse_threshold = threshold_bin;
  topk -= histogram[threshold_bin + 1];

  if (topk == 0) {
    const auto append_if_above = [&](int index, int bin) {
      if (bin > threshold_bin) {
        const int position = atomicAdd(&counter, 1);
        indices[position] = index;
      }
    };
    for (int i = tid; i < vec4_prefix; i += kThreads) {
      append_if_above(i, coarse_bin(input[i]));
    }
    for (int vec_index = tid; vec_index < vec4_length; vec_index += kThreads) {
      const float4 values = input_vec4[vec_index];
      const int index = vec4_prefix + vec_index * 4;
      append_if_above(index, coarse_bin(values.x));
      append_if_above(index + 1, coarse_bin(values.y));
      append_if_above(index + 2, coarse_bin(values.z));
      append_if_above(index + 3, coarse_bin(values.w));
    }
    for (int i = vec4_tail + tid; i < length; i += kThreads) {
      append_if_above(i, coarse_bin(input[i]));
    }
    __syncthreads();
    return;
  }

  __syncthreads();
  for (int i = tid; i < kRadix + 1; i += kThreads) {
    histogram[i] = 0;
  }
  __syncthreads();

  const auto append_or_stage = [&](int index, float value, int bin) {
    if (bin > threshold_bin) {
      const int position = atomicAdd(&counter, 1);
      indices[position] = index;
    }
    else if (bin == threshold_bin) {
      const int position = atomicAdd(&num_input[0], 1);
      if (position < kSmemInputSize) {
        staged_indices[0][position] = index;
        const int sub_bin = (refine_bin(value) >> 24) & 0xff;
        atomicAdd(&histogram[sub_bin], 1);
      }
    }
  };
  for (int i = tid; i < vec4_prefix; i += kThreads) {
    append_or_stage(i, input[i], coarse_bin(input[i]));
  }
  for (int vec_index = tid; vec_index < vec4_length; vec_index += kThreads) {
    const float4 values = input_vec4[vec_index];
    const int index = vec4_prefix + vec_index * 4;
    append_or_stage(index, values.x, coarse_bin(values.x));
    append_or_stage(index + 1, values.y, coarse_bin(values.y));
    append_or_stage(index + 2, values.z, coarse_bin(values.z));
    append_or_stage(index + 3, values.w, coarse_bin(values.w));
  }
  for (int i = vec4_tail + tid; i < length; i += kThreads) {
    append_or_stage(i, input[i], coarse_bin(input[i]));
  }
  __syncthreads();

  // `staged_indices` holds `kSmemInputSize` candidates.  When the threshold bin
  // is wider than that, the surplus is not dropped any more -- it is not
  // staged at all and the refinement re-scans the row under the same predicate
  // the staging would have encoded (`coarse bin == threshold`, plus the refine
  // bits the previous rounds resolved).  Truncating instead answered for an
  // arbitrary subset of the bin: a wrong value multiset, and a different one
  // each run, since the race that fills the buffer is won by different
  // elements.  Only the rows whose bin is that wide pay the extra passes.
  const bool rescanned = num_input[0] > kSmemInputSize;
  int prefix_bits = 0;
  uint32_t selected_prefix = 0;
  if (rescanned) {
    // The staged elements' histogram misses whatever was not staged; rebuild
    // the first round's from the whole bin.
    for (int i = tid; i < kRadix + 1; i += kThreads) {
      histogram[i] = 0;
    }
    __syncthreads();
    for (int i = tid; i < length; i += kThreads) {
      const float value = input[i];
      if (coarse_bin(value) == coarse_threshold) {
        atomicAdd(&histogram[(refine_bin(value) >> 24) & 0xff], 1);
      }
    }
    __syncthreads();
  }

#pragma unroll 4
  for (int round = 0; round < 4; ++round) {
    __shared__ int last_remain;
    const int read = round & 1;
    const int raw_num_input = num_input[read];
    const int current_num_input = min(raw_num_input, kSmemInputSize);
    const int offset = 24 - round * 8;

    const auto for_each_candidate = [&](const auto &body) {
      if (!rescanned) {
        for (int i = tid; i < current_num_input; i += kThreads) {
          const int index = staged_indices[read][i];
          body(index, input[index]);
        }
      }
      else {
        for (int i = tid; i < length; i += kThreads) {
          const float value = input[i];
          if (coarse_bin(value) != coarse_threshold) continue;
          if (prefix_bits != 0
              && static_cast<uint32_t>(refine_bin(value) >> (32 - prefix_bits))
                     != selected_prefix) {
            continue;
          }
          body(i, value);
        }
      }
    };

    warp_cumsum_histogram<kRadix>(histogram);
    if (tid < kRadix && histogram[tid] > topk && histogram[tid + 1] <= topk) {
      threshold_bin_id = tid;
      num_input[read ^ 1] = 0;
      last_remain = topk - histogram[tid + 1];
    }
    __syncthreads();

    threshold_bin = threshold_bin_id;
    topk -= histogram[threshold_bin + 1];

    if (topk == 0) {
      for_each_candidate([&](int index, float value) {
        const int bin = (refine_bin(value) >> offset) & 0xff;
        if (bin > threshold_bin) {
          const int position = atomicAdd(&counter, 1);
          indices[position] = index;
        }
      });
      __syncthreads();
      break;
    }

    __syncthreads();
    for (int i = tid; i < kRadix + 1; i += kThreads) {
      histogram[i] = 0;
    }
    __syncthreads();
    for_each_candidate([&](int index, float value) {
      const int bin = (refine_bin(value) >> offset) & 0xff;
      if (bin > threshold_bin) {
        const int position = atomicAdd(&counter, 1);
        indices[position] = index;
      }
      else if (bin == threshold_bin) {
        if (round == 3) {
          const int position = atomicAdd(&last_remain, -1);
          if (position > 0) {
            indices[requested_topk - position] = index;
          }
        }
        else if (rescanned) {
          // There is no next staging buffer to fill -- the next round re-scans
          // under this bin, and this is its histogram.
          atomicAdd(&histogram[(refine_bin(value) >> (offset - 8)) & 0xff], 1);
        }
        else {
          const int position = atomicAdd(&num_input[read ^ 1], 1);
          if (position < kSmemInputSize) {
            staged_indices[read ^ 1][position] = index;
            const int sub_bin = (refine_bin(value) >> (offset - 8)) & 0xff;
            atomicAdd(&histogram[sub_bin], 1);
          }
        }
      }
    });
    if (rescanned) {
      selected_prefix = (selected_prefix << 8) | static_cast<uint32_t>(threshold_bin);
      prefix_bits += 8;
    }
    __syncthreads();
  }
}

template <bool Transform>
__global__ __launch_bounds__(kThreads) void topk_single(TopKParams params)
{
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  int begin;
  int length;
  const TransformInfo trans_info =
      initialize_row_context<Transform>(params, row, begin, length);
  int32_t *row_output = params.output + static_cast<int64_t>(row) * params.top_k;
  float *row_values = params.output_values == nullptr
                        ? nullptr
                        : params.output_values + static_cast<int64_t>(row) * params.top_k;
  const float *row_input =
      params.scores + static_cast<int64_t>(row) * params.n_cols + begin;
  if (length <= params.top_k) {
    write_trivial_topk_indices<Transform>(trans_info, row_output, length, params.top_k);
    for (int i = tid; i < params.top_k; i += kThreads) {
      if (row_values != nullptr)
        row_values[i] = i < length ? row_input[i] : kNegativeInfinity;
    }
    return;
  }

  __shared__ int selected_indices[kMaxTopK];
  topk_single_impl(row_input, selected_indices, length, params.top_k);
  for (int pos = tid; pos < params.top_k; pos += kThreads) {
    row_output[pos] = transform_index<Transform>(trans_info, selected_indices[pos]);
  }
  gather_topk_values(row_input, selected_indices, row_values, params.top_k);
}

template <bool Transform>
__global__ __launch_bounds__(kCoarse12Threads) void topk_coarse12(TopKParams params)
{
  const int row = blockIdx.x;
  const int tid = threadIdx.x;
  int begin;
  int length;
  const TransformInfo trans_info =
      initialize_row_context<Transform>(params, row, begin, length);
  int32_t *row_output = params.output + row * params.top_k;
  float *row_values = params.output_values == nullptr
                        ? nullptr
                        : params.output_values + static_cast<int64_t>(row) * params.top_k;
  const float *row_input = params.scores + row * params.n_cols + begin;
  if (length <= params.top_k) {
    write_trivial_topk_indices<Transform>(trans_info, row_output, length, params.top_k);
    for (int i = tid; i < params.top_k; i += kCoarse12Threads) {
      if (row_values != nullptr)
        row_values[i] = i < length ? row_input[i] : kNegativeInfinity;
    }
    return;
  }

  if constexpr (Transform) {
    // On the measured large-batch region, direct transform is cheaper for
    // short rows.  Once the effective row is at least four times top-k, first
    // writing compact local indices and then transforming them cooperatively
    // gives page-table loads a denser, dedicated writeback phase.  Smaller
    // batches retain the proven post-transform path.  `length` is CTA-uniform,
    // so this runtime branch adds no intra-CTA divergence or host sync.
    if (params.n_rows >= 1024 && (params.top_k < 1024 || length < 4 * params.top_k)) {
      topk_coarse12_impl<true>(row_input, row_output, length, params.top_k, trans_info);
    }
    else {
      topk_coarse12_impl<false>(row_input, row_output, length, params.top_k, trans_info);
      __syncthreads();
      for (int i = tid; i < params.top_k; i += kCoarse12Threads) {
        row_output[i] = transform_index<true>(trans_info, row_output[i]);
      }
    }
  }
  else {
    topk_coarse12_impl<false>(row_input, row_output, length, params.top_k, trans_info);
  }
  __syncthreads();
  gather_topk_values(row_input, row_output, row_values, params.top_k);
}

template <bool Transform, int ChunkCount>
__global__ void
topk_chunks_init(TopKChunksWorkspace<ChunkCount> *workspaces, TopKParams params)
{
  // One CTA owns one row: Wave 0 initializes its state and Wave 1 resolves the
  // optional page row.  Auto dispatch uses chunks only for small row counts,
  // but an explicit chunks policy may be used with an arbitrarily large page
  // batch, so Wave 1 must cover the whole cu_seqlens_row array.
  const int tid = threadIdx.x;
  const int row = blockIdx.x;
  if (tid < kWarpSize) {
    if (tid == 0) {
      int begin;
      int length;
      if (params.is_decode) {
        begin = 0;
        length = __ldg(params.seq_lens + row);
      }
      else {
        get_row_begin_length(params, row, begin, length);
      }
      workspaces[row].row_begin = begin;
      workspaces[row].row_length = length;
      workspaces[row].arrival = 0;
    }
  }
  else {
    const int lane = tid - kWarpSize;
    if constexpr (!Transform) {
      if (lane == 0) workspaces[row].trans_info = {nullptr, false, 0};
    }
    else {
      if (lane == 0) {
        workspaces[row].trans_info.q_position =
            params.q_positions == nullptr ? 0 : __ldg(params.q_positions + row);
        workspaces[row].trans_info.region_pack = params.region_pack;
      }
      if (!params.lookup_page_row) {
        // Decode has one page-table row per score row and needs no
        // cu_seqlens_row lookup.
        if (lane == 0)
          workspaces[row].trans_info.page_table_row =
              params.page_table + row * params.page_table_stride;
      }
      else {
        for (int batch = lane; batch < params.page_batch_size; batch += kWarpSize) {
          if (row >= params.cu_seqlens_row[batch]
              && row < params.cu_seqlens_row[batch + 1])
            workspaces[row].trans_info.page_table_row =
                params.page_table + batch * params.page_table_stride;
        }
      }
    }
  }
}

template <bool Transform, int NChunks>
__global__ __launch_bounds__(kThreads) void topk_chunks_coarse_hist(
    TopKParams params, TopKChunksWorkspace<NChunks> *workspaces)
{
  const int chunk = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  TopKChunksWorkspace<NChunks>& workspace = workspaces[row];
  const int begin = workspace.row_begin;
  const int length = workspace.row_length;
  if (length <= params.top_k) {
    if (chunk == 0) {
      const TransformInfo trans_info = workspace.trans_info;
      int32_t *row_output = params.output + static_cast<int64_t>(row) * params.top_k;
      float *row_values =
          params.output_values == nullptr
              ? nullptr
              : params.output_values + static_cast<int64_t>(row) * params.top_k;
      const float *row_input =
          params.scores + static_cast<int64_t>(row) * params.n_cols + begin;
      write_trivial_topk_indices<Transform>(trans_info, row_output, length, params.top_k);
      for (int pos = tid; pos < params.top_k; pos += kThreads) {
        if (row_values != nullptr)
          row_values[pos] = pos < length ? row_input[pos] : kNegativeInfinity;
      }
    }
    return;
  }

  __shared__ int histogram[kCoarseBins + kHistogramPadding];
  histogram[tid] = 0;
  __syncthreads();

  const int chunk_begin = length * chunk / NChunks;
  const int chunk_end = length * (chunk + 1) / NChunks;

  const auto row_input = params.scores + row * params.n_cols + begin;
  const auto chunk_input = row_input + chunk_begin;
  const int chunk_length = chunk_end - chunk_begin;
  const uintptr_t address = reinterpret_cast<uintptr_t>(chunk_input);
  const int prefix_unclamped = static_cast<int>(
      (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
  const int prefix =
      (address & (alignof(float4) - 1)) == 0 ? 0 : min(chunk_length, prefix_unclamped);
  const float4 *vector_input = reinterpret_cast<const float4 *>(chunk_input + prefix);
  const int vector_length = (chunk_length - prefix) / 4;
  const int tail = prefix + vector_length * 4;
  for (int index = tid; index < prefix; index += kThreads) {
    atomicAdd(&histogram[coarse_bin(chunk_input[index])], 1);
  }
  for (int index = tid; index < vector_length; index += kThreads) {
    const float4 values = vector_input[index];
    int bin_x, bin_y, bin_z, bin_w;
    coarse_bins_pair(values.x, values.y, bin_x, bin_y);
    coarse_bins_pair(values.z, values.w, bin_z, bin_w);
    atomicAdd(&histogram[bin_x], 1);
    atomicAdd(&histogram[bin_y], 1);
    atomicAdd(&histogram[bin_z], 1);
    atomicAdd(&histogram[bin_w], 1);
  }
  for (int index = tail + tid; index < chunk_length; index += kThreads) {
    atomicAdd(&histogram[coarse_bin(chunk_input[index])], 1);
  }
  __syncthreads();
  workspace.coarse[chunk][tid] = histogram[tid];
  __threadfence();
  __syncthreads();

  __shared__ int is_last;
  if (tid == 0) {
    is_last = atomicAdd(&workspace.arrival, 1) == NChunks - 1;
  }
  __syncthreads();
  if (!is_last) return;

  int total = 0;
#pragma unroll
  for (int source = 0; source < NChunks; ++source) {
    total += workspace.coarse[source][tid];
  }
  histogram[tid] = total;
  if (tid == 0) histogram[kCoarseBins] = 0;
  __syncthreads();
  warp_cumsum_histogram<kCoarseBins>(histogram);

  __shared__ int threshold;
  if (histogram[tid] >= params.top_k && histogram[tid + 1] < params.top_k) {
    threshold = tid;
  }
  __syncthreads();
  __shared__ int chunk_guaranteed[NChunks];
  __shared__ int chunk_boundary[NChunks];
  if (tid < NChunks) {
    int guaranteed = 0;
    for (int bin = threshold + 1; bin < kCoarseBins; ++bin) {
      guaranteed += workspace.coarse[tid][bin];
    }
    chunk_guaranteed[tid] = guaranteed;
    chunk_boundary[tid] = workspace.coarse[tid][threshold];
  }
  __syncthreads();
  if (tid == 0) {
    int guaranteed_base = 0;
    int boundary_base = 0;
#pragma unroll
    for (int source = 0; source < NChunks; ++source) {
      workspace.guaranteed_bases[source] = guaranteed_base;
      workspace.boundary_bases[source] = boundary_base;
      guaranteed_base += chunk_guaranteed[source];
      boundary_base += chunk_boundary[source];
    }
    workspace.threshold = threshold;
    workspace.guaranteed_count = guaranteed_base;
    workspace.boundary_take = params.top_k - guaranteed_base;
    workspace.candidate_count = boundary_base;
    workspace.arrival = 0;
  }
}

template <bool Transform, int NChunks>
__global__ __launch_bounds__(kThreads) void topk_chunks_compact_refine(
    TopKParams params, TopKChunksWorkspace<NChunks> *workspaces)
{
  const int chunk = blockIdx.x;
  const int row = blockIdx.y;
  const int tid = threadIdx.x;
  TopKChunksWorkspace<NChunks>& workspace = workspaces[row];
  if (workspace.row_length <= params.top_k) return;

  __shared__ TopKChunksShared shared;
  extern __shared__ int staged_indices[][kCandidateCapacity];
  const int begin = workspace.row_begin;
  const int length = workspace.row_length;
  const int chunk_begin = length * chunk / NChunks;
  const int chunk_end = length * (chunk + 1) / NChunks;
  const auto row_input = params.scores + row * params.n_cols + begin;
  int32_t *row_output = params.output + static_cast<int64_t>(row) * params.top_k;
  float *row_values = params.output_values == nullptr
                        ? nullptr
                        : params.output_values + static_cast<int64_t>(row) * params.top_k;
  const TransformInfo trans_info = workspace.trans_info;

  if (tid < kFineBins) shared.histogram[tid] = 0;
  if (tid == 0) {
    shared.guaranteed_counter = 0;
    shared.boundary_counter = 0;
  }
  __syncthreads();

  const int threshold = workspace.threshold;
  const int guaranteed_base = workspace.guaranteed_bases[chunk];
  const int boundary_base = workspace.boundary_bases[chunk];
  const float *chunk_input = row_input + chunk_begin;
  const int chunk_length = chunk_end - chunk_begin;
  const uintptr_t address = reinterpret_cast<uintptr_t>(chunk_input);
  const int prefix_unclamped = static_cast<int>(
      (alignof(float4) - (address & (alignof(float4) - 1))) / sizeof(float));
  const int prefix =
      (address & (alignof(float4) - 1)) == 0 ? 0 : min(chunk_length, prefix_unclamped);
  const float4 *vector_input = reinterpret_cast<const float4 *>(chunk_input + prefix);
  const int vector_length = (chunk_length - prefix) / 4;
  const int tail = prefix + vector_length * 4;
  const auto compact = [&](int local_index, float value, int bin) {
    const int column = chunk_begin + local_index;
    if (bin > threshold) {
      const int pos = atomicAdd(&shared.guaranteed_counter, 1);
      row_output[guaranteed_base + pos] = transform_index<Transform>(trans_info, column);
    }
    else if (bin == threshold) {
      // Every member of the bin is histogrammed, not only the staged ones: the
      // merge phase resolves its threshold from this histogram, and it may have
      // to refine a bin too wide for `candidate_indices` (see below), where the
      // unstaged members are the majority.
      atomicAdd(&shared.histogram[refine_bin(value) >> 24], 1);
      const int local_pos = atomicAdd(&shared.boundary_counter, 1);
      const int global_pos = boundary_base + local_pos;
      if (global_pos < kCandidateCapacity) {
        workspace.candidate_indices[global_pos] = column;
      }
    }
  };
  for (int i = tid; i < prefix; i += kThreads)
    compact(i, chunk_input[i], coarse_bin(chunk_input[i]));
  for (int i = tid; i < vector_length; i += kThreads) {
    const float4 values = vector_input[i];
    const int index = prefix + i * 4;
    int bin_x, bin_y, bin_z, bin_w;
    coarse_bins_pair(values.x, values.y, bin_x, bin_y);
    coarse_bins_pair(values.z, values.w, bin_z, bin_w);
    compact(index, values.x, bin_x);
    compact(index + 1, values.y, bin_y);
    compact(index + 2, values.z, bin_z);
    compact(index + 3, values.w, bin_w);
  }
  for (int i = tail + tid; i < chunk_length; i += kThreads)
    compact(i, chunk_input[i], coarse_bin(chunk_input[i]));
  __syncthreads();
  if (tid < kFineBins) workspace.fine[chunk][tid] = shared.histogram[tid];
  __threadfence();
  __syncthreads();
  if (tid == 0) shared.is_last = atomicAdd(&workspace.arrival, 1) == NChunks - 1;
  __syncthreads();
  if (!shared.is_last) return;

  // `workspace.candidate_count` is the true width of the coarse threshold bin,
  // known from the merged per-chunk coarse histograms before any member is
  // staged; `candidate_indices` holds only the first `kCandidateCapacity` of
  // them.  A bin wider than that is no longer truncated: its members are not
  // staged at all.  The fine histogram above is complete either way (every
  // chunk histograms each member, staged or not), and every pass over the bin
  // re-reads the row under the same predicate the staging would have encoded
  // (`coarse_bin == threshold`, plus the refine bytes resolved so far) -- the
  // shape `topk_single` uses.  Truncating instead answered for an arbitrary
  // subset of the bin: a wrong value multiset, and a different one each run,
  // since the race that fills the workspace is won by different elements.  Only
  // the rows whose bin is that wide pay the extra passes.
  const bool rescanned = workspace.candidate_count > kCandidateCapacity;
  const int candidate_count = min(workspace.candidate_count, kCandidateCapacity);
  int remaining = workspace.boundary_take;
  if (remaining == 0) {
    gather_topk_values(row_input, row_output, row_values, params.top_k);
    if (tid == 0) workspace.arrival = 0;
    return;
  }
  if (tid < kFineBins) {
    int count = 0;
#pragma unroll
    for (int source = 0; source < NChunks; ++source) count += workspace.fine[source][tid];
    shared.histogram[tid] = count;
  }
  if (tid == 0) {
    shared.histogram[kFineBins] = 0;
    shared.candidate_counter = 0;
  }
  __syncthreads();

  // The current candidate set: `staged` when the bin fits, the row re-scan
  // (restricted, once any refine byte has been resolved) when it does not.  The
  // re-scan branch ignores both arguments -- `staged_count` is meaningless
  // there, as nothing is ever staged.
  uint32_t selected_prefix = 0;
  int prefix_bits = 0;
  const auto for_each_candidate = [&](const int *staged, int staged_count,
                                      const auto &body) {
    if (!rescanned) {
      for (int pos = tid; pos < staged_count; pos += kThreads) {
        body(staged[pos], row_input[staged[pos]]);
      }
    }
    else {
      for (int i = tid; i < length; i += kThreads) {
        const float value = row_input[i];
        if (coarse_bin(value) != threshold) continue;
        if (prefix_bits != 0
            && (refine_bin(value) >> (32 - prefix_bits)) != selected_prefix) {
          continue;
        }
        body(i, value);
      }
    }
  };

  warp_cumsum_histogram<kFineBins>(shared.histogram);
  if (tid < kFineBins && shared.histogram[tid] >= remaining
      && shared.histogram[tid + 1] < remaining) {
    shared.threshold = tid;
    shared.last_remain = remaining - shared.histogram[tid + 1];
  }
  __syncthreads();
  int fine_threshold = shared.threshold;
  remaining -= shared.histogram[fine_threshold + 1];
  if (remaining == 0) {
    for_each_candidate(workspace.candidate_indices, candidate_count,
                       [&](int column, float value) {
      if (static_cast<int>(refine_bin(value) >> 24) > fine_threshold) {
        const int out = atomicAdd(&shared.candidate_counter, 1);
        row_output[workspace.guaranteed_count + out] =
            transform_index<Transform>(trans_info, column);
      }
    });
    __syncthreads();
  }
  else {
    if (tid < kFineBins) shared.histogram[tid] = 0;
    if (tid == 0) shared.boundary_counter = 0;
    __syncthreads();
    for_each_candidate(workspace.candidate_indices, candidate_count,
                       [&](int column, float value) {
      const uint32_t key = refine_bin(value);
      const int bin = key >> 24;
      if (bin > fine_threshold) {
        const int out = atomicAdd(&shared.candidate_counter, 1);
        row_output[workspace.guaranteed_count + out] =
            transform_index<Transform>(trans_info, column);
      }
      else if (bin == fine_threshold) {
        if (rescanned) {
          // There is no staging buffer to fill, so none to overflow: the next
          // round re-reads the row under this byte, and this is its histogram.
          atomicAdd(&shared.histogram[(key >> 16) & 0xffu], 1);
        }
        else {
          const int staged = atomicAdd(&shared.boundary_counter, 1);
          if (staged < kCandidateCapacity) {
            staged_indices[0][staged] = column;
            atomicAdd(&shared.histogram[(key >> 16) & 0xffu], 1);
          }
        }
      }
    });
    if (rescanned) {
      selected_prefix = static_cast<uint32_t>(fine_threshold);
      prefix_bits = 8;
    }
    __syncthreads();
#pragma unroll
    for (int round = 0; round < 3; ++round) {
      const int read_buffer = round & 1;
      const int count = min(shared.boundary_counter, kCandidateCapacity);
      warp_cumsum_histogram<kFineBins>(shared.histogram);
      if (tid < kFineBins && shared.histogram[tid] >= remaining
          && shared.histogram[tid + 1] < remaining) {
        shared.threshold = tid;
        shared.last_remain = remaining - shared.histogram[tid + 1];
      }
      __syncthreads();
      fine_threshold = shared.threshold;
      remaining -= shared.histogram[fine_threshold + 1];
      const int offset = 16 - round * 8;
      if (remaining == 0) {
        for_each_candidate(staged_indices[read_buffer], count,
                           [&](int column, float value) {
          const int bin = (refine_bin(value) >> offset) & 0xffu;
          if (bin > fine_threshold) {
            const int out = atomicAdd(&shared.candidate_counter, 1);
            row_output[workspace.guaranteed_count + out] =
                transform_index<Transform>(trans_info, column);
          }
        });
        __syncthreads();
        break;
      }
      if (tid < kFineBins) shared.histogram[tid] = 0;
      if (tid == 0) shared.boundary_counter = 0;
      __syncthreads();
      for_each_candidate(staged_indices[read_buffer], count,
                         [&](int column, float value) {
        const uint32_t key = refine_bin(value);
        const int bin = (key >> offset) & 0xffu;
        if (bin > fine_threshold) {
          const int out = atomicAdd(&shared.candidate_counter, 1);
          row_output[workspace.guaranteed_count + out] =
              transform_index<Transform>(trans_info, column);
        }
        else if (bin == fine_threshold) {
          if (round == 2) {
            const int left = atomicAdd(&shared.last_remain, -1);
            if (left > 0)
              row_output[params.top_k - left] =
                  transform_index<Transform>(trans_info, column);
          }
          else if (rescanned) {
            atomicAdd(&shared.histogram[(key >> (offset - 8)) & 0xffu], 1);
          }
          else {
            const int staged = atomicAdd(&shared.boundary_counter, 1);
            if (staged < kCandidateCapacity) {
              staged_indices[read_buffer ^ 1][staged] = column;
              atomicAdd(&shared.histogram[(key >> (offset - 8)) & 0xffu], 1);
            }
          }
        }
      });
      if (rescanned && round < 2) {
        selected_prefix =
            (selected_prefix << 8) | static_cast<uint32_t>(fine_threshold);
        prefix_bits += 8;
      }
      __syncthreads();
    }
  }
  __syncthreads();
  gather_topk_values(row_input, row_output, row_values, params.top_k);
  if (tid == 0) workspace.arrival = 0;
}

enum class TopKPolicy {
  Auto = 0,
  Chunks = 1,
  Single = 2,
  Coarse12 = 3,
};

__host__ __forceinline__ int
select_chunk_count(int64_t n_rows, int64_t n_cols, int num_sms)
{
  constexpr int kMinChunkCount = 3;
  constexpr int kMaxChunkCount = 6;
  constexpr int64_t kMinElementsPerChunk = 4096;
  // n_cols is the host-visible length proxy; reading seq_lens here would
  // introduce a device synchronization. Keep enough estimated work per CTA,
  // maximize parallelism in one wave, and reject under-filled tail waves.
  const int max_chunks =
      min(kMaxChunkCount, static_cast<int>(n_cols / kMinElementsPerChunk));
  if (max_chunks < kMinChunkCount) return 0;

  num_sms = max(num_sms, 1);
  const auto tail_blocks = [num_sms](int64_t blocks) {
    const int waves = static_cast<int>((blocks + num_sms - 1) / num_sms);
    return static_cast<int>(blocks - static_cast<int64_t>(waves - 1) * num_sms);
  };

  // Chunks add initialization, synchronization, and merge work. Only use them
  // when at least one candidate fills the last scheduling wave better than the
  // single-CTA-per-row launch. An exact multiple occupies a full last wave.
  const int single_tail = tail_blocks(n_rows);
  int best_chunk_tail = 0;
  for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks)
    best_chunk_tail = max(best_chunk_tail, tail_blocks(n_rows * chunks));
  if (best_chunk_tail <= single_tail) return 0;

  // Within a single wave, use as many chunks as possible to expose row-level
  // parallelism without adding another scheduling wave.
  int single_wave_chunks = 0;
  for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks) {
    const int64_t blocks = n_rows * chunks;
    if (blocks <= num_sms) single_wave_chunks = chunks;
  }
  if (single_wave_chunks != 0) return single_wave_chunks;

  // For multi-wave launches, prefer the smallest chunk count with a
  // sufficiently populated tail. This accounts for the fixed histogram,
  // arrival, and merge cost that a waves/chunks-only model misses.
  int selected = 0;
  int best_tail = -1;
  for (int chunks = kMinChunkCount; chunks <= max_chunks; ++chunks) {
    const int64_t blocks = n_rows * chunks;
    const int tail = tail_blocks(blocks);
    if (2 * tail >= num_sms) return chunks;
    if (tail > best_tail) {
      selected = chunks;
      best_tail = tail;
    }
  }
  return selected;
}

// The target (yellow) Python front end does not pass a policy, so `Auto` must
// still resolve host-side; the master tree moved this decision into its Python
// wrapper.  Keeping the shape-based choice here preserves the yellow port's
// routing (and the per-kernel arms of its candidate-overflow test) for both
// front ends sharing this csrc.
__host__ __forceinline__ TopKPolicy
select_topk_policy(const TopKParams& params, int chunk_count)
{
  if (params.n_rows <= 32)
    return chunk_count == 0 ? TopKPolicy::Single : TopKPolicy::Chunks;
  if (params.n_rows >= 128 && params.n_cols >= 2049 && params.top_k >= 256)
    return TopKPolicy::Coarse12;
  // Preserve the previous dispatch for smaller top-k values, which are not
  // covered by the relaxed-condition performance matrix yet.
  if (params.n_rows >= 1024 && params.n_cols >= 65536) return TopKPolicy::Coarse12;
  return TopKPolicy::Single;
}

template <bool Transform, int NChunks>
void _launch_topk_chunks(
    const TopKParams& params, TopKChunksWorkspace<NChunks> *workspace_ptr,
    cudaStream_t stream)
{
  // The original allocated this with `torch::empty` here.  The buffer is
  // caller-owned now: it must hold `params.n_rows * sizeof(TopKChunksWorkspace<NChunks>)`
  // bytes, and `deep_gemm_topk_chunks_workspace_bytes()` reports that size.
  topk_chunks_init<Transform, NChunks>
      <<<params.n_rows, 2 * kWarpSize, 0, stream>>>(workspace_ptr, params);
  DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

  const dim3 grid(NChunks, static_cast<unsigned int>(params.n_rows));
  topk_chunks_coarse_hist<Transform, NChunks>
      <<<grid, kThreads, 0, stream>>>(params, workspace_ptr);
  DG_CUDA_RUNTIME_CHECK(cudaGetLastError());

  static const cudaError_t smem_result = cudaFuncSetAttribute(
      topk_chunks_compact_refine<Transform, NChunks>,
      cudaFuncAttributeMaxDynamicSharedMemorySize, kTopKCandidateSmemBytes);
  DG_CUDA_RUNTIME_CHECK(smem_result);
  topk_chunks_compact_refine<Transform, NChunks>
      <<<grid, kThreads, kTopKCandidateSmemBytes, stream>>>(params, workspace_ptr);
}

template <bool Transform>
void launch_topk_single(const TopKParams& params, cudaStream_t stream)
{
  static const cudaError_t smem_result = cudaFuncSetAttribute(
      topk_single<Transform>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kTopKCandidateSmemBytes);
  DG_CUDA_RUNTIME_CHECK(smem_result);
  topk_single<Transform>
      <<<static_cast<unsigned int>(params.n_rows), kThreads, kTopKCandidateSmemBytes,
         stream>>>(params);
}

template <bool Transform>
void launch_topk_chunks(
    const TopKParams& params, TopKChunksWorkspace<3> *workspace_ptr,
    cudaStream_t stream)
{
  // The doubling is the original's, kept verbatim
  // (src/kernels/fp32_topk.cu:1553-1554: `num_sms * 2`); note that the *policy*
  // decision in `fp32_indexer_topk_impl` (line 1653-1655) passes the undoubled
  // count.  Either way the answer is 0 or 3..6, so `workspace_ptr` sized for
  // the largest of 3..6 is always big enough.
  const int num_sms = deep_gemm_device_sm_count() * 2;
  const int chunk_count = select_chunk_count(params.n_rows, params.n_cols, num_sms);
  switch (chunk_count) {
    case 3:
      _launch_topk_chunks<Transform, 3>(params, workspace_ptr, stream);
      return;
    case 4:
      _launch_topk_chunks<Transform, 4>(
          params, reinterpret_cast<TopKChunksWorkspace<4> *>(workspace_ptr), stream);
      return;
    case 5:
      _launch_topk_chunks<Transform, 5>(
          params, reinterpret_cast<TopKChunksWorkspace<5> *>(workspace_ptr), stream);
      return;
    case 6:
      _launch_topk_chunks<Transform, 6>(
          params, reinterpret_cast<TopKChunksWorkspace<6> *>(workspace_ptr), stream);
      return;
  }
  launch_topk_single<Transform>(params, stream);
}

template <bool Transform>
void launch_topk_coarse12(const TopKParams& params, cudaStream_t stream)
{
  static const cudaError_t smem_result = cudaFuncSetAttribute(
      topk_coarse12<Transform>, cudaFuncAttributeMaxDynamicSharedMemorySize,
      kCoarse12SmemBytes);
  DG_CUDA_RUNTIME_CHECK(smem_result);
  topk_coarse12<Transform>
      <<<static_cast<unsigned int>(params.n_rows), kCoarse12Threads, kCoarse12SmemBytes,
         stream>>>(params);
}

}  // namespace detail

// ---------------------------------------------------------------------------
// Torch-free host entry point.  This is the only piece the extraction adds: it
// reproduces the `Auto` arm of `fp32_indexer_topk_impl<false>` (the body that
// lived at src/kernels/fp32_topk.cu:1649-1673) with the torch pieces removed.
//
// Deliberately NOT reproduced from the original host wrapper, all of which is
// out of the 1..1584 kernel layer:
//   * `Transform = true` entries (`_transform`, `_region_pack`) -- they need the
//     page table / cu_seqlens_row / q_positions plumbing.
//   * `seq_starts`, `topk_values`, `page_table`, `region_pack`.
//   * `DeviceGuard` (the caller's stream/device is assumed current) and
//     `cudaGetLastError()` after the launch (the driver reports the same fault
//     at the next synchronizing call).
// ---------------------------------------------------------------------------

namespace {

// `select_topk_policy` reads `chunk_count` only for `n_rows <= 32`, so this is
// the original Auto arm's precondition (src/kernels/fp32_topk.cu:1652-1656)
// factored out: the SM count is only queried where it can change the answer.
int resolve_chunk_count(const detail::TopKParams &params, int num_sms, bool *needed = nullptr)
{
  if (params.n_rows <= 32) {
    if (needed != nullptr) *needed = true;
    return detail::select_chunk_count(params.n_rows, params.n_cols, num_sms);
  }
  return 0;
}

}  // namespace

int32_t deep_gemm_topk_policy(
    int64_t n_rows, int64_t n_cols, int32_t top_k, int num_sms)
{
  const detail::TopKParams params{
      nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr,
      n_rows,  n_cols,  0,       0,       0,       top_k,   false,   false,
      false};
  return static_cast<int32_t>(
      detail::select_topk_policy(params, resolve_chunk_count(params, num_sms)));
}

size_t deep_gemm_topk_chunks_workspace_bytes(int64_t n_rows, int chunk_count)
{
  switch (chunk_count) {
    case 3: return static_cast<size_t>(n_rows) * sizeof(detail::TopKChunksWorkspace<3>);
    case 4: return static_cast<size_t>(n_rows) * sizeof(detail::TopKChunksWorkspace<4>);
    case 5: return static_cast<size_t>(n_rows) * sizeof(detail::TopKChunksWorkspace<5>);
    case 6: return static_cast<size_t>(n_rows) * sizeof(detail::TopKChunksWorkspace<6>);
  }
  return 0;
}

int32_t deep_gemm_topk_chunk_count(int64_t n_rows, int64_t n_cols, int num_sms)
{
  return detail::select_chunk_count(n_rows, n_cols, num_sms * 2);
}

void deep_gemm_topk_selector(
    const float *scores, const int32_t *seq_lens, int32_t *out_indices,
    float *out_values, int32_t *chunks_workspace, int64_t n_rows, int64_t n_cols,
    int32_t top_k, cudaStream_t stream)
{
  // `is_decode` is the original predicate `!lookup_page_row && starts_ptr ==
  // nullptr && seq_size == n_rows`; with `seq_starts` absent, torch-free and
  // untransformed all that is left is "one seq_len per row", which for a plain
  // (n_rows, n_cols) score matrix is the case the caller wants.
  const detail::TopKParams params{
      scores,
      seq_lens,
      nullptr,
      nullptr,
      nullptr,
      nullptr,
      out_indices,
      out_values,
      n_rows,
      n_cols,
      0,
      0,
      static_cast<int32_t>(n_rows),  // next_n: one row per sequence
      top_k,
      true,                          // is_decode
      false,                         // lookup_page_row
      false};                        // region_pack

  using namespace detail;
  bool needs_sm_count = false;
  const TopKPolicy policy =
      select_topk_policy(params, resolve_chunk_count(params, 0, &needs_sm_count));
  switch (policy) {
    case TopKPolicy::Chunks: {
      if (chunks_workspace == nullptr)
        deep_gemm_abort(
            __FILE__, __LINE__,
            "chunks policy selected but chunks_workspace is null");
      launch_topk_chunks<false>(
          params, reinterpret_cast<TopKChunksWorkspace<3> *>(chunks_workspace),
          stream);
      break;
    }
    case TopKPolicy::Coarse12:
      launch_topk_coarse12<false>(params, stream);
      break;
    case TopKPolicy::Single:
    case TopKPolicy::Auto:
      launch_topk_single<false>(params, stream);
      break;
  }
}

}  // namespace deep_gemm::indexer
