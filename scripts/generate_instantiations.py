"""
Generate one explicit-instantiation .cu per device kernel this tree carries.

Usage, from the repo root:
    python3 scripts/generate_instantiations.py                  # xcore1000
    python3 scripts/generate_instantiations.py <instantiation_dir>
    # e.g. csrc/maca_kernels/xcore1600/v3/instantiations
    #      csrc/maca_kernels/xcore1600/v3_fp32/instantiations
    # [MACA] the old v3_cluster/instantiations went with cluster and TMA.

Creates one .cu per config, then prints the paths (repo-root relative) to copy
into `setup.py`'s `sources`.

Two kernel trees, two shapes of table:

  * **`csrc/maca_kernels/xcore1600/`** -- the ported upstream kernels, which
    this build does not compile.  Its tables are upstream's
    `TopkSelectConfigs` tuples with the MACA refits applied, and each generated
    file instantiates `run_topk_select_kernel<Config>`.

  * **`csrc/maca_kernels/xcore1000/`** -- what every family builds, and the
    default argument.  There the unit is a *launcher*: a host function whose
    body owns the `<<<>>>` launch of exactly one device kernel, so a tuple's
    file carries that kernel's device code and the launch that reaches it in
    one translation unit.  `setup.py` takes the list from `xcore1000_sources()`
    rather than from the printed output, so the two cannot drift.
"""

import dataclasses
import os
import shutil
import sys
from typing import List

# [MACA] The largest shared memory per SM any MACA part has.  Upstream's tables
# are sized for an H100's 227 KiB, so most of its tuples describe a launch no
# MACA part can perform -- this lets the generator refuse one at generation time
# with the arithmetic, instead of failing at compile time on whichever
# architecture happens to be building.  The same number `structs.h` gets through
# `-DDEEP_SELECT_NATIVE_ARCH`.
MACA_SMEM_CAPACITY_BYTES = 128 * 1024

# Where the repo root is, from this file rather than from the caller's cwd: this
# tree is vendored inside another one, so "run it from somewhere else" is a real
# place to be standing.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
XCORE1000_INSTANTIATION_DIR = os.path.join(
    REPO_ROOT, "csrc", "maca_kernels", "xcore1000", "instantiations")


@dataclasses.dataclass
class TopkSelectConfigs:
    ValueT: str
    OutIdxT: str
    sorted_value: bool
    sorted_index: bool
    return_value: bool
    max_topk: int
    num_threads: int
    target_occupancy: int
    elements_per_round: int
    reconstruct_threshold: int
    tma_buffer_depth: int
    cluster: int = 1

    def shared_memory_bytes(self) -> int:
        """`sizeof(SharedMemoryPlan)` for this config, in bytes.

        A hand-kept transcription of `SharedMemoryPlanBase` (common_parts.cuh)
        and the constants that size it -- keep the two in step.

        `tma_buffer_depth` deliberately does not appear: the MACA port has no
        TMA and no pipelining, so `NUM_TMA_LOAD_BUFS` is pinned to 1 and the
        parameter survives only as a record of upstream's tuning.
        """
        elem = 2 if self.ValueT == "nv_bfloat16" else 4
        num_warps = self.num_threads // 32
        num_extra_slots = self.reconstruct_threshold + self.elements_per_round
        num_bucket_slots = (1 << 8) + 4

        def align_up(x, a):
            return (x + a - 1) // a * a

        off = 0
        off = align_up(off, 1024) + 2 * self.max_topk * 8
        off = align_up(off, 1024) + num_extra_slots * 8
        off = align_up(off, 1024) + self.elements_per_round * elem
        off += num_warps * 4 + 4 + 4
        off = align_up(off, 16) + 2 * num_bucket_slots * 4
        return align_up(off, 1024)          # struct alignment = max member align

    def check_fits_maca(self):
        """Refuse a tuple that cannot occupy one SM of a 128 KiB MACA part.

        `target_occupancy` CTAs of this footprint share the SM, so the product
        is the quantity the capacity gate compares -- the same one upstream's
        own runtime check uses.
        """
        total = self.shared_memory_bytes() * self.target_occupancy
        assert total <= MACA_SMEM_CAPACITY_BYTES, (
            f"{self.max_topk=} {self.num_threads=} {self.target_occupancy=} "
            f"{self.elements_per_round=} {self.reconstruct_threshold=}: "
            f"{self.shared_memory_bytes()} B/CTA x {self.target_occupancy} = "
            f"{total} B/SM exceeds the {MACA_SMEM_CAPACITY_BYTES} B a MACA SM has"
        )

    def check_validity(self):
        assert self.ValueT in ["nv_bfloat16", "float"], "Invalid `ValueT`"
        assert self.OutIdxT in ["int32_t", "int64_t"], "Invalid `OutIdxT`"
        if self.sorted_value:
            assert self.return_value, "`return_value` must be `True` for `sorted_value`"
            assert not self.sorted_index, "`sorted_value` and `sorted_index` cannot be specified at the same time"
        if self.ValueT == "nv_bfloat16":
            assert not self.sorted_value, "`sorted_value` is fp32-only"
        assert self.max_topk in [512, 1024, 4096]
        assert self.elements_per_round == self.num_threads * 16, "contract ABI: B = NUM_THREADS * 16"
        assert self.reconstruct_threshold >= self.max_topk, "RECONSTRUCT_THRESHOLD >= MAX_TOPK"
        assert 1 <= self.cluster <= 16, "Invalid `cluster`"


def generate_instantiation_file(instantiation_dir: str, namespace: str, config: TopkSelectConfigs) -> str:
    # The config vocabulary is upstream's; the emitted source uses MACA's
    # spelling.  `nv_bfloat16` is not a type on MACA (`structs.h` includes
    # `<maca_bfloat16.h>`), so the upstream spelling verbatim does not compile.
    emitted_value_t = {"nv_bfloat16": "maca_bfloat16", "float": "float"}[config.ValueT]
    file_content = \
f"""#include "../topk_select.cuh"

namespace {namespace} {{

using Config = TopkSelectConfig<{emitted_value_t}, {config.OutIdxT}, {str(config.sorted_value).lower()}, {str(config.sorted_index).lower()}, {str(config.return_value).lower()}, {config.max_topk}, {config.num_threads}, {config.target_occupancy}, {config.elements_per_round}, {config.reconstruct_threshold}, {config.tma_buffer_depth}, 512, {config.cluster}>;

template
void run_topk_select_kernel<Config>(const TopkSelectArgs &args);

}}   // {namespace}
"""
    value_abbrev = {"nv_bfloat16": "bf16", "float": "fp32"}[config.ValueT]
    outidx_abbrev = {"int32_t": "i32", "int64_t": "i64"}[config.OutIdxT]
    file_name = f"value_{value_abbrev}_outidx_{outidx_abbrev}_sv{int(config.sorted_value)}_si{int(config.sorted_index)}_rv{int(config.return_value)}_maxtopk_{config.max_topk}_numthreads_{config.num_threads}_occupancy_{config.target_occupancy}"
    file_name += f"_b_{config.elements_per_round}_b2_{config.reconstruct_threshold}_tma_{config.tma_buffer_depth}"
    if config.cluster != 1:
        file_name += f"_cluster_{config.cluster}"
    file_name += ".cu"
    file_path = os.path.join(instantiation_dir, file_name)

    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(file_content)

    return file_path


def generate_instantiations(instantiation_dir: str, namespace: str, configs: List[TopkSelectConfigs]):
    for config in configs:
        config.check_validity()
        config.check_fits_maca()
    setup_py_source_files = []
    for config in configs:
        setup_py_source_files.append(generate_instantiation_file(instantiation_dir, namespace, config))
    for t in setup_py_source_files:
        print(f"\"{t}\",")


# ═══════════════════════════════════════════════════════════════════════════
# csrc/maca_kernels/xcore1000/ -- the tree every family builds
# ═══════════════════════════════════════════════════════════════════════════
#
# The unit here is a *launcher*: the host function whose body owns the
# `<<<>>>` launch of one device kernel.  Instantiating it in its own file puts
# the kernel's device code and the only launch that reaches it in the same
# translation unit, which is what makes the artifact carry the kernel exactly
# once -- the dispatch side declares the launcher `extern template` instead of
# defining it, so a tuple missing from these tables is an undefined symbol at
# link rather than a duplicate.
#
# Six tables, one per group of kernels the artifact carries.  The file count
# per group is the receipt: `ls instantiations/*.cu | wc -l` equals the number
# of `__device_stub__` symbols the artifact defines.


@dataclasses.dataclass(frozen=True)
class Instantiation:
    """One device kernel: the file that carries it, and its one instantiation."""

    family: str
    name: str            # the file stem -- the tuple, spelled out
    namespace: str       # where the launcher template is declared
    header: str          # relative to the instantiations/ directory
    instantiation: str   # the `template ...<args>(params);` it writes

    @property
    def file_name(self) -> str:
        return self.name + ".cu"

    def file_text(self) -> str:
        return (
            "// Generated by scripts/generate_instantiations.py -- do not edit.\n"
            f"// {self.family}: {self.name}\n"
            "\n"
            f'#include "{self.header}"\n'
            "\n"
            f"namespace {self.namespace} {{\n"
            "\n"
            f"{self.instantiation}\n"
            "\n"
            f"}}  // namespace {self.namespace}\n"
        )


def _inst(family, name, namespace, header, instantiation) -> Instantiation:
    return Instantiation(family, name, namespace, header, instantiation)


ROW_HEADER = "../contract.cuh"
FP32_ROW_HEADER = "../radix_row_fp32.cuh"
CHUNKED_HEADER = "../radix_chunked.cuh"
COARSE12_HEADER = "../topk_coarse12.cuh"
CHUNKS_HEADER = "../topk_chunks.cuh"

# ── row: `topk_kernel_radix<Cfg>` ───────────────────────────────────────────
#
# The axes are `docs/code-architecture.zh.md` §5b's, in the config's own
# spelling `TopkRadixConfig<ValueT, OutIdxT, BLOCK, SI, RV, SV, PRE>`, which is
# also the spelling the table in `contract.cuh` uses.  The two lists must agree;
# a tuple here that the table does not carry is an unused instantiation, and one
# the table wants that this list does not carry is an undefined symbol.
ROW_VALUE_TYPES = (("bf16", "maca_bfloat16"), ("fp32", "float"))
ROW_INDEX_TYPES = (("i32", "int32_t"), ("i64", "int64_t"))
# (SI, SV, RV).  `sorted_value` implies `return_value` and is exclusive with
# `sorted_index`, so five of the eight combinations are reachable and they are
# these five.
ROW_FLAGS = ((0, 0, 0), (0, 0, 1), (1, 0, 0), (1, 0, 1), (1, 1, 1))
# The wide block is the 16-bit row's; the fp32 row has one width.
ROW_BLOCKS = {"bf16": (512, 1024), "fp32": (512,)}
ROW_SIGNATURE = ("const deep_select::RowParams &params, uint32_t batches, "
                 "cudaStream_t stream, size_t smem")


def _b(flag: int) -> str:
    return "true" if flag else "false"


def row_instantiations() -> List[Instantiation]:
    out = []
    for value_tag, value_type in ROW_VALUE_TYPES:
        for idx_tag, idx_type in ROW_INDEX_TYPES:
            for block in ROW_BLOCKS[value_tag]:
                for si, sv, rv in ROW_FLAGS:
                    for pre in (0, 1):
                        cfg = (f"deep_select::TopkRadixConfig<{value_type}, "
                               f"{idx_type}, {block}, {_b(si)}, {_b(rv)}, "
                               f"{_b(sv)}, {_b(pre)}>")
                        out.append(_inst(
                            "row",
                            f"value_{value_tag}_outidx_{idx_tag}_block_{block}"
                            f"_si{si}_rv{rv}_sv{sv}_pre{pre}",
                            "deep_select::detail", ROW_HEADER,
                            f"template void launch_radix_config<{cfg}>(\n"
                            f"    {ROW_SIGNATURE});"))
    return out


# ── chunked: the two splits' stages ─────────────────────────────────────────
#
# Four arms, twelve kernels: the 16-bit split's static-k arm (stage 1 x
# `kNan`, stage 2 per `TOPK`), the same split's runtime-`topk` arm (stage 1 x
# `kNan`, stage 2 once), and the fp32 split (stage 1 x `kNan`, stage 2 once).
# The composing entry points (`launch_topk_bf16_chunked_k`,
# `launch_topk_bf16_chunked`, `launch_topk_f32_chunked`) keep their names and
# call these; they own no launch of their own.
CHUNK_BLOCK = "kChunkBlockSize"
F32_CHUNK_BLOCK = "kBlockSize"


def chunked_instantiations() -> List[Instantiation]:
    out = []
    for topk in (512, 1024):
        for nan in (0, 1):
            out.append(_inst(
                "chunked", f"bf16_stage1_k_topk{topk}_nan{nan}",
                "topk_select_radix", CHUNKED_HEADER,
                f"template cudaError_t launch_bf16_chunk_stage1_k<{topk}u, "
                f"{_b(nan)}>(\n"
                "    const maca_bfloat16 *scores, const int32_t *lengths,\n"
                "    int32_t *candidate_indices, maca_bfloat16 *candidate_values,\n"
                "    int B, int L, int num_chunks, cudaStream_t stream,\n"
                "    int64_t score_stride, int32_t *nan_flags);"))
        out.append(_inst(
            "chunked", f"bf16_stage2_k_topk{topk}",
            "topk_select_radix", CHUNKED_HEADER,
            f"template cudaError_t launch_bf16_chunk_stage2_k<{topk}u>(\n"
            "    const maca_bfloat16 *candidate_values,\n"
            "    const int32_t *candidate_indices, int32_t *indices, int B,\n"
            "    int num_chunks, cudaStream_t stream);"))
    for nan in (0, 1):
        out.append(_inst(
            "chunked", f"bf16_stage1_topk_runtime_nan{nan}",
            "topk_select_radix", CHUNKED_HEADER,
            f"template cudaError_t launch_bf16_chunk_stage1_runtime<{_b(nan)}>(\n"
            "    const maca_bfloat16 *scores, const int32_t *lengths,\n"
            "    int32_t *candidate_indices, maca_bfloat16 *candidate_values,\n"
            "    int B, int L, int topk, int num_chunks, cudaStream_t stream,\n"
            "    int64_t score_stride, int32_t *nan_flags);"))
    out.append(_inst(
        "chunked", "bf16_stage2_topk_runtime",
        "topk_select_radix", CHUNKED_HEADER,
        f"template cudaError_t launch_bf16_chunk_stage2_runtime<{CHUNK_BLOCK}>(\n"
        "    const maca_bfloat16 *candidate_values,\n"
        "    const int32_t *candidate_indices, int32_t *indices, int B,\n"
        "    int num_chunks, int topk, cudaStream_t stream);"))
    for nan in (0, 1):
        out.append(_inst(
            "chunked", f"fp32_stage1_nan{nan}",
            "topk_select_radix", CHUNKED_HEADER,
            f"template cudaError_t launch_f32_chunk_stage1<{_b(nan)}>(\n"
            "    const float *scores, const int32_t *lengths, int32_t *cols,\n"
            "    float *vals, int B, int L, int topk, int num_chunks,\n"
            "    cudaStream_t stream, int64_t score_stride, int32_t *nan_flags);"))
    out.append(_inst(
        "chunked", "fp32_stage2", "topk_select_radix", CHUNKED_HEADER,
        f"template cudaError_t launch_f32_chunk_stage2<{F32_CHUNK_BLOCK}>(\n"
        "    const float *vals, const int32_t *cols, int32_t *out, int B,\n"
        "    int topk, int num_chunks, cudaStream_t stream);"))
    return out


# ── coarse12: the ported `topk_coarse12` selector ───────────────────────────


def coarse12_instantiations() -> List[Instantiation]:
    return [_inst(
        "coarse12", "coarse12_kernel", "topk_select_coarse12", COARSE12_HEADER,
        "template cudaError_t launch_coarse12_kernel<kThreads>(\n"
        "    const float *scores, const int32_t *lengths, int32_t *out,\n"
        "    int32_t *nan_flags, int B, int topk, int64_t stride,\n"
        "    int default_length, cudaStream_t stream);")]


# ── chunks: the ported `topk_chunks` selector ───────────────────────────────
#
# Three kernels per `ChunkCount`, and `ChunkCount` is what
# `select_chunk_count` resolves the shape to.

CHUNK_COUNTS = (3, 4, 5, 6)


def chunks_instantiations() -> List[Instantiation]:
    out = []
    for count in CHUNK_COUNTS:
        out.append(_inst(
            "chunks", f"chunks_init_c{count}", "topk_select_chunks", CHUNKS_HEADER,
            f"template cudaError_t launch_chunks_init<{count}>(\n"
            "    void *workspace, const int32_t *lengths, int rows,\n"
            "    int default_length, cudaStream_t stream);"))
        out.append(_inst(
            "chunks", f"chunks_coarse_hist_c{count}", "topk_select_chunks",
            CHUNKS_HEADER,
            f"template cudaError_t launch_chunks_coarse_hist<{count}>(\n"
            "    const float *scores, void *workspace, int32_t *nan_flags,\n"
            "    int topk, int64_t stride, int default_length, int rows,\n"
            "    cudaStream_t stream);"))
        out.append(_inst(
            "chunks", f"chunks_compact_refine_c{count}", "topk_select_chunks",
            CHUNKS_HEADER,
            f"template cudaError_t launch_chunks_compact_refine<{count}>(\n"
            "    const float *scores, int32_t *out, void *workspace, int topk,\n"
            "    int64_t stride, int default_length, int rows, cudaStream_t stream);"))
    return out


# ── the two groups that are carried but not routed to ───────────────────────
#
# Neither is reached by a shape predicate: the fp32 static-k2048 pair has no
# caller outside `radix_row_fp32.cuh`'s own launchers, and the contract layer's
# two kernels are launched from the ladder and the entry point rather than
# selected.  They are in the tables because the artifact carries them -- a
# `__global__` is emitted whether or not anything launches it -- and the file
# count is what says so.


def fp32_static_k_instantiations() -> List[Instantiation]:
    signature = ("    const float* scores, const int32_t* lengths,\n"
                 "    const int32_t* row_starts, int32_t* indices, int B, int L,\n"
                 "    cudaStream_t stream);")
    return [
        _inst("fp32_static_k", "fp32_k2048_b1024", "topk_select_radix",
              FP32_ROW_HEADER,
              "template cudaError_t launch_f32_k2048_b1024<kF32K2048BlockSize>(\n"
              + signature),
        _inst("fp32_static_k", "fp32_k2048_b1024_warp_scan", "topk_select_radix",
              FP32_ROW_HEADER,
              "template cudaError_t launch_f32_k2048_b1024_warp_scan<"
              "kF32K2048BlockSize>(\n" + signature),
    ]


def contract_instantiations() -> List[Instantiation]:
    return [
        _inst("contract", "nan_scan_fp32", "deep_select", ROW_HEADER,
              "template void launch_nan_scan<float>(\n"
              "    const void *input, const int32_t *lengths, int64_t stride_elems,\n"
              "    uint32_t vocab_size, uint32_t chunks, int32_t *flags,\n"
              "    uint32_t batches, cudaStream_t stream);"),
        _inst("contract", "nan_abort", "deep_select", ROW_HEADER,
              "template void launch_nan_abort<kNanAbortBlock>(\n"
              "    const int32_t *flags, uint32_t batches, cudaStream_t stream);"),
    ]


def xcore1000_instantiations() -> List[Instantiation]:
    """Every device kernel this tree carries, one record per kernel."""
    return (row_instantiations() + chunked_instantiations()
            + coarse12_instantiations() + chunks_instantiations()
            + fp32_static_k_instantiations() + contract_instantiations())


def xcore1000_sources() -> List[str]:
    """The generated sources, repo-root-relative -- what `setup.py` compiles."""
    return [os.path.relpath(os.path.join(XCORE1000_INSTANTIATION_DIR, i.file_name),
                            REPO_ROOT)
            for i in xcore1000_instantiations()]


def xcore1000_counts() -> List[tuple]:
    """`(family, tuple count)` in table order -- the per-group receipt."""
    counts = []
    for inst in xcore1000_instantiations():
        if counts and counts[-1][0] == inst.family:
            counts[-1] = (inst.family, counts[-1][1] + 1)
        else:
            counts.append((inst.family, 1))
    return counts


def write_xcore1000() -> List[Instantiation]:
    """Remake the directory and write one file per tuple."""
    if os.path.exists(XCORE1000_INSTANTIATION_DIR):
        shutil.rmtree(XCORE1000_INSTANTIATION_DIR)
    os.makedirs(XCORE1000_INSTANTIATION_DIR)
    everything = xcore1000_instantiations()
    names = [i.file_name for i in everything]
    if len(set(names)) != len(names):
        raise ValueError("two tuples share a file name")
    for inst in everything:
        with open(os.path.join(XCORE1000_INSTANTIATION_DIR, inst.file_name),
                  "w", encoding="utf-8") as f:
            f.write(inst.file_text())
    return everything


def main(instantiation_dir: str):
    def remove_and_remake_dir():
        if os.path.exists(instantiation_dir):
            shutil.rmtree(instantiation_dir)
        os.makedirs(instantiation_dir, exist_ok=True)
    if instantiation_dir == "csrc/maca_kernels/xcore1600/v3/instantiations":
        remove_and_remake_dir()
        # bf16: sv0 x si{0,1} x rv{0,1}
        valid_si_rv_combinations = [
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ]
        # [MACA] Upstream's tuples, re-derived for the 128 KiB a MACA SM has
        # instead of the 227 KiB they were sized for.  Both flagship tuples drop
        # target_occupancy 2 -> 1 (two CTAs do not fit, and there is no TMA
        # pipelining here to overlap anyway); the mk=1024 wave1 tuple drops B2
        # 4096 -> 3584 to fit.  There is no mk=4096 tuple and cannot be:
        # `surviving_topk_pairs` alone is 2 * MAX_TOPK * 8 = 65536 B and the
        # extra-pairs region at least (MAX_TOPK + B) * 8 = 65536 B, which fills
        # the SM before a single input element is staged.  `topk` in
        # (1024, 4096] is served by `csrc/maca_kernels/xcore1000/maca_topk.cu` instead.
        #
        # TMA4 costs 109.6 KB of smem per CTA, so occupancy 2 only fits at
        # max_topk <= 512.
        fast_path_tuples_by_max_topk = {
            512:  [(256, 1, 4096, 4096, 4),     # flagship (num_waves >= 2)
                   (512, 1, 8192, 4096, 5)],    # wave1
            1024: [(256, 1, 4096, 4096, 3),     # flagship (num_waves >= 2)
                   (512, 1, 8192, 3584, 5)],    # wave1
        }
        configs = []
        for out_idx_t in ["int32_t", "int64_t"]:
            for si, rv in valid_si_rv_combinations:
                for max_topk, fast_path_tuples in fast_path_tuples_by_max_topk.items():
                    for num_threads, occ, b, b2, tma in fast_path_tuples:
                        configs.append(TopkSelectConfigs("nv_bfloat16", out_idx_t, False, si, rv, max_topk, num_threads, occ, b, b2, tma))
        generate_instantiations(instantiation_dir, "topk_select_bf16_normal", configs)
    elif instantiation_dir == "csrc/maca_kernels/xcore1600/v3_fp32/instantiations":
        remove_and_remake_dir()
        # fp32: sv0 x si{0,1} x rv{0,1} + sv1_si0_rv1
        valid_sv_si_rv_combinations_fp32 = [
            (False, False, False),
            (False, False, True),
            (False, True, False),
            (False, True, True),
            (True, False, True),
        ]
        # [MACA] B2 lowered and mk=4096 dropped, both for the reasons in the
        # bf16 table above.  fp32 pays more than bf16 for the same B2, because
        # `tma_load_buf` holds B elements of the value type.
        tuple_by_max_topk = {
            512: (512, 1, 8192, 2560, 3),
            1024: (512, 1, 8192, 1536, 3),
        }
        configs = []
        for out_idx_t in ["int32_t", "int64_t"]:
            for sv, si, rv in valid_sv_si_rv_combinations_fp32:
                for max_topk, (num_threads, occ, b, b2, tma) in tuple_by_max_topk.items():
                    configs.append(TopkSelectConfigs("float", out_idx_t, sv, si, rv, max_topk, num_threads, occ, b, b2, tma))
        generate_instantiations(instantiation_dir, "topk_select_fp32", configs)
    # [MACA] Upstream's `v3_cluster` branch (bf16 + mk1024 + a 16-CTA cluster)
    # is gone: MACA has no cluster and no TMA, so the variant went with them.
    elif os.path.normpath(instantiation_dir) in (
            "csrc/maca_kernels/xcore1000/instantiations",
            os.path.relpath(XCORE1000_INSTANTIATION_DIR, REPO_ROOT)):
        write_xcore1000()
        for family, count in xcore1000_counts():
            print(f"# {family}: {count}", file=sys.stderr)
        print(f"# total: {len(xcore1000_instantiations())}", file=sys.stderr)
        for path in xcore1000_sources():
            print(f"\"{path}\",")
    else:
        raise ValueError(f"Invalid `instantiation_dir: {instantiation_dir}")


if __name__ == '__main__':
    if len(sys.argv) > 2:
        print(f"Usage: {sys.argv[0]} [<instantiation_dir>]", file=sys.stderr)
        sys.exit(1)
    # No argument is the tree this build compiles.
    main(sys.argv[1] if len(sys.argv) == 2
         else "csrc/maca_kernels/xcore1000/instantiations")
