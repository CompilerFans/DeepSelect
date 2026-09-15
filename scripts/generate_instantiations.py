"""
Generate explicit template instantiation .cu files for a topk kernel template.

Usage, from the repo root:
    python3 scripts/generate_instantiations.py <instantiation_dir>
    # e.g. csrc/xcore1600/v3/instantiations
    #      csrc/xcore1600/v3_fp32/instantiations
    # [MACA] the old v3_cluster/instantiations went with cluster and TMA.

Creates one .cu per config, then prints the paths (repo-root relative) to copy
into `setup.py`'s `sources`.
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

def main(instantiation_dir: str):
    def remove_and_remake_dir():
        if os.path.exists(instantiation_dir):
            shutil.rmtree(instantiation_dir)
        os.makedirs(instantiation_dir, exist_ok=True)
    if instantiation_dir == "csrc/xcore1600/v3/instantiations":
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
        # (1024, 4096] is served by `csrc/xcore1000/maca_topk.cu` instead.
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
    elif instantiation_dir == "csrc/xcore1600/v3_fp32/instantiations":
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
    else:
        raise ValueError(f"Invalid `instantiation_dir: {instantiation_dir}")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <instantiation_dir>")
        sys.exit(1)
    instantiation_dir = sys.argv[1]
    main(instantiation_dir)
