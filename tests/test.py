import argparse
import dataclasses
import os
import time
from typing import Optional, List
import copy
import sys

import torch
import kernelkit as kk
import random

import deep_select

import lib
from lib import TestParam, Testcase, UniformUIntDistribution, NormalFloatDistribution, UintDistributionWithHotspotAndSpecifiedPivot, get_fp_config

def get_space(t: torch.Tensor):
    return t.numel() * t.element_size()

_counter = kk.Counter()

def check_call_contract(p: TestParam, ans_topk_value, ans_topk_index):
    """The assertions that are about the *call*, not the values it returned."""
    assert ans_topk_index.dtype == p.out_idx_dtype
    if p.return_value:
        assert ans_topk_value is not None
    else:
        assert ans_topk_value is None


def check_result(p: TestParam, t: Testcase, ans_topk_value, ans_topk_index) -> bool:
    """The official correctness assertions, as a predicate.  Split out so
    `scripts/perf_snapshot.py` checks other backends against this code, not a
    copy of it; the two answers are the caller's clones."""
    is_correct = True
    has_nan_mask = torch.zeros((p.batch_size,), dtype=torch.bool)
    if t.input.isnan().any().item():
        has_nan_mask = t.input.isnan()
        if t.end is not None:
            lib.row_wise_masked_fill_(has_nan_mask, t.end, False)
        has_nan_mask = has_nan_mask.any(dim=-1)  # [batch_size]

    selected_counts = (
        torch.full((p.batch_size,), min(p.vocab_size, p.topk), dtype=torch.int32)
        if t.end is None
        else torch.clamp(t.end, max=p.topk)
    )
    selected_mask = torch.arange(p.topk).unsqueeze(0) < selected_counts.unsqueeze(1)

    # Assert: index[0] = 0x3F3F3F3F for rows contain NaN
    valid_nan_guard = ans_topk_index[has_nan_mask, 0] == 0x3F3F3F3F
    valid_nan_guard |= p.vocab_size <= p.topk if t.end is None else t.end[has_nan_mask] <= p.topk
    is_correct &= kk.check_is_bitwise_equal("NaN guard", valid_nan_guard, torch.ones_like(valid_nan_guard))

    selected_mask &= ~has_nan_mask.unsqueeze(1)
    selected_index = ans_topk_index.to(torch.int64)
    if t.output_idx_offset is not None:
        selected_index -= t.output_idx_offset.to(torch.int64).unsqueeze(1)

    # Assert: 0 <= index < valid_len
    valid_len = torch.full((p.batch_size,), p.vocab_size, dtype=torch.int64) if t.end is None else t.end.to(torch.int64)
    index_in_range = (selected_index >= 0) & (selected_index < valid_len.unsqueeze(1))
    valid_index_mask = index_in_range | ~selected_mask
    is_correct &= kk.check_is_bitwise_equal("index range", valid_index_mask, torch.ones_like(valid_index_mask))

    # Assert: index[i] != index[j] (i != j)
    sorted_selected_index = torch.sort(
        torch.where(selected_mask, selected_index, torch.iinfo(torch.int64).max), dim=1
    ).values
    duplicate_mask = sorted_selected_index[:, 1:] == sorted_selected_index[:, :-1]
    duplicate_mask &= selected_mask[:, 1:]
    is_correct &= kk.check_is_bitwise_equal("unique index", duplicate_mask, torch.zeros_like(duplicate_mask))

    safe_selected_index = torch.where(selected_mask & index_in_range, selected_index, 0)
    gathered_value = t.input.gather(1, safe_selected_index)
    valid_row_mask = ~has_nan_mask
    if bool(torch.any(valid_row_mask).item()):
        valid_selected_mask = selected_mask[valid_row_mask]
        valid_gathered_value = gathered_value[valid_row_mask]

        # Assert: value_i = input[index_i]
        if ans_topk_value is not None:
            expected_value = valid_gathered_value.masked_fill(~valid_selected_mask, p.value_oob_fill_value)
            is_correct &= kk.check_is_bitwise_equal(
                "topk gathered value", ans_topk_value[valid_row_mask], expected_value
            )

        # Assert: min(selected) >= max(unselected)
        selected_min = valid_gathered_value.masked_fill(~valid_selected_mask, float("inf")).amin(dim=1)
        unselected_input = t.input[valid_row_mask].clone()
        unselected_input.scatter_(1, safe_selected_index[valid_row_mask], float("-inf"))
        lib.row_wise_masked_fill_(unselected_input, valid_len[valid_row_mask], float("-inf"))
        topk_condition = selected_min >= unselected_input.amax(dim=1)
        is_correct &= kk.check_is_bitwise_equal("topk condition", topk_condition, torch.ones_like(topk_condition))

    # Assert: index[i] <= index[i+1] if sorted_index
    ordered_mask = selected_mask[:, 1:]
    if p.sorted_index:
        index_ordered = ans_topk_index[:, 1:] >= ans_topk_index[:, :-1]
        index_ordered |= ~ordered_mask
        is_correct &= kk.check_is_bitwise_equal("sorted index", index_ordered, torch.ones_like(index_ordered))

    # Assert: value[i] >= value[i+1] if sorted_value
    if p.sorted_value:
        assert ans_topk_value is not None
        value_ordered = ans_topk_value[:, :-1] >= ans_topk_value[:, 1:]
        value_ordered |= ~ordered_mask
        is_correct &= kk.check_is_bitwise_equal("sorted value", value_ordered, torch.ones_like(value_ordered))
    return bool(is_correct)


def topk_total_size(p: TestParam, t: Testcase, ans_topk_value, ans_topk_index) -> int:
    """The operator's own traffic in bytes, as `run_testcase`'s TB/s line
    computes it.  An output may be `None` for the reference backend, which
    writes its own."""
    return (t.end.sum() if t.end is not None else p.batch_size * p.vocab_size) * t.input.element_size() \
        + (get_space(ans_topk_value) if ans_topk_value is not None else 0) \
        + (get_space(ans_topk_index) if ans_topk_index is not None else 0)


def bench_topk(fn, p: TestParam, t: Testcase, ans_topk_value, ans_topk_index):
    """The official timing rule: `(time_usage, total_size)` in seconds and bytes.

    One matching kernel name is that kernel's time, several is the span over
    them.  Zero matches is `None`, not 0 -- "0 us" reads as implausibly good."""
    total_size = topk_total_size(p, t, ans_topk_value, ans_topk_index)
    bench_result = kk.bench(fn, p.num_runs)
    kernel_names = [s for s in bench_result.get_kernel_names() if "topk" in s.lower()]
    if len(kernel_names) == 1:
        time_usage = bench_result.get_kernel_time(kernel_names[0])
    elif kernel_names:
        time_usage = bench_result.get_e2e_time(kernel_names)
    else:
        time_usage = None
    return time_usage, total_size


def bench_torch_reference(p: TestParam, t: Testcase):
    """The `torch.topk` reference backend, timed by the rule above.

    A *bare* `torch.topk`, not `deep_select.topk(backend="torch")`: that one
    pads, masks and converts, launching ~20 kernels where this launches one.
    The guard is the runner's own, repeated so this is never called where a bare
    `torch.topk` would raise."""
    if t.end is not None or t.output_idx_offset is not None or p.vocab_size < p.topk:
        return None

    def run_torch_topk():
        return torch.topk(t.input, p.topk, dim=1, sorted=p.sorted_value)

    bench_result = kk.bench(run_torch_topk, p.num_runs)
    kernel_names = [s for s in bench_result.get_kernel_names() if "topk" in s.lower()]
    if len(kernel_names) == 1:
        return bench_result.get_kernel_time(kernel_names[0])
    if kernel_names:
        return bench_result.get_e2e_time(kernel_names)
    return None


@torch.inference_mode()
def run_testcase(p: TestParam, backend: str = "maca_c"):
    """Run one case and answer whether it selected correctly.

    `backend` is passed to `deep_select.topk` at the call site.  The default is
    `maca_c` -- the kernel this tree exists to ship -- so a bare call tests the
    kernel and not the reference implementation that validates it.
    """
    if p.seed == -1:
        global _counter
        p.seed = _counter.next()

    print(f"Running on {p}")

    t = lib.generate_testcase(p)

    def run_topk_select(testcase: Testcase = t, start_batch_idx: int = 0, end_batch_idx: Optional[int] = None):
        if end_batch_idx is None:
            end_batch_idx = testcase.input.size(0)
        return deep_select.topk(
            testcase.input[start_batch_idx: end_batch_idx],
            p.topk,
            sorted=p.sorted_value,
            begin=None,
            end=testcase.end,
            indices_type=p.out_idx_dtype,
            sorted_index=p.sorted_index,
            hint=None,
            output_idx=None,
            output_idx_offset=testcase.output_idx_offset[start_batch_idx: end_batch_idx] if testcase.output_idx_offset is not None else None,
            idx_oob_fill_value=p.idx_oob_fill_value,
            value_oob_fill_value=p.value_oob_fill_value,
            return_value=p.return_value,
            abort_when_nan_found=False,
            backend=backend,
        )

    ans_topk_value, ans_topk_index = run_topk_select()
    batch_topk_value = ans_topk_value.clone() if ans_topk_value is not None else None
    batch_topk_index = ans_topk_index.clone()
    check_call_contract(p, ans_topk_value, ans_topk_index)

    is_correct = True
    if p.check_correctness:
        is_correct &= check_result(p, t, batch_topk_value, batch_topk_index)

    if p.num_runs > 0:
        time_usage, total_size = bench_topk(run_topk_select, p, t, ans_topk_value, ans_topk_index)
        if time_usage is None:
            print("topk           : (no kernel name contains \"topk\"; not timed)")
        else:
            print(f"topk           : {time_usage * 1e6:9.3f} us, {total_size / time_usage / 1e12:.3f} TB/s")

        if t.end is None and t.output_idx_offset is None and p.vocab_size >= p.topk:
            torch_time = bench_torch_reference(p, t)
            if torch_time and time_usage is not None:
                print(f"torch.topk     : {torch_time * 1e6:9.3f} us, {total_size / torch_time / 1e12:.3f} TB/s  (speedup {torch_time / time_usage:.2f}x)")

    return is_correct

def correctness_cases_() -> List[TestParam]:
    """The official correctness table; a function so a caller can add to it,
    the same reason `performance_cases` is one."""
    valid_sv_si_rv_combinations = [ # sv_si_rv: sorted_value, sorted_index, return_value
        (False, False, False),
        (False, False, True),
        (True, False, True),
        (False, True, True),
        (False, True, False)
    ]

    correctness_cases = []
    for dtype in [torch.float, torch.bfloat16]:
        for out_idx_dtype in [torch.int32, torch.int64]:
            for sv, si, rv in valid_sv_si_rv_combinations:
                if sv and dtype == torch.bfloat16:
                    # sorted_value is fp32-only
                    continue
                for b in [random.randint(1, 20), random.randint(1, 100), 121, 512, 4096]:
                    for vocab_size in [
                        1, random.randint(2, 500), 1602, 32768, 123245, 225467, 262144, 418673, 682965, 998123, 1048576, (1 << 23) - 1
                    ]:
                        if b == 4096 and vocab_size > 1048576:
                            continue    # To avoid OOM
                        for topk in [
                            1, random.randint(2, 500), 512, 1024, 1231, 2048, 2132, 2333, 4096
                        ]:
                            enable_end_position = topk%2 == 1   # derived, not enumerated, to keep the case count down
                            for distrib in [
                                NormalFloatDistribution(post_proc_func)
                                for post_proc_func in [
                                    None,
                                    lambda x: x + (100 if enable_end_position else -100),
                                    lambda x: x * (100 if enable_end_position else 1/100),
                                    lambda x: x * (1000 if enable_end_position else 1/1000),
                                    lambda x: x * (10000 if enable_end_position else 1/10000),
                                ]
                            ] + [
                                UniformUIntDistribution(
                                    get_fp_config(dtype).negative_0_as_int if enable_end_position else get_fp_config(dtype).positive_0_as_int,
                                    get_fp_config(dtype).negative_inf_as_int if enable_end_position else get_fp_config(dtype).positive_inf_as_int
                                ),
                                UniformUIntDistribution(0x0, 0x1),
                                UniformUIntDistribution(0x0, 0x20),
                                UniformUIntDistribution(0x0, 0x1000),
                            ] + [
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).positive_nan, min(8, vocab_size))], True),
                                UintDistributionWithHotspotAndSpecifiedPivot(None, topk, [(get_fp_config(dtype).negative_nan, min(8, vocab_size))], True),
                            ]:
                                enable_output_idx_offset = b%2 == 1
                                cur_case = TestParam(b, vocab_size, topk, sv, si, rv, dtype, out_idx_dtype, enable_end_position, enable_output_idx_offset, num_runs=0, idx_oob_fill_value=-2000000+vocab_size, input_distrib=distrib)
                                correctness_cases.append(cur_case)
    return correctness_cases

# ── the `deep_gemm` selector's shapes ───────────────────────────────────────
# `deep_gemm/tests/test_indexer_topk_selector.py`'s `SELECTOR_PERF_SHAPES`
# (= test-topk + sglang + dsa), all `top_k = 2048`, fp32.  Without them the
# `deep_gemm` backend is `unsupported` on every row, so nothing is compared.
# `csrc/structs.h`'s `kDeepGemmSelectorPerfShapes` is the C++ mirror -- edit together.
#
# (n_rows, n_cols, seq_len): `seq_len` is the `deep_gemm` grid's window, this
# repo ranks the whole row, so the two are NOT numerically comparable.
DEEP_GEMM_SELECTOR_PERF_SHAPES = (
    [(b, 66551, 66551) for b in (1, 16, 132, 512)]                    # test-topk
    + [(b, 131072, s) for b in (1, 132, 256, 4096)                    # sglang
       for s in (2048, 4096, 16384, 65536)]
    + [(1, 107520, 107520), (16, 66551, 66551), (132, 107520, 107520),  # dsa
       (256, 107520, 107520), (4096, 107520, 107520)]
)
DEEP_GEMM_SELECTOR_PERF_TOPK = 2048


def deep_gemm_selector_perf_cases() -> List[TestParam]:
    """`DEEP_GEMM_SELECTOR_PERF_SHAPES` as `TestParam`s: fp32, `top_k=2048` -- all the
    `deep_gemm` backend serves (`topk <= 2048`, fp32 only)."""
    return [
        TestParam(b, v, DEEP_GEMM_SELECTOR_PERF_TOPK, False, False, False,
                  torch.float32, torch.int32, num_runs=10)
        for b, v, _seq in DEEP_GEMM_SELECTOR_PERF_SHAPES
    ]

# The `deep_gemm` grid's own correctness shapes (`SELECTOR_CORRECTNESS_SHAPES`):
# (n_rows, n_cols, top_k).  That grid declares `seq_lens` / `seq_starts` windows;
# this repository ranks the whole row instead -- valid here, deliberately not its
# semantics, and the shapes a `deep_gemm` column exists on at all.
DEEP_GEMM_SELECTOR_CORRECTNESS_SHAPES = (
    (16,  257,   31),     # deep_gemm `chunks` kernel
    (512, 65536, 2048),   # deep_gemm `coarse12` kernel
)


def deep_gemm_selector_correctness_cases() -> List[TestParam]:
    """`DEEP_GEMM_SELECTOR_CORRECTNESS_SHAPES` as `TestParam`s, fp32."""
    return [
        TestParam(b, v, k, False, False, False, torch.float32, torch.int32,
                  num_runs=0)
        for b, v, k in DEEP_GEMM_SELECTOR_CORRECTNESS_SHAPES
    ]


def performance_cases() -> List[TestParam]:
    """The official performance grid, `test.py`'s own case list.  A function so
    `scripts/perf_snapshot.py` drives *these* cases, not a transcription.

    `seed=-1` is left as-is, so a snapshot's data differs between runs exactly
    as two official runs do -- seed the cases yourself for an A/B."""
    return [
        # Lightning Indexer
        TestParam(b, compressed_seqlen, topk, False, False, False, torch.bfloat16, torch.int32, num_runs=10)
        for topk in [512, 1024]
        for b in [
            6,      # RL rollout
            256,    # Decoding
            512,
            768,
            4096    # Prefill
        ]
        for compressed_seqlen in [256, 1024, 4096, 16384, 65536, 131072, 262144, 524288, 1048576]
    ] + [
        # Sampler
        TestParam(b, vocab_size, 512, True, False, True, torch.float, torch.int64, num_runs=10)
        for b in [6, 256, 512, 768, 4096]
        for vocab_size in [129280]
    ]


if __name__ == '__main__':
    torch.set_default_device("cuda")

    parser = argparse.ArgumentParser()
    lib.stick_unit_test_args(parser)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default=None,
                        help="Only run testcases whose input dtype matches")
    parser.add_argument("--perf-only", action="store_true",
                        help="Only run performance testcases (num_runs > 0)")
    parser.add_argument("--backend", default="maca_c",
                        choices=["maca_c", "torch", "deep_gemm"],
                        help="which implementation every case runs: `maca_c` "
                             "(the kernel, the default), `torch` (the reference "
                             "implementation) or `deep_gemm`.  Passed to "
                             "`deep_select.topk` at the call site.")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed the case table before it is built, so the "
                             "same command draws the same shapes.  Unset = the "
                             "unseeded table, as upstream builds it.")
    parser.add_argument("--sample", type=int, default=0,
                        help="draw this many correctness cases instead of all "
                             "of them (the perf grid is always kept whole).  "
                             "0 or unset = the whole table.")
    args = parser.parse_args()

    import deep_select

    # An explicit `--backend deep_gemm` on a box that cannot run it is refused
    # before anything is measured: otherwise every case reports a failure and
    # the run reads as a kernel defect rather than a missing package.
    if args.backend == "deep_gemm" and not deep_select.deep_gemm_available():
        raise SystemExit(
            "tests/test.py: --backend deep_gemm was asked for, but `import "
            "deep_gemm` does not succeed here or the package does not carry "
            "`fp32_indexer_topk_selector`.  Nothing was run.")
    if args.seed is not None:
        random.seed(args.seed)

    correctness_cases = correctness_cases_()

    performance_cases = performance_cases()
    # The selector shapes ride along whenever the backend that answers them can
    # run here.  A package question, not a location one, and not a switch: no
    # flag, because an override would have to reach every runner that probes for
    # itself (measured: `run_bench.sh --no-deep-gemm-shapes` reported 95 cells
    # and ran 120).
    selector_cases = []
    if deep_select.deep_gemm_available():
        selector_cases = deep_gemm_selector_correctness_cases()
        performance_cases = performance_cases + deep_gemm_selector_perf_cases()

    # The selector shapes are prepended rather than drawn, as they were when a
    # separate driver held them: they are 2 cases out of 105,140, so sampling
    # would leave the one backend that answers them with nothing to compare.
    if args.sample:
        drawn = random.Random(args.seed).sample(
            correctness_cases, min(args.sample, len(correctness_cases)))
    else:
        drawn = correctness_cases
    testcases = selector_cases + drawn + performance_cases

    if args.dtype is not None:
        wanted_dtype = {"fp32": torch.float32, "bf16": torch.bfloat16}[args.dtype]
        testcases = [t for t in testcases if t.dtype == wanted_dtype]
    if args.perf_only:
        testcases = [t for t in testcases if t.num_runs > 0]

    # To sweep batch x sequence length instead of the grid above, build the same
    # list here (`num_runs=10`, `NormalFloatDistribution(lambda f: f*1000)`).

    # Every case ends in exactly one of these, and the run reaches its summary
    # whatever happened: which cases went wrong has to be readable, not buried
    # behind the first one.  `unsupported` is the backend being narrower than
    # the operator (a coverage gap, not a defect) and `skip` is the environment.
    status = {"pass": [], "check_fail": [], "crash": [], "skip": [], "unsupported": []}
    stopped_early = ""

    def context_alive() -> bool:
        """Whether the device can still run anything.

        A device-side fault does not raise once -- it poisons the CUDA context,
        so every later case reports the same error.  Left alone, a 105k-case
        table prints 105k identical crashes and calls that a result; this is how
        the loop tells "this case was bad" from "there is no device any more".
        """
        try:
            torch.cuda.synchronize()
            return True
        except Exception:
            return False

    for test_idx, test in enumerate(testcases):
        if test != testcases[0] and test.num_runs > 0 and not args.no_cooldown:
            time.sleep(0.2)
        print("================")
        print(f"[{test_idx+1:6d}/{len(testcases):6d}, {test_idx/len(testcases)*100:2.0f}%] ", end="")
        try:
            is_correct = run_testcase(test, backend=args.backend)
        except torch.cuda.OutOfMemoryError as exc:
            status["skip"].append((test, f"out of memory: {str(exc)[:120]}"))
            torch.cuda.empty_cache()
            print("    SKIPPED: out of memory", flush=True)
            continue
        except deep_select.UnsupportedByBackend as exc:
            status["unsupported"].append((test, str(exc)[:120]))
            print(f"    UNSUPPORTED: {str(exc)[:120]}", flush=True)
            continue
        except Exception as exc:
            status["crash"].append((test, f"{type(exc).__name__}: {str(exc)[:200]}"))
            print(f"    CRASHED: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            if not context_alive():
                stopped_early = (f"the device stopped responding at case "
                                 f"{test_idx + 1}/{len(testcases)}; stopping, "
                                 f"because every later case would report the "
                                 f"same fault")
                break
            if not args.run_to_finish:
                stopped_early = (f"stopping at the first crash; `-rf` runs the "
                                 f"whole table and reports every case")
                break
            continue
        status["pass" if is_correct else "check_fail"].append((test, ""))
        if not is_correct:
            print("    SELECTED WRONG", flush=True)
            if not args.run_to_finish:
                stopped_early = (f"stopping at the first case that selected "
                                 f"wrong; `-rf` runs the whole table and reports "
                                 f"every case")
                break

    print(f"\n{'=' * 64}")
    for name in ("pass", "check_fail", "crash", "skip", "unsupported"):
        print(f"  {name:<12} {len(status[name]):>7}")
    if stopped_early:
        print(f"  (run stopped early: {stopped_early})")
    for name in ("check_fail", "crash", "skip", "unsupported"):
        for test, why in status[name]:
            print(f"  {name:<12} {test}" + (f"\n{'':<15}[{why}]" if why else ""))
    # Every case that was attempted landed in exactly one status list, so this is
    # the count that ran -- which is not `len(testcases)` when the run stopped
    # early, and saying "of 125 run" after running one is a false report.
    attempted = sum(len(v) for v in status.values())
    if status["check_fail"] or status["crash"]:
        ran = (f"of {attempted} run" if attempted == len(testcases)
               else f"of {attempted} run of {len(testcases)} (stopped early)")
        print(f"\033[31m\033[1m{len(status['check_fail'])} case(s) selected wrong, "
              f"{len(status['crash'])} crashed, {ran}\033[0m")
        sys.exit(1)
    print(f"\033[32m\033[1mAll {len(status['pass'])} cases passed!\033[0m")
