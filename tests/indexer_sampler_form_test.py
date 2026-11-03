"""The indexer and sampler call forms, as their production callers write them.

This operator's two supported scenarios each have one production call site,
and each site's keyword set differs from the official table's:

* the DSA lightning indexer -- bf16 `p_sum` rows, a per-row `end`, a
  caller-allocated `output_idx`, an `output_idx_offset`, `-1` as the
  out-of-window fill, indices only;
* the sampler -- fp32 logprobs over the 129280-wide vocabulary, values and
  indices, ordered by value.

`tests/test.py` drives one call site with its own keyword set; this file
drives those two argument for argument, at the production shapes, and checks
the answers with the same `check_result` predicate the official table uses --
imported, not copied.

The indexer grid is batch 1 (decode), 8 and 256 (prefill chunks) over
vocabularies 4096 and 16384, with the three `end` regimes that site sees (the
full row, half a row, and a window shorter than `topk`, which is the `-1` fill
path); the sampler grid is batch 1, 8 and 64 with `topk` in {1, 5, 20, 512}.

    python tests/indexer_sampler_form_test.py                 # maca_c, device 0
    python tests/indexer_sampler_form_test.py --backend torch
    CUDA_VISIBLE_DEVICES=1 python tests/indexer_sampler_form_test.py

Standalone, like the rest of this tree's suites; exit status is the verdict.
"""
import argparse
import glob
import hashlib
import os
import sys

# The path rule the other runners use: the checkout answers when it has a
# built extension, the installed wheel otherwise.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if glob.glob(os.path.join(_REPO, "deep_select", "deep_select_maca*.so")):
    sys.path.insert(0, _REPO)
sys.path.insert(0, os.path.join(_REPO, "tests"))

import torch

torch.set_default_device("cuda")

import deep_select
import lib
import test as suite
from lib import TestParam

PASS = []
FAIL = []

# The production shapes both call sites run.
INDEX_TOPK = 512
SAMPLER_VOCAB = 129280


def check(name, cond, detail=""):
    if cond:
        PASS.append(name)
        print(f"  PASS  {name}")
    else:
        FAIL.append(name + (f" -- {detail}" if detail else ""))
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def section(title):
    print(f"\n== {title}")


def build_case(batch, vocab, topk, *, dtype, sorted_value, sorted_index,
               return_value, seed, idx_oob_fill_value):
    """A `TestParam`/`Testcase` pair whose input carries the aligned row stride.

    `lib.generate_testcase` sizes the backing allocation from
    `get_stride_requirement()` and slices it back to `vocab` columns, which is
    the same padded-row shape both call sites hand the operator.
    """
    p = TestParam(batch_size=batch, vocab_size=vocab, topk=topk,
                  sorted_value=sorted_value, sorted_index=sorted_index,
                  return_value=return_value, dtype=dtype,
                  out_idx_dtype=torch.int32,
                  idx_oob_fill_value=idx_oob_fill_value,
                  input_distrib=lib.NormalFloatDistribution(),
                  seed=seed)
    return p, lib.generate_testcase(p)


def indexer_end(batch, vocab, mode, seed):
    """The indexer's per-row window: whole row, half row, or shorter than `topk`."""
    if mode == "full":
        return torch.full((batch,), vocab, dtype=torch.int32, device="cuda")
    if mode == "half":
        return torch.full((batch,), vocab // 2, dtype=torch.int32, device="cuda")
    generator = torch.Generator(device="cuda").manual_seed(seed)
    return torch.randint(INDEX_TOPK // 2, INDEX_TOPK, (batch,),
                         dtype=torch.int32, device="cuda", generator=generator)


def indexer(backend):
    """The indexer's call: block selection over a variable-length row."""
    section(f"1. indexer call form ({backend})")
    for batch in (1, 8, 256):
        for vocab in (4096, 16384):
            for mode in ("full", "half", "lt_topk"):
                name = f"indexer b{batch} v{vocab} end={mode}"
                seed = 700 + batch + vocab
                p, t = build_case(
                    batch, vocab, INDEX_TOPK, dtype=torch.bfloat16,
                    sorted_value=False, sorted_index=True, return_value=False,
                    seed=seed, idx_oob_fill_value=-1)
                end = indexer_end(batch, vocab, mode, seed)
                generator = torch.Generator(device="cuda").manual_seed(seed + 1)
                offset = torch.randint(0, 4096, (batch,), dtype=torch.int32,
                                       device="cuda", generator=generator)
                # The caller owns this buffer; the sentinel shows if the kernel
                # wrote somewhere else.
                out_idx = torch.full((batch, INDEX_TOPK), -7,
                                     dtype=torch.int32, device="cuda")
                try:
                    values, indices = deep_select.topk(
                        t.input, INDEX_TOPK, sorted=False, end=end,
                        indices_type=torch.int32, sorted_index=True,
                        output_idx=out_idx, output_idx_offset=offset,
                        idx_oob_fill_value=-1, return_value=False,
                        backend=backend)
                except BaseException as exc:  # noqa: BLE001 - the contract is "answers"
                    check(name, False,
                          f"raised {type(exc).__name__}: {str(exc)[:160]}")
                    continue
                check(name + ": indices only, caller's buffer",
                      values is None and
                      (indices is out_idx or torch.equal(indices, out_idx)),
                      f"values={values!r}, same buffer={indices is out_idx}")
                t.end, t.output_idx_offset = end, offset
                check(name + ": official contract",
                      suite.check_result(p, t, None, out_idx))
                del t, out_idx


def sampler(backend):
    """The sampler's call: top logprobs over the full vocabulary."""
    section(f"2. sampler call form ({backend})")
    for batch in (1, 8, 64):
        for topk in (1, 5, 20, 512):
            name = f"sampler b{batch} k{topk}"
            p, t = build_case(
                batch, SAMPLER_VOCAB, topk, dtype=torch.float32,
                sorted_value=True, sorted_index=False, return_value=True,
                seed=710 + batch + topk, idx_oob_fill_value=2147483647)
            try:
                values, indices = deep_select.topk(
                    t.input, topk, True, indices_type=torch.int32,
                    backend=backend)
            except BaseException as exc:  # noqa: BLE001
                check(name, False,
                      f"raised {type(exc).__name__}: {str(exc)[:160]}")
                continue
            check(name + ": values and int32 indices",
                  values is not None and values.dtype == torch.float32
                  and indices.dtype == torch.int32
                  and tuple(indices.shape) == (batch, topk),
                  f"values={None if values is None else values.dtype}, "
                  f"indices={indices.dtype} {tuple(indices.shape)}")
            check(name + ": official contract",
                  suite.check_result(p, t, values, indices))
            del t, values, indices


def alignment(backend):
    """Both call sites allocate rows in whole 1024-byte units.

    The input requirement has to divide 1024: a stricter one would reject an
    allocation the caller is entitled to make.  The indexer's caller-owned
    `output_idx` is (batch, 512) int32, a 2048-byte row, so the output
    requirement has to divide that.
    """
    section("3. stride requirements against the callers' allocations")
    input_align, output_align = deep_select.get_stride_requirement()
    check("a 1024-byte input row satisfies the requirement",
          1024 % input_align == 0, f"input requirement {input_align} B")
    check("a 2048-byte output_idx row satisfies the requirement",
          2048 % output_align == 0, f"output requirement {output_align} B")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", default="maca_c",
                        choices=["maca_c", "torch"],
                        help="which implementation every case runs")
    parser.add_argument("--device", type=int, default=None,
                        help="CUDA_VISIBLE_DEVICES shorthand; set before torch "
                             "initialises a context")
    parser.add_argument("--sections", default="1,2,3")
    args = parser.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        torch.cuda.init()
    done = set(args.sections.split(","))
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}, "
          f"backend {args.backend}")
    # Which package answers and which artifact it resolves: a passing count is
    # only a receipt for the binary it names.
    from deep_select import _binding
    so = _binding.extension_path("deep_select_maca")
    md5 = hashlib.md5(open(so, "rb").read()).hexdigest()[:12] if so else "none"
    print(f"package  {os.path.dirname(os.path.abspath(deep_select.__file__))}")
    print(f"artifact {so or '(none resolves)'} md5 {md5}")
    if "1" in done:
        indexer(args.backend)
    if "2" in done:
        sampler(args.backend)
    if "3" in done:
        alignment(args.backend)
    print(f"\n{len(PASS)} passed, {len(FAIL)} failed")
    for failure in FAIL:
        print(f"  FAILED: {failure}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
