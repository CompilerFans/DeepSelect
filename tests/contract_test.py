"""The contract and its refusals -- what the official suite does not check.

`tests/test.py` builds the official table and asserts on *values*: index range,
uniqueness, `value == input[idx]`, the definitional `min(selected) >=
max(unselected)`, the orderings and the NaN guard.  It never constructs a call
the operator is supposed to refuse, and its official call site passes `begin`,
`hint` and caller-allocated buffers as `None`.  Both gaps are recorded in
CLAUDE.md ("Not covered by either suite, recorded rather than papered over").

This file is that coverage, as a standalone runner rather than a pytest suite
(this tree has none):

    python tests/contract_test.py                              # maca_c, device 0
    python tests/contract_test.py --backend torch              # the reference
    CUDA_VISIBLE_DEVICES=1 python tests/contract_test.py

Which package is tested follows the rule `run_bench.sh` and
`tools/sample_gate.py` use: the checkout answers when it has a built extension,
the installed wheel otherwise -- so `python <repo>/tests/contract_test.py` from
anywhere is an acceptance test for a delivered wheel, not just for a develop
tree.

Sections:

  1. `refusals`   -- every `DS_HOST_CHECK` in `maca_topk.cu` plus the two
                     interface-level asserts and the backend-name check, run
                     against the backend under test (the two word the same
                     rejection differently).  A refused call must raise, and
                     raise the *reason*, because a silent acceptance is a wrong
                     answer somewhere downstream.
  2. `semantics`  -- `end` windows and their fill values, `output_idx_offset`
                     (and that the out-of-band fill is not offset), the two
                     orderings, caller-allocated `output_idx`, `check_nan`'s two
                     dispositions, and the default backend's resolution.
  3. `shapes`     -- the coarse12 route's boundaries: a row shorter than a
                     float4, a row with a prefix (length % 4 != 0) and a tail,
                     the vector region's partial last iteration, `topk` at the
                     public ceiling.  fp32 cells are compared elementwise
                     against `backend="torch"` with `sorted_index` -- the one
                     differential this project documents as meaningful
                     (CLAUDE.md, "What backend=torch is"); bf16 is checked
                     against the contract only, because ties make set agreement
                     meaningless there.

Exit status is the verdict: 0 only when every case passes.
"""
import argparse
import glob
import hashlib
import os
import subprocess
import sys

# The path rule `run_bench.sh` and `tools/sample_gate.py` use: the checkout
# answers only when it has an extension to answer with, so a tree with none
# tests the installed wheel -- the package a deployment would actually import.
_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if glob.glob(os.path.join(_REPO, "deep_select", "deep_select_maca*.so")):
    sys.path.insert(0, _REPO)

import torch

import deep_select

FAIL = []
PASS = [0]
IDX_OOB_FILL = 2147483647


def check(name, cond, detail=""):
    if cond:
        PASS[0] += 1
        print(f"  PASS  {name}")
    else:
        FAIL.append(name + (f" -- {detail}" if detail else ""))
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def expect_raise(name, fn, needle, kind=None):
    """A refused call has to raise, and say why.

    `needle` matches the exception text: `DS_HOST_CHECK` composes
    `__FILE_NAME__:__LINE__: message`, so the message is the part a caller can
    act on and the file:line is the part a maintainer can.  `kind` names an
    exception class when the refusal is a Python `assert` and has no message.
    """
    try:
        fn()
    except BaseException as exc:  # noqa: BLE001 - the contract is "raises"
        text = str(exc)
        ok = (needle in text) if needle is not None else True
        if kind is not None:
            ok = ok and isinstance(exc, kind)
        check(name, ok, f"raised {type(exc).__name__}: {text[:160]}")
        return
    check(name, False, "did not raise")


def row_major(b, v, dtype, stride_elems=None, seed=0):
    """A (b, v) row-major tensor whose row stride is `stride_elems` (>= v).

    The operator requires `stride(0) * itemsize` to be a multiple of 1024 B, so
    a ragged `v` is served from a padded allocation and sliced: the slice keeps
    the padded row stride and `stride(1) == 1`.
    """
    g = torch.Generator(device="cpu").manual_seed(seed)
    # The default pad is the smallest stride the operator accepts, so every
    # non-refusal case in this file is a call it is supposed to serve.
    pad = stride_elems if stride_elems is not None else min_row_stride(v, dtype)
    backing = torch.randn((b, pad), generator=g, dtype=torch.float32)
    return backing.to(dtype).to("cuda")[:, :v]


def min_row_stride(v, dtype):
    """The smallest row stride >= v the 1024-byte rule accepts."""
    items = torch.tensor([], dtype=dtype).element_size()
    per = 1024 // items
    return (v + per - 1) // per * per


def contract_holds(x, k, out_idx, out_val, end=None, offset=None):
    """The operator's contract, row by row, with the fill path included.

    Real entries are `out_idx[:m] - offset`, `m = min(length, k)`; past `m` the
    row holds `idx_oob_fill_value`, and the fill is *not* offset.  The
    definitional check is exact under ties: the number of visible elements
    strictly greater than the smallest selected one must be below `k`.
    """
    b, v = x.shape
    lengths = (end.to(torch.int64) if end is not None
               else torch.full((b,), v, dtype=torch.int64, device=x.device))
    offs = (offset.to(torch.int64) if offset is not None
            else torch.zeros((b,), dtype=torch.int64, device=x.device))
    for r in range(b):
        n = int(lengths[r])
        m = min(n, k)
        row = out_idx[r][:k].to(torch.int64)
        real = row[:m] - offs[r]
        tail = row[m:]
        if tail.numel() and not bool((tail == IDX_OOB_FILL).all()):
            return f"row {r}: unfilled tail {tail.tolist()[:4]} (n={n}, k={k})"
        if m == 0:
            continue
        if bool((real < 0).any()) or bool((real >= n).any()):
            return f"row {r}: index out of range [0, {n})"
        if int(torch.unique(real).numel()) != m:
            return f"row {r}: duplicate index"
        sel = x[r][real]
        if out_val is not None:
            if not torch.equal(sel, out_val[r][:m]):
                return f"row {r}: value != input[index]"
        if n > k:
            greater = int((x[r][:n] > sel.min()).sum())
            if greater >= k:
                return f"row {r}: min(selected) < max(unselected)"
    return None


def expect_ok(name, x, k, backend, **kw):
    """Run one call and check the contract; returns (val, idx) or None."""
    try:
        val, idx = deep_select.topk(x, k, backend=backend, **kw)
    except BaseException as exc:  # noqa: BLE001
        check(name, False, f"raised {type(exc).__name__}: {str(exc)[:160]}")
        return None
    why = contract_holds(x, k, idx, val, kw.get("end"), kw.get("output_idx_offset"))
    check(name, why is None, why or "")
    return val, idx


def section(title):
    print(f"\n== {title}")


def refusals(backend):
    """Every refusal, against the backend named.

    **The two backends word the same rejection differently, and one rule is
    the kernel's alone.**  `torch` checks the contract in Python
    (`_backends.py`), so its messages name Python's spellings
    (`topk must be positive`, `indices_type must be ...`, `output_idx`), while
    `maca_c` is refused by `DS_HOST_CHECK` inside `maca_topk.cu`
    (`topk must > 0`, `output_index ...`).  Neither wording is the contract;
    the *refusal* is, so the needle follows the backend.  The 1024-byte row
    stride is a kernel requirement and nothing else's: `torch` serves a padded
    row, which this section asserts in its place.
    """
    section(f"1. refusals ({backend})")
    dtype = torch.float32
    x = row_major(2, 1024, dtype)
    k = 8
    kern = backend == "maca_c"

    def t(**kw):
        return lambda: deep_select.topk(x, k, backend=backend, **kw)

    def be(maca, torch_word):
        """The wording the chosen backend raises."""
        return maca if kern else torch_word

    expect_raise("input 1-D",
                 lambda: deep_select.topk(x[0], k, backend=backend),
                 "input must be 2-D")
    expect_raise("topk = 0",
                 lambda: deep_select.topk(x, 0, backend=backend),
                 be("topk must > 0", "topk must be positive"))
    expect_raise("topk > 4096",
                 lambda: deep_select.topk(x, 4097, backend=backend),
                 "topk must be <= 4096")
    expect_raise("sorted without return_value", t(sorted=True, return_value=False),
                 "`return_value` must be enabled")
    expect_raise("sorted_value with sorted_index", t(sorted=True, sorted_index=True),
                 "cannot be used at the same time")
    expect_raise("float64 input",
                 lambda: deep_select.topk(x.to(torch.float64), k, backend=backend),
                 "input dtype must be float32 or bfloat16")
    expect_raise("int32 input",
                 lambda: deep_select.topk(x.to(torch.int32), k, backend=backend),
                 "input dtype must be float32 or bfloat16")
    expect_raise("indices_type float32", t(indices_type=torch.float32),
                 be("output_index dtype must be int32 or int64",
                    "indices_type must be int32 or int64"))
    expect_raise("input.stride(1) != 1",
                 lambda: deep_select.topk(x.t().contiguous().t(), k, backend=backend),
                 "input.stride(1) must be 1")
    if kern:
        expect_raise("input.stride(0) not a multiple of 1024 B",
                     lambda: deep_select.topk(
                         row_major(2, 1000, dtype, stride_elems=1000), k,
                         backend=backend),
                     "must be a multiple of 1024 bytes")
    else:
        # No alignment rule in the reference: the same padded row is a call it
        # serves, and serving it correctly is what the rule costs the kernel.
        padded = row_major(2, 1000, dtype, stride_elems=1000)
        expect_ok("padded input.stride(0) is served", padded, k, backend)

    small = torch.empty((2, k - 1), dtype=torch.int32, device="cuda")
    expect_raise("undersized output_idx",
                 t(indices_type=torch.int32, output_idx=small),
                 "must be at least (batch_size, topk)")
    wide = torch.empty((2, 64), dtype=torch.int32, device="cuda")[:, ::2]
    expect_raise("output_idx.stride(1) != 1",
                 t(indices_type=torch.int32, output_idx=wide),
                 be("output_index.stride(1) must be 1",
                    "output_idx.stride(1) must be 1"))
    # `interface.py` asserts the dtype before it dispatches, so both backends
    # meet the same bare `AssertionError`.
    expect_raise("output_idx dtype != indices_type",
                 t(indices_type=torch.int64,
                   output_idx=torch.empty((2, k), dtype=torch.int32, device="cuda")),
                 None, AssertionError)

    bad_end = torch.zeros((3,), dtype=torch.int32, device="cuda")
    # `torch` reaches the same refusal through a torch broadcast and reports
    # it as one; the kernel's own check names the argument.
    expect_raise("end of the wrong length", t(end=bad_end),
                 be("must be a contiguous tensor of `batch_size` entries", None))
    expect_raise("begin is not supported",
                 t(begin=torch.zeros((2,), dtype=torch.int32, device="cuda")),
                 "`begin` is not supported now")
    expect_raise("hint is not supported",
                 t(hint=torch.zeros((2, k), dtype=torch.int32, device="cuda")),
                 "`hint` is not supported now")
    expect_raise("unknown backend",
                 lambda: deep_select.topk(x, k, backend="cuda"),
                 "Unsupported backend")


def semantics(backend):
    section(f"2. semantics ({backend})")
    b, v, k = 4, 4096, 32
    x = row_major(b, v, torch.float32, seed=1)

    # -- end: the window is [0, end), and the answer is topk over that prefix.
    end = torch.tensor([4096, 4095, 1024, 16], dtype=torch.int32, device="cuda")
    val, idx = expect_ok("end windows: contract", x, k, backend, end=end,
                         indices_type=torch.int32)
    if idx is not None:
        bad = [(r, int(idx[r].max()), int(end[r])) for r in range(b)
               if int(end[r]) > k and int(idx[r].max()) >= int(end[r])]
        check("end windows: every index inside [0, end)", not bad, str(bad[:3]))
        row3 = idx[3]
        check("end <= topk: real indices first",
              bool((row3[:16] < 16).all()), f"row3[:16]={row3[:16].tolist()[:6]}")
        check("end <= topk: the tail is idx_oob_fill_value",
              bool((row3[16:] == IDX_OOB_FILL).all()), f"row3[16:]={row3[16:].tolist()[:4]}")

    off = torch.tensor([0, 100, -50, 7], dtype=torch.int32, device="cuda")
    val2, idx2 = expect_ok("output_idx_offset: contract", x, k, backend, end=end,
                           output_idx_offset=off)
    if idx2 is not None:
        check("output_idx_offset: the fill is not offset",
              bool((idx2[3][16:] == IDX_OOB_FILL).all()),
              f"{idx2[3][16:].tolist()[:4]}")

    # -- the two orderings.  `sorted_index` is *ascending index order* -- the
    # selected set, emitted in column order -- not "sorted by value"; the
    # official suite's own assertion is `ans_topk_index[:, 1:] >=
    # ans_topk_index[:, :-1]` (`tests/test.py`, "sorted index").
    val3, idx3 = expect_ok("sorted_index: contract", x, k, backend, sorted_index=True)
    if idx3 is not None and val3 is not None:
        check("sorted_index: indices ascend",
              bool((idx3[:, 1:] >= idx3[:, :-1]).all()))
    val4, _ = expect_ok("sorted value: contract", x, k, backend, sorted=True)
    if val4 is not None:
        check("sorted value: descends", bool((val4[:, 1:] <= val4[:, :-1]).all()))

    # -- a caller-allocated buffer is written in place and handed back.
    mine = torch.empty((b, k), dtype=torch.int32, device="cuda")
    _, got = deep_select.topk(x, k, backend=backend, indices_type=torch.int32,
                              output_idx=mine)
    check("caller-allocated output_idx is the returned tensor", got is mine)
    _, fresh = deep_select.topk(x, k, backend=backend, indices_type=torch.int32)
    check("caller-allocated output_idx holds the same answer",
          bool(torch.equal(mine.sort().values, fresh.sort().values)))

    # -- return_value=False returns None first and still a correct index row.
    val5, idx5 = deep_select.topk(x, k, backend=backend, return_value=False,
                                  indices_type=torch.int32)
    check("return_value=False: value is None", val5 is None)
    if idx5 is not None:
        why = contract_holds(x, k, idx5, None)
        check("return_value=False: contract", why is None, why or "")

    # -- NaN dispositions.
    nanx = x.clone()
    nanx[1, 100] = float("nan")
    val6, idx6 = deep_select.topk(nanx, k, backend=backend, indices_type=torch.int32,
                                  abort_when_nan_found=False)
    check("NaN row: sentinel in output_idx[row, 0]",
          int(idx6[1, 0]) == 0x3F3F3F3F, f"{int(idx6[1, 0])}")
    check("NaN row: clean rows unaffected",
          int(idx6[0, 0]) != 0x3F3F3F3F and int(idx6[3, 0]) != 0x3F3F3F3F)
    val7, idx7 = deep_select.topk(nanx, k, backend=backend, indices_type=torch.int32,
                                  abort_when_nan_found=False, check_nan=False)
    check("check_nan=False: no sentinel, the NaN is ranked",
          int(idx7[1, 0]) != 0x3F3F3F3F)
    short_end = torch.tensor([4096, 16, 4096, 4096], dtype=torch.int32, device="cuda")
    val8, idx8 = deep_select.topk(nanx, k, backend=backend, indices_type=torch.int32,
                                  end=short_end, abort_when_nan_found=False)
    check("a row whose visible length <= topk is never NaN-checked",
          int(idx8[1, 0]) != 0x3F3F3F3F, f"{int(idx8[1, 0])}")
    # The same rule holds for a NaN *inside* such a row's window: with the row
    # no longer than `topk`, the whole window is the answer, so nothing is
    # scanned -- the kernel's `length <= topk` shortcut returns before its
    # scan, and both reference backends used to scan here as soon as another
    # row in the batch turned the scan on.
    nanx2 = x.clone()
    nanx2[1, 5] = float("nan")
    try:
        _, idx9 = deep_select.topk(nanx2, k, backend=backend,
                                   indices_type=torch.int32, end=short_end,
                                   abort_when_nan_found=True)
        check("a short row's in-window NaN is ranked, not flagged",
              int(idx9[1, 0]) != 0x3F3F3F3F, f"{int(idx9[1, 0])}")
    except BaseException as exc:  # noqa: BLE001
        check("a short row's in-window NaN is ranked, not flagged", False,
              f"raised {type(exc).__name__}: {str(exc)[:160]}")

    # -- the default backend is the process default, resolved per call.
    # `_default_backend` is the resolution the signature documents; it lives
    # in `interface`, not in the package's public surface.
    code = ("from deep_select.interface import _default_backend;"
            "print(_default_backend())")
    out = subprocess.run([sys.executable, "-c", code],
                         env={**os.environ, "DS_TOPK_BACKEND": "maca_c",
                              "PYTHONPATH": os.getcwd()},
                         capture_output=True, text=True)
    check("DS_TOPK_BACKEND=maca_c resolves the default",
          out.stdout.strip().endswith("maca_c"), out.stdout.strip() + out.stderr[-200:])
    out2 = subprocess.run([sys.executable, "-c", code],
                          env={**os.environ, "DS_TOPK_BACKEND": "not-a-backend",
                               "PYTHONPATH": os.getcwd()},
                          capture_output=True, text=True)
    check("an unrecognized DS_TOPK_BACKEND is ignored",
          out2.stdout.strip().endswith("torch"), out2.stdout.strip() + out2.stderr[-200:])


def shapes(backend):
    section(f"3. coarse12 boundaries ({backend})")
    # `v` around the vector region's edges: shorter than a float4, a prefix
    # (v % 4 != 0), an exact multiple of the thread count, and one that leaves
    # a partial last iteration in pass 1 (v/4 not a multiple of 512).
    cases = [
        (2, 3, 1), (2, 4, 2), (2, 5, 2), (2, 7, 3), (2, 8, 4), (2, 9, 5),
        (2, 1023, 8), (2, 1024, 8), (2, 1025, 8),
        (2, 2048, 2048),                       # topk == the visible length
        (2, 4096, 4096),                       # topk == the public ceiling
        (2, 4099, 512),                        # prefix + tail + partial iteration
        (2, 8191, 1024),
        (1, 129283, 512),                      # v % 4 != 0 at the Sampler width
        (1, 131072, 2048),                     # the selector shape, fp32
    ]
    for b, v, k in cases:
        if k > v:
            continue
        x = row_major(b, v, torch.float32, seed=v + k)
        name = f"fp32 b{b} v{v} k{k}"
        val, idx = expect_ok(name + ": contract", x, k, backend,
                             indices_type=torch.int32, sorted_index=True)
        if idx is None:
            continue
        ref_val, ref_idx = deep_select.topk(x, k, backend="torch",
                                            indices_type=torch.int32, sorted_index=True)
        same = bool(torch.equal(idx.to(torch.int64), ref_idx.to(torch.int64)))
        check(name + ": elementwise vs torch", same,
              "" if same else f"{int((idx != ref_idx).sum())} of {idx.numel()} differ")

    # bf16 is checked against the contract only: a k-th value can tie, and
    # which member of a tie is selected is unspecified (CLAUDE.md).
    for b, v, k in [(2, 8192, 512), (2, 8195, 512), (1, 65536, 2048)]:
        x = row_major(b, v, torch.bfloat16, seed=v)
        name = f"bf16 b{b} v{v} k{k}"
        expect_ok(name + ": contract", x, k, backend, indices_type=torch.int32)
        # bf16 + sorted_value: this kernel orders bf16 too (upstream's ported
        # kernel rejects it).  The arm stays on the kernel path because that is
        # the acceptance it pins; the reference backend here serves it as well.
        if backend == "maca_c":
            expect_ok(name + ": sorted value", x, k, backend, sorted=True,
                      indices_type=torch.int32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="maca_c",
                    help="maca_c (default) or torch; deep_gemm is a subset and "
                         "is not what these cases are about")
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES shorthand; set before torch "
                         "initialises a context")
    ap.add_argument("--sections", default="1,2,3")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
        torch.cuda.init()
    done = set(args.sections.split(","))
    print(f"torch {torch.__version__}, device {torch.cuda.get_device_name(0)}, "
          f"backend {args.backend}")
    # Which package answered and which artifact it resolves -- the receipt a
    # passing count is worthless without (same as `tools/sample_gate.py`).
    from deep_select import _binding
    so = _binding.extension_path("deep_select_maca")
    md5 = hashlib.md5(open(so, "rb").read()).hexdigest()[:12] if so else "none"
    print(f"package  {os.path.dirname(os.path.abspath(deep_select.__file__))}")
    print(f"artifact {so or '(none resolves)'} md5 {md5}")
    if "1" in done:
        refusals(args.backend)
    if "2" in done:
        semantics(args.backend)
    if "3" in done:
        shapes(args.backend)
    print(f"\n{PASS[0]} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
