"""Correctness check for the MACA-native top-K against torch.topk.

Compares gathered VALUES, not index sets: ties can be broken differently, so
index equality is not the right contract.  The operator promises:
  * values equal torch.topk's values (as a multiset)
  * value[i] == input[index[i]]            (the pair stays together)
  * every returned index is in range and unique
  * min(selected) >= max(unselected)       (a genuine top-k)
  * sorted_index emits ascending indices, sorted_value descending values
  * the padded tail carries the exact `idx_oob_fill_value` /
    `value_oob_fill_value` sentinels
  * a row whose visible length exceeds `topk` and contains a NaN gets
    0x3F3F3F3F at output_index[row, 0]

Padding is never gathered: `idx_oob_fill_value` is a huge/negative sentinel that
would fault the device (and read the wrong value), so every gather is restricted
to `n_valid = min(visible, topk)`, and the sentinel is asserted
position-by-position instead.  Several cases use a *non-default* sentinel
precisely so that a gather of the pad would be impossible to mistake.

`torch.topk` runs in its own process (see `_ref_topk`): on this MACA torch,
mixing a large `normal_()` fill with `torch.topk` in one process intermittently
aborts in `c10::cuda::SetDevice`, which is unrelated to this kernel.  The input
draw is a code snippet (`INPUT_GENERATORS`) that both sides execute against the
same seed, so the reference ranks exactly the values the kernel sees.
"""
import argparse
import json
import os
import subprocess
import sys

import torch

import deep_select


# Backend under test, set from `--backend`.  `maca_c` (the default) is the
# kernel this device has; `torch` drives `deep_select.interface.topk_torch`.
# Running the same table against both is the point: the table is the operator's
# contract, so an implementation that disagrees with it is wrong whichever one
# it is.
BACKEND = "maca_c"


def _topk(*args, **kwargs):
    """`deep_select.topk` against the backend selected on the command line."""
    return deep_select.topk(*args, backend=BACKEND, **kwargs)


# ── input draws ──────────────────────────────────────────────────────────────
# Each snippet runs against `buf` on the device, with only `torch`, `buf` and
# `topk` in scope -- the same three names exist in the parent and in the
# reference subprocess, so a snippet cannot drift into depending on one side.
INPUT_GENERATORS = {
    "normal": "buf.normal_(0, 1)",
    "wide": "buf.normal_(0, 1); buf.mul_(1000.0)",
    "narrow": "buf.normal_(0, 1); buf.mul_(1e-3)",
    # few distinct values -> every key byte below the first is one big tie
    "ties4": "buf.normal_(0, 1); buf.copy_(buf.mul(4).floor())",
    # one value only: the k-th element sits inside an unbounded tie group
    "equal": "buf.fill_(7.0)",
    # ties spanning the k boundary exactly: k-1 strictly larger, the rest equal
    "tie_split": "buf.fill_(7.0); buf[:, :topk - 1] = 9.0",
    # sign bit and byte-boundary coverage
    "signmix": "buf.normal_(0, 1); buf[:, buf.shape[1] // 2:] *= -1",
    "inf": ("buf.fill_(1.0); buf[:, 0::3] = float('inf');"
            " buf[:, 1::3] = float('-inf'); buf[:, 2::3] = 0.0"),
    # subnormal (fp32) / all-zero (bf16 underflows here): ftz must be off
    "denormal": "buf.normal_(0, 1); buf.mul_(1e-42)",
    "zeros": "buf.fill_(-0.0)",
}

# NaN bit patterns.  The bf16 value is the top half of the fp32 one.
NAN_PATTERNS = {
    torch.float32: {"nan_pos": 0x7F800001, "nan_neg": 0xFF800001,
                    "qnan_pos": 0x7FC00000, "qnan_neg": 0xFFC00000,
                    "nan_allones": 0x7FFFFFFF},
    torch.bfloat16: {"nan_pos": 0x7F81, "nan_neg": 0xFF81,
                     "qnan_pos": 0x7FC0, "qnan_neg": 0xFFC0,
                     "nan_allones": 0x7FFF},
}

DTYPE_NAMES = {torch.bfloat16: "bf16", torch.float32: "fp32"}
UINT_OF = {torch.float32: torch.uint32, torch.bfloat16: torch.uint16}


def _make(batch, vocab, dtype, seed, gen, topk, plant=()):
    """Draw the padded buffer the kernel reads; returns (view, vocab_rounded)."""
    stride_bytes = deep_select.get_stride_requirement()[0]
    align = stride_bytes // torch.empty((), dtype=dtype).element_size()
    vocab_rounded = ((vocab + align - 1) // align) * align
    # Generate the whole padded buffer on the device: the same values the
    # reference process sees for the first `vocab` columns, and nothing the
    # kernel can read for the pad.  (Filling via `copy_` from a CPU tensor
    # aborts in this MACA torch's SetDevice path.)
    torch.manual_seed(seed)
    buf = torch.empty((batch, vocab_rounded), dtype=dtype, device="cuda")
    exec(gen, {"torch": torch, "buf": buf, "topk": topk})
    for col, bits in plant:
        _plant(buf, col, bits, dtype)
    return buf[:, :vocab], vocab_rounded


def _plant(buf, col, bits, dtype):
    """Write the exact bit pattern `bits` into column `col` of every row."""
    buf[:, col] = torch.tensor([bits], dtype=UINT_OF[dtype]).view(dtype)


def _ref_topk(batch, vocab_rounded, ends, topk, dtype, seed, gen):
    """Per-row torch.topk values, in a fresh process.

    The buffer must be built the same way as `_make` (same padded shape, same
    generator, same seed) -- otherwise the reference ranks a different draw
    than the kernel sees.
    """
    code = f"""
import json, torch
torch.manual_seed({seed})
buf = torch.empty({batch}, {vocab_rounded}, dtype={dtype!r}, device="cuda")
topk = {topk}
{gen}
rows = []
for r, e in enumerate({list(ends)!r}):
    n = min(topk, e)
    if n:
        v, _ = torch.topk(buf[r, :e], n)
        rows.append(v.float().cpu().tolist())
    else:
        rows.append([])
print("RESULT" + json.dumps(rows))
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=os.environ.copy())
    for line in out.stdout.splitlines():
        if line.startswith("RESULT"):
            return json.loads(line[len("RESULT"):])
    print(out.stdout[-2000:], out.stderr[-2000:])
    raise RuntimeError("reference torch.topk subprocess failed")


# ── the check ────────────────────────────────────────────────────────────────
def check(name, batch, vocab, topk, dtype, idx_dtype, seed,
          gen="normal", sorted_index=False, return_value=True,
          sorted_value=False, end_len=None, alloc_output=False,
          output_offset=None, idx_fill=2147483647, value_fill=float("-inf"),
          out_pad=None):
    code = INPUT_GENERATORS[gen]
    x, vocab_rounded = _make(batch, vocab, dtype, seed, code, topk)
    visible = ([vocab] * batch if end_len is None
               else [end_len] * batch if isinstance(end_len, int)
               else list(end_len))
    end = (None if end_len is None
           else torch.tensor(visible, dtype=torch.int32, device="cuda"))
    off = (None if output_offset is None
           else torch.tensor(output_offset, dtype=torch.int32, device="cuda"))
    if alloc_output:
        # `out_pad` gives the caller buffer a padded row stride
        if out_pad is None:
            out_idx = torch.empty((batch, topk), dtype=idx_dtype, device="cuda")
        else:
            out_idx = torch.empty((batch, out_pad), dtype=idx_dtype,
                                  device="cuda")[:, :topk]
    else:
        out_idx = None
    val, idx = _topk(
        x, topk, sorted=sorted_value, end=end, indices_type=idx_dtype,
        sorted_index=sorted_index, return_value=return_value,
        output_idx=out_idx, output_idx_offset=off,
        idx_oob_fill_value=idx_fill, value_oob_fill_value=value_fill,
        abort_when_nan_found=False)
    torch.cuda.synchronize()

    n_valid = [min(v, topk) for v in visible]
    xc = x.float().cpu()
    idx64 = idx.to(torch.int64).cpu()
    if off is not None:
        idx64 = idx64 - off.to(torch.int64).unsqueeze(1).cpu()
    # Gather only the valid prefix of each row; the sentinel tail is asserted
    # separately (it is deliberately far outside the row).
    gathered = [xc[r].gather(0, idx64[r, :n_valid[r]]) if n_valid[r]
                else torch.empty(0) for r in range(batch)]
    ref = [torch.tensor(row).float() for row in
           _ref_topk(batch, vocab_rounded, visible, topk, dtype, seed, code)]

    # ── values as a multiset ────────────────────────────────────────────────
    vals_ok = all(bool(torch.equal(torch.sort(gathered[r]).values,
                                   torch.sort(ref[r]).values))
                  for r in range(batch))

    # ── value[i] pairs with index[i] (upstream's "topk gathered value") ─────
    pair_ok = True
    if return_value:
        for r in range(batch):
            pair_ok &= bool(torch.equal(val[r, :n_valid[r]].float().cpu(),
                                        gathered[r]))

    # ── indices in range and unique ─────────────────────────────────────────
    in_range = all(bool(((idx64[r, :n_valid[r]] >= 0)
                         & (idx64[r, :n_valid[r]] < visible[r])).all())
                   for r in range(batch))
    uniq = all(int(torch.unique(idx64[r, :n_valid[r]]).numel()) == n_valid[r]
               for r in range(batch))

    # ── a genuine top-k: every selected >= every unselected (visible only) ──
    topk_ok = True
    for r in range(batch):
        if not n_valid[r]:
            continue
        rest = xc[r, :visible[r]].clone()
        rest.scatter_(0, idx64[r, :n_valid[r]], float("-inf"))
        topk_ok &= bool(gathered[r].amin() >= rest.amax())

    # ── the padded tail carries the exact sentinels ─────────────────────────
    pad_ok = True
    for r in range(batch):
        if topk <= n_valid[r]:
            continue
        # `idx_oob_fill_value` crosses the FFI as an int in both directions, so
        # an int64 output carries it sign-extended -- not INT64_MAX.
        want_idx = int(torch.tensor(idx_fill, dtype=idx_dtype).item())
        pad_ok &= bool((idx64[r, n_valid[r]:] == want_idx).all())
        if return_value:
            # `value_oob_fill_value` reaches the kernel as a C++ float and is
            # converted from there, so route the expectation through fp32 too:
            # a direct double->bf16 cast could round differently.
            want_val = torch.full((topk - n_valid[r],), value_fill,
                                  dtype=torch.float32).to(dtype)
            pad_ok &= bool(torch.equal(val[r, n_valid[r]:].cpu(), want_val))

    rv_ok = (val is not None) == return_value
    si_ok = True
    if sorted_index:
        si_ok = all(bool((idx64[r, 1:n_valid[r]] >= idx64[r, :n_valid[r] - 1]).all())
                    if n_valid[r] > 1 else True for r in range(batch))
    sv_ok = True
    if sorted_value:
        sv_ok = all(bool((val[r, 1:n_valid[r]].float()
                          <= val[r, :n_valid[r] - 1].float()).all())
                    if n_valid[r] > 1 else True for r in range(batch))
        sv_ok &= pair_ok

    sub = dict(vals=vals_ok, pair=pair_ok, range=in_range, uniq=uniq,
               topk=topk_ok, pad=pad_ok, rv=rv_ok, si=si_ok, sv=sv_ok)
    ok = all(sub.values())
    desc = (f"b={batch:<4} v={vocab:<8} k={topk:<5} {DTYPE_NAMES[dtype]:<5} "
            f"{str(idx_dtype).split('.')[-1]:<7} {gen:<9} "
            f"si={int(sorted_index)} sv={int(sorted_value)} "
            f"rv={int(return_value)} "
            f"end={'ragged' if isinstance(end_len, list) else end_len or '-'} "
            f"off={'y' if off is not None else '-'} "
            f"alloc={int(alloc_output)}"
            f"{'' if out_pad is None else f'(pad{out_pad})'}")
    bad = " ".join(f"{k}=0" for k, v in sub.items() if not v)
    print(f"  {desc:<84} -> {'PASS' if ok else 'FAIL'}"
          f"{'' if ok else '  [' + bad + ']'}", flush=True)
    if not vals_ok:
        for r in range(batch):
            if not torch.equal(torch.sort(gathered[r]).values,
                               torch.sort(ref[r]).values):
                print(f"    {name} row {r} mine={torch.sort(gathered[r]).values[:5].tolist()}")
                print(f"    {name} row {r} ref ={torch.sort(ref[r]).values[:5].tolist()}")
                break
    return ok


# ── NaN checks (no torch.topk reference: NaN rows are out of contract) ──────
def check_nan(dtype, idx_dtype, patterns, end_len=None):
    """Every NaN encoding in `patterns` is planted in its own row.

    With `abort_when_nan_found=False` the operator must write the 0x3F3F3F3F
    guard to output_index[row, 0] for rows whose visible length exceeds `topk`.
    Rows at or below `topk` skip the check entirely, so only range/uniqueness
    is asserted there.
    """
    batch, vocab, topk = len(patterns), 4096, 512
    plant = [(10 + 20 * i, bits) for i, bits in enumerate(patterns)]
    x, _ = _make(batch, vocab, dtype, 7, INPUT_GENERATORS["normal"], topk,
                 plant=plant)
    end = (None if end_len is None else
           torch.full((batch,), end_len, dtype=torch.int32, device="cuda"))
    idx = torch.zeros((batch, topk), dtype=idx_dtype, device="cuda")
    _val, idx = _topk(x, topk, end=end, indices_type=idx_dtype,
                                 output_idx=idx, abort_when_nan_found=False)
    torch.cuda.synchronize()
    tag = (f"NaN {DTYPE_NAMES[dtype]:<5} {str(idx_dtype).split('.')[-1]:<7} "
           f"end={'-' if end_len is None else end_len:<4} n={batch}")
    if end_len is not None and end_len <= topk:
        nv = min(end_len, topk)
        ok = bool(((idx[:, :nv].to(torch.int64) >= 0)
                   & (idx[:, :nv].to(torch.int64) < end_len)).all())
        detail = "check skipped (window <= topk): indices in range"
    else:
        guarded = [int(idx[r, 0].item()) for r in range(batch)]
        miss = [i for i, g in enumerate(guarded) if g != 0x3F3F3F3F]
        ok = not miss
        detail = (f"guard on {batch}/{batch} rows" if ok else
                  f"guard MISSING on rows {miss} (patterns "
                  f"{[hex(patterns[i]) for i in miss]})")
    print(f"  {tag:<84} -> {'PASS' if ok else 'FAIL'}  [{detail}]", flush=True)
    return ok


def check_nan_abort(dtype, patterns, expected_abort=True):
    """`abort_when_nan_found=True` must abort instead of returning a result.

    The abort is a device-side trap, so it runs in its own process; the case
    passes when the "no abort" marker never reaches stdout (the sync raises, or
    the process dies first).  The marker must come *after* a sync -- the launch
    is asynchronous, so printing it right after the call proves nothing.
    """
    bits = list(patterns)
    code = f"""
import torch, deep_select
torch.manual_seed(7)
buf = torch.empty(1, 4096, dtype={dtype!r}, device="cuda")
buf.normal_(0, 1)
for i, b in enumerate({bits!r}):
    buf[:, 10 + 20 * i] = torch.tensor([b], dtype={UINT_OF[dtype]!r}).view({dtype!r})
deep_select.topk(buf, 512, abort_when_nan_found=True, backend={BACKEND!r})
torch.cuda.synchronize()
print("NO_ABORT")
"""
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, env=os.environ.copy())
    ok = ("NO_ABORT" not in out.stdout) == expected_abort
    print(f"  NaN {DTYPE_NAMES[dtype]:<5} abort_when_nan_found=True"
          f"{'':<34} -> {'PASS' if ok else 'FAIL'}  "
          f"[{'aborted' if 'NO_ABORT' not in out.stdout else 'returned a result'}]",
          flush=True)
    return ok


# ── contract rejections ─────────────────────────────────────────────────────
def check_rejections():
    """Calls the operator documents as invalid must raise, not corrupt."""
    x = torch.empty(2, 4096, dtype=torch.float32, device="cuda")
    x.normal_(0, 1)
    one_row = torch.zeros(2, dtype=torch.int32, device="cuda")
    wide = torch.empty(2, 2048, dtype=torch.int32, device="cuda")
    cases = [
        ("topk=0", lambda: _topk(x, 0)),
        ("topk=4097", lambda: _topk(x, 4097)),
        ("sorted_value with return_value=False",
         lambda: _topk(x, 16, sorted=True, return_value=False)),
        ("sorted_value with sorted_index",
         lambda: _topk(x, 16, sorted=True, sorted_index=True)),
        ("begin (unsupported)", lambda: _topk(x, 16, begin=one_row)),
        ("hint (unsupported)", lambda: _topk(x, 16, hint=one_row)),
        ("input.stride(1) != 1", lambda: _topk(x[:, ::2], 16)),
        ("value dtype fp16", lambda: _topk(x.half(), 16)),
        ("indices_type fp32",
         lambda: _topk(x, 16, indices_type=torch.float32)),
        # The kernel writes output_index[row, i] assuming stride(1) == 1, so a
        # strided view must be refused rather than silently scrambled
        # (upstream: csrc/api.cpp KU_CHECK_LAST_DIM_CONTIGUOUS).
        ("output_idx.stride(1) != 1",
         lambda: _topk(x, 512, output_idx=wide[:, ::2],
                                  indices_type=torch.int32)),
        ("output_idx.size(1) < topk",
         lambda: _topk(x, 512, output_idx=wide[:, :100],
                                  indices_type=torch.int32)),
    ]
    ok = True
    for label, fn in cases:
        try:
            fn()
            torch.cuda.synchronize()
            print(f"  reject {label:<40} -> FAIL  [accepted]", flush=True)
            ok = False
        except Exception as exc:  # noqa: BLE001
            msg = " ".join(str(exc).splitlines()) or type(exc).__name__
            print(f"  reject {label:<40} -> PASS  [{type(exc).__name__}: "
                  f"{msg[:60]}]", flush=True)
    return ok


# ── case table ─────────────────────────────────────────────────────────────
def C(name, group, batch, vocab, topk, dtype, idx_dtype, **kw):
    return dict(name=name, group=group, batch=batch, vocab=vocab, topk=topk,
                dtype=dtype, idx_dtype=idx_dtype, **kw)


BF16, FP32 = torch.bfloat16, torch.float32
I32, I64 = torch.int32, torch.int64

CASES = [
    # ── Lightning Indexer: bf16, topk <= 4096, batch/vocab 1..+inf ──────────
    C("bf16-small", "indexer", 8, 4096, 256, BF16, I32),
    C("bf16-core", "indexer", 8, 4096, 512, BF16, I32),
    C("bf16-si", "indexer", 8, 32768, 1024, BF16, I32, sorted_index=True),
    C("bf16-si-novalue", "indexer", 8, 32768, 512, BF16, I32,
      sorted_index=True, return_value=False),
    C("bf16-i64", "indexer", 8, 32768, 512, BF16, I64),
    C("bf16-novalue", "indexer", 4, 65536, 2048, BF16, I32,
      return_value=False),
    C("bf16-cluster-shape", "indexer", 4, 524288, 1024, BF16, I32),
    C("bf16-max-topk", "indexer", 2, 1048576, 4096, BF16, I32),
    C("bf16-vocab1", "indexer", 4, 1, 512, BF16, I32),
    C("bf16-batch256", "indexer", 256, 16384, 1024, BF16, I64),
    C("bf16-wide-range", "indexer", 4, 32768, 512, BF16, I32, gen="wide"),
    C("bf16-narrow-range", "indexer", 4, 32768, 512, BF16, I32,
      gen="narrow"),
    # ── Sampling: fp32, vocab ~128K ─────────────────────────────────────────
    C("fp32-sampler-sv", "sampler", 6, 129280, 512, FP32, I64,
      sorted_value=True),
    C("fp32-si", "sampler", 8, 4096, 512, FP32, I64, sorted_index=True),
    C("fp32-core", "sampler", 8, 32768, 1024, FP32, I32),
    C("fp32-sampler-novalue", "sampler", 6, 129280, 512, FP32, I32,
      return_value=False),
    C("fp32-sampler-si", "sampler", 6, 129280, 1024, FP32, I32,
      sorted_index=True),
    C("fp32-max-topk", "sampler", 4, 262144, 4096, FP32, I64),
    C("fp32-vocab-2^23", "sampler", 1, 1 << 23, 512, FP32, I32),
    # ── stride / alignment traps (element offsets vs byte offsets) ──────────
    # a padded input row stride makes an element/byte mix-up read a wrong row
    C("pad-bf16-vocab300", "stride", 8, 300, 100, BF16, I32),
    C("pad-fp32-vocab100", "stride", 8, 100, 5, FP32, I64),
    # topk not a multiple of the 32 B output alignment -> padded output rows
    C("pad-bf16-idx32B", "stride", 8, 4096, 5, BF16, I32),
    C("pad-bf16-val32B", "stride", 4, 4096, 100, BF16, I64),
    C("pad-fp32-both", "stride", 4, 4096, 1231, FP32, I32),
    # caller-owned output buffer (interface-allocated -> padded view)
    C("alloc-output", "stride", 8, 4096, 512, BF16, I32, alloc_output=True),
    C("alloc-padded", "stride", 4, 4096, 700, FP32, I64, alloc_output=True,
      out_pad=2048),
    # ── order preservation / tie truncation ────────────────────────────────
    C("tie-bf16-4vals", "ties", 4, 4096, 512, BF16, I32, gen="ties4"),
    # sorted_value on bfloat16: works on MACA, upstream rejects it as fp32-only
    C("bf16-sorted-value", "ties", 4, 16384, 512, BF16, I32, gen="ties4",
      sorted_value=True),
    C("tie-fp32-equal", "ties", 4, 4096, 512, FP32, I32, gen="equal"),
    C("tie-fp32-split", "ties", 2, 16384, 2333, FP32, I32, gen="tie_split"),
    C("tie-bf16-subnormal", "ties", 4, 32768, 1024, BF16, I32,
      gen="denormal"),
    C("tie-fp32-zeros", "ties", 4, 4096, 512, FP32, I64, gen="zeros"),
    C("order-signmix", "ties", 4, 8192, 512, FP32, I32, gen="signmix"),
    C("order-inf", "ties", 4, 8192, 1024, FP32, I32, gen="inf"),
    C("order-fp32-subnormal", "ties", 4, 8192, 512, FP32, I32,
      gen="denormal"),
    C("tie-fp32-split-offset", "ties", 4, 16384, 512, FP32, I32,
      gen="tie_split", output_offset=[5, -6, 0, 1 << 20]),
    # ── windows, offsets, non-default sentinels ────────────────────────────
    C("end-const", "window", 8, 32768, 256, BF16, I32, end_len=1000),
    C("end-ragged", "window", 6, 4096, 512, FP32, I32,
      end_len=[0, 1, 300, 1000, 4096, 4096]),
    C("end-ragged-i64", "window", 6, 8192, 512, FP32, I64,
      end_len=[7, 511, 512, 513, 8192, 4000], idx_fill=-2000000 + 8192),
    C("end-shortcut-eq", "window", 4, 4096, 512, FP32, I32, end_len=512),
    C("end-shortcut-lt", "window", 4, 4096, 512, FP32, I32, end_len=511),
    # The window being the whole answer and the row having to come back in
    # value order are independent: whichever exit produces the answer still
    # owes the caller `sorted_value`.  These four are that intersection -- a
    # window that ends on/under `topk`, taken with and without `end`, on both
    # dtypes (upstream's suite caught this on a `vocab < topk` shape, which is
    # the no-`end` case here).
    C("sv-shortcut-eq", "window", 4, 4096, 512, FP32, I32, end_len=512,
      sorted_value=True),
    C("sv-shortcut-lt", "window", 4, 4096, 512, FP32, I64, end_len=511,
      sorted_value=True),
    C("sv-vocab-under-k", "window", 4, 300, 512, FP32, I32,
      sorted_value=True),
    C("sv-vocab-under-k-bf16", "window", 4, 200, 512, BF16, I32,
      sorted_value=True),
    C("end-minimal-refine", "window", 4, 4096, 512, FP32, I32, end_len=513),
    C("end-zero", "window", 4, 2048, 512, FP32, I64, end_len=0),
    C("offset", "window", 4, 4096, 512, BF16, I32,
      output_offset=[0, 1, 2, 3]),
    C("offset-negative", "window", 4, 4096, 512, FP32, I32,
      output_offset=[-(2 ** 30), 2 ** 30 - 1, -7, 12345]),
    C("fills-nondefault", "window", 6, 8192, 512, FP32, I64,
      end_len=[100, 4000, 8192, 8192, 3, 512], idx_fill=-2000000 + 8192,
      value_fill=-1.25e30),
    C("fills-extreme", "window", 4, 4096, 512, BF16, I32,
      end_len=[10, 4096, 4096, 2], idx_fill=-2147483647,
      value_fill=-1234123412341234.0),
]


# ── performance ─────────────────────────────────────────────────────────────
#
# Two implementations, benchmarked as peers: `maca_c`, and `topk` -- raw
# `torch.topk`, the baseline this kernel exists to beat.  `topk` is
# deliberately not a `backend=` value: it implements no part of the contract
# (`end`, the out-of-band fills, `output_idx_offset`), so it is a fair
# comparison only where the two coincide -- no window, no offset, and a row at
# least as long as `topk` -- which is how every shape below is drawn.  It is
# the same guard upstream's perf block applies (`tests/test.py:147`).
#
# The timer is `tests/kernelkit/bench.py`'s kineto harness (L2 flushed, marker
# kernel around the measured range) -- the one upstream's `--perf-only` uses,
# so the numbers are comparable with it.
PERF_BACKENDS = ("maca_c", "topk")


# Byte accounting, as upstream's: the row that was read, plus the outputs that
# were written.  Stated here because "equivalent bandwidth" means nothing
# without it.
def _perf_bytes(shape):
    itemsize = torch.empty((), dtype=shape["dtype"]).element_size()
    idx_bytes = torch.empty((), dtype=shape["index_dtype"]).element_size()
    total = shape["batch"] * shape["vocab"] * itemsize
    if shape["return_value"]:
        total += shape["batch"] * shape["topk"] * itemsize
    return total + shape["batch"] * shape["topk"] * idx_bytes


# The two scenarios the operator serves, with the flags upstream's
# `performance_cases` gives them (`tests/test.py:225-242`): the Lightning
# Indexer (bfloat16, indices only) and the Sampler (float32, sorted values,
# int64 indices).
def _indexer_shapes(batches, seqlens, topks):
    return [dict(name=f"indexer-b{b}-s{seqlen}-k{k}", batch=b, vocab=seqlen,
                 topk=k, dtype=BF16, index_dtype=I32, return_value=False)
            for k in topks for b in batches for seqlen in seqlens]


def _sampler_shapes(batches, vocab=129280, topk=512):
    return [dict(name=f"sampler-b{b}", batch=b, vocab=vocab, topk=topk,
                 dtype=FP32, index_dtype=I64, return_value=True)
            for b in batches]


# Decode (small b) to prefill (b=4096), short rows to long, at both topk
# values -- a subset of upstream's 95-shape grid, which `--perf-full` runs.
PERF_SHAPES = (
    _indexer_shapes([6, 256, 4096], [4096, 65536, 262144, 1048576], [512])
    + _indexer_shapes([256, 4096], [65536, 262144], [1024])
    + _sampler_shapes([6, 256, 4096])
)

PERF_SHAPES_FULL = (
    _indexer_shapes([6, 256, 512, 768, 4096],
                    [256, 1024, 4096, 16384, 65536, 131072, 262144, 524288,
                     1048576],
                    [512, 1024])
    + _sampler_shapes([6, 256, 512, 768, 4096])
)


# ── cross-kernel comparison ─────────────────────────────────────────────────
#
# A third row for the one kernel outside this repository that solves the same
# problem on this platform: the host repository's topk selector
# (`deep_gemm.fp32_indexer_topk_selector`, kernel `topk_coarse12`).  It is worth
# measuring because it is the mature MACA implementation of exactly this
# operator, so it says what the shapes below should cost.
#
# It is not a `backend=` value and not a row in the table above, because the
# contract it implements is a strict subset: float32 only, `top_k <= 2048`,
# int32 indices relative to `seq_starts` with `-1` for the padding, no ordered
# output, no NaN contract, and values as a separate gather.  So this table gives
# all three implementations the sub-problem they *all* implement -- float32,
# no window, no offset, indices only, unsorted -- and says so in its header.  A
# row here is comparable to the row beside it; it is not the full contract, and
# for a shape with `sorted_value` above it is not the same work.
#
# `deep_gemm` is an optional dependency: this file is also run standalone (the
# repository ships as a submodule of that same host), so the table is skipped
# with a note when the import is not available.
PERF_CROSS_BACKENDS = ("maca_c", "deep_gemm", "topk")


def _fp32_shapes(batches, seqlens, topks=(512,)):
    """float32 and indices only -- the sub-problem the cross table compares on."""
    return [dict(name=f"fp32-b{b}-s{seqlen}-k{k}", batch=b, vocab=seqlen,
                 topk=k, dtype=FP32, index_dtype=I32, return_value=False)
            for k in topks for b in batches for seqlen in seqlens]


PERF_CROSS_SHAPES = (
    _fp32_shapes([256, 4096], [65536, 262144, 1048576])
    + _fp32_shapes([6, 256, 4096], [129280])
)


def _deep_gemm():
    """The host repository's module, or None when it is not importable."""
    try:
        import deep_gemm
    except ImportError:
        return None
    return deep_gemm


def _perf_launcher(shape, backend, scores):
    """A callable running one shape on one backend, for the timer to run."""
    if backend == "topk":
        def launch():
            return torch.topk(scores, shape["topk"], dim=1,
                              sorted=shape["return_value"])
    elif backend == "deep_gemm":
        # Through the backend, not around it: the row then measures what a
        # caller gets -- the selector plus the contract work this repository
        # adds on top of it (the NaN check, the fill/offset mapping), which is
        # what the row beside it does for `maca_c`.
        def launch():
            return deep_select.topk(scores, shape["topk"],
                                    return_value=shape["return_value"],
                                    indices_type=shape["index_dtype"],
                                    backend="deep_gemm")
    else:
        def launch():
            return deep_select.topk(
                scores, shape["topk"], sorted=shape["return_value"],
                indices_type=shape["index_dtype"],
                return_value=shape["return_value"], backend="maca_c")
    return launch


# What the profiler records for each backend's own kernels, matched
# case-insensitively.  `maca_c` is one kernel, named for what it does.
# `torch.topk` is not one thing here: it dispatches to the vendor's `mbtopk*`
# on long rows and to ATen's `gatherTopK` + `radixSortKVInPlace` on short ones,
# so the match is a set of stems rather than one substring.  Neither stem can
# match the benchmark's own traffic -- the L2 flush (`...FillFunctor<int>`) and
# the runtime's API events (`mcLaunchKernel`, `mcMemsetAsync`, ...), which the
# profiler also records.
_PERF_KERNEL_MATCH = {
    "maca_c": ("topk",),
    "topk": ("topk", "radixsort"),
    # `deep_gemm::indexer::detail::topk_coarse12<...>` / `topk_chunks<...>`.
    "deep_gemm": ("topk",),
}


def _perf_time_ms(launch, num_iters, backend):
    """Mean per-run time of `launch`'s kernels, in milliseconds.

    Kineto records everything between two markers, so the match is what
    separates the measured kernels from the benchmark's own traffic.  A
    several-name match is the op's span, not an error: both backends here can
    be more than one kernel.
    """
    try:
        from kernelkit import bench as kk_bench
    except ImportError:                  # `python -m tests.check_maca`
        from tests.kernelkit import bench as kk_bench
    result = kk_bench(launch, num_iters)
    keys = _PERF_KERNEL_MATCH[backend]
    names = [n for n in result.get_kernel_names()
             if any(k in n.lower() for k in keys)]
    if not names:
        raise RuntimeError(
            f"no kernel of {backend!r} matched {keys}; the profiler saw "
            f"{result.get_kernel_names()}")
    if len(names) == 1:
        return result.get_kernel_time(names[0]) * 1e3
    return result.get_e2e_time(names) * 1e3


def _perf_table(title, note, shapes, backends, num_iters, ratio_base):
    """Benchmark every (shape, backend) and print one row per pair.

    `ratio_base` names the backend the ratio column divides by; the column is
    left out when it is None.
    """
    print(f"\n{title}\n{note}\n", flush=True)
    head = (f"{'Shape':<30} {'Backend':>9} {'Latency(ms)':>12} {'BW(GB/s)':>10}"
            + (f" {'vs ' + ratio_base:>9}" if ratio_base else ""))
    print(head)
    print("-" * len(head), flush=True)

    failures = []
    for shape in shapes:
        moved = _perf_bytes(shape)
        # A fixed draw per shape, as the correctness table does; the padded
        # buffer keeps the row stride the kernel requires.
        scores, _ = _make(shape["batch"], shape["vocab"], shape["dtype"],
                          seed=shape["batch"] * 131 + shape["vocab"] * 7
                          + shape["topk"], gen=INPUT_GENERATORS["normal"],
                          topk=shape["topk"])
        times = {}
        for backend in backends:
            launch = _perf_launcher(shape, backend, scores)
            try:
                times[backend] = _perf_time_ms(launch, num_iters, backend)
            except Exception as exc:  # noqa: BLE001 - report the shape, keep going
                print(f"{shape['name']:<30} {backend:>9}   ERROR "
                      f"{type(exc).__name__}: {str(exc).splitlines()[0][:60]}",
                      flush=True)
                failures.append(f"{shape['name']}/{backend}")
        for backend in backends:
            if backend not in times:
                continue
            ms = times[backend]
            bw = moved / (ms * 1e-3) / 1e9
            ratio = "--"
            if ratio_base in times and backend != ratio_base:
                ratio = "%.2fx" % (times[ratio_base] / ms)
            print(f"{shape['name']:<30} {backend:>9} {ms:>12.4f} "
                  f"{bw:>10.1f} {ratio:>9}", flush=True)
        del scores
        torch.cuda.empty_cache()
    return failures


def run_perf(num_iters, full):
    shapes = PERF_SHAPES_FULL if full else PERF_SHAPES
    failures = _perf_table(
        f"performance: {' vs '.join(PERF_BACKENDS)} -- {len(shapes)} shapes x "
        f"{num_iters} iters, kineto timer, L2 flushed",
        "bytes moved = input row + index output"
        " (+ value output when the shape asks for one)",
        shapes, PERF_BACKENDS, num_iters, ratio_base="topk")

    if _deep_gemm() is None:
        print("\ncross-kernel table skipped: deep_gemm is not importable here",
              flush=True)
    else:
        failures += _perf_table(
            f"cross-kernel: {' vs '.join(PERF_CROSS_BACKENDS)} -- "
            f"{len(PERF_CROSS_SHAPES)} shapes x {num_iters} iters",
            "the sub-problem all three implement: float32, no window, no "
            "offset, indices only, unsorted\n(each row is one backend's whole "
            "cost, contract work included; the ratio is against maca_c, so a "
            "number above 1.00x is how far ahead that backend is)",
            PERF_CROSS_SHAPES, PERF_CROSS_BACKENDS, num_iters,
            ratio_base="maca_c")

    print(flush=True)
    if failures:
        print(f"{len(failures)} shape(s) could not be measured: {failures}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="run two cases only (smoke test)")
    ap.add_argument("--group", default=None, help="only run this group")
    ap.add_argument("--no-nan", action="store_true",
                    help="skip the NaN and rejection blocks")
    ap.add_argument("--list", action="store_true", help="list cases and exit")
    ap.add_argument("--backend", default="maca_c",
                    choices=["maca_c", "torch", "deep_gemm"],
                    help="implementation under test: 'maca_c' (default) is the "
                         "MACA kernel for this device, whichever that is; "
                         "'torch' is the reference implementation; "
                         "'deep_gemm' is the host repository's selector, which "
                         "serves a subset (cases outside it are reported as "
                         "gaps, not failures)")
    ap.add_argument("--perf", action="store_true",
                    help="benchmark instead of checking: maca_c against "
                         "torch.topk over the perf shapes (ignores --backend, "
                         "--group, --quick)")
    ap.add_argument("--perf-iters", type=int, default=10,
                    help="runs per measurement in --perf (default 10, as "
                         "upstream's performance_cases)")
    ap.add_argument("--perf-full", action="store_true",
                    help="--perf over upstream's whole performance grid "
                         "instead of the default subset")
    args = ap.parse_args()

    global BACKEND
    BACKEND = args.backend

    if args.list:
        for c in CASES:
            print(f"  {c['group']:<8} {c['name']}")
        return

    from deep_select._arch import native_target
    # `maca_c` resolves to the kernel built for this device, so name both: a
    # run that loaded some other extension should be visible in the header.
    print(f"device: {torch.cuda.get_device_name(0)}  "
          f"backend: {BACKEND}{' -> ' + native_target() if BACKEND == 'maca_c' else ''}",
          flush=True)

    # `interface.get_stride_requirement` falls back to a constant when no
    # kernel is built (so `backend="torch"` works on a bare machine).  A
    # constant that mirrors another value can drift from it, so whenever a
    # kernel IS present, assert the two agree -- the fallback stays a mirror
    # instead of quietly becoming a second source of truth.
    from deep_select.interface import _ALIGNMENT_REQUIREMENT_BYTES
    try:
        built = tuple(deep_select.interface._backend_for(
            native_target()).get_alignment_requirement())
    except RuntimeError:
        print("alignment contract: no kernel built for this device, "
              "fallback not checked", flush=True)
    else:
        if built != tuple(_ALIGNMENT_REQUIREMENT_BYTES):
            print(f"ALIGNMENT CONTRACT MISMATCH: the kernel says {built}, "
                  f"the fallback constant says {tuple(_ALIGNMENT_REQUIREMENT_BYTES)}")
            return 1
        print(f"alignment contract: kernel and fallback agree at {built}", flush=True)

    if args.perf:
        return run_perf(num_iters=args.perf_iters, full=args.perf_full)

    cases = CASES
    if args.group:
        cases = [c for c in cases if c["group"] == args.group]
    if args.quick:
        cases = cases[:2]

    failures, gaps = [], []

    def gap(label, exc):
        """A case outside this backend's contract is named, not failed.

        `UnsupportedByBackend` says the operator offers the case and the chosen
        implementation does not -- `deep_gemm` is float32-only, say.  That is a
        property of the backend, so it is counted and printed apart from the
        verdict; anything else that raises is still a failure.
        """
        gaps.append(label)
        print(f"  {label} -> GAP   [{str(exc).splitlines()[0][:80]}]", flush=True)
        return True

    def run_check(label, fn, *fn_args, **fn_kwargs):
        try:
            return fn(*fn_args, **fn_kwargs)
        except deep_select.UnsupportedByBackend as exc:
            return gap(label, exc)

    for case in cases:
        spec = dict(case)
        name, group = spec.pop("name"), spec.pop("group")
        spec["seed"] = (spec["batch"] * 131 + spec["vocab"] * 7
                        + spec["topk"]) % 10000
        try:
            ok = check(name=name, **spec)
        except deep_select.UnsupportedByBackend as exc:
            ok = gap(f"{group}/{name}", exc)
        except Exception as exc:  # noqa: BLE001 - report and keep going
            print(f"  {name} -> ERROR {type(exc).__name__}: "
                  f"{str(exc).splitlines()[0][:100]}", flush=True)
            ok = False
        if not ok:
            failures.append(f"{group}/{name}")

    if not args.no_nan and args.group in (None, "nan"):
        print(flush=True)
        for dtype in (FP32, BF16):
            pats = NAN_PATTERNS[dtype]
            for label, keys in [("signed", ["nan_pos", "nan_neg"]),
                                ("quiet", ["qnan_pos", "qnan_neg"]),
                                ("allones", ["nan_allones"])]:
                for idx_dtype in (I32, I64):
                    name = f"nan/{DTYPE_NAMES[dtype]}-{label}"
                    if not run_check(name, check_nan, dtype, idx_dtype,
                                     [pats[k] for k in keys]):
                        failures.append(name)
            name = f"nan/{DTYPE_NAMES[dtype]}-short-window"
            if not run_check(name, check_nan, dtype, I32,
                             [pats["nan_pos"], pats["nan_neg"]], end_len=100):
                failures.append(name)
            name = f"nan/{DTYPE_NAMES[dtype]}-abort"
            if not run_check(name, check_nan_abort, dtype,
                             [pats["nan_pos"], pats["nan_neg"]]):
                failures.append(name)

        print(flush=True)
        if not run_check("reject/contract", check_rejections):
            failures.append("reject/contract")

    print(flush=True)
    if gaps:
        print(f"{len(gaps)} case(s) outside backend {BACKEND!r}'s contract: "
              f"{gaps}")
    if failures:
        print(f"{len(failures)} check(s) FAILED: {failures}")
        sys.exit(1)
    print("all cases passed" if not gaps else
          f"all cases passed ({len(gaps)} gap(s) above)")


if __name__ == "__main__":
    main()
