"""Can a DeepSelect topk call be captured into a CUDA graph, and replayed?

This is a *capture* question, not a speed question.  Stream capture records the
device work a call enqueues and nothing else: any host-side branch, host
allocation, or pointer read that the call performs at launch time is frozen at
the values it had the first time and silently replayed forever after.  The
operator has three routes chosen by a host predicate (`f32_chunks_applies`,
`f32_coarse12_applies`, `chunked_f32_applies`) over the *shapes*, plus a
process-wide grow-only scratch allocator (`ChunkedScratch`, raw `cudaMalloc`
inside the entry).  Both are exactly the kind of thing capture punishes, and
neither is visible from the python facade.

So: for each of a few cells covering the three routes, do an eager call, capture
the same call, replay it, and compare.  Then the case that catches a stale
pointer or a data-dependent branch -- overwrite the input **in place** with new
data, replay, and check against a fresh eager call on the new data.

Correctness goes through `tests/test.py:check_result`, the suite's own
predicate, rather than a hand-written comparator: a hand-written one is how a
harness artifact gets mistaken for a kernel bug (twice in this tree already --
see `chunks_arm_official.py`).  The replay-vs-eager comparison is the suite's
`check_is_bitwise_equal`, which is the right relation here: the same kernel on
the same bytes must give the same bytes, and "close enough" would hide a graph
that replayed a stale answer.
"""
import os, sys, traceback, torch
torch.set_default_device("cuda")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kernelkit as kk, deep_select, lib, test as suite
from lib import TestParam, NormalFloatDistribution

_REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def build(tag, b, V, k, end=False, si=True, rv=False, idx_dtype=torch.int32,
          distrib=None, seed=3):
    """One testcase, on the device, shaped like `chunks_arm_official.case`."""
    p = TestParam(batch_size=b, vocab_size=V, topk=k, sorted_value=False,
                  sorted_index=si, return_value=rv, dtype=torch.float32,
                  out_idx_dtype=idx_dtype, enable_end_position=end,
                  input_distrib=distrib or NormalFloatDistribution(), seed=seed,
                  idx_oob_fill_value=-2000000 + V)
    kk.set_random_seed(seed)
    t = lib.generate_testcase(p)
    t.input = t.input.to("cuda")
    if t.end is not None:
        t.end = t.end.to("cuda")
    return p, t


def call(p, t, k, si=True, rv=False, idx_dtype=torch.int32):
    """The call under test.  `output_idx`/`output_val` are left to the operator,
    which is what a real caller does and therefore what a graph would have to
    capture."""
    return deep_select.topk(t.input, k, sorted=False, begin=None, end=t.end,
                            indices_type=idx_dtype, sorted_index=si, hint=None,
                            output_idx=None, output_idx_offset=None,
                            idx_oob_fill_value=p.idx_oob_fill_value,
                            value_oob_fill_value=p.value_oob_fill_value,
                            return_value=rv, abort_when_nan_found=False,
                            backend="maca_c")


def describe(exc):
    """The first line of the error, plus the frames of our own code it passed
    through -- innermost last, so the last `our code` entry is the origin."""
    first = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    first = first.splitlines()[0] if first else type(exc).__name__
    frames = traceback.extract_tb(exc.__traceback__)
    ours = [f"{os.path.relpath(f.filename, _REPO)}:{f.lineno} in {f.name}"
            for f in frames if os.path.abspath(f.filename).startswith(_REPO + os.sep)]
    allf = [f"{f.filename}:{f.lineno} in {f.name}" for f in frames]
    return first, ours, allf


def capture(p, t, k, si, rv, idx_dtype):
    """Capture `call` into a fresh graph.  Returns (graph, val, idx) or raises."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            call(p, t, k, si=si, rv=rv, idx_dtype=idx_dtype)
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        val, idx = call(p, t, k, si=si, rv=rv, idx_dtype=idx_dtype)
    return g, val, idx


def _pair(b, V, k):
    """A bare (input, callable) pair for the probes, which do not need a
    `TestParam` -- they are about the scratch allocator, not about correctness
    against the suite's predicate."""
    import math
    al = deep_select.get_stride_requirement()[0] // 4
    vr = int(math.ceil(V / al)) * al
    x = torch.empty((b, vr), device="cuda")[:, :V].normal_()

    def fn():
        return deep_select.topk(x, k, backend="maca_c",
                                indices_type=torch.int32, sorted_index=False,
                                return_value=False, abort_when_nan_found=False)
    return x, fn


def capture_plain(fn):
    """`capture` for a bare callable, with the same stream isolation."""
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        out = fn()
    return g, out[0], out[1]


def cell(tag, b, V, k, end=False, si=True, rv=False, idx_dtype=torch.int32,
         distrib=None, seed=3):
    """Eager, capture, replay, replay-on-new-input.  One printed line per stage."""
    p, t = build(tag, b, V, k, end=end, si=si, rv=rv, idx_dtype=idx_dtype,
                 distrib=distrib, seed=seed)
    head = f"{tag:<34} b={b:<3} V={V:<7} k={k:<5} end={int(end)} si={int(si)}"

    # ── (a) eager ──────────────────────────────────────────────────────────
    val_e, idx_e = call(p, t, k, si=si, rv=rv, idx_dtype=idx_dtype)
    torch.cuda.synchronize()
    eager_ok = suite.check_result(p, t, val_e.clone() if val_e is not None else None,
                                  idx_e.clone())
    print(f"  {head}  eager={'PASS' if eager_ok else 'FAIL'}")

    # ── (b) capture ────────────────────────────────────────────────────────
    try:
        g, val_c, idx_c = capture(p, t, k, si, rv, idx_dtype)
        cap = "OK"
    except Exception as exc:
        first, ours, allf = describe(exc)
        print(f"  {head}  capture=RAISED  {first}")
        for fr in ours[-4:] or allf[-4:]:
            print(f"  {'':<34}    at {fr}")
        return dict(tag=tag, eager=eager_ok, captured=False, replay=False,
                    reinput=False, first=first, ours=ours)

    # ── (b') replay, same input ────────────────────────────────────────────
    try:
        g.replay()
        torch.cuda.synchronize()
        replayed_ok = suite.check_result(
            p, t, val_c.clone() if val_c is not None else None, idx_c.clone())
        replay_same = kk.check_is_bitwise_equal("replay idx vs eager idx", idx_c, idx_e)
        if val_c is not None and val_e is not None:
            replay_same &= kk.check_is_bitwise_equal("replay val vs eager val", val_c, val_e)
        err = None
    except Exception as exc:
        first, ours, allf = describe(exc)
        print(f"  {head}  replay=RAISED  {first}")
        for fr in ours[-4:] or allf[-4:]:
            print(f"  {'':<34}    at {fr}")
        replayed_ok, replay_same, err = False, False, first

    print(f"  {head}  capture=OK  replay={'MATCH' if replay_same else 'MISMATCH'}"
          f"  (official predicate on replay: {'PASS' if replayed_ok else 'FAIL'})")

    # ── (c) replay with DIFFERENT input, written in place ──────────────────
    # In place, so the captured pointer is still valid: this is the
    # data-dependent-branch probe, not the stale-pointer one.
    try:
        p.input_distrib.generate(t.input)
        torch.cuda.synchronize()
        val_e2, idx_e2 = call(p, t, k, si=si, rv=rv, idx_dtype=idx_dtype)
        torch.cuda.synchronize()
        eager2_ok = suite.check_result(
            p, t, val_e2.clone() if val_e2 is not None else None, idx_e2.clone())
        g.replay()
        torch.cuda.synchronize()
        reinput_same = kk.check_is_bitwise_equal("replay(new input) idx vs eager idx", idx_c, idx_e2)
        if val_c is not None and val_e2 is not None:
            reinput_same &= kk.check_is_bitwise_equal("replay(new input) val vs eager val", val_c, val_e2)
        reinput_ok = suite.check_result(
            p, t, val_c.clone() if val_c is not None else None, idx_c.clone())
        print(f"  {head}  replay(new input)={'MATCH' if reinput_same else 'MISMATCH'}"
              f"  (eager on new data: {'PASS' if eager2_ok else 'FAIL'};"
              f" official predicate on replay: {'PASS' if reinput_ok else 'FAIL'})")
    except Exception as exc:
        first, ours, allf = describe(exc)
        print(f"  {head}  replay(new input)=RAISED  {first}")
        for fr in ours[-4:] or allf[-4:]:
            print(f"  {'':<34}    at {fr}")
        reinput_same, reinput_ok = False, False

    return dict(tag=tag, eager=eager_ok, captured=True, replay=replay_same,
                reinput=reinput_same, first=err, ours=[])


# ── the cells: one per route ────────────────────────────────────────────────
# The route each one lands on is a fact about the C++ gates, not a label:
#   row      -- `V <= 32768` denies the f32 split (`kF32ChunkedMinVocab`), `b < 16`
#               denies coarse12's narrow arm, `b > 2` denies chunks.
#   coarse12 -- `b = 16 >= kF32Coarse12MinBatchesNarrow` at `V <= 131072`.
#   chunks   -- `b <= 2`, `V >= 2048`, `topk <= 2048`.
CELLS = [
    ("row/radix",  dict(b=8, V=32768, k=2048)),
    ("coarse12",   dict(b=16, V=66551, k=2048)),
    ("chunks",     dict(b=2, V=66551, k=2048)),
    ("chunks+end", dict(b=1, V=107520, k=2048, end=True)),
]

print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}  "
      f"sms={torch.cuda.get_device_properties(0).multi_processor_count}")
print(f"artifact={deep_select._binding.extension_path('deep_select_maca')}")
print()

results = [cell(tag, **kw) for tag, kw in CELLS]

# ── probe: capture a route whose scratch is NOT yet allocated ──────────────
# The four cells above each do an eager call first, which warms the
# process-wide grow-only scratch (`ChunkedScratch`) and hides the entry's
# `cudaMalloc`.  A cold probe is wider than any of them, so the chunks
# workspace and the coarse12 column buffer must grow -- and it is captured with
# no eager call, so that growth happens *inside* capture if it happens at all.
# Not part of the three required checks; it is here because it separates "the
# route is graph-unsafe" from "the allocator is".
#
# **This probe is also its own process**, and it is only meaningful if nothing
# before it has grown the scratch.  Measured: `capture=OK`.  The growth did not
# land inside the capture window here because the probe's own warmup calls (the
# three inside `capture`) reached the `cudaMalloc` first.  That is a fact about
# the order this file calls things in, not a guarantee: a caller whose first
# call at a shape *is* the capture call has a raw `cudaMalloc` inside the
# capture region.  The probe that does not depend on ordering is the next one.
def cold_probe():
    p, t = build("cold-chunks", b=2, V=131072, k=2048)
    try:
        g, val_c, idx_c = capture(p, t, 2048, True, False, torch.int32)
        g.replay()
        torch.cuda.synchronize()
        return "OK", None
    except Exception as exc:
        return "RAISED", describe(exc)[0]


if __name__ == "__main__" and "--probe" in sys.argv:
    print("probe: cold scratch, first chunks call in the process is inside capture")
    state, first = cold_probe()
    print(f"  cold-scratch  capture={state}" + (f"  {first}" if first else ""))
    sys.exit(0)

# ── probe: grow the scratch AFTER capture, then replay ─────────────────────
# The stale-pointer probe, and the one case the four cells cannot reach: they
# all capture a shape they already warmed, and none of them calls a *larger*
# shape afterwards.  `ChunkedScratch` is grow-only (`maca_topk.cu:2003-2011`:
# `if (need_cols > scratch.coarse12_cols_count) { cudaMalloc; cudaFree(old); }`),
# so a later wider call frees the buffer the captured graph recorded -- and the
# graph goes on reading that address.
#
# **Measured: MISMATCH, deterministically.**  `replay=MISMATCH`, ~262k of 270336
# indices differ, reproduced three times.  This is the hazard realised, not a
# theoretical one.
#
# The first version of this probe *passed*, because it compared replay against
# an eager call that had itself re-warmed the scratch at the same shape -- so
# both sides read the same (stale but intact) buffer and agreed.  The eager
# reference has to be taken on data the replay was never run against, which is
# what `xs.normal_()` before `small()` below does: the graph re-reads the
# buffer it recorded, the eager call reads the one the allocator handed back
# for the same shape afterwards.  A comparator that lets the two agree by
# accident is the same failure this repository has now made three times.
def grow_probe():
    xs, small = _pair(132, 66551, 2048)
    small(); torch.cuda.synchronize()
    g, val_c, idx_c = capture_plain(small)
    g.replay(); torch.cuda.synchronize()
    xb, big = _pair(4096, 131072, 2048)
    big(); torch.cuda.synchronize()          # grows -> cudaFree(captured buffer)
    xs.normal_(); torch.cuda.synchronize()
    # Take the reference FIRST, on the new data, so `small()` here is what
    # re-allocates the same shape's scratch -- and then replay reads the
    # address the graph froze, not the one this call just used.
    val_e, idx_e = small()
    torch.cuda.synchronize()
    g.replay(); torch.cuda.synchronize()
    return bool(kk.check_is_bitwise_equal("grow-probe replay vs eager", idx_c, idx_e))


if __name__ == "__main__" and "--grow-probe" in sys.argv:
    print("probe: capture small, grow the scratch with a wider call, then replay")
    ok = grow_probe()
    print(f"  grow-scratch  replay={'MATCH' if ok else 'MISMATCH'}  "
          f"(a MATCH means the freed page was not reused, not that this is safe)")
    sys.exit(0)

print()
print("=" * 78)
print(f"{'cell':<16}{'eager':<8}{'capture':<10}{'replay':<10}{'replay(new input)':<18}")
for r in results:
    print(f"{r['tag']:<16}{'PASS' if r['eager'] else 'FAIL':<8}"
          f"{'OK' if r['captured'] else 'RAISED':<10}"
          f"{'MATCH' if r['replay'] else 'MISMATCH':<10}"
          f"{'MATCH' if r['reinput'] else 'MISMATCH':<18}")
n_cap = sum(r['captured'] for r in results)
n_rep = sum(bool(r['replay']) for r in results)
n_rei = sum(bool(r['reinput']) for r in results)
print(f"\ncaptured {n_cap}/{len(results)} cells; replay matched {n_rep}/{n_cap}; "
      f"replay-on-new-input matched {n_rei}/{n_cap}")
print("cold-scratch probe is a separate process (the scratch is process-wide): "
      "python3 tests/cases/graph_capture.py --probe")
print("ALL PASS" if (n_cap == len(results) and n_rep == n_cap and n_rei == n_cap)
      else "SOME FAILED")
