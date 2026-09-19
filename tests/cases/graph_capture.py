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
# **This file measures the checkout, not the installed wheel.**  It puts the
# *repository* on `sys.path` (below), which is the only way a `deep_select`
# import can resolve to a tree that also carries `kernelkit` and `lib` -- the
# two modules this case is built on.  That is deliberate, and it is also the
# trap: a bare `python tests/cases/foo.py` run from the repository root gets
# `.` on `sys.path` from the interpreter, and with the package importable from
# site-packages the name can resolve there instead.  Run these with
# `PYTHONPATH=.` and read the `artifact=` line every one of them prints, or
# `run_bench.sh`, which resolves the extension the same way and records its
# md5 in the run header.
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


def _pair(b, V, k, si=False):
    """A bare (input, callable) pair for the probes, which do not need a
    `TestParam` -- they are about the scratch allocator, not about correctness
    against the suite's predicate.

    `si=True` is what makes a replay comparison bitwise: the unsorted path
    writes its indices in atomic order, so two calls on the same bytes are not
    bitwise equal (measured: 261k of 270336 slots differ) and only a multiset
    comparison is fair.  The four cells above run sorted for that reason; the
    probe runs sorted too, because its whole question is whether the graph read
    the same thing twice."""
    import math
    al = deep_select.get_stride_requirement()[0] // 4
    vr = int(math.ceil(V / al)) * al
    x = torch.empty((b, vr), device="cuda")[:, :V].normal_()

    def fn():
        return deep_select.topk(x, k, backend="maca_c",
                                indices_type=torch.int32, sorted_index=si,
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

# ── probe: call another shape AFTER capture, then replay ───────────────────
# The probe the four cells cannot reach: they all capture a shape they already
# warmed and then stay at it.  A captured graph freezes every *host* decision the
# call made -- the route predicate, the scratch pointer, the `end` table pointer
# -- so the question this asks is whether a later call at another shape can move
# what the graph recorded out from under it.
#
# Two hazards were found here, and both are fixed:
#
#   1. **The scratch was freed on growth.**  `ChunkedScratch` was grow-only with
#      a `cudaFree` of the old buffer, and the graph had recorded that buffer's
#      address.  Fixed by *retiring* the old buffer (`scratch_retire`) instead of
#      freeing it -- the peak is held, which is what the cache already does for a
#      shape that never shrinks.
#   2. **The row table was rewritten in place.**  One `lengths` buffer plus an
#      epoch keyed on `(batches, vocab_size)` meant a call at another shape wrote
#      the *same address* with different lengths, and the graph re-read it: a
#      `b132 V66551` capture replayed after a `b16 V131072` call ranked a
#      131072-wide window against a 66551-wide row, **131 of 132 rows wrong**.
#      Fixed by keying the table on its shape (`ChunkedScratch::lengths_table`),
#      so a key never reuses another key's address.
#
# **What this probe can and cannot see.**  It compares *multisets*, because the
# `return_value=False` path writes its indices in atomic order and two eager
# calls on the same bytes are not bitwise equal either (measured: the control
# below differs in 261k of 270336 slots, exactly as the probe does).  A bitwise
# comparator here would be red on a correct kernel; a multiset comparator is
# still sharp enough for both hazards, because both corrupt *which columns are
# selected* -- hazard 2 turned 131 rows into a different window's answer, and a
# freed-and-reused scratch turns rows into whatever the allocator left there.
# What it would miss is a reordering-only corruption, which is what
# `sorted_index=True` exists to rule out (the four cells above run that mode and
# do compare bitwise).
def grow_probe():
    """Every intervening shape that has a distinct table, then a replay.

    The shapes are not arbitrary.  The second hazard above is keyed on
    `(batches, vocab_size)`, so a probe that changes only one of the two -- or
    changes neither, like the shipped one did -- cannot see it.

    **No eager call at the captured shape may run between the intervening call
    and the replay.**  That is not a detail; it is the difference between a
    probe that works and one that cannot.  Under the old policy the eager call
    *is* the repair: `small()` refills the one table back to `V=66551`, so a
    probe that takes its reference first (as the shipped version did) hands the
    graph a correct table and goes green on a broken kernel.  Measured: the
    version with the reference taken first reports MATCH on the pre-fix build.

    So the reference is the graph's own first replay, on input that is never
    written to again, and the comparison is bitwise.  `sorted_index=True` is
    what makes bitwise the right relation here: the same kernel on the same
    bytes must give the same bytes, and the unsorted path's atomic write order
    would not.
    """
    xs, small = _pair(132, 66551, 2048, si=True)
    small(); torch.cuda.synchronize()
    g, val_c, idx_c = capture_plain(small)
    g.replay(); torch.cuda.synchronize()
    ref = idx_c.clone()
    ok = True
    for ib, iV in ((4096, 131072), (16, 131072), (1, 262144), (64, 131072)):
        xb, big = _pair(ib, iV, 2048, si=True)
        big(); torch.cuda.synchronize()      # the shape that used to move the table
        g.replay(); torch.cuda.synchronize()
        same = bool(kk.check_is_bitwise_equal(
            f"replay after b={ib} V={iV} vs the first replay", idx_c, ref))
        ok &= same
        print(f"    after an intervening b={ib} V={iV}: "
              f"{'MATCH' if same else 'MISMATCH'}")
        idx_c.copy_(ref)                     # each shape gets its own replay
    return ok


if __name__ == "__main__" and "--grow-probe" in sys.argv:
    print("probe: capture small, call another shape, then replay (bitwise)")
    ok = grow_probe()
    print(f"  grow-scratch  replay={'MATCH' if ok else 'MISMATCH'}")
    sys.exit(0 if ok else 1)

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
