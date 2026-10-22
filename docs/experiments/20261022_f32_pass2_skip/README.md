# Does coarse12's second pass have to read the whole row?

Measured 2026-10-22 on a `MetaX C600-U` (28 AP, `xcore1600`, MACA 3.0.1436.20261017,
torch 2.10.0+metax20261017.1120), device 1, one probe (`pass2_skip.cu`, self-contained,
`cucc -O3 --offload-arch=xcore1600 -shared`) driven by `pass2_skip_run.py` on the official
data (`tests/lib.py`, `dtype=float32`, `topk` as in the grid, `return_value=False`).
Five timed iterations after three warm-ups; the clock ramp is discarded by the warm-ups.

Pass 2 of `topk_coarse12_row` (`csrc/maca_kernels/common/topk_coarse12.cuh`) walks every
element of the row to act on the ~`k`+bin ones at or above the 12-bit threshold. At
`b4096 V131072 k2048` that is **2.6% of the row**. The 16-bit row already answers the same
question with a per-block max and a kept-block list (`radix_row_bf16.cuh`,
`docs/experiments/20261010_skim_rate/`); this asks whether the fp32 walk has the same lever.

## The two candidate shapes

| shape | what pass 2 does |
| --- | --- |
| dense (today) | every element, `bin >= threshold` tested per element |
| table-gated | walks `block_max[]`, skips a block whose max bin is under the threshold |
| compacted list | after the threshold, compacts kept blocks into a list; the walk strides that list, four lanes per block |

All three were measured against the same data with the same collect rule (above-threshold
→ `out` via one shared atomic; threshold-bin → arena + refine histogram). The lists and
tables were built by the driver with the *true* 12-bit threshold bin, so these numbers are
the walk, not the summary.

| arm (ms) | b4096 V131072 k2048 | b4096 V129280 k512 |
| --- | --- | --- |
| **dense collect** (today) | **2.782** | **2.165** |
| table-gated collect | 2.811 | 2.157 |
| table-gated load-only | 0.867 | 0.649 |
| listed G=16 collect | **1.482** | **0.514** |
| listed G=16 load-only | 0.864 | 0.313 |
| listed G=32 collect (52.1% / 18.1% kept) | 2.185 | 0.795 |
| listed G=64 collect (77.1% / 32.9% kept) | 3.040 | 1.262 |

Kept-block fraction at G=16: **30.8%** (flagship) and **9.5%** (k512 cell).

## Finding 1: the table saves the traffic, the list saves the time

The table-gated walk reads the same bytes the list walk does — its load-only arm is
identical to the list's (0.867 against 0.864, 0.649 against 0.313) — and still comes out at
the dense walk's time. The reason is divergence: with 30.8% of blocks kept, essentially
every 64-lane warp contains a kept block, so every warp executes the block body and the
*issue* cost stays at the dense walk's level while only the traffic falls. The compaction
removes that: after it, every entry in the walk is kept, four lanes share a block, and the
traffic saving is what the clock sees — 1.88x at the flagship and 4.21x on the k512 cell.

G=32 and G=64 are measured for the granularity trade: coarser blocks keep more of the row
(52.1% and 77.1%), and at G=64 the walk is slower than the dense one. G=16 is the choice.

This is the same conclusion `docs/experiments/20261010_skim_rate/` reached on the 16-bit
row ("a test in front of the walk measures at 0.45 of the full walk, the decoupled form at
0.14"), re-derived here for fp32 rather than inherited.

## Finding 2: the summary costs 0.72–0.83 ms in pass 1, and it is issue-bound

Pass 1's own marginal cost for writing the table (two xor-shuffle steps per 16-element
block, one uint16 store per four lanes):

| arm (ms) | b4096 V131072 k2048 | b4096 V129280 k512 |
| --- | --- | --- |
| pass 1, histogram only | 1.534 | 1.513 |
| pass 1 + summary (`__shfl_xor_sync`) | 2.257 | 2.337 |
| pass 1 + summary (`__builtin_mxc_bsm_bpermute`) | 2.270 | 2.339 |

Two things this rules out:

* **Not an occupancy cliff.** `--resource-usage` on the probe: 24 MT registers without the
  summary, 30 MT with (48 → 60 registers; the AP's 131072-register file holds four
  512-thread CTAs up to 64). The real kernel measured 50 MT before the change and 62 MT
  after, which is still two CTAs per SM (124 registers/thread at 512 threads).
* **Not the shuffle wrapper's index arithmetic.** The platform permute is one instruction
  per step against the wrapper's ~5, and it measures the same — 2.270 against 2.257.

What it is: pass 1 is **instruction-issue bound**, not bandwidth-bound. Its 1.534 ms is
1400 GB/s, which is the element-atomic walk's ceiling on this part (1348 GB/s measured with
`filter_stage.cu`), not the 1595 GB/s streaming read. On an issue-bound walk every
instruction added per element is paid at face value: ~2.5 extra instructions per element on
top of ~6 costs ~47%, which is what the table shows.

So the summary has to earn its cost against the walk it saves, and it does: on the flagship
`-1.30 (pass 2) + 0.72 (pass 1) = -0.58 ms` of a 4.7 ms kernel (**-12%**), and on the k512
cell `-1.65 + 0.82 = -0.83 ms` (**-19%**).

## The kernel, A/B'd

`tools`-free paired run: the two artifacts swapped in place, one process per
(arm, round), base then new, three rounds on device 1; the same two artifacts
were also run over the **whole fp32 grid** (`tests/test.py --perf-only --dtype
fp32 -nc`, two alternating rounds).  Kernel time is the sum over kernels whose
name contains `topk`, which is `tests/test.py`'s own rule.  Medians of three
rounds on the six cells below, where the spread across rounds was under 0.2%;
the grid totals are the two rounds' own sums.

| cell | base | new | delta | GB/s (base -> new) | kept blocks |
| --- | --- | --- | --- | --- | --- |
| b4096 V131072 k2048 (selector) | 5176.8 us | 4784.0 us | **-7.6%** | 415 -> 449 | 30.8% |
| b4096 V129280 k512 (Sampler) | 5070.4 us | 4395.3 us | **-13.3%** | 418 -> 482 | 9.5% |
| b4096 V107520 k2048 (dsa) | 4337.8 us | 4006.0 us | **-7.7%** | 406 -> 440 | 30.8% |
| b4096 V66551 k2048 (test-topk) | 2887.7 us | 2943.5 us | **+1.9%** | 378 -> 370 | 45.5% |
| b256 V131072 k2048 | 399.1 us | 366.0 us | **-8.3%** | 336 -> 367 | 30.8% |
| b6 V129280 k512 | 68.7 us | 69.0 us | +0.4% | 45.2 -> 45.0 | 9.5% |

GB/s is the operator's own traffic (`B*V*4`, one read; the outputs are 1.5% at
most) over the kernel time -- the repo's convention, and not the DRAM the two
passes actually move: at the flagship the row is read twice, but pass 2 now
touches only the 30.8% of blocks the list names, so the kernel's real read
traffic is ~1.31 rows.

Over the whole fp32 grid the two rounds sum to **36,340.1 -> 33,304.0 us
(-8.35%)** and **36,366.3 -> 33,317.0 us (-8.39%)**.  Per cell: -13.0..-13.2% on
`V129280 k512` (all batches from 256 up), -7.2..-7.5% on `b4096 V131072 k2048`,
-7.5..-7.7% on `b4096 V107520 k2048`, -7.9..-8.5% on `b256 V131072`, and
+1.0..+3.7% on the three `V66551 k2048` cells below `b4096`.  The `b1` cells
(93-110 us) move under 0.5% either way: at one row there is no list to amortise.

The kept fractions are `kept_frac.py`'s: a block is kept when its max key's
12-bit bin is at or above the k-th value's bin, which is exactly the test the
compaction applies.

**The one regression is the densest cell on the grid** — `V66551 k2048` keeps
45.5% of its blocks. The list is still used there (the gate is 75%) and still
beats the dense walk, but by less than the pass-1 summary costs: the same
mechanism that pays at 9.5% and 30.8% turns into a 1.9% tax. Both halves are
measured and the sign follows the density, which is the trade the gate cannot
remove — falling back to the dense walk would cost the summary *and* the walk,
so the cell would be worse, not better. It is left in and reported.

Small batches are where the absolute numbers are tiny either way: `b6` is
68.7 us and moves 0.3 us, so the `b256` row (-8.3%) is the one that says the
skip is not a large-batch artifact.

The probe's listed arms at `V66551` report count mismatches of 1-4 elements per
row; that is the probe, not the kernel — `pass2_skip.cu` was written around
16-element blocks and does not walk a ragged tail, while the shipping kernel
walks the sub-block remainder densely (`kBlockVecs`-granular and scalar). The
shipping kernel's own grid check covers the shape (30/30 fp32 perf cells, and
the 343/343 full gate recorded under "What shipped").

## On the official grid (the recording, 2026-10-22)

The shipped artifact was recorded by `./run_bench.sh` on the same pod, device 1, as
`perf_data/MetaX_C600-U/20261022_175424/` (commit `6b5928c`, artifact `6379e1cb`). All
the stages ran and passed in one session: snapshot 147/147 `maca_c` and 147/147 `torch`
(plus 25 `deep_gemm` selector rows), official 147/147, official_fp32 30/30, the
`--seed 20260911 --sample 400` correctness draw at 0 `check_fail` / 0 `crash` /
13 OOM `skip` over 549 cases, and the chunks/graph arms with both scratch probes.

Against the previous recording (`20261021_175631`, the 512-thread tree before the
list, same board and toolkit), kernel time, both currencies:

| cell (official shape) | before | after | ratio |
| --- | ---: | ---: | ---: |
| fp32 b4096 V131072 k2048 int32 | 5234.4 us / 416.7 GB/s | 4745.5 us / 459.6 GB/s | 0.907x |
| fp32 b4096 V107520 k2048 int32 | 4394.0 us / 408.6 GB/s | 3988.9 us / 450.0 GB/s | 0.908x |
| fp32 b4096 V129280 k512 sorted int64 | 5127.3 us / 418.0 GB/s | 4380.5 us / 489.3 GB/s | 0.854x |
| fp32 b256 V129280 k512 | 396.5 us / 337.8 GB/s | 332.4 us / 403.0 GB/s | 0.838x |

The fp32 column (17 cells) sums **18,999.4 -> 16,917.0 us (-10.96%)**, median ratio
0.907; 12 cells improved, ten of them by 9-19%. The four that moved the other way are
under 1.4% -- three b1/b16 cells where the kernel is launch-bound, and `b132 V66551`
(+1.4%), the densest cell and the density trade above. bf16 is the control (117 cells,
median 0.993): the change cannot route a bf16 call.

It reproduces the paired A/B on a second instrument and a second session, cell for cell
-- 0.854 here against 0.867 there at `b4096 V129280 k512`, 0.907 against 0.924 at
`b4096 V131072 k2048`. The column total is larger than the A/B's -8.35% because the
snapshot's 17 distinct shapes are mostly the `V >= 107520` rows that gain, while the
A/B's 30 cases also carry the b1/b16 draws that cannot.

Kernel to kernel, the fp32 selector path is now ahead of `deep_gemm`'s `topk_coarse12`
at every batch >= 132: **0.880x** at `b4096 V131072 k2048` (4745.5 against 5393.5 us)
and **0.879x** at `b4096 V107520` (3988.9 against 4538.0), against 0.946x / 0.940x in
the previous recording; `deep_gemm` itself moved only -2.1..-3.6% between the sessions,
so the ratio moving is this change. The b1 cells stay behind (1.37x at V66551,
2.1-2.4x at the two longer rows): 45-111 us, launch-latency bound. Both sides' `time(us)`
is the official "name contains topk" rule, so `deep_gemm`'s torch-level NaN check sits
in its `wall_us` (17.1 ms against our 4.76 ms at `b4096 V131072`), not in its kernel.

The baseline was not moved: `run_header.txt` lists the pod's `/root/busy.py`
device-0 keep-alive (the measurement ran on device 1), and the pod's checkout carries
no `baseline` symlink -- the run was not given `--set-baseline` and lands as a directory.

## The boundary the first version got wrong (caught by the gate, not by the grid)

The first build passed all 30 fp32 **performance** cells and still had a
correctness hole. The full suite (`./run_test.sh --all`, 349 cases) reported
`1 case(s) selected wrong`: `b512 V998123 k2048 sorted_value=True`, 5 slots of
one row each below the true k-th value, **deterministic** — three runs on the
new artifact reproduced it exactly, three on the base artifact were clean
(`repro_case63.py`).

The mechanism is the group shuffle at the loop edge. Pass 1's vector loop is
per-lane (`vec = tid, tid + kThreads, ...`), and when `vector_length` is not a
multiple of `kBlockLanes * kThreads` its last iteration runs a *prefix* of the
lanes. A four-lane group can straddle that edge, and the summary's
`__shfl_xor_sync(kFullMask, ...)` then names lanes that are not executing it —
undefined behavior, and on this part it returns the inactive lane's stale
register. One block per row (`vec >> 2 == 62254` at this shape) was recorded
with a max over two of its four float4s, so a block whose above-threshold
element sat in the other two was dropped, and a row could lose a candidate.

The 16-bit row does not have this bug, and not by luck: its summary loops are
block-granular (`((blk + 4) << 6) <= vec_len`), so a group enters whole or not
at all. The fp32 summary was grafted onto a per-lane loop, which is where the
partial group came from.

The first fix was two lines: shuffle on `__activemask()` rather than a full
mask, and a group that is not whole records `kCoarseBins - 1` — the maximum bin
— so its block is always kept. That walks at most one block per row that did not
need walking and can never drop one. It passed the full suite (343/343) and it
**cost 1.4% of the fp32 grid**: -6.76% and -6.78% over the same cells and rounds
that measure -8.35% and -8.39% below. The reason is `Finding 2` again — pass 1
is issue-bound, so a mask read, a four-lane bit test and a select wrapped around
each of the two shuffles is paid per block, and it lands on every iteration of
the walk even though only one iteration per row can be at the edge.

So the loop is split at `bulk_iters = vector_length / kThreads`, the iteration
count *every* lane runs (`tid + m * kThreads < vector_length` for all m under
it, whatever the lane). Inside that bulk the whole block is live: the mask is
the constant one and the summary needs no test at all. The tail — at most one
iteration per lane, a prefix of the lanes — keeps the guarded spelling. Both
arms compile to the same walk they did before; the only thing that moved is
where the guard sits, and it is now off the ~99.8% of the walk that is bulk.

**The grid says nothing about this**: the 30 fp32 perf cells never produced the
boundary, and the case the gate drew (`V998123`, a correctness-table shape) is
not on the grid at all. A perf grid is not a correctness gate. And the guarded
version *did* pass the grid at -6.76%, which is the other half of the same
lesson: a perf number taken before the correctness gate cannot be the number
this change ships with.

## What shipped

`topk_coarse12_row` now writes `block_max` in pass 1's vector loops, compacts the blocks at
or above the narrow's threshold into `block_list`, and walks that list in pass 2 with four
lanes per block. Both tables are per-row scratch (`RowParams::coarse12_block_max` /
`coarse12_block_list`, `maca_topk.cu`), allocated on the same gate as the route's answer
buffer and grown like the rest. The list is not used past three quarters of the blocks
(the walk then reads most of the row anyway and the list is traffic on top of it); a row
that trips that, or a call whose allocation failed, takes the dense walk unchanged.

The prefix (≤3 floats), the tail (≤3 floats) and the up-to-3 float4s past the last whole
block are outside every table entry and are walked densely on both arms. The
`topk == 0` early-return path also stays dense — it is a pure filter and rare.

The summary itself is `block_max_bin` (constant mask) over the bulk of pass 1's
vector loop and `block_max_bin_edge` (`__activemask()`, a group that is not whole
claims the maximum bin) over its last iteration — see the boundary section above
for why both spellings exist and what the guarded one costs when it is the only
one.

Gate: `./run_test.sh --all` on the shipped artifact (`6379e1cb`), **343/343
passed** (6 skipped on `torch.cuda.OutOfMemoryError`), plus the repro case
(`repro_case63.py`) at 0 bad slots, three runs. The perf grid is checked case by
case before it is timed, so the 30 fp32 cells in the A/B above are correctness
passes too.
