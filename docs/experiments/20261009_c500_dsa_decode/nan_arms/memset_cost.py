#!/usr/bin/env python3
"""What one `mcMemsetAsync` of the abort table costs.

This is the number behind `RowParams::nan_abort_flags`'s "no per-call memset"
decision: the table is written for every row by the row kernel instead of being
zeroed on the stream, because the memset is not free and the calls that pay it
are the small-batch ones that gain nothing from the occupancy the change buys.

It is a *launch* cost, not a bandwidth one -- the table is 16 KiB and the same
figure comes out for 4 KiB -- so the loop runs the memset `repl` times inside
one event pair to show the per-call figure is not the event overhead.

`mcMemsetAsync` is reached through ctypes rather than a torch call: the symbol
is `mcMemsetAsync` in `libmcruntime.so` and there is no `cudaMemsetAsync` on
this platform, so a `torch` spelling would be measuring something else.

    memset_cost.py
"""
import ctypes
import statistics

import torch

torch.set_default_device("cuda")
mcr = ctypes.CDLL("libmcruntime.so")
mcr.mcMemsetAsync.restype = ctypes.c_int
mcr.mcMemsetAsync.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t,
                              ctypes.c_void_p]
buf = torch.empty(4096, dtype=torch.int32)
ptr = ctypes.c_void_p(buf.data_ptr())
stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)


def timed(n_bytes, repl):
    def one():
        for _ in range(repl):
            mcr.mcMemsetAsync(ptr, 0, n_bytes, stream)
    for _ in range(5):
        one()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    one()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / repl * 1e3          # ms -> us per memset


print(f"torch {torch.__version__}")
for n in (4096 * 4,):
    for repl in (1, 10):
        vals = [timed(n, repl) for _ in range(5)]
        print(f"  mcMemsetAsync({n} B) x{repl}: median "
              f"{statistics.median(vals):.1f} us  {[round(v, 1) for v in vals]}")

# The floor: an event pair with nothing between it, so the figures above can be
# read as "a launch" rather than as "the clock".
torch.cuda.synchronize()
s = torch.cuda.Event(enable_timing=True)
e = torch.cuda.Event(enable_timing=True)
s.record()
e.record()
torch.cuda.synchronize()
print(f"  empty event pair: {s.elapsed_time(e) * 1e3:.1f} us")
