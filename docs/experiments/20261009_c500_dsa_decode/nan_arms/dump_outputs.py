#!/usr/bin/env python3
"""Run a fixed set of bf16 cases through one arm and save the outputs.

Ties are broken identically by both arms (same algorithm, integer keys), so
this compares bit-for-bit; a shape whose output differs is a real change.

    dump_outputs.py <arm-dir> <out.npz>
"""
import sys
import numpy as np
import torch

ARM, OUT = sys.argv[1], sys.argv[2]
sys.path.insert(0, ARM)
torch.set_default_device("cuda")
torch.manual_seed(0)
import deep_select  # noqa: E402

assert deep_select.__file__.startswith(ARM), deep_select.__file__
res = {}
def run(tag, x, k, **kw):
    _, idx = deep_select.topk(x, k, backend="maca_c", **kw)
    res[tag] = idx.cpu().numpy()

g = torch.Generator(device="cpu").manual_seed(12345)
def rand(*shape):
    t = torch.randn(*shape, generator=g, device="cpu").to(torch.bfloat16)
    return t.to("cuda")

for b, v, k in ((8, 4096, 512), (64, 65536, 512), (4, 524288, 512), (8, 8192, 1024)):
    run(f"rand_b{b}_v{v}_k{k}", rand(b, v), k)

# NaN rows: the row must come back with the 0x3F3F3F3F sentinel in slot 0.
for tag, fill in (("nan", float("nan")), ("pinf", float("inf")), ("ninf", float("-inf"))):
    x = rand(4, 4096)
    x[2, 777] = fill
    run(f"{tag}_abort_off", x, 512, abort_when_nan_found=False)
    run(f"{tag}_nocheck", x, 512, check_nan=False, abort_when_nan_found=False)

# a windowed row and an offset, to exercise the contract half around the fold
x = rand(6, 8192)
run("windowed", x, 512,
    end=torch.tensor([8000, 4000, 1000] * 2, dtype=torch.int32, device="cuda"))
np.savez(OUT, **{k: v for k, v in res.items()})
print(f"{ARM.split('/')[-1]}: {len(res)} outputs -> {OUT}")
