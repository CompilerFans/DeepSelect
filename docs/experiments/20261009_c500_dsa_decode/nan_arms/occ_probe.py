#!/usr/bin/env python3
"""Ask the MACA runtime -- not a model -- for each arm's CTAs per SM.

Usage:  occ_probe.py <arm-dir> <dynamic-smem-bytes>
Loads the arm through its own package (so the extension is registered the
way production registers it), then calls the driver's own
mcOccupancyMaxActiveBlocksPerMultiprocessor on the row kernel this cell
runs, with the block size and dynamic smem that arm launches with.
"""
import ctypes
import glob
import sys

import torch

ARM, DYN = sys.argv[1], int(sys.argv[2])
torch.cuda.init()
sys.path.insert(0, ARM)
import deep_select  # noqa: F401  -- registers the kernels

so = sorted(glob.glob(ARM + "/deep_select/deep_select_maca_xcore*.so"))[-1]
lib = ctypes.CDLL(so)
NAME = ("_ZN16deep_select_maca32__device_stub__topk_kernel_radixI15__maca_bfloat16i"
        "Li1024ELb0ELb0ELb0ELb0EEEvNS_9RowParamsE")

rt = ctypes.CDLL("libmcruntime.so")
occ = rt.mcOccupancyMaxActiveBlocksPerMultiprocessor
occ.restype = ctypes.c_int
occ.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_void_p,
                ctypes.c_int, ctypes.c_size_t]

fn = ctypes.cast(getattr(lib, NAME), ctypes.c_void_p)
nb = ctypes.c_int(-1)
rc = occ(ctypes.byref(nb), fn, 1024, DYN)
arm = ARM.rstrip("/").split("/")[-1]
print(f"  {arm:<4} dyn={DYN:<6} rc={rc}  blocks/SM={nb.value}")
