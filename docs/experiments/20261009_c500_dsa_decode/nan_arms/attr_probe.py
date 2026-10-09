#!/usr/bin/env python3
"""Ask the driver what it thinks each kernel's register count is."""
import ctypes, glob, sys
import torch
ARM = sys.argv[1]
torch.cuda.init()
sys.path.insert(0, ARM)
import deep_select  # noqa: F401
so = sorted(glob.glob(ARM + "/deep_select/deep_select_maca_xcore*.so"))[-1]
lib = ctypes.CDLL(so)
NAME = ("_ZN16deep_select_maca32__device_stub__topk_kernel_radixI15__maca_bfloat16i"
        "Li1024ELb0ELb0ELb0ELb0EEEvNS_9RowParamsE")
rt = ctypes.CDLL("libmcruntime.so")
get = rt.mcFuncGetAttribute
get.restype = ctypes.c_int
get.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_void_p]
fn = ctypes.cast(getattr(lib, NAME), ctypes.c_void_p)
out = []
for attr in range(1, 12):
    v = ctypes.c_int(-1)
    rc = get(ctypes.byref(v), attr, fn)
    out.append(f"{attr}:{'--' if rc else v.value}")
print(f"  {ARM.rstrip('/').split('/')[-1]:<3} attrs(1..11) = " + "  ".join(out))
