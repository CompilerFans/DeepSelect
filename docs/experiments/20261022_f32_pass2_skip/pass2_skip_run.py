# Driver for pass2_skip.cu: build the official fp32 case, compute each row's
# true 12-bit threshold bin (the bin of the k-th largest key -- exactly what
# coarse12's pass 1 narrows to), then hand the kernel a kept-block list at
# G = 16/32/64 built from that bin.
#
#   PYTHONPATH=<tree>:<tree>/tests CUDA_VISIBLE_DEVICES=1 \
#     python pass2_skip_run.py /tmp/ps.so
#
# The lists are built here, not in the kernel: the question this probe answers
# is what the *walk* costs, and production's summary/compaction bill is the
# bf16 row's (docs/experiments/20261021_blockhist/), not this probe's.
import ctypes
import os
import sys

sys.path.insert(0, os.environ.get("DS_TESTS", os.path.join(os.path.dirname(__file__), "../../../tests")))
import torch
import lib

torch.set_default_device("cuda")

SHAPES = [(4096, 131072, 2048), (4096, 129280, 512)]
if os.environ.get("PS_SHAPES"):   # e.g. PS_SHAPES=4096x66551x2048
    SHAPES = [tuple(int(n) for n in s.split("x")) for s in os.environ["PS_SHAPES"].split(",")]
ITERS = 5

P = ctypes.c_void_p
lib_ = ctypes.CDLL(sys.argv[1])
lib_.probe_dense.argtypes = [P, P, P, P, P, ctypes.c_int, ctypes.c_int,
                             ctypes.c_int, P, P, P, ctypes.c_int]
lib_.probe_listed.argtypes = [P, P, P, P, P, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, P, P, P, ctypes.c_int,
                              ctypes.c_int, ctypes.c_long, ctypes.c_long]
lib_.probe_pass1.argtypes = [P, ctypes.c_int, ctypes.c_int, P, P, ctypes.c_int,
                             ctypes.c_int]
lib_.probe_table.argtypes = [P, P, P, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                             P, P, P, ctypes.c_int]


def key_of_bits(bits):
    """The order-preserving key of an int32 bit pattern, as int32.

    The key is an *unsigned* 32-bit value; torch's int32 shifts are arithmetic,
    so every consumer must mask -- `bin_of` below is the one that matters."""
    return torch.where(bits < 0, ~bits, bits | (-2**31))


def bin_of_bits(bits):
    """The coarse12 bin: the key's top 12 bits, masked back to unsigned."""
    return (key_of_bits(bits) >> 20) & 0xFFF


def build_bmax(x):
    """The block-max table pass 1 would write: max 12-bit bin per 16 elements."""
    B, V = x.shape
    nblk = V // 16                     # whole blocks; a ragged tail is not tabled
    bb = x[:, :nblk * 16].view(B, nblk, 16).amax(dim=2).view(torch.int32)
    return bin_of_bits(bb).to(torch.int16).contiguous()


def build_list(x, thr, G):
    """Kept blocks at granularity G: block max key >= thr, packed row-major."""
    B, V = x.shape
    nblk = V // G
    bb = x[:, :nblk * G].view(B, nblk, G).amax(dim=2).view(torch.int32)
    keep = bin_of_bits(bb) >= thr[:, None]
    idx = keep.nonzero()                      # (nnz, 2), sorted row-major
    cols = idx[:, 1].to(torch.int32).contiguous()
    lens = torch.bincount(idx[:, 0], minlength=B).to(torch.int32)
    off = torch.zeros(B, dtype=torch.int32, device="cuda")
    off[1:] = torch.cumsum(lens, 0)[:-1]
    del idx, keep, bb
    return cols, off.contiguous(), lens.contiguous()


for B, V, K in SHAPES:
    p = lib.TestParam(batch_size=B, vocab_size=V, topk=K, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32, out_idx_dtype=torch.int32,
                      seed=0, num_runs=1)
    tc = lib.generate_testcase(p)
    x = tc.input.contiguous()
    del tc

    kth = torch.topk(x, K, dim=1).values[:, -1]
    thr = bin_of_bits(kth.view(torch.int32)).to(torch.int32).contiguous()

    out = torch.empty((B, K), dtype=torch.int32, device="cuda")
    above = torch.empty(B, dtype=torch.int32, device="cuda")
    inbin = torch.empty(B, dtype=torch.int32, device="cuda")

    print(f"\n=== b{B} V{V} k{K}  ({x.numel() * 4 / 1e9:.2f} GB) ===")
    lib_.probe_dense(x.data_ptr(), thr.data_ptr(), ctypes.c_void_p(0),
                     ctypes.c_void_p(0), ctypes.c_void_p(0), B, V, K,
                     out.data_ptr(), above.data_ptr(), inbin.data_ptr(), ITERS)
    ref_above = above.cpu()
    ref_inbin = inbin.cpu()
    print(f"  counts: above {ref_above.min().item()}..{ref_above.max().item()} "
          f"(k={K}), bin {ref_inbin.min().item()}..{ref_inbin.max().item()} "
          f"(arena 2048)")

    bmax = build_bmax(x)
    hist = torch.empty((B, 256), dtype=torch.int32, device="cuda")
    lib_.probe_pass1(x.data_ptr(), B, V, ctypes.c_void_p(0), hist.data_ptr(),
                     ITERS, 0)
    lib_.probe_pass1(x.data_ptr(), B, V, bmax.data_ptr(), hist.data_ptr(),
                     ITERS, 1)
    lib_.probe_pass1(x.data_ptr(), B, V, bmax.data_ptr(), hist.data_ptr(),
                     ITERS, 2)
    chk = bmax.view(torch.int16)
    lib_.probe_table(x.data_ptr(), thr.data_ptr(), chk.data_ptr(), B, V, K,
                     out.data_ptr(), above.data_ptr(), inbin.data_ptr(), ITERS)
    a = above.cpu()
    b = inbin.cpu()
    ok = bool((a == ref_above).all() and (b == ref_inbin).all())
    print(f"  table: counts {'match dense' if ok else 'DIFFER from dense'}")
    if not ok:
        print(f"    above dense {ref_above[:5].tolist()} table {a[:5].tolist()}")
    del bmax, chk, hist
    torch.cuda.empty_cache()

    for G in (16, 32, 64):
        cols, off, lens = build_list(x, thr, G)
        lib_.probe_listed(x.data_ptr(), thr.data_ptr(), cols.data_ptr(),
                          off.data_ptr(), lens.data_ptr(), B, V, K,
                          out.data_ptr(), above.data_ptr(), inbin.data_ptr(),
                          ITERS, G, int(cols.numel()), B * (V // G))
        a = above.cpu()
        b = inbin.cpu()
        same = bool((a == ref_above).all() and (b == ref_inbin).all())
        print(f"  G={G}: counts {'match dense' if same else 'DIFFER from dense'}")
        if not same:
            print(f"    above dense {ref_above[:5].tolist()} listed {a[:5].tolist()}")
            print(f"    inbin dense {ref_inbin[:5].tolist()} listed {b[:5].tolist()}")
            print(f"    delta rows: above {(a - ref_above).abs().max().item()}, "
                  f"inbin {(b - ref_inbin).abs().max().item()}, "
                  f"rows differing {(a != ref_above).sum().item()}/{(b != ref_inbin).sum().item()}")
        del cols, off, lens

    del x, kth, thr, out, above, inbin, ref_above, ref_inbin
    torch.cuda.empty_cache()
