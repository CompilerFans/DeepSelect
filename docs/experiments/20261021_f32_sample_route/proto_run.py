# Driver for proto_sample.cu -- the fp32 sampled-threshold one-pass prototype.
#   python proto_run.py /tmp/proto.so
# Builds official fp32 cases (tests/lib.py), runs the prototype, verifies the
# contract per row (min(selected) >= max(unselected)) plus set agreement with
# torch.topk, counts fallbacks, and times the kernel (events, 2 warm + 5).
import os
import sys
import ctypes

sys.path.insert(0, os.environ.get("DS_TESTS", "/root/github/DeepSelect/tests"))
import torch, lib

torch.set_default_device("cuda")

CELLS = [(4096, 131072, 2048), (4096, 129280, 512), (4096, 107520, 2048)]

lib_ = ctypes.CDLL(sys.argv[1])
lib_.proto_launch.argtypes = [ctypes.c_void_p] + [ctypes.c_int] * 5 + [
    ctypes.c_void_p, ctypes.c_void_p]
lib_.proto_launch.restype = ctypes.c_int

for B, V, K in CELLS:
    margin = 1.4 if K >= 1024 else 1.6
    m_sample = max(1, round(margin * K / 16))
    s_len = V // 16
    p = lib.TestParam(batch_size=B, vocab_size=V, topk=K, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32, out_idx_dtype=torch.int32,
                      seed=0, num_runs=1)
    tc = lib.generate_testcase(p)
    x = tc.input.contiguous()
    out_idx = torch.full((B, K), -1, dtype=torch.int32)
    counters = torch.zeros((B, 12), dtype=torch.int32)

    rc = lib_.proto_launch(x.data_ptr(), B, V, K, m_sample, s_len,
                           out_idx.data_ptr(), counters.data_ptr())
    torch.cuda.synchronize()
    assert rc == 0, f"launch rc={rc}"
    c = counters.cpu()
    n_fallback = int((c[:, 1] == 2).sum())
    nan_rows = int((c[:, 2] > 0).sum())
    staged = c[:, 0]

    # contract: every selected value >= the row's k-th value (definitional)
    ok_rows = c[:, 1] == 1
    idx = out_idx[ok_rows].to(torch.int64)
    xok = x[ok_rows]
    sel = xok.gather(1, idx)
    kth = torch.topk(xok, K, dim=1).values[:, -1]
    n_contract_bad = int((sel.min(dim=1).values < kth).sum())
    n_short = int((out_idx[ok_rows] < 0).any(dim=1).sum())
    emitted = c[ok_rows, 7]

    # set agreement with torch.topk on up to 64 of the ok rows
    n_set_mismatch = 0
    if ok_rows.any():
        vv, ii = torch.topk(xok[:64], K, dim=1)
        s_ref = ii.sort(dim=1).values
        s_got = out_idx[ok_rows][:64].to(torch.int64).sort(dim=1).values
        n_set_mismatch = int((s_ref != s_got).any(dim=1).sum())

    # timing (events; includes the whole kernel, single launch)
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    for _ in range(2):
        lib_.proto_launch(x.data_ptr(), B, V, K, m_sample, s_len,
                          out_idx.data_ptr(), counters.data_ptr())
    torch.cuda.synchronize()
    e0.record()
    for _ in range(5):
        lib_.proto_launch(x.data_ptr(), B, V, K, m_sample, s_len,
                          out_idx.data_ptr(), counters.data_ptr())
    e1.record()
    torch.cuda.synchronize()
    ms = e0.elapsed_time(e1) / 5
    gbs = B * V * 4 / (ms * 1e6)
    clk = 1.6e9  # device clock, Hz -- ticks -> ms
    print(f"b{B} V{V} k{K} m={m_sample}: {ms:8.3f} ms  {gbs:7.1f} GB/s"
          f" | fallback {n_fallback} nan {nan_rows}"
          f" | staged med {int(staged.median())}"
          f" | contract_bad {n_contract_bad} short {n_short}"
          f" set_mismatch {n_set_mismatch}/64")
    if ok_rows.any():
        ph = c[ok_rows].double().mean(dim=0) / clk * 1e3
        print(f"   phases ms/row: sample {ph[4]:.4f} filter {ph[5]:.4f}"
              f" refine {ph[6]:.4f} total {ph[3]:.4f}"
              f" | emitted med {int(emitted.median())} (want {K})")
    del x, out_idx, counters, tc
    torch.cuda.empty_cache()
