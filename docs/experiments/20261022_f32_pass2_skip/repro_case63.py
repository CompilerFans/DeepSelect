# The one gate failure: b512 V998123 k2048 fp32 sorted_value, seed 62.
# Prints the contract's own verdict per row (min(selected) >= max(unselected),
# value == input[index]) plus the count of slots below the true k-th value.
# Run it several times per artifact: a race does not reproduce identically.
import sys, os, hashlib, torch, lib, deep_select
from deep_select._binding import extension_path

torch.set_default_device("cuda")
print("lib", hashlib.md5(open(extension_path("deep_select_maca"), "rb").read()).hexdigest()[:8])

p = lib.TestParam(batch_size=512, vocab_size=998123, topk=2048, sorted_value=True,
                  sorted_index=False, return_value=True, dtype=torch.float32,
                  out_idx_dtype=torch.int64, enable_end_position=False,
                  enable_output_idx_offset=False, seed=62, num_runs=1)
t = lib.generate_testcase(p)
x = t.input
val, idx = deep_select.topk(x, p.topk, sorted=True, return_value=True,
                            backend="maca_c", indices_type=torch.int64,
                            abort_when_nan_found=False)
torch.cuda.synchronize()
gathered = x.gather(1, idx)
kth = torch.topk(x, p.topk, dim=1).values[:, -1:]
bad_sorted = (val[:, :-1] < val[:, 1:]).sum().item()
bad_pair = (gathered != val).sum().item()
below = (gathered < kth).sum(1).sum().item()
rows = ((gathered < kth).sum(1) > 0).sum().item()
print("bad_sorted %d  bad_pair %d  slots_below_kth %d  rows %d" %
      (bad_sorted, bad_pair, below, rows))
