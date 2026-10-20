import torch
import deep_select

x = torch.randn((1, 262144), dtype=torch.bfloat16, device="cuda")
x[0, 17] = float("nan")
deep_select.topk(x, 512, sorted_index=False, return_value=False,
                 indices_type=torch.int32, abort_when_nan_found=True,
                 backend="maca_c")
torch.cuda.synchronize()
print("NO_TRAP")
