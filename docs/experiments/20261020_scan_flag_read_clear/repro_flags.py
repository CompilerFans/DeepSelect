"""Cross-call NaN-vote regression: a clean row must never inherit a previous
call's scan answer.

The original failure (2026-10-20, C600-U): the bf16 split kept its per-row NaN
flags inside the process-wide grow-only workspace, so a b1 v262144 clean call
that landed on the slot a b6 v1048576 call had filled with a candidate index
returned 0x3F3F3F3F in column 0.  The dispatcher-owned table plus the contract
half's read-and-clear is what these sequences check.
"""
import torch
import deep_select

torch.set_default_device("cuda")
SENTINEL = 0x3F3F3F3F

ok = True


def check(tag, cond):
    global ok
    cond = bool(cond)
    ok &= cond
    print(f"  {tag}: {'OK' if cond else 'FAIL'}")


def call(x, k, check_nan=True):
    _val, idx = deep_select.topk(
        x, k, sorted_index=False, return_value=False,
        indices_type=torch.int32, abort_when_nan_found=False,
        check_nan=check_nan, backend="maca_c")
    torch.cuda.synchronize()
    return idx


def sentinel_rows(idx):
    return (idx[:, 0] == SENTINEL).nonzero().flatten().tolist()


for dtype in (torch.bfloat16, torch.float32):
    tag = str(dtype).replace("torch.", "")
    gen = torch.Generator(device="cuda").manual_seed(1234)
    k = 512
    clean = torch.randn((1, 262144), dtype=dtype, generator=gen)
    dirty = torch.randn((6, 1048576), dtype=dtype, generator=gen)
    dirty[0, 3] = float("nan")

    idx = call(clean, k)
    check(f"{tag} cold clean has no sentinel", not sentinel_rows(idx))

    idx = call(dirty, k)
    rows = sentinel_rows(idx)
    check(f"{tag} dirty call guards only the NaN row", rows == [0])

    idx = call(clean, k)
    check(f"{tag} warm clean after dirty (state)", not sentinel_rows(idx))

    idx = call(clean, k, check_nan=False)
    check(f"{tag} clean with check_nan off", not sentinel_rows(idx))

    # A second dirty call must still see its own NaN after the table was
    # cleared by the clean calls above.
    idx = call(dirty, k)
    check(f"{tag} dirty again", sentinel_rows(idx) == [0])

    del clean, dirty

print("RESULT", "OK" if ok else "FAIL")
raise SystemExit(0 if ok else 1)
