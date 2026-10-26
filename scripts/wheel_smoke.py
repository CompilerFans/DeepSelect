#!/usr/bin/env python3
"""Acceptance smoke test for an *installed* `deep_select` -- run it from outside
the repository, so the package that answers is the wheel and not the checkout.

    cd /tmp && python /path/to/scripts/wheel_smoke.py
    cd /tmp && CUDA_VISIBLE_DEVICES=1 python /path/to/scripts/wheel_smoke.py

What it establishes, in order:

  1. **Which artifact a deployment loads.**  `deep_select._binding.extension_path`
     is the same glob `load()` uses, so the printed path and its md5 are the
     "which binary answered" receipt -- including the family suffix, which the
     wheel carries one file per family for.
  2. **The three production cells the operator is sold for**: the bf16
     Lightning-Indexer shape, the fp32 Sampler shape, and an fp32 row with a
     ragged length (a padded row stride, a prefix and a tail -- the vector
     region's edges).  Each is checked against the contract -- index range,
     uniqueness, `value == input[index]`, `min(selected) >= max(unselected)` --
     not against a reference except on fp32, where `backend="torch"` under
     `sorted_index` agrees elementwise (the one differential CLAUDE.md records
     as meaningful).
  3. **The declared dependencies are really there**: the extension resolves its
     `libtvm_ffi.so` through `import tvm_ffi`, and the package imports with no
     checkout on the path.
  4. **The default backend resolution** (`DS_TOPK_BACKEND`, unrecognized values
     ignored) still answers in a fresh process.

Exit status is the verdict.  There is no timing here on purpose: this is a
smoke test for a delivered wheel, and a record is a different act with its own
rules (`run_bench.sh`).
"""
import argparse
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# The receipt's md5 length is `tools/provenance.py`'s question (12 for a
# human-facing line).  Only the helper is taken from there: the *path* rule is
# deliberately not applied here -- this gate's job is to refuse when the
# checkout answers, so it must not put the checkout on the path.
sys.path.insert(0, os.path.join(REPO, "tools"))
from provenance import md5_of  # noqa: E402
FAIL = []
PASS = [0]


def check(name, cond, detail=""):
    if cond:
        PASS[0] += 1
        print(f"  PASS  {name}")
    else:
        FAIL.append(name + (f" -- {detail}" if detail else ""))
        print(f"  FAIL  {name}" + (f" -- {detail}" if detail else ""))


def import_package(allow_checkout):
    import deep_select
    where = os.path.dirname(os.path.abspath(deep_select.__file__))
    print(f"deep_select {deep_select.__version__} from {where}")
    in_repo = os.path.commonpath([where, REPO]) == REPO
    check("the imported package is not this checkout", allow_checkout or not in_repo,
          f"{where} is inside {REPO}; run from outside the repository, or the "
          f"wheel was never installed")
    return deep_select


def min_row_stride(v, dtype):
    """The smallest row stride >= v the operator's 1024-byte rule accepts."""
    per = 1024 // dtype.itemsize
    return (v + per - 1) // per * per


def row_major(torch, b, v, dtype, seed):
    """A (b, v) tensor whose row stride is the operator's minimum, so a ragged
    `v` is served from a padded allocation exactly as a caller would slice it."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    pad = min_row_stride(v, dtype)
    backing = torch.randn((b, pad), generator=g, dtype=torch.float32)
    return backing.to(dtype).to("cuda")[:, :v]


def contract(torch, x, k, out_idx, out_val):
    """The operator's contract, row by row.  Exact under ties."""
    b, v = x.shape
    for r in range(b):
        row = out_idx[r][:k].to(torch.int64)
        if int(row.min()) < 0 or int(row.max()) >= v:
            return f"row {r}: index out of range [0, {v})"
        if int(torch.unique(row).numel()) != k:
            return f"row {r}: duplicate index"
        sel = x[r][row]
        if out_val is not None and not torch.equal(sel, out_val[r][:k]):
            return f"row {r}: value != input[index]"
        if v > k and int((x[r] > sel.min()).sum()) >= k:
            return f"row {r}: min(selected) < max(unselected)"
    return None


def run_case(torch, ds, name, x, k, **kw):
    try:
        val, idx = ds.topk(x, k, backend="maca_c", **kw)
    except BaseException as exc:  # noqa: BLE001 - the smoke test is "it serves"
        check(name, False, f"raised {type(exc).__name__}: {str(exc)[:160]}")
        return None
    why = contract(torch, x, k, idx, val)
    check(name, why is None, why or "")
    return val, idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", type=int, default=None,
                    help="CUDA_VISIBLE_DEVICES shorthand")
    ap.add_argument("--allow-checkout", action="store_true",
                    help="skip the 'not the checkout' guard (a develop.sh tree)")
    args = ap.parse_args()
    if args.device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)

    ds = import_package(args.allow_checkout)
    import torch
    from deep_select import _binding

    # -- 1. the artifact the deployment resolves.
    so = _binding.extension_path("deep_select_maca")
    check("an extension artifact resolves", so is not None,
          "no deep_select_maca*.so beside the installed package")
    if so is not None:
        print(f"artifact {so} md5 {md5_of(so)}")
        suffix = _binding.device_family_suffix()
        check("the artifact is this device's family",
              suffix in os.path.basename(so) if suffix else True,
              f"device family {suffix!r}, artifact {os.path.basename(so)}")
        props = torch.cuda.get_device_properties(0)
        print(f"torch {torch.__version__}, {props.name}, "
              f"{props.multi_processor_count} APs")

    # -- 2. the production cells.
    xb = row_major(torch, 8, 65536, torch.bfloat16, seed=1)
    run_case(torch, ds, "bf16 b8 v65536 k512", xb, 512, indices_type=torch.int32)
    run_case(torch, ds, "bf16 b8 v65536 k512 sorted value", xb, 512, sorted=True,
             indices_type=torch.int32)

    xf = row_major(torch, 8, 131072, torch.float32, seed=2)
    _, idx = run_case(torch, ds, "fp32 b8 v131072 k2048 (Sampler)", xf, 2048,
                        indices_type=torch.int32, sorted_index=True)
    if idx is not None:
        _, ref = ds.topk(xf, 2048, backend="torch", indices_type=torch.int32,
                         sorted_index=True)
        same = bool(torch.equal(idx.to(torch.int64), ref.to(torch.int64)))
        check("fp32 Sampler cell: elementwise vs torch", same,
              "" if same else f"{int((idx != ref).sum())} of {idx.numel()} differ")

    # A ragged row: v % 4 != 0 at a width the coarse12 route serves, sliced from
    # a padded allocation (stride(0) stays a multiple of 1024 B).
    xr = row_major(torch, 4, 4099, torch.float32, seed=3)
    check("the ragged row is padded and served", xr.stride(0) * xr.element_size() % 1024 == 0)
    _, idx = run_case(torch, ds, "fp32 b4 v4099 k512", xr, 512,
                        indices_type=torch.int32, sorted_index=True)
    if idx is not None:
        _, ref = ds.topk(xr, 512, backend="torch", indices_type=torch.int32,
                         sorted_index=True)
        same = bool(torch.equal(idx.to(torch.int64), ref.to(torch.int64)))
        check("ragged cell: elementwise vs torch", same,
              "" if same else f"{int((idx != ref).sum())} of {idx.numel()} differ")

    # -- 3. the process default still resolves per call.
    code = ("import deep_select;"
            "print(deep_select.interface._default_backend())")
    env = {k: v for k, v in os.environ.items() if k != "DS_TOPK_BACKEND"}
    out = subprocess.run([sys.executable, "-c", code], env=env,
                         capture_output=True, text=True)
    check("the default resolves to 'maca_c' with no DS_TOPK_BACKEND",
          out.stdout.strip().endswith("maca_c"), out.stdout.strip() + out.stderr[-200:])
    out = subprocess.run([sys.executable, "-c", code],
                         env={**env, "DS_TOPK_BACKEND": "torch"},
                         capture_output=True, text=True)
    check("DS_TOPK_BACKEND=torch resolves the reference",
          out.stdout.strip().endswith("torch"), out.stdout.strip() + out.stderr[-200:])
    out = subprocess.run([sys.executable, "-c", code],
                         env={**env, "DS_TOPK_BACKEND": " TORCH "},
                         capture_output=True, text=True)
    check("a padded, upper-case name still resolves to the reference",
          out.stdout.strip().endswith("torch"), out.stdout.strip() + out.stderr[-200:])

    print(f"\n{PASS[0]} passed, {len(FAIL)} failed")
    for f in FAIL:
        print(f"  FAILED: {f}")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
