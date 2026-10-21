"""Targeted thread-width A/B: the five official fp32 cells + two bf16 controls.

Run on the C600U pod (32741) with the two arms staged as `/tmp/t640` and
`/tmp/t512`, each a full checkout copy with its own built
`deep_select/deep_select_maca_xcore1600.so` (md5 ccf5631f / 5958dcea).  The
child is a fresh process per (arm, round) so the two artifacts never share an
address space; rounds alternate t640/t512 three times, each arm takes 9 calls
per cell per round, and the printed value is the median of the three round
medians.  The per-round medians are kept beside the result so a reader can see
the spread -- which is what makes the 0.94-0.96x verdict a measurement rather
than a single pairing.

The bf16 cells are the control: bf16 does not route through coarse12, so they
must not move.  Output: `ab_targeted.txt`, this directory.
"""
import sys, json, subprocess, statistics as st, os

CELLS = [[4096, 131072, 2048, "f32"], [4096, 129280, 512, "f32"],
         [4096, 107520, 2048, "f32"], [256, 131072, 2048, "f32"],
         [512, 66551, 2048, "f32"],
         [4096, 1048576, 512, "bf16"], [4096, 262144, 1024, "bf16"]]

CHILD = r'''
import sys, torch, time, json
sys.path.insert(0, sys.argv[1])
sys.path.insert(0, "/root/github/DeepSelect/tests")
import deep_select, lib
torch.set_default_device("cuda")
out = []
for b, v, k, dt in json.loads(sys.argv[2]):
    p = lib.TestParam(batch_size=b, vocab_size=v, topk=k, sorted_value=False,
                      sorted_index=False, return_value=False,
                      dtype=torch.float32 if dt == "f32" else torch.bfloat16,
                      out_idx_dtype=torch.int32, seed=0, num_runs=1)
    x = lib.generate_testcase(p).input
    for _ in range(3):
        deep_select.topk(x, k, indices_type=torch.int32, return_value=False,
                         abort_when_nan_found=False, backend="maca_c")
    torch.cuda.synchronize()
    ts = []
    for _ in range(9):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        deep_select.topk(x, k, indices_type=torch.int32, return_value=False,
                         abort_when_nan_found=False, backend="maca_c")
        torch.cuda.synchronize(); ts.append((time.perf_counter() - t0) * 1e6)
    out.append([b, v, k, dt, sorted(ts)[len(ts)//2]])
    del x
    torch.cuda.empty_cache()
print("R " + json.dumps(out))
'''

results = {"t640": [], "t512": []}
for rnd in range(3):
    for arm in ("t640", "t512"):
        env = dict(os.environ)
        env["PYTHONPATH"] = f"/tmp/{arm}"
        env["CUDA_VISIBLE_DEVICES"] = "1"
        r = subprocess.run(["/opt/conda/bin/python", "-c", CHILD, f"/tmp/{arm}",
                            json.dumps(CELLS)], capture_output=True, text=True,
                           env=env)
        line = [l for l in r.stdout.splitlines() if l.startswith("R ")]
        if not line:
            print(arm, "FAILED:", r.stdout[-300:], r.stderr[-300:])
            sys.exit(1)
        results[arm].append(json.loads(line[0][2:]))

for i, (b, v, k, d) in enumerate(CELLS):
    a = [results["t640"][r][i][4] for r in range(3)]
    b_ = [results["t512"][r][i][4] for r in range(3)]
    ma, mb = st.median(a), st.median(b_)
    print(f"{d:4s} b{b:<5d} V{v:<7d} k{k:<5d}: t640 {ma:9.1f}  t512 {mb:9.1f} us"
          f"  {mb/ma:.4f}x   rounds {['%.0f' % x for x in a]} vs {['%.0f' % x for x in b_]}")
