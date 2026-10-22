#!/bin/bash
# The whole fp32 official grid, alternating arms.  test.py's case seeds come
# from a per-process counter, so every arm sees identical data; each case is
# checked before it is timed, so a wrong answer is a failure, not a clock.
set -u
REPO=/root/github/DeepSelect
SO=$REPO/deep_select/deep_select_maca_xcore1600.so
M=/root/maca/maca-sdk-20261017-1436/opt/maca-20261017
export CUDA_VISIBLE_DEVICES=1 MACA_PATH=$M PYTHONPATH=$REPO
export LD_LIBRARY_PATH=$M/lib:$M/mxgpu_llvm/lib:$M/ompi/lib:${LD_LIBRARY_PATH:-}
for r in 1 2; do
  for arm in base new; do
    if [ "$arm" = base ]; then src=/root/base_004efa7.so; else src=${NEW_SO:-/root/new_fix.so}; fi
    cp -f "$src" "$SO"
    ( cd "$REPO" && /opt/conda/bin/python tests/test.py --perf-only --dtype fp32 -nc \
        > "/tmp/grid_fp32_${arm}_r${r}.log" 2>&1 )
    echo "round $r $arm rc=$? $(grep -c '^topk *:' /tmp/grid_fp32_${arm}_r${r}.log) cells"
  done
done
