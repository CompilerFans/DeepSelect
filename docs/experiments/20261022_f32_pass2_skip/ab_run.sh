#!/bin/bash
# Paired A/B: alternating arms, one process per (arm, round), base first.
# The artifact is swapped in place -- both arms are the same checkout, so
# only the .so differs; the child prints the md5 that answered.
set -u
REPO=/root/github/DeepSelect
SO=$REPO/deep_select/deep_select_maca_xcore1600.so
BASE=/root/base_004efa7.so
NEW=/root/new_pass2skip.so
ROUNDS=${ROUNDS:-3}
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=$REPO
M=/root/maca/maca-sdk-20261017-1436/opt/maca-20261017
export LD_LIBRARY_PATH=$M/lib:$M/mxgpu_llvm/lib:$M/ompi/lib:${LD_LIBRARY_PATH:-}
for r in $(seq 1 $ROUNDS); do
  for arm in base new; do
    if [ "$arm" = base ]; then src=$BASE; else src=$NEW; fi
    cp -f "$src" "$SO"
    echo "--- round $r arm $arm"
    /opt/conda/bin/python -u "$(dirname "$0")/ab_cells32.py" "$REPO" 2>&1 | grep -vE "warning|Warning"
  done
done
