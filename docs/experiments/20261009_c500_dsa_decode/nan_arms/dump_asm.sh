#!/usr/bin/env bash
# Device asm + resource usage for one arm's maca_topk.cu, with the build's own
# flag list (taken from buildA.log's cucc invocation).
set -euo pipefail
TREE="$1"
OUT="$2"
CC=/opt/maca/tools/cu-bridge/bin/cucc
TT=/home/compiler_gfx/miniconda3/lib/python3.10/site-packages
"$CC" \
  -I"$TREE"/csrc -I"$TREE"/csrc/ffi \
  -I/opt/maca/include -I"$TT"/tvm_ffi/include \
  -I"$TT"/torch/include -I"$TT"/torch/include/torch/csrc/api/include \
  -I"$TT"/torch/include/TH -I"$TT"/torch/include/THC \
  -I/opt/maca/include/mcr -I/opt/maca/include/mcblas -I/opt/maca/include/mcfft \
  -I/opt/maca/include/mcsolver -I/opt/maca/include/mcdnn -I/opt/maca/include/common \
  -I/opt/maca/include/mcsparse -I/opt/maca/include/mcrand -I/opt/maca/include/mckl \
  -I/opt/maca/include/mcsml -I/opt/maca/include/mctx -I/opt/maca/include/thrust/detail \
  -I/opt/maca/tools/cu-bridge/include \
  -I"$TT"/torch/include/ATen/native/cuda \
  -I/home/compiler_gfx/miniconda3/include/python3.10 \
  -D__CUDA_NO_HALF_OPERATORS__ -D__CUDA_NO_HALF_CONVERSIONS__ \
  -D__CUDA_NO_BFLOAT16_CONVERSIONS__ -D__CUDA_NO_HALF2_OPERATORS__ \
  --expt-relaxed-constexpr -O3 -std=c++20 -DNDEBUG \
  -DDEEP_SELECT_IS_BUILD_ON_MACA -Wno-deprecated-declarations -fPIC \
  -use-fast-math -Xclang -fdenormal-fp-math-f32=ieee \
  -offload-arch=xcore1000 -DDEEP_SELECT_ARCH=1000 \
  -DTORCH_EXTENSION_NAME=deep_select_maca_xcore1000 \
  -D_GLIBCXX_USE_CXX11_ABI=1 -DUSE_MACA -DNV_ARCH_A100 \
  -aop -S -maca-device-only --resource-usage \
  "$TREE"/csrc/maca_kernels/xcore1000/maca_topk.cu -o "$OUT" 2> "${OUT%.s}.ru"
