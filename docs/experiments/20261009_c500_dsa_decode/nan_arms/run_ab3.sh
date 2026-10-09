#!/usr/bin/env bash
# A (production) vs Y1 (SWAR fold) vs W (SWAR test only), alternating, 3 rounds.
LOG=/tmp/ds_nan_ab/ab_fold.log
: > "$LOG"
pgrep -af python | grep -vE 'pgrep|socks5|networkd|unattended|vscode|pet server|lsp_server|jedi|run_ab3' >> "$LOG" || echo "  no other torch process" >> "$LOG"
for r in 1 2 3; do
  for a in A W Y1; do
    CUDA_VISIBLE_DEVICES=2 timeout 900 python3 /tmp/ds_nan_ab/measure_sum.py \
      "/tmp/ds_nan_ab/$a" "$a-r$r" 2>&1 \
      | grep -vE 'flash_attn|^  import|Warning' >> "$LOG"
  done
done
echo "ALLDONE" >> "$LOG"
