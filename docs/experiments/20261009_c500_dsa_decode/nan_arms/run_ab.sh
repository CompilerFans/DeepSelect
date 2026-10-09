#!/usr/bin/env bash
# Wait for the outer repo's suite to release the box, then measure the five
# arms in one session, alternating, three rounds.
#
#   A  production (two bodies)
#   B  fold_nan=false literal (one body)
#   T  B + 2,588 B more dynamic smem requested -> the same total footprint as A
#   S1 separate nan_scan_kernel in front of B, scan grid = batches * 1
#   S2 same, scan grid = batches * 16
LOG=/tmp/ds_nan_ab/ab_final.log
: > "$LOG"
PID=1960576
for _ in $(seq 1 240); do
  kill -0 "$PID" 2>/dev/null || break
  sleep 30
done
if kill -0 "$PID" 2>/dev/null; then
  echo "TIMEOUT: the outer suite is still alive after 2h" >> "$LOG"; exit 1
fi
sleep 20
echo "=== contender gone? ===" >> "$LOG"
pgrep -af python | grep -vE 'pgrep|socks5|networkd|unattended|vscode|pet server|lsp_server|jedi|run_ab' \
  >> "$LOG" || echo "  no other torch process" >> "$LOG"
echo "=== five arms, three rounds ===" >> "$LOG"
for r in 1 2 3; do
  for a in A B T S1 S2; do
    CUDA_VISIBLE_DEVICES=2 timeout 900 python3 /tmp/ds_nan_ab/measure_sum.py \
      "/tmp/ds_nan_ab/$a" "$a-r$r" 2>&1 \
      | grep -vE 'flash_attn|^  import|Warning' >> "$LOG"
  done
done
echo "ALLDONE" >> "$LOG"
