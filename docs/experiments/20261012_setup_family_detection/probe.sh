#!/usr/bin/env bash
# The two probes behind this directory's README, in one script.
#
#   ./probe.sh targets   -- the CUCC_TARGETS input matrix, through the real
#                           `setup.py` (it parses CUCC_TARGETS at import, before
#                           any command runs, so a run with no command reaches
#                           the whole decision and stops at "no commands
#                           supplied").  ~20 s per case: torch is imported first.
#   ./probe.sh arch      -- which `-offload-arch` spellings the device compiler
#                           takes, and what each one compiles to.  Seconds.
#
# `MACA_PATH` is authoritative; read it, do not assume `/opt/maca`.
set -uo pipefail

REPO=$(cd "$(dirname "$0")/../../.." && pwd)
export MACA_PATH="${MACA_PATH:-${MACA_HOME:-/opt/maca}}"
CUCC="$MACA_PATH/tools/cu-bridge/bin/cucc"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"' EXIT

probe_targets() {
    local case_label="$1"; shift
    local out rc
    out=$(cd "$REPO" && env "$@" python setup.py 2>&1); rc=$?
    printf '=== %s === rc=%s\n' "$case_label" "$rc"
    printf '%s\n' "$out" | grep -E 'deep_select: compiling|deep_select: CUCC_TARGETS|RuntimeError|error:' | tail -3
    echo
}

case "${1:-}" in
targets)
    # The last two lines of each case are the verdict: either the resolved
    # target list, or the refusal's own message.
    probe_targets "a) unset"           -u CUCC_TARGETS
    probe_targets "b) empty"           CUCC_TARGETS=
    probe_targets "c) whitespace only" CUCC_TARGETS='   '
    probe_targets "d) xcore1000"       CUCC_TARGETS=xcore1000
    probe_targets "e) 1000,1600"       CUCC_TARGETS=xcore1000,xcore1600
    probe_targets "f) native"          CUCC_TARGETS=native
    probe_targets "g) xcore1008"       CUCC_TARGETS=xcore1008
    probe_targets "h) xcore1502"       CUCC_TARGETS=xcore1502
    probe_targets "i) xcore1610"       CUCC_TARGETS=xcore1610
    probe_targets "j) xcore999"        CUCC_TARGETS=xcore999
    probe_targets "k) 1000,1000"       CUCC_TARGETS=xcore1000,xcore1000
    probe_targets "l) 1000 bare"       CUCC_TARGETS=1000
    probe_targets "m) XCORE1000"       CUCC_TARGETS=XCORE1000
    probe_targets "n) trailing comma"  CUCC_TARGETS=xcore1000,
    probe_targets "o) spaces around"   CUCC_TARGETS=' xcore1000 , xcore1600 '
    ;;
arch)
    cat > "$WORK/probe.cu" <<'EOF'
__global__ void probe_kernel(int *out) { out[threadIdx.x] = 1; }
EOF
    cat > "$WORK/arch.cu" <<'EOF'
#define STR2(x) #x
#define STR(x) STR2(x)
#if defined(__MACA_ARCH__)
#pragma message("PROBE __MACA_ARCH__=" STR(__MACA_ARCH__))
#else
#pragma message("PROBE __MACA_ARCH__ is UNDEFINED")
#endif
__global__ void arch_probe(int *out) { out[threadIdx.x] = 1; }
EOF
    # A second kernel, because the 2-instruction one cannot separate targets:
    # arch-specific lowering of the cross-lane primitives is where they differ.
    cat > "$WORK/k3.cu" <<'EOF'
__global__ void k3(const float *a, float *out, int n) {
  __shared__ unsigned s_hist[256];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  float v = (i < n) ? a[i] : 0.0f;
  unsigned mask = __activemask();
  float r = v;
  for (int d = 32; d > 0; d >>= 1) r += __shfl_down_sync(mask, r, d);
  unsigned long long bal = __ballot_sync((unsigned long long)mask, r > 0.5f);
  if ((i & 63) == 0) atomicAdd(&s_hist[((int)v) & 255], 1u);
  __syncthreads();
  if (i < n) out[i] = r + (float)__popcll(bal) + (float)s_hist[i & 255];
}
EOF
    printf '%-11s %-4s %-14s %-14s %s\n' spelling rc MACA_ARCH trivial_md5 crosslane_md5
    for a in xcore1000 xcore1002 xcore1004 xcore1008 \
             xcore1500 xcore1501 xcore1502 xcore1504 xcore1520 xcore1521 \
             xcore1600 xcore1610 xcore1620 native; do
        out=$("$CUCC" --offload-arch="$a" -c "$WORK/probe.cu" -o "$WORK/p.o" 2>&1); rc=$?
        if [ $rc -ne 0 ]; then
            printf '%-11s %-4s %s\n' "$a" "$rc" \
                "$(printf '%s' "$out" | grep -o 'invalid target ID[^)]*' | head -1)"
            continue
        fi
        aa=$("$CUCC" --offload-arch="$a" -c "$WORK/arch.cu" -o /dev/null 2>&1 \
             | grep -o '__MACA_ARCH__=[0-9]*' | head -1 | cut -d= -f2)
        "$CUCC" --offload-arch="$a" -aop -S -maca-device-only -x maca -O3 \
            -o "$WORK/t.s" "$WORK/probe.cu" >/dev/null 2>&1
        t=$(md5sum "$WORK/t.s" 2>/dev/null | cut -c1-12)
        "$CUCC" --offload-arch="$a" -aop -S -maca-device-only -x maca -O3 \
            -o "$WORK/c.s" "$WORK/k3.cu" >/dev/null 2>&1
        c=$(md5sum "$WORK/c.s" 2>/dev/null | cut -c1-12)
        printf '%-11s %-4s %-14s %-14s %s\n' "$a" "$rc" "$aa" "$t" "$c"
    done
    # Two spellings sharing a crosslane_md5 (and not the trivial one) is the
    # whole alias question: it says their device code is the same, not that
    # they are the same family -- `xcore1500` and `xcore1600` share one too.
    ;;
*)
    echo "usage: $0 {targets|arch}" >&2
    exit 2
    ;;
esac
