#!/usr/bin/env bash
# Kernel set with attn_fa256 at a 262144-row capacity (2026-11-25): export_fa262k.sh [out] [build-root] [extra args]
# A probe export (QWEN36MOE_PROBE_*) must get its own build root: the shared one would be overwritten.
. "$(dirname "$0")/paths.sh"
out=${1:-$QW/kernels-fa262k}; root=${2:-$QW/build}; shift 2 2>/dev/null || shift $#
cd "$QR" && "$QPY" open_kernels/export_qwen36_kernels.py --model-dir "$QM" --out "$out" --build-root "$root" \
  --max-ctx 262144 -j "${JOBS:-8}" "$@"
echo "export exit $?"
