#!/usr/bin/env bash
# attn_fa256 build at one window (2026-11-25): fa256_build.sh <L> <outdir> <max_rows>
. "$(dirname "$0")/paths.sh"
cd "$QR" && AF_L=$1 AF_MAX_ROWS=$3 OFLM_KEEP_FAILED=1 "$QPY" open_kernels/build_design.py open_kernels/designs/attn_fa256/attn_fa256.py "$2"
echo "build exit $?"
