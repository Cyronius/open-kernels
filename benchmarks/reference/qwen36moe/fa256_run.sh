#!/usr/bin/env bash
# run attn_fa256 tests and compare: fa256_run.sh <testdir...> (2026-11-25)
. "$(dirname "$0")/paths.sh"
cd "$QW/fa256"
for t in "$@"; do
  echo "== $t"
  NPUQ_ETA=5m "$QD/npuq.sh" timeout 300 "$QRUNK" "$(qpath "$PWD/$t/run.cfg")" > $t/run.log 2>&1; echo "exit $?"
  grep "run k" $t/run.log | tail -3
  "$QPY" "$QR/open_kernels/designs/attn_fa256/compare.py" $t | tail -1
done
