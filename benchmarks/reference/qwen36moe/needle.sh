#!/usr/bin/env bash
# Needle runs through the CLI on the block route: needle.sh <kernels> <max_ctx> <prompt names...> (2026-11-24).
. "$(dirname "$0")/paths.sh"
D=$QW/ctx/needle
kern=$1; ctx=$2; shift 2
for n in "$@"; do
  NPUQ_ETA=${NPUQ_ETA:-60m} "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "$kern" \
    --ids-file "$(qpath "$D/$n.txt")" --gemm-block --max-tokens 12 --max-ctx "$ctx" \
    --timeout-ms 600000 > $D/$n.log 2>&1
  echo "$n exit $? $("$QPY" -P "$QD/needle.py" check $D/$n.log)"
done
