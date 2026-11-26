#!/usr/bin/env bash
# Block-prefill per-layer profile at several prompt lengths: prefill_prof.sh <kernels> <prompt names...> (2026-11-25).
. "$(dirname "$0")/paths.sh"
O=$QW/ctx/prof
N=$QW/ctx/needle
mkdir -p $O
kern=$1; shift
for n in "$@"; do
  len=$("$QPY" -P -c "import sys; print(len(open(sys.argv[1]).read().replace(',', ' ').split()))" "$N/$n.txt")
  rm -f $O/$n.jsonl
  OFLM_OPEN_PROFILE="$(qpath "$O/$n.jsonl")" NPUQ_ETA=${NPUQ_ETA:-20m} "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "$kern" \
    --ids-file "$(qpath "$N/$n.txt")" --gemm-block --max-tokens 1 --max-ctx $((len + 512)) \
    --timeout-ms 600000 > $O/$n.log 2>&1
  echo "$n ($len tokens) exit $?: $(grep -E '^prefill ' $O/$n.log)"
done
"$QPY" -P "$QD/prefill_prof.py" $O "$@"
