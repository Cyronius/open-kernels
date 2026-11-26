#!/usr/bin/env bash
# Greedy tokens plain against MTP draft + verify pass, compared and timed (OPEN-SPEC-MTP-QWEN36MOE, 2026-11-25).
. "$(dirname "$0")/paths.sh"
O=$QW/mtp; mkdir -p $O
OFLM_OPEN_MTP=1 OFLM_OPEN_VERIFY=1 "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "${KERN:-$(qpath "$QW/kernels-v")}" \
  --max-ctx 4096 --gemm-block --ids-file "${IDS:-$(qpath "$QW/ctx/needle/short32.txt")}" --spec-mtp ${1:-128} > $O/spec_mtp${TAG:-}.log 2>&1
echo "exit $?"; grep -E "plain|per cycle|SPEC|ERROR|error" $O/spec_mtp${TAG:-}.log | tail -6
