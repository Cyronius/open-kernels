#!/usr/bin/env bash
# Verify pass against plain steps, bit for bit (OPEN-VERIFY-ROWS-QWEN36MOE, 2026-11-25).
. "$(dirname "$0")/paths.sh"
O=$QW/mtp; mkdir -p $O
OFLM_VERIFY_PROFILE=1 OFLM_OPEN_VERIFY=1 "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "${KERN:-$(qpath "$QW/kernels-v")}" \
  --max-ctx 4096 --gemm-block --ids-file "$(qpath "$QW/ctx/needle/short32.txt")" --verify-check ${1:-4} > $O/verify_check.log 2>&1
echo "exit $?"; grep -E "verify |verify:|commit [0-9]|layer|cycle|identical|DIFFER|VERIFY|ERROR|error|position" $O/verify_check.log | head -40
