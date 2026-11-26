#!/usr/bin/env bash
# Verify pass against plain steps, bit for bit (OPEN-VERIFY-ROWS-QWEN36MOE, 2026-11-25).
CLI='C:\code\open-kernels-qwen36moe\src\open_qwen36\out\open_qwen36_cli.exe'
O=/c/models/qwen36moe/mtp
OFLM_VERIFY_PROFILE=1 OFLM_OPEN_VERIFY=1 "$CLI" --model 'C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S' --kernels 'C:\models\qwen36moe\kernels-v' \
  --max-ctx 4096 --gemm-block --ids-file C:/models/qwen36moe/ctx/needle/short32.txt --verify-check ${1:-4} > $O/verify_check.log 2>&1
echo "exit $?"; grep -E "verify |verify:|commit [0-9]|layer|cycle|identical|DIFFER|VERIFY|ERROR|error|position" $O/verify_check.log | head -40
