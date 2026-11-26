#!/usr/bin/env bash
# Greedy tokens plain against MTP draft + verify pass, compared and timed (OPEN-SPEC-MTP-QWEN36MOE, 2026-11-25).
CLI='C:\code\open-kernels-qwen36moe\src\open_qwen36\out\open_qwen36_cli.exe'
O=/c/models/qwen36moe/mtp
OFLM_OPEN_MTP=1 OFLM_OPEN_VERIFY=1 "$CLI" --model 'C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S' --kernels 'C:\models\qwen36moe\kernels-v' \
  --max-ctx 4096 --gemm-block --ids-file ${IDS:-C:/models/qwen36moe/ctx/needle/short32.txt} --spec-mtp ${1:-128} > $O/spec_mtp${TAG}.log 2>&1
echo "exit $?"; grep -E "plain|per cycle|SPEC|ERROR|error" $O/spec_mtp${TAG}.log | tail -6
