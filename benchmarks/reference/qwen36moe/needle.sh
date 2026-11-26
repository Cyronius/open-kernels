#!/usr/bin/env bash
# Needle runs through the CLI on the block route: needle.sh <kernels> <max_ctx> <prompt names...> (2026-11-24).
CLI='C:\code\open-kernels-qwen36moe\src\open_qwen36\out\open_qwen36_cli.exe'
M='C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S'
D=/c/models/qwen36moe/ctx/needle
kern=$1; ctx=$2; shift 2
for n in "$@"; do
  NPUQ_ETA=${NPUQ_ETA:-60m} /c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe/npuq.sh "$CLI" --model "$M" --kernels "$kern" \
    --ids-file "C:/models/qwen36moe/ctx/needle/$n.txt" --gemm-block --max-tokens 12 --max-ctx "$ctx" \
    --timeout-ms 600000 > $D/$n.log 2>&1
  echo "$n exit $? $(python -P /c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe/needle.py check $D/$n.log)"
done
