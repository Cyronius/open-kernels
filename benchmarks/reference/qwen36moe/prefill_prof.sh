#!/usr/bin/env bash
# Block-prefill per-layer profile at several prompt lengths: prefill_prof.sh <kernels> <prompt names...> (2026-11-25).
CLI='C:\code\open-kernels-qwen36moe\src\open_qwen36\out\open_qwen36_cli.exe'
M='C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S'
O=/c/models/qwen36moe/ctx/prof
mkdir -p $O
kern=$1; shift
for n in "$@"; do
  len=$(python -P -c "import sys; print(len(open(sys.argv[1]).read().replace(',', ' ').split()))" "C:/models/qwen36moe/ctx/needle/$n.txt")
  rm -f $O/$n.jsonl
  OFLM_OPEN_PROFILE="C:/models/qwen36moe/ctx/prof/$n.jsonl" "$CLI" --model "$M" --kernels "$kern" \
    --ids-file "C:/models/qwen36moe/ctx/needle/$n.txt" --gemm-block --max-tokens 1 --max-ctx $((len + 512)) \
    --timeout-ms 600000 > $O/$n.log 2>&1
  echo "$n ($len tokens) exit $?: $(grep -E '^prefill ' $O/$n.log)"
done
python -P /c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe/prefill_prof.py $O "$@"
