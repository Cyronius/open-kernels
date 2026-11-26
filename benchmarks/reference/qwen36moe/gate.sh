#!/usr/bin/env bash
# Teacher-forced logits on the llama.cpp KL chunks: gate.sh <tag> <kernels> [chunks] [extra CLI args...] (2026-11-23).
set -u
tag=$1; kern=$2; n=${3:-8}; shift 3 || shift $#
CLI='C:\code\open-kernels-qwen36moe\src\open_qwen36\out\open_qwen36_cli.exe'
M='C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S'
KL=/c/models/qwen36moe/kl
mkdir -p $KL/$tag
for k in $(seq 0 $((n-1))); do
  [ -f "$KL/$tag/chunk${k}_t0.bin" ] && { echo "chunk $k kept"; continue; }
  NPUQ_ETA=10m /c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe/npuq.sh "$CLI" --model "$M" --kernels "$kern" \
    --ids-file "C:\models\qwen36moe\kl\ids\chunk$k.txt" ${GATE_ROUTE---gemm-block} --prefill-logits \
    --dump-logits "C:/models/qwen36moe/kl/$tag/chunk$k" --max-tokens 1 --max-ctx 1024 "$@" > $KL/$tag/chunk$k.log 2>&1
  echo "chunk $k exit $? $(grep -E 'prefill [0-9]+ tokens|ERROR' $KL/$tag/chunk$k.log | head -2 | tr '\n' ' ')"
done
python -P /c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe/kld.py score --base $KL/base_vk_c512.kld --dir $KL/$tag --chunks $n --json $KL/$tag/score.json
