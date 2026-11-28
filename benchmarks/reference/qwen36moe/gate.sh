#!/usr/bin/env bash
# Teacher-forced logits on the llama.cpp KL chunks: gate.sh <tag> <kernels> [chunks] [extra CLI args...] (2026-11-23).
set -u
. "$(dirname "$0")/paths.sh"
tag=$1; kern=$2; n=${3:-8}; shift 3 || shift $#
KL=$QW/kl
mkdir -p $KL/$tag
for k in $(seq 0 $((n-1))); do
  [ -f "$KL/$tag/chunk${k}_t0.bin" ] && { echo "chunk $k kept"; continue; }
  NPUQ_ETA=10m "$QD/npuq.sh" "$QCLI" --model "$(qpath "$QM")" --kernels "$kern" \
    --ids-file "$(qpath "$KL/ids/chunk$k.txt")" ${GATE_ROUTE---gemm-block} --prefill-logits \
    --dump-logits "$(qpath "$KL/$tag/chunk$k")" --max-tokens 1 --max-ctx 1024 "$@" > $KL/$tag/chunk$k.log 2>&1
  echo "chunk $k exit $? $(grep -E 'prefill [0-9]+ tokens|ERROR' $KL/$tag/chunk$k.log | head -2 | tr '\n' ' ')"
done
"$QPY" -P "$QD/kld.py" score --base $KL/base_vk_c512.kld --dir $KL/$tag --chunks $n --json $KL/$tag/score.json
