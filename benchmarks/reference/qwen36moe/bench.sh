#!/usr/bin/env bash
# phlegm bench --development on Qwen3.6-35B-A3B UD-Q4_K_S (512 to 4096 input tokens, 128 output), under arbiter npu
# (2026-11-26): bench.sh [out.json] [notes]
. "$(dirname "$0")/paths.sh"
O=$QW/bench; mkdir -p $O
out=${1:-$O/phlegm_bench_$(date +%Y%m%d_%H%M).json}
notes=${2:-"NPU power mode Default (pmode needs root on this box); shared machine, other sessions on CPU/GPU"}
NPUQ_ETA=${NPUQ_ETA:-25m} "$QD/npuq.sh" "$QPHLEGM" bench --development --reference-engine "$QBRIDGE" --model-dir "$QM" \
  --kernels "${KERN:-$QW/kernels-fa262k}" --model-id unsloth/Qwen3.6-35B-A3B-MTP-GGUF \
  --model-revision 5bc3e238d916f48a861bac2f8a1990a0e9b7e98d --quantization UD-Q4_K_S --context ${CTX:-8192} \
  --notes "$notes" --output "$out" > "${out%.json}.log" 2>&1
echo "exit $?"
tail -12 "${out%.json}.log"
