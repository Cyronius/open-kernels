#!/usr/bin/env bash
# A long needle through phlegm serve (needle_serve.py): needle_serve.sh <tokens> [max_ctx] (2026-11-26).
# Run it inside `arbiter run npu` (phlegm takes its own NPU lock as well).
. "$(dirname "$0")/paths.sh"
O=$QW/ctx/serve; mkdir -p $O
N=$1; CTX=${2:-$((N + 1024))}; PORT=${PORT:-52634}
QWEN36_BRIDGE_STATS=1 "$QPHLEGM" serve --model-dir "$QM" --kernels "${KERN:-$QW/kernels-fa262k}" --bridge "$QBRIDGE" \
  --max-ctx $CTX --speculative ${SPEC:-mtp} --timeout 36000 --port $PORT > $O/serve_$N.log 2>&1 &
pid=$!
"$QPY" -P "$QD/needle_serve.py" http://127.0.0.1:$PORT $N | tee $O/needle_$N.txt
kids=$(pgrep -P $pid); kill $pid; wait $pid 2>/dev/null; [ -n "$kids" ] && kill $kids 2>/dev/null
grep -E "prompt [0-9]+ tokens|error|Error" $O/serve_$N.log | tail -3
