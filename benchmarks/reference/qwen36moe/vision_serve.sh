#!/usr/bin/env bash
# phlegm serve with an image conversation (vision_serve.py), speculative mode $SPEC (default mtp), 2026-11-26.
. "$(dirname "$0")/paths.sh"
O=$QW/vision/serve; mkdir -p $O
PORT=${PORT:-52633}
QWEN36_BRIDGE_STATS=1 QWEN36_BRIDGE_TRACE=${TRACE:-} "$QPHLEGM" serve --model-dir "$QM" --kernels "${KERN:-$QW/kernels-fa262k}" --bridge "$QBRIDGE" \
  --max-ctx ${CTX:-8192} --speculative ${SPEC:-mtp} --port $PORT > $O/serve_${SPEC:-mtp}.log 2>&1 &
pid=$!
"$QPY" -P "$QD/vision_serve.py" http://127.0.0.1:$PORT "${1:-$QW/vision/llama0-banner.png}" | tee $O/answers_${SPEC:-mtp}.txt
kill $pid; wait $pid 2>/dev/null; pkill -f "qwen36moe-bar/target/release/phlegm-reference-bench"
grep -E "bridge:|prompt [0-9]+ tokens|error|Error" $O/serve_${SPEC:-mtp}.log | tail -12
