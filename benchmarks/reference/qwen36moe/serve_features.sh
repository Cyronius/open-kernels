#!/usr/bin/env bash
# phlegm serve: thinking on / off, streaming, a tool round trip (serve_features.py), 2026-11-26. Run inside arbiter npu.
. "$(dirname "$0")/paths.sh"
O=$QW/serve; mkdir -p $O
PORT=${PORT:-52635}
"$QPHLEGM" serve --model-dir "$QM" --kernels "${KERN:-$QW/kernels-fa262k}" --bridge "$QBRIDGE" --max-ctx 8192 \
  --speculative ${SPEC:-mtp} --port $PORT > $O/serve_features.log 2>&1 &
pid=$!
"$QPY" -P "$QD/serve_features.py" http://127.0.0.1:$PORT | tee $O/features.txt
kids=$(pgrep -P $pid); kill $pid; wait $pid 2>/dev/null; [ -n "$kids" ] && kill $kids 2>/dev/null
