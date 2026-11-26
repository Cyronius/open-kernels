#!/usr/bin/env bash
# Rebuild the phlegm bridge and phlegm, then rewrite the bridge sidecar (2026-11-25).
D=/c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe
O=/c/models/qwen36moe/mtp
cd /c/models/qwen36moe/bridge && cmd //c 'C:\models\qwen36moe\bridge\build.cmd' > build11.log 2>&1 || { echo "bridge build failed"; grep -E "error C|fatal" build11.log | head; exit 1; }
cd /c/code/phlegm-qwen36moe && cargo build --release -p phlegm-cli > $O/cargo.log 2>&1 || { echo "cargo failed"; grep -E "^error" -A5 $O/cargo.log | head -20; exit 1; }
python -P $D/bridge_sidecar.py /c/code/open-kernels-qwen36moe /c/code/phlegm-qwen36moe /c/models/qwen36moe/bridge/phlegm-reference-bench.exe | tail -1
