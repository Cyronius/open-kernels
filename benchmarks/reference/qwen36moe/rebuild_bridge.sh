#!/usr/bin/env bash
# Rebuild phlegm and its bridge from this tree's engine (2026-11-25; Linux 2026-11-26).
# Linux: cargo --features npu builds the bridge (phlegm build.py) from phlegm's open-kernels submodule, so the
# engine sources of this checkout are copied into it first; the sidecar then reads dirty (development builds).
. "$(dirname "$0")/paths.sh"
O=$QW/mtp; mkdir -p $O
if [ -z "${WINDIR:-}" ]; then
  for p in src/open_qwen36 src/include/nlohmann; do
    rsync -a --delete --exclude build --exclude out "$QR/$p/" "$QPH/open-kernels/$p/"
  done
  cp "$QR/open_kernels/harness/stream_patch.hpp" "$QPH/open-kernels/open_kernels/harness/stream_patch.hpp"
  cd "$QPH" && cargo build --release -p phlegm-cli --features npu > $O/cargo.log 2>&1 ||
    { echo "cargo failed"; grep -E "^error|error:" -A5 $O/cargo.log | head -30; exit 1; }
  ls "$QPHLEGM" "$QBRIDGE"
  exit 0
fi
cd /c/models/qwen36moe/bridge && cmd //c 'C:\models\qwen36moe\bridge\build.cmd' > build11.log 2>&1 || { echo "bridge build failed"; grep -E "error C|fatal" build11.log | head; exit 1; }
cd "$QPH" && cargo build --release -p phlegm-cli > $O/cargo.log 2>&1 || { echo "cargo failed"; grep -E "^error" -A5 $O/cargo.log | head -20; exit 1; }
"$QPY" -P "$QD/bridge_sidecar.py" "$QR" "$QPH" "$QBRIDGE" | tail -1
