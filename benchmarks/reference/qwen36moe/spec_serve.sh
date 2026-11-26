#!/usr/bin/env bash
# phlegm serve with --speculative off / lookup / mtp, each timed by spec_serve.py (2026-11-25; Linux 2026-11-26).
# phlegm takes its own NPU lock; run this inside `arbiter hold npu` (or npuq.sh) so other sessions wait.
. "$(dirname "$0")/paths.sh"
O=$QW/mtp; mkdir -p $O
CTX=${CTX:-4096}; PORT=${PORT:-52632}
sed -n '/^PROMPT=/,/Output only the code.'"'"'}/p' "$QD/spec_time.sh" | sed "1s/^PROMPT=\${PROMPT:-'//; \$s/'}\$//" > $O/rewrite_prompt.txt
for mode in ${MODES:-off lookup mtp}; do
  QWEN36_BRIDGE_STATS=1 "$QPHLEGM" serve --model-dir "$(qpath "$QM")" --kernels "${KERN:-$(qpath "$QW/kernels-v")}" \
    --bridge "$(qpath "$QBRIDGE")" --max-ctx $CTX --speculative $mode --port $PORT > $O/serve_$mode.log 2>&1 &
  pid=$!
  "$QPY" -P "$QD/spec_serve.py" http://127.0.0.1:$PORT $mode $O/rewrite_prompt.txt
  if [ -n "${WINDIR:-}" ]; then
    taskkill //T //F //PID "$(cat /proc/$pid/winpid)" > /dev/null 2>&1
    wait $pid 2>/dev/null
    # the bridge holds the NPU lock until it exits
    while tasklist //fo csv //nh 2>/dev/null | grep -qiE "phlegm(-reference-bench)?\.exe"; do sleep 2; done
    # a forced kill skips the lock release, and phlegm cannot see a dead owner on Windows
    rm -f "$(cygpath "$LOCALAPPDATA")/Temp/phlegm-npu.lock"
  else
    kill -INT $pid; wait $pid 2>/dev/null
    while pgrep -f phlegm-reference-bench > /dev/null; do sleep 2; done
  fi
done
for n in rewrite explain; do for m in lookup mtp; do cmp -s $O/serve_off_$n.txt $O/serve_${m}_$n.txt && echo "$n $m IDENTICAL" || echo "$n $m DIFFERS"; done; done
