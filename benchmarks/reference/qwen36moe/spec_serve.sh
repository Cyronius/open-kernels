#!/usr/bin/env bash
# phlegm serve with --speculative off / lookup / mtp, each timed by spec_serve.py (2026-11-25).
P=/c/code/phlegm-qwen36moe/target/release/phlegm.exe
D=/c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe
O=/c/models/qwen36moe/mtp
sed -n '/^PROMPT=/,/Output only the code.'"'"'}/p' $D/spec_time.sh | sed "1s/^PROMPT=\${PROMPT:-'//; \$s/'}\$//" > $O/rewrite_prompt.txt
for mode in ${MODES:-off lookup mtp}; do
  QWEN36_BRIDGE_STATS=1 "$P" serve --model-dir 'C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S' --kernels 'C:\models\qwen36moe\kernels-v' \
    --bridge 'C:\models\qwen36moe\bridge\phlegm-reference-bench.exe' --max-ctx 4096 --speculative $mode --port 52632 > $O/serve_$mode.log 2>&1 &
  pid=$!
  python -P $D/spec_serve.py http://127.0.0.1:52632 $mode $O/rewrite_prompt.txt
  taskkill //T //F //PID "$(cat /proc/$pid/winpid)" > /dev/null 2>&1
  wait $pid 2>/dev/null
  # the bridge holds the NPU lock until it exits
  while tasklist //fo csv //nh 2>/dev/null | grep -qiE "phlegm(-reference-bench)?\.exe"; do sleep 2; done
  # a forced kill skips the lock release, and phlegm cannot see a dead owner on Windows
  rm -f "$(cygpath "$LOCALAPPDATA")/Temp/phlegm-npu.lock"
done
for n in rewrite explain; do for m in lookup mtp; do cmp -s $O/serve_off_$n.txt $O/serve_${m}_$n.txt && echo "$n $m IDENTICAL" || echo "$n $m DIFFERS"; done; done
