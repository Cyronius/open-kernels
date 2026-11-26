#!/usr/bin/env bash
# Run a command under arbiter npu once no other loaded NPU client is alive (2026-11-23; Linux 2026-11-26).
# Another process on the NPU can hang this run's queue (F-0185), so foreign contexts seen during the run are logged too.
# NPUQ_MEASURE=1 takes the whole machine (arbiter --measure) for benchmark numbers.
if [ -z "${WINDIR:-}" ]; then
  # Linux: arbiter run makes its own process group, so it waits, runs and releases in one call
  m=(); [ -n "${NPUQ_MEASURE:-}" ] && m=(--measure)
  exec arbiter run npu "${m[@]}" --label "${NPUQ_LABEL:-qwen36moe}" --eta "${NPUQ_ETA:-20m}" -- bash -c '
    xrt-smi examine -r aie-partitions 2>&1 | grep -q "No hardware contexts" ||
      echo "npuq $(date +%H:%M:%S): NPU contexts already open when the run started" >&2
    exec "$@"' npuq "$@"
fi
others() {
  # any process with an NPU hardware context, except Windows' own (their contexts sit idle on the NPU)
  /c/Windows/System32/AMD/xrt-smi.exe examine -r aie-partitions 2>&1 | grep -oE '[A-Za-z0-9_.-]+\.exe' |
    grep -viE '^(WorkloadsSessionHost|svchost|FrameServer|xrt-smi)\.exe$' | sort -u | tr '
' ' '
}
# `arbiter run` cannot make its Windows Job Object inside this session, so hold and release around the command
# hold first: the hold can wait behind a session whose NPU client is still alive when it is granted
arbiter hold npu --label qwen36moe --eta "${NPUQ_ETA:-20m}" >&2
i=0; quiet=0
while [ "$quiet" -lt 3 ]; do
  o=$(others)
  if [ -z "$o" ]; then quiet=$((quiet + 1)); else
    quiet=0
    [ $((i % 60)) = 0 ] && echo "npuq $(date +%H:%M): waiting for loaded NPU clients: $o" >&2
    i=$((i + 1))
  fi
  sleep 10
done
echo "npuq $(date +%H:%M): running $*" >&2
( while sleep 3; do o=$(others | sed 's/open_qwen36_cli\.exe//; s/phlegm-reference-bench\.exe//; s/phlegm\.exe//');
    [ -n "${o// /}" ] && echo "npuq $(date +%H:%M:%S): FOREIGN NPU CLIENT during the run: $o" >&2; done ) &
watch=$!
"$@"
rc=$?
arbiter release npu >&2
kill $watch 2>/dev/null
echo "npuq $(date +%H:%M): exit $rc" >&2
exit $rc
