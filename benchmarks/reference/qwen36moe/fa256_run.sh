# run attn_fa256 tests and compare (2026-11-25)
cd /c/models/qwen36moe/fa256
for t in "$@"; do
  echo "== $t"
  timeout 300 /c/code/open-kernels/open_kernels/harness/out/run_kernel.exe "$(cygpath -w $PWD/$t/run.cfg)" > $t/run.log 2>&1; echo "exit $?"
  grep "run k" $t/run.log | tail -3
  python /c/code/open-kernels-qwen36moe/open_kernels/designs/attn_fa256/compare.py $t | tail -1
done
