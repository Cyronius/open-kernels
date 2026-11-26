@echo off
rem kernel set with attn_fa256 at an 262144-row capacity, for the short-context gates (2026-11-25)
call C:\dev\mlir-aie\iron_env.cmd
cd /d C:\code\open-kernels-qwen36moe
python open_kernels\export_qwen36_kernels.py --model-dir C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S --out C:\models\qwen36moe\kernels-fa262k --build-root C:\models\qwen36moe\build --max-ctx 262144 -j 4
echo export exit %errorlevel%
