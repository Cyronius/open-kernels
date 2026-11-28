@echo off
rem attn_fa256 build at one window (2026-11-25). Usage: build.cmd <L> <outdir>
call C:\dev\mlir-aie\iron_env.cmd
set AF_L=%1
set AF_MAX_ROWS=%3
set OFLM_KEEP_FAILED=1
cd /d C:\code\open-kernels-qwen36moe
python open_kernels\build_design.py open_kernels\designs\attn_fa256\attn_fa256.py %2
echo build exit %errorlevel%
