@echo off
rem The bridge compile from phlegm benchmarks/reference/build.ps1, against the qwen36moe worktree (2026-11-23).
set "VCVARS64=%ProgramFiles(x86)%\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
call "%VCVARS64%" >nul
set "R=C:\code\open-kernels-qwen36moe"
set "E=%R%\src\open_qwen36"
cd /d C:\models\qwen36moe\bridge
cl /nologo /EHsc /O2 /MD /std:c++17 /Zc:__cplusplus /D_CRT_SECURE_NO_WARNINGS /DDISABLE_ABI_CHECK=1 /bigobj /openmp /arch:AVX2 /I "C:\dev\XRT\src\runtime_src\core\include" /I "%R%\src" /I "%R%\src\include" /I "%R%\open_kernels\harness" "%E%\core.cpp" "%E%\block_host.cpp" "%E%\pools.cpp" "%E%\manifest.cpp" "%E%\q4nx_file.cpp" "%E%\gguf_file.cpp" "%E%\omp_policy.cpp" "C:\code\phlegm-qwen36moe\benchmarks\reference\bridge.cpp" "C:\dev\xrtNPUfromDLL\xrt_coreutil.lib" /Fe:phlegm-reference-bench.exe /Fo:.\ /link /DELAYLOAD:VCOMP140.DLL delayimp.lib
exit /b %errorlevel%
