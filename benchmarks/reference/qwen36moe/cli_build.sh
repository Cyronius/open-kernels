#!/usr/bin/env bash
# The CLI and its unit tests on Linux, against the system XRT (2026-11-26). Windows: cli_build.cmd.
. "$(dirname "$0")/paths.sh"
mkdir -p "$QW/cli"
cmake -S "$QR/src/open_qwen36" -B "$QW/cli" -DCMAKE_BUILD_TYPE=Release \
  -DXRT_INCLUDE_DIR="${XRT_INCLUDE_DIR:-/usr/include}" -DXRT_LIB_DIR="${XRT_LIB_DIR:-/usr/lib}" > "$QW/cli/configure.log" 2>&1 ||
  { tail -20 "$QW/cli/configure.log"; exit 1; }
cmake --build "$QW/cli" -j "${JOBS:-16}" && ctest --test-dir "$QW/cli" --output-on-failure
