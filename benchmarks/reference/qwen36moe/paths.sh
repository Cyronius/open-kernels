# Paths for the qwen36moe helper scripts (2026-11-26). Source it; override any variable from the environment.
# Linux (Strix Halo box) and Windows git-bash (the old Strix Point box) defaults.
if [ -n "${WINDIR:-}" ]; then
  : "${QR:=/c/code/open-kernels-qwen36moe}"          # this repo
  : "${QPH:=/c/code/phlegm-qwen36moe}"               # phlegm feat/qwen36moe-bar
  : "${QW:=/c/models/qwen36moe}"                     # work root: kernels, logs, kl, ctx
  : "${QCLI:=$QR/src/open_qwen36/out/open_qwen36_cli.exe}"
  : "${QRUNK:=/c/code/open-kernels/open_kernels/harness/out/run_kernel.exe}"
  : "${QPHLEGM:=$QPH/target/release/phlegm.exe}"
  : "${QBRIDGE:=$QW/bridge/phlegm-reference-bench.exe}"
  : "${QPY:=python}"
  qpath() { cygpath -w "$1"; }                       # a path as the native tools want it
else
  : "${QR:=$HOME/code/open-kernels}"
  : "${QPH:=$HOME/code/phlegm-wt/qwen36moe-bar}"
  : "${QW:=$HOME/models/qwen36moe}"
  : "${QCLI:=$QW/cli/open_qwen36_cli}"
  : "${QRUNK:=$QW/harness/run_kernel}"
  : "${QPHLEGM:=$QPH/target/release/phlegm}"
  : "${QBRIDGE:=$QPH/target/release/phlegm-reference-bench}"   # cargo --features npu builds it beside phlegm
  : "${QPY:=$HOME/ironenv142/bin/python}"
  # the build's tools: the venv's python (build_design.py runs `python`), and aiebu-asm, which the distro XRT
  # package lacks (any XRT build's bin directory has it)
  : "${QXRTBIN:=$HOME/code/phlegm/target/npu-dev/xrt/bin}"
  case ":$PATH:" in *":$(dirname "$QPY"):"*) ;; *) PATH="$(dirname "$QPY"):$PATH:$QXRTBIN" ;; esac
  export PATH
  qpath() { printf '%s' "$1"; }
fi
: "${QM:=$QW/Qwen3.6-35B-A3B-UD-Q4_K_S}"             # model dir: model.gguf, mmproj, tokenizer, config
QD="$QR/benchmarks/reference/qwen36moe"
export QR QPH QW QCLI QRUNK QPHLEGM QBRIDGE QPY QM QD
