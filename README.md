# open-kernels

**Open NPU kernels and a C++ engine for running LLMs on the AMD Ryzen AI (XDNA2) NPU.**

The kernels are written with the IRON / mlir-aie toolchain. They replace FastFlowLM's closed kernels and run whole
models on the NPU: Qwen3 / Qwen3.5 / Qwen3.6-MoE, Llama, Gemma 3 and 4, GPT-OSS, LFM2, Granite, Whisper and others.

This repository is used by:

- [phlegm](https://github.com/Cyronius/phlegm), which mounts it as a git submodule at `open-kernels/`.
  phlegm adds the Rust CLI, server, tokenizer, sampling and tray.
- [openflowlm-next](https://github.com/Atomic-Germ/openflowlm-next), which keeps the same paths and receives this
  work as pull requests. [`open_kernels/UPSTREAM.md`](open_kernels/UPSTREAM.md) records what moves between the two.

## Layout

| dir | what |
|---|---|
| [`open_kernels/`](open_kernels/) | AIE kernel designs, per-family recipes, the build harness and the fp64 replicas. See its README. |
| [`src/open_qwen36/`](src/open_qwen36/) | The C++ engine that reads a kernel set's `manifest.json` and runs it. Derived from OpenFlowLM. |
| [`npu_offload/`](npu_offload/) | Embedding GEMM sources and block-attention build dependencies. |
| [`viz/`](viz/) | Animated view of one decode step on the NPU (exporter in `open_kernels/viz/`). |
| [`utilities/`](utilities/) | Kernel export, replay, probe and audit tools. |
| [`specs/open-engine/`](specs/open-engine/) | Requirements (`OPEN-*`) and tests for the kernels, recipes and engine. |
| [`specs/viz/`](specs/viz/) | Requirements (`VIZ-*`) and tests for the viz. |

The paths match openflowlm-next, so patches and cherry-picks apply in both directions without path changes.

## Tests

The offline suite needs no NPU:

```sh
python -m pip install -r .github/ci-requirements.txt
python -m pytest -q -rs
```

Tests that need the NPU, XRT or a native compiler skip when those are absent. A skip is untested coverage, not a pass.

## History

This repository was split from phlegm on 2026-11-15 with `git filter-repo`. It keeps the full history of the moved
files, including the history imported from openflowlm-next.

## License

MIT. See [`LICENSE`](LICENSE) and [`NOTICE.md`](NOTICE.md) for third-party terms.
