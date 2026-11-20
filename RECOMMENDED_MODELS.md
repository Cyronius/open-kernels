# Recommended models: progress to the production bar

The top-tier models from OFLM [#240](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/240),
and how far each one is toward the production bar in `AGENTS.md`. OFLM owns the list. This file
tracks the kernel and engine side. The registry entry, `oflm add` / `serve`, `oflm bench`, the
benchmark pages and app features (reasoning parse, tool calling) go in separate OFLM PRs and are
listed per model under "OFLM side".

Update this file in the same commit as the evidence. Each model's full audit is a
`## Production bar` table in its spec; a model passes when that table has no open row.

## Status

`done` = measured and recorded; `open` = work left; `n/a` = does not apply, reason in the audit;
`?` = not audited yet.

| Model | Block prefill | Near optimal | Real context | All on NPU | Drafter | Quantized GGUF | Kernel side |
|---|---|---|---|---|---|---|---|
| flux2-klein:4b | n/a | done | n/a | done | n/a | done (Q8_0) | **passes** (Linux GGUF run open) |
| Granite 4.2 3B | done (`attn_fa` to 131k) | open (decode at its floor to 2k, 1.15x at 8k, 1.4x at 32k; prefill 2.2-2.9x the old default, 0.89-1.07x Vulkan) | done (131k runs; needle = llama.cpp to 64k) | done (host stages named) | done (lookup) | done (IBM Q4_1 GGUF) | open: decode attention past 8k ([audit](specs/open-engine/spec.md), OPEN-FAMILY-GRANITE) |
| Gemma 4 E2B / E4B | done for E4B text (#238); E2B not rerun | open (unprofiled; prefill fell 85 to 62 tok/s) | open (32k once, not repeated) | open (towers refused) | open (Unsloth MTP, prompt lookup) | open (Google QAT Q4_0; Unsloth reason to record) | [planned](specs/open-engine/plans/gemma4e-production-bar.md); vision, video, audio not open |
| BERT embeddings (bge x3, minilm, nomic, gte) | ? | ? | ? | ? | n/a | ? | not started |
| whisper-v3:turbo | ? | ? | ? | open (decoder on host) | ? | ? | not started |
| qwen3.6-moe:35b-a3b | ? | ? | open (32k test) | ? | open (DFlash planned) | ? | not started |
| gpt-oss:20b | ? | ? | open | ? | open | ? | not started (OFLM #237) |
| qwen3.5:4b (or :9b) | ? | open (prefill numbers) | open | ? | open | open | not started |
| Ternary Bonsai 2 27B | ? | ? | open (8k of 262k) | ? | done (DFlash 2 + lookup) | open (reads PQ2_0 via q4nx-build) | not started; work not on main |

The `open` and `done` cells for models with no audit come from #240's own notes.

## Per model

### flux2-klein:4b

Audit: [flux2-klein-native.md](specs/open-engine/flux2-klein-native.md#production-bar-oflm-agentsmd-240)
(2026-11-14). Kernel side passes: profiled with what is left named, every op on the NPU, Q8_0 GGUF
quality-gated against the reference, t2i and edits at 512 and 1024.

- Open: the GGUF config has not run on Linux.
- OFLM side: the registry entry names the prepacked `Cyronius/FLUX.2-klein-4B-NPU2` bundle and
  states `bfp16`; no `oflm serve` Images API pass; no benchmark page.

### Granite 4.2 3B

Audit: `spec.md` OPEN-FAMILY-GRANITE, "Production bar" (2026-11-19). Plan:
[granite-production-bar.md](specs/open-engine/plans/archive/granite-production-bar.md).

- Done: IBM's Q4_1 GGUF runs directly (multipliers folded at load); `attn_fa` covers every window to 131,072; runs
  complete at 120k and 130k; needle answers match llama.cpp at 32k and 64k, depths 10 / 50 / 90 %, on the default
  route; prompt lookup is SPEC IDENTICAL at 2k / 8k / 32k (1.5-2.3x on copy-heavy replies); Uno rejected as the
  default (q4nx sidecar, at most +0.03x); host stages named.
- Speed (2026-11-19, `g131072_v4`): prefill 666 / 727 / 428 / 162 tok/s at 512 / 2k / 8k / 32k (was 226 / 248 / 194 /
  103; Vulkan 747 / 680 / 457 / 157); decode 41.7 / 47.2 / 63.3 / 128.2 ms a step at depth 0 / 2k / 8k / 32k (was 41.8 /
  ~50 / 73.2 / 167.1). Needles and prompt lookup unchanged.
- Open (near optimal): decode attention past 8k (score trees and spills; an `mmul` scorer in `dx` and `dxl`); `attn_fa`
  arithmetic in long prefill; prefill host stages (fp64 residual tail, three-term qkv pack).
- OFLM side: registry entry (IBM Q4_1 GGUF, q4_1, granite, 131,072); `oflm add` / `serve` / `bench`; benchmark
  page; reasoning block not parsed; tool calling not checked.

### Gemma 4 E2B / E4B

Plan: [gemma4e-production-bar.md](specs/open-engine/plans/gemma4e-production-bar.md) (2026-11-18).
The text work is in OFLM draft #238 and is not in this repo yet. The model is not on #240's list yet.

- Done (E4B, #238): block prefill, KV sharing, per-layer inputs on the NPU, `oflm serve` text.
- Open: import #238; E2B on the NPU; 131,072 context; profile; drafter (Unsloth MTP); the vision
  tower, video and the audio tower on the NPU from the mmproj GGUF.
- OFLM side: registry entries with mmproj and drafter; video request shape; thinking-on answer
  lost to reasoning; tool calling on the open path; benchmark page.

### Others

Not audited. Local evidence to start from:

- BERT embeddings, whisper-v3:turbo: no spec here; FLM rows in
  [flm-kernel-matrix.md](specs/open-engine/flm-kernel-matrix.md).
- qwen3.6-moe:35b-a3b: OPEN-FAMILY-QWEN36MOE-GGUF;
  [dflash-drafters.md](specs/open-engine/plans/dflash-drafters.md).
- gpt-oss:20b: OPEN-FAMILY-GPTOSS, OPEN-PREFILL-GPTOSS;
  [gptoss-kernels.md](specs/open-engine/gptoss-kernels.md).
- qwen3.5:4b / 9b: OPEN-FAMILY-QWEN35;
  [qwen35-remaining-sizes.md](specs/open-engine/plans/qwen35-remaining-sizes.md).
- Ternary Bonsai 2 27B: OPEN-QUANT-T2, OPEN-CONVERT-PRISM-TERNARY;
  [bonsai2-dflash2.md](specs/open-engine/plans/archive/bonsai2-dflash2.md).

## Honorable mention

Other models we support and test, but do not recommend (#240 discussion). They are not top tier,
so the bar does not gate them.

| Model | Reason | Local evidence |
|---|---|---|
| Qwen3.8-27B | Removed from #240: decode about 2.2 tok/s on the open engine | [qwen38-main-rebaseline.md](specs/open-engine/plans/qwen38-main-rebaseline.md), [qwen35-27b-bringup.md](specs/open-engine/plans/qwen35-27b-bringup.md) |
