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
| qwen3.6-moe:35b-a3b | done (layer-major block route; image prompts too) | done, rest named (MoE passes pipelined, DeltaNet host fused, `attn_fa256`; decode 1.5x its stream floor: lx / ax context switches, routed-expert round trips) | done (needle at 262,000, also through `phlegm serve` in 36 min; `attn_fa256` 6x at 262k) | done (host stages named; vision tower on the host) | done (MTP head + two-row verify, 1.24-1.40x) | done (Unsloth UD-Q4_K_S, q8 projections kept) | **passes** ([audit](specs/open-engine/spec.md), OPEN-FAMILY-QWEN36MOE-GGUF); benches are development runs |
| gpt-oss:20b | done (`attn_fa_gx` to 131k) | done, rest named (decode 82-84 ms short; 409 ms a step at 131k, compute-bound over a 170 ms stream floor) | done (needle at 130,500) | done (host stages named) | done (measured rejection: prompt lookup, DFlash) | done (ggml-org MXFP4 GGUF) | **passes** ([audit](specs/open-engine/spec.md), OPEN-FAMILY-GPTOSS); bench is a development run |
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

### gpt-oss:20b

Audit: `spec.md` OPEN-FAMILY-GPTOSS, "Production bar: gpt-oss:20b" (2026-11-21). Branch `gptoss-att8`.

- Done: ggml-org's MXFP4 GGUF runs directly; decode is one merged core image (attention and experts, 16 attention cores
  with KV windows, 8-row blocks); prefill attention on the NPU (`attn_fa_gx`: 64 heads, learned sink, runtime window)
  to 131,072 rows; the needle answers at 32k, 65k and 130.5k; YaRN ramp bounds as llama.cpp's; 32 of 32 greedy tokens
  equal llama.cpp's; `phlegm run` / `serve` with reasoning effort and tool calls; host stages named.
- Speed (`phlegm bench --development`, 2026-11-21): prefill 224 / 276 / 295 / 296 tok/s and decode 11.7 / 11.4 / 11.1 /
  10.7 tok/s at 512 / 1k / 2k / 4k. Prefill 177 s at 32k, 1882 s at 130.5k; decode 457 ms a token at 130.5k.
- Drafter: rejected with measurements. A rows pass costs 1.6 decode steps for 2 tokens (expert unions); DFlash accepts
  2.67 tokens a cycle offline. EAGLE3 not tried.
- Open (optional): decode attention at depth (`ATTN_G5` cuts the step 28 % but gives wrong logits at this shape); a tagged phlegm release
  for a publishable bench report.
- OFLM side: registry entry (MXFP4 GGUF, gptoss, 131,072); `oflm add` / `serve` / `bench`; benchmark page; README row.

### qwen3.6-moe:35b-a3b

Audit: `spec.md` OPEN-FAMILY-QWEN36MOE-GGUF, "Production bar: qwen3.6-moe:35b-a3b" (2026-11-24, updated 2026-11-26).
Plans: [qwen36moe-production-bar.md](specs/open-engine/plans/qwen36moe-production-bar.md),
[qwen36moe-attn-fa256.md](specs/open-engine/plans/qwen36moe-attn-fa256.md), handoff
[qwen36moe-handoff.md](specs/open-engine/plans/qwen36moe-handoff.md).

- Done: Unsloth's UD-Q4_K_S GGUF runs directly (Q4_K experts exact, Q8_0 projections at q8, the q8 shared expert on
  the host in decode), KL 0.0235 against llama.cpp Vulkan (Strix Point; backend spread 0.019) and 0.0278 against
  llama.cpp CPU (Strix Halo, another text); layer-major block prefill; needles at 4k, 8k, 32k, 131k and 262,000;
  `attn_fa256` (the full layers' block attention in one NPU dispatch: 262k prefill 6x); the MTP drafter on a
  two-row verify pass (`phlegm serve --speculative mtp` 1.24-1.40x, output identical to `off`); vision (the mmproj
  tower on the host, image prompts on the block route, `phlegm serve` image parts, the test image answered word for
  word as llama.cpp answers it); `phlegm run` / `serve` with reasoning and a tool call; host stages named; lost
  routed dispatches recovered on a new hardware context (OPEN-ROUTE-REDO). Runs on Linux (Strix Halo) and Windows.
- Speed (Strix Point, Windows, performance mode, `phlegm bench --development`, 2026-11-24): prefill 156 / 180 / 188 /
  188 tok/s and decode 9.1 / 8.9 / 8.6 / 8.1 tok/s at 512 / 1k / 2k / 4k; with `attn_fa256` 4k 218 tok/s, 32k 193,
  262k 107 (CLI). Strix Halo, Linux, Default power mode on a shared box: see the audit.
- Left, named in the audit: decode at 1.5x its 56 ms stream floor (20 lx / ax context switches a step, the routed
  experts' h round trip, the q8 front at 38 GB/s); the DeltaNet conv and delta rule and the vision tower on the host;
  the token-batched expert kernel core-bound at 8 tokens a visit; `attn_fa256` at 2.3x its transfer floor.
- OFLM side: registry entry (Unsloth MTP GGUF, `UD-Q4_K_S`, qwen35moe, 262,144, mmproj and MTP drafter); `oflm add` /
  `serve` / `bench`; benchmark page; README row.

### Others

Not audited. Local evidence to start from:

- BERT embeddings, whisper-v3:turbo: no spec here; FLM rows in
  [flm-kernel-matrix.md](specs/open-engine/flm-kernel-matrix.md).
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
