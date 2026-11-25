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
| Granite 4.2 3B | done (`attn_fa` to 131k) | done (profiled; decode 19% over the measured window stream at 32k, prefill 0.89-1.07x Vulkan) | done (130k runs; needle = llama.cpp to 64k) | done (host stages named) | done (lookup) | done (IBM Q4_1 GGUF) | **passes** ([audit](specs/open-engine/spec.md), OPEN-FAMILY-GRANITE) |
| Gemma 4 E2B / E4B | done for E4B text (#238); E2B not rerun | open (unprofiled; prefill fell 85 to 62 tok/s) | open (32k once, not repeated) | open (towers refused) | open (Unsloth MTP, prompt lookup) | open (Google QAT Q4_0; Unsloth reason to record) | [planned](specs/open-engine/plans/gemma4e-production-bar.md); vision, video, audio not open |
| BERT embeddings (bge x3, minilm, nomic, gte) | ? | ? | ? | ? | n/a | ? | not started |
| whisper-v3:turbo | ? | ? | ? | open (decoder on host) | ? | ? | not started |
| qwen3.6-moe:35b-a3b | ? | ? | open (32k test) | ? | open (DFlash planned) | ? | not started |
| gpt-oss:20b | done (`attn_fa_gx` to 131k) | done, rest named (decode 82-84 ms short; 409 ms a step at 131k, compute-bound over a 170 ms stream floor) | done (needle at 130,500) | done (host stages named) | done (measured rejection: prompt lookup, DFlash) | done (ggml-org MXFP4 GGUF) | **passes** ([audit](specs/open-engine/spec.md), OPEN-FAMILY-GPTOSS); bench is a development run |
| Inkling 2 Mini | 256-token Phlegm blocks exercised; near-range block attention on the NPU | open (8K Phlegm prefill 23.9–28.4 to 40.1–41.6 tok/s in both orders; 4K decode 423 to 271 ms/token) | open (clean 131K recall in 89 min of profiled prefill, 1,048,576 target) | open | open (DSpark opt-in, lookup slower) | open (IQ4_NL candidate, 8K perplexity, local prompts and ARC samples compared) | [in development](specs/open-engine/plans/inkling2-mini-production-bar.md), outside #240 |
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
Kernel-side production-bar work merged in [open-kernels #4](https://github.com/Cyronius/open-kernels/pull/4).

- Done: IBM's Q4_1 GGUF runs directly (multipliers folded at load); `attn_fa` covers every window to 131,072; runs
  complete at 120k and 130k; needle answers match llama.cpp at 32k and 64k, depths 10 / 50 / 90 %, on the default
  route; prompt lookup is SPEC IDENTICAL at 2k / 8k / 32k (1.5-2.3x on copy-heavy replies); Uno rejected as the
  default (q4nx sidecar, at most +0.03x); host stages named.
- Speed (2026-11-19, `g131072_v4`): prefill 666 / 727 / 428 / 162 tok/s at 512 / 2k / 8k / 32k (was 226 / 248 / 194 /
  103; Vulkan 747 / 680 / 457 / 157); decode 41.7 / 47.2 / 63.3 / 128.2 ms a step at depth 0 / 2k / 8k / 32k (was 41.8 /
  ~50 / 73.2 / 167.1). Needles and prompt lookup unchanged.
- Profiled headroom: decode at 32k takes 128.1 ms per step against a measured 107.7 ms window-stream
  floor. The audit also records the long-prefill attention arithmetic and host costs (fp64 residual tail and three-term QKV pack).
- OFLM side: [#209](https://github.com/Atomic-Germ/OpenFlowLM-Next/pull/209) recorded serving with and without speculation.
  The IBM Q4_1 GGUF route still needs a registry entry (q4_1, granite, 131,072), `oflm add` / `serve` / `bench`
  evidence, a benchmark page and README row, reasoning parsing, and a tool-calling check.

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

### Inkling 2 Mini

- IQ4_NL and Q8_0 experts run on the NPU during block prefill and decode, alongside the Q8 logits head.
  The version-4 five-token block followed by NPU decode matched 64 llama.cpp greedy IDs on the Ryzen AI 9 HX PRO 370.
  Sharing the expert layout cut measured bridge NPU memory from 32.70 to 15.54 GiB at short context.
  `phlegm run` and plain and streamed `phlegm serve` passed text smoke tests.
  [Production audit](specs/open-engine/plans/inkling2-mini-production-bar.md#production-bar).
- On the 48 sampled ARC prompts, Phlegm NPU IQ4_NL and all three llama.cpp CPU quants matched
  every answer choice from the BF16 CPU reference. All routes scored 44/48.
  [Quant comparison](specs/open-engine/plans/inkling2-mini-production-bar.md#candidate-quant-comparison).
- The optional relative-position attention route passed the isolated million-key test and phlegm text smoke.
  Sharing attention and expert contexts cut short warm decode from 11.06 to 7.43 s over 16 steps.
  The earlier bundle differs from llama.cpp at greedy token 59 on that prompt.
- Optional grouped expert prefill reduced the recorded 1,024-token phlegm prefill from 227.22 to 187.27 s
  with bit-identical boundary logits. The 128-token ranking flipped when the bundle order was reversed.
- Version 8 adds split-scale Q8 attention projections for block prefill. It ran the 1,024-token prefix
  in 78.71 s, halved the paired warm 128-token time and matched all 64 llama.cpp short greedy IDs.
- The fused Q8_0 host projections preserved the CPU target's logits and cut its nine-token decode time
  by about 24% in both test orders. With the new batch path, warm Phlegm decode for eight steps fell
  from 4.76–5.30 s to 2.48–2.75 s with the same greedy IDs in both orders.
- In the ordered 512-token comparison, the v11 development bridge took 6.66–7.97× Vulkan's prefill
  time and 3.83–6.24× its decode time. The fixed trajectory matched 32 of 33 greedy predictions.
  A subsequent NPU logit capture matched all 33 llama.cpp CPU predictions. Tiled host F32
  projections cut warm prefill by 6.0–8.1% in a newer 512-token profile with identical logits.
- The integrated launch-overlap route cut warm 512-token prefill from 23.17 to 14.11 seconds in
  both test orders. One matched 8,225-token pair fell from 510.19 to 295.96 seconds. Both
  comparisons preserved the final logits bit for bit.
  [Prefill evidence](specs/open-engine/plans/inkling2-mini-production-bar.md#host-projections).
- Phlegm passed short text inference at none, low, high and max effort and a plain and streamed tool-call round trip.
- `phlegm serve` passed 8K and 32K recall at 10%, 50% and 90% depth with the default bridge timeout.
  The three 32K runs took 61.5, 49.0 and 64.8 minutes respectively. The 131K midpoint returned the
  exact key. Its 12 h 59 min 55 s of wall time included two standby-correlated gaps totaling
  8 h 12 min 40 s. The 4 h 47 min 15 s outside the gaps include request overhead and decode.
  The 131K 10% retry also returned the exact key. Full prefill was first observed at 3 h 47 min
  and the response completed at 3 h 50 min, with brief standby intervals. The 90% recall returned
  `ONYX-3179` exactly. The observer saw full prefill at 4 h 6 min, and the response finished at
  4 h 9 min. Three power events exclude that interval from clean timing. A subsequent packed-key
  run returned the exact 90%-depth key after 89 min 27 s of profiled prefill, with zero power and
  PCI context events during the request. The 1M run waits for a faster prefill path and validation
  of its memory footprint.
  The 18,465-token candidate matched all 33 terminal greedy IDs and missed the 5% speed
  threshold in both run orders, so it was rejected.
  [Long-context evidence](specs/open-engine/plans/inkling2-mini-production-bar.md#phlegm-text-integration).
- Native host vision matched the pinned GGUF reference above 0.9999 correlation through 1,792 px. Standalone
  NPU vision attention and projection probes passed, including the 42,437-key stream for 4,096 px.
  Phlegm passed the NPU image smoke in JSON and SSE. The v9 tower missed the 0.9999 correlation gate at
  896, 1,792 and 4,096 px. The assembled v11 runtime with split keys and values and FP16 input rounding
  passed all seven fixtures, including 0.99990436 at 4,096 px in 345.81 s. Phlegm passed all eight expanded
  image QA cases. Full-resolution OCR kept 10,609 image rows and returned the expected key in 1,787.77 s.
  Two CPU Phlegm image requests matched plain decode with DSpark and accepted 53 of 91 proposals.
  The NPU pair matched the first six image responses and accepted 138 of 227 DSpark proposals. It hit an
  expert dispatch timeout on the 1,792-pixel case with DSpark enabled. The fresh-server retry matched the
  saved off response and accepted 28 of 41 proposals. The fresh 4,096-pixel DSpark OCR run matched the
  saved plain-decode stream and accepted 35 of 46 proposals. Its Windows Modern Standby event excludes
  that run from performance comparisons.
- Inkling's CPU speculative verification and DSpark feature captures matched direct replay at every
  eight-token acceptance length and in four rollback cases around the sliding-window boundary.
  The corrected Phlegm NPU shared image matched direct decode bit for bit in all 18 forced
  acceptance and rollback cases at 5- and 1,023-token prefixes, including the subsequent draft
  proposals and target logits.
  The Unsloth DSpark CPU oracle consumed captured features and matched a six-token greedy replay.
  Native CPU DSpark matched all 21 oracle proposals using the same borrowed weights and target features.
  The AVX2 Q8_0 draft head cut its warmed projection time from 55.01 to 10.05 ms and matched all 42
  scalar proposal IDs across short prompts and the 1,024-token draft window boundary.
  The small-block Q8_0 projection also preserved all draft logits. Paired CPU replay measured
  1.16–1.48× faster full drafts across six positions. Phlegm throughput for that change remains open.
  Both Phlegm DSpark builds preserved the short streamed outputs. The optimized pair had 1.29× code
  throughput and 0.80× repetition throughput compared with plain decode.
  Phlegm also matched all 24 greedy output IDs with and without DSpark after both a five-token prefix
  and a 1,023-token prefix that crosses the sliding window during decode.
  The corrected bundle matched plain and speculative HTTP streams in both short workloads. It
  sped up code continuation and slowed repetition in the off-first screening run. DSpark remains
  opt-in. Phlegm prompt lookup matched streams and was slower in both benchmark orders.
  The optional two-row Q8 head matched verification logits bit for bit and halved isolated
  two-row head time. The normal Phlegm HTTP bracket also matched streams at 83 and 1,320 prompt
  tokens. Code improved in both orders and repetition reversed. Control drift exceeded 12%.
- IQ4_NL remains the shipping candidate after the three-way 8,192-token Vulkan comparison. Unsloth's
  UD-Q4_K_XL was closer to the CPU BF16 reference on this prompt. Its mean KL was 0.002320 nats
  versus 0.006193 for IQ4_NL. The existing NPU expert kernels support IQ4_NL. The Unsloth model
  requires native NVFP4 block and decode kernels.
- The clean `phlegm bench --development` sweep measured 29.1–36.1 tokens/s prefill and
  2.2–2.6 tokens/s decode at 512–4,096 tokens on the Ryzen AI 9 HX PRO 370. The published
  `v0.1.0` CLI rejects the current adapter identity. Remaining work includes default-route attention
  and expert performance, the 1M recall, and release-build feature and
  `phlegm bench` runs.
- The shared singleton candidate cut 32-step decode time by 10.6–17.7% against the normal
  Phlegm bundle after 512 prompt tokens and preserved all 33 greedy IDs in a clean bracket.
  Matching the singleton dense and Q8 expert arithmetic passed all 18 verification cases.
  The tracked exporter offers the route under `--singleton-decode`. The rebuilt image passed
  the mixed operator fixture and matched both focused rollback cases. Through `phlegm serve`,
  it preserved the streamed output and cut decode time by 9.9–15.0% after a 1,320-token prompt.
  Sampled private memory was up 5.0 GiB. Standard exports omit this route and its artifacts.
- OFLM side covers the registry, application integration, benchmark page, README row and #240.

### Others

Not audited. Local evidence to start from:

- BERT embeddings, whisper-v3:turbo: no spec here; FLM rows in
  [flm-kernel-matrix.md](specs/open-engine/flm-kernel-matrix.md).
- qwen3.6-moe:35b-a3b: OPEN-FAMILY-QWEN36MOE-GGUF;
  [dflash-drafters.md](specs/open-engine/plans/dflash-drafters.md).
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
