# Gemma 4 E2B / E4B to the production bar

Spec: `spec.md` OPEN-FAMILY-GEMMA4 (12B here; #238 extends it to the E-series) and the multimodal
primitives in [gemma4-text-kernels.md](../gemma4-text-kernels.md) and
[gemma4-multimodal-kernels.md](../gemma4-multimodal-kernels.md). Bar: `AGENTS.md` "Production bar for
OFLM kernel sets". Progress: [RECOMMENDED_MODELS.md](../../../RECOMMENDED_MODELS.md). Written 2026-11-18.
Scope includes images, video and audio: the user requires all three.

## Where E2B / E4B stand (2026-11-18)

The text work exists, but not in this repo. OFLM draft PR #238 (`Cyronius:feat/gemma4-e-open`, head
`e23f4d35`; local worktree `C:/code/oflm-gemma4e`, branch at `dc57d3cc`) adds family `gemma4e`:
`recipes/gemma4e.py`, one-context `designs/dense/gx.py` built from `dxkit.py` (a frozen copy of `dx.py`'s
pieces), KV sharing, per-layer inputs (PLE) on the NPU, 256-token block prefill (`OPEN-PREFILL-G4`),
`fa_wide` block attention past 4096, an `oflm-add` installer and a text-only registry entry. Images and
audio are refused with a 400. Nothing in OFLM, open or closed, takes video.

| Bar item | State | Evidence |
|---|---|---|
| Block route | E4B yes (#238). E2B not run on the NPU since the port | #238 body |
| Near optimal | **Gap.** E4B serve: prefill 62-65 tok/s at 850 tokens, decode 10.7 tok/s, busy box, not `dibs run --measure`. No profile. October's branch: 85-86 tok/s; the drop is not explained | #238, #236 |
| Context | **Gap.** Model states 131,072. A 32k prompt ran in October; not repeated. `fa_wide` accuracy gate not repeated | #238 |
| All on the NPU | Text: PLE on the NPU; no host-stage table. **Towers: none on the NPU** (primitives only) | gemma4-multimodal-kernels.md |
| Drafter | **Gap.** Unsloth ships an MTP drafter (`mtp-gemma-4-E{2,4}B-it.gguf`, arch `gemma4-assistant`, Q8_0, shares the target's KV; llama.cpp #24282). Unsloth reports 0.55 acceptance on E2B (B200). Prompt lookup exists for Granite (OPEN-SPEC-LOOKUP), not on the `gemma4e` route | HF repo, checked 2026-11-18 |
| Quantized GGUF | Google's QAT Q4_0 (`google/gemma-4-E{2,4}B-it-qat-q4_0-gguf`). Unsloth also publishes Q4_0, Q4_1, Q4_K_M, BF16. Reason for the non-Unsloth source not yet recorded | #238 |
| `oflm add` / `serve` | E4B text: `serve` passes, `oflm-test --llm` 5/5 with reasoning off. E2B not checked | #238 |
| Benchmarks | None recorded | |
| Features | Vision, video, audio: **not on the open path.** Thinking on: answer lost to reasoning and a raw `<\|channel>thought` block leaks (OFLM). Tool calling: fixed for the closed engine (#59), not checked on the open path | #238 |

Not on OFLM #240's target list yet. Adding it there is an OFLM decision.

### Multimodal facts (Google `config.json`, `processor_config.json`, both mmproj headers, 2026-11-18)

- **Vision** (`gemma4_vision`): 16 layers x 768, 12 heads x 64, MLP 3072 GELU-tanh, patch 16, 2-D RoPE
  theta 100, 3x3 pooling, 10,240-entry position table. Images default to 280 soft tokens (2520 patches).
- **Video**: the same tower per frame. 32 frames sampled, 70 soft tokens each (630 patches), so 2240
  tokens a clip. `video_token_id` is in the config.
- **Audio** (`gemma4_audio`): 12-layer conformer x 1024, 8 heads, chunked local attention (chunk 12,
  left 13), logit cap 50, 5-tap depthwise conv, subsampling convs 128 / 32 channels, clipped linears,
  128 mel bins at 16 kHz, 40 ms a token, 750 tokens (30 s) a clip.
- Both mmproj GGUFs (Google QAT repo and Unsloth BF16) carry vision **and** audio: 1411 tensors, F32 and
  BF16, projectors `gemma4v` / `gemma4a`. Weights can come from GGUF; no q4nx is needed.
- **Hazard:** Unsloth's mmproj says audio `layer_norm_epsilon` 1e-5; Google's says 1e-6, as does the
  config. The reader must not take eps from the Unsloth file without a check.
- Text `use_bidirectional_attention` is null for E2B / E4B, so the decoder mask stays causal over media
  tokens. Confirm in `modeling_gemma4.py`, with the per-layer input that media positions take.

## Spec impact

New requirements (names provisional):
- **OPEN-VISION-GEMMA4:** the Gemma 4 vision tower and projector on the NPU from the mmproj GGUF, every
  soft-token budget the processor allows.
- **OPEN-VIDEO-GEMMA4:** frame decode, sampling and timestamps on the host; frames through the tower;
  the token layout of Google's chat template.
- **OPEN-AUDIO-GEMMA4:** the audio conformer and projector on the NPU; the mel front end on the host.
- **OPEN-SPEC-MTP-GEMMA4:** the MTP drafter on the `gemma4e` route, if it ships (step 5).

Modified requirements:
- **OPEN-FAMILY-GEMMA4:** takes #238's E-series text; states the GGUF source, the 131,072 context and a
  `## Production bar` audit table for E2B and E4B.
- **OPEN-VISION-EMBED:** Gemma's 1-D positions, and media rows through block prefill, not one token a
  step (2240 video tokens one at a time would fail the block-route item).

## Decisions for you

1. **Source GGUF.** Recommendation: keep Google's QAT Q4_0 for the text weights. QAT weights are
   trained for Q4_0, and Q4_0 is the engine's native pool law. Step 1 measures it against Unsloth's
   Q4_0 and Q4_1 by perplexity and KL to the BF16 model, and the bar's exception reason is that number.
   The drafter comes from Unsloth either way: Google publishes none.
2. **Video API.** The OpenAI chat API has no standard video part. Recommendation: accept a `video_url`
   content part (vLLM's shape) with a local path or URL, and a `frames` option. This is OFLM server
   work, a separate effort (`AGENTS.md`); the plan only defines what the engine here takes.
3. **Order.** Recommendation: import and text bar first (steps 0-4), then vision, then video (it reuses
   vision), then audio. The MTP drafter can run in parallel with the towers.

## Steps

0. **Import #238 here.** This repo is the canonical home of `open_kernels/`, `src/open_qwen36/` and
   `specs/open-engine/`. Bring in #238's parts of those trees with `cherry-pick -x` where they apply,
   and a ledger row in `UPSTREAM.md`. This repo's 12B route moved past OFLM main (bfp16 prefill GEMMs,
   Q4_0-native 4608-byte chunk, 8-core KV-slice decode attention); `gx.py` keeps `dxkit.py`'s frozen
   copy for now. Gates: #238's E4B gates reproduce (45/45 argmax, the same 24 and 32 greedy tokens, the
   same instruction streams). Then export and run **E2B** on the NPU with the same gates.
1. **Source and quality.** Fetch both Google QAT and Unsloth Q4_0 / Q4_1 for E2B and E4B (record
   revisions and sha256). Reference: transformers BF16 from `google/gemma-4-E{2,4}B-it`
   (R-0004). Gate: perplexity and KL on a long real document, and greedy decode, for each source.
2. **Context to 131,072.** Own-KV layers only: E4B has 4 global layers x 2 kv heads x 512, about
   16 KiB a token, 2 GiB at 131,072; E2B about 0.75 GiB. Sliding layers hold 512 rows. Read R-0005 and
   R-0011 first. Gates at 8k, 32k and 128k: the `fa_wide` accuracy gate, greedy against the reference,
   no NaN, and the shared-layer windows at their boundaries.
3. **Near optimal.** `dibs run --measure`, performance mode (R-0006). Profile decode per stage at 0 /
   2k / 8k / 32k / 128k and prefill per op family at 512 / 2048 / 8192 / 32768. llama.cpp Vulkan on
   the same machine as the baseline. Find the 85 to 62 tok/s prefill drop first. Then port this repo's
   12B gains into `gx.py` one at a time, each gated for bit identity or KL: Q4_0-native chunk (12B:
   -6-7% decode), KV-slice decode attention, bfp16 prefill GEMMs. Name what is left, largest first.
   Read R-0001, R-0003 / R-0008, R-0012 / R-0013 before kernel changes.
4. **Host stages.** A table of every host stage and its cost: the PLE row gather from the
   262,144-row table, the softcap, sampling, the tokenizer, and (later) image resize, frame decode and
   the mel front end.
5. **Drafter.** Read llama.cpp #24282 for the `gemma4-assistant` graph and dump the MTP GGUF's tensor
   table. Measure prompt lookup on `gemma4e` first (the verify step is the cost). Then MTP: it reuses
   the target's KV, so it needs the shared-cache aliasing #238 already has. Matrix as OPEN-SPEC-LOOKUP:
   SPEC IDENTICAL on every run, acceptance and speed at 0 / 2k / 8k. Ship the best, or reject each with
   the measured reason.
6. **Vision tower on the NPU.** The primitives exist (table in gemma4-multimodal-kernels.md). Open:
   - the bidirectional attention: `whisper_fa` fails at real magnitudes (0.096 against 0.04); the
     online-attention path passes (3.37e-7). Build on the online path or fix the scaled one.
   - weights from the mmproj GGUF (`v.*`, `mm.input_projection`), a scheduler for the 16 layers,
     the position-table gather, the projector into the text width.
   - preprocessing on the host for every soft-token budget the processor accepts (confirm the set;
     70 / 140 / 280 / 560 / 1120 expected).
   - the engine: media rows into block prefill at 1-D positions.
   Gates: tower output against transformers per budget (corr > 0.9999); image QA answers through
   `oflm serve`; the cost of each image budget.
7. **Video.** FFmpeg decode (OPEN-VISION-IMAGE-READ's link), 32-frame sampling, 70 tokens a frame,
   timestamps as Google's chat template writes them. Gates: token layout equal to the HF processor's on
   a test clip; frame embeddings against transformers; a clip QA through `oflm serve`; the time per
   clip, with host and NPU parts named.
8. **Audio tower on the NPU.** Primitives exist for every op family. Open: the conformer wiring, the
   relative-key projection packing, clipped linears from the GGUF's stored ranges, weights from `a.*`,
   the projector, the mel front end on the host (the eps hazard above). Gates: features and tower
   output against transformers; transcription of reference clips against transformers' text; a 30 s
   clip through `oflm serve`.
9. **Record.** The audit table into OPEN-FAMILY-GEMMA4; the new requirements; `RECOMMENDED_MODELS.md`;
   `benchmarks/reference/engine-changes.json`; archive this plan. Do not change #238 or other OFLM
   code from this plan (`AGENTS.md`).

## For the separate OFLM effort (not done here)

- `src/model_list.json`: entries for E2B and E4B naming the text GGUF, the mmproj and the drafter, with
  the real quant, family `gemma4e` and context 131,072. Drop the text-only refusal when the towers land.
- `oflm-add` for the mmproj and the MTP files; `oflm serve` image, video and audio requests; the video
  content part (decision 2).
- Thinking on: the answer goes to reasoning and the raw `<|channel>thought` block leaks (#238). Tool
  calling on the open path.
- `oflm bench` on a benchmark page; the README row (#241); adding the model to #240.
