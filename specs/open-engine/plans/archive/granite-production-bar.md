# Granite 4.2 3B to the production bar (OFLM #240)

**Done 2026-11-19.** Steps 1-6 ran; results are in `spec.md` OPEN-FAMILY-GRANITE's production bar. Decisions 1
and 2 took the recommendations. Step 3 took neither option: one stream every 4096 rows (47 in all) and a sliced KV
stream in the kernel. The near-optimal row stays open, with what is left named there.

Spec: `spec.md` OPEN-FAMILY-GRANITE and the requirements it links. Bar: `AGENTS.md` "Production bar
for OFLM kernel sets". Progress: [RECOMMENDED_MODELS.md](../../../../RECOMMENDED_MODELS.md). Written 2026-11-18. Kernel and engine
side only; OFLM work is listed at the end for separate PRs.

## Where Granite stands (from `spec.md`, 2026-11-18)

| Bar item | State | Evidence |
|---|---|---|
| Block route | Yes up to 4096: layer-major `gemm_block`, `attn_fa` on the NPU, bf16 and bfp16 routes. Past `attn_fa.l_max` 4096 the attention falls back to the host | OPEN-PREFILL-HOST-ATTN, -ATTN-FA, -BFP16, -GLUE-PREP |
| Near optimal | Decode profiled at depth, remainders named (window stream, arithmetic, KVCORES 2 reload; 0.72x Vulkan at 2048). Prefill measured per route, no consolidated profile of the current set | OPEN-ATTN-KVSLICE, OPEN-ATTN-SCORE-TREE, OPEN-PREFILL-BFP16 |
| Context | **Gap.** Tested to ~2100 tokens. The model states 131,072 (`max_position_embeddings`, `ibm-granite/granite-4.2-3b`). Decode capacity is a runtime value (OPEN-ATTN-KVSLICE: nothing compiled in) | OPEN-FAMILY-GRANITE-UNO depth tables |
| All on the NPU | Decode yes. Prefill yes up to 4096 on the bfp16 route (glue prep on the array); the bf16 route preps on the host. No table of host stages | OPEN-PREFILL-GLUE-PREP |
| Speculative decoding | **Shipped**: prompt lookup with the per-cycle choice. Served output equals plain decode; 1.10-1.97x geomean, 3.06x on a whole-file reply | OPEN-SPEC-LOOKUP |
| `oflm add` / `serve` | `serve` checked 2026-10-15 and 10-17. `add` not checked end to end. No registry entry | OPEN-FAMILY-GRANITE |
| Quantized | Q4_1. The GGUF-direct route (`dx_f32`, `model.gguf`) passed byte-identical logit gates; the source file's origin is not recorded. No Unsloth GGUF exists (checked 2026-11-18); IBM publishes `ibm-granite/granite-4.2-3b-GGUF` with a Q4_1 | OPEN-ATTN-KVSLICE gates |
| Benchmarks | CLI numbers only, to 2048 | as above |
| Features | Text-only model. Thinking toggle done. Reasoning block not parsed; tool calling not checked. Both are app code (OFLM) | OPEN-FAMILY-GRANITE |

## Spec impact

Modified requirements:
- **OPEN-FAMILY-GRANITE:** the source is IBM's published Q4_1 GGUF, loaded directly, with no q4nx
  container. The context capacity is the model's. Adds a `## Production bar` audit table.
- **OPEN-PREFILL-ATTN-FA:** the block attention covers windows past 4096, up to the capacity, on the NPU.
- **OPEN-FAMILY-GRANITE-UNO:** Uno is not the shipped drafter (decision 2). The requirement stays, opt-in.

No new requirement IDs unless step 3 picks a new kernel (then `OPEN-PREFILL-ATTN-FA-MERGE`).

## Decisions for you

1. **Context target.** The bar says to test to the real context: 131,072. KV is bf16, 2048 B a
   layer a token, so 80 KiB a token, **10 GiB at 131,072**. That is under R-0005's ~26 GiB NPU limit,
   but it needs a machine with the RAM for it. Decode at depth slows with the window stream
   (0.28 ms a layer at 2048, linear), so expect roughly 1-2 tok/s near 131k.
   Recommendation: support and gate the full 131,072. Publish speeds at 4k / 8k / 32k / 64k / 128k.
2. **Uno.** Its adapter ships as `uno.q4nx`, and q4nx is deprecated. Measured on 2026-10-15, adaptive
   with Uno beats lookup alone by at most 0.03x geomean (short 1.12 vs 1.10, 1024 1.15 vs 1.16, 2048
   1.19 vs 1.17, lookup prompts 1.94 vs 1.97). Recommendation: ship prompt lookup and record Uno
   as rejected for the default with that reason. Do not port the adapter format.

## Steps

1. **Source.** Fetch `ibm-granite/granite-4.2-3b-GGUF` `granite-4.2-3b-Q4_1.gguf` (record revision and
   sha256). Build the kernel set from it with no q4nx step. Confirm how the multiplier fold applies on
   the GGUF route (OPEN-GGUF-CONFIG says GGUF cannot supply the folded multipliers; the `dx_f32` gate
   says the route runs). Gate: OPEN-FAMILY-GRANITE's procedure, logits against the CPU reference.
2. **Decode to 131,072.** Run with `--max-ctx 131072`. Measure resident memory (KV, weights, bfp16
   copies) against R-0005. Gates at 8k, 32k and 128k: `--rows-check` ROWS PASS, greedy identical to
   the reference over 64 tokens, no NaN. Read R-0005 and R-0011 (buffers) first.
3. **Prefill attention past 4096.** Read the `attn_fa` design, R-0001 (budgets) and R-0003/R-0008
   (contexts, dispatch) first. Two options:
   - More streams: `af256` .. `af131072` by 256 is 512 instruction streams in one xclbin. No kernel
     change. Measure load time, xclbin size and BD/program budgets first.
   - Split the window: run `attn_fa` over 4096-row slices and merge the partial results (row max, row
     sum, O). Needs the kernel to write its max and sum; one more dispatch per slice.
   Pick by measurement. Gate: `prefill_mode_check.py` NPU against host attention at 8192 and 32768,
   OPEN-PREFILL-ATTN-FA's thresholds; perplexity on a long real document within 1 %.
4. **Drafter at depth.** OPEN-SPEC-LOOKUP's matrix at 8k and 32k, plus the existing prompts. SPEC
   IDENTICAL on every run.
5. **Profile and host stages.** One profile of the final set, `dibs run --measure`, performance mode
   (R-0006): prefill per op family at 512 / 2048 / 8192 / 32768; decode per stage at 0 / 2k / 8k / 32k /
   128k. llama.cpp Vulkan on the same machine as the baseline (R-0004). A table of every host stage
   and its cost. Name what is left, largest first.
6. **Record.** Results and the audit table into OPEN-FAMILY-GRANITE; update `RECOMMENDED_MODELS.md`;
   `benchmarks/reference/engine-changes.json` if the engine changed; archive this plan.

## For OFLM PRs (out of scope here)

- Registry entry in `src/model_list.json`: IBM's Q4_1 GGUF, quant Q4_1, family granite, context 131,072.
- `oflm add` and `oflm serve` from that entry; `oflm bench` results on a benchmark page.
- Reasoning parse (`modeling_granite.cpp` has no `parse_stream_content`; IBM ships
  `granite_thinking_parser.py` beside the GGUF) and a tool-calling check.
- The bar's README "Recommended models" row has no table to go in (#241 closed).
