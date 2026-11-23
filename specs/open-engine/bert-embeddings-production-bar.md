# BERT embedding production audit

This audit covers the six encoders in [OFLM #240](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/240).
The inference entry point is `oflm serve --embed 1 --embeddingmodel TAG`. The benchmark entry point is
`oflm bench-embed TAG`, which reports encoder latency and throughput across batch sizes.

## Production bar

| Gate | Evidence and remaining work |
|---|---|
| Block execution | The checked release designs have four operations per layer and four batch tiers per family, under one resident xclbin. MiniLM loaded all 16 streams on the NPU and passed the batch-identity warm-up. The long-input designs need the same runtime check. |
| Performance | The published OFLM page measures short-input batching. Add matched CPU comparisons, at-limit latency and throughput, and the cost of the host stages. The Windows host is a Ryzen AI 9 HX PRO 370, Windows build 26200, NPU driver 32.0.20102.3930. |
| Context | Release designs process 64 tokens per sequence. Staged publisher packs served exact 512-token fixtures. Nomic served exact 8,192 tokens but missed publisher parity. Two exact GTE 8,192-token fixtures matched the publisher. The deployment still needs request-length routing and the correct error for over-limit input. |
| NPU coverage | QKV projection, attention output projection, and both FFN projections dispatch to the NPU. Tokenization, embedding lookup, attention/softmax, normalization, activation, and pooling are host stages. Stage costs over long inputs remain open. |
| Speculative decode | Not applicable to an encoder. |
| Published input weights | The present registry pulls and packs publisher safetensors into the tiled NPUE format. Establish published GGUF coverage per checkpoint and compare quality before choosing the default quant. |
| Serving and features | Run `oflm-test --embedding` on each tag. Exercise Nomic task prefixes and GTE multilingual inputs. Check GTE sparse output support and the publishers' elastic dimensions separately from full-width dense vectors. |

| Tag | Published input length to verify | Design family |
|---|---:|---|
| `all-minilm:l6-v2` | 256 in the published Sentence Transformers configuration | `BERT-h384-bfp16` |
| `bge-small:en-v1.5` | 512 | `BERT-h384-bf16` |
| `bge-base:en-v1.5` | 512 | `BERT-h768-bfp16` |
| `bge-large:en-v1.5` | 512 | `BERT-h1024-bfp16` |
| `nomic-embed-text:v1.5` | 8,192 with the publisher's dynamic RoPE scaling | `BERT-h768-gated-bfp16` |
| `gte-multilingual:base` | 8,192 | `BERT-h768-gated-bfp16` |

The 512-token comparison also includes MiniLM as an exploratory run. Its encoder has 512 position
embeddings and the publisher uses a 256-token truncation limit for serving. See the
[MiniLM model card](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2),
[Nomic model card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#long-context),
and [GTE model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base).

## Artifact and input checks

- All five design families from [OFLM pre-v.0.1.0](https://github.com/Atomic-Germ/OpenFlowLM-Next/releases/tag/pre-v.0.1.0)
  passed `check_design_sets.py`. Linux tarball SHA-256
  `58997fb7115bf40903e3edda8184d4ced749a4c622d3c2e53f09f5c0ebf50a13`.
  The release, 512-token and 8,192-token artifacts also passed the instruction-file and slot
  checks. The checker requires every declared stream file, a nonempty xclbin and bootstrap
  instruction image, and the exporter slot assignment for each operation and batch tier.
- The extended comparison uses `--seq 512 --batch 16 --batches 1,4,16`, keeping the largest
  stream at the release design's 8,192 rows. The original batch-multiple-of-four guard rejected
  the one-sequence tier. Its matrix has 512 rows, which meets the exporter's actual 256-row
  tile requirement. The audit still needs to verify the resulting artifact on hardware.
- A standalone build of OFLM's NPUE packer packed all six checkpoints at the lengths needed for
  the extended comparison. The automatic pack path uses 256 positions for BERT and Nomic and
  64 for GTE. The isolated comparison directories carry separately generated containers, with
  512 positions for BERT and 8,192 for the two RoPE models. Container length alone does not
  prove that an encoder computes the publisher's long-context RoPE.
- GTE's live `config.json` changed its model type from `new` to `gte` at
  [publisher commit 6dde5e2](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/commit/6dde5e2ebcac02a5459a3cf310346901fa1dc8ff).
  OFLM's downloader fetched the new file and warned about a hash mismatch. The existing automatic
  packer selects the GTE path by `model_type == "new"`. A subsequent cold-pack probe failed
  with missing `embeddings.word_embeddings.weight` after the config fell through to the BERT
  packer. The config from
  [revision 9bbca17](https://huggingface.co/Alibaba-NLP/gte-multilingual-base/tree/9bbca17d9273fd0d03d5725c7a4b0f6b45142062)
  exactly matches the recorded Git blob hash, `9f640b05ee684e9680fdc90000de85559061c5ef`, and the isolated
  pack succeeded with it.
- The current publisher config encodes the same GTE rotary frequencies as the pinned config.
  For head width 64 the legacy formula uses base `20000 * 8` and divides frequencies by
  `8 ** (2 / 64)`. The current linear formula specifies that base and divisor directly.
  All 32 frequencies agreed exactly in the comparison. Translating the equivalent metadata
  in an isolated directory and running the existing packer produced the same 8,192-position
  NPUE bytes as the pinned checkpoint. Both packs have SHA-256
  `c4b1c592b7164de7e6559abab7e4b7f000a7c8cd0e4e6be04f97e0292d83bbdc`.
  Raw evidence is in `.plans/bert/gte-current.log`.

Use [the comparison utility](../../utilities/bert-embeddings/check.py) with a local checkpoint directory
to make the same exact-length requests to OFLM and the checkpoint's CPU implementation. Use
`--prompt-name document --prefix "search_document: "` for Nomic serving. The utility checks the
server's model identity, processed token count, and vector normalization before recording the
comparison. Long-input performance needs a quiet-machine repeat with the driver and design named.
Use `--mixed --batches 4 16` to compare varied, exact-length fixtures through the same CPU and
serving paths. The utility checks the full batch token count and ordered result indices and
records the lowest cosine in each batch.

## Release design control

Windows OFLM v0.1.0 development binary, performance power mode, release xclbins and independently
packed publisher checkpoints. The control ran during a compiler build, so these records establish
vector quality and serving behavior. The timings are exploratory. For each model, both fixtures
at 32 and 64 tokens passed the server's token-count, model identity and normalization checks.

| Model | CPU cosine range, four requests | `oflm-test --embedding` |
|---|---:|---|
| MiniLM | 0.999593–0.999741 | Pass |
| BGE small | 0.999987–0.999993 | Pass |
| BGE base | 0.999715–0.999806 | Pass |
| BGE large | 0.999750–0.999783 | Pass |
| Nomic | Pending | Eight errors. The suite sends its normal requests without `prompt_name` and the server requires it. Explicit `document` requests passed both lengths. |
| GTE | Pending | Pass with the pinned source config and separately packed container |

Raw reports are under `.plans/bert/release-quality-*.log` and `.plans/bert/release-oflm-test-*.log`
in the bring-up worktree.

The matched control used 64-token varied fixtures, three warmed timed requests per case, and
batch sizes 1, 4 and 16. The CPU reference used FP32 PyTorch with four threads. This table gives
the batch-16 median wall times and the lowest CPU cosine in each batch.

| Model | OFLM HTTP, ms | CPU, ms | CPU / OFLM | Lowest cosine |
|---|---:|---:|---:|---:|
| MiniLM | 52.62 | 134.91 | 2.56 | 0.999559 |
| BGE small | 123.78 | 270.79 | 2.19 | 0.999989 |
| BGE base | 322.81 | 1,014.43 | 3.14 | 0.999789 |
| BGE large | 575.14 | 3,156.25 | 5.49 | 0.999603 |
| Nomic | 399.83 | 1,285.73 | 3.22 | 0.998692 |
| GTE | 387.97 | 1,259.27 | 3.25 | 0.999531 |

All six checks returned the requested 1,024 batch tokens. Dibs ended the run as NOISY and
reported a concurrent Inkling benchmark process. Treat these timings as provisional and repeat
the comparison on a quiet machine before closing the performance gate. Raw measurements are
under `.plans/bert/bench64-quality-*.log`.

## 512-token candidate

All five families compiled and passed the design checker on the same Windows toolchain. Each
family has four streams per tier, at batches 1, 4 and 16. Raw exports are retained under
`.plans/bert/seq512/xclbins`. The SHA-256 values identify the checked designs.

| Family | `design.json` SHA-256 |
|---|---|
| BERT-h384-bfp16 | `e853a0796fdec0d669c24a424e779537da28fa7c2dee8189f1b03f2e4d391a37` |
| BERT-h384-bf16 | `c90e0943ce432a846129b6a52e02461049ecc00fda4d0b0917b5f2cb65fd05aa` |
| BERT-h768-bfp16 | `629f4dd319f20f8e13af048250120ecfc10ec0089d79ea3928decfa9ee857482` |
| BERT-h768-gated-bfp16 | `209efdb401275a80bab9c215668c4e8ead72eebca6a4469562e895688d58f763` |
| BERT-h1024-bfp16 | `61006bcde2e0beb3367482b87fc4abe0f17bc2f4fa6b7b3c0ac5fc9319ee0c95` |

MiniLM's re-export loaded on the NPU with 12 instruction streams and batch tiers 1, 4 and 16.
`oflm-test --embedding` passed. Both 256-token probes agreed with the publisher CPU reference at
cosines 0.999015–0.999122. Both 512-token probes returned a 512-token usage count and agreed at
0.999020–0.999065. The comparison was run during a compiler build, so use it for quality and
coverage. Raw report `.plans/bert/seq512-quality-all-minilm_l6-v2.log`, SHA-256
`2bd1b7d74f209070f95011a4a5b74ce23c19861c164e89292221f0d41fb5257c`.

BGE small served both 512-token requests at CPU cosines 0.999982–0.999984. Its embedding suite
passed. An exact 513-token request received HTTP 500 with a generic server error. The engine
rejected the oversized input, and the REST layer needs to expose that as a client error.

The publisher checkpoint packs for BGE-base, BGE-large, Nomic and GTE also served both
512-token fixtures. Their minimum CPU cosines were 0.999585, 0.999442, 0.997504 and 0.999394,
respectively. In each case the over-limit request returned HTTP 500. Raw checks are in
`.plans/bert/seq512-quality-*.log`.

The single-sequence 8,192-token Nomic/GTE experiment compiled and passed the artifact checker.
It uses four streams and the same 8,192 total rows as the largest release tier. Its
`design.json` SHA-256 is `b73db99879c4bb42d194caf781cd43f7523f3e76fa491ff554e5dbf55f4f67a3`.
Runtime and Nomic dynamic RoPE parity are separate checks.

The [sequence derivation tool](../../npu_offload/gemm_rtp/derive_sequence.py) also generated
an 8,192-token design directly from the 512-token design's batch-16 streams. Its xclbin and
bootstrap instructions are byte-identical to the source 512-token design. Its four per-operation
instruction files are byte-identical to the independently built 8,192-token instructions.
The derived artifact passed the checker. The derived and independently compiled images
also returned the same bytes for one exact 8,192-token GTE request through OFLM.
The vector SHA-256 was
`9e0f8680a72aed87d413871cc9ba2f3e793979ebe10f7504402fabd0a0a8064e`.
A 2,048-token view derived from the same 512-token image also passed the checker. It uses
the 2,048- and 8,192-row streams for batches 1 and 4. The same Nomic fixture is queued
for profiling at 2,048 tokens with both 2,048- and 8,192-row padding.

The 64- and 512-token artifacts have eight shared streams for each family. All 40 corresponding
instruction pairs were byte-identical. The gated family's AIE partition payload, connectivity,
IP layout and memory topology matched using `xclbinutil`. The other four AIE partitions
differed, so this audit does not treat arbitrary release xclbins as interchangeable.

A standalone MiniLM profile using the same 512-token design dispatched four GEMMs per layer
(24 total). Five warm runs measured host attention at 18.1–21.7 ms, softmax at 3.9–5.2 ms,
normalization at 1.5–1.7 ms, conversions and bias at 2.9–4.0 ms, and buffer sync at 0.8–1.5 ms.
GELU ran in the fused host packing path. Kernel dispatch and wait took 9.2–10.7 ms for the first
three runs and rose to 71.5 and 116.6 ms during the last two. Collect a quiet-machine repeat
before using these figures as performance claims.

The full batch-16 comparison demonstrates the cost of padding short requests to the
512-token design. With the same 64-token inputs, the larger design took 4.1–9.6 times
the HTTP wall time of the release design. At the longer input lengths the candidate
was faster than the four-thread FP32 CPU reference. Times are warmed medians in milliseconds.

| Model | 64 tokens, release | 64 tokens, 512 design | Long tokens per input | Long NPU HTTP | Long CPU |
|---|---:|---:|---:|---:|---:|
| MiniLM | 52.6 | 505.0 | 256 | 526.4 | 563.1 |
| BGE small | 123.8 | 1,118.7 | 512 | 1,180.9 | 2,584.0 |
| BGE base | 322.8 | 1,609.6 | 512 | 2,535.6 | 8,403.4 |
| BGE large | 575.1 | 4,630.9 | 512 | 4,838.2 | 28,557.7 |
| Nomic | 399.8 | 1,635.4 | 512 | 1,645.5 | 10,906.9 |
| GTE | 388.0 | 2,031.0 | 512 | 1,988.0 | 11,185.8 |

All requests used the publisher checkpoint packs. Dibs reported concurrent Inkling jobs
during both measurements. The results establish a short-input regression worth addressing.
The quiet-machine production rate gate remains open. Raw results are in
`.plans/bert/bench512-quality-*.log` and `.plans/bert/measure-exact.log`.

## MiniLM cold GGUF at 256 tokens

OFLM packed the public Second State Q8_0 GGUF on first use in an isolated model
directory and loaded the 256-token view of the release GEMM image. It loaded 16
streams over batch tiers 1, 4, 8 and 32. Single and batch-4/16 requests at 64, 128
and 256 tokens matched the publisher CPU model. The minimum cosine at 256 was
0.999477. The batch of 16 took 195.2 ms through OFLM and 568.9 ms on the CPU
reference in this run. The ordinary embedding suite passed its nine applicable
checks. The exact 257-token request produced HTTP 500 and failed the separate
over-limit gate.

Raw report `.plans/bert/minilm256-quality-all-minilm_l6-v2.log`, SHA-256
`3d64db2c1488c83950d45ae6ab5694935454eda5e2c9df348d07663558c9cc63`.

## Long-context Nomic and bidirectional attention

The 8,192-token design served exact Nomic inputs through `oflm serve` with the staged
publisher weights. Both 2,048-token fixtures scored at least 0.997861 against the
publisher's CPU model. The two exact 8,192-token fixtures scored 0.456015 and 0.455719.
The observed HTTP timings were 47.9?74.9 s at 2,048 tokens and 74.4?94.7 s at 8,192,
with the 8,192-token design padding both lengths to the same capacity. The exact
8,193-token request returned HTTP 500. Raw results are in
`.plans/bert/seq8192-quality-nomic-embed-text_v1.5.log`, SHA-256
`5848771e0d0b2432d201df3dc3d744133dcbe8156e3a90a942b4051bd06f3613`.

The same hardware uses the pack's static theta. The publisher's dynamic RoPE uses the
original theta at or below 2,048 tokens and scales it as the sequence grows. A
diagnostic pack changed only theta to 7453.483032, the value for an exact 8,192-token
request. It reduced cosine at 2,048 tokens to 0.939621 and 0.940555, compared with
0.997938 and 0.997862 on the original pack. On the first full-length fixture it
restored cosine from 0.456015 to 0.997399. The second changed from 0.455719 to
0.997418. All 150 packed tensor payloads were unchanged. The rotary base must be
selected by each input's true length, across both request order and batches. For
length L above 2048, the publisher's dynamic NTK base in this model is
1000 ? (2L/2048 ? 1)^(64/62). At or below 2048 it is 1000.
Raw report `.plans/bert/rope-fixed-seq8192-quality-nomic-embed-text_v1.5.log`,
SHA-256 `c7ae0cb66d7f9b44e7838660c31f8f096b56fd8093987e2449570c19df79dda3`.

The in-progress host profile already has a paired Nomic measurement with the same
2,048-token input. In five warm runs per route the 2,048-capacity view took
1.95?2.05 s and the 8,192-capacity view took 53.28?67.20 s. The host QK and AV
passes grew from 1.40?1.47 s to 50.06?63.78 s, while host softmax grew from
0.16?0.18 s to 2.33?2.52 s. The profile measures a local encoder call. Its remaining
cases, including actual 8,192-token input, are still running.

A local integration build is queued on the two long-context families. It routes
post-projection Q/K/V through the NPU attention image, selects the mask from the
actual input length, and chooses Nomic's rotary table per input. The existing NPUE
weight layout folds the attention scale into Q. The probe reverses that power-of-two
scale during BF16 packing before using the already checked attention image.
The local prototype and the serving code require their own parity checks.

The existing bidirectional [FlashAttention design](../../open_kernels/designs/whisper_fa/README.md)
compiled at 512 and 8,192 tokens with 12 heads and head width 64, the gated encoder
geometry. The new [hardware check](../../utilities/bert-embeddings/check_attention.py)
compares BF16 I/O with a float64 CPU oracle at query-tile boundaries and verifies that
the first query attends to the last valid key. The Strix Point NPU passed the
static 512- and 8,192-token checks, including the 8,192-capacity build with 2,051 valid
keys. Outputs were finite and bit-identical across two dispatches per case.

| Attention capacity | Valid keys | Minimum sampled row cosine | First / second dispatch (ms) |
|---:|---:|---:|---:|
| 512 | 512 | 0.999495 | 3.81 / 1.11 |
| 8192 | 8192 | 0.999584 | 88.51 / 70.35 |
| 8192 | 2051 | 0.999336 | 87.12 / 72.96 |

The harness measured submit and wait for a single layer with synthetic BF16 inputs.
Raw report `.plans/bert/attention-run.log`, SHA-256
`83ab92d7335e42e8698c51e62880f30965777d5e30048e5c7b4eec0b2b0b1dbf`.

The 2,051-valid-key build completed. An isolated runtime-parameter prototype also
compiled at 512-token capacity. It writes a length register and releases a barrier on
each of the 32 compute tiles before those tiles run the existing key-mask arithmetic.
The transaction verifier found 32 distinct length writes and changed the mask from
381 to 259 valid keys without changing the xclbin (SHA-256
`25d9aaf2737ff0c4346c6042c2853f88d052563382a6ed6e74efe4e77b7eb246`).
This verifies how the length is conveyed in the stream. The NPU parity check must
still establish that the new value reaches each worker on repeated dispatches.
The 8,192-capacity prototype also compiled. The same verifier patched all 32 tile
writes to set 2,051 valid keys without changing its xclbin. The optional
`FA_RUNTIME_VALID_LEN` path is now in the shared design. At 512 capacity the tracked
runtime build matched the prototype's instruction stream and XRT connectivity. The
tracked static build matched its pre-change instruction stream and connectivity.
The compiled PDI bytes differ between builds. The tracked 512-capacity runtime
image passed at key lengths 259, 381 and 512 with bit-identical repeats and a minimum
sampled row cosine of 0.999449 over all three lengths. Warm dispatch times ranged from
1.13 to 1.28 ms. Running 8,192 capacity with a dynamic length and alternating lengths
in one resident context remain separate checks. The harness now patches and
syncs the same instruction buffer for each requested length after validating and
caching all 32 tile register offsets from the original transaction. A 2,048-capacity build is queued to
measure the intermediate route without padding attention to 8,192. The checker also
accepts captured Q/K/V in float32, comparing the device output against float64
attention over BF16-rounded inputs and reporting the extra error due to rounding the
original tensors. The layer 0 and 11 GTE captures are queued at 512 and 8,192 tokens.
Model-level quality and latency still need an integrated run.

GTE served both 2,048-token fixtures on the 8,192-capacity encoder and matched
the publisher CPU vectors with a minimum cosine of 0.999475. The observed single
request timings were 83.2?86.0 s. The two exact 8,192-token requests scored 0.999570
and 0.999543, taking 93.6 and 81.8 s through HTTP compared with 26.1 and 28.0 s for the
CPU references. The exact 8,193-token request returned HTTP 500. Raw results are in
`.plans/bert/seq8192-quality-gte-multilingual_base.log`, SHA-256
`7698157fba16eb420f7e7dd3a82e41c11fdf7f604ea6c93e9ccb5fe5a6325daf`.

## Published GGUF candidates

Five pinned Q4_K_M downloads matched their Hub SHA-256. The loader's compatibility depends on the
tensor types inside each file. The tag alone does not establish that coverage.

| Model | Candidate source | Input check |
|---|---|---|
| MiniLM | Jarbas, `cd9d5518` | Contains Q5_0 tensors. OFLM add refuses this file. |
| BGE small | CompendiumLabs, `d32f8c04` | Contains Q5_0 tensors. |
| BGE base | CompendiumLabs, `24914efd` | F32, Q4_K and Q6_K. Runtime and quantized quality still to check. |
| BGE large | CompendiumLabs, `03b9bf10` | F32, Q4_K and Q6_K. Runtime and quantized quality still to check. |
| Nomic | Nomic AI, `0188c9bf` | Q4_K_M includes Q5_K. |
| GTE | No matching published GGUF found in the initial Hub search | The current NPUE GGUF adapter accepts `bert` and `nomic-bert`. GTE needs its own conversion and loader validation. |

A follow-up Hub search found a related [GPUStack reranker GGUF](https://huggingface.co/gpustack/gte-multilingual-reranker-base-GGUF)
with `general.architecture=new` and the same fused projection tensor names. It is a
separate checkpoint with different weights and rotary metadata. The isolated quantization
experiment uses the original embedding checkpoint, whose tokenizer has 250,002 real
pieces and whose embedding table has 250,048 rows. The spare 46 rows are marked unused
in the GGUF vocabulary, and the exact publisher tokenizer is retained. The
prototype emitted 138 tensors into a 348,620,160-byte Q8_0 GGUF and verified their
names, source shapes and finite dequantized values on readback. SHA-256 is
`c12175db0b1b7f2dc6daacc2e72dc0203b89921433e8d4f6cc5273c76bedc574`. The readback tensors were replayed through the
existing GTE packer to stage an NPU quality check. On CPU, embeddings made from the
dequantized tensors matched the original publisher weights at 64, 512 and 8,192 tokens
with cosines 0.999816, 0.999789 and 0.999789. These are single dense CLS fixtures.
Raw report `.plans/bert/gte-gguf-cpu.log`, SHA-256
`3eca5adcc2ff1c9e2480ec10d906316a4e74f55f7b60d9e4a561a4901728bcdd`. The reproducible
[converter](../../utilities/bert-embeddings/convert_gte_gguf.py) accepted the current
publisher directory and emitted a byte-identical GGUF. It takes the checkpoint directory
and a new output path and requires the Python `gguf`, `numpy` and `safetensors` packages.

The replayed NPUE pack served 64- and 512-token single and batch-4 requests with a
minimum cosine of 0.999379 against the original publisher CPU model. It passed
`oflm-test --embedding`. A direct GGUF reader and validation through the standard
installation path are still required.

OFLM add installed Q8_0 for MiniLM from Second State, BGE small/base/large from CompendiumLabs,
and Nomic from the original publisher. Each file was checked as GGUF file type 7. The five
models loaded from GGUF and passed the batch-4 bit-identity gate on the 64-token release design.
Nomic needed `--prompt-name document`. These runs used an isolated registry overlay with the
canonical tags because the installed backend rejects custom embedding tags even after the
installer registers them. The generated entries label Q8_0 weights as bf16. The local
overlay records Q8_0 and leaves the user registry alone. The 32- and 64-token HTTP probes also
matched the original CPU checkpoints. The sample is two fixtures per length.

| Q8_0 model | Minimum CPU cosine | `oflm-test --embedding` |
|---|---:|---|
| MiniLM | 0.999314 | Pass |
| BGE small | 0.999867 | Pass |
| BGE base | 0.999574 | Pass |
| BGE large | 0.999696 | Pass |
| Nomic | 0.997948 | The explicit document requests passed. The embedded suite still omitted the required task prompt. |

Longer-context validation is open for the quantized route. Raw reports are
`.plans/bert/gguf-quality-*.log`, `.plans/bert/smoke-gguf-canonical.log` and
`.plans/bert/nomic-gguf-document.log`. The compatible Q4 comparison uses MiniLM Q4_0 from
Second State, BGE small Q4_0 from ChristianAzinn, BGE base and large Q4_K_M from CompendiumLabs,
and Nomic Q4_0 from the original publisher.

| Q4 model | Lowest CPU cosine across single requests and batch-4 fixtures |
|---|---:|
| MiniLM Q4_0 | 0.953409 |
| BGE small Q4_0 | 0.973202 |
| BGE base Q4_K_M | 0.985465 |
| BGE large Q4_K_M | 0.985452 |
| Nomic Q4_0 | 0.861227 |

The Q4 trials ran at 32 and 64 tokens. They used varied context at 64 tokens and the same short
fixtures at 32 tokens. All loaded on the NPU. The Q4 results favor Q8_0 for the provisional
selection. Both formats unpack to the same NPU weight layout, so the primary benefit of Q4
here is the smaller download. Raw reports are under `.plans/bert/q4-quality-*.log`.

For the extended Q8_0 trials, the [position-table utility](../../utilities/bert-embeddings/extend_npue.py)
restores the missing rows from the original verified GGUF. Nomic uses the existing zero-table
convention for RoPE within its GGUF's declared context. All five staged 512-token containers
matched the original automatic packs byte for byte outside the position tensor.

The staged Q8_0 packs for MiniLM and all three BGE models served 512 tokens per input, including
batch 4 at 2,048 total tokens. Their lowest cosine across the two single inputs and the batch
was 0.999443 for MiniLM, 0.999891 for BGE-small, 0.999720 for BGE-base and 0.999717 for BGE-large.
MiniLM's 512-token run is exploratory beyond its published 256-token serving limit. Raw outputs
are under `.plans/bert/q8-long-quality-*.log`. This verifies the staged weights and kernels
while the automatic release pack still needs to preserve those position rows.

The staged Nomic Q8_0 pack also served exact 512-token singles and batch 4 with
`search_document`. The minimum CPU cosine was 0.997649. The unmodified OFLM test suite
still failed its generic requests without the required prompt. This run covers 512 tokens
with the standard rotary settings. It leaves the 8,192-token dynamic RoPE gate open.

## Model features

- GTE dense retrieval passed a short multilingual probe using an English query, related
  German, Spanish, Chinese and Russian passages, and unrelated English and French passages.
  The lowest CPU cosine was 0.999642 across all seven vectors. The gap between the weakest
  relevant passage and the strongest distractor was 0.2652 on NPU and 0.2634 on CPU.
- Nomic accepted `search_query`, `search_document`, `clustering` and `classification`.
  All four vectors differed for the same text and matched their task-specific CPU references
  with a minimum cosine of 0.998892. The server returned 768 values for a request with
  `dimensions: 256`. Resizable output remains open.

Raw report `.plans/bert/features.log`, SHA-256
`52028291087aa646355099bce62fd0a3053bfa0df835e1f21bd20ce60d224901`.

## OFLM side

- Make a clean GTE pull reproducible by pinning the known config revision or accepting and
  validating the publisher's updated metadata. Demonstrate a cold pull and pack on the chosen route.
- Keep the declared model lengths, packed position capacity and shipped design coverage aligned.
  Verify `oflm serve`, `oflm-test --embedding`, the benchmark page, the Recommended models table,
  and the requested published-GGUF setup for each tag.
- Teach the embedding test suite to send a supported task prompt for its ordinary Nomic requests.
  Keep its separate task and error-path coverage.
- Map the encoder's over-limit error to an HTTP 400 with the input length and supported limit.
- Route registered embedding entries to NPUE by declared family and design. Keep the actual
  GGUF quant in the registry metadata and report the runtime packing format separately.
- Honor and validate supported embedding dimensions. Expose GTE's sparse output or document
  the dense-only route and its reason.
- Keep the 64-token throughput route and select larger compiled rows for longer inputs.
  Cache compatible sequences in one context where the physical image matches. Measure the
  steady-state dispatch path across length changes before promoting a release default.
