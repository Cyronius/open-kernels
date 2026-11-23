# BERT embedding production audit

This audit covers the six encoders in [OFLM #240](https://github.com/Atomic-Germ/OpenFlowLM-Next/issues/240).
The OFLM inference entry point is `oflm serve --embed 1 --embeddingmodel TAG`.
`oflm bench-embed TAG` reports encoder latency and throughput across batch sizes.
The Phlegm development route uses `phlegm serve --embed-model PACK --embed-worker EXE
--embed-design DIR [--embed-attention DIR]`.
`phlegm bench embed` uses the same resident adapter. Its JSON records exact
fixture token counts, worker latency with IPC, median throughput, machine
metadata and artifact hashes. HTTP remains separately measured with
`utilities/bert-embeddings/check.py`.

## Production bar

| Gate | Evidence and remaining work |
|---|---|
| Block execution | The checked release designs have four operations per layer and four batch tiers per family, under one resident xclbin. MiniLM loaded all 16 streams on the NPU and passed the batch-identity warm-up. The long-input designs need the same runtime check. |
| Performance | The published OFLM page measures short-input batching. Add matched CPU comparisons, at-limit latency and throughput, and the cost of the host stages. The Windows host is a Ryzen AI 9 HX PRO 370, Windows build 26200, NPU driver 32.0.20102.3930. |
| Context | MiniLM served its published 256-token limit and BGE served 512-token inputs. GTE matched the publisher at 8,192 tokens. Nomic passed local publisher parity at 8,192 tokens with dynamic RoPE. Phlegm's length routing needs hardware parity and measurement. |
| NPU coverage | QKV projection, attention output projection, and both FFN projections dispatch to the NPU. The local GTE and Nomic worker also dispatches attention and softmax to the NPU. Tokenization, embedding lookup, normalization, activation, and pooling remain on the host. Release serving integration and profiling are open. |
| Speculative decode | Not applicable to an encoder. |
| Published input weights | The registry currently pulls safetensors. The cold tool rebuilt all six Q8_0 candidates directly from GGUF. Serving evidence covers MiniLM at 256 and BGE, GTE and Nomic at 512. The long quantized Phlegm runs and automatic registry installation remain open. |
| Serving and features | Run `oflm-test --embedding` on each tag. Exercise Nomic task prefixes and GTE multilingual inputs. Check GTE sparse output support and the publishers' elastic dimensions separately from full-width dense vectors. |

| Tag | Published input length to verify | Design family |
|---|---:|---|
| `all-minilm:l6-v2` | 256 in the published Sentence Transformers configuration | `BERT-h384-bfp16` |
| `bge-small:en-v1.5` | 512 | `BERT-h384-bf16` |
| `bge-base:en-v1.5` | 512 | `BERT-h768-bfp16` |
| `bge-large:en-v1.5` | 512 | `BERT-h1024-bfp16` |
| `nomic-embed-text:v1.5` | 8,192 with the publisher's dynamic RoPE scaling | `BERT-h768-gated-bfp16` |
| `gte-multilingual:base` | 8,192 | `BERT-h768-gated-bfp16` |

The Phlegm `5103c86` release build passed a synthetic benchmark with text and
8,192-ID inputs, rejecting token-count mismatches and preserving existing
reports. The six-model hardware matrix is staged with Q8_0 packs, mixed exact
fixtures and batches of one and four. It covers 64 tokens and the lengths in
the table, plus 512 tokens for both long-context models. The live Inkling
session still owns the shared NPU. Hardware results through this entry point
remain open.

Phlegm `718125f` added `--embed-short-design` and optional
`--embed-short-attention` to both serving and benchmarking. Text routing
uses the model tokenizer and task prefix to check capacity. Pretokenized
inputs use their exact lengths. The entire batch takes the short route
only when every input fits. Synthetic HTTP exercised the short boundary,
a mixed-length batch, the 8,192-token ceiling, invalid requests and
short-long-short recovery. A synthetic benchmark recorded both routes.
The queued HTTP run repeats the same 64-token request twice after serving
at maximum context. It records both latencies and requires both results
to equal the earlier short-route vector bit for bit. This measures the
first short request after a route change against an immediate repeat.

The development implementation uses two resident workers because the NPUE
encoder geometry is process-wide. It keeps one shared Phlegm device lock
and dispatches one request at a time. The queued serving check records each
worker's private bytes and working set after startup, with and without the
short route. A Windows dry run using synthetic workers verified the one- and
two-process snapshots. The counters cover the worker processes. The totals
for working sets may count shared pages twice. Measure on hardware and check
the context-switch cost before enabling the route by default. R-0003 and
R-0008 in xdna2-notes document the cost of adding contexts for variants.

The resident worker can report per-request stage timings to Phlegm `06c3e47`.
`phlegm bench embed` records the selected context, projection dispatch count,
total encoder time, and stage times alongside each worker round-trip sample.
The stages cover input preparation, rotary transforms, projections and
synchronization, host and NPU attention, normalization, activation, and
pooling. Stage collection is requested by the benchmark. Older workers may
omit it. The Windows CLI accepted synthetic profiles at contexts 64 and
8,192 and rejected an invalid timing without writing a report. The runtime
patch applied to the pinned OFLM source and reproduced the compiled worker
header. Stage measurements from real Phlegm NPU requests remain open.
Phlegm `f124813` verifies the reported context against the route expected
for each fixture. The pending short-input comparison uses the same two
64-token cases (batch sizes one and four) with and without length routing
for each model, followed by the full context sweep.

The queued Windows run also serves each Q8_0 GGUF with the benchmark's
worker, cache, projection and routing options. It compares single requests
at each tested length and a four-input 64-token batch with the publisher's
FP32 reference. The gate requires exact token counts, ordered unit vectors,
cosine similarity of at least 0.995, and HTTP 400 at the model limit plus one.
MiniLM publisher vectors replayed through a synthetic worker passed this
check through Phlegm HTTP. All six hardware comparisons remain open.

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

## Local encoder stage profile

All 16 cases completed with five warm measurements each on the Windows Strix Point
host documented above. The local driver used 12 host threads divided across two
encoder lanes. Every case was a single input and used the same four NPU GEMM routes
per layer. MiniLM used 24 dispatches, BGE-large 96, and the other models 48.

| Model | Input / capacity | Wall ms | NPU GEMM wait ms | NPU sync ms | Host QK and AV ms | Host softmax ms | Host norm ms | Conversion and bias ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MiniLM | 256 / 512 | 31.56 | 6.83 | 0.62 | 16.58 | 3.47 | 1.07 | 2.50 |
| BGE small | 512 / 512 | 167.65 | 114.64 | 1.79 | 34.85 | 7.50 | 2.31 | 4.94 |
| BGE base | 512 / 512 | 220.29 | 142.86 | 2.44 | 53.40 | 8.18 | 3.78 | 8.59 |
| BGE large | 512 / 512 | 761.36 | 466.87 | 6.37 | 202.37 | 25.25 | 10.28 | 24.62 |
| Nomic | 2048 / 2048 | 2002.67 | 329.68 | 11.71 | 1422.83 | 162.73 | 14.96 | 44.53 |
| Nomic | 2048 / 8192 | 61591.80 | 553.43 | 21.81 | 58407.00 | 2361.66 | 65.11 | 157.57 |
| Nomic | 8192 / 8192 | 61854.60 | 560.24 | 22.33 | 58659.10 | 2398.53 | 63.80 | 154.19 |
| GTE | 512 / 512 | 305.66 | 193.66 | 2.99 | 56.08 | 9.84 | 4.32 | 31.72 |
| GTE | 8192 / 8192 | 59871.10 | 589.37 | 21.26 | 56155.10 | 2453.07 | 64.10 | 515.47 |

Columns report separate medians. The fused activation runs in the conversion path.
The short cases showed large variation in device wait. For example, MiniLM at 64
tokens ranged from 7.09 to 124.02 ms wall time, and its device wait ranged from 3.61
to 117.80 ms. At 8,192 tokens, Nomic and GTE spent over 90% of the median call in
the host QK and AV passes. Moving those operations remains the priority for long input.

Raw report `.plans/bert/profile-matrix.log`, SHA-256
`2a683c0d4cf24737e37fee381f45ae6d11aebabef62534d91f3f30bc5a5f8ab4`.

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
0.16?0.18 s to 2.33?2.52 s. The profile measures a local encoder call. Five warm
calls with an exact 8,192-token Nomic input took 61.00?63.71 s. Host QK and AV
accounted for 57.68?60.46 s and softmax for 2.36?2.46 s. Five corresponding GTE
calls took 58.28?61.68 s, with 54.58?58.04 s in QK and AV.

A local integration build is queued on the two long-context families. It routes
post-projection Q/K/V through the NPU attention image, selects the mask from the
actual input length, and chooses Nomic's rotary table per input. The existing NPUE
weight layout folds the attention scale into Q. Detiling the first projection of
each actual NPUE pack produced exact BF16 matches against the publisher tensors
with Q multiplied by 1/8 and K and V unchanged. The GTE QKV bias matched the same
scaling rule, and the Nomic bias was all zeros. The probe reverses that power-of-two
scale during BF16 packing before using the checked attention image.
The local probe built successfully with MSVC. It reuses the static rotary table
through 2,048 tokens and builds the dynamic table once per unique longer length
in the request. It checks a
2,048 ? 8,192 ? 2,048 sequence within one instance. The same-binary GTE comparison at 8,192 tokens measured 74.58 s with host
attention and 1.77 s with NPU attention. The two vectors had cosine 0.999731.
Dibs reported background compilation and another benchmark at the end of the
paired run, so these are observed times. Raw report
`.plans/bert/integrated-npue-match.log`, SHA-256
`7afcf43580d67d5ccf8525aead14d92a2ae2fc5b969a85e157c99a0fc0d2fffd`.

| Model | Input / capacity | Local wall ms | NPU attention ms | Attention packing and sync ms | Publisher CPU cosine |
|---|---:|---:|---:|---:|---:|
| GTE | 512 / 512 | 363.34 | 72.56 | 3.92 | 0.999194 |
| GTE | 8192 / 8192 | 1703.32 | 673.61 | 121.95 | 0.999424 |
| Nomic | 2048 / 8192 | 1415.95 | 672.80 | 31.81 | 0.997835 |
| Nomic | 8192 / 8192 | 1580.31 | 676.46 | 126.37 | 0.997498 |

All four local integration cases passed parity. Nomic's two 2,048-token outputs
were byte-identical across the 2,048 ? 8,192 ? 2,048 sequence. The prototype used
one lane and 12 host threads. It is a standalone NPUE run. The normal serving
route still uses host attention and requires its own integration and validation.
The [tracked attention adapter](../../npu_offload/gemm_rtp/bert_attention.hpp) and
[per-input rotary base](../../npu_offload/gemm_rtp/nomic_rope.hpp) reproduced all
four validated local outputs byte for byte. The adapter reads the exported device
image and instruction stream after artifact verification, consumes the existing
NPUE-scaled QKV layout, and returns token-major FP32. Raw replay
`.plans/bert/tracked-npue-run.log`, SHA-256
`604a8d1f6d2ef11a3c3438a35936aa5d04a127af31c0b337f8d91a434f2c3a0c`.

The [NPUE probe build](../../utilities/bert-embeddings/build_npue_probe.py) builds
the same route from tracked sources in an x64 MSVC developer shell. It archives
`src/open_npue` from OFLM commit `9b1409d3e150e3602936307f5ff725c7bb4ac8af`
and applies the [encoder patch](../../open_kernels/upstream-patches/bert-embedding-probe-runtime.patch)
in a temporary directory. The build checks the patch and both versions of the
encoder header against the SHA-256 values in `open_kernels/upstream-patches/sources.json`,
using LF line endings for the hashes. Use `--oflm` for a checkout containing that commit,
`--xrt-include` for the XRT core include directory, `--xrt-lib` for the coreutil
import library, `--boost-include` for the Boost headers, and `--output` for the
executable. Put the matching XRT runtime libraries on `PATH` for execution.

Set `NPUE_BERT_ATTENTION_PROBE` to an exported attention directory. Set
`NPUE_NOMIC_DYNAMIC_ROPE=1` for Nomic v1.5. Run the executable with `PACK DESIGN
FIXTURE PREFIX OUTPUT` and append `FIXTURE OUTPUT` pairs to reuse the resident
encoder. The build replay matched all four recorded vectors byte for byte and
again produced identical Nomic outputs before and after the 8,192-token request.

Pass `--worker` to the same build command to build the resident NPUE text
worker. Run it with `PACK DESIGN [ATTENTION-DIRECTORY]`. The worker
loads one model, enables Nomic v1.5's per-input rotary scaling, reports its
capacity and prompt names in a protocol-1 greeting, and accepts newline-delimited
JSON requests with `texts` or `token_ids`. Text callers may use `prompt_name`.
Token-ID callers supply unpadded sequences including any task prefix and special
tokens. Its stdout carries only protocol records. Device diagnostics go to stderr.

The worker and probe verify the attention manifest geometry, both payload sizes
and both SHA-256 digests at startup. They check that the attention and projection
designs have the same sequence length, then give XRT the checked bytes from
memory. The native loader accepted the 512-, 2,048- and 8,192-capacity exports
and XRT read the same xclbin UUIDs from memory and disk. The worker refused
a bit change in either payload, a shortened instruction stream, a changed
runtime marker and a missing manifest before opening the model. Raw report
`.plans/bert/native-attention-loader.log`, SHA-256
`7f196a767250d44d143facb3025870d84b3ff2d43e2779f829c0d3d6a6f6e1ea`.

On the Strix Point NPU, the built worker returned byte-identical vectors for
GTE at 8,192 tokens and Nomic at 2,048 and 8,192 tokens. It preserved the Nomic
batch-of-two order and output bytes. Requests above capacity returned the actual
token count and the 8,192-token limit as an invalid request. A valid Nomic
request after that error produced the original short vector. Raw log
`.plans/bert/npue-worker-smoke.log`, SHA-256
`0b1ffe1006e62a3ed71a1e06d8586a82b5cfd35da8980b4b27e032957919e043`.

Pretokenized inputs use the same tier selection, masks, embeddings and
pooling as text. The offline native check loaded the extended Nomic and GTE
Q8_0 packs. For both encoders it reproduced every token, mask and padding
entry from their 8,192-token text fixtures. It rejected the 8,193rd input
token, empty sequences and IDs outside each vocabulary. Raw report
`.plans/bert/token-ids-prep.log`, SHA-256
`bcc455644c0e580a02d81ab0603f2ee75d3d77efc3a5d811ec855d78ea082426`.
The Phlegm HTTP test with a synthetic worker exercised 8,192 flat IDs, an
ordered mixed-length batch and HTTP 400 at 8,193 IDs. Live vector equivalence
through Phlegm remains to be verified on the NPU.

Raw run `.plans/bert/integrated-npue-run.log`, SHA-256
`446873b6e77e41690d294def3268e3bd83a78b19107bdb914349f0d604880fa1`.
CPU comparison `.plans/bert/integrated-npue-verify.log`, SHA-256
`99544812427dfd9a8634b20a035ede02a4b9f66602c006925159ddf218008a09`.
The [BERT attention exporter](../../open_kernels/export_bert_attention.py) now records
capacity, head geometry, BF16 layout, scale, runtime length marker, and both payload
hashes. The hardware checker uses that contract, and the integration probe checks
it before starting the encoder. Offline validation accepted the two known images
and rejected both a different requested capacity and a bit-changed instruction stream.
An independent 512-token build through the exporter passed manifest validation
and reproduced the earlier instruction bytes. Its NPU smoke is queued.

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
1.13 to 1.28 ms. The 2,048-capacity runtime passed at 1,027 and 2,048 valid keys.
Warm dispatches took 14.39 and 12.97 ms in those two cases. At capacity 8,192,
the runtime passed at 2,051 and 8,192 valid keys with warm dispatches of 56.18 and
58.71 ms. Alternating valid lengths in one resident context reproduced the
isolated outputs byte for byte at capacities 512 and 8,192. The harness patches
and syncs the same instruction buffer and caches all 32 register offsets. After
the first lookup, the observed full-capacity patches took 0.009?0.011 ms. The checker also
accepts captured Q/K/V in float32, comparing the device output against float64
attention over BF16-rounded inputs and reporting the extra error due to rounding the
original tensors. Layers 0 and 11 were captured at 512 and 8,192 tokens. In the independent CPU
calculation, BF16 input rounding alone gave a minimum sampled row cosine of
0.9998407 and worst head-relative L2 of 0.006247 across the four captures.
Raw report `.plans/bert/gte-real-qkv-rounding.jsonl`. NPU replay of all four captures passed. The lowest sampled row cosine against the
BF16-rounded float64 oracle was 0.997377, and the largest head-relative L2 was
0.028853.

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
`oflm-test --embedding`. Both exact 8,192-token fixtures scored at least
0.999449 against the publisher, with 79.6 and 84.0 s through HTTP. The
serving checker can enforce a named minimum cosine across all configured single
and batch fixtures in addition to the token-count and input-limit checks. The new
[GGUF staging utility](../../utilities/bert-embeddings/stage_gte_gguf.py) reconstructs
the tokenizer, tensors and equivalent legacy RoPE config using the GGUF as its only
input. This leaves native NPUE packing and tokenizer generation to the existing
packer. Staging from GGUF alone produced the same dequantized weights as the
validated replay, SHA-256
`f122116c8429466db7d262be8466a366c34e7964ed3b02bd52e462587e81392e`.
The tokenizer JSON also matched the publisher bytes. The isolated model directory
initially contained only the reconstructed config, weights and tokenizer. The
default registry compared the staging files with its entry for the original
safetensors, redownloaded that checkpoint and its modern config, and then fell
back to the wrong packer. With a local registry and the new provenance record,
OFLM cold-packed all 150 tensors and generated the tokenizer table. Its automatic
pack capped the position table at 64 rows, so startup refused the 512-token
design. The extension utility verified the baked rotary frequencies against
the GGUF and expanded the unused position table. The resulting 8,192-token NPUE
was byte-identical to the separately built and served reference, SHA-256
`ac9b5460d9a1bf4b278f03cebcd12282594cd747e60a6991b93d43950a30c9a7`.
Stock `oflm serve` loaded this exact extended pack with the 512-token design and
the local registry. Two single requests and one batch of four at each of 64 and
512 tokens passed the CPU comparison, with minimum cosine 0.999379 against the
0.995 gate. `oflm-test --embedding` passed all nine applicable checks. The suite
skipped its bundled reference and task prompt checks for this model. The separate
CPU comparison supplies the numerical reference. Raw logs under `.plans/bert/`
are `gte-gguf-local-extended-quality-gte-multilingual_base.log` (SHA-256
`bcab236ad34a0e5bc587a8fcc7750bb612389aea3a01adb2b295c27421d59d58`) and
`gte-gguf-local-extended-oflm-test-gte-multilingual_base.log` (SHA-256
`ffbe16050dd30e677e6a5f537647231fceeb1a2a7078da28c02db884d1472203`).
Native registry, automatic full-length packing and GGUF dispatch remain OFLM work.

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

These release-design runs cover 32 and 64 tokens. Raw reports are
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

The CPU replay also compared these four Q8_0 GGUFs with their publisher
checkpoints at the serving limits. Each comparison used two mixed inputs.
Every tokenizer ID matched the rebuilt WordPiece vocabulary. The comparison
pools the final hidden states, so the original classifier pooler is unused.

| Model | Input tokens | Minimum cosine |
|---|---:|---:|
| MiniLM | 256 | 0.999767 |
| BGE small | 512 | 0.999891 |
| BGE base | 512 | 0.999900 |
| BGE large | 512 | 0.999910 |

Raw report `.plans/bert/bert-absolute-q8-cpu-limit-parity.json`, SHA-256
`071b2f20cfbb63c67ada7416b20dae9a196dc766df465e43c54d38a681a752ca`.

The staged Nomic Q8_0 pack also served exact 512-token singles and batch 4 with
`search_document`. The minimum CPU cosine was 0.997649. The unmodified OFLM test suite
still failed its generic requests without the required prompt. This run covers 512 tokens
with the standard rotary settings. It leaves the 8,192-token dynamic RoPE gate open.

Nomic's published GGUF declares 2,048 tokens as its default and documents an
8,192-token extended context. The `--nomic-dynamic-rope` option in
`extend_npue.py` exposes that capacity for packs with matching Nomic v1.5
identity and rotary geometry. The ordinary packing ceiling stays 2,048.
[Publisher GGUF card](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF)

The published Q8_0 GGUF with SHA-256
`3e24342164b3d94991ba9692fdc0dd08e3fd7362e0aacc396a9a5c54a544c3b7`
produced an 8,192-position NPUE pack with SHA-256
`121598c1e75bc06030dd572a5d2a585669abcfb305d289cde94163e5076047f9`.
The 149 non-position tensors were byte-identical to the 512-token GGUF pack.
The utility rejected 8,192 without the option and 8,193 with it.
Runtime parity for the extended quantized pack remains open.

## Attention candidates for MiniLM and BGE

The runtime-length design now compiles 12 heads of width 32 for MiniLM at
256 tokens and BGE-small at 512 tokens, and 16 heads of width 64 for BGE-large
at 512 tokens. The binary loader verifies these manifest geometries and
requires them to agree with the selected encoder. The worker and profiler
both build with the new shapes. The original 12-by-64 format remains readable.
Its 512-token control rebuild reproduced the recorded instruction bytes.

The instruction checker found one runtime length update on each of the
32 compute tiles for all three new images. A separate 512-token rebuild
reproduced the instruction stream for each geometry. The linked core
programs and allocated local-memory window fit the R-0001 budgets.

| Geometry and capacity | Largest program | Stack reservation | Largest reserved local extent plus static data |
|---|---:|---:|---:|
| 12 × 32 at 512 | 13,904 B | 2,048 B | 31,620 B |
| 16 × 64 at 512 | 14,992 B | 2,048 B | 62,340 B |

An offline check of each first-layer GGUF-derived QKV pack found the expected
`1/sqrt(head_width)` fold on Q weights and biases. K and V matched their
unscaled sources. The adapter reverses the Q fold before the FlashAttention
image applies its native scale. CPU reference captures at the first and last
layers of all three encoders are staged for device comparison.

The hardware accuracy, dispatch and HTTP gates remain open. Raw instruction
and ELF report `.plans/bert/attention-shape-offline.log`, SHA-256
`bf1ced9c0716cf7da08a6f1e92c8139bffba57f697920db5a8077fd7922710de`.
Packed operand check `.plans/bert/attention-shape-qkv.log`, SHA-256
`e4483d1d655843795702cc042153ffe21660bd7140cc1647a16b555ab3d271b4`.

## Cold packing from GGUF

Build the CPU-only packer from an x64 MSVC developer shell with
`build_npue_probe.py --packer --oflm CHECKOUT --output pack-npue.exe`.
`pack_gguf.py WEIGHTS DESIGN OUTPUT --packer EXE --gguf-sha256 HASH` converts
one of the six Q8_0 files to a deployable NPUE container. The input directory
needs only the GGUF, and the tool rebuilds the tokenizer and staged tensors.
The projection design supplies sequence capacity and tile dimensions. GTE
includes its F32 sparse head, and Nomic packs the published dynamic context.

All six cold packs matched the corresponding staged packs in every tensor.
GTE was byte-identical as a container. MiniLM, BGE and Nomic have different
staged safetensors hashes. BGE records the corrected 512-position slice in its
fusion metadata. Nomic keeps the GGUF's layer-norm epsilon metadata, which
the current native runtime implements as 1e-12.

A second cold run reproduced all six file hashes. The tool
refused a mismatched GGUF checksum, a 512-token MiniLM design and a gated
layout for ungated BGE. Raw comparison `.plans/bert/direct-gguf-parity.log`
has SHA-256 `b7ebe567465027be4ce6d691ad9aa8d436d70c184ede5258ccdb5d5b0e5a6da3`.
Raw replay `.plans/bert/direct-gguf-replay.log` has SHA-256
`d7893f5073b26f9e1e0aecce514ea169746be943a67f792438da5e00babe1ac8`.

Phlegm `5c133b9` accepts the GGUF directly in `serve --embed-model` and
`bench embed --embed-model` with `--embed-packer` and
`--embed-gguf-sha256`. Use `--embed-pack-script` to point a development
checkout at the packing script. The Python packaging environment needs
`gguf`, `numpy` and `safetensors`. Packing runs before Phlegm acquires
the device lock. The cache keys on source content, design, packer,
packing scripts and dependency versions. It verifies the GGUF on every
startup and checks the NPUE against its receipt before reuse.

Two concurrent MiniLM cold starts published one cache entry. The warm
lookup reused the same file. The cache rejected a modified NPUE and an
incorrect GGUF checksum. A Phlegm startup test used the actual Q8_0
MiniLM GGUF, then a native test worker opened the pack and returned a
384-dimensional synthetic result. The benchmark reused the pack and
reported the GGUF and NPUE hashes. Device parity and performance on the
GGUF startup route remain open.

The Windows benchmark environment rebuilt MiniLM's 256-token pack in the
same cache used by the queued run. The verified warm lookup returned the
same bytes as the independent cold pack, SHA-256
`82da004f2857af6d0a764b316c119260707f75eb361e1e9f123f90285f3992cd`.
Preflight checked that both projection designs for each encoder use the
same packed weight layout and that the reference checkpoint metadata is
available.

Cache evidence `.plans/bert/cached-gguf-pack.log`, SHA-256
`2dd1a76b18adc918abd7fa7261d1774367d57650e7766294743ad3709d6384ec`.
Startup evidence `.plans/bert/phlegm-gguf-startup.log`, SHA-256
`03f1ba139327fa1733aee1c0540f1c56ad08dcd66489c0bf1febb57aad2b0797`.

## Model features

- GTE dense retrieval passed a short multilingual probe using an English query, related
  German, Spanish, Chinese and Russian passages, and unrelated English and French passages.
  The lowest CPU cosine was 0.999642 across all seven vectors. The gap between the weakest
  relevant passage and the strongest distractor was 0.2652 on NPU and 0.2634 on CPU.
- The resident worker accepts GTE elastic dense widths from 128 through 768
  and normalizes the requested prefix. Replaying its native postprocessing on
  the captured NPU vectors at 512 and 8,192 tokens matched the publisher CPU
  reference at widths 128, 129, 256, 512 and 767. The minimum cosine was
  0.999195. Live HTTP parity remains open.
  [GTE model card](https://huggingface.co/Alibaba-NLP/gte-multilingual-base)
- The stock GTE NPUE pack drops the classifier that the GGUF preserves.
  `extend_npue.py --gte-sparse-head` adds the original F32 weights and bias as
  `sparse.weight` and `sparse.bias`. The 8,192-token pack with SHA-256
  `0d971e06a80b016e4efa67989dec272d564ddfdd0c934d2e4ec9c8967baa7cd3`
  preserves all 150 existing tensors byte for byte. On English and Chinese
  fixtures, applying these tensors to the publisher CPU model's final hidden
  states reproduced `GteForTokenClassification` logits exactly.
- Phlegm accepts `return_sparse: true` for GTE packs carrying the head.
  The worker scores occupied token rows from the resident encoder. Each item
  returns its dense vector and `sparse_embedding` with sorted token-ID indices
  and positive scores. Special IDs are excluded, and repeated token IDs keep
  their highest score. The native scoring replay on publisher CPU hidden states
  matched all positive IDs on English, Chinese and 8,192-token fixtures. The
  maximum absolute score difference was 4.77e-7. Live sparse parity through the
  quantized NPU route remains open.
- Replaying the converted Q8_0 weights on the CPU preserved the positive
  token IDs for the short English and Chinese probes. Sparse cosine was
  0.999995 and 0.999998, with maximum score differences of 0.00704 and
  0.01234. This isolates the effect of quantization before the live NPU
  comparison. Raw report `.plans/bert/gte-q8-sparse-cpu.jsonl`, SHA-256
  `2cd5d9c15392178014f3ae7d7673602460ee47e69407a72f6d207845f7ee24c1`.
- The same CPU comparison at 8,192 tokens preserved all 54 positive token
  IDs. Sparse cosine was 0.999975 and the maximum score difference was
  0.01618. The fixture is the one queued for the live Phlegm sparse check.
  Raw report `.plans/bert/gte-q8-sparse-cpu-8192.json`, SHA-256
  `ab39a4f90b2487d6b7eb711e1f99f8bfd86bfb87b17c335ab8c708da6a529156`.
- Nomic accepted `search_query`, `search_document`, `clustering` and `classification`.
  All four vectors differed for the same text and matched their task-specific CPU references
  with a minimum cosine of 0.998892. Stock OFLM returned 768 values for a request
  with `dimensions: 256`. The resident worker and Phlegm accept 64, 128, 256,
  512 and 768. The reduced-width route centers the full vector before taking and
  normalizing the prefix. [Publisher Matryoshka reference](https://huggingface.co/nomic-ai/nomic-embed-text-v1.5#adjusting-dimensionality)
- A native replay of the postprocessing code on saved NPU vectors matched the
  publisher CPU reference at 2,048 and 8,192 tokens for all four reduced widths.
  The minimum cosine was 0.997466 and all norms were within 1e-6 of 1.
  End-to-end Phlegm validation remains open while the Inkling run holds the
  shared NPU lock. Raw report `.plans/bert/nomic-matryoshka-native.log`, SHA-256
  `e5c0433b2b8694a3605a80b5f0f3c2a0d38f43c7046805c83c7af6444c26d0bc`.
- A CPU replay of the publisher's Nomic Q8_0 GGUF matched the original
  checkpoint using dynamic RoPE at 2,048 and 8,192 tokens. Across dimensions
  64, 128, 256, 512 and 768 the minimum cosines were 0.99840 and 0.99821,
  respectively. The GGUF has all 30,522 tokenizer IDs. The original
  checkpoint has six additional, unassigned vocabulary padding rows.
  Raw report `.plans/bert/nomic-q8-cpu-context-parity.json`, SHA-256
  `13dc285fb34e1bc3f2ce297ef5a84c2e99cfe2a01a86c56c97b2931badba5f00`.

Raw report `.plans/bert/features.log`, SHA-256
`52028291087aa646355099bce62fd0a3053bfa0df835e1f21bd20ce60d224901`.
GTE elastic replay `.plans/bert/gte-elastic-native.log`, SHA-256
`baadf3a3268aac18b105abc485aeda045fcc7b555a6ee79d525bdb759ec180e5`.
Classifier check `.plans/bert/gte-sparse-head-check.log`, SHA-256
`09bea59684021247b3d9d9b82ab788c1db5c394f9488b6d7e0480f15d98e536b`.
Native sparse replay `.plans/bert/gte-sparse-native.log`, SHA-256
`4367ef8b69b4eb553ed7a10831b095fac794cf8ead8d356ce262ea0770494f73`.

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
