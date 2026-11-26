# Inkling 2 Mini to the production bar

Bar: `AGENTS.md` "Production bar for OFLM kernel sets". Progress: [RECOMMENDED_MODELS.md](../../../RECOMMENDED_MODELS.md).
Written 2026-11-21 after the model's release on 2026-11-20. Scope covers text, image input, DSpark, reasoning effort
and tool calls. Model acceptance uses `phlegm run`, `phlegm serve` and `phlegm bench`.
Reusable findings live in the [Inkling 2 Mini port notes](../notes/inkling2-mini-port.md).

## The model

The 2026-11-22 check used the publisher's pinned config, the raw GGUF headers and the pinned llama.cpp graph.
[Publisher config](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/b85c3cfa615fc413cc9af3379f491b963f3b586a/config.json).
[llama.cpp Inkling 2 merge](https://github.com/ggml-org/llama.cpp/commit/29b52d398b294509746ec0a3f28948a3107c0213).

| | Inkling 2 Mini |
|---|---|
| size | ~26.2-26.4B total (pages disagree), 3.4B active a pass, ~5.4B effective a token with the loop |
| layers | 24 physical, **48 visits a token**: 0-5 once, **6-17 three times** (`loop_config`: core 6-18, 3 passes, gate rank 64), 18-23 once |
| hidden | 2304 |
| attention | 24 q heads, 8 kv heads, head dim 128. Sliding window 1024 in 20 layers; full in 5, 11, 17, 23 (L L L L L G) |
| attention extras | query-dependent sink and output gate, learned relative-position logits (`d_rel` 16, `rel_extent` 1024), no text RoPE, q/k L2 clip 22.63, value RMS norm without gamma. Full attention scales Q and the relative query by log position after clipping. Each visit has causal four-tap convolutions on K, V, attention output and MoE output. Full attention adds dynamic per-head K/V taps |
| loop extras | cross-pass messages read by each core layer through a learned query; four rank-64 gates into q, k, v and the MoE input; in passes 2-3 each core layer attends over two residual blocks of the last pass (`n_layers_per_attn_res_block` 6). Each visit has its own KV and conv state. Controller weights are F32 in the GGUF |
| MoE | 128 experts, top 8, no shared expert, width 1152. Add the correction bias to sigmoid router scores for expert selection. Normalize the selected unbiased scores and multiply by 8. SiTU applies separate softcaps to the gate and up projections (4.0 and 25.0), with sigmoid on the raw gate |
| vocab / head | 201,024 (200,058 used), final logit softcap 30 |
| context | 1,048,576 |
| vision | `inkling2_vision` in the publisher config, projector type `inkling2v` under GGUF architecture `clip`. 23 layers x 1024, 16 heads, MLP x4, patch 20, temporal patch 2, merge 2x2, 32x32 position grid, projector 4096 into 2304. Images 40-4096 px a side. mmproj BF16 642 MB / Q8_0 346 MB. Image input, text output |
| drafter | **DSpark**: 3 layers (sliding in 0 and 2), RoPE theta 1e6, width 9216, block size 8, rank-256 Markov head. Reads the inputs to chronological visits 5, 23 and 41, plus the final normalized output at index 48. `dspark-Inkling-2-Mini-Q8_0.gguf`, 402 MB. [SGLang release notes](https://huggingface.co/blog/inkling2mini#speculative-decoding) |
| reasoning | `reasoning_effort` none / minimal / low / medium / high (default) / xhigh / max, or a number 0-0.99, written into a system message |

GGUF sources:

- **ggml-org** (mainline llama.cpp layout, arch shown as `inkling2_loop`): IQ4_NL 17.6 GB (experts of layers 0-19
  IQ4_NL, the rest Q8_0, imatrix), Q8_0 27.8 GB, BF16 52.2 GB. KLD to BF16: IQ4_NL 0.0095, Q8_0 0.0030.
- **Unsloth** publishes a fork layout and a `Mainline_Support/` layout. Both inspected mainline files use
  `inkling2_loop` and retain 40 NVFP4 expert tensors. UD-Q4_K_XL has F32, BF16 and Q8_0 for the rest. UD-IQ4_XS
  also has K-quants and several other types. [Unsloth mainline files](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/tree/45fd08764f2b0bf16b03a5483a80e5c7aaec8772/Mainline_Support).

## Reference evidence (2026-11-22)

| Artifact or check | Result |
|---|---|
| Publisher config | Revision `b85c3cfa615fc413cc9af3379f491b963f3b586a`, SHA-256 `a7a349601f0984c4abf999c480ef775d8567d21f5f84b4e8a3e035b44f288dc2` |
| ggml-org IQ4_NL weights | Revision `e8a2cecf0e78914f07cc27b9be711a9b2c7561d9`, local SHA-256 `cbdf3fa9dc50a4124977980f37da4f1fa764b0cfe800d275cd43d26ce2a81af4` matches publisher LFS |
| IQ4_NL, Q8_0, BF16 text headers | 704 tensors each. IQ4_NL has 40 IQ4_NL, 186 Q8_0 and 478 F32 tensors. The Q8_0/BF16 files use their named types for all 226 non-F32 tensors |
| Vision and DSpark headers | BF16 mmproj has 422 tensors and projector `inkling2v`. DSpark has 54 tensors, architecture `dflash`, Markov rank 256 and target layers 5, 23, 41, 48 |
| Vision and DSpark weights | Local SHA-256 `e914997fe07fb2066bfdce855ec7275c15bc2b624e5423149f1027839593d9cb` for BF16 mmproj and `2ec78510da837e8fc3be54a14757a0c5fbd2d8d9e85a0d92853fafdf674f3773` for Unsloth's DSpark Q8_0 |
| IQ4_NL host reader | Native `gguf_pack_test` passes an exact byte and row comparison covering all 16 codes, both nibble halves, two blocks per row and two expert planes. The C++ and Python readers also agreed exactly on 13,824 values from real IQ4_NL and Q8_0 expert rows in layers 0 and 20 |
| Short text prefill | Windows, Ryzen AI 9 HX PRO 370. Same IQ4_NL weights in llama.cpp at `29b52d398b294509746ec0a3f28948a3107c0213` (MSVC 19.44, CPU, eight threads) and the [FP32 reference](../../../open_kernels/model/inkling2_gguf_reference.py). Five input tokens for “The capital of France is”. All 48 residuals have Pearson correlation ≥ 0.99918. Final logits correlate at 0.99994, KL(llama.cpp ‖ FP32) is 0.000133 nats, and the top five IDs agree in order |
| Short cached decode | The FP32 reference matched the first eight of llama.cpp's 64 greedy IDs on the same prompt and IQ4_NL file, using separate KV and four convolution histories per layer visit |
| On-device IQ4_NL codebook | The Strix NPU reproduced 64 of 64 signed codes in a shuffled all-code fixture with two bank-separated tables, each holding the full 16-entry table twice. The 8-entry interleaving experiment gave 28 of 64. This covers the nibble lookup in isolation. Windows XRT 2.21.0, driver 32.0.20102.3930, Peano `c9c5ecb725fc8c765e4b687356e6ec1e54da7a0e`. Probe source `.plans/inkling2/iq4_lookup.cc` SHA-256 `24bc9d073e530c561cf6bee98e50b9577d67e449864283a6ba7af62d829d54f0` |

### Candidate quant comparison

One 1,024-token chunk of WikiText-2 test (raw text SHA-256
`173c87a53759e0201f33e0ccf978e510c2042d7f2cb78229d9a50d79b9e7dd08`) in pinned llama.cpp on the same host,
CPU only, eight threads, batch 1,024 and microbatch 128.

| GGUF | PPL | Mean KL from ggml-org IQ4_NL | Same top token |
|---|---:|---:|---:|
| ggml-org IQ4_NL | 3.6442 ± 0.3499 | baseline | baseline |
| Unsloth mainline UD-Q4_K_XL | 3.6195 ± 0.3478 | 0.004323 ± 0.000411 nats | 97.26% |

Unsloth's published GGUF SHA-256 `e64276656f072e9b2197a61723667cbdb71da66a716ffd966e572c21088f3cd3`
matches the local file. The initial Unsloth run gave PPL 100.294 until the llama.cpp NVFP4 loop-scale fix was
applied. The corrected run adds the patch from llama.cpp #32704 (patch SHA-256
`2cdc04a28422648274758d1d032ea57af50be99e0d1be16d44bfe402e4ffebbc`). The corrected result is in
`.plans/inkling2/unsloth-wiki-patched.log` (SHA-256
`14c640c4d19622b25e480dce6b03cad30196304a61d5e791ffb913c71670915d`). [llama.cpp scale fix](https://github.com/ggml-org/llama.cpp/pull/32704).

Use IQ4_NL for engine bring-up. The one-chunk comparison leaves the shipping quant open. Unsloth's mainline
candidate requires per-expert NVFP4 scales to stay attached to the physical weights across each loop pass.

The 27.8 GB ggml-org Q8_0 file from the same revision verified against publisher SHA-256
`d57542f8aea5d7639beba5fc4ae7f584754916f63ac5d5a69328c27f652f45da`. The pinned llama.cpp build with
Vulkan SDK 1.4.363.0 enumerated the Radeon 890M (Vulkan driver 2.0.395) and completed a Q8_0
1,024-token smoke on WikiText-2's *American Beauty (1999 film)* article with PPL 6.5119 ± 0.7377.
The comparison on the first contiguous 8,192 tokens of that same article used batch 1,024,
microbatch 128, eight host threads and Vulkan on the Radeon 890M. The reference build used llama.cpp
commit `29b52d398b294509746ec0a3f28948a3107c0213` and the NVFP4 loop scale fix above.

| GGUF | PPL | Mean KL from Q8_0 | Same top token |
| --- | ---: | ---: | ---: |
| ggml-org Q8_0 | 8.1072 ± 0.3123 | baseline | baseline |
| ggml-org IQ4_NL | 8.1338 ± 0.3129 | 0.006251 ± 0.000194 nats | 96.557% |
| Unsloth UD-Q4_K_XL | 8.0998 ± 0.3117 | 0.002246 ± 0.000090 nats | 97.998% |

Keep ggml-org IQ4_NL as the shipping candidate. Unsloth is closer to Q8_0 on KL and top-token agreement,
and its perplexity is 0.034 below IQ4_NL on this article. Its NVFP4 expert layout still needs a native
pack and projection path with physical-expert scales. The two files both occupy about 17.6 GB.
The article input has SHA-256 `7454ed0f2b889fe15f71d7ac9a82e0b53fcc03bca8c7cd13b2383e251f9656c1`. Summary
`.plans/inkling2/quant-quality/vulkan/summary.json` SHA-256
`7b8138400de806ee7429422697a2329cbfebaa1ddad49ceb1c013372dd9b19d2`.
[ggml-org artifacts](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9).

The fixed-choice continuation test scored 32 local prompts on the pinned llama.cpp CPU build with
four threads, memory-mapped weights and BF16 K/V. The four task types cover two-step arithmetic,
Python code tracing, JSON field and value selection, and choosing an entry from a table under a rule.
Each answer label appears twice as the gold option per task type. An independent parser and solver
verified the rendered answers. All three GGUFs gave identical prompt tokens, and each answer letter
was a single appended token. The score picks the largest next-token logit among the four option
labels. KL is computed after normalizing over those four labels.

| GGUF | Arithmetic | Code | JSON | Evidence | Total | Same top choice as Q8_0 | Mean choice KL from Q8_0 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| ggml-org Q8_0 | 2/8 | 2/8 | 8/8 | 8/8 | 20/32 | 32/32 | 0 nats |
| ggml-org IQ4_NL | 2/8 | 3/8 | 8/8 | 8/8 | 21/32 | 30/32 | 0.010779 nats |
| Unsloth UD-Q4_K_XL | 2/8 | 2/8 | 8/8 | 8/8 | 20/32 | 32/32 | 0.003996 nats |

IQ4_NL corrected one code answer and changed one wrong arithmetic answer to another. On these
prompts Unsloth remained closer to Q8_0 in option probabilities, consistent with the earlier
WikiText result. Retain the measured IQ4_NL route as the Phlegm candidate. At this stage, completing
the 52.2 GB BF16 download required more local storage. The completed BF16 comparison below uses
the publisher GGUF with the pinned llama.cpp runtime.

The publisher BF16 reference is pinned at `c7679166d5612e1b7e47b71febd8b4a6bca8df30`,
52,226,509,504 bytes, SHA-256 `7074c9762e6769a608c7b23960a7fd2444880b1174a3910149749f6d01389aec`.
The repository headers and publisher pointer agree. An initial sample showed 74.6 GB free.
Free space fell to 44.1 GB before transfer, so the resumable download left no partial file.
Reserve room for the weights and the approximate 1.64 GB WikiText logit tape. The BF16 preflights
match the existing 8,192-token WikiText fixture and verify the prompt and option IDs across all
48 retained ARC cases. [Publisher BF16 file](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/blob/main/Inkling-2-Mini-BF16.gguf)

| Preflight under `.plans/inkling2/quant-quality/` | SHA-256 |
| --- | --- |
| `vulkan-bf16/input.json` | `7640ca3d0ebdc5bfa13972a6503fd63c207948debb97ad575197370031ccc1f1` |
| `arc-test/bf16-preflight.json` | `f926787e7f0517296898a5b8e22a7c1c8019fcbc3f0e7f8fbc72c64df746a155` |

Four 8 MiB samples of the pinned BF16 file used 75–87.5% of their original allocated space under
transparent NTFS compression. Extrapolating the sample mean gives 40.8 GB for the full file.
The measured range varied across the sampled offsets, and the estimate excludes output tapes and
paging. Compressing the existing Q8_0 WikiText logit tape saved 1,582,261,528 bytes with identical
SHA-256 before and after. Sampled LZX ratios for Q8_0, Unsloth and the borrowed DSpark weights are
recorded as estimates. Those weights were kept in their original storage.

The compressed 256 MiB prefix came back as HTTP 206 for the exact published length. Its SHA-256
matched a subsequent read through a Windows memory map. The GGUF header
reports `inkling2_loop`, 226 BF16 tensors, 478 F32 tensors and 26,070,385,152 elements. The
resumable transfer checks the HTTP range and length of each 64 MiB response. It limits bandwidth
to 12 MiB/s alongside the 131K job and stops at the configured 32 GiB reserve. A truncated-range test
rewound to the last checkpoint before retrying. A partial stays under `.part` until the complete
length and publisher SHA-256 pass. The standard llama.cpp evaluation needs the complete file.

On November 23 the C: volume was the only fixed volume and had 31.75 GiB free. The retained
partial occupies 2.87 GiB of allocated space for 3,758,096,384 logical bytes. At the sampled
75–87.5% compression ratios, completing the remaining bytes would take about 33.9–39.5 GiB
of additional allocation, plus the 1.53 GiB WikiText logit tape. The space estimate uses
sampled compression and retains the 32 GiB reserve during the active Phlegm job.

The downloader now accepts a destination directory on another volume and can preflight that
volume's available bytes without opening the transfer. The ARC and WikiText runners accept the
same BF16 path. WikiText can capture the reference before loading any of the other quants, then
run comparisons using the retained tape after that volume is detached. The capture checks the
published model digest, the compiled executable and libraries, the complete tape length, and
the 8,192-token sequence against the retained pinned Q8_0 evaluation. The later comparison
rechecks tape bytes and receipt before scoring. Real fixture preflights and a miniature replay
covered interrupted range recovery, capture and comparison in separate phases, and rejection
of a changed tape. The November 23 resumed transfer verified all 52,226,509,504
bytes against the publisher SHA-256. The first full-offload Vulkan WikiText run
exited before producing a tape. Its temporary native log was removed on exit.
The updated runner retains that log on failure and can limit GPU offload for the
BF16 reference while keeping the candidate quant settings. A recovery run is
queued to diagnose the load before launching a second 8K capture.
The retry's short full-offload Vulkan probe loaded the pinned BF16 model and
completed a 32-token prompt. At 8K the Vulkan driver ran out of device memory
allocating a 209,715,200-byte KV buffer. The 24-layer offload retry keeps the
same BF16 GGUF, context, batches and fixtures. Candidate quant evaluation
keeps the full-offload settings. `bf16-capture-q2v4dq62-failed.log` in
`.plans/inkling2/quant-quality/vulkan-bf16/` has SHA-256
`4deec5d59714d1307d50785e61cf0b72dae7071f1bfc55d9ca683a8de534348d`.
The 24-layer retry also failed, during Vulkan queue submission while loading
the weights. The next run keeps the pinned BF16 model and uses the CPU for
its 8K reference capture. Cached logits still let the three candidate quants
run at full Vulkan offload in a separate step.
The CPU capture finished in the pinned `llama-perplexity` executable with
BF16 perplexity `8.1005 ± 0.31212`. Its 1,638,540,568-byte tape passed the
header, token and length checks. The reuse step validated the saved digest and
began evaluating the candidate quants. `bf16-reference.json` has SHA-256
`c1a2e6eb59d4ec5f013a242bd5153aaa6c650170bdac634024ba8640b398a08a`
and the tape digest is
`054f84ba0ce85e15e9aa16cd9b22ee598c1ccc110bec6e29d57c7dc0abb66ae7`.
The full-vocabulary comparison used the verified CPU BF16 tape and scored
all three candidate quants on Vulkan on the same 8,192-token prompt.

| Weights | Perplexity | Mean KL from BF16 (nats) | Top-ID agreement |
| --- | ---: | ---: | ---: |
| Publisher BF16 | 8.100500 ± 0.312120 | reference | reference |
| ggml-org Q8_0 | 8.107185 ± 0.312327 | 0.002515 ± 0.000111 | 97.778% ± 0.230% |
| ggml-org IQ4_NL | 8.133819 ± 0.312949 | 0.006193 ± 0.000179 | 96.288% ± 0.295% |
| Unsloth UD-Q4_K_XL | 8.099835 ± 0.311732 | 0.002320 ± 0.000100 | 97.924% ± 0.223% |

Unsloth's candidate had the lowest mean KL among the three quants on this
fixture. IQ4_NL remains the Phlegm NPU candidate because the existing NPU
expert kernels support its tensor layout. The Unsloth GGUF still needs
NVFP4 block and decode kernels with separate per-group and per-expert scales.
These measurements include the comparison across CPU and Vulkan backends.
The separate ARC sample below also checks the same-backend answer choices.

| Capture under `.plans/inkling2/quant-quality/vulkan-bf16/` | SHA-256 |
| --- | --- |
| `summary.json` | `3376d7433cadc23bc33d280b51afcadbf32002f82629343c6652c17aca82b1f0` |
| `q8_0.log` | `9052a589f852c62b09a09f02acde6137dee6bcc856fa1d7241f0361a120ed58d` |
| `iq4_nl.log` | `96e52122608317e37675b03db63e79dcc58a4d9627b346bb2dc93decb21d2967` |
| `ud_q4_k_xl.log` | `9e68fa025ee66abbee26afb719cfbffd31f7a63dddd3afc981cac1f8a42b83b6` |

The separate matched 48-case ARC run completed. The full BF16 CPU reference,
the three retained llama.cpp CPU quant runs, and the Phlegm NPU IQ4_NL route
chose the same option in all 48 cases. All scored 44/48 by choice logit. The
tokenizer matches covered all prompt and answer-choice IDs. The BF16 scorer
took 343.53 seconds after the file verification and tokenizer check.

| Route | ARC choices matched against BF16 | Mean four-choice KL from BF16 (nats) |
| --- | ---: | ---: |
| Q8_0 CPU | 48/48 | 0.004541 |
| IQ4_NL CPU | 48/48 | 0.005780 |
| UD-Q4_K_XL CPU | 48/48 | 0.003573 |
| Phlegm NPU IQ4_NL | 48/48 | 0.004477 |

`.plans/inkling2/quant-quality/arc-test/bf16-summary.json` has SHA-256
`0c806b278737388a485e0480ec5124cc128fc2b43a1f68f812021996dea073c4`.
The 8K WikiText tape measures the separate full-vocabulary comparison.

| Capture under `.plans/inkling2/quant-quality/` | SHA-256 |
| --- | --- |
| `vulkan-bf16/input.json` | `d6ff7516108766880317ce5b5ec84372020cfe73d823bc202bd61c6370f16696` |
| `bf16-storage-validation.log` | `dd26598a710c07aa2498f62575603e41a85b6794abbdb8407741401f8bf8d0db` |

At the 3,758,096,384-byte checkpoint, both layer-0 expert tensors and the first 63 complete
experts of layer 1's gate/up tensor were available. The sample reads five eight-row spans per
physical expert and evaluates each against 16 fixed signed input vectors. This compares
23,500,800 weight elements per model. BF16, Q8_0, IQ4_NL and NVFP4 row dequantization agreed
element for element between the pinned llama.cpp DLL and the independent NumPy decoders.
The NVFP4 comparison includes each expert's F32 scale from the Unsloth GGUF.

| Expert tensor | Expert sample | Q8_0 projection error | IQ4_NL projection error | Unsloth NVFP4 projection error |
| --- | ---: | ---: | ---: | ---: |
| Layer 0 gate/up | 128 / 128 | 0.580% | 7.303% | 0.171% |
| Layer 0 down | 128 / 128 | 0.578% | 7.385% | 0.167% |
| Layer 1 gate/up | 63 / 128 | 0.586% | 7.257% | 0.166% |

Projection error is relative L2 against the sampled BF16 rows, accumulated across the tested
experts and inputs. The sampled NVFP4 weights also had the lowest relative weight L2 on all
three tensors. This makes NVFP4 worth evaluating for the shipping quant once the engine can
apply its per-expert scales before SITU and after the down projection. The samples cover two
physical layers and synthetic inputs. Full BF16 WikiText, ARC and per-visit results remain open.
Reproduce with `.plans/inkling2/quant-quality/bf16-expert-sample.py` and the retained BF16 part.
The JSON result includes row offsets and sampled wire hashes. Its SHA-256 is
`638999bd94c7e1afec633727a0666607cb5bfb8d6157a7c57624491ab7a9d75f`.

A separate CPU run uses the pinned BF16 GGUF to score the existing five-token fixture.
The independent NumPy forward pass selects experts itself. Its tensor reader uses the
retained local prefix and fetches missing byte ranges on demand, holding at most 1.5 GiB
of raw cached ranges in RAM. It verifies each HTTP 206 range and logs its offset, length,
and SHA-256. A 64 KiB live range matched the retained local bytes. An offline comparison
matched the ordinary GGUF reader across local, range-backed, checkpoint-straddling,
and cached BF16 and F32 tensor reads. The prepare-only input is
`.plans/inkling2/bf16-sparse-reference/preflight/input.json`, SHA-256
`fcde69d1daa01140679211eaab426426df90d4c6775a2dc0915dc1865cc9ca19`.
The CPU run completed and its audit checked all 48 visits, 2,006 unique weight
ranges, and the three full-vocabulary logit arrays. BF16 and native IQ4_NL matched
the top five IDs. Full-vocabulary KL from BF16 to native IQ4_NL was `0.00448901`
nats, compared with `0.000170969` for the independent Q8_0 replay. The IQ4_NL replay
with the same inputs matched the archived replay exactly. The audit under
`.plans/inkling2/bf16-sparse-reference/five-token-bf16-20261123/audit.json`
has SHA-256 `a9462737c1402251e5bdfa6b3083543f219d7746c03abc4736d2df219e4a1fe4`.
The 8K WikiText and ARC quality sweeps still need the full reference and their own
receipts.

An independent local IQ4_NL run of the same five IDs with BF16 K/V matched all five
top IDs from the archived native CPU output. Full-vocabulary KL from NumPy to the
native output was `7.05e-8` nats and Pearson correlation was `0.999999986`.
The completion audit checks all 48 visits, the tensor bounds and repeated-range hashes,
the three terminal logit tapes and the companion IQ4_NL replay. It also measures BF16
directly against the archived native IQ4_NL output. A fixture replay reproduced the
metrics and rejected a changed logit tape and a conflicting read of the same range.
The local control receipt `.plans/inkling2/bf16-sparse-reference/native-control-check.json`
has SHA-256 `9d0376d63f0d411b54adece4bf1201c004d202dcab59eb3378763865c00aa49e`.

The November 23 evening space check found 166.23 GB free and 48.47 GB left to
download. This fits with a 32 GiB reserve even without counting compression. The
resumable publisher download was restarted after the packed-score timing pair.
The 8K Vulkan reference-capture and 48-case ARC BF16 preflights both passed again.
The completed download passed the publisher's size and SHA-256 checks, and the
8K reference tape came from the CPU fallback described above.

The native host now opens that Unsloth GGUF with all 40 NVFP4 expert tensors. It validates
the exact row size and the 128-entry F32 scale for each tensor. The block scales are decoded
with the wire format, and the separate expert scales are applied to the projection output.
On one fixed five-token prompt and four forced decode steps, it matched all five greedy IDs
from the pinned, scale-patched llama.cpp CPU build with BF16 KV. The mean and maximum
full-vocabulary KL were 0.000709 and 0.002247 nats. The last prefill row's relative L2 from
the reference ranged from 0.51% to 4.38% across all 48 chronological visits. On the existing
IQ4_NL route, the same fixture preserved all 200,058 prefill logits byte for byte and all
five output IDs. `GGUF-PACK` also passed the signed-nibble, independent group-scale and
misaligned-row cases.

The native host answered all eight retained ARC questions correctly and matched the pinned
Unsloth llama.cpp answer choice in each case. Mean four-option KL was 0.000189 nats. The
run used the exact saved prompt and answer option IDs, with one case per gold letter in each
split. `.plans/inkling2/quant-quality/nvfp4-native/arc-summary.json` has SHA-256
`185001698ffa5eee42f8e6a78b58477b1beb3ba17109b8fe05a48a3573b9316d`.

The report under `.plans/inkling2/quant-quality/nvfp4-native/summary.json` has SHA-256
`fee0815f162e02b27310d65248fd1b07e39134a9f0e4aa127169dfd108779f05`.
The current Phlegm NPU bundle still needs the NVFP4 block and decode kernels for this GGUF.
The existing 96-by-32 compact weight tile has space for 16 code bytes and two scale bytes
per output row. An offline prototype puts two raw UE4M3 scales in those two bytes and
preserves the 1,728-byte tile size. Eight real expert planes from the first and looped
layers round-tripped byte for byte, including all 17,915,904 sampled bytes. A separate
offset calculation checked the code and both scale locations at row, column and K-tile
boundaries. The core must rescale each 16-value partial independently before accumulating.
Compiler resources, on-device numerics and runtime throughput remain open for that layout.
`.plans/inkling2/quant-quality/nvfp4-native/wire.json` has SHA-256
`879ebda7265f7a0c04998536ed553a51c9276eb8c3f5975e6214ad877f934777`.

| Capture in `.plans/inkling2/reference/bf16-storage-probe/` | SHA-256 |
| --- | --- |
| `ntfs-compression.json` | `355d20b8978a4a90a6687918c647184940b18564d84d536d98ffdebb90e6d1ae` |
| `cold-artifact-compression.json` | `e8b126601659a0ef2e9eeb365cae04e91b7359c8b906e0b304cbb8189e35ee49` |
| `tape-compression.json` | `e3b5aeffd0a19cd808e04728044b3d8d54a9ce10223f2b91e6d1624c41af5181` |
| `tensor-header.json` | `72c3229fd6986e072be0033050d833402e6b8af03bba6910e027fdaeb4681604` |

| Artifact under `.plans/inkling2/quant-quality/continuations/` | SHA-256 |
| --- | --- |
| `fixtures/cases.json` | `69dd5bb60f6355789dc7afb55f6e6221f389710ea43af67eae67f01dff33199a` |
| `fixture-verification.json` | `80c74dea51e413290d8436a6e62fe224b721fd2ff3e97590a9ebed3ed6dee309` |
| `q8_0.jsonl` | `7c70cec58a4ee4e219edd3bbac4a4f10f1d91676bbaa730b9941ecf318483bf3` |
| `iq4_nl.jsonl` | `97fa97b518314848cc5ea4c24337870bcccc1452a6e307bcd2c13c95893472d5` |
| `ud_q4_k_xl.jsonl` | `d51b4152d7cd91b194295ad000084dad4b7ec8fcba59a662d636f4cc477cd17f` |
| `summary.json` | `bccdba2ce507be358600b3ac18b2218760e0e191bc6f5a0c57634a7210f56179` |
| `score-verification.json` | `3b0a62e533483c92a2b56ef1c06b5f96d3b4bef8f66315fb6f2cde5aed9a57a5` |

An external transfer check selects 24 Easy and 24 Challenge questions from the published
AI2 ARC test splits at revision `210d026faf9955653af8916fad021475a3f00453`. It takes the first
six hash-ranked eligible four-option questions per gold label in each split and preserves
published option order. Each prompt ends in `Answer:` and scores the four one-token labels.
All three weights produced identical prompt and answer token IDs. The native CPU replay covers
eight cases, one per gold label per split. Phlegm used the same 48 prompts with the IQ4_NL GGUF
and the v11 bundle. [Pinned ARC dataset](https://huggingface.co/datasets/allenai/ai2_arc/tree/210d026faf9955653af8916fad021475a3f00453).

The pinned llama.cpp CPU build gave the same selected answers for all 48 questions on all three
weights. An independent NumPy/SciPy calculation verified the source labels, selected answers
and four-option KL. These scores cover the fixed 48-case sample.

| Weights | Easy | Challenge | Total | Four-option KL from Q8_0 |
| --- | ---: | ---: | ---: | ---: |
| ggml-org Q8_0 | 23/24 | 21/24 | 44/48 | 0 |
| ggml-org IQ4_NL | 23/24 | 21/24 | 44/48 | 0.003404 nats |
| Unsloth UD-Q4_K_XL | 23/24 | 21/24 | 44/48 | 0.000638 nats |

The native CPU route and pinned llama.cpp IQ4_NL both selected the correct option in all eight
subset cases. Mean four-option KL from llama.cpp to the native host was 0.000237 nats. The
native engine returned a finite full vocabulary row for each prompt. The Phlegm NPU run scored
23/24 Easy and 21/24 Challenge and matched all 48 selected answers from the pinned llama.cpp
IQ4_NL CPU run. Mean four-option KL from CPU to NPU was 0.001037 nats. A separate calculation
checked all 48 raw vocabulary captures, gold labels, option logits, and answer choices.

| Artifact under `.plans/inkling2/quant-quality/arc-test/` | SHA-256 |
| --- | --- |
| `fixtures/cases.json` | `e574a26a55e58313310942fe978e3c7171092b511606f3cf9a6e5b7a6f578d9b` |
| `preflight.json` | `3deb18eca8f82f6e607c1e7b4b5eab57e80bf3c5e5a712dbecc68ebfb1727768` |
| `phlegm-preflight.log` | `4ca6bd634047ea9c5eca20f207f855ce12ec7c88f3e1a911a91569e7e9122e90` |
| `summary.json` | `17e2a0fe046b61a4bc3ece7c21ed5efc4f064ae6239520665e4091a8008b34e3` |
| `verified.json` | `c1c2318829d6dc7a598ec8bc4a2831582430221ce3d01d90c5cdb4dd91c428ee` |
| `native-summary.json` | `363f9ea5d1c3f3d4d2c9a5ffbed27412165d0477683befa23dee6f12be720f36` |
| `phlegm-summary.json` | `9bcc539bc18e0e9fc17bf6401a764819ebc1594ccb54960602a19535bbf2b125` |
| `phlegm-independent-check.json` | `83c3c32bd0141decd2807ce779a78027f971b3c2d574863de03fba4e548e0cef` |

### Native text and expert projection bring-up

The [native host driver](../../../src/open_qwen36/inkling2_host_check.cpp) reads the fixed-loop GGUF directly.
It runs block prefill and cached decode with 24 physical weight sets and 48 separate KV and convolution states.
Sliding layers use bounded KV rings. Each block stages updates until its last visit and requested logits succeed.
The native constructor checks the loop dimensions, signed GGUF head-count arrays, and local/global pattern.

| Gate | Result on the Ryzen AI 9 HX PRO 370 |
|---|---|
| Short native prefill and decode | On the five-token prompt above, a single block and 2+2+1 blocks produced bit-identical logits. Native vs FP32 logits correlate at 0.99998. All 48 residuals correlate at ≥ 0.99920 vs FP32. Native vs llama.cpp KL is 0.0000997 nats and all 64 greedy IDs match. Native logits `.plans/inkling2/native-short.f32` SHA-256 `ba976018ea7aa452130220da129eba264e97ec045e07cca0917d213f3b355467`. |
| Sliding and relative-position boundary | First WikiText-2 chunk plus two fixed tokens (1,026 total, native split 1,024+2). All paths returned finite logits and the same top-five IDs in order. Native vs llama.cpp correlation is 0.99520 and KL is 0.04577 nats (0.06004 with BF16 in both KV caches). The independent NumPy graph with BF16-rounded KV gives correlation 0.99909 and KL(NumPy ‖ native) 0.00301. KL(llama.cpp BF16 KV ‖ NumPy) is 0.04366. Boundary parity across implementations remains open. Prefix IDs `.plans/inkling2/window-ids.txt` SHA-256 `2e1c80ed65e3d2a3643eea98e5eb5acf79c5bea084f938f9d74f0cd260a40c8c`. NumPy logits and 48 residuals in `.plans/inkling2/numpy-window-bf16/`, logits SHA-256 `6a230db19dced8420d660783d19d0f60f9e88f7766c742013890d4e073b9e921`. |
| IQ4_NL NPU gate/up | Layer 0, expert 0, 2,304×2,304, four Strix cores. Relative L2 is 4.92e-6 against an independent FP64 dot using GGUF codes and BF16-rounded scales/activation, and 0.00229 against source GGUF scales and FP32 activation. Output `.plans/inkling2/iq4-gate-up-expanded/output.bin` SHA-256 `7179f539d0d6331f49adab179d017e471d8c2ef3e5dcb21f986ef3b0114101d7`. |
| IQ4_NL NPU down | Layer 0, expert 127, 2,304×1,152 with K padded to 1,280. Relative L2 is 4.93e-6 against BF16-rounded inputs and 0.00229 against source GGUF and FP32 input. Output `.plans/inkling2/iq4-down/output.bin` SHA-256 `25ad1bf457685d708f1a50a332ed00a7339e963bfc6806be9abff7ad0762f41b`. |

The [IQ4_NL design](../../../open_kernels/designs/inkling2/iq4_gemv.py) streams compact weight tiles from DDR,
expands each tile into an L1 buffer and uses the existing Q8 GEMV. Two direct lookup-and-MAC experiments
corrupted alternating output lanes, including 171 non-finite values on the fused gate/up fixture. Both failures
are retained under `.plans/inkling2/iq4-gate-up*`. The working 2,304-wide image uses 3,552 bytes of program text
per core and 40,128 bytes of stack plus buffers. The highest occupied L1 address is below 49,216 bytes.
The gate/up fixture produced bit-identical outputs on four and 12 cores. The XRT harness reported
2.148 ms on four cores and 1.449 ms on 12 for the individual smoke dispatches. Those timings cover
one expert projection each.

### Routed expert decode on the NPU

The [routed design](../../../open_kernels/designs/inkling2/routed_gemv.py) processes eight selected experts per
projection. The four streams share a Strix image and XRT context. Each visit runs gate/up and down, with SiTU,
routing and weighted accumulation on the host. The 48 visits share the 24 physical layers' expert buffers.
The runtime packs each expert on its first use and patches the stream's eight slots to the selected expert IDs.
The probe runs multi-token prefill on the host block path.

| Check | Result on the Ryzen AI 9 HX PRO 370 |
| --- | --- |
| Four-stream hardware fixture | Eight real expert planes per stream with distinct activations. Maximum relative L2 against independent FP64 dots using BF16-rounded scales and activations was 5.21e-6 for IQ4_NL gate/up, 6.01e-6 for IQ4_NL down, 5.07e-6 for Q8_0 gate/up and 5.71e-6 for Q8_0 down. All streams and a repeat of IQ4_NL gate/up ran on one context. |
| Resources | 12 cores, 3,808 program bytes and 40,192 stack and buffer bytes per core, with the highest buffer ending at 49,216. The four builds have identical core ELFs and hardware initialization binaries. Each stream has 96 weight fills, one activation fill and 12 output drains. |
| Dispatch timing | XRT reported 3.959 ms on the repeated IQ4_NL gate/up, 2.292 ms on IQ4_NL down, 1.800 ms on Q8_0 gate/up and 1.153 ms on Q8_0 down. The first IQ4_NL gate/up took 5.473 ms. These are single measurements from a resident-buffer smoke test. Trace `.plans/inkling2/routed-run.log` SHA-256 `938eb99170ab929d92c0d50dd1e0690c58b050d987f73ce2decb67d6862fec7c`. |
| Native text, single-token appends | On the five-token prompt, all 64 greedy IDs match llama.cpp. Prompt KL is 0.000129 nats against llama.cpp and 0.000139 nats against the native host route. The repeated run took 8.87 s for five prompt tokens and 54.80 s to decode 63 tokens (1.15 tok/s) with on-demand packing and host stages. Logits `.plans/inkling2/routed-v1-short.f32` SHA-256 `0cc0c2ce2a089782480b16636543a45efae3e3e185cdabee8060c7afc2d0c195`. Greedy IDs `.plans/inkling2/routed-v1-greedy64.txt` SHA-256 `a4d0b016c6a458e94fda02cf478e31569c86d9fac27f4695204a59bcc22c1371`. |
| Host block prefill followed by NPU decode | IDs 1–5 agree with llama.cpp. ID 6 changes from 161721 to 220. Keep this as an open cross-route quality gate. Greedy IDs `.plans/inkling2/routed-greedy6.txt` SHA-256 `87cb7aeb40f5cc61569baf92753e9a28f40d0f07521031939e7428c5bfd6b682`. |
| Sliding window boundary, 1,024+1+1 | Host block prefill plus two NPU expert decode steps on the fixed WikiText-2 prefix. All logits are finite and the top five match the native host and both llama.cpp runs. KL is 8.23e-6 nats against the native 1,024+2 route and 0.04498 nats against llama.cpp with FP16 KV, so the llama.cpp boundary discrepancy remains. Logits `.plans/inkling2/routed-window.f32` SHA-256 `649f931fff390632fa8721f79225c20c83bbfb6b0eb5b7919e2b3103de161acd`. |

Feeding the fixed five-token prompt and first five reference continuation IDs, the logit for ID 161721
leads 220 by 0.22013 with all NPU expert projections and 0.16894 with all host projections. With host
block prefill and NPU expert decode, 220 leads by 0.00573. The mixed-route logits are in
`.plans/inkling2/routed-teacher-5.f32`, SHA-256 `f571e9bba1c3de64a36bd3d7d19b834213283ba5f2a4055c4a7fc43338b27eb1`.

The pinned llama.cpp CPU graph uses Q8_0 activation blocks for Q8_0 and IQ4_NL dot products. Emulating its
32-value quantizer ahead of those NumPy matmuls increased KL against llama.cpp from 0.000133 to 0.000717 on
the five-token prompt and from 0.04366 to 0.08598 at 1,026 tokens with BF16 KV. That test reproduced the activation
rounding with FP32 accumulation. The source and long logits are `.plans/inkling2/quantized_activations.py`
SHA-256 `1ca3a5cd03a82da1cf1fec5a158afb287e54efc982bb1bc2daffd9b3b45277a8` and
`.plans/inkling2/numpy-q8-window.f32` SHA-256 `30007b3ad067d73c8ee6e996d34b1e979f0a45f7cb0c2ae4aa64ed21bdcd8ae1`.

At the isolated dispatch timings, all 48 visits would spend about 287 ms per token on the expert projections.
On two warm replays of five fixed tokens, the native probe spent 187–193 ms per token in gate/up dispatches
and 109–113 ms in down dispatches, including buffer syncs. Logits and state commit took 89 ms and the
other host stages took 340–344 ms. Total decode was 729–735 ms per token. The first replay spent 1,445 ms
per token with on-demand packing. Probe `.plans/inkling2/route_profile.cpp`, trace
`.plans/inkling2/route-stage-quiet.log` SHA-256 `94fdfc39db792a355e85592ed4bca137c3868e90f777d1e174bb40276e34cbc6`.

An isolated eight-core [Q8 logits kernel](../../../open_kernels/designs/lm_head_q8/lm_head_q8.py) on Inkling's
real `output.weight` streamed 200,064 rows (200,058 usable IDs) and ran in 10.72–10.91 ms after a 13.59 ms
first run. On 512 sampled rows, relative L2 against FP64 dots was 4.88e-6 with BF16-rounded scales and
0.00174 with source GGUF scales. Trace `.plans/inkling2/lm-head-q8/run.log`
SHA-256 `b4d43fb2d1c3d03b33a1b2e3042ae4cf5c90f1bf1d30e419075d635c38062873`.
On the same device with two contexts registered, two consecutive warm head launches took 9.95–10.17 ms,
and five head launches alternating with IQ4_NL gate/up took 10.56–10.90 ms. Trace
`.plans/inkling2/head-moe-quiet.log` SHA-256 `a219605efb74c70c4831915f73750280bbab443435dfd46a5a7abf04f1655055`.

The version-2 routed bundle runs the Q8 head after the native final norm. It packs the first 200,058 GGUF
rows once and zero-fills the final tile, then keeps the head resident in its own XRT context. The host
still applies the model's final logit softcap. In a paired five-token prompt and 64-token greedy probe,
the NPU-head and host-head variants both matched all llama.cpp IDs. The NPU-head prompt KL against the
host head was 8.41e-6 nats, and against llama.cpp it was 0.000139. With six OpenMP threads and passive
waiting, 63 decode steps took 51.94 s with the NPU head and 57.18 s with the host head. This is one
paired run with on-demand expert packing. The prompt logits are in `.plans/inkling2/head-model-short.f32`
SHA-256 `84a3713397c3ce4d9c9b23b1f4c45100aea3a537472c309661f8508c3a5ef19e`.
Trace `.plans/inkling2/head-model-64.log` SHA-256 `f37a81c77e04e14ce708ef547a551e535579bd3986c8eded928de3474e364317`.
On the reconstructed normalized last row of the independent 1,026-token NumPy reference, the NPU head
preserved the top five IDs with a KL divergence of 4.44e-6 nats after softcap. This isolates the head
at the window boundary. Logits `.plans/inkling2/head-window-logits.f32` SHA-256
`2242d12bda40cc4ae7c9ccdfae1c4bfe9cdb5b9ee668802522ac87c86723826f`.

Build `routed_gemv.py` four times using `INKLING2_ROUTED_FORMAT=iq4` or `q8` and `INKLING2_ROUTED_K=2304` or `1280`.
Use sibling output directories `routed-iq4-up`, `routed-iq4-down`, `routed-q8-up` and `routed-q8-down`.
`designs/inkling2/make_routed_test.py MODEL.gguf BUILDS` writes `routed.cfg` for the XRT harness. Run the harness
and rerun the script with `--verify`. `recipes/inkling2_routed.py BUILDS BUNDLE` writes the versioned bundle for
`inkling2_npu_check BUNDLE MODEL.gguf IDS.txt LOGITS.f32 64 1`. Add `--head HEAD_BUILD` to include the Q8 head.
Build the head with `LMHEAD_N=200064`, `LMHEAD_K=2304`, `LMHEAD_CORES=8` and `LMHEAD_SCALES_F32=1`.

### Routed expert block prefill on the NPU

The version-3 bundle adds one expert block image per weight format. Each image runs two instruction streams for
fused gate/up (K=2,304) and down (K=1,152). The block route groups tokens by routed expert within each visit,
then batches their projections. It retains the source weight codes and BF16 scales. The NPU multiplies signed
int16 activation blocks by signed int8 codes and accumulates rescaled output in FP32. The host runs attention,
the loop controller, dense projections, convolution, routing, SwiGLU and the expert mix. The single-token path
keeps its two grouped expert dispatches per visit.

On the Ryzen AI 9 HX PRO 370 with the Windows XRT stack used above, the M=32 image uses 16 active cores.
IQ4_NL and Q8_0 ELF `.text` sizes are 1,952 and 1,280 bytes per core. The larger Q8_0 route uses 19,460 bytes
per core including the 4 KiB reserved stack, 6,144-byte accumulation tile, double-buffered operands, two 64-byte
table buffers and the 4-byte K-tile count. The last table ends at `0xC040`. Each stream has 728 instruction words,
with eight weight fills, two activation fills and eight output drains. Within each format the up/down core ELFs
and initialization CDOs are byte-identical. Peano rejected an initial per-vector dynamic shift during register-bank
selection. The tested image uses BF16 products with powers of two for the high/low split. Failed build log
`.plans/inkling2/block-iq4-up.log` SHA-256 `df8921584dd594610282891d6ba3824743a0a6732eb0e49269606adff69f8259`.

| M=32 expert projection | Warm XRT dispatch (five runs) | Max per-token relative L2 against independent FP64 dots |
| --- | ---: | ---: |
| IQ4_NL gate/up | 1.403–1.428 ms | 2.60248e-6 |
| IQ4_NL down | 0.786–0.842 ms | 2.61557e-6 |
| Q8_0 gate/up | 0.811–0.933 ms | 2.59260e-6 |
| Q8_0 down | 0.491–0.557 ms | 2.56675e-6 |

| XRT trace in `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `int-block-iq4-up/run.log` | `d7f677a80d01512fb30411cd18659266f22863478e738c60b071008bc7d9958c` |
| `int-block-iq4-down/run.log` | `16d467392f7e2abedc119715dc214159d3cf09043bb21c636dca8f3f25f2121e` |
| `int-block-q8-up/run.log` | `0409b8aec0a0145945ca1ceb4c6a9f72bd823321089092a15f8c5be97e81596b` |
| `int-block-q8-down/run.log` | `c90e7a028d2f4f94b525abd89a9d696d9ed7ac8eb6c49878cc6105a3b98d4b2d` |

The five-token prompt consumed by one NPU expert block produced bit-identical logits to the version-2 singleton
route. All 64 greedy IDs matched the pinned llama.cpp reference. This reproduces short-prompt parity with block
prefill and NPU expert decode. The version-1 host-prefill mismatch at token six remains recorded above.
Short-prompt logits `.plans/inkling2/block-short.f32` SHA-256
`84a3713397c3ce4d9c9b23b1f4c45100aea3a537472c309661f8508c3a5ef19e`.

The 1,026-token WikiText-2 prefix ran in four 256-token blocks and a two-token tail. The final logits were finite
and preserved the ordered top five from the native host path and both llama.cpp KV precisions. KL to the native
host was 0.001292 nats. KL to llama.cpp was 0.03862 with FP16 KV and 0.05278 with BF16 KV. The reference
discrepancy remains open. Cold prefill took 287.85 s including on-demand packing. Logits
`.plans/inkling2/block-window.f32` SHA-256 `103cd361874678340420afa05ce17f119f626a5bee196295b305b270e5500b95`.

From the BO sizes, the prefill layout reserves 14.74 GiB and the decode layout 17.23 GiB when each physical
layer has been visited. Including the Q8 head, the combined reservation is 32.45 GiB before KV. In the short
HTTP smoke, the bridge process's WDDM `Shared Usage` reached 32.70 GiB once decode allocated its pools, and
both responses completed. R-0005 reports earlier Windows residency problems near 26 GiB, which prompted the
shared pool in version 4 below. Samples
`.plans/inkling2/phlegm-block-adapter-memory.jsonl` SHA-256
`6e434d484a8b31d2a9ca2fdd6c2bbcefacac1474a549227652ccba491cb121f7`.

Build `designs/inkling2/block_expert.py` with `INKLING2_BLOCK_FORMAT=iq4` or `q8` and
`INKLING2_BLOCK_K=2304` or `1152`. The measured builds used `INKLING2_BLOCK_ARRAY_ROWS=2`. Run
`designs/inkling2/make_block_test.py` with the model, tensor, expert and build directory, run its `run.cfg` in the
XRT harness, then rerun the fixture with `--verify`. Stage the four builds in sibling directories named
`block-iq4-up`, `block-iq4-down`, `block-q8-up`, and `block-q8-down`. Pass their parent with `--blocks` to
`recipes/inkling2_routed.py`. Keep `--head` to use the NPU logits projection.

### Shared expert weights for block prefill and decode

Version 4 holds one packed copy of each expert for both routes. The decode stream sends a different expert to
each of eight shim columns. Four cores per column consume disjoint stripes from the prefill weight layout. A
strided weight DMA interleaves the four stripes in K order. Each core runs the prefill integer dot with one active
row of a four-row tile. This retains the FP32 sums and uses the same activation and weight scales as block prefill.

On Strix Point the grouped decode image has 32 cores. Its two format builds use 1,952 (IQ4_NL) and 1,280
(Q8_0) bytes of `.text` per core. The larger Q8_0 route uses 14,468 bytes of L1 per core including a reserved
4 KiB stack, double-buffered operands and result, two LUT buffers and the K count. A shim column uses two
fills and one drain. All four streams have 1,004 instruction words. Gate/up and down have identical core ELFs
and initialization CDOs within each format.

| Eight-expert decode dispatch | Warm XRT latency range | Max relative L2 against FP64 dots | Log SHA-256 in `.plans/inkling2/compact-r4a-<case>/run.log` |
| --- | ---: | ---: | --- |
| `iq4-up` | 1.142–1.528 ms | 2.58434e-6 | `8cb20b35ef33697aadc51770f8d0b63e7b30bf7ff5f8d66deaf708b94b233445` |
| `iq4-down` | 0.618–0.708 ms | 2.58442e-6 | `4a005361fceeb66e7bd3f64a780718d59ca0a61d8fea1dc0e4a6573fcdd890b2` |
| `q8-up` | 1.003–1.081 ms | 2.59352e-6 | `7b627892332a0a5923c0f9993e022e3a01f9ad2ba81c9aa9b89e821be512c789` |
| `q8-down` | 0.511–0.686 ms | 2.60495e-6 | `1c8e5eff00b298ae91e18e00f4cb5edca7c57dfd6c063378bc1ec4ff946c27ab` |

The five-token block prefill produced bit-identical logits to version 3 and the next 64 greedy IDs matched
llama.cpp. Logits `.plans/inkling2/compact-short.f32` SHA-256
`84a3713397c3ce4d9c9b23b1f4c45100aea3a537472c309661f8508c3a5ef19e`.

On the same phlegm bridge binary and five-token prompt, each bundle ran one cold and two warm trials with 16
forced reference decode steps. Each run matched all 17 llama.cpp greedy predictions. The bridge had capacity
1,048,576, used six threads with passive OpenMP waiting, and reset sequence state between trials.

| Expert layouts | WDDM process shared usage after cold decode | Cold decode (16 steps) | Warm decode (16 steps) |
| --- | ---: | ---: | ---: |
| Version 3, separate pools | 32.70 GiB | 11.23 s | 7.91–7.96 s |
| Version 4, shared pool | 15.54 GiB | 11.57 s | 6.50–6.76 s |

Warm decode throughput increased by 20% in this comparison. Warm five-token prefill was 2.71–2.78 s across
the four warm trials. Trace `.plans/inkling2/phlegm-compact-speed.jsonl` SHA-256
`48839c973662d7fec2605befa1906d80c83e0be0af51937f991491f230e7abd6`.

The version-4 phlegm bridge also took the fixed WikiText-2 prefix through 1,024 tokens of NPU expert block
prefill followed by two singleton decode steps. It retained 15.54 GiB of WDDM shared usage after both decode
steps. The final logits were bit-identical to version 3's 256+256+256+256+2 partition and matched the
llama.cpp ordered top five. The whole-distribution discrepancy above still applies. This cold 1,024-token
prefill took 320.50 s. Trace `.plans/inkling2/phlegm-compact-window.jsonl` SHA-256
`8481373da580af6c8b25b154f04d115901645903e5ec4abe807773d0c6e093d7`.

At the same 1,026-token prefix, the pinned llama.cpp BF16-KV oracle captured the last residual row at all 48
visits and reproduced its original final logits bit-for-bit. Relative L2 against the independent NumPy BF16-KV
graph was 0.78–0.88% through physical layers 0–3, then 4.26% at layer 4 (sliding) and 8.69% at layer 5 (full).
These increases precede the first repeated layer. Per-visit comparison
`.plans/inkling2/llama-window-bf16-residual-comparison.csv` SHA-256
`ee5c1c52747c074e862c30a2477952cbb0f0d6a2c0c4f915c5b1d0927c0a95dd`.

The follow-up capture measured 0.599% error after layer 4 attention and 24.90% after the MoE post-convolution.
llama.cpp chose expert 23 in slot eight, while NumPy chose 11. Their rank-eight/nine gate margins were
0.000318 and 0.000031 respectively. Forcing llama.cpp's expert IDs for the last row in NumPy brought the
layer 4 MoE error to 1.36% and the residual error from 4.26% to 0.65%. At layer 5, forcing the last-row IDs
reduced the MoE error from 29.53% to 4.47% and the residual error from 8.69% to 3.50%. These interventions
hold the incoming NumPy residuals fixed and show that routing explains most of each jump on this prefix.
The captures `.plans/inkling2/llama-window-stage-comparison.csv` and
`.plans/inkling2/llama-window-stage-counterfactual.csv` have SHA-256
`a5a5c154a06d69859bf7df2a2dc21fcf8b21d728df1002eff2d04dff0a71eb24` and
`3dfde8579c04d52c6459a528e57a246ac5cbb754b545948a4d88fe1db3aa1bcc`.

Build `designs/inkling2/compact_decode.py` with `INKLING2_COMPACT_FORMAT=iq4` or `q8` and
`INKLING2_COMPACT_K=2304` or `1152`. The measured builds use `INKLING2_COMPACT_ROWS=4`.
Use `designs/inkling2/make_compact_test.py MODEL TENSOR BUILD`, the generated `run.cfg`, and `--verify`.
Name the four builds `compact-iq4-up`, `compact-iq4-down`, `compact-q8-up` and `compact-q8-down`.
Pass their parent to `recipes/inkling2_routed.py BUILDS BUNDLE --compact --blocks BLOCK_BUILDS --head HEAD_BUILD`.

### Relative-position attention on the NPU

The optional version-5 bundle runs attention in 16-query tiles for both block prefill and decode. Eight columns
each stream one KV head to three query-head workers. The core forms the 128-wide QK dot, adds the learned
relative score from a 16-channel bias strip, and applies causal and sliding masks before its online softmax.
It seeds the softmax with each query's zero-value sink. The query, relative bias and probability use BF16
high and low parts with FP32 sums. The host prepares Q/K/V and the log-position scale, stages at most 4,096
keys per dispatch, and merges the resulting maxima, denominators and weighted values across key ranges.
This caps the staging BO at 20 MiB while the per-visit KV caches stay in host memory.

The 24-core image uses 3,868 bytes of program and 52,484 bytes of local memory per core including the 8 KiB
reserved stack. Each shim column has two fills and one drain. Six stream tiers (1, 4, 16, 32, 64 and 128 chunks
of 32 keys) each have 956 instruction words and share byte-identical core ELFs and initialization CDOs.

All three fixtures use the real learned bias table from physical layer 4 or 5 and controlled query and KV rows.
The dense FP64 reference computes the full score matrix and one softmax including the sink.

| Case | Real keys | Key ranges | Max relative L2 across query heads and rows | Max relative mass error | XRT dispatch |
| --- | ---: | ---: | ---: | ---: | ---: |
| Causal | 21 | 1 | 7.41e-5 | 1.21e-5 | 2.729 ms |
| Sliding-window boundary | 1,036 | 1 | 3.77e-5 | 3.69e-5 | 4.167 ms |
| Full, crossing 8,192 | 8,194 | 3 | 4.70e-5 | 4.95e-5 | 7.120 + 4.726 + 0.852 ms |

Traces `.plans/inkling2/attn-rp-b-{1,64,128}/run.log` have SHA-256
`7510ad141a499d6a840de676cd8f136ec0cf5e938653d502308082b1a401e7a2`,
`fa74b54f48c9ae84806c714ba192f68f1663090dffee1b31c065ca13b942fdb1`, and
`4ea496882900f6efa9444f41311274c23bd4321dc967d6ecb4afbfbfc332c3cb` respectively.

The native attention entry point also ran at absolute query position 1,048,575. The full-cache probe used
all 1,048,576 generated BF16 keys and values across 256 key ranges. An independent FP64 calculation counted
the exact multiplicities of the periodic test keys and evaluated the last 1,024 learned distances individually.
The worst relative L2 was 1.66e-5 in 1.805 s including host staging and merge. A 1,024-key wrapped sliding
cache at the same position gave 2.05e-5 in 2.132 ms. These are single attention operations using the real GGUF
bias tables. Trace `.plans/inkling2/attention-max-context/result.log` SHA-256
`fae0d0f8d4304df0ff9b111202edf04110ae28a0d67a3d49a3868661b9b64420`.

On the original five-token prompt, the version-5 logits have KL 6.76e-5 nats against version 4 and 1.80e-4
against llama.cpp. Greedy IDs 1–58 agree and ID 59 changes from 152416 to 220. Keep version 5 opt-in while
this parity gate is open. Logits `.plans/inkling2/attention-short.f32` SHA-256
`e11e44fe79ae82a5785edfbdc473d1eba9402eac96136f781974b65d1ccc6503`.

On the same forced 58-token continuation, token 152416 leads token 220 by 0.00527 logits with host attention.
The NPU route gives 220 a 0.02966 lead. KL between routes is 4.53e-4 nats at this decision. In matched
phlegm bridge trials, short warm decode with NPU attention took 11.24–11.35 s for 16 steps versus 6.97–7.04 s
with host attention. Both routes matched all 17 predictions in those shorter trials. WDDM shared usage after
the cold trials was 15.62 GiB with NPU attention and 15.54 GiB with host attention. Trace
`.plans/inkling2/phlegm-attention-speed.jsonl` SHA-256
`77dcfe8d5ba5e95b8ed5df5f77b98e30c59dec428fa605f3244e299d9985fe3c`.

Alternating the attention and compact IQ4_NL gate/up contexts in the XRT harness raised median attention
dispatch from 0.192 to 2.107 ms and median gate/up from 1.054 to 3.866 ms. The combined increase was 4.73 ms
per pair, enough to explain much of the short decode regression when repeated across 48 visits. This isolated
experiment measures the context change for these two images. Trace `.plans/inkling2/attention-switch-profile/run.log`
SHA-256 `3c3fa661e3f2f4a326c693a90c19221450259dbb4fb688373dc232ed8c832e2f`.

The version-5 phlegm bridge prefilled the fixed 1,024-token WikiText-2 prefix in 213.40 s and decoded two
more tokens with 15.62 GiB of WDDM shared usage. The final top five agreed in order with version 4 and both
llama.cpp KV precisions. KL was 0.002127 nats against version 4 and 0.03238/0.04477 against llama.cpp with
FP16/BF16 KV. The 320.50 s version-4 prefill came from a separate run. Trace
`.plans/inkling2/phlegm-attention-window.jsonl` SHA-256
`3bd47df2e9da199a83c18fa5e4723700ac5c5b588787016370241596706c7f9a`.

Build `designs/inkling2/relative_attention.py` with `INKLING2_ATTN_CHUNKS` set for each tier. Name the builds
`attention-<chunks>`, including `attention-1`. Add `--attention ATTENTION_BUILDS` to the compact bundle command.
Use `designs/inkling2/make_attention_test.py MODEL BUILD --chunks CHUNKS --case short|sliding|long`, then run
its `run.cfg` in the XRT harness and rerun the test with `--verify`.

### Shared attention and expert decode contexts

Version 6 uses one image for each expert format. When a layer's gate/up tensor is IQ4_NL or Q8_0, its
attention and singleton expert projections use that format's XRT context. The two operations share a broadcast
input, a split input and a drain on every column. The first split record orders the worker's mode read after the
runtime parameter writes. Attention uses the split records for query heads and the broadcast records for KV.
Expert decode uses the split records for weight stripes and the broadcast records for groups of four activation
tiles. Block experts and the logits head use dedicated images.

All 16 streams were rebuilt. The two expert streams and six attention tiers have byte-identical core ELFs and
initialization CDOs within each format. The shared images contain 32 workers and use at most 9,276 (IQ4_NL)
or 8,412 (Q8_0) program bytes per core. The largest local-memory footprint is 58,696 B including the 8 KiB
stack. Each shim column has two fills and one drain. Each stream is 1,196 words.

The same-context XRT fixtures alternated attention, gate/up and down. All attention tiers, including the
sliding and 8,194-key fixtures, and both expert projections matched the original separate images bit-for-bit.

| Format | Attention, 21 keys | Gate/up | Down |
| --- | ---: | ---: | ---: |
| IQ4_NL | 0.141 ms | 1.094 ms | 0.602 ms |
| Q8_0 | 0.136 ms | 0.933 ms | 0.515 ms |

These are medians of seven warm triplets per format. The IQ4_NL attention plus gate/up pair took 1.235 ms
in the shared-image test. The earlier separate-image pair took 5.973 ms. Traces
`.plans/inkling2/shared-b-iq4-matrix/run.log` and `.plans/inkling2/shared-c-q8-matrix/run.log` have SHA-256
`4de17c75f048fee71afbd032808d6d85502d24ef7bec136d889e8c2091323d71` and
`933b1c36573623f428bdcb659f73621162a9ce42de1364b4c258b6fa1736b937`.

On the same rebuilt phlegm bridge and five-token prefix, warm 16-step decode took 10.97–11.15 s with
version 5 and 7.41–7.46 s with version 6. The shared context reduced mean elapsed time by 33%. Cold decode
took 16.53 s and 12.26 s respectively, with shared adapter usage of 15.62 and 15.56 GiB. Both repeated runs
matched all 17 reference predictions in their short trial. The 64-token native test reproduced version 5's
predictions and bit-identical prefill logits. The forced-prefix logits at token 59 also reproduced version 5
bit-for-bit, including the known ordering difference against llama.cpp. Timing trace
`.plans/inkling2/phlegm-shared-speed.jsonl` SHA-256
`bed57d2aa534c71a2481c46fae1232605fb17446879722efaff8643984dde795`.

The shared image passed the maximum-position single-operation probe again. The Q8_0 context streamed all
1,048,576 keys in 1.964 s and gave relative L2 1.66e-5. The IQ4_NL context ran the wrapped 1,024-key window
at the same position and gave 2.05e-5. Trace `.plans/inkling2/attention-max-context/shared-result.log`
SHA-256 `b55d5326d34826f4444fdda2ca92799032a2760682af166ad826792a482ed913`.

On the fixed WikiText-2 boundary, the version-6 phlegm bridge prefilled 1,024 tokens in 227.22 s and
decoded two more with 15.56 GiB of shared adapter usage. The logits at both positions were bit-identical
to version 5. Trace `.plans/inkling2/phlegm-shared-window.jsonl` SHA-256
`35043b57bac32f1de6a36236506f40262cb38941b15eee5b5dfb6e9ab2e52df6`.

Build `designs/inkling2/shared_decode.py` with `INKLING2_SHARED_FORMAT` set to `iq4` or `q8`. Set
`INKLING2_SHARED_K=2304` for `-up` or `1152` for `-down`, with `INKLING2_SHARED_CHUNKS=0`.
For `-attention-<chunks>`, set `INKLING2_SHARED_K=0` and set the chunk count. Put these `shared-<format>-*`
build directories under `BUILDS`. Pass `--compact --shared-attention --blocks BLOCK_BUILDS --head HEAD_BUILD`
to `recipes/inkling2_routed.py BUILDS BUNDLE`. The attention fixture takes `--shared-format iq4|q8`.

### Grouped expert prefill

The isolated IQ4_NL gate/up experiment reused the compact expert layout and ran eight independent expert
planes with 16 rows each in one dispatch. It compared the same work against eight launches of the current
32-row block image with 16 valid rows per launch. The warmed median was 2.741 ms grouped and 8.642 ms
summed over the separate launches, a 3.15× gain on that projection. Outputs matched bit-for-bit, and the
worst per-row relative L2 against independent FP64 dots was 2.62e-6.

The 32-core prototype uses at most 2,032 bytes of program and 22,532 bytes of L1 including stack. Each
column uses two fills and one drain. The experiment covers IQ4_NL gate/up. Trace
`.plans/inkling2/grouped-prefill-iq4-up-m16/run.log`
SHA-256 `b66d55a1a68b37e5af3831e10e378c266e76cd3776bb97b2922491978260c074`.

The version-6 native stage profile used the first 128 tokens of the fixed WikiText-2 prefix. The warm
prefill took 26.90 s, including 3,961 gate/up and 3,961 down calls totaling 11.08 s. Relative attention
took 0.48 s, the head took 0.015 s, and the remaining host work took 15.33 s. Stage timings include host
packing, dispatch and readback. Eight warm decode steps took 4.02 s, including 2.20 s outside those timed
stages. Grouping expert work targets the measured dispatch count. Dense projections, control and convolution
still need separate host timings. Trace `.plans/inkling2/shared-stage-profile/result.log` SHA-256
`e5a7e8a136fa9bfcf1a81776c4a8f6b4d6a348d29cb09a63afda76976dd2f4b3`.

Version 7 groups up to eight expert slices with up to 16 token rows each. It uses the resident compact
weight pool and visits experts in ascending order so each token keeps its FP32 accumulation order.
The runtime pads short slices with zero activations and unpacks only their valid rows for SiTU and
accumulation. One expert can occupy several slices. An unused slot reuses the first expert's weight
address. Decode and relative attention use the version-6 images. Repackaging version 6 with the current
recipe reproduced every artifact byte-for-byte.

The up and down instruction streams within each quant format share an image and its hardware
initialization. Per-core program text is 2,032 bytes for IQ4_NL and 1,360 for Q8_0. The sum of local
buffers and stack is 22,532 and 25,604 bytes respectively, including both lookup tables and the runtime
parameter. Each direct stream has eight fills for weights, eight fills for activations and eight
drains. Independent FP64 dot products on 52 active rows per stream gave these worst relative L2 errors.
The other 76 rows per stream returned zero, including the unused expert slot.

| Format | Gate/up | Down |
| --- | ---: | ---: |
| IQ4_NL | 2.588e-6 | 2.579e-6 |
| Q8_0 | 2.620e-6 | 2.660e-6 |

An instrumented native CPU prefill captured the expert counts at all 48 visits for the first
256 WikiText-2 tokens on the pinned IQ4_NL GGUF. The following census replays those counts in
expert order. It charges for all eight slots in each launch, including empty slots, as the
current instructions do. Each expert projection has one weight read per slot. Activation DMA
includes six passes over the same packed activations. The counts describe the traced CPU router
decisions and compiled transfer layouts. NPU timing and router decisions remain open.

| Rows per slot | Up and down dispatches | Weights | Activation DMA | Output | Fixed matrix rows per projection sequence |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 2,216 | 42.66 GB | 7.35 GB | 2.61 GB | 141,824 |
| 24 | 1,760 | 33.92 GB | 8.76 GB | 3.11 GB | 168,960 |
| 32 | 1,532 | 29.55 GB | 10.17 GB | 3.61 GB | 196,096 |

The staged masked 32-row image keeps the same DMA layout and skips completely padded four-row
groups. Its packer reserves a zero scale for padding. A real zero activation row still has a
nonzero BF16 scale. On the first K tile the worker writes zeros for skipped output rows, including
entire empty slots. The route census counts 105,600 computed rows for the masked 32-row layout,
against 141,824 for the current fixed 16-row layout. The output still transfers the full tile.

All twelve staged streams compiled for 24, 32 and masked 32 rows across both weight formats and
projection legs. Up and down share identical core ELFs and hardware initialization within each
format and width. The masked image uses at most 2,432 B of core text, a 1,024 B kernel frame plus
64 B entry frame on a 4,096 B stack, and placed local buffers end at 49,216 B. Each column uses
three shim descriptors and up to 36 memtile descriptors. The staged native packer matched 78
independent NumPy wires byte for byte, including legacy 16-row packing, empty slots and partial
four-row groups. The Phlegm bridge built and passed its handshake and provenance check. On the
NPU, all 96 sparse, medium and skewed operator comparisons matched bit for bit across the
16-row, 24-row, 32-row and masked 32-row layouts. The largest relative L2 against FP64 was
2.70e-6. The bracketed 16-row controls moved by at least 27.8%, so these runs establish numerical
parity. The follow-up Phlegm model pair exited after preflight. The wrapper retained its first
stderr line. A rerun with separate streams exited successfully and matched all nine full logit
rows at each of 128, 256 and 1,023 prompt tokens. The Windows event window had no PCI context
errors or sleep/wake records during the comparison.

| Prompt tokens | Two warm runs, 16 rows | Two warm runs, masked 32 rows | Paired prefill ratio |
| ---: | ---: | ---: | ---: |
| 128 | 10.436 / 10.052 s | 11.025 / 12.411 s | 0.878× |
| 256 | 17.772 / 17.312 s | 17.391 / 15.483 s | 1.070× |
| 1,023 | 75.959 / 83.910 s | 95.293 / 91.156 s | 0.859× |

Part of the 1,023-token comparison overlapped a separate BelowNormal host compile, and the
16-row control times drifted about 10%. Retain the 16-row bundle given the mixed timing results.

| Capture in `.plans/inkling2/grouped-expert-width/` | SHA-256 |
| --- | --- |
| `routes-256.csv` | `576a244f4cd2e84bc7aeb8a8d0995d46599e652fbc7f0b0af4a73021fe54919e` |
| `census.json` | `b0fbf66f1bd35869e86f657a954ec0c2b521c0b04f9a1532ee6ca25ac5c1adf0` |
| `resources.json` | `25933ecbac5678f6b610b00832fe683e37690e8ebae179a9ff755f8e40856999` |
| `fixture/plan.json` | `b3a731f5c889635da29174d0d9bf3027f2b9508b17ff338d553d899744042fdf` |
| `verify-wire.log` | `27f6f948c0ed0349d83fab63414f75768a78a73def2588b7272c4fe46d29cb07` |
| `board-verify.log` | `0158bfc39cd9b853684e03f60159248159090a1fc96028d5e42c528a3f7a5dbe` |
| `phlegm-preflight.log` | `25435870ea8102ac5ad465c9001c390744b9424511b4e7a0bd106b0a4822a57e` |
| `model-replay.stdout.log` | `129a489f99ce0e8a43758282d72c3eb13988175206cb1b376cfd58f2aedcc2c5` |
| `pci-model-replay.json` | `600a8c4f1ae344f0966daa9c3f9d5dde135c389881ee2ef243ab377b4cabf3dd` |

A masked 48-row group retains the current four-row compute mask and the ascending expert
reduction order. The extension compiled for both projection legs in IQ4_NL and Q8_0. Each
quant's up and down streams share 32 core ELFs and hardware initialization. The maximum
linked text is 2,432 B, allocated data ends at 55,748 B, and the largest kernel and entry
frames are 1,024 B and 64 B within the 4 KiB reservation. The audit verified three shim
descriptors per column and up to 36 memtile descriptors. The native pack matched NumPy
byte for byte at all 49 valid row counts for both projection widths, including real zero
rows and masked padding.

| Router input | Expert rows per slot | Up and down launches | Weight, activation and output DMA |
| --- | ---: | ---: | ---: |
| Captured 256 | 16 | 2,216 | 52.63 GB |
| Captured 256 | 32 | 1,532 | 43.33 GB |
| Captured 256 | 48 | 1,330 | 43.59 GB |
| Repeated 512 | 16 | 3,674 | 87.22 GB |
| Repeated 512 | 32 | 2,216 | 62.60 GB |
| Repeated 512 | 48 | 1,760 | 57.67 GB |

The 512-token estimate doubles the captured 256-token expert counts. The same-build
Phlegm comparison held block size at 512, used identical warm-up and prompt IDs, and
changed only the grouped-expert manifest settings, images, and instructions. It matched
all final logits, with 1,854 launches for M48 and 3,788 for M16. M48 took 20.226 and
19.709 s against 19.827 and 19.371 s for M16, and its MoE stage also took longer in
both orders. One M48 arm recorded a power event. The experiment establishes parity
and launch savings. Keep M16 for this workload pending a measured wall time gain.
`ordinary-device-20261123/expert-m48/model-512/pair-result.json` has SHA-256
`a1084e2aec5234508c40a25a3756e6d778b561a9e8dea34a5be9d538e103e7ac`.

| Capture under `.plans/inkling2/expert-m48/` | SHA-256 |
| --- | --- |
| `wire.json` | `ffd3d013cddc544c3ccd3f7e589aae65a199c3ae7c2e16e51365c1decda732e0` |
| `resources.json` | `4dbe42ac41039e477156af2d9a2e33f9e16e39ee60fc488dcdbdc99ece8f93f4` |
| `wire-check.log` | `9af4953d2cd8242c13e7038bbacb808585ea100b0c350d6b6084757f96ccece4` |
| `fixture/plan.json` | `16d849c723211f2fd2c603f0d2e4199c5884588288298fdbfc13309a72ac8973` |
| `bridge-block-512/phlegm-reference-bench.exe` | `5dc110347d9261bf46d3f8d5eeb043235eab26d45ad94e96cdf166e1e189b344` |

Build `designs/inkling2/compact_decode.py` with `INKLING2_COMPACT_TOKENS=16`,
`INKLING2_COMPACT_ROWS=4`, `INKLING2_COMPACT_FORMAT=iq4|q8` and `INKLING2_COMPACT_K=2304|1152`
for gate/up and down respectively. Put the four builds under `PREFILL_BUILDS` as
`prefill-{iq4,q8}-{up,down}`. Pass `--compact --shared-attention --grouped-blocks PREFILL_BUILDS --head HEAD_BUILD`
to `recipes/inkling2_routed.py BUILDS BUNDLE`. Use `make_compact_test.py --token-rows 16` for the
independent projection check.

The native version-7 bundle matched the version-6 five-token prefill logits and all 64 greedy predictions,
including the known difference from llama.cpp at token 59. Version 7 decoded 63 tokens in 36.32 s. The
native capture `.plans/inkling2/grouped-short.f32` has SHA-256
`e11e44fe79ae82a5785edfbdc473d1eba9402eac96136f781974b65d1ccc6503`.

The rebuilt phlegm bridge ran both bundles on the same 128 WikiText-2 tokens in separate resident processes,
using six host threads and passive OpenMP waiting. The warm trials retained the packed weights and reset
sequence state. All twelve trials gave bit-identical logits. The mean warm time rose by 27% with
grouping when version 7 ran second and fell by 22% when it ran first. These trials do not establish a
stable 128-token performance change.

| 128-token prefill by run order | Cold | Warm 1 | Warm 2 |
| --- | ---: | ---: | ---: |
| First comparison, version 6 | 37.96 s | 21.08 s | 21.77 s |
| First comparison, version 7 | 44.60 s | 27.13 s | 27.30 s |
| Reversed order, version 7 | 32.33 s | 17.86 s | 18.17 s |
| Reversed order, version 6 | 39.31 s | 23.08 s | 22.93 s |

The version-7 bridge prefilled the full fixed 1,024-token prefix in 187.27 s, 18% below the earlier
227.22 s version-6 run. It used 15.56 GiB of shared adapter memory. Both following singleton steps
gave bit-identical logits to version 6, with 529 and 517 ms elapsed. Version 7 remains opt-in while
the crossover and variance are measured across prompt lengths.

| Capture in `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `phlegm-grouped-128.jsonl` | `3fe4b31bf1244e0789455adb9f3709b5a864ad41a654e202f9493369cd6b6d22` |
| `phlegm-grouped-128-reverse.jsonl` | `38b68e0148a5bf3bf59fd77c60ab46027b34a66320e9a0b40ccdc3670fee5f77` |
| `phlegm-grouped-window.jsonl` | `3674e398eb74c1734f4c0ad7da93792d57d8e29dd584acb82edae356ee9f765f` |

A separate native stage profile ran version 7 first and version 6 second. The same warm 128 tokens
took 18.10 s and 23.04 s respectively. Grouping cut the gate/up and down projection time from 9.44 to
4.93 s and the dispatch count per projection from 3,961 to 713.

| Version-7 host projections | Warm prefill, 128 tokens | Warm decode, 8 steps |
| --- | ---: | ---: |
| Q, K and V | 3.634 s | 0.847 s |
| Attention gate | 2.175 s | 0.452 s |
| Attention output | 2.292 s | 0.371 s |
| Relative query and sinks | 1.232 s | 0.175 s |
| Loop controllers | 1.102 s | 0.159 s |
| Expert router | 0.403 s | 0.029 s |
| Dynamic K/V tap projections | 0.044 s | 0.005 s |
| All host projections | 10.882 s | 2.038 s |
| Other host work | 1.756 s | 0.138 s |

The stage trace uses instrumented native code with the same GGUF and six host threads. Each projector's
stage totals include CPU operand packing and output unpacking. The trace files
`.plans/inkling2/grouped-stage-profile/grouped-bundle.log` and
`.plans/inkling2/grouped-stage-profile/shared-bundle.log` have SHA-256
`0941aeab267a539ad300a5dd18f98ef193cafa0c102e8f58ad48ecb5d8e4d663` and
`d32e5659cb9526986f812655ef12c031dc01df489909071eb53ea409b02d8322`.

### Dense attention projection probe

The largest measured host projections are Q/K/V, the attention gate and the attention output. All
use Q8_0 on the pinned GGUF. The gate, relative query and sinks share Q's normalized input on every
visit. K and V share it on sliding visits without loop controller messages. The isolated NPU probe
packed all six layer-0 tensors into one 2,304-input by
9,216-output stream, with 8,600 valid outputs. A second stream projected the 3,072-input by
2,304-output attention output. For visits with separate inputs, the Q group used 6,912 outputs and K
and V each used a 1,536-output stream. All ran 64 rows per dispatch and share the same 32-core image
and initialization. Core program text is 1,280 bytes and local buffers plus stack occupy 19,460 bytes.

| Layer-0 projection | Warm device median | Worst relative L2 vs packed FP64 | Worst relative L2 vs source FP64 |
| --- | ---: | ---: | ---: |
| Fused sliding input, six tensors | 1.884 ms | 4.102e-6 | 0.003902 |
| Q, gate, relative query and sinks | 1.431 ms | 4.102e-6 | 0.003902 |
| Separate K | 0.616 ms | 2.650e-6 | 0.002533 |
| Separate V | 0.432 ms | 2.684e-6 | 0.002559 |
| Attention output | 0.673 ms | 2.588e-6 | 0.002510 |

Each device median covers seven launches with resident operands. The packed FP64 reference applies the
kernel's BF16-rounded scales and activation quantization. The source reference uses the GGUF scales
and FP32 activations directly. The padded outputs were exactly zero. Engine dispatch, context switches
and model parity for this route remain open.

| Capture in `.plans/inkling2/dense-q8-probe/` | SHA-256 |
| --- | --- |
| `fused-sliding-m64/run.log` | `78ee1e706b08fb2167b0f1e304635df4c4d1afc0e558eb54ff0034589a9af80e` |
| `query-group-m64/run.log` | `4354a4fac61690abca976cf8999b5ccce89ce8645c53365ee23bcd2af2fd147f` |
| `key-m64-case/run.log` | `d15cf42e2c63d093932cf9568921f067c229a1b4ad1c0415b28e608be1f24fee` |
| `value-m64-case/run.log` | `5e99d48d583ccf80aa73653d266b5a2dc383c7caad29980863f8fd0cb2dbda6f` |
| `attention-out-m64/run.log` | `80f7d974b58ff5fa0e101b1e157ff57d36dc844d0bf2da0e04235247793f758a` |

### Dense Q8 block prefill

The first engine integration tried BF16-rounded scales and activations for the dense projections in
both prefill and decode. It cut the recorded 1,024-token prefill to 89.61 s, but its short greedy run
diverged from llama.cpp at token 6 and took 95.94 s for 63 singleton steps. The same singleton workload
took 36.32 s with version 7. The 64-row dense image is suitable for blocks. Its singleton route was rejected.

Version 8 keeps each original Q8 scale as the exact sum of two BF16 values and quantizes FP32 activations
directly to int16. It clamps boundary values before the int16 cast. The shared dense image runs Q/K/V,
relative query, sinks, gate and attention output on multi-token blocks. The existing singleton route
handles decode. Distinct controller inputs use the separate Q group and K/V streams. Query clipping,
dynamic taps, convolutions, normalization and gating remain on the host.

All four streams share the 32-core image and initialization. Core text is 1,312 bytes, and buffers and
stack total 19,844 bytes per core. An added scale plane raises packed Q8 weight traffic by 5.9%. The
largest packed FP64 error was 4.75e-6. The largest error against FP64 using the unrounded GGUF scales
and FP32 activations was 5.06e-5, down from 0.00390 with one BF16 scale. The padding returned zeros.

| Projection, 64 rows | Warm device median |
| --- | ---: |
| Fused sliding input | 1.987 ms |
| Q with relative query, sinks and gate | 1.480 ms |
| Separate K | 0.406 ms |
| Separate V | 0.426 ms |
| Attention output | 0.737 ms |

The five-token native prefill and 63 singleton steps produced all 64 llama.cpp greedy IDs. Initial
KL from llama.cpp was 0.000191 nats and the first five ranks agreed. The previously observed token-59
difference remains reproducible on version 7. Version 8's short result covers one prompt. Native
singleton time was 39.00 s for the 63 steps.

On the fixed 1,024+1+1 WikiText prefix, version 8 prefilled in 78.71 s against the earlier version-7
187.27 s. The 1,025 and 1,026 singleton steps took 526 and 520 ms. Shared adapter memory was 16.62 GiB,
up 1.06 GiB from the version-7 boundary run. At 1,026, KL from llama.cpp with FP16 KV was 0.02637 nats
against 0.03238 for version 7. The top-three ranks matched, and ranks four and five were exchanged.
This boundary check leaves corpus-level quality and the long-context ladder open.

The rebuilt phlegm bridge also ran the same first 128 WikiText tokens in both bundle orders with six
host threads. The second trial on each process kept its packed weights. Version 8 was roughly twice
as fast in both orders. The version-7 logits retained their prior hash, and the version-8 logits were
bit-identical across repeats. Top five ranks agreed between versions, with KL from version 7 of
0.00253 nats. Eight forced singleton steps following each warm prefill used the existing decode route.

| Bundle order | V7 warm prefill | V8 warm prefill | V7 decode, 8 steps | V8 decode, 8 steps |
| --- | ---: | ---: | ---: | ---: |
| V7 first | 16.40 s | 8.06 s | 3.60 s | 3.53 s |
| V8 first | 17.56 s | 8.26 s | 4.07 s | 3.60 s |

The instrumented native version-8 run took 11.12 s for a warm 128-token prefill. Timed NPU
projections included host packing and readback. Grouped expert projections took 4.67 s, dense attention
projections 1.18 s, relative attention 0.47 s and logits 0.012 s. The remaining host work totaled
4.79 s, of which 1.73 s was controller projections, 0.59 s the router, 0.052 s dynamic K/V tap
projections and 2.42 s other host work. The eight warm singleton steps took 5.37 s in the native trace,
including 3.47 s of host projections. Native stage timings and phlegm bridge wall times use different
executables.

Build `designs/inkling2/block_expert.py` with `INKLING2_BLOCK_FORMAT=q8`,
`INKLING2_BLOCK_ARRAY_ROWS=4` and `INKLING2_BLOCK_SPLIT_SCALE=1`. Name the builds
`dense-fused` for K/N 2,304/9,216, `dense-query` for 2,304/6,912, `dense-kv` for 2,304/1,536 and
`dense-output` for 3,072/2,304. Add `--dense DENSE_BUILDS` to the version-7 packaging command.
Version 8 declares `dense_q8_format: split-scale` and verifies all DMA spans at load.

| Capture in `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `dense-short.log` | `309afec554b016eed4dd52ce053a6ef1e7062a93eef755c6dfa0d4131293cf88` |
| `phlegm-dense-window.jsonl` | `e1c04b7af28927f4511e7a45f89882a876a1d13fff217929d8dd25e62472a37f` |
| `dense-precise-short-greedy.txt` | `a4d0b016c6a458e94fda02cf478e31569c86d9fac27f4695204a59bcc22c1371` |
| `phlegm-dense-precise-window.jsonl` | `34e2626ce9ac2428b40871a58092250cba7acfe7929dd6521a5bea821e4e4c87` |
| `phlegm-dense-precise-paired.jsonl` | `e62c656843d978ed30f751cae00fdee3abf9f71959c39c419ce9e4328fc3004a` |
| `dense-precise-builds/dense-fused/run.log` | `5b91df4bc2d4b434dba48b2cbcd49abd38def03a750fb534ee0ddad51bd64bae` |
| `dense-stage-profile/result.log` | `9b4a5d20b8c701ae54c7b98393b3d747d598a434529d8d1411408932070ac9a5` |

An eight-core, four-row split-scale dense image also ran eight forced singleton steps after the fixed
five-token prompt. The warm test took 8.87 s against 8.07 s with host projections on the same instrumented
native driver. The new dense dispatches took 2.78 s, and the relative-attention and expert dispatches
also took longer. The smaller standalone context did not improve singleton time in this run. It stays
outside the bundle. Traces under `.plans/inkling2/dense-m4-fixed-probe/` are `control.log` SHA-256
`ff3ff26fff35848f60cbaf7568e2c58cbd8f0c6c0c56810616368e60dd738ec5` and `m4.log` SHA-256
`7617dc5f4ff3691a568307c92e13c76d625dc475e1acc0ac972092ecbe1a5f80`.

An isolated shared-image design puts singleton dense Q8 projections alongside relative attention and
routed experts. A 3,456-byte weight frame holds one split-scale Q8 tile or two original IQ4_NL tiles.
Packing adjacent IQ4_NL tiles into the frame preserves their original byte count and accumulation order.
The worker selects the projection or attention stream and the projection band count at launch.

The fused input stream can use separate rows of the four-row tile for the Q, K and V inputs of looped
layers. One launch computes the six input projections. The host would select the relevant output columns
from each activation row. The offline fixture uses the six real layer-6 tensors, reconstructs every Q8 code
and FP16 scale from the packed bytes, and retains independent FP64 results for one- and three-input cases.
The separate output stream packs the real 3,072-by-2,304 projection.

All 13 dense, expert and attention streams have the same 32 core ELFs and hardware initialization.
Maximum linked program text is 11,712 B. Linked data ends at 61,724 B, and the fixed IQ4 lookup table
at the top of L1 ends at 65,536 B. Each worker reserves 8 KiB of stack. Every stream uses three
single-descriptor shim transfers per column. The native host packer matched the independent weights
and activations byte for byte for both fused input cases and the output projection. A framed-output
fixture verified all 8,600 input-projection outputs and 2,304 output-projection values against the
selected activation rows. The host loader checked offsets and transfer spans for each compiled stream.

The experimental bundle carries the shared singleton image alongside the v11 artifacts. The staged
Phlegm bridge and native driver both compiled. The bridge passed handshake and provenance, and the
paired 512-token prompt with 32 forced decode steps passed offline preparation. The mixed-launch
fixture will check dense output and attention against FP64 and IQ4_NL against its packed-weight
reference. Device execution, numerical quality and throughput remain unmeasured.

The same experiment now builds against the current host projections and NPU runtime. A queued
comparison holds the bridge, GGUF, prompt and continuations fixed at 512 prompt tokens and 32
decode steps. It compares the v11 bundle with host projections, the shared NPU image with host
projections, and the shared NPU image with its singleton projections enabled. Turning off
singleton projections on the shared bundle still selects its shared attention and expert image.
The three runs in each order isolate context sharing from projection offload. The preflight
verified the bundle overlap, bridge handshake and build provenance. The device queue runs the
12-launch mixed operator check before the model comparison. The first projection completed, then
the IQ4 expert dispatch reported ERT state 8. Windows PCI record 188576 falls in the same window
and has the lost-command signature. The model comparison still needs a successful operator run.
`.plans/inkling2/shared-dense-decode/mixed/run.log` SHA-256
`e54b9cf82c63c82f01039c8624d93e707fd338372391898d74a558f307bcaf4f`.

The 12-launch fixture now stages all outputs in its own run directory, with
the same instructions, weights, queries and references. A new preflight passed
and the retry is queued after BF16 evaluation. The Phlegm model comparison
starts only after the mixed operator passes its independent numerical tests
and the Windows event check.
It again completed the Q8 projection and failed on dispatching the following
IQ4_NL expert, with ERT state 8 and one PCI context error. The Phlegm
singleton model comparison is still pending diagnosis of this transition.

The generated worker revealed a stream-count error. The generator searched
for `range_` when shortening the expert loops. The upstream loops used
`range`, so the generated IQ4_NL worker tried to consume four weight records
per activation group while DMA supplied two. The extra loop indices also
selected activations outside that group. The generator now checks both source
anchors and emits `ACT_GROUP // pair` for both loops. An offline census
confirmed the emitted loop bound matches the DMA record count in both modes
at widths 1,152, 2,304 and 3,072, and that the last activation index is three
in each four-tile group. All four streams rebuilt and passed shared-image
preflight. All 12 corrected mixed launches completed with zero power or PCI
events. Relative L2 was at most `2.54e-6` for the dense and expert projections
and `4.71e-5` for attention. All 13 corrected streams now share one audited
32-core image in the rebuilt bundle. The Phlegm comparison passed preflight
and is running. The 63-file bundle digest list has SHA-256
`90596bd0ba417f1f04eacf1eb7c7a73fec50276fe8ae3cf5456b1109935633c9`.
The corrected singleton image digest is
`89fd7877a4d1158155a5fe070d858e8d9bb88d32e001cd092d07d93bb6542a66`.
`.plans/inkling2/shared-dense-decode-fixed/fifo-census.json`
has SHA-256 `66ea423d327e3234f7289c9d2b862a455ddde7a99ffc3043557c7139798c704e`.
`mixed/run.log` has SHA-256
`f99cb11c6fe1013aa6432c1f6f2130355e40b64b65d7b05ee05dc650eaa2790e`.

Both model comparison orders reproduced their own full logit tapes. Shared
NPU dense decode retained all 33 top IDs against the shared-image host
projection control, with mean KL `0.000102` nats and maximum KL `0.000847`.
Warm 32-step decode took 9.203 and 8.729 s on the NPU dense route, versus
10.346 and 9.625 s with host projections on the same image. The v11 host
controls took 10.452 and 9.623 s. Their 8.26% decode drift exceeded the 5%
limit, and one PCI event fell in the run's event window. This qualifies the
fixed prompt and forced continuation numerics. The observed speed gains
need a clean repeat before adoption. The capture under
`.plans/inkling2/shared-singleton-current/fixed-model/model.log` has SHA-256
`65aff3f8592fd052ff582322da68e266a5f2df4e0c05091b9c676b5081540fd3`.

The focused repeat held the verified shared image fixed and compared host and NPU
singleton projections in both orders. At a 512-token prefix, warm 32-step decode
took 12.032 and 12.116 s with host projections and 9.344 and 10.347 s with NPU
projections, a 22.3% and 14.6% reduction. Control drift was 0.69%, and Windows
recorded zero power and PCI events. The 33 greedy IDs matched, with maximum KL
`0.0008475`. The calculation isolates projection offload on this image. A second
bracket compares the full shared route with the normal Phlegm bundle on the
current engine, which includes wide attention and the optional paired head.
`shared-singleton-current/clean-recheck-175284/result.json` has SHA-256
`d608398f347b2c7ab478c7f397a56ce8a61ca2b4fac0dc2a7f3c512fbf7a308e`.

The current-engine bracket compared the normal v11 bundle with the shared
singleton image using the same Phlegm bridge. Warm 32-step decode took 11.561
and 11.038 s with the normal bundle and 9.512 and 9.873 s with the shared image.
Both matched all 33 greedy IDs. The maximum KL was `0.0008354`, the final prefill
logits were exact, and the control drift was 4.63%. Windows recorded zero power
or PCI context events. In the speculative replay, verification with host dense
projections changed accepted logits and the immediate subsequent decode at both
tested prefixes. Greedy IDs and subsequent draft IDs matched. Using the same NPU
dense projections per verification row reduced maximum accepted-logit KL from
`0.00128845` to `0.00000508` at prefix 5. Matching the Q8 expert arithmetic as well
made the 5-token case and the 1,023-token sliding-window case exact across
accepted logits, subsequent draft IDs and logits, and the next direct decode
after partial commit. Both cases ran with zero power and PCI context events.
The same bridge and bundle then passed all 18 acceptance and rollback cases at
prefixes 5 and 1,023. Accepted logits, subsequent draft and target logits, and
draft proposals matched exactly. The suffix-replacement checks and the two
invalid request cases passed. The interval had no power or PCI context events.
`shared-singleton-current/canonical-recheck/comparison-138528/result.json`
has SHA-256 `09469058d11561d6292f611caca3ba22e2eea828524b67dfe941cf8ae8d971c6`.
The two diagnostic receipts in the same directory's parent are
`verify-148472/result.json` SHA-256 `f4a19cd71567b54686b742d80635675de3f6fbbfc4aa202729a504f0c0d1553f`
and `verify-169636/result.json` SHA-256 `cd1376a405f832e362e85dd302ffe1b52207e472490aaa7418fe8665feafcab1`.
The dense-and-expert match receipt `verify-128412/result.json` has SHA-256
`513369e66c59e68cfc53c00fe26fbff2ccad92a057a05202af5f65adf6500293`.
`shared-singleton-current/matched-verify-q8/full-matrix-175632/force-accept/result.json`
has SHA-256 `f4688672ffb52b28272d7896e845f404ac3f3149d583562bc8dbe7311ef1e1e1`.

The exporter now builds and packages this route with `--singleton-decode`. The
optional v11 manifest shares one aligned image across four expert streams, six
attention tiers and two dense projections. Verification uses the same singleton
dense and split-scale Q8 expert arithmetic where the batch route would differ.
The C++ targets compiled. The first existing IQ4 stream rebuilt with identical
instructions, worker ELFs and hardware initialization. The new singleton IQ4 stream
reproduced its instruction bytes. Its linked worker image changed when the tracked
source reused the existing expert arithmetic, so it needs fresh device evidence.
All 12 tracked singleton streams then compiled, shared one 32-worker image and
passed the exporter with the paired head and wide attention enabled. Maximum
program text was 11,376 B, with an 8,192-byte stack reservation per worker and
three shim DMA descriptors per column. The rebuilt image passed 12 mixed
projection, expert and attention launches with zero power or PCI context events.
Maximum relative L2 was `2.53e-6` for the expert fixture and `4.70e-5` for attention.
The tracked engine replayed both partial-commit cases exactly with the rebuilt
image at prefixes 5 and 1,023, with no power or PCI context events. The offline
import audit passed with 55 unchanged files and 42 recorded engine changes.
The normal `phlegm serve` bracket compared the rebuilt package with the same
bundle without the singleton streams, after a 1,320-token prompt. All streamed
bytes, finish reasons and usage matched. With speculation off, timed decode fell
from 12.437 to 11.203 s and from 12.984 to 11.031 s in the respective run orders,
or 9.9% and 15.0% less time. Control drift was 4.3%. All four timed decode windows
had zero power or PCI context events. The first control had one PCI context
event 16 seconds before the request. The receipt retains the full-arm events.
`shared-singleton-current/source-builds/resources-complete.json` has SHA-256
`4fb897fa73c7f2fa9f38cfd3b2f9b5f18ac44b09e413750c0b534fa31dfec00d`.
`shared-singleton-current/source-mixed/run.log` has SHA-256
`5c8b47f5b1a361340efb53884ca8a7f74d12bdfbca570588c97a793b5393658e` and
`shared-singleton-current/canonical-recheck/verify-177288/result.json` has SHA-256
`37ed2a020ca10e1dad40aef2ac803cd4d024da84707b254c4c3cc5d8fa273cff`.
`shared-singleton-current/tracked-serve-169872/result.json` has SHA-256
`e2ae38ae16a803f88648666a552277f1f68c0b110d8652c7ebd74f499d24f807`.

The ggml-org GGUF has Q8_0 gate/up and down experts in layers 20–23. The
singleton image uses separate split-scale pools for those tensors. The GGUF
dimensions and wire sizes account for 4.27 GiB of additional expert weight
capacity and 0.77 GiB for the singleton dense matrices. The 1,320-token server
run sampled peaks of 22.02 GiB private and 35.56 GiB working set for the candidate,
versus 16.99 and 30.55 GiB for the final control. Sampling began during the first
candidate arm, after the first control ended. The private-memory difference
matches the additional weight estimate. The full-length fit must account for
both these allocations and growth of the KV cache after a warmed decode.
Keep this route off by default. A standard package-only export produced all 54
baseline files byte for byte even with the singleton kernels in the build
directory. Only `--singleton-decode` adds the streams and manifest entry.
`shared-singleton-current/singleton-residency-budget.json` has SHA-256
`c49f8380e1de309cd9f7912df910ce0840ac5235b094096ad8de9bfe66a12d5c`.
`shared-singleton-current/tracked-serve-169872/memory.jsonl` has SHA-256
`26e696062f6cc70a69dc7075c9e00862635f64e499541a9591a25f3d9f5b02f9`.

The grouped Q8 prefill kernel can read the singleton's split-scale wire.
The two prototype streams share a 32-core image. On the same eight expert
planes the kernel using both scale limbs matched independent FP64 dots
within `2.67e-6` relative L2. Its median warm launches took 1.338 ms up and
0.652 ms down, compared with 1.215 ms and 0.695 ms for the standard kernel.
A second build used the same wire and kept the standard prefill arithmetic
by reading only the high scale limb. That build matched every output byte,
including inactive rows. Its clean run took 1.438 ms up and 0.644 ms down,
versus 1.281 and 0.584 ms for the control. The two formats use separate
images, and these timings cover only the isolated operator. The prototype
retains the 4.27 GiB split-scale pool and can remove the separate 4.03 GiB
rounded-scale pool in a singleton bundle. The dense projection pool is
separate. The standard export keeps singleton
decode off. The 768-token Phlegm comparison matched all logits and DSpark
proposals bit for bit through decode, speculative partial accept, resumed
block prefill and reset. Private memory was 241 MiB higher after prefill
and 4.05 GiB lower after the first decode. Prefill took 49.29 s for the
control and 57.27 s for sharing the pool. In this single-order run, the
first block accounted for 7.19 s of that 7.98 s difference. The interval
had no power or PCI events. The shared-pool route remains a prototype.
`expert-m48/shared-q8-scale/board-179340/result.json` has SHA-256
`72f3d611300f423598cb90241d1a3566ea961dd2233fa9e765cbc12fc1f6cace`.
`expert-m48/shared-q8-scale-baseline/board-152932/result.json` has SHA-256
`66137ff1761ef0bfca20aadf5b209bd4830e082315c4d8a70eabf1466e89c4ac`.
`expert-m48/shared-q8-scale-baseline/phlegm-177360/result.json` has SHA-256
`84dd19c5f8c6d50120c7c98a1ed20b71da941eceebc6791f4094a56a9ec3e5f3`.

The source-built singleton package also passed the 1,320-token `phlegm serve`
request with DSpark enabled. It preserved all streamed bytes, usage and the
finish reason and accepted 20 of 29 proposals. Throughput was 2.32 tokens/s
against 2.81 for the retained plain-decode arm. There were no power or PCI events
during decode. One PCI event occurred before the request. This single-order
observation supports keeping DSpark opt-in on this route.
`shared-singleton-current/tracked-dspark-177808/result.json` has SHA-256
`f04d1813daa67b882708f9f0c6d373c609d91745fe5057d02d50251045f5ed10`.

| Capture in `.plans/inkling2/shared-singleton-current/` | SHA-256 |
| --- | --- |
| `phlegm-bridge/phlegm-reference-bench.exe` | `36fa96cbee07f74109fa28d7d16e4b05ed222faaeda04d643b2fee09e330efed` |
| `compare-preflight.log` | `e50bc7427bfd8435abb87f132282bce905f665843f357a9e3056c87b8f6ef243` |

| Capture in `.plans/inkling2/shared-dense-decode/` | SHA-256 |
| --- | --- |
| `resources-first-four.json` | `dd1bed232cf1c228734ff3f2091524a291a0f4d94c5baa2b39415904f4bed91e` |
| `resources-complete.json` | `57ae8deadaa3adab1583244738c598873fee06a59b15d867085bbf430b10f7b4` |
| `mixed/run.cfg` | `441aa1ab169a57fcaa26ef7437046169f858a16395897ed75c68816ddf2d7c66` |
| `fused-split-2304/expected.npy` | `126f7b37224cc71d5d77ad915202d0e72ac7a6b842f9c027b71e6353e69536cf` |
| `check-io-complete.log` | `f7e3b810d5809bc5642a587b716895166931c835591add2883e55da4977ac420` |
| `bundle/manifest.json` | `6b705144ac52013c9af79699c016bdef4cb57ba2e23771e3ab90f476273c1f60` |
| `phlegm-bridge/phlegm-reference-bench.exe` | `b23beb422571d9024e0ec9cad2745c31c756005685a234a8315c444af61dbb7d` |

### Host projections

The activation packer now converts eight lanes at a time on AVX2 for the Inkling expert and dense
projection routes. It retains BF16 rounding, the per-row scale, signed FP32 saturation, padding,
and the existing device byte layout. MSVC AVX2, Clang AVX2, and the scalar build each matched 512
boundary cases and 16 separate NumPy fixtures built from captured model values. The boundary
check covers all four rounding modes, subnormals, the finite exponent range, partial tiles,
unaligned storage, and rejection of nonfinite inputs.

On the HX PRO 370, the isolated MSVC AVX2 pack of 16 by 2,304 BF16-rounded activations went
from a median 142.88 µs to 11.78 µs. For the 64 by 3,072 direct FP32 activation tile, it went
from 761.78 µs to 53.45 µs. Across sampled shapes the median paired ratios were 10.37–17.42×
under MSVC and 14.98–26.16× under Clang. These timings used a single worker at BelowNormal
priority while the NPU was running a separate comparison. The full Phlegm speedup still needs
the paired 256-token model test with v11 and the same source except for the packer.
The native `GGUF-PACK` test passed, the source import audit still passes, and the two Phlegm
bridge sidecars and handshakes confirm `pools.cpp` is the only changed build input. The paired
model test waits behind the existing recalls and smoke tests.

| Capture in `.plans/inkling2/activation-pack-avx2/` | SHA-256 |
| --- | --- |
| `operands.bin` | `2ff5e7c65c04182945a138373cc5b0ec67bd4aa87d4e39f56be8d72de490cd98` |
| `msvc-avx2-final.log` | `0a46b0e5f849658d78c199d2bd2a71b5d600e4ff5b2ff12105c2593d800744c0` |
| `clang-avx2-final.log` | `4faf04bccb68e9593409f0abca6af625e240dc00b6519e29c4b69f93c5f4361f` |
| `msvc-portable-final.log` | `0a467722a63c8ab49c3b1be8f21bee622173fda28dd67fe757a7cb515bf75aa1` |
| `pair-preflight.log` | `43c6c9cdfa1a86ea4f9977f8499f61f8b67ee77dbf6fe994c30e38539c5d02a4` |

The AVX2 singleton path reads Q8_0 rows directly from the mapped GGUF. It keeps the scalar projection's
four accumulation chains and final reduction order. The general path handles multi-row inputs and folded
weights. Real GGUF tests compared packed projections against dequantization and the scalar dot on Q, relative
query, sinks, output, expert plane 127 and the complete language head. All rows matched byte for byte
under MSVC and under Clang with floating-point contraction enabled.

The CPU-only model replay matched all 10 greedy IDs and 2,000,580 logits byte for byte. The paired
decode times for nine steps were 16.79 and 17.47 s on the scalar control and 12.83 and 13.31 s on AVX2.
Both binaries were built with AVX2 enabled and used four threads during the 32K recall. These timings
apply to the CPU-only model. The replay SHA-256 is
`9a453db13261bbd679346615fb61253c7242ff0fe3e2bef17894141e1e5d45ad`.

The host singleton path now schedules the six Q8_0 attention input projections together. Each worker
uses the existing packed dot product and keeps the previous sum order. Sliding visits with no
controller message reuse the already-normalized Q input. The host uses the individual projections
for other quant types, folded weights or a non-AVX2 build.

Twelve rotated trials per layer used saved model features and the actual IQ4_NL GGUF's Q8_0 weights
from layers 0, 6, 11 and 23. The 8,600 outputs per layer matched byte for byte for shared and
separate Q, K and V inputs. At six threads the median paired stage speedup was 1.22–1.35× across
the four layers. The single-projection test also compared hardware half conversion and two wider
load variants against independently unpacked weights. At one input row and six threads the widened
weights variant's median paired speedup was 1.05× across seven target matrices. The runtime adopts
the scheduling change and uses the existing dot product.

Four ordered CPU-only replays of a five-token prefill and eight decode steps returned the same
nine greedy IDs and all 1,800,522 logits bit for bit. The before build took 8.65–8.66 s for the
eight steps, and the after build took 8.57–8.58 s. These measurements overlapped the 131K request.
Native CPU verify passed every accepted length from zero through eight. Both official Phlegm bridge
builds passed the protocol handshake, and their provenance differs only in `inkling2_host.cpp`.
The paired Phlegm NPU replay completed at 128 and 1,023 prompt tokens with 16 fixed decode
steps. The full 17 logit rows matched bit for bit in all four runs at each prefix. At 1,023,
the control took 8.560 and 8.609 s for warm decode and the grouped projection took 8.459 and
8.185 s. At 128 the paired ratio was 1.048×, with build activity overlapping part of the timing.
These development measurements support the existing arithmetic and scheduling change.

| Capture in `.plans/inkling2/q8-decode-scales/` | SHA-256 |
| --- | --- |
| `summary.json` | `6ca4bda5883498a59fa4a7a5d76ef3c5d684e3cb0da685906f9d72f1e3c7e3f7` |
| `grouped-4.csv` | `82d3bfe8c55f95bbe1fd3c68182224a28a9ad2f47613b2d4d8d46e15a28b3b77` |
| `grouped-6.csv` | `cff03a93f5b3c393ba89144d6577d4dbc621b31e70d261ba28406e1217a2727c` |
| `full-0-before.f32` through `full-3-before.f32` | `1b68845300f78563fd06b658c80f57547e11f27a2e26ee2ad43a22b601709ca3` |
| `verify-after.log` | `3b592c9e5f6a950f11d27ba0002f6bff64b937d592cc7a568172bdf4ab915fdb` |
| `phlegm-preflight.log` | `49ba1c69814336151bebf590b606775996a705f87e66cc44da309130e947273b` |
| `phlegm-pair-queue.log` | `27bfd38b4f3c0c540eae36aa906596a41cd24bb8bb6b5db113a83798a899e039` |

Small F32 controller and router projections now use the existing row-parallel loop when a batch has at
least 2,097,152 input-weight products. At 128 rows, the measured paired ratios for the two isolated
matrices were 3.93× and 3.54×. The CPU-only 128-token replay had identical logits and took 66.44 s on
the original route and 67.07 s with both changes. This sample establishes arithmetic parity for the
CPU route. Replay SHA-256
`f9c5db7dc4333ee5df7331ba0ad74f401c4b4e4fde562d22815a9a32220362d5`.

An experimental F32 projection processes four input rows per weight load while preserving each row's
four accumulation chains. It matched the original projection byte for byte on three actual GGUF
matrices and batch sizes 1, 5, 32, 128 and 257 under MSVC and Clang. At batch 128 the isolated MSVC
projection was 2.31–4.60× faster. All five full logit rows from the 32-token CPU replay matched across
the control–candidate–candidate–control order. Control prefills took 29.18 and 49.84 s, and candidate
prefills took 39.93 and 41.25 s during the concurrent Phlegm vision run. These timings are inconclusive
for the full model. The experiment is staged for on-device profiling before any runtime change.
The four replay dumps in `.plans/inkling2/f32-kernel-probe/` share SHA-256
`6415d0b048845ed81b8df9349f72b5cacf449b9b8e2a3bd2417781bf9e22565c`.

The subsequent 512-token NPU profile evaluated control, tiled F32, tiled F32 and control, each
with a warm trial and 32 forced decode steps. Every run produced the same full set of 6,601,914
float logits. The first control's cold load overlapped a CPU correctness check that ended before
its warm trial. Six host threads were used throughout.

| Order | Warm control prefill | Warm tiled prefill | Control controller and router | Tiled controller and router |
| --- | ---: | ---: | ---: | ---: |
| Control first | 38.580 s | 35.475 s | 4.131 s | 1.183 s |
| Tiled first | 38.805 s | 36.472 s | 4.092 s | 1.116 s |

Warm prefill fell 6.0–8.1%. Warm decode went from 14.925 and 14.942 s to 14.729 and 14.696 s.
The capture `.plans/inkling2/f32-kernel-probe/npu-pair-queue.log` has SHA-256
`2fe480bc59bf4df5dccde2e88148ee2a674c5407fd0db9636c6e8083c2009d87`.
The verified F32 path is now in the host engine. A real 32-token replay on the current cache
implementation retained all five predictions and 1,000,290 logits. The first draft oracle check
used the UD-Q4_K_XL target and differed on three proposals. With the saved oracle's UD-IQ1_S
target and IQ4_XS embedding and head, native Q8_0 DSpark matched all 21 proposal IDs and kept
exact state across the sliding-window acceptance test. Its `release/draft.log` SHA-256 is
`3a83e0a309c4a4f84a76283277121bb2c62409dd3a5806436d5ebb5fa6fbc531`.

The target short convolutions now transpose the static taps once at model load and reuse each
dynamic tap across a head. The tap accumulation order and stored input history are unchanged.
A replay using the IQ4_NL file's actual tap tensors and dynamic projections matched output rows,
history and speculative input tape byte for byte across blocks of 1 to 256 tokens. It passed on
MSVC AVX2, Clang AVX2 with FMA contraction, and portable Clang. In the MSVC operator test, the
128-row sliding K convolution took 0.508 ms before and 0.111 ms after, and the attention and MoE
post-convolutions fell from 3.51 to 0.93 ms and from 3.35 to 1.08 ms. The full-layer dynamic K/V
operators improved 1.20–1.23×, including their tap projections. Each case alternated nine paired
samples and discarded the first pair. The 131K Phlegm request was active during these host tests.
An independent 32-token CPU model replay retained all 1,000,290 logits and five greedy IDs.
Its before and after dumps share SHA-256
`6415d0b048845ed81b8df9349f72b5cacf449b9b8e2a3bd2417781bf9e22565c`.
The rebuilt native reference suite passed all four checks, and the native verifier matched logits
and DSpark features for accepted lengths 0 through 8. The official Phlegm bridge built with the
same traced sources as the preceding bridge except for `inkling2_host.cpp`. Its handshake and
the remaining 131K and full-context fixture preflights passed. Hardware timing of this change is
pending.

| Operator replay under `.plans/inkling2/conv-layout/` | SHA-256 |
| --- | --- |
| `msvc.log` | `ea11404695c62141d8f97b6f8f6c5cd5ced9a90b9cf9255ff415f9629c3a7ae4` |
| `clang.log` | `4161f21d38b93568ad30b061cbc133ef45609ffc5bd35b2f9d741d742cc6bdff` |
| `portable.log` | `a3db6cd355ea725150769ca83a7793b8d2a4745b05d040879f1044c4285df203` |
| `verify.log` | `3b592c9e5f6a950f11d27ba0002f6bff64b937d592cc7a568172bdf4ab915fdb` |
| `bridge-preflight.log` | `5263240f49060ae747eccf25708667c93910d8e7a1d89a92f30143b3c8e21992` |

The Phlegm development bridge then ran the fixed 128-token prefix and eight forced continuations with
the v11 NPU bundle, six host threads and passive OpenMP waiting. The host-projection build and the
previous bridge each used a fresh resident process in both orders. The second trial retained packed
weights. All nine greedy IDs matched across all eight trials.

| Order | Control warm prefill | Optimized warm prefill | Control warm decode, 8 steps | Optimized warm decode, 8 steps |
| --- | ---: | ---: | ---: | ---: |
| Control first | 10.754 s | 9.081 s | 4.761 s | 2.748 s |
| Optimized first | 10.816 s | 9.760 s | 5.302 s | 2.481 s |

Prefill improved 1.11–1.18× and decode 1.73–2.14× in these development pairs. The warm decode control
varied by 11% between orders. A release sweep remains open.
`.plans/inkling2/singleton-host/phlegm-paired.jsonl` SHA-256
`ac28352277102f03de05e7c261d59a629151c8a2bcc664d05d86f413548d7239`.

A separate cold bridge run prefilled 1,023 tokens and forced ordinary decode through positions
1,024–1,026. All four greedy IDs matched the control across the sliding boundary. The run establishes
ordinary decode parity there. The Phlegm DSpark trace later crosses the same boundary.
`.plans/inkling2/singleton-host/phlegm-window.jsonl` SHA-256
`c111d803934f3c726b0c50919d9674ffe5891a6f9b58cb1f39144caac0c7a7dd`.

The pinned llama.cpp reference used ggml-org IQ4_NL, BF16 K/V and Vulkan on the Radeon 890M.
It consumed the same 128 prompt IDs and eight forced continuations as the Phlegm pairs and
matched all nine greedy IDs. The reference also matched all three greedy IDs at positions
1,024–1,026. Comparing those three full logit rows against the retained v8 NPU capture gave
0.0283–0.0856 relative L2, 0.9966–0.9994 Pearson correlation and at most 0.007794 nats KL
from Vulkan to the NPU. Rankings beyond the best ID differed. These were functional comparisons
while the NPU recall and CPU replay were active. An isolated timing comparison remains open.

| Capture in `.plans/inkling2/verify-matrix/` | SHA-256 |
| --- | --- |
| `vulkan-128-functional.jsonl` | `ca3d70fa42514c2ffa2be5cc90ba6aa15357850014a2987fa411d9248601afe4` |
| `vulkan-window-parity.json` | `7439e3404e121102875ee8658c981c604d5e0dffd15e2eccf8efaaf96b7e78d0` |

The ordered 512-token comparison used the same IQ4_NL file and 32 fixed continuation tokens on
Phlegm's experimental v11 bridge and the pinned Vulkan reference. Each backend ran a cold and a
warm trial in its own process, with the other backend stopped. The Vulkan runner used BF16 K/V,
256-token batches and 128-token microbatches. Both used six host threads.

| Order | Phlegm warm prefill | Vulkan warm prefill | Phlegm 32-step decode | Vulkan 32-step decode |
| --- | ---: | ---: | ---: | ---: |
| Phlegm first | 60.641 s | 7.610 s | 28.155 s | 4.513 s |
| Vulkan first | 42.724 s | 6.416 s | 23.932 s | 6.253 s |

In both orders Phlegm took 6.66–7.97× as long to prefill and 3.83–6.24× as long to decode.
All four runs per backend reproduced their own IDs. The backends matched 32 of 33 predictions
and disagreed after 522 processed rows. Vulkan preferred 149917 with probability 0.1865 and
ranked Phlegm's 150266 second at 0.1752, a logit margin of 0.0620. The Vulkan logit replay
reproduced its original trace. Phlegm's full logit comparison at this row and the current
dispatch/host split remain pending. This development test leaves the near-optimal performance
gate open.

| Capture in `.plans/inkling2/query-pack-reuse/` | SHA-256 |
| --- | --- |
| `paired-phlegm-vulkan-512.log` | `5739e41c680659dccc92eccc4faed94f276b2e451c5be9be0ab9198586ed3ad1` |
| `vulkan-512-margin.json` | `5aeac6d4f5a1b4ea4f1d834000cc838032d711dc9da688d412921a33c57e89de` |

A CPU-only replay with the same pinned llama.cpp revision, GGUF, BF16 K/V, batch geometry and
forced IDs agreed with Phlegm on all 33 greedy predictions. At position 522 the CPU reference
preferred Phlegm's ID 150266 over Vulkan's ID 149917 by 0.0399 logits. CPU-to-Vulkan KL over
the full logit rows averaged 0.001417 nats with a maximum of 0.007229. The row at position 522
had KL 0.002558. The 131K recall was active during this four-thread CPU run. These results
resolve the observed greedy discrepancy against the CPU reference. Full NPU logit comparison
and the release performance comparison remain open.

| Capture in `.plans/inkling2/conv-layout/` | SHA-256 |
| --- | --- |
| `cpu-512.json` | `95fbde0003c50d10d9d52af08aa26384d2e8061f3ceac3479932c5b7fb7e8f99` |
| `cpu-512.f32` | `f32c259bcbe64c25643a122c5b1b7b22180d975fc8f2f0fdf539246ba6018bc7` |

The instrumented v11 NPU replay captured all 33 full-vocabulary rows for the same inputs.
It reproduced all 33 earlier Phlegm predictions and all 33 llama.cpp CPU predictions. At
position 522 its margin for ID 150266 was 0.07847 logits. Mean KL(NPU ‖ CPU) was 0.002078
nats and the maximum was 0.02364 at position 533. Mean KL(NPU ‖ Vulkan) was 0.000902 nats
and the maximum was 0.005365 at position 533. The held-out quality and release gates remain open.

The warm NPU control spent 17.14 of 38.58 s in grouped expert prefill interfaces, including
device waits, packing and readback. It spent 14.35 s outside measured device interfaces,
including 4.34 s in the measured host projections. Over 32 forced decode steps, it spent
5.34 of 14.92 s in expert interfaces, 3.42 s in relative attention and 5.64 s outside the
device interfaces. Decode transferred 8.05 GB of physical K/V records and 0.51 GB of query
records across 1,536 attention dispatches. These are development trace totals.

Grouped experts and dense projections accounted for 21.73 s of the 38.58 s prefill.
Halving the entire time of both stages with the other stages held fixed would give
27.72 s. Removing every measured device interface would still leave 14.35 s from
this trace. Warm Vulkan took 6.42–7.61 s in the separate 512-token comparison.
These are arithmetic scenarios from the retained trace. Pairing launches preserves
the matrix work and bytes. Its actual stage savings require hardware measurements.
The evidence is `.plans/inkling2/f32-kernel-probe/npu-512-audit.json`, SHA-256
`2b5e93a092cfb307a32408ccb99a09f598487c2db1cf822ed2c90d6b41aca514`.

The 8,225-token control prefilled at 15.0 tokens/s. Ordinary prefill is now the performance
priority. In the retained 512-token trace, 4,452 grouped expert launches took 17.14 s at
the interface, including 9.43 s to submit and wait and 3.92 s for readback and validation.
That run issued 7,205 device launches in total. Another 10.01 s lay outside both the timed
device interfaces and the named host projections. The new Phlegm profiling bridge tracks
these per block, with separate counters for expert packing, sync, wait, finite scan, route
gather, SiTU, reduction, residual mix, controller, convolution, and dense and relative
attention. Its ordinary v11 profile is separate from the 18K older-key qualification.
The bridge built and passed preflight at 512 and 8,225 tokens. A five-token CPU smoke
captured all 48 visits in each of three chunks and reproduced the control logits bit for
bit. NPU timings from this profiler are still pending.

| Capture under `.plans/inkling2/ordinary-prefill-profile/` | SHA-256 |
| --- | --- |
| `preflight-512/input.json` | `9c8e54b47b1ed29c9091174f6abc8864c1e87a8b237bcb7900091d4dc27eb7fc` |
| `preflight-8225/input.json` | `56ed67a8db29bba8490e02fccd70585732f7ed67cb2eac5162aa803cc6c2d583` |
| `smoke-profile-raw.log` | `53b866804aaab40edc5e572084cfae4845b7769c2396e8c5082cb8033aea7dc4` |
| `smoke-profile-raw.f32` | `ba976018ea7aa452130220da129eba264e97ec045e07cca0917d213f3b355467` |

An AVX2 experiment checks the exponent bits of all 294,912 floats in a grouped
projection result, including the padded rows. It matched the scalar predicate on
both signs, all 256 exponents, five mantissa boundaries, and every position in
unaligned buffers of lengths 1–37. Captured IQ4_NL gate/up and Q8_0 down output
buffers both passed. On the Windows host, the 1.125 MiB scan took 15.2–15.4 μs with
a reused buffer and 25.7–27.9 μs cycling through 48 buffers. The corresponding
scalar ranges were 419.8–434.3 μs and 428.7–457.1 μs. These are CPU-only timings on
the captured outputs. The combined readback and validation time from the previous
NPU profile cannot serve as the denominator for a speedup claim.

The isolated Phlegm build switches between both checks at the same grouped dispatch
and keeps scalar SiTU in both arms. It passed handshake and provenance checks for
512 and 8,225 tokens. On-device full-logit equality and separate validation,
readback, and overall prefill timings remain open. The capture is
`.plans/inkling2/expert-finite-scan/probe.log`, SHA-256
`953bbaeb1b6dd89c8c700a0bd14bd8c3d5f209b5b0a58afc2bc21c1896277a27`.
The replay bridge SHA-256 is
`ab52aff503d43ae239044f4ae5dc3f35efa146b500cb17dcf81d58cc4013889f`.

A block-size experiment keeps the M16 expert kernels and lets the Phlegm fast route
ingest 512 tokens per layer-major block. Repeating the captured 256-token routing
histograms and merging each pair reduces the estimated grouped projection count from
4,432 to 3,674 (17.1%). Both cases account for the same 196,608 token/expert
assignments across 48 visits. The repeated histogram is a dispatch estimate. The
device test below measures the actual routing schedule.

The profiled bridge advertises either 256 or 512 tokens from the same executable.
Each comparison arm warms on the same 512-token prefix. The saved profiler
verified both requested block shapes, a partial warm-up block, and rejection of a
wrong advertised route in a simulated bridge. The real build passed both protocol
preflights. In both run orders the 512-token block preserved every final logit and
reduced grouped launches from 4,452 to 3,788. Warm prefill fell from 23.090 to 21.240 s
and from 23.054 to 20.824 s, gains of 8.0% and 9.7% with 0.15% control drift. The
control and candidate used synchronous launches. The larger block still needs
a joint comparison with overlap and longer-window parity before changing Phlegm's
block size. The receipt `ordinary-device-20261123/block-512/pair-result.json` under
`.plans/inkling2/prefill-block-512/` has SHA-256
`64b0d038eba7199bd508b0923ba81c96ece12ac28abde1eaafc343940205b0a4`.
The joint comparison is staged under `.plans/inkling2/overlap-block-size/`. Its source
manifest matches the integrated profiled build except for the Phlegm adapter's
test-only choice of block size. All four protocol preflights passed. It uses a
1,536-token prompt to cross the 1,024-token sliding window, with the same context
allocation and 512-token warm-up in both arms. The board comparison followed the
integrated parity check. All four arms matched their final logits exactly and
recorded no power or PCI events. Grouped launches fell from 13,446 to 11,384. The
256-token controls ran in 47.528 and 48.146 s. The 512-token candidate ran in 49.869
and 48.137 s, with 1.29% control drift. The larger block provided no consistent
speed gain once overlap was enabled. Keep 256 as the fast-mode block size.
`device-1536/pair-result.json` has SHA-256
`9610c59255cfdf5f6a408111f8b4c9157a4569c1f3e62195f06f38b93762885d`. Bridge SHA-256
`7c9922bc15d321103a7b963ea5fc390ec79f6b4d281284ade6c4e4079968e910`.
The build SHA-256 is `1609b98a4ef1910e211f55d9323ffe5423b49929ec42252a9dd7b97ec2b26456`.
The inputs and expected launch count are in `.plans/inkling2/prefill-block-512/census.json`
(SHA-256 `7fe5fb726f5e5d89a1d127416fcbfb25f2feb3ab4405eaa6ee0ab02dd36f581f`).

A separate CPU probe packed each input token once for the grouped gate/up operation
and gathered the resulting wire rows for each of its eight expert assignments. It
replayed the captured 256-token expert counts across all 48 visits using synthetic
activations and token assignments. All 8,864 slots matched the existing gather and
pack byte for byte, including padding. The probe also covered every partial slot
length and rejected a non-finite source. The cache took 23.3–23.5 ms versus
66.0–70.7 ms for the current path, a saving of about 45 ms for one 256-token
prompt. Both arms wrote host buffers. The probe leaves device buffer sync and
dispatch unchanged. This result did not warrant changing the runtime interface.
Capture `.plans/inkling2/expert-activation-reuse/probe.log`, SHA-256
`c7be8bbb5d1de527396e577f89c2c8f841433947e67bb1114770c37b92cd3afd`.

A separate dense Q8 experiment issues two existing 64-row waves in one NPU launch.
It keeps the core arithmetic and output order, supplies a separate activation fill
and result drain for each wave, and repeats the same weight DMA. It saves submissions
and host buffer syncs while transferring the same weight bytes as two single launches.
For a 512-token prompt, the layer schedule predicts 1,216 dense launches in the control
and 608 in the paired arm. This is about 8.4% fewer launches overall relative to the
retained 7,205-launch trace. The model has not yet run the paired route on the NPU.

All four paired streams compiled and match the single-wave image's 32 worker ELFs
and hardware initialization. The DMA audit verified 32 shim descriptors per stream, with
two drains per output stream, two fills per activation stream and one repeating fill
per weight stream. The native packer's 128-row input matched two separate 64-row packs
byte for byte in all 18 full, partial and boundary cases. Prepared operator fixtures
cover fused input, query, key, value and output projections with separate activation
rows in the two waves. The isolated Phlegm bridge passed handshake and provenance
checks at 512 and 8,192 tokens. Device parity and model timing remain open.

| Capture under `.plans/inkling2/dense-paired-launch/` | SHA-256 |
| --- | --- |
| `preflight/input.json` | `5f4a56cab628db2d07d6eb3b8a516a19116d9e0d024ab9cc31afa4506e17c975` |
| `bridge/phlegm-reference-bench.exe` | `4f42fe6f5f986a5c13f1254aeee0da38fe06a3a1e5b3cc032b77943838e7a1e4` |

The grouped expert experiment similarly brings two eight-slot M16 groups into one
launch. Each wave has its own eight patched expert addresses. IQ4_NL and Q8_0 gate/up
and down reuse the current 32-core worker images and hardware initialization. The
host gathers in ascending expert order, runs SiTU on each valid row, and reduces
in the original order. A final unpaired group uses the existing single-wave stream.
The route reduces submissions and buffer sync calls. It retains the same physical
weight and activation transfers for the same routing decisions.

The captured 256-token routing histogram predicts 2,216 control launches and 1,134
paired-plus-tail launches, a 48.8% reduction within the grouped expert stage. The latter
contains 1,082 two-wave launches and 52 unpaired launches across both projection legs.
All four streams compiled. The operator fixtures checked all 48 shim DMA descriptors
in each stream and patched 16 weight offsets from two separate archived group launches.
Each wave has different expert assignments in the fixture. The model bridge passed
preflight in both modes at 512 and 8,192 tokens. Its device comparison checks the
launch count for each measured block, exact final logits, and paired elapsed times.
The 512-token expert and dense comparisons are queued behind the block-size replay.

| Capture under `.plans/inkling2/expert-paired-launch/` | SHA-256 |
| --- | --- |
| `preflight/input.json` | `75c6f202f10b5441725c9de83c57855fb532490803f91c2b7a80819665a31545` |
| `bridge/phlegm-reference-bench.exe` | `ce91a96730b3c7e8c5b786f755362fac5c102e8c5ad85b2a2cf9463f888c12d8` |

Both 512-token device comparisons preserved all logits. Expert pairing reduced
grouped launches from 4,452 to 2,280. Dense pairing reduced dense launches from
1,216 to 608. Control drift was 17.0% for experts and 34.0% for dense, outside
the 5% acceptance limit. These runs establish parity and launch counts. A wall
time improvement still needs a stable comparison with the integrated overlap route.

| Capture under `.plans/inkling2/expert-paired-launch/ordinary-device-20261123/` | SHA-256 |
| --- | --- |
| `expert-paired-launch/pair-result.json` | `5fada28c6cccdec5cc7c198e1f0a68d2df99f5442a58340f07a3a83779f726be` |
| `dense-paired-launch/pair-result.json` | `b806f7784cbc9cd2db6f0951fbf6e7fbf19705fa0426a3325f28231daf76d970` |

The decode-tier census reproduced the 8.05 GB exactly. All 32 steps in this fixture required
17 chunks and selected the 32-chunk stream. Adding a tier at 17 would reduce those K/V records
to 4.28 GB at the same launch count. This is a dispatch-size calculation. Hardware parity and
timing for a finer tier remain open.

Both 17-chunk streams are prepared for the existing v11 shared IQ4_NL and Q8_0 images.
Re-lowering the placed compiler IR reproduced the deployed 16- and 32-chunk instruction
streams byte for byte in both formats. The 17-chunk stream changes the count written to all 32
workers and the K/V DMA spans for all eight columns. K/V per launch decreases from 5,242,880
to 2,785,280 bytes. All other stream instructions and all packaged images are identical. Only
the new instruction paths and tier list change in the manifest. The queued 512-token model
comparison uses the same instrumented binary in both arms. It checks all 33 logit rows, the
prefill traffic, and the predicted decode bytes over 1,536 attention launches. Hardware parity
and timing remain open.

| Artifact under `.plans/inkling2/decode-attention-tier/` | SHA-256 |
| --- | --- |
| `iq4-17.bin` | `c9bccf4162751f9ec179c0c60abd3a25a12d1ee045e71a49a66b2609b1e457f3` |
| `q8-17.bin` | `ec5a58793087840a08a0b2e274cf97b3b72419a3da0837e8f861f09ea7e7d316` |
| `bundle/manifest.json` | `72c371a25b3ea276acb52df2c85e2455ee6eb21cdafc3e467351981acafeb9d8` |
| `model-preflight.log` | `4d42004780c7704bccb970d7faf862aeadc5e8478c3cc75e456a33822b894709` |

The clean development trace ties the 4,096-token measured trial to
31 singleton steps. Each averaged 418.67 ms of profiled time,
including 182.19 ms in attention and 124.58 ms in MoE. The trace
records 48 attention visits and 96 expert dispatches per token.
The new 33-chunk stream covers a 1,024-token aligned window with
one extra leading chunk. The 64-chunk fallback reads 10,485,760 B
per dispatch, and the exact tier reads 5,406,720 B. At the tested
4K positions, the schedule predicts about 6.30 GB less K/V traffic
across 31 tokens and 40 sliding visits. That is a transfer count.
The two new streams match their shared images and change only the
expected tier words and K/V spans. The packaged manifest and Phlegm
protocol preflight passed. The mirrored Phlegm comparison is queued
after the vectorized expert smoke.

The completed 1,024-token decode bracket matched every logit in all four arms
and recorded zero power and PCI events. Control drift was 0.82%. The 20-step
control took 8.911 and 8.984 s, and the tier took 9.454 and 8.861 s in the
matched orders. Attention timing also switched rank. The 64-chunk tier stays
in use. `decode-attention-tier/rolling-33-phlegm-146240/result.json` has SHA-256
`0373f6bd23ec7820d405697fddeb5567dffea39b4ee13dcf8c3f946d73142e8c`.

| Capture | SHA-256 |
| --- | --- |
| `release-bench/decode-stage-receipt.json` | `03db077c9d7a46a89fb4484d4d093d10a2d15b6104a0253ff9554115d71f6d49` |
| `decode-attention-tier/rolling-33/bundle/manifest.json` | `2a02ea3e497bafea99c237674a015039afc669796903d963beb65c9e91177838` |
| `decode-attention-tier/rolling-33/iq4-33.bin` | `24ad8daf2f8ae95c0a3cfee3d2142738e61bfcd04b708e6c7e85f96719e62724` |
| `decode-attention-tier/rolling-33/q8-33.bin` | `75fe54a5141335c312b40fc736aabd9e17571c3d136708916c5be4a1b9bdcbeb` |

| Artifact under `.plans/inkling2/f32-kernel-probe/` | SHA-256 |
| --- | --- |
| `npu-0-probe.f32` | `e76ceed57d8f0f7454f11383ab315e990a675183f07516ee2e161c69a7797d6f` |
| `npu-512-audit.json` | `2b5e93a092cfb307a32408ccb99a09f598487c2db1cf822ed2c90d6b41aca514` |
| `decode-tier-census.json` | `ac23ae5b81cefc6537bf83517211c60457e198f26b6d53d3ad19a413542d2f6f` |

An experiment ran the grouped prefill host SiTU by independent expert slot once the group had at
least 32 valid rows. The following weighted expert sums retained their original order.
An isolated-stage probe used the model's exact softcaps and captured IQ4_NL and Q8_0 NPU gate/up
outputs. The 52-row group used the capture's routing mask. The 128-row group reused captured
values in formerly padded rows. Serial and threaded outputs were bit-identical for each format,
both groups, and the 27-row serial fallback. The probe also checked that inactive rows were left
untouched. Six-thread medians for this stage are below. Other active workloads can affect these
timings.

| Format | 52 rows, serial / parallel | 128 rows, serial / parallel | Log SHA-256 in `.plans/inkling2/prefill-situ/` |
| --- | ---: | ---: | --- |
| IQ4_NL | 1.054 / 0.572 ms | 2.771 / 0.894 ms | `iq4-caps.log` `134d69ff8e620b307495442c7ab420d3864ad9f1ebdab507927c23b87e73c9f1` |
| Q8_0 | 0.864 / 0.488 ms | 3.338 / 1.122 ms | `q8-caps.log` `d66b9317def6eec1d2442771832e01233bddd8ae090c84787458942c9dfd33f6` |

An IQ4_NL probe also matched the output bits at 31, 32 and 33 active rows, including the change
from serial to parallel at 32. `.plans/inkling2/prefill-situ/iq4-boundary.log` SHA-256
`78f2f8a67abe12f074d1d7fe417c3d87b851bcf4f8d23adf61745c15871075bf`.

The ordered Phlegm pairs used the same 128-token prefix. All eight runs produced identical full
logits. With the control first, warm prefill took 11.895 s serial and 13.510 s threaded. With the
threaded build first, the times were 11.870 s serial and 11.873 s threaded. The paired run used
Performance power mode. Observed processor load was about 14–16% before it and 19–30% afterward.
The engine retains serial SiTU based on the application timings. The immutable threaded experiment
is identified separately in subsequent measurements. `.plans/inkling2/prefill-situ/phlegm-exact.jsonl`
SHA-256 `9d0d1c5fc3e9c82e67f04240021793977a3b7d07e5cafa703c3b837710a5ba8e`.

An AVX2 SiTU experiment uses the existing vector exponential and computes softcap tanh
with a small-argument polynomial and the exponential formula outside that interval.
The unguarded variant reached about `9.2e-8` relative L2 on captured IQ4_NL and Q8_0
gate/up outputs. Some values crossed BF16 rounding boundaries in the next projection.
The guarded variant recomputes lanes near those boundaries with scalar SiTU. It produced
the same BF16-based packed expert input bytes for the captured 52-row masks, 128-row
copies, and a synthetic fixture with very small, saturated, and boundary values. On the
captured fixtures it took 0.110–0.115 ms for 52 rows and 0.272 ms for 128 rows, compared
with 0.522–0.545 ms and 1.197–1.203 ms for scalar. This measures the host activation
alone. The FP32-direct pack path requires separate validation. The experiment is staged
on the grouped BF16 Phlegm path for same-binary replay. Full model timing and logits
remain open, and the ordinary engine still uses scalar SiTU.

The instrumented Phlegm build passed protocol and artifact preflight for 512 and 8,225
tokens. The comparison uses one binary in scalar, guarded, guarded, scalar order and
checks the route marker, exact vocabulary output, stage time, total prefill time, and
Windows power and PCI events. A simulated result verified rejection of a route mismatch
or changed logits. On-device measurements are pending.

| Capture under `.plans/inkling2/situ-avx2/` | SHA-256 |
| --- | --- |
| `iq4-guarded.log` | `c54bef868b29ef5b68975a00347efbc1f5705dce1305c10b5ae54410cf8c782d` |
| `q8-guarded.log` | `55eff85f9287eae442bd7bf60b35fcc812355bd63cfd4199c3db389ee867e2d0` |
| `boundary-stress.log` | `4486d341cd020707a3cca7b95fe6c5739199daa5eaf0c1dbefcab6ceab95f24e` |
| `bridge/phlegm-reference-bench.exe` | `6d6e7a2f6ae794d9efab26f87b0e59816a11b18c2f88068005912c4f07881bb1` |

Two offline experiments targeted the remaining residual mix. Caching normalized bank rows gave
about 6% lower isolated time and matched the complete CPU logits after 17 prompt tokens and
three greedy decode steps. The single end-to-end CPU pair did not show a prefill gain. Splitting
independent token rows across six host threads matched the isolated outputs and took 45.67 ms
against 107.00 ms for 12 mixes of 256 rows and a four-block bank. The parallel prototype also
matched all four complete CPU logit vectors after 32 prompt tokens and three greedy steps. The
experiments used concurrent workloads. The active Phlegm candidate retains the existing
residual-mix implementation until the parallel version can be timed in an ordered Phlegm pair.

| Capture in `.plans/inkling2/residual-mix/` | SHA-256 |
| --- | --- |
| `mix.log` | `34efc432bf8db6bb136b2d631b487ae70d0cb6435685da802391f62c631232c9` |
| `mix-parallel.log` | `90d6c28431d5e0c9031ec480412a9b6eab8c420b5355b5ae28c77115d5fe0679` |
| `parallel-32.f32` | `d90ca219aaeb364d6c9dc5b5dafecd2310e91a147d479621148914a6b6670ff9` |

The saved Phlegm profiler now times the controller's elementwise update inside the
overall controller stage and times the attention output gate separately. An offline
six-thread probe on 256 rows preserved every scalar output bit across row counts
1, 17, 63, 64 and 256. It used synthetic controller values and attention gate values
from an archived NPU projection fixture. The controller update took 2.50–2.54 ms
instead of 9.22–9.96 ms. The attention gate took 0.41–0.51 ms instead of 1.29–1.37 ms.
The isolated timings overlapped the 131K run. The controller-only Phlegm replay
uses the same binary for both arms and requests row parallelism on blocks of at
least 64 rows. Protocol and provenance checks passed at 512 and 8,225 prompt tokens.
A five-token native CPU smoke matched all archived vocabulary values and reported
96 controller updates and 48 attention gates in each of its three blocks. The
512-token model-level timing and exact-logit comparison is queued after the expert and
dense launch comparisons. Its four protocol and provenance preflights passed.

| Capture under `.plans/inkling2/host-row-parallel/` | SHA-256 |
| --- | --- |
| `probe.log` | `c31d84ac0907aa3d554ae35451d9aa98891ae40e4944475c574875a0c6d2bf8e` |
| `gate.f32` | `2d5bee16381f89d31ce0c2d7ace0fa5e84e2a30d07f9e4e8cff63083a932be9a` |
| `smoke-raw.log` | `0cdf7e118e657e8a1cfa21a1a81dd827dc94851fd3f0fd502c4bd3935c447cb5` |
| `bridge/phlegm-reference-bench.exe` | `0935e8c5a86fda485e4ade9b390893b7e7615d222b02ae838feebcac22aa8323` |

The profiler also separates layer normalization, Q/K/V postprocessing, KV commit,
embedding, head evaluation and CPU fallback. A native five-token CPU run with
blocks of 2, 2 and 1 matched all 200,058 archived logits. Each block reported 152
layer normalizations, 48 Q/K/V postprocessing calls, 48 host attention calls and
one cache commit. Only the final block reported a head evaluation and host head
fallback. The 512-token Phlegm reprofile passed its handshake and provenance check
and is queued after the controller comparison. It will measure how much of the
earlier 10.01 seconds these stages account for and whether any visits fall back.

| Capture under `.plans/inkling2/prefill-attribution-172514/` | SHA-256 |
| --- | --- |
| `smoke.log` | `1bf9f5b8c5e33fadad5fa151c5712f54678ad1968675bacb44358ca98ec0d2aa` |
| `bridge/phlegm-reference-bench.exe` | `60d56b77b1073f2d2ea650ddbc43b377a9c7e32a9c50a31a979fc02b3c01a191` |

One host probe split the Q/K/V postprocessing loop over six OpenMP threads. It replays
archived NPU projection outputs for synthetic activations and loads the real clipping
weights and scale coefficients from the target GGUF. It matched the original serial
arithmetic and both encoded K/V arrays bit for bit in 30 cases. The cases cover one to
256 rows, sliding and full attention, and the 8,192-token query scaling boundary.
At 256 rows, the isolated stage took 0.62–0.63 ms serially and 0.19–0.35 ms in parallel.
The 131K request was running during these host timings. Projecting the largest observed
saving over two full 48-visit blocks gives about 43 ms, roughly 0.1% of the retained
512-token prefill wall time. This is an isolated-stage calculation. The unchanged serial
loop remains in the Phlegm comparison.

| Capture under `.plans/inkling2/qkv-post-parallel/` | SHA-256 |
| --- | --- |
| `fixture.json` | `e4eac3702b03088af0d183c9c2ebebc0b9563a5950d69ad184eaf93bd9fe0d20` |
| `probe-msvc.log` | `9df15de116a67f3fe6c23d98d7a38f56e96530e30d3ae2ecf4a3da300ddd23b3` |

A source inspection on November 23 traced much of the ordinary prefill gap to the launch
structure. Every grouped expert, dense and relative-attention launch in `inkling2_npu.cpp`
packs, syncs, starts, waits, reads back, validates and unpacks before the host continues.
The host and the NPU never work at the same time. In the retained 512-token trace the
expert and dense interfaces took 21.73 s, including 11.91 s to submit and wait. The other
9.82 s was host work inside those two interfaces with the array idle. Gather, SiTU and
reduction run between the two expert launches and were not timed separately in that trace.
The block GEMM route removed the same pattern for Qwen and measured 19–22% lower
1,000-token prefill. That result also recorded a busy CPU slowing the array, so 9.82 s is an
arithmetic ceiling for these two interfaces. Submit and wait across experts, dense
projections and relative attention totaled 13.37 s. Overlap leaves that device time, which
alone exceeds the 6.42–7.61 s Vulkan prefill. [Stage-major prefill](prefill-pipeline.md)

A staged candidate applies the block GEMM schedule to Inkling. It keeps one launch in
flight, patches expert addresses only between a wait and the next start, keeps each stage
inside one hardware context, and follows each device-to-host sync with `MFENCE`. Grouped
experts run in device order U0, U1, D0, U2, D1 and so on. The host unpacks gate/up(g), runs
SiTU and stages down(g) under gate/up(g+1). It reduces down(g-1) and stages gate/up(g+2)
under down(g). Each projection leg has two input and two output buffers. The slices, expert
order and per-token accumulation order match the synchronous loop, so its gate is
bit-identical logits. Dense projections pack the next 64-row wave and scatter the previous
one under the current wave. Relative attention is unchanged. A separate toggle splits the
residual mix and the controller update across token rows. The isolated probes above project
about 1.2 s and 1.3–1.4 s for those stages over the 238 mixes and 192 updates of a
512-token prompt.

The handoff session could read and edit files. Its permissions denied Python, the compiler,
process inspection and the NPU. One bridge serves a mirrored eight-trial order with a
control, overlap with readback before or after the next launch, and overlap with the row split and vector
finite scan. The comparison checks exact logits, equal launch counts, control drift, and
Windows power and PCI events. Readback after the next launch has no precedent on this
stack. The xdna2-notes recommendations lie outside that session's readable directory and
were not re-read. The dispatch rules come from OPEN-PREFILL-BATCH, OPEN-DECODE-PIPELINE
and OPEN-REQUEST-ISOLATION in the engine spec.

| File under `.plans/inkling2/prefill-overlap/` | SHA-256 |
| --- | --- |
| `stage.py` | `39f0271d79830eb374940c1bdcdb66210c27f19c1b74db76a180d7cb0a1d5a02` |
| `compare.py` | `504c3be54303818d53858ed43d33b93eee0bf8ba8f4fd012fa3f2ac36533b333` |
| `README.md` | `f957280d303d29ceac395727adafce7de37c39472f077035d3e7b2f9b5ee07d0` |

The follow-up staged the unchanged candidate in `.plans/inkling2/prefill-overlap-verify/`
and built Phlegm's protocol-2 development bridge with MSVC, OpenMP, AVX2 and the local
Phlegm XRT import library. The bridge SHA-256 is
`89ea17795458b037a3c94616c617f79c21e27e7f500bd6218ce3437bfa103296` and its
`bridge/phlegm-reference-bench.build.json` has SHA-256
`265fc538678b6221d144b1d3e0f7c666be59f9dabee277f58d1b4b3a1325ac11`.
All eight `preflight-512/` arms checked the staged source identity, pinned bundle and
prompt, and the protocol handshake. Hardware initialization is part of `device-512/`.
The extracted grouped launch sequence passed 32,768 state-machine cases over 1–128
groups and 256 patterns of cold staging, which can retire a run early. The probe checked
that a lane was free before staging, each leg launched and read its groups in order, and
the schedule left no live lanes. XRT synchronization and output values are covered by
the device runs. `verify-schedule.cpp` has SHA-256
`6caff04122a659621013f8502171f01cb48c37c3da16c0080ad1628f8e0397e9`.

The eight device arms each warmed on 512 tokens before timing the same prompt in two
256-token blocks. All eight returned the same 200,058 FP32 logits, SHA-256
`6920e0a4a283b9ba256e87842731920c84a40c6b5219e4c3a81c1784c8daaeed`, and issued the
same 4,452 grouped launches. The logit hash also matches the separately built scalar
profiler control. Each arm recorded zero Windows power or PCI events.

| Arm | First order | Reverse order |
| --- | ---: | ---: |
| Synchronous control | 22.879 s | 24.123 s |
| Overlap with early readback | 17.048 s | 21.669 s |
| Overlap with late readback | 16.440 s | 16.692 s |
| Late readback, parallel host rows and AVX2 finite scan | 14.028 s | 14.088 s |

Control drift of 5.29% exceeded the 5% acceptance threshold. The exact-logit result
stands and the timing bracket is provisional. The early-readback arm's second trial
spent more time in host SiTU and scalar finite scanning. The full arm's two trials were
within 0.43% of each other. The repeat brackets only the full arm against the control.
The complete receipt `device-512/result.json` has SHA-256
`3b9cd0fddad03a165f428c807b2106f987a58330a7af77427b67b1058629c8b4`.

The focused repeat ran control, full, full, control on the same bridge. The controls
took 23.608 and 22.731 s, a 3.79% drift. The full candidate took 14.416 and 13.798 s,
38.9% and 39.3% less time in the respective orders. The mean fell from 23.170 to
14.107 s (39.1%). All four matched the earlier logits and 4,452 launches, with no
CPU fallbacks for attention or the head and no Windows power or PCI events. Mean MoE
time fell from 13.827 to 7.478 s, grouped finite scanning from 2.351 to 0.135 s, and
controller update from 1.719 to 0.448 s. The group timer counts blocked host waits,
which change as work overlaps. These nested stage savings cannot be summed. The
`device-512-repeat/result.json` receipt has SHA-256
`1aca568192bd677592df347fb4bbdb68073b5c536fcf8aeaf8577bf7216609a4`.

The 8,225-token comparison used one control and one full-candidate trial, each with
the same 512-token warm-up. Timed prefill fell from 510.191 to 295.963 s (42.0%), or
16.12 to 27.79 tokens/s. Both completed 32 full blocks and a 33-token tail, matched
all 200,058 final logits bit for bit, and issued 72,652 grouped launches in the same
per-block counts. Neither recorded a power or PCI event or CPU fallback for attention
or the head. MoE still accounted for 122.822 s and relative attention for 91.585 s
of the candidate's time. The 512-token repeat covers both run orders. The longer
pair covers one order and the ordinary block route through a partial final block.
The `device-8225/result.json` receipt has SHA-256
`13b0d71a858849a9dcc3a0637f4ce077f369a5ddfc5f91f6ef170894ec15e5c3`.

Of the 122.822 s in MoE, the host SiTU loop accounted for 38.778 s. The expert
down projection rounds its input to BF16 before block packing. A host AVX2
prototype approximates SiTU and replays the scalar lane when a conservative
interval crosses a BF16 rounding boundary or the inputs fall outside its range.
It matched all 59,904 active BF16 values in each saved IQ4_NL and Q8_0 group and
4,718,592 seeded finite inputs. In the single-threaded active-row replay, IQ4_NL
took 0.498 ms on scalar and 0.115 ms with the candidate. Q8_0 took 0.498 and
0.114 ms. There were 67 and 72 scalar fallbacks in the real groups respectively.
The scalar fallback also preserved all 171,648 boundary cases, including padding,
with both signs and small and large gate values. The staged Phlegm profiler
compared every value against scalar before packing on a 512-token warm-up and
the fixed 512-token prompt. It matched the known final logits and all 4,452
grouped launches with zero power or PCI events. The shadow run performs both
calculations. The same-build scalar/vector/vector/scalar bracket preserved all
logits, launched the same kernels, and recorded zero power or PCI events. The
SiTU stage fell from 2,204 and 2,091 ms to 670 and 599 ms. Wall times went
from 14.444 and 12.985 s to 13.628 and 12.668 s in the two matched orders.
The 10.6% control drift disqualifies the 2.4–5.7% apparent wall-time gain.
The scalar implementation remains the default. A full-domain error bound and
a measured wall-time gain would be needed to adopt the prototype.
`.plans/inkling2/prefill-situ/vector-wire.log` has SHA-256
`13af22a2fd60cd8d4d30892f1251f9602643e3199253cee72088e37c66a19a8f`.
`vector-wire-boundaries.log` has SHA-256
`29e08c9883525461e6236e6cd624e5b8928502c3fe41d031a214a681f1739a4d` and
`vector-shadow/result.json` has SHA-256
`7b2c0055239045c8587a1874fbac05665a68bde903673dd7afa02248c98cd517`.
`vector-pair.json` has SHA-256
`03c5ff97efa3ed4d62978b55631beafeca1fbe5f87e67fe236a5c68e38c1234c`.

The engine now enables the verified late-readback schedule on Windows. Other platforms
use the established synchronous NPU route. Both platforms parallelize independent host
rows when OpenMP is available, and the Windows overlap route uses an AVX2 finite scan
when built with AVX2. The kernel manifest and Phlegm block size are unchanged.
Both the normal and profiled Phlegm development bridges built and passed protocol
preflight. The rebuilt CPU driver passed all four focused reference tests. A canonical
engine import audit passed with 55 unchanged files and 40 matching recorded changes.
The normal and profiled bridges have SHA-256
`68a03ce003dee4c24eb49365c0e57565037dafec2e159660455b187cbac4c74d`
and `52642f37de38f69465974d6c9ca6ca31000cc1736d0203f4f5facefb4f209141`, respectively.
The integrated normal and profiled builds both reproduced the validated 512-token
logit hash on the NPU, with zero power and PCI events. The profiled build recorded
4,452 grouped launches. Prefill observations were 13.784 and 13.568 s respectively.
The earlier matched and mirrored staged experiment provides the timing comparison.
Receipts under `.plans/inkling2/integrated-prefill-profile/` are
`normal-512/result.json` SHA-256 `cc5f19b4801fa341f8f65d847a1070865d1a2aadf3db1990bbe6c4f68cf7e314`
and `profile-512/result.json` SHA-256 `1be1fe11c9642d2c6ba0aeb293c744fd76ab5367c71883c132ac1815300f6cf9`.

### Full-attention prefix reuse

For a multi-query full-attention block, the runtime now uploads each complete key tier older than
the relative-position extent once and traverses that tier for each group of 16 queries. Later tiers
still build the query-dependent relative stripe, and every query merges the same sequence of partial
softmax results in key order. This reuses host packing and device sync. The NPU still reads each tier
per query group.

A synthetic probe at depth 32,768 with 256 queries reduced key uploads from 144 to 39, a calculated
saving of 2.05 GiB of packed transfers to the shared buffer per attention visit. Warm median time
fell from 1,473 to 1,302 ms in one order and from 1,582 to 1,355 ms in reversed order. Both compiled
formats, the exact relative-position cutoff at positions 5,118 and 5,119, a wrapped sliding window
and singleton attention were covered. All seven outputs were bit-identical to the previous runtime.
The independent FP64 spot check had worst relative L2 of 3.96e-5. The full version-8 short prefill
also reproduced its logits byte-for-byte.

| Capture in `.plans/inkling2/prefix-kv-profile/` | SHA-256 |
| --- | --- |
| `0-control/result.log` | `2ae3519c795dfe557e8cdb831487c365694ee4f9259ff0e5c10e2ca747ae3f30` |
| `1-probe/result.log` | `de08ece46771cf54f68be4bb230da04da7a9bf63a5bfff98631048dac3eaf07a` |
| `2-probe/result.log` | `7c189d0dad85315fa760bdab657f9db1b7f62394fedd134f4ded0a98f89bbbc7` |
| `3-control/result.log` | `c2e74d25846e2cde7734396226fc8421b7d7bd1af5f2cc35c90f09b4fd95f9a9` |

Full-attention dispatches can also reuse a packed query across K/V tiers. For consecutive
dispatches with the same query group, the host changes only the tier header in the resident
query buffer. For interleaved groups it retains one host copy of each packed query, then copies
the relevant record and updates its header. The device still receives the same query layout and
uses the same instructions and synchronization. Every record byte matched the original packer
in both shared formats and the standalone layout. The replay included two relative cutoffs, a
wrapped sliding window, singleton decode, and the last 256 queries at 1,048,576 positions.

| Synthetic query replay | Dispatches packed | Previous host packing | With reuse |
| --- | ---: | ---: | ---: |
| 32,768 cached tokens and 256 queries, shared Q8_0 | 144 | 18.788 ms | 3.168 ms |
| 131,072 cached tokens and 256 queries, standalone | 528 | 68.661 ms | 4.971 ms |
| 1,048,320 cached tokens and 256 queries, shared IQ4_NL | 4,096 | 1,131.12 ms | 70.373 ms |

These are host packing medians with concurrent recall and reference vision work. They exclude
NPU dispatch and buffer synchronization.
`.plans/inkling2/query-pack-reuse/real-wire-layout.log` SHA-256
`22e0fbc69ed502ac9f6b5782049720d05484fa37ac300c480f7f813b10bfd174`.

On the NPU, all eight attention cases were bit-identical across the two variants and both
orders. They included the sliding-window wrap, both relative-position cutoff cases, singleton
decode, both quant formats and 33 query rows at 131,072 cached keys. The independent FP64
comparison had worst relative L2 of 3.96e-5.

| Synthetic NPU case | Control first, old / reused | Reuse first, old / reused |
| --- | ---: | ---: |
| 32,768 keys, 256 query rows | 1,160.55 / 760.92 ms | 1,156.04 / 1,128.36 ms |
| 131,072 keys, 33 query rows | 967.25 / 839.93 ms | 689.08 / 699.84 ms |
| 16,384 keys, one query row | 40.02 / 56.41 ms | 41.85 / 42.88 ms |

The large spread between orders leaves the aggregate attention speedup unproven. The device
result establishes output parity for the tested ranges. Model-level timings across contexts
are pending.

| Capture in `.plans/inkling2/query-pack-reuse/` | SHA-256 |
| --- | --- |
| `trial-0-control.log` | `86d92108ebd7d91ca49dba051dbcf998687e2e1099d137c86c86a0f8baf8c0e5` |
| `trial-1-probe.log` | `edea15d59945222877cb53fe30f2e98793590961522ba625327e05a5af62567c` |
| `trial-2-probe.log` | `c453b1a523f9d41f0529606c432eab57b160856710d2756537900bab3660d634` |
| `trial-3-control.log` | `d3ee327e92ec8bb5d0cb0bcdb97a46f974a026582fc2888b9c3c0e7fbea98c37` |

The host now merges numerators, maxima and masses directly from the device's output frames. The
previous shared path copied every head's complete tile even for singleton queries. Captured IQ4_NL,
Q8_0 and standalone results replayed bit-for-bit for 1, 2, 5 and 16 active rows and for one or
16 query groups under MSVC and Clang. The isolated shared-layout singleton merge went from
1.71–3.49 µs to 0.21–0.35 µs. A full 16-row merge went from 4.62–18.32 µs to 3.09–6.59 µs.
The timings apply to replayed host merges during the vision run. Whole-model throughput with the
change remains unmeasured. Transposing the relative bias table and hoisting cached K/V offsets
preserved all packed bytes in eight cases but had mixed timings, including some slower sliding
and cutoff cases. The engine uses the original K/V packer.

A separate host prototype streams complete old tiers into the existing v11 row-major wire,
leaving the unused relative stripe at its original contents. The old-range cutoff uses the last
key. Partial, uncommitted, wraparound and near-relative ranges follow the original packer.
The AVX2 version uses non-temporal stores to aligned buffers and fences each writer before the
dispatcher can sync. A portable path handles unaligned output. The benchmark extracted the
current C++ packer as the control and checked the entire destination, including the untouched
relative stripe. The independent builds also exercised an unaligned source range and destination.

| Complete old tier | Control, MSVC 6 threads | Streamed, MSVC 6 threads | Median paired speedup |
| --- | ---: | ---: | ---: |
| 4,096 keys at the cutoff | 2.025 ms | 0.485 ms | 4.13× |
| 4,096 keys at a nonzero offset | 2.106 ms | 0.521 ms | 3.98× |

At three threads the paired gains were 3.63× and 3.45×. The portable Clang path reproduced
the bytes. These measurements cover host packing only, with the longer Phlegm request active.
The isolated Phlegm build enables the full 4,096-key path with an experiment flag and uses the
existing v11 bundle. Protocol 2 handshake and source audit passed. The bridge differs from
the running reference in the NPU source and packing helper only. Its SHA-256 is
`77cc59714c30ba4875317cb7e1ae1db3fb79361c905fbd287b399beb3cf5b01d`.
The device comparison reuses the same compiled projector in both modes and checks that
the streaming path actually ran. It requires bit-identical outputs across both orders,
including full attention at 131K, a sliding wrap and the two sides of the relative cutoff.
It runs with the other attention probes before the next long recall. All eight checks passed
preflight. `f32-phlegm/priority-preflight-v11-pack-20261123/results.jsonl` has SHA-256
`e809f9d2768eaea2b26b3667a18f09cd645293e6ac6d3711498a8314ddd7940c`.
`kv-pack-study/v11-old-fast-avx2-6.log` has SHA-256
`7b9b473d58c343b0a4f33d542b6ccb042579f1d7d6733b0b668d5dc8eb73a523`.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `attention-merge/merge-final-iq4.log` | `c370bc58b33d359708e5cfdeabf9133a9a326f5b4dc4bf96b3724255958b53e7` |
| `attention-merge/merge-final-q8.log` | `5591574d8dfd92e0042d2faafa9bc84454406fafb4078d58106a5e744d8e7060` |
| `attention-merge/merge-clang-final-iq4.log` | `f73ef86f57ed9146d9850845dd4a145523536df9cb83c26116e893d66c0b8b10` |
| `attention-merge/merge-clang-final-q8.log` | `34a26b48ac04c7d333908f3b8f855983620b5615514249de586453f0aeb16b2d` |
| `kv-pack-study/pack.log` | `a7fe0e5017f38f363c9c602c9815446439d513c72a58affc4cff3a36ce65b361` |
| `kv-pack-study/pack-clang.log` | `32b346c81a10e0ed13d201d3e77ea43a9bfd3aa9a6046549cdfb9d2cfe1b2a91` |

An offline numerator prototype shares each probability broadcast across two 32-channel
accumulators. Fresh Peano objects used 2,304 B of text for the control's chunk function and
2,352 B for the prototype. Both had a 1,216 B function frame, and all 392 generated FIFO,
buffer and DMA records agreed. The first packaging attempts could not discover `aiebu-asm`, and
WSL reported its configured disk image was missing. A read of the native Windows XRT SDK found
both packaging executables. Workspace-local copies successfully packaged separate 128-chunk
control and prototype images. Their instruction stream and control ELF matched byte for byte.
Across all 24 linked cores, maximum program occupancy was 3,776 B for the control and 3,824 B
for the prototype. The L1 high-water mark including the reserved stack and linked data was
52,576 B in both. The prototype's hardware output and timing checks remain open.
[Windows XRT SDK setup](https://github.com/Xilinx/mlir-aie/blob/main/docs/buildHostWinNative.md#3-install-the-windows-xrt-sdk)

`.plans/inkling2/attention-numerator/static-resources.json` SHA-256
`d7d9c27614e7c0dff2d4d822578d6aea3d182161fa2ec624b42988f9417f291d`.
Packaged per-core resource report `.plans/inkling2/attention-numerator/packaged-resources.json`
SHA-256 `666df8b51956bf012021eeb19cf228886fa0ad8963c2a4eba5ddcf20c061dcea`.

| Packaged artifact | SHA-256 |
| --- | --- |
| Control `final.xclbin` | `d44b6de1b0732e3e82c44d1cacb45990de09efb0b3971e875f665738275377ed` |
| Prototype `final.xclbin` | `2679262633383226f9519b503514c6b614d95e111283356dc71a3a03f7938ab0` |
| Both `insts.bin` | `69c21afb6fcd574c7768cab441e2f54920c5a9739ede97277ea7fa10a4473b94` |
| Both `insts.elf` | `9ccf692b286ad397cca80edeae2f3deee114783409a9272bf2774deddc02d3a4` |

A separate score prototype reuses the split query loads for two adjacent keys in the old full-attention
prefix. Each key retains its original accumulation order. The fast path requires all 32 keys in the
chunk to be present and outside the relative-position extent for every query. Sliding-window, relative
and partial chunks use the original path. The 128-chunk packages compiled for standalone attention and
both shared expert layouts. Each matched the control's `insts.bin`, control ELF and placed FIFO,
buffer and DMA records. The linked data ends include the reserved stack and final read-only data.

| 128-chunk image | Workers | Control text, max | Candidate text, max | Linked data end, both |
| --- | ---: | ---: | ---: | ---: |
| Standalone | 24 | 3,776 B | 4,560 B | 52,636 B |
| IQ4_NL shared | 32 | 9,184 B | 9,968 B | 59,996 B |
| Q8_0 shared | 32 | 8,320 B | 9,104 B | 61,532 B |

The retained FP64 fixture covers 128, 95 and zero accelerated chunks in its three ranges, including
one range that crosses the relative-position cutoff. The paired harness verified artifact and input
provenance offline. Device parity and speed remain unmeasured, and the production kernel still uses the
original score loop. `.plans/inkling2/attention-score-pair/resources.json` SHA-256
`3951455f53dbac87e523e6a13ce0094e8fb4ce4e11bc919bcd674ca75d46ff36`.

### Phlegm text integration

The phlegm working tree at base `4aa1493199807737d9cb7c0fa89b489e3d5c9dbb` accepts the Inkling bundle
through its resident C++ bridge. Rust reads the `inkling2` pre-tokenizer from the GGUF, keeps the publisher
template's whitespace controls around its training markers, and splits text and thinking frames for CLI
and HTTP responses. Seven tokenizer cases, including two rendered conversations and multilingual text,
matched the pinned llama.cpp token IDs, round trips and stop tokens. Corpus output
`.plans/inkling2/phlegm-token-parity.json` SHA-256 `9902e8b17a663e25dd2f979c2f93b8d6b0f66cd6586ebc8b35937276bc0a8db2`.
The sequential bridge smoke exercised prefill, decode, a greedy timing trial and reproducible reset.
Trace `.plans/inkling2/phlegm-bridge-smoke.log` SHA-256 `0d528f39b4f4e74bba095a5aa8fba51a6dd9561e5b852e19f4314739d09b4e0c`.

`phlegm bench` now reads the embedded GGUF encoder when the model directory has no `tokenizer.json`.
The benchmark report fingerprints that source using the same GGUF SHA-256 already recorded for the
weights. An existing `tokenizer.json` retains precedence and its own hash. The tokenizer, reference
and compiled-bridge tests passed. The full benchmark command stopped at Phlegm's requirement for the
exact `v<Cargo version>` release tag, so release benchmark data remain pending.

The phlegm tokenizer renders the publisher's tool JSON with sorted keys and the specified separators.
The reply parser recognizes named Inkling model frames, keeps reasoning separate, and buffers possible
tool frames until valid JSON and the matching recipient have been checked. Unit tests cover split headers,
two calls, the terminator spelled inside a JSON argument and malformed frames.

On the version-6 bundle, a greedy `phlegm serve` request with reasoning effort `none` returned
`finish_reason=tool_calls` and the call `lookup({"key":"ember"})`. Its streamed equivalent returned
the same call without emitting protocol markers as content. Sending the assistant message and a tool
result identified by its call ID produced `CRIMSON-42` with `finish_reason=stop`. The tool result
provided the value at runtime.

The real GGUF template rendered the phlegm effort ladder at 0, 0.1, 0.2, 0.7, 0.9 and 0.99 with a
default of 0.9. Rendering via template kwargs accepted the publisher's `max` alias and a numeric 0.42,
and rejected 1.0. The version-6 server returned `OK` with a separate thinking frame at `low`,
`high` and `max` on the short greedy prompt, producing 41, 53 and 72 completion tokens respectively.
The `max` alias was passed in `chat_template_kwargs`. The original CLI and server smoke covers `none`.

| Capture in `.plans/inkling2/phlegm-tools/` | SHA-256 |
| --- | --- |
| `effort.jsonl` | `25140f7c4d7a6134e45fed4ea7234879381ae20df8be4d2ed9a1c4a15dbf6134` |
| `live-effort.jsonl` | `21c410bb6a4c6b1e6c291003bed97fdb2d23da25dd6964f25c0cecd6a1477fa3` |
| `invoke.json` | `adc50db7aeff0cbe09c2bc6c7a5d8270b441c5675980a0bf477ff5938f9096ee` |
| `invoke-stream.sse` | `e6758050688ca567c29b342852dd3dd9a9ae97dc695454761a9547d2ec5c9c0b` |
| `tool-result.json` | `54470bc3209012d8de18d0202b9a3e0808acf67eee1b3d034ef072c73bf06449` |

A hardware-free replay fed scripted, real Inkling tokenizer IDs to the development Phlegm
server. Eight HTTP checks passed. JSON and SSE preserved separate thinking and text, Unicode,
two typed tool argument objects, call order, and distinct call IDs. One argument contained the
literal end-message marker. Plain and streamed follow-ups supplied both results in reverse
order. The rendered template assigned `AZURE-73` to `convert` and `EMBER-41` to `lookup` by
call ID. The server accepted the selected named call, withheld a mismatched structured call,
rejected an undeclared tool choice with HTTP 400, and honored `tool_choice=none`. The three
existing Inkling Rust tokenizer and parser tests also passed. Live multi-call model behavior
and the release build still need validation.

| Capture in `.plans/inkling2/phlegm-tools/replay-verified/` | SHA-256 |
| --- | --- |
| `server.log` | `9707f725bc2d00f84ae952cadbbb645f1626b94fc02fd0bc7f28f491c40661d9` |
| `parallel.sse` | `3a4714aa4d8de507fb9faa4c8de3b8ff533cf62949c06626f195dd1298ccc9c0` |
| `response.sse.request.json` | `f7832d13240a77585dbf6b2735170f34938e302af4e444d143322ad37f8cfc6b` |
| `response.sse` | `96544d3664da561761d1d3b53e7c5a8674c81e9b35dbc65046e0da72718943bb` |

`phlegm run` accepted `--max-ctx 1048576` and answered `OK` on a short prompt. This checks the app's
capacity configuration. `phlegm serve` returned `OK` with separate `reasoning_content` and `finish_reason=stop`
for a 20-token prompt and 45-token completion. Repeating the request via SSE returned the same answer and reasoning.
Both paths used host block prefill, NPU expert decode and the NPU Q8 head. The server test exercised a request
with `reasoning_effort=none`. The template emitted effort 0 and the model still generated a thinking frame.
The complete text inference and feature matrix remains open.

With the version-3 bundle and rebuilt bridge, `phlegm run --prefill-route block` also returned `OK` with a
separate thinking frame. The bridge reports the layer-major route as `mixed` when the expert blocks are
available. CLI output `.plans/inkling2/phlegm-run-block.out` SHA-256
`c7a1ba7a18c0ba7c5d1c47c9251171ea565cb782f1e4f607e5cd80b0875f67c5`.
Plain HTTP and SSE returned the same `OK`, thinking frame and stop reason through the resident bridge with
the version-3 bundle. The non-streamed request used 20 prompt tokens and 42 completion tokens.

| Version-3 app capture in `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `phlegm-block-inkling-response.json` | `4fecd41ef18e6e317037171ad51b16f83f3887f8c72c8b896fe409dc4ec6d118` |
| `phlegm-block-inkling-stream.txt` | `71b6d27b5e0ea6b5533581d64c76aaf83417547947370cfae862855e14bf4b42` |

The rebuilt phlegm bridge ran the same 128 WikiText-2 tokens with each bundle in its own resident process,
six host threads and passive OpenMP waiting. It reset sequence state between trials and retained the packed
weights. Both bundles used the NPU Q8 head. The cold trial includes on-demand preparation. The warm trials
show 1.73× faster prefill with NPU experts. Each route reproduced its logits bit-for-bit across all three trials.
The best token and set of top five agreed between routes, with ranks four and five exchanged and KL of
0.002798 nats from host to NPU expert prefill.

| Prefill experts | Cold | Warm 1 | Warm 2 |
| --- | ---: | ---: | ---: |
| NPU IQ4_NL + Q8_0 | 41.83 s | 24.01 s | 23.89 s |
| Host IQ4_NL + Q8_0 | 42.80 s | 41.38 s | 41.31 s |

Timing trace `.plans/inkling2/phlegm-block-128.jsonl` SHA-256
`55c9cff638362b317c6746b274d6d2f437f84bc28e718336fc762e529fca68fc`.

`phlegm run` with the version-4 bundle and `--max-ctx 1048576` returned `OK` on the same short prompt.
Plain HTTP and SSE again agreed on the answer, thinking frame and stop reason at the short test capacity.
The version-4 captures `.plans/inkling2/phlegm-compact-inkling-response.json` and
`.plans/inkling2/phlegm-compact-inkling-stream.txt` have SHA-256
`10af2590b7df0bf7bdbeb84831b690ba77f61cd624f3852655ef5b99d95c5031` and
`2bc4093c7ecb6938710e04ae5c611ce8680dca61d6456bf21d3b01c6b27c66a5`.

`phlegm run` and plain HTTP and SSE also returned `OK` with the version-5 attention bundle. Both HTTP paths
agreed on the answer, thinking frame and stop reason. The version-5 captures
`.plans/inkling2/phlegm-attention-inkling-response.json` and `.plans/inkling2/phlegm-attention-inkling-stream.txt`
have SHA-256 `cfd225fcd29d20d8bad12d7a34918366342920709defc63f4f5483dda7d44529` and
`8b1de01b9fbca2758f9347f72cb297eb5c05e8d1b3bfa6d6bd581a1b4d5d8a14`.

The version-6 bundle also passed `phlegm run --max-ctx 1048576` and both HTTP modes on the short `OK` case.
The version-6 captures `.plans/inkling2/phlegm-shared-inkling-response.json` and
`.plans/inkling2/phlegm-shared-inkling-stream.txt` have SHA-256
`fc98c2e37feb44cbb5623e3b0449b31089d62e787470acb6bd06f872ec9e4c98` and
`2ed4ebbd85db6f0a1b4fcdcbcb94e01e93b578b58fee6c74a2d5a02adaa0f122`.

Version 7 passed the same CLI and HTTP checks. The CLI accepted the configured million-token capacity
on the short prompt. Plain HTTP and SSE returned `OK` with the same separate thinking text and
`finish_reason=stop`. The response used 20 prompt tokens and 42 completion tokens.

| Version-7 capture in `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `phlegm-run-grouped.out` | `a12b7cb43c9d9134b5bb1b35e9096b66775d9e92e7611d1cc92b02edd6782a87` |
| `phlegm-grouped-inkling-response.json` | `09e37c9372641c67d18770ef6325267727bc6b186123ea535763a072347f4178` |
| `phlegm-grouped-inkling-stream.txt` | `97cd6aa86aca0c385b815ff96024157f280866403764ef214d7840f8d31ac44d` |

Version 8 passed the same `phlegm run` smoke at the configured million-token capacity. Non-streaming
and streaming `phlegm serve` both returned `OK` with the same separate thinking text and stop reason
after 20 prompt tokens and 38 completion tokens. The version-8 response and stream in `.plans/inkling2/`
are `phlegm-dense-precise-inkling-response.json` and `phlegm-dense-precise-inkling-stream.txt`, with SHA-256
`17b3f1fbae194c001d73e1eaf90e62a74c21f5bfccdad095179517cb6cd8637e` and
`29e2c04cfafd3055b7454236ec2597a0f0172b0459edc2c0c05da6506ad2d86a`.

The first 8,192-token phlegm recall request failed at the 600-second reference-bridge request timeout.
The bridge had one response outstanding for the entire prompt. The Rust backend now uses the
Inkling bridge's advertised prefill chunk size, checks its committed position after each block, and
requests logits on the final block. Each acknowledgement gets the normal per-request timeout.
The original HTTP error at `.plans/inkling2/needle-8k/8k-depth-50.response-error.json` has SHA-256
`3b91d9a086d132ea75ff26b1050c8b4e24bdf70549e26e68cec5fee383f47c88`.
The rebuilt bridge produced byte-identical logits when a 513-token prompt was sent in one request
and when it was sent in three requests of 256 + 256 + 1 tokens, and again on the next fixed decode
token. Both used internal blocks of at most 256 tokens. The intermediate replies acknowledged
positions 256 and 512 without returning logits. The capture at
`.plans/inkling2/phlegm-chunked-prefill/protocol.json` has
SHA-256 `223fc56b81681ffd38c21cfd73ae30213428821ef4ed10e7ff140a13ade87562`.

The retry through `phlegm serve` returned `AURIC-7492` from token 4,110 of the 8,192-token prompt. The
version-8 bundle, default 600-second per-request timeout, temperature 0 and reasoning effort `none`
produced a `stop` response in 836.97 s including 89 completion tokens. A mid-request bridge sample showed
16.13 GiB of shared adapter memory. The 10% and 90% runs used the same build and settings. All three fully
prefilled their prompts and returned exactly the requested key. The edge runs overlapped local vision
checks on the host. The wall times are application observations for these prompts.

| Requested depth | Prompt tokens | Needle token | Key | Wall time (s) |
| ---: | ---: | ---: | --- | ---: |
| 10% | 8,191 | 832 | `OPAL-6827` | 699.33 |
| 50% | 8,192 | 4,110 | `AURIC-7492` | 836.97 |
| 90% | 8,192 | 7,384 | `TOPAZ-5913` | 790.73 |

| Recall artifact in `.plans/inkling2/needle-8k/` | SHA-256 |
| --- | --- |
| `8k-depth-50.json` | `ebf8de7ac5faa7527a7972225acf6e7ed9b7cab1f5cea1b3e78de87e0b063dd8` |
| `8k-depth-50.chunked-response.json` | `fbe4e4d8e44a0403aca08750d24c0d903abb8ecd292bb20748df814845755695` |
| `8k-depth-50.chunked-summary.json` | `b0ace9acb90d78a5e25e8b16235e5c6f45eff97e718ba186fae5d801b3b10dbb` |
| `8k-depth-10.chunked-summary.json` | `1ea36860d4235a86e97feb7dff4df093179aa0bd9aaaf2611bc2eeb2801e865f` |
| `8k-depth-90.chunked-summary.json` | `02dfea2926feb2a86481e92e4b371146f75a9cdd468bd1c2f9ad5d3f1ef2d33e` |

The version-8 `phlegm serve` run at 32K returned the exact needle `CITRINE-9274` at token 16,397
of a 32,766-token prompt. It stopped normally after 97 completion tokens and took 2,942.19 s
end to end on the same Ryzen AI 9 HX PRO 370. CPU oracle work overlapped part of the run.
The version-8 10% run returned `BERYL-5927` at token 3,290 of a 32,766-token prompt. It stopped
normally after 86 completion tokens and took 3,689.06 s. CPU and Vulkan development work overlapped
the run. A WDDM process counter showed 17,324,630,016 shared adapter bytes in flight. That point
sample has no measured token boundary or peak claim.

The v11 `phlegm serve` run at 90% depth returned `ZIRCON-1824` exactly from token 29,505 of a
32,767-token prompt. It used speculation off, had zero cached prompt tokens, stopped after 94
output tokens, and took 3,889.78 s. The largest available WDDM process sample during the latter
part of the run was 17,843,675,136 shared adapter bytes (16.62 GiB). CPU and Vulkan development
work overlapped the run, and the elapsed time measures this request under that load. All three
32K depths have now passed across v8 and v11. Local artifacts in `.plans/inkling2/needle-32k/`:

| Recall artifact | SHA-256 |
| --- | --- |
| `32k-depth-50.json` | `137039db75fbf8a3d472afc167b811b6a93d7f612c512e0e165ea72245dca976` |
| `32k-depth-50.chunked-summary.json` | `e987ff739e0497cde91d9d0612ba061d0292724f561de0fa83d564ee3e8541ef` |
| `32k-depth-50.chunked-response.json` | `b47b964beeed4cb4827b76bb29f0401defc2ef5632aefc0596d4a4bfd41dce28` |
| `32k-depth-10.json` | `2a64063b24bbe31fe72eb20ff13b07824cd12904adbb9801e473f2c3606e31cb` |
| `32k-depth-10.chunked-summary.json` | `227d95832c03432f2f5424f4724301d8edcba6273d8aeedf849e9d867ce708ae` |
| `32k-depth-10.chunked-response.json` | `678e321233bee87f4a0295367a8f869fb137abb95df8050f83f01b09eaba59e8` |
| `depth-10-memory-snapshot.json` | `0be41f91b7a2f2c58b50e13a227e332758620633c0202e148cbc2caeb2dc0a5e` |
| `32k-depth-90.json` | `31539be01cb4fda76cbcaf5e65ee0103e1e37b43f972a0038de982acb729b7ef` |
| `32k-depth-90.chunked-summary.json` | `d6a4a7e20ae3fd9113521e12fdc2982542ad4763b316650c1074d98523cc24d6` |
| `32k-depth-90.chunked-response.json` | `214d8b1ee608737c27ba09cd78b0a544655dda35c175752862b8b8b84357d9d8` |
| `depth-90-memory-series.jsonl` | `7dfefbba606f292064f55a38f7e37a986279087bfe5299a19d2eaf061f8303b4` |

The 131K recall fixtures use Phlegm's real GGUF tokenizer and chat template. Actual prompt counts are
131,071, 131,072 and 131,071 tokens, and each unique key begins within 0.05 percentage points of its
target depth. The 131,072-token midpoint NPU request returned exactly `JASPER-6942` from token
65,547 and stopped normally after 89 generated tokens. Phlegm reported 131,072 prompt tokens.
The development v11 query-cache bridge took 12 h 59 min 55 s from request to response on the
Ryzen AI 9 HX PRO 370, ending at 09:08 CST on November 23. Across 285 memory samples, the maximum
bridge working set was 34.54 GiB, maximum private bytes were 20.91 GiB, and maximum shared adapter
memory was 16.62 GiB. The sampler has two gaps, 21:15:51–00:54:36 and 02:58:45–07:32:41.
Windows logged Modern Standby exits within both gaps, and the bridge accumulated only
69.44 seconds of CPU time across them. The gaps cover 8 h 12 min 40 s, leaving 4 h 47 min
15 s of wall time outside the gaps. The uninterrupted prefill time and continuous memory peak
are unknown. The midpoint passes correctness.

The live 10% retry acquired a Windows idle-sleep request through a helper tied to the running
recall process. It was on AC at acquisition. This retry had already crossed one Modern Standby
interval from 12:00:52 to 12:03:02 CST. Subsequent recall runs hold their own scoped request and
save the exact request's Windows power and PCI events with the result. The commit observer now
records when it first sees the full prompt commit, separately from response wall time. The bridge
log is polled every 15 seconds. [Windows power requests](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest)

Windows logged another standby entry at 12:29:46 CST (System record 188592) while the retry held
the request. The two bridge samples spanning that event arrived 61.07 seconds apart, and the
bridge used 31.09 CPU seconds between them. The event query from 12:03:03 through 12:47:36 found
no PCI context errors. The retry reached 65,536 of 131,071 prompt tokens in 1 h 33 min 49 s of
elapsed time. The 16,384 tokens leading to that point took 1,531 seconds (10.70 tokens/s).
These interim observations come from `f32-phlegm/recall-1m.stdout.log` and
`f32-phlegm/recall-131k-155032/runtime-111956-memory.jsonl` under `.plans/inkling2/`.
The two subsequent 8,192-token intervals reached 73,728 at 9.92 tokens/s and 81,920 at
8.95 tokens/s. The system log recorded standby exit 188598 at 13:05:31. The bridge samples
spanning it remained about 61 seconds apart with 26.25 CPU seconds between them. The query
from 12:47:36 through 13:20:06 found zero PCI context errors. These interim rates include all
observed request elapsed time in each interval.

The same request reached 108,800 of 131,071 prompt tokens at 14:12 CST. Between the
81,920 and 90,112 commit observations it ran at 8.95 tokens/s, followed by 8.03 tokens/s
through 98,304 and 7.86 tokens/s through 108,800. Each sampled interval covers 32–41
consecutive blocks. From 13:06 through 14:12, Windows logged zero matching power or PCI
events and the largest of the 65 bridge sample gaps was 61.29 seconds. These are elapsed
Phlegm throughput measurements over partial windows. Commit observation granularity is
15 seconds.

Windows logged two additional standby entry/exit pairs near 14:47 CST, lasting about 3 and
29 seconds. The bridge samples around both pairs stayed about 61 seconds apart and still
accumulated CPU time. The request reached 126,464 tokens at 14:50 CST with no PCI context
errors recorded from 14:12 through 14:50.

The 10% retry returned exactly `AGATE-8113` from token 13,121 and stopped normally. Phlegm
reported 131,071 prompt tokens, zero cached tokens, and 80 generated tokens. The trace
contains 512 prefill commits and first showed the full prompt at 3 h 47 min 9 s. The
response completed at 3 h 49 min 58 s. Both timings include the recorded standby intervals.
Across 227 samples, peak bridge working set was 32.05 GiB, peak private bytes were 20.90 GiB,
and peak shared adapter memory was 16.62 GiB. The process-window query found eight power
events and zero PCI context errors. Uninterrupted prefill timing remains open.

| Capture in `.plans/inkling2/f32-phlegm/recall-131k-155032/` | SHA-256 |
| --- | --- |
| `late-window.baseline.json` | `c31c2f90fce09eff2c9b606ecc525ab9516628f75d3d909b16b679674248242d` |
| `late-window.events.json` | `68e8ac45d082493cbe12769d4717a40c3cdc7d29e024947fbe9536c6bc94c70b` |
| `late-standby.events.json` | `51cf8931985b5a9bc6b0db4def2f955d5650bd1272812132e2d405481a4681de` |
| `131k-depth-10.chunked-summary.json` | `f001e228e2c5df38304c0eaf10c8e363fbe091d3296dfb2a7a8712213540f902` |
| `131k-depth-10.chunked-response.json` | `adfac356997b6ce0fce0ad71c85f8027591222537fcb58ac1ccd3dfaad212ea2` |
| `needle-chunked-serve.log` | `808ed47e168f1a26aa78b341fced5d700bc1cc98a2ff6a53e2b1c1ff2ca847bb` |
| `runtime-111956-memory.jsonl` | `73aa6868cf903a2f7ea0de1af5d2db157d64bbf22f60767f428892ed527b500c` |
| `process-window.events.json` | `24e5d7d4d8967a2d45438a62b8a43ebd4a21f5f0232d898f78b636076396983e` |

| Capture in `.plans/inkling2/needle-131k/` | SHA-256 |
| --- | --- |
| `cases.json` | `c0405585163f876fa1f2f9ab112f7d2fc143c7497b9f3562de83d6ad11daa7a0` |
| `131k-depth-50.chunked-summary.json` | `0f59bc6bb7267da35ff152362570f25d432a47bb3ffda359f81d2ca56ffa717a` |
| `131k-depth-50.chunked-response.json` | `d61bf135f03e8df339c2425d826a293284e4ffaf2799da6e1776f9f607454659` |
| `runtime-128996-memory.jsonl` | `d602be1fb8bcf2ff7254f9feaf798607bb0f6d6d9969475a03ec8b1ccc1555a8` |
| `midpoint-windows-events.json` | `430bcd662554b9fc97e54169942c25944c177f1192ce4108d56dcada633b5f96` |
| `midpoint-power-audit.json` | `45e9b12a221f649688df2d9c9546622f41741373307326cb8abbea9ba5d0f817` |

The midpoint full-context fixture has 1,048,449 rendered prompt tokens and a 128-token output budget
with Phlegm's GGUF tokenizer. The unique key `ZEBRA-6291` begins near token 524,238. It uses repeated
WikiText-2 filler to reach capacity. `.plans/inkling2/needle-1m/1m-depth-50.json` SHA-256
`98e0474a98af97b9239f40de2620d9baad62b6270a41a5f0a4d74d3b3957bdb6`. The full-context hardware run is pending.

A native storage replay used the production `commit`, `accept` and `reset` bodies with synthetic
K/V and convolution rows. At capacity 131,328 it exercised all 48 visits, including eight full
caches and 40 sliding rings. It checked sampled K/V rows and all convolution lanes, rejected
and partially accepted a speculative block, and filled the final available positions. An
injected allocation failure during reserve preserved the old state and allowed the smaller
accepted prefix to commit on retry. At the 131,328 boundary the target K/V vectors held
4.164 GiB and the process peak working set was 4.466 GiB.

A second storage replay reached 1,048,576 positions with one actual-width full-attention visit,
the 40 sliding rings and seven one-row placeholders. It checked the 131K and 524K offsets,
the final eight speculative positions, the wrap boundaries and storage release on reset.
The isolated run held 4.156 GiB of K/V. The complete target layout is 32.156 GiB of K/V at
that length. DSpark adds 4.008 GiB for one full visit and two sliding rings, bringing the
combined K/V payload to 36.164 GiB. These figures exclude weights, temporary buffers and XRT
allocations. The actual 1M Phlegm run still has to establish total residency and model recall.

| Storage replay under `.plans/inkling2/kv-state-boundary/` | SHA-256 |
| --- | --- |
| `state-131k.log` | `2ac2be867b0050eb98d66fd62bd706687fe762a1d430bfeaba20522f83dd5ad3` |
| `state-1m-one.log` | `fafcf27c780b74b331e46dc4d2e54e35bf2bb95963cfd424dcbfd27a2edb16c5` |

On 64-bit Windows, full-attention target caches configured for at least 65,536 tokens now
reserve a contiguous address range and commit the needed pages as tokens arrive. The existing
transaction still prepares all visits before changing accepted K/V or convolution state. Short
caches and sliding rings keep their vector storage.

An ordered 131,328-token storage replay covered all 48 visits, including injected reserve failure
and retry, window rollover, reject, partial accept and reset. The full-attention buffers moved
160 times under vector growth and kept stable addresses with page-backed storage, eliminating
about 8.0 GiB of old-row copies in this replay. The process peak working set fell from 4.47 to
4.24 GiB. Timed cache commits took 4.77–4.99 s before and 2.88–2.94 s after. These isolated
storage timings overlapped the original 131K recall.

At a declared 1,048,576-token capacity, all sixteen full-cache lanes reserved 32 GiB of
address space. The first 32 tokens committed 1 MiB across those lanes, and reset released all
sixteen ranges. A separate full-boundary storage replay used one full-attention visit and
all 40 normal sliding rings. At position 655,360 its private bytes fell from 4.22 to 2.72 GiB.
The two storage backends preserved the exact K/V and convolution values through position
1,048,576. The real IQ4_NL model, also configured for 1,048,576, reproduced all 1,000,290
logits and five greedy IDs on the fixed 32-token input. The rebuilt native verifier passed
all nine acceptance lengths with identical logits and DSpark features. The traced Phlegm
bridge built and prepared the remaining 131K depths and the 1M fixture. Real full-context
residency, recall and wall time remain open.

The traced bridge was rebuilt with the measured F32 projection change. Its provenance differed
from the preceding page-backed bridge only in `inkling2_cpu.hpp`. The rebuilt bridge passed the
protocol handshake and prepared both remaining 131K depths and the 1M fixture. A same-bridge
comparison against the v11 attention route passed preflight at 8,225 and 17,441 prompt tokens
with 32 forced decode steps. The separate older-key stream is scheduled for 784 and 4,112 launches
respectively, with a 33-row tail in both cases. The 8,225-token comparison stopped on the
baseline at `blk.12.dense-fused` during prefill. The alternate older-key route was unexercised.
The separate 512-token finer-tier comparison also stopped on its baseline at
`blk.17.ffn_gate_up_exps.weight`. Both dispatches reported XRT ERT state 8. These failures
supplied no comparison timings or evidence for replacing production streams.

The 131K 10% HTTP recall uses the traced bridge with the established v11 attention bundle.
The 90% recall uses the same bridge and tests the candidate bundle if the shorter pairs pass.
The runner verifies the full prompt commit sequence and requires zero
cached prompt tokens, the exact response and a normal stop. Progress now records elapsed time
at each sampled commit. Both 131K fixtures and the million-token fixture passed preparation
with a one-hour timeout per bridge request. The HTTP deadlines remain 24 hours per 131K
request and 14 days for 1M.

The first 131K 10% attempt returned HTTP 500 at `blk.11.ffn_gate_up_exps.weight` (XRT ERT state 8)
before a commit marker appeared. That runner exited before the 90% case. The queued recovery
starts a fresh Phlegm server for each depth and retains partial commits and exit codes. The 1M
runner checks the speed and 131K receipts before opening a server. The response under
`f32-phlegm/recall-131k-157444/131k-depth-10.chunked-response-error.json`
has SHA-256 `ccc83534f6badd08f51526dd8b88ebaf5a5ab6eb1a5a2f0258867b2efdc108fa`.

The 10% case finished before the next device slot. That slot runs eight short performance
probes and then the two Phlegm model comparisons. The speed decision
is recorded before the DSpark rollback matrix, separate fresh-server DSpark image requests at
1,792 and 4,096 px, and the source-rebuilt bundle smoke. The 90% case starts after those feature
tests. Performance and feature checks retain independent logs, exit status and Windows events.
The selected bundle's digest is checked again before the 90% server starts. Both phases passed
preparation. Simulated handoffs verified the order for accepted and rejected speed results and
rejected a bundle changed between the speed decision and the 90% request.

All four feature checks finished on November 23. The 90% HTTP needle started at
17:06 CST with the established v11 bundle after the older-key speed candidate was
rejected. Phlegm returned exactly `ONYX-3179` from token 117,978 and stopped after
80 generated tokens. The trace records 512 block commits for 131,071 prompt tokens.
The observer saw the completed prompt at 4 h 6 min 26 s, and the response finished
at 4 h 9 min 38 s. The Windows event log has three power events and zero PCI context
errors. The exact answer qualifies as a functional pass. The observed durations are
excluded from clean performance results. The 1M runner held the request because the
faster attention route had no qualifying timing bracket.

| Capture under `.plans/inkling2/f32-phlegm/recall-131k-160824/` | SHA-256 |
| --- | --- |
| `131k-depth-90.chunked-summary.json` | `3725702d89e29708aab79d06b503de868cc0b1dca3041ce7991e1ea68241d053` |
| `131k-depth-90.windows-events.json` | `957de09ea21177b86c5e2935153627ab554f728c97028412afbc03c9c219fdd1` |

| Capture under `.plans/inkling2/f32-phlegm/` | SHA-256 |
| --- | --- |
| `priority-paired-keys-preflight/results.jsonl` | `803017c7d1219a14274ef14b922ef328503ad3ceb7efe829143b0f379374fdb7` |
| `priority-phased-feature-preflight/results.jsonl` | `c93605a5cdf866073e156c7ec29767e43b04397a5912e878a6c38221d9f0c667` |
| `priority-phased-handoff.log` | `95475509dd06e2ec5d193a71aa1afce2334856fa4b323e0b8810cc14404e91dc` |

The full 1,048,449-token HTTP recall is queued after the existing device runs. It uses the
existing traced bridge, the same midpoint fixture and the full 1,048,576-token capacity.
Both bundles passed preparation. Before opening the 131K 90% HTTP request, the managed runner gives the
8,225-token comparison one fresh attempt if it has no usable result. That fixture checks
older-key output parity. After it passes, the 18,465-token fixture measures four complete
256-query blocks starting at position 17,408. Preflight predicts 32 full-attention visits using
the 16K old-key tier in that window. The earlier 17,441-token fixture exercises the 16K tier in
its 33-row tail. Both current fixtures must preserve every greedy ID and keep maximum
full-vocabulary KL at or below 0.01 nats. The longer fixture also requires a minimum 5% prefill
saving in both orders across the four full blocks and no more than 5% control drift. Full-prefix
and late-window timings are recorded separately.
The short fixture evaluates one full prompt and 32 forced decode steps per route. It has a
numerical acceptance gate. Together with the longer prefill preview, this brings the planned
qualification to 57,476 evaluated prefill rows, down from 106,760 before warm-up.

The actual 8,225-token comparison passed its quality gate. All 33 greedy IDs matched.
An independent FP64 replay of both logit tapes confirmed mean KL of `0.001654` and
maximum KL of `0.009754` nats. The candidate trace contained all 104 older-key visits and
784 launches with a total 8.767 s of device dispatch. Windows logged no power or PCI
events over the comparison. In the single control-then-candidate order, prefill took
547.73 s and 671.14 s respectively. The candidate's elapsed time was 22.5% higher.
The comparison has one timing sample per arm and qualifies numerical behavior only.
The following 18,465-token preview started the same-state, bracketed speed comparison.

| Capture under `.plans/inkling2/f32-phlegm/` | SHA-256 |
| --- | --- |
| `old-range-8225-result.json` | `a0bb35f4075311b0e558c9a0bc13a40054f213f81e0e9d9be18ce778a48e97c7` |
| `million-route-8225-zplw2o2q/stdout.log` | `ff281bd63756462ac3668901bf336cea1debe8ea393e35877fc8e9dd0a2784a3` |
| `million-route-8225-zplw2o2q/phlegm-8225-1-2.stderr` | `a4163ad1b0ccf7a1c5883a3b750d8873234c932c81fa935adb9fbc11947a7485` |
| `million-route-8225-zplw2o2q/phlegm-8225-windows-events.json` | `50c292f085183034bd4d3aa87e4f654ba5f4600711175a5ee607ad8eade234fa` |

The managed 18,465-token comparison now prefers a separately built Phlegm bridge with a
non-committing prefill preview. It runs both attention routes against identical committed
state at each of the four 256-token positions. Each position has both run orders and compares
full-vocabulary logits. The control append then commits the block and must reproduce the
control preview bit for bit. Each route separately prefills the full fixture and returns 32
forced-decode vocabularies. This reduces the comparison from 73,860 to 41,026 evaluated
prefill rows before warm-up, about 44%. The timing includes the last-row LM head in both
arms. It excludes the append used to advance the common state.

All four late-block previews matched greedy IDs, with a maximum KL divergence of
0.00110913 nats. The candidate reduced prefill time by 0.13% when the control ran first
and took 2.05% longer when the candidate ran first. Control totals differed by 2.84%.
The candidate missed the 5% speed threshold in both orders. The interim timing receipt is
`.plans/inkling2/f32-phlegm/million-route-preview-tmmb78xl/block-timing-interim.json`
(SHA-256 `37effa7460865475118c03eeb8f4f369c876361b185119c499a6fd3cb3b3f8ec`).
The full 18,465-token replay then matched all 33 terminal greedy IDs (one prompt row
and 32 forced decode rows), with maximum KL divergence 0.00349842 nats. Both route
traces matched their expected dispatch counts. Windows recorded zero power events
and zero PCI context errors. The manager rejected the candidate on timing and
advanced to the independent feature checks. Final receipt
`.plans/inkling2/f32-phlegm/million-route-preview-tmmb78xl/comparison/result.json`,
SHA-256 `469e34a7154f1325e56590aff680b3d93f7aecf82367e77c60c4c2c71ad52729`.

Future managed comparisons stop after the first arm when the late-block speed
gate fails. The receipt keeps both timing orders and marks the terminal-logit
comparison as unevaluated. This saves the second 18,465-token prefill when the
route already fails qualification. A standalone comparison can still request
both arms for quality evidence.

An actual IQ4_NL host run matched the full 200,058 logits across repeated and discarded
previews, a normal append, and the subsequent decode after reset. The isolated Phlegm bridge
passed build provenance and protocol preparation with both the copying and direct-DMA
bundles. Simulated handoffs accepted and rejected replay results as intended.
The subsequent long recall requests use the traced Phlegm acceptance bridge.

| Capture under `.plans/inkling2/prefill-preview/` | SHA-256 |
| --- | --- |
| `bridge/phlegm-reference-bench.exe` | `1780593e4edc9f05ed0626ba66681c2d42a3064fa2465d2e547922327a2f3163` |
| `check-state.log` | `8912aa595826b27bb8a652f15db4913a4f39a8e3e14a4533ffae27b7cf47f8ba` |
| `preflight-copy/input.json` | `5752ff3a8c44ad82fc996130357767d2d242be91f4b121cefbeb21e222f7298a` |
| `preflight-direct/input.json` | `e4dd3cd7c13d861bfd179bdde9684784f9aad1dda9390635aea31eb0b7ed1c4a` |

Each comparison records the exact bridge, manifest, bundle and token-ID hashes and the Windows power
and PCI events. PCI context errors reject either comparison. A power transition inside a
timed four-block window disqualifies the longer result. Rejected and unusable results select
v11, and each new attempt keeps its logs and model captures in a fresh directory. A candidate
that clears both pairs must also complete the 131K 90% request on a fresh server. The receipt
requires the exact needle, full commit sequence, zero cached prompt tokens, and all 3,936
expected older-key visits and 145,152 launches. It binds the bridge, Phlegm CLI, candidate
bundle and trace to the launcher. A failed 131K candidate recall falls back to v11 at 90%.
The managed 1M run uses the candidate only when the same launcher's completed receipt still
matches. Otherwise it emits `million_recall_held` and exits before opening a Phlegm server.
The 131K control remains available to complete the shorter ladder. The 1M journal's
`million_route_selected` event records the selected manifest. Preparation passed
for both 131K 90% and 1M routes and both comparison fixtures. A simulated handoff exercised
missing receipts, mismatched binaries, wrong answers and missing trace entries. Device timing
and both longer candidate results are pending.

After a successful 131K candidate recall, the runner can also publish a 1M prefill estimate
from a clean source interval. It checks all expected older-key trace entries and separates
query packing and upload, K/V packing and sync, device dispatch, and host merge. It scales the
dispatch time by launches and by device K/V bytes in separate scenarios, using both the full
131K trace and the quarter beginning at position 98,304. Other prefill time scales by token
count. The work census reproduces the independently compiled schedule's 145,152 source
launches and 8,501,480 target launches, plus 559.55 TB of target older-key K/V reads. Estimates
require complete prefill timing and a clean power and PCI trace. They exclude decode and
future interruptions. The actual 1M recall remains the context gate.

The board timings now provide a dispatch-only estimate before a full model trace is
available. The 1,048,449-token fixture schedules 8,239,392 two-group dispatches at the
16K tier, which account for 98.8% of the older-key device K/V bytes. Multiplying by the
29.98 ms repeated-copy median gives 68.6 hours for that tier. Adding the remaining
launches at their recorded tier rates gives 69.5–69.6 hours for older-key dispatch alone.
The direct-DMA median gives 67.9 hours. The 27.22 ms median from the two stable paired-key
brackets gives 63.2 hours, pending the warmed repeat. At 131K, the same calculation
estimates 1.05 hours of older-key dispatch for the copying candidate. These scenarios
assume the isolated operator rates carry over. They omit host packing and sync, relative
attention, experts, all other prefill work, decode, and interruptions.

`.plans/inkling2/f32-phlegm/priority-device-20261123/operator-dispatch-estimate.json` SHA-256
`4db4baff628eb7c4214ca357547067e17fe8247dfee29cf06256f6a8e441600b`.

The 1,048,449-token fixture requires about 53.9 PFLOP for the two older-key dot
products across eight full-attention visits, counting two operations per
multiply-add. At the 8.6 TFLOP/s measured on the existing diffusion attention
operator on this machine class, that arithmetic would take 1.74 hours. This
estimate excludes data movement, exponentials, masks, local attention and the
rest of the model. Separately, linear extrapolation of the measured non-attention
remainder of the 8,225-token profile gives 7.24 hours. These are estimates for
prioritizing work. The full-context fixture still needs a real Phlegm run.

The existing head-dimension-128 diffusion attention image compiled for one
512-query tile against 4,096 old keys with Inkling's 24 query and eight KV heads.
It held all query-row partial state on 32 cores. Linked program text reached
11,312 B, linked data ended at byte 60,160, and the largest kernel frame used
3,328 B of a 4,096 B stack reservation. The probe repeats 16 real Inkling query
rows across the tile and uses real old K/V. It omits the sink and the recent
relative-position range. Single-BF16 query packing alone raised FP64-normalized
output relative L2 to 0.002508. Rounding only the exact output to BF16 raised it
to 0.001666. Both by themselves exceed the older-key 0.0005 operator target.
A usable wide-query route needs higher-precision Q and an unnormalized partial
state with enough precision to merge the sink and recent range. The stock
image took 3.422 ms median over 16 warmed dispatches, approximately 7.53 TFLOP/s
for this 512-query by 4,096-key operator. Repeated query rows reproduced the
same bytes throughout the tile. Relative L2 was 0.02345 against FP64 and
0.02332 against FP64 using the rounded Q. There were no power or PCI events.
This is evidence for wider K/V reuse. The stock arithmetic exceeded the
Inkling error budget. `.plans/inkling2/dit-old-keys/fixture/result.json` has
SHA-256 `02c26d1b557942dd0f658a6ddd76f31444681fcf029ba607817d145348d1c6f4`.

A second wide-query image uses the HD128 form of the existing split-precision
flash core. Four groups of eight cores hold 256 rows of one query head per
group. Each K or V chunk is read once per query head and broadcast to its
eight workers. The image carries Q hi/lo and probability hi/lo and emits FP32
numerator, base-2 center and softmax mass for an independent merge. It uses
the old-key segment only. The 4,096-key build fit 32 cores, with 8,032 B of
maximum text, data placed through byte 62,272 and a 2,624 B maximum kernel
frame within a 6,144 B reservation. Each column queues up to three fills and
one drain per head pass on separate channels. Fixture preflight verified
the wire layout and matched the synthetic FP64-reference partial state within
`3.37e-8` relative L2 after merging the sink. The real NPU accuracy and timing
probe is queued. `.plans/inkling2/dit-old-keys/wide-resources.json` has SHA-256
`9760139209e452af8e285a43e3b264de9dbfab091a7bee46fa8a375d4c7351f2`.

The fixed-4K board run passed, with `6.07e-5` relative L2 after merging the
sink, exact repetition across copies of the same rows, and zero power or PCI
events. The 16 warmed dispatches had a 22.361 ms median. The test repeated
16 real query rows across 256 positions. It did not exercise Phlegm cache
packing, context switches or recent relative attention.
`wide-fixture/result.json` has SHA-256
`d69f5bccf7d52d144bb1000ef2bf035e34f5d99abf1f358b77ef5b72bf943f66`.

The next build moves the old-key chunk count into a runtime parameter so the
4,096- and 16,384-key instruction streams can use one image. All 32 worker
ELFs and hardware initialization matched across the two builds. The streams
set 128 and 512 chunks on every worker. Program text reached 7,600 B and
linked data ended at 62,304 B, with the same 2,624 B frame and 6,144 B stack
reservation. The 16K route keeps at most two 64-chunk K/V slices in flight
alongside the query fill. The fixture compares a single 16K run, four 4K runs
with an FP64 host merge, and an independent FP64 answer including the sink.
Its offline merge matched the exact full result at `7.54e-16` relative L2
and rejected a changed numerator. The device run passed with relative L2 of
`6.09e-5` and `6.11e-5` against FP64 for the four slices and the full 16K tier.
The two results differed by `4.09e-6`. Median times were 62.887 ms across
the four 4K dispatches and 61.209 ms for one 16K dispatch, with zero power
or PCI events. That interval omits host operations and context switches.
`.plans/inkling2/dit-old-keys/tier-fixture/input.json` has SHA-256
`f211b99d6a3aca4ed28a37c92df9d2b194a07eb93f12eed234c98a06cf3818e1`
and `result.json` has SHA-256
`79dec7e2297edd9b7516dbfbe4ec90a6bdc80be2b76a0559f1ce38445db61c3b`.

The 61.209 ms measurement corresponds to 0.842 TFLOP/s for the 256-query
attention tile. At that rate the 1M fixture's older-key arithmetic and
in-dispatch DMA alone would take about 17.8 hours. Phlegm-level gains at
shorter contexts still need measurement. A single-image 16K diagnostic
compiles the same DMA traffic with four settings: no arithmetic, score
products, score products plus softmax, and full attention. The full build
fits in 8,000 B of text and uses the same local memory and stack as the tier
build. The queued device comparison verifies the DMA-only and score-only
outputs agree, checks that value accumulation preserves the normalizer,
and validates the full output against FP64 before attributing time.

The 16K same-image diagnostic recorded median dispatch times of 4.063 ms
for DMA only, 22.681 ms with QK, 30.872 ms with QK and softmax, and
61.329 ms for full attention. Adding PV increased the measured time by
30.457 ms, the largest increment. Each setting ran six times in mirrored
order after warm-up. The QK-only and DMA-only states matched. The two
normalizers with softmax matched, and the full output had `6.11e-5` relative
L2 against FP64. There were zero power or PCI events. These increments
include interactions with overlapping work. `.plans/inkling2/dit-old-keys/stage-profile/result.json`
has SHA-256 `d8256adf6ed6c4bf1f253e77c3c2df5bb5e98f6a621a72ef2d277226c9fdc1e3`.

The next PV candidate uses native 8-by-8 BF16 matrix products with separate
high and low probability terms and FP32 output accumulation. QK, softmax,
the input layout and DMA stay the same. A runtime parameter switches the
two PV implementations in one 32-core image. Both streams compiled and
passed preflight, with 9,456 B of program text, data placed to byte 62,304,
and a 2,624 B largest kernel frame with 6,144 B reserved. The board comparison
is queued after the singleton Phlegm run and checks the exact normalizer,
FP64 output accuracy and warmed dispatch time in both orders. The board run
preserved normalizer bits and had `6.11e-5` relative L2 against FP64, within
`2.45e-7` relative L2 of the vector route. It took 69.041 ms versus 59.031 ms
for the same-image control with no power or PCI events. Native BF16 matrix
PV with on-core transposes is 17.0% slower on this fixture. Keep the vector
route while testing the BFP16 implementation. `pv-comparison/result.json`
has SHA-256 `5a626aad9fc2fc5c3da0cb27b2fc1d7474096ece5ab94bb1e9a9dbe72ff32924`.

An independent host screen tested BFP16 exponent sharing with the existing
eight-value block encoder on the captured 16-query by 4K-key tile. It rounded
the residuals to BF16 before the second BFP16 encoding, as in the existing
compensated GEMM probe. The full QK and PV comparison, including the attention
sink, measured relative L2 `0.01783` for high products only and at least
`0.01265` with either two-product variant. Three products (high/high,
high/residual and residual/high) reduced error to `1.07e-4`, below the
`5e-4` operator limit. Adding residual/residual reduced it to `6.35e-5`.
The three-product PV-only error was `7.07e-5`. These are input representation
checks using FP64 products. On-array conversion, accumulation and throughput
remain to be measured. `.plans/inkling2/dit-old-keys/bfp-screen-bf16-residual.json`
has SHA-256 `b972d7855deca779ff1b4b3d781bd42827084b1b49d1efd9c1e0f4e778da3eac`.

The staged on-array BFP16 PV variant packs each panel of probabilities once
per key chunk and each V strip once for all query panels. It uses three
`mac_8x8_8x8T` products per tile into the existing FP32 numerator. QK and
softmax use the unchanged wide core. The setup keeps both limbs in local
scratch and retains the original K/V stream. The build and paired operator
test completed. The shared 32-core image used 9,264 B of text, data up to
byte 62,304, and a maximum kernel frame of 2,944 B with 6,144 B reserved.
Median time fell from 59.121 to 47.951 ms at 16K old keys (18.9%). The
normalizer matched bit for bit, full-output relative L2 was `9.36e-5`
against FP64, and there were zero power or PCI events. The 4K companion
stream compiled to the same core image. A Phlegm paired comparison is
running at committed prefixes 5,120 and 17,408 to exercise both streams on
real KV. `pv-comparison-bfp/result.json` has SHA-256
`a30c6e8786ba31cae5d5839c6906640bfe8ed554111a1be9f52320ebaf3893d8`.

The 5,120-prefix BFP16 Phlegm bracket retained the greedy token, passed
rollback and repeated-logit checks, and measured maximum KL `2.91e-5` nats.
The 256-token block ran 4.75% and 5.53% faster in the two orders, with
2.72% control drift and no power or PCI events. At the 17,408 prefix the
same process used the 16K stream. The candidate took 8.446 and 8.634 s
against 10.538 and 10.623 s in the mirrored orders, reductions of 19.86%
and 18.72%. Control drift was 0.80%. It matched the top token, passed
rollback and repeated-logit checks, and measured maximum KL `0.001043`
nats. There were no power or PCI events. For the 16 timed attention visits,
median query packing, KV packing and sync, dispatch and readback, and merge
were 1.002, 4.261, 52.394 and 0.171 ms respectively. The paired result under
`.plans/inkling2/dit-old-keys/bfp-model-comparison/result.json` has SHA-256
`d1781f128e8789beabb54e17842955cd136a77549b5030af5821c6f77cb8f4fe`.

The same kernel streams passed the 90% needle through normal
`phlegm serve`. The request committed 32,767 prompt tokens, issued
472 dispatches on the 16K attention tier, and returned `ZIRCON-1824`
with a stop finish. The 128 profiled prefill blocks totaled 1,223.355 s
(26.78 tokens/s), and the request took 1,287.0 s including decode.
There were zero power events and one PCI context event during startup.
The event named PID 19148, while the serving bridge had PID 69208.
This is evidence for committed-cache recall at the tested depth.
The elapsed time is an observation from a run with a PCI event.
`bfp-pv-recall-52720/result.json` has SHA-256
`6379b4615b07079baec5d599b18a4c75af916e33e62c7b1abd3df9b2f9cdfcf6`.
The post-run verifier checked every prefill block and the expected
recurrent visit schedule. All 128 blocks ran 48 attention visits
without CPU attention fallback. The trace has 856 wide visits and
1,416 short and 472 long wide dispatches. The verifier rejects a
missing recurrent visit. `schedule-check.json` has SHA-256
`72b4e8a226d552e2490cf5e44e9ad7ed88d8b4e372b974cde99411b9bfbb104f`.

A companion QK experiment packs both limbs of Q and K before dispatch and
uses the same three-product BFP16 accumulation. The existing BFP16 PV
and softmax remain in both arms. The two QK streams use matching 9,216-byte
DMA frames, with the BF16 control padded to that size, so the comparison
isolates score computation. The CPU layout check confirmed that all query
records fit and round-tripped 33 real K panels through the interleaved
format. Split representation relative L2 for those key panels was `6.80e-6`.
`qk-bfp-host-layout.log` has SHA-256
`23cea7a51f96efae163c41df0b5d53002c1ac0c4e4f82a6b4f854eac99ca35d4`.
Both instruction streams compiled. With prepacked inputs on the board, the
candidate reduced median 16K operator time from 47.216 to 37.621 ms (20.3%).
The score normalizers changed. The normalized output had relative L2
`1.254e-4` against the FP64 reference, within the `5e-4` gate, with zero
power or PCI events. The test excluded runtime packing and model
execution. The image has 32 cores, a maximum text size of 9,984 bytes,
and a maximum placed data end of 64,352 bytes with a 6,144-byte stack
reservation. `pv-comparison-qk/result.json` has SHA-256
`3ac2a8508ce242d3c0c7f9f7f3c304e12b5187f80d0ccbc6d3a75767adeb61fe`.

A host probe reused the AVX2 encoder and packed the captured 16K key
cache from the runtime token-major BF16 layout. Its output matched all
37,748,736 reference bytes. With six threads, rebuilding the keys per
dispatch took a median 12.882 ms, against the 9.595 ms board-side saving.
Packing each 256-row append took 0.221 ms, with all 64 appends verified
against the full packed reference. Copying the resulting 16K key segment
to a host buffer took 1.253 ms. These CPU medians were taken while the
32K recall was running and exclude V packing, XRT sync and model
execution. The margin favors encoding stored keys once. A separate
two-limb cache would add 18 GiB at the full 1,048,576-token capacity
across eight full-attention visits. Its production viability still
requires memory and reset, rollback, and repeated-prefix checks through
Phlegm. `bench-qk-pack.log` has SHA-256
`eb0ef3a6659596b9b23fbed79647d2bf27602ff308225b72b7906b34e842f3cd`.

The scratch bridge now packs each 4K committed range on first use and
keeps it by recurrent visit until reset. It wires the recurrent visit
index separately from the reused weight index. The host fixture passed
repeated preparation, prefix growth, recurrent visit isolation, reset,
and rejection of an uncommitted range. The 4K and 16K streams share all
32 compiled core images and initialization binaries. The bridge passed
protocol startup and provenance checks. The host fixture result
`cache-prototype.log` has SHA-256
`053d760933af6fd4cf4ac88ace831067ca47bc1cbb8e504fae6bd4f5896c0065`.

The paired Phlegm preview at a 5,120-token prefix took 7.860 and 8.337 s
with packed QK and PV, compared with the original v11 attention controls
at 8.687 and 8.848 s. Control drift was 1.84%, maximum KL was
`0.0001495` nats and the top ID matched. Both modes repeated their
logits exactly and the following commit matched control. There were
zero power or PCI events. The measured 4K dispatch and readback
median was 14.835 ms. The prior PV-only run's 16.964 ms is a separate
measurement. `qk-model-5120.json`
has SHA-256 `4e1953f36795124f2eb0392a739a56352933c65527ce79e1875ddf591b2bc64c`.

At position 17,408, packed QK and PV took 8.633 and 8.200 s versus
10.791 and 11.064 s in their matched control orders. The reduction
was 22.0% and 24.0% with 2.50% control drift, equal top IDs and
maximum KL `0.00104605` nats. Repeated logits and the subsequent
control commit matched within each route. There were zero power or
PCI events. All eight full-attention visits used the 16K tier.
Across the 16 timed candidate visits, median query preparation,
K/V upload, dispatch and readback, and merge were 1.505, 5.044,
42.779 and 0.187 ms. QK and PV together account for the comparison
with v11. The earlier PV-only run was a separate bracket.
`qk-model-comparison/result.json` has SHA-256
`08c5aa7f14188340970ecc05af3dfe765432de6334d29bc99f8849a47deb62d4`.

The repository now contains the experimental
`open_kernels/designs/inkling2/wide_old_attention.py` design and its
BFP16 kernel helpers. It selects the HD128, 32-key mode of the existing
flash-attention core. Rebuilding both tiers from those sources matched
the scratch bridge's 32 core ELFs, hardware initialization and complete
instruction streams byte for byte. The 4K stream SHA-256 is
`1f7f3e5da33c7f4befbac569c3dbab9900e4fce6feae73e2a799e34d69c8b2ac`.
The 16K stream SHA-256 is
`a55b47234e2da76ec3e89ff936024a3cf96ea61085d6f78c346ce90e91f4ab8f`.
The XCLBIN container differed. A default-mode regression build of the
256-row Inkling vision stream matched the existing 32 core ELFs,
initialization and instructions.

`export_inkling2_kernels.py --wide-old-attention` now provides the
optional v11 format. The writer checks that both streams share the
32-core image and each sets the expected chunk count and split BFP16
score and value parameters. The reader selects it from the bundle
manifest, keeps packed keys per recurrent visit and releases them
on sequence reset. A package check reproduced the existing v11
bundle byte for byte with the flag off. The flag added exactly
one image and two streams and the manifest field. A same-image
BF16-score stream was rejected. The new manifest SHA-256 is
`43c17aa13454f59fed21febbada71c8a9c4104de7d17abc28f9ccc1a55c73107`.
The normal and profiled Phlegm bridges compiled and passed the
protocol handshake. The offline import audit passed with 55
unchanged engine files and 41 matching the ledger. Its log
SHA-256 is `03337df809244e2ea230f6141ff89c08af9d62eb47744ec8e6099414039886d7`.
The running 131K recall uses its pinned scratch bridge. A 5,376-token
replay using only the new manifest is queued after the development
benchmark to compare the selected tier and logits against the
verified scratch route.

An independent postflight is waiting for the 131K runner to exit.
It checks the original HTTP response against the pinned 90% fixture,
every prefill block and recurrent visit including the 255-row tail,
the decode positions, the reported memory peaks, and Windows
power and PCI events. It also checks that the run result carries the
same artifact hashes as the original request launch. Offline, it
accepted the archived 32K recall and flagged its PCI event. Requiring
wide-tail coverage on that older trace correctly rejected it.

The packed-key 131K Phlegm run returned exactly `ONYX-3179` and
stopped normally after 80 generated tokens. The full 131,071-token
prompt committed in 512 blocks, including eight wide visits in the
255-token tail. The trace contains 3,936 wide visits, 6,048 4K
dispatches, and 14,112 16K dispatches with zero host attention
fallbacks. Profiled prefill totaled 5,367.285 s (89 min 27 s, 24.42
tokens/s). The bridge peaked at 39.17 GiB working set and 23.24 GiB
private. An event-collector timeout occurred after the HTTP response
was saved. Repeating the query for the original interval, 02:48:08 to
04:20:25 CST, found zero power and PCI context events. The independent
postflight verified the answer and block schedule from the original
artifacts. Its 92 min 18 s event interval includes server startup and
shutdown. The runner did not save separate HTTP wall time. The
benchmark and packaged manifest replay use separate captures.
`qk-wide-recall-131k/postflight.json` has SHA-256
`a091ada399b64162c3ea55dca82ddf1cb9304e59f36a72ae57bb59e425db5cb0`.

The source-built manifest selected eight packed 4K attention visits
on the 5,376-token Phlegm replay. All 200,058 logits matched the
pinned scratch route byte for byte. Its original event collector
timed out after saving the trace and logits. A retry over the actual
run interval found zero power and PCI context events. The replay
covered the 4K boundary. The packaged 16K model replay and singleton
cache retirement still need device evidence. The receipt
`tracked-wide-smoke/comparison-recovered.json` has SHA-256
`ed73b611a1030cceffb8accb6fd85bc0c42401e5851ba131a8374de56c6d173e`.

At 1,048,576 tokens, the eight full visits need 32 GiB of raw BF16
K/V and up to 18 GiB of packed older keys, plus about 160 MiB for the
40 rolling K/V rings. The early 131K run sample near 32K showed
33.49 GiB in the bridge working set, 18.08 GiB private and about
32.22 GiB available system memory. Growing both cache representations
to 1M would add about 48.5 GiB of payload over that point. File
residency and operating-system eviction prevent a peak prediction
from those counters. The full-length memory gate remains open.
`wide-memory-budget.json` has SHA-256
`87f1c9c7ef1003fd198c2bd41739308b76b774fa8180e2693a7372100352dca9`.

One way to avoid keeping duplicate old K/V is to store the BFP limbs
and retain a small correction record for the original BF16. A host
prototype rounded the reconstructed limbs to BF16 and stored the
positions and original bits of exceptions. It reconstructed all
37,748,736 original BF16 words exactly across the 16K operator
fixture and a separate 256-token capture covering all eight full
attention visits. The operator fixture repeated one 4K range. The
independent model capture needed at most 127 key and 88 value
corrections per 4,096-word frame. Its average record sizes project
414 MiB for K and 308 MiB for V at 1M, before lookup offsets. Both
packed planes plus corrections would occupy about 36.7 GiB before
recent and rolling KV, compared with 68 GiB if both packed planes
were added alongside all raw K/V. This tests the representation.
Runtime storage, restore throughput, fallback and longer coverage
remain open. `packed-kv-exceptions.json` has SHA-256
`c7b296be53cd642e3b7892bea02d47bd315da239ccd3085c40e8881f105adaa2`.

A native restore probe read independent correction fixtures and
reproduced all 16,777,216 key and 16,777,216 value words in the 16K
operator fixture. Both portable MSVC and AVX2 passed, including
rejection of duplicate correction indices. In the AVX2 build with
six threads, restoring 32 MiB each of K and V took medians of
1.61 ms and 2.27 ms. The V path includes a register transpose
back to the original layout. These were warm host-buffer timings
with the long Phlegm run active, excluding the token-major
destination, XRT transfer, and attention. Linear scaling would
charge about 2 s for restoring the whole 1M cache once. Decode
integration needs a route that restores only the tiles it consumes
and measures its part of end-to-end time. The AVX2 receipt
`check-restore-packed-kv-transpose.log` has SHA-256
`7bd78cf38ce926083f0ad0630dd78b2ae0f304ed8172488832294410ad1b82b1`.

A Windows segment probe populated a token-major KV reservation and
independently encoded one committed 4,096-token segment. The native
packed bytes matched both Python device fixtures for all eight KV
heads. After publishing the packed segment and corrections, the
probe decommitted 16 MiB of original K/V. It then restored the last
17 archived rows and read the next 23 live rows, with exact BF16
across the boundary in both planes and all heads. A later append
kept the archived pages decommitted, and a non-finite input left
the original pages committed. Model-cache integration still needs
to route every narrow and CPU fallback read around retired pages.
`check-packed-kv-segment.log` has SHA-256
`f8f0223dfc11b3f46b36c2d89b8c67e9f6df48bf8d00f0974a89213ea65d2f02`.

The Phlegm key-cache prototype now retires a Windows key segment after its
existing wide-attention bytes and exact BF16 corrections have been saved. Narrow
attention restores only the required 32-row frames. Values keep their original
storage. The AVX2 and portable native runs restored all 8,388,608 archived BF16
words exactly while 16 MiB of raw key pages remained decommitted. Both checked a
refused retirement, an append across the decommitted ranges, visit isolation and
reset. An injected archive allocation failure also preserved the earlier archived
keys and the new raw segment. Retrying then retired the new segment exactly. The
extracted runtime upload reproduced all 10,485,760 output words,
including the untouched relative-position stripe. In that host-only sample the
median 4K upload preparation was about 0.49 ms from raw keys and 0.64 ms from
archived keys. Creating the first 4K archive took 8.88 ms. This excludes XRT sync
and model execution.

Using the measured correction size, retiring the complete old key segments at
1M would save about 15.5 GiB compared with retaining raw keys beside the packed
copy. This is a memory estimate. The Phlegm replay compares 5,376-token prefill,
partial speculative commit, singleton decode, resumed block prefill and reset
with and without retirement. It released 64 MiB across the eight full-attention
visits and preserved all captured logits and draft proposals bit for bit. Prefill
took 227.42 s for the control and 227.00 s for the archive in one order. One PCI
event occurred eight seconds into startup. The pair is correctness and memory
evidence. The tracked exporter now offers `--reclaim-packed-keys` with
`--wide-old-attention`, and the engine selects the archive from the manifest.
The package check found only that manifest field different from the normal
bundle. The native NPU and verify targets built. The Phlegm replay with the
tracked bridge and standard decode matched all captured logits and DSpark
proposals bit for bit through prefill, partial accept, decode, resumed prefill,
and reset. After the first wide block, the archive arm used 61.6 MiB less
private memory, or 61.2 MiB after adjusting for the starting difference. The
5,376-token prefill took 223.42 s for raw keys and 215.62 s for the archive.
The single-order pair had one startup PCI event and establishes no speedup.
`wide-key-reclaim/check-avx2.log` and `check-portable.log` have SHA-256
`95771cc58eb1c30f3ad1033f9c341adcecbbaec6c923b2fcb1569b235b711500`.
`wide-key-reclaim/check-upload-result.log` has SHA-256
`4b0eed2758bbb00be049e4c49a1c7bdb4747df8116e09644cf451ac21b810a8b`.
The matching allocation-failure receipts `check-avx2-failure-retry.log` and
`check-portable-failure-retry.log` have SHA-256
`3da9406a8e7997a48842d8f3c93e8e8f0d4d75df43c3c59614f60c006481f1c8`.
`wide-key-reclaim/phlegm-178584/result.json` has SHA-256
`827056c9d2c87b4395771258456eda1bef9c9023b89de010026d08e150922c91`.
`wide-key-reclaim/phlegm-172216/result.json` has SHA-256
`ae97bf66167031d4bb7a2d66ea51f06ffd294e45a0446a8c86b8bcc7b8e0156b`.

In the packaged route the first singleton releases the prefill-only
packed cache while retaining raw K/V for decode. Later prefill can
rebuild it from committed keys. If the host packing allocation
fails, the current sequence uses the narrow path and retries wide
attention after reset. Allocation happens before wide attention
initializes the sink state. The extracted runtime failure check
verified that allocation failure leaves the narrow path's inputs
untouched and suppresses repeated allocation attempts. Native
portable and AVX2 packing checks passed discard and same-sequence
rebuild. These are host tests. The live recall and the queued
initial package parity comparison use their earlier pinned bridges.
`check-wide-memory-fallback.log` has SHA-256
`99784987d24056f8da9e89a2f6d20f3c6da091eaa384e3a3c431245d445c7a34`.

A follow-up kernel pairs two eight-row tiles on each worker and shares
both K limbs between the two accumulators. Each tile retains its
original three-MAC order. Both 16K streams compiled with identical
core and hardware initialization images. The maximum program is
10,992 B and placed data ends at 64,352 B. The largest kernel frame
remains 2,944 B within a 6,144 B reservation. The on-device comparison
uses identical packed inputs, checks raw attention state equality,
and times both orders. The run preserved raw output bytes with
`1.254e-4` relative L2 against FP64 and zero power or PCI events.
The median went from 44.772 to 42.563 ms. Per-order median pair
reductions were 4.47% and 2.58%. Keep this as an operator result
pending a model comparison. `pv-comparison-qk-paired/result.json`
has SHA-256 `58a588125d1c2fda7d8dc6c17498dd40248b966ffad85279bce0ee172bc1f8ab`.

Another PV candidate writes the same reconstructed probability as
FP32 in the score scratch once and keeps the separate hi and lo mass
accumulation. It uses two supported 4-by-8 transposes to feed the
existing BFP16 PV path. The toolchain rejected a single 8-by-8 FP32
transpose. Both corrected streams built and match all 32 cores and
their hardware initialization. The maximum text is 10,672 B, placed
data ends at 64,352 B, and the 2,944 B largest frame fits in 6,144 B.
The 16K device comparison matched raw outputs and normalizers bit for bit.
The control median was 38.221 ms and the candidate median was 38.449 ms,
with zero power or PCI events. Keep the original probability layout. The preflight
`pv-comparison-pv-fp32-prob-preflight/prepare.json` has SHA-256
`838adb3704b949f30c18cb4fb09391ca7015531c8ec4ef98260f7ba53829dcfd`.
`pv-comparison-pv-fp32-prob/result.json` has SHA-256
`df4bc00ad768418b0268657e8372ff3e3cbeafbc7fc4752e9f270101d181c118`.

Storing each BF16 probability limb once kept the math and removed the
duplicated high-limb writes and PV lane filters. The compact layout matched
the full 16K state bit for bit and took 37.883 ms, compared with 38.276 ms
for the control. There were zero power or PCI events. Keep the original
layout given the 1.0% isolated gain. `pv-comparison-pv-compact-prob/result.json`
has SHA-256 `50148082278e07b67adee5cf8e9fe8ae6004e23aba9314e19f1f9de69e4fd61e`.

A separate PV stream shares each V panel across two row tiles while
keeping per-tile product order and the original probability and V
conversion stages. Both arms compiled and passed the shared-image
audit. The image has 11,440 B maximum text, data placed through byte
64,352, and a 3,072 B maximum kernel frame within the unchanged
6,144 B reservation. The board comparison holds packed QK constant
and preserved raw output bytes with zero power or PCI events.
The median increased from 37.642 to 39.198 ms. The median pair
was 4.12% and 4.19% slower in the two orders. Keep the original
PV tiling. `pv-comparison-pv-paired/result.json` has SHA-256
`5b30e3d63e43c795f254945d09c693f801617045524cda6a40be380f23cb09aa`.

A second PV candidate shares probability panels across two output feature
tiles using extra worker-local staging. It matched both the control and the
earlier 16K capture bit for bit. The per-order control and candidate medians
were 36.976 and 36.989 ms, then 37.470 and 37.273 ms. Both streams used the
same image, with at most 12,272 B of program text per worker. The interval
had zero power or PCI events. The gain was below 1% in each order, so the
kernel stays in the experiment. `.plans/inkling2/wide-pv-pair/board-161012/result.json`
has SHA-256 `ddfc2482e1ce1644a1c3a7508c6159cc23a35715f70328c7060a74632f5224f2`.

The wide route's former exact-width check sent the last 255 rows of
the 32K needle through the narrow stream. That block took 13.770 s
in the completed Phlegm profile. The scratch tail bridge now uses
the wide route for 128–256 query rows, with zero-padded query records
and a merge limited to live rows. The extracted AVX2 host code
matched explicit padding and the full-width merge on all live
lanes at 128, 129, 255 and 256 rows, with intact output guards.
The same image and streams remain valid. This reduces the predicted
older-key dispatches in the 32K tail from 896 to 32 and in the 1M
fixture's 129-row tail from 18,360 to 528. The bridge passed protocol
preflight.
`check-qk-tail.log` has SHA-256
`3938ca00b7fd78d9baadfafa9e34976dfb9734dbd95dda6a2508c57d34ba202e`.
Portable and AVX2 host builds also matched all 37,748,736
reference key bytes and each other across 15,040,512 query bytes for
the four widths. The query capture SHA-256 is
`21d9ffb207ff2465025ee05b44d5434fc978c7931b195ef2e613f4c460e53534`.

The 129-row Phlegm pair at position 5,120 passed. Maximum KL was
`4.397e-6` nats, the top ID matched, repeated logits were exact within
each arm, and the subsequent control commit matched. The candidate
exercised the 4K tier at all eight full visits. Zero power or PCI
events were recorded. The candidate took 5.219 and 5.228 s, against
5.504 and 6.134 s for the controls. The 10.83% control drift exceeds
the 5% comparison limit, so this qualifies parity only.
`qk-tail-129/result.json` has SHA-256
`2eb0565c07bb492abc322c12a50f18cbad76368715c708d813b63fd02a395ff6`.

The 131,071-token HTTP needle at 90% depth then started through the
normal Phlegm adapter. The runner captures prompt and tail block
coverage, prefill time, process memory and Windows events. The bridge
SHA-256 is `a82b8b9c62fc95c68554fc4873693a5327af57eff7788838ad48925803d65b94`.

The 32K Phlegm HTTP recall trace also identifies what limits shorter prompts.
In seven completed 256-token blocks starting at 17,408 and ending at
19,200 tokens, all eight full-attention visits used the 16K tier. Median
block time was 8.974 s, including 3.503 s in MoE and 2.912 s in attention.
Wide older-key work contributed 0.484 s per block by median and 5.37% of
aggregate block time. The 56 single-16K dispatches had a 52.571 ms median.
Applying the isolated 9.595 ms QK saving to all eight visits would save only
about 77 ms per block here before packing costs. Older-key attention
remains the full-context scaling problem. Ordinary prefill also needs fewer
or faster relative-attention and expert operations. The tracked profiler now
breaks out sliding and full attention dispatch and readback, K/V and query
uploads, and the partial-result merge. The in-progress capture is under
`.plans/inkling2/dit-old-keys/bfp-pv-recall-52720/`: `stage-summary.json`
SHA-256 `fa208dedbe88685cfc9132c5d20f86d2aa1cfaff905de8342038645a35250a64`.

For a 256-token block at position 1,024, each of the 40 sliding visits
issues 16 attention dispatches. The 32-key alignment and 16-query group
size make every group span 33 key chunks. The 64-chunk tier reserves
10,485,760 K/V bytes per dispatch. A 33-chunk tier reserves 5,406,720
bytes with the same query and output buffers. The first two groups in
each of the eight full visits also select the new tier at that position,
so the block schedule predicts 656 narrower transfers and 3.332 GB
fewer K/V bytes. The dispatch count stays the same.

Both quant streams were re-lowered from deployed 32-chunk IR. This
reproduced the deployed 16-, 32- and 64-chunk streams byte for byte.
For the 33-chunk streams, the binary audit permits changes only to the
K/V DMA records and the 32 worker chunk counts. All existing artifacts
remain byte-identical. The scratch Phlegm build passed the protocol
preflight. The manifest SHA-256 is
`de1a341f75bf74679eae66ffe6c03f7f0a079cfa16e47ebe08e7dac1bc11bff7`.
The Phlegm pair preserved all 200,058 logits and confirmed 640 sliding and
16 full tier hits in both candidate arms. There were no power or PCI
events. K/V packing and upload fell from 432 to 397 ms in the first
order, and from 464 to 310 ms in the second. Sliding dispatch and
readback took 1.252 and 1.351 s for the control, and 1.260 and 1.177 s
for the candidate. Block wall times moved from 7.808 to 7.909 s and
from 8.311 to 7.499 s. The control drift was 6.25%, above the 5%
limit. Retain the byte and parity findings and keep the current default
pending a stable speed measurement. `sliding-attention-tier/phlegm-pair/result.json`
has SHA-256 `154f81257b78b7100d0abc58c952611edafde628ff87fa8a87e81436b28522c8`.

Across the original 4K captured cache, the two encoded BFP16 limbs changed
40,379 of 4,194,304 K values and 39,949 of 4,194,304 V values on reconstruction.
Relative L2 was `7.00e-6` and `6.94e-6` respectively. Retain the original BF16
cache for exact fallback and decode if prepacked limbs are staged with it.
`bfp-limb-coverage.json` has SHA-256
`360a91949aa0f55d00fea6211fd38ef1776dabdbf88d0575a147f5a7b2795023`.

The paired-route census counts the actual 256-query schedule, including
the partial final block and only the keys outside the relative range of all
queries. Device K/V traffic counts all three query heads per KV head.

| Prompt tokens | Wide 4K dispatches | Wide 16K dispatches | Device old K/V traffic | Estimated wide dispatch time |
| ---: | ---: | ---: | ---: | ---: |
| 8,192 | 96 | 0 | 4.83 GB | 1.63 s |
| 32,768 | 1,440 | 480 | 169 GB | 47.4 s |
| 131,072 | 6,048 | 14,112 | 3.15 TB | 13.0 min |
| 1,048,449 | 49,032 | 1,029,672 | 209.8 TB | 13.95 h |

This uses a 16.964 ms median from the 4K Phlegm bracket and the 47.951 ms
isolated 16K median. The 1M schedule also leaves 18,360 narrow old-key
dispatches in the 129-token tail, outside the estimated time. It would pack
69.9 TB of old K/V on the host and read back 3.50 TB of partial state across
the wide dispatches. The estimate also excludes other attention, packing,
merging, all other layers and decode. The current rate still leaves the full
1M recall on hold. `wide-census.json` has SHA-256
`c03fdf27c40eae9abce619b028a2fff2d79cc8602f5ba63995f2adb3867af9be`.
Substituting the newly captured 52.394 ms in-model 16K median increases
that dispatch-only estimate to 15.22 hours. This reflects the route's
additional in-model dispatch and readback costs on the two tested prefixes.

A projection from the complete 32K Phlegm PV trace includes the other
prefill phases plus wide-query preparation, K/V packing, dispatch,
readback and merge. It uses the exact schedule and measured 18.898
and 56.952 ms per complete 4K and 16K visit after query preparation.
At 32K the median scenario predicts 1,201 s across complete blocks
versus 1,210 s in the source trace. For 131,072 tokens the scenario
range is 85–101 min, with a median of 91 min. For 1,048,449 tokens
it projects 25.8–27.9 h across complete blocks, including 16.6 h in
the older-key stage alone. These scenarios apply observed non-older
stage rates and omit the partial tail and decode. They do not
measure or establish full-length throughput.
`recall-time-estimate.json` has SHA-256
`ea9dd35f9a31121751e9b193c94c4405b0128edb6d6737c505d77624928826f0`.

The packed QK/PV run completed 39,424 prompt tokens in 1,464.748 s
of profiled block time when sampled. Its median complete 4K and 16K
older-key stage times after query preparation were 16.222 and
47.811 ms. With the same schedule this gives a median projection of
about 88.5 min for 131K and 23.9 h for the full fixture. At full
length the modeled older-key stage accounts for 13.9 h and the
remainder for about 10 h. Removing older-key work alone would still
leave ordinary block prefill too slow for an efficient full-context
test. The scenario omits tails and decode, and low-priority compiler
work overlapped part of the source run. `packed-projection.json`
has SHA-256 `41c0b9f2d5ae174f1a89b92e40a4a1ad55bffc6268b5215a333643944fb8d498`.

A later trace through 93,184 tokens separates the prefill stages.
Fitting through 59,392 tokens and holding out the subsequent blocks
predicted 1,367 s against 1,417 s observed, 3.5% low. The held-out
older-key dispatch and readback took 202.0 s against 202.7 s
predicted. The same fit projects 86.6 minutes at 131K and 23.5 hours
at 1M before the partial tail and decode. The 1M projection assigns
12.2 hours to older-key dispatch and readback, 1.4 hours to its packing
and sync, 3.1 hours to other attention, and 1.2 hours to host SiTU.
The 131K holdout is still partial, and the 1M numbers remain a scenario.
`long-context-stage-budget.json` has SHA-256
`888721460e6d3f5caba21ebde5cc36eb69364d923fb35789b8e06bf416d3f4ec`.

Four 16K diagnostic streams now build against one common image for
the current packed QK/PV format. They retain the same query and KV
traffic and progressively enable QK, softmax, and PV. Preparation
verified all 32 core images and initialization payloads, the worker
mode writes, and the packed input hashes. The board test is queued
behind the grouped expert probe and checks the full stage against
the independent FP64 attention result. The receipt
`packed-stage-profile/prepare.json` has SHA-256
`d8c4d4f90485479cab8db6ce78c3faf105500baa8cbecdf60607c7e035733c7b`.

The diagnostic medians were 5.079 ms for DMA and local copies,
13.519 ms with QK, 21.523 ms with softmax, and 38.235 ms with PV.
The largest incremental cost was the PV stage at 16.712 ms. Full
output relative L2 against FP64 was `1.254e-4`. No power or PCI
context events were recorded. `packed-stage-profile/result.json`
has SHA-256 `8bd4732d51c6c99f43be168d46a59a9daa380abf2840e31beef6746850c337ca`.

A packed-V probe replaces the wide worker's per-chunk V transpose
and BFP split with loads from the existing 9,216-byte V frame. The
current and candidate streams share all 32 worker images and the
same DMA geometry. An independent host encoder packed all 512
chunks for each of the eight KV heads. The split representation's
largest per-head relative L2 was `7.11e-6`, and every K byte remained
unchanged. The queued board comparison requires identical softmax
state and checks both outputs against FP64. Its timings cover only
the operator. Keeping an additional packed V cache would add up to
18 GiB at 1M, so any gain must also be checked against packing and
memory costs. `packed-v-profile/prepare.json` has SHA-256
`95be12ec416809caa55719aa5585d29c9cfa6c43c7e2695a1d3a995460c24712`.

The board comparison reduced the warm 16K operator time from
38.007 to 32.058 ms and 38.144 to 32.070 ms in the two orders.
Control drift was 0.36%, softmax state was bit-identical, and both
arms were within `1.254e-4` relative L2 of FP64. No power or PCI
context events were recorded. The result supports profiling the
resident V route in Phlegm. It excludes host encoding, cache
resident bytes, and model-level logit checks.
`packed-v-profile/result.json` has SHA-256
`264be7039aff853fc73ac7443ba2be6c30926dc7861b1a834d113d08f1bed21d`.

The Phlegm experiment now carries BF16 V and prepacked BFP V as
separate 4K and 16K instruction streams under one image. Its native
cache copied all 37,748,736 B of the independent 16K V fixture
exactly, checked incremental committed growth and visit isolation,
and cleared both packed planes on reset and singleton decode. A
six-thread host-only 16K pack took 22.34 ms median while the NPU
benchmark was running, versus about 6 ms of measured operator
savings per launch. The Phlegm experiment therefore packs each
committed segment once and measures cold materialization separately.
The preview bridge passed protocol and provenance checks. Its
comparison is queued after the clean development benchmark, at
prefixes 5,120 and 17,408, with both warm run orders and a subsequent
commit check. `check-value-cache.log` has SHA-256
`ce3a705fb128b114d114bb25609efc5471f96425334f71d4cf76753849d414cb`.

At 17,408 committed tokens, the Phlegm comparison preserved the final greedy ID
with KL `0.001428` and reproduced the control logits after committing the preview.
Across four warm samples per arm, median old-attention time fell from 433.9 to
362.9 ms, including a reduction in the upload from 44.2 to 32.7 ms. Median wall
time for the full 256-token block fell from 10.120 to 9.826 s. The two arms switched
rank in the last pair and the warm controls ranged from 9.627 to 10.983 s. One PCI
context event occurred during prefix startup, outside the preview window, and no
power events occurred. Keep the current path as the default. The stage improvement
amounts to about 2.1 minutes at 131K and 2.54 hours at 1M if it holds over every
16K dispatch in the counted schedule. This scenario excludes cold materialization,
memory pressure and changes in cache reuse. The 1M prefill would still take about
21 hours under that assumption.
`dit-old-keys/packed-v-phlegm-146328/result.json` has SHA-256
`36408dca5020a562515bd3e07522926e14e0462432e2a62d211dec5f24a0fac5`.

A standalone AIE SiTU probe uses Inkling's gate and up softcaps
and the existing vector sigmoid and exponential. This checks
arithmetic for a future fused expert path. It built for one
1,152-wide row with 3,968 B of program text and 6,144 B reserved
for stack. Ten fixtures cover real IQ4 and Q8 gate/up outputs,
the cap range and near-zero inputs. Device comparison against
the independent BF16 outputs is queued after the package replay.
The standalone dispatch will measure arithmetic feasibility.
Production benefit still requires fusion with the expert projections.

The expert trace through 48,640 prompt tokens counted 213,030
gate/up and down groups in 189 warm blocks. The current group
reads back 1,179,648 bytes of gate/up output and uploads a
368,640-byte down activation. Eliminating those two transfers
would remove about 330 GB of host-facing traffic from that
capture. Host SiTU took 209.4 s, or 11.8% of profiled time.
This is a stage and wire census, so overlap prevents treating
these values as wall-time savings.

The real projection stripes place all 12 matching gate/up tile
pairs on different workers within each expert column. A fused
route needs to rearrange them on the device or change the
weight stripes. Simply running the activation on each worker's
own output would pair the wrong features. The host replay
reconstructed the tiles by physical worker and reproduced every
byte of the existing down activation wire for IQ4_NL and Q8_0,
using captured, fully active and fully masked rows. Per column
the intermediate gate/up payload is 147,456 B and the packed
down input is 46,080 B. The host byte and traffic receipt
`prefill-situ/fused-wire-result.json` has SHA-256
`4f8048f78215681f9e6d7563adbae68bb10720fb0336f9cc6fb0e4149c83e933`.

A native pack probe computes the down tile's int16 codes directly
from BF16 fields. It matched a floating-point rounding reference
on 652,800 finite encoding and scale combinations. The test also
covered the scale layout and ignoring masked non-finite rows.
The device pack and worker exchange remain open.
`prefill-situ/check-down-tile.log` has SHA-256
`452e84430f84e3c016dcb5288a4954b71c7ad667b1656560aa6be7a1615135ba`.

A candidate weight order puts all 12 gate/up pairs on the
same worker. Each worker sees gate then up for each of its
three consecutive 96-feature tiles. This keeps one 6,144-byte
gate tile live at a time and produces 11,520 bytes of packed
down intermediates per worker. Reordering the compact blocks
and packing directly from the permuted IQ4_NL and Q8_0 GGUF
rows produced identical bytes. Applying the inverse permutation
restored the captured projection outputs exactly. This changes
the gate/up stripe order, so the shared weight reader and any
legacy output unpack must use the matching order when enabled.
`prefill-situ/paired-weight-layout.json` has SHA-256
`cf33bc385695e3b8a6d1b253be12a86fa621cea611a636b431f858c3bc5bddca`.

The isolated fused SiTU and down-pack core now consumes two
native 16-by-96 tiles and produces three down K32 tiles. The
first compile stalled in the optimizer. Keeping bounded loops
rolled made the revised build finish, with 5,136 B of text and
a 6,144 B stack reservation. Separate 16-, 9- and 0-row streams
share one core image and initialization. Four real paired
tile fixtures use their own down-wire reference. The queued
device run will test the arithmetic and masks. The full fused
projection schedule is still open.

The grouped projection probe compiles for both IQ4_NL and Q8_0 across
all 32 workers. It accumulates each pair of gate and up tiles on the same
worker and emits three packed K32 records plus a 64-byte status record.
The IQ4_NL workers use 7,136 B of program text and the Q8_0 workers use
6,464 B, each with 6,144 B reserved for stack. The probe fixes all slots
at 16 active rows and drains to the host. It still needs device parity,
per-slot masks, and an on-device connection to the down projection.
The XCLBIN captures under `prefill-situ/` are `grouped-situ-iq4/final.xclbin`
(`218c5b5cc01d703537c9d6d4c35c49b3cdd8ff8b7ff43855361caa31533f26bf`) and
`grouped-situ-q8/final.xclbin`
(`795a00bd932c51845d7baf19d5eb8b67ec6a799949a41f284cd35b8fac56d871`).

A two-stage DMA probe compiled for IQ4_NL and Q8_0. Its up stage
drains packed down input and status into one BO. Its down stage
reads the same buffer layout, skips all 12 status records per
slot, and feeds K32 tiles to the workers in ascending K order.
A byte-index check of the generated shim descriptors verified all
eight slots and six output bands per slot in both formats. The descriptor receipt
`prefill-situ/grouped-down-wire.json` has SHA-256
`d9763f303545b8f94a2d719334deca0ce305d40e3b0b1d96f227190e7478a28b`.

The shared-image build then compiled separate up and down instruction
streams against byte-identical worker ELFs, initialization, and enable
payloads within each quant format. Both stages use a 6,144-byte output
record to share the core DMA layout. The up stage writes three packed
K32 tiles and status into each record. The down stage traverses the
shared buffer in K order and reads only its three K32 tiles, as verified
from the generated DMA descriptors for all eight slots. Workers read the
mode only after their first input and weight arrive so the dispatch can
set the new mode before starting the transfer. The BO handoff, mode
transitions, masked rows, and parity still need board validation.
`prefill-situ/shared-grouped-down-wire.json` has SHA-256
`a09f2cadadb7b39b42a1dc7bc8e81aaaa186325c1c06268984fd938e9198e23d`.

The masked variant uses the same 32 worker images and configuration
payloads for full up, mixed-row up, and down within each quant format.
The compiled instruction audit checks the active-row write at every
worker before DMA starts. The mixed fixture covers 16, 9, 4, 1, 13, 2,
7, and 0 active rows, with nonzero inputs beyond the valid rows. The
board test uses eight real expert planes each from IQ4_NL layer 0 and
Q8_0 layer 20. It runs up and down consecutively with the same BO and
then compares the resulting floats bitwise with a separate launch
fed the captured packed input. A second control quantifies the
difference from host SiTU on the original gate/up projection. The
host comparison was checked on archived inputs and rejected an injected
byte mismatch. The board test waits behind the existing Phlegm queue.
`prefill-situ/shared-rtp-down-wire.json` has SHA-256
`dda3683bfe350938d2307e279e9fad462f2b99d474b2b77e7810778b175f6c24`.

The native mask patcher identifies the 32 worker writes in each
full-row up stream. It validates the RTP register coordinates and
patches the row counts without changing the mode or DMA words.
Its mixed-row output matched the independently compiled instruction
stream byte for byte for both quant formats. The test also verified
restoring full counts and rejecting a bad coordinate, truncated or
wrong-stage stream, and out-of-range count. The queued mixed-row
board case uses this host-patched stream. The patcher remains a
probe until the board result supports integration.
`prefill-situ/check-expert-mask-patch.log` has SHA-256
`7d65d697f6458f4220c65c7e20397b1394647836378c347949c282e00d32d6c0`.

The board handoff passed at full and mixed row counts in both
formats. All 294,912 output floats in each case matched the
canonical down projection fed the captured packed input. The
mixed and zero-row slots returned zero for every masked output.
Relative L2 against the original host SiTU route ranged from
`4.27e-4` to `5.04e-4`, with zero power or PCI context events.
This validates the operator and handoff. Phlegm integration,
end-to-end quality and speed remain open.
`prefill-situ/shared-board-fixture/result.json` has SHA-256
`9acb7acafa75d023683c1aadf667a120a63e3d3fd5f09fbd953421126d6af76b`.

The native weight packer now has a tested paired-order prototype
for sharing one packed gate/up pool between the resident prefill
kernel and singleton decode. The IQ4_NL and Q8_0 output bytes
matched both independent legacy and paired board fixtures. The
matching FP32 scatter restored all 18,432 singleton and 294,912
grouped output positions, including preserving the sentinels for
masked rows. This avoids budgeting a second gate/up weight pool
when the resident Phlegm route is connected. The native capture
`prefill-situ/check-native-resident-layout.log` has SHA-256
`2e20589e134f73f113c8a57840feac05c1f2de7a051da8622740130b7abb6e61`.

The experimental Phlegm bridge now routes grouped prefill through
the resident up and down streams using the existing overlapped
lanes. Up and singleton decode share one paired gate/up weight
pool. The host checks the SiTU status before binding the same BO
to the down input. It no longer runs host SiTU or uploads a separate
down activation. A 512-token smoke and four forced decode steps
preserved all greedy IDs with maximum KL `0.003398` nats. A separate
singleton-only replay matched all five full logit rows bit for bit.
Host SiTU calls were 2,226 in control and zero in the candidate.
Activation syncs were 4,452 for 4,452 dispatches in control and
2,228 for 4,456 dispatches in the candidate. The checker originally
required equal group counts across arms, but the slightly changed
logits can change routing. The saved trace independently verified
the correct same-arm dispatch and upload counts. The timing bracket
recorded one PCI context event and supplies no accepted speed claim.
The recovered receipt
`prefill-situ/resident-model-smoke-137088/comparison-recovered.json`
has SHA-256 `c4d903f42c156578590e34956aa5f6a04ed75a654f500336b7c647c768fcb698`.

The reverse order reproduced the quality and singleton parity.
The resident arms profiled 16.21 and 20.03 s against control
17.57 and 17.67 s across the two orders. Both intervals contained
one PCI context event and no power events. Resident expert waits
were 5.98 and 7.38 s against control 3.83 and 3.21 s. The saved
host SiTU time therefore cannot be interpreted as a speedup.
`prefill-situ/resident-model-smoke-56296/result.json` has SHA-256
`b07971b38b08c79de2db539e0dd1567c0fb3bb64fcc7691549cfbc3fa3470e82`.

The revised kernel replaces the scalar down activation pack
with an exponent reduction, one vector fixed-point conversion
per active row, and vector stores to the existing K32 wire.
Its isolated board test matched every word of the four existing
operator fixtures, including the all-masked case. It rejected
active gate infinity, active up infinity and active up NaN, and
ignored a non-finite masked row. There were zero power and PCI
context events. The full grouped builds passed for IQ4_NL and
Q8_0. Their up and down streams share a core image within each
format, and the shim instruction bytes match the earlier resident
streams. A Phlegm smoke using the new images is queued after the
17K packed-value comparison. The isolated result
`prefill-situ/vector-packer-board-168660/result.json` has SHA-256
`fef102f126c8c4bde0307a6f45eed8656998e8f6fe469372046efe3cab183280`.

The vectorized Phlegm smoke preserved all five greedy IDs after the 512-token
prefix with maximum KL `0.003398`. All five pure-singleton logit rows matched bit
for bit. It eliminated 2,226 host SiTU calls and cut activation uploads from
4,452 to 2,228. Grouped device waits increased from 1.964 to 5.381 s, offsetting
the 3.296 s of host SiTU removed. Full block prefill took 18.278 s against
18.141 s for the control, with zero PCI context and power events in this order.
Keep the current default. This comparison isolates a device bottleneck for the
resident approach and gives no reason to run a longer prefill with this version.
`prefill-situ/resident-model-smoke-148976/result.json` has SHA-256
`10f51b06ec7872961110d1212a8d5145379cf2239b6fb3a1c50f3a35a0229180`.

A follow-up experiment interleaves the four workers' weight tiles in each expert slot so
the shim can read a contiguous span. Grouped prefill and singleton decode can share the
reordered pool. The original eight streams rebuilt byte for byte. In each candidate,
only the eight weight DMA descriptors changed, and the four reordered GGUF fixtures
reproduced the original worker tile bytes. On the board, all 12 final outputs and
the 11 warm intermediate buffers matched exactly within each quant format. The
first intermediate buffer had 53,760 zero words where subsequent runs left data
in padding. Its full-buffer comparison marked the raw receipt false. The final
output already matched on that first run. Warm IQ4_NL up and down times were
essentially flat, and Q8_0 up fell from 1.840 to 1.796 ms. The interval had zero
power and PCI events. Keep the existing layout.
`prefill-situ/contiguous-expert-dma/prepare.json` has SHA-256
`c35b0d3438e8f9848d7899168edd08f40cfe24ef0e3be378d5622d30fae37c39`.
`prefill-situ/contiguous-board-167344/result.json` has SHA-256
`5f31f390b02fb4c59876cdfc16ddfa059978a9ab5dba35d9cdb02282414e4bb9`.

The scratch overlap bridge reuses the existing one-in-flight protocol.
It prepares the next old K/V segment in a second buffer while the
current kernel is running. It waits before launching the next segment
and merges partials in the original order, with separate readback
buffers. The additional buffer payload is about 75 MiB. The bridge
build and protocol preflight passed. If all measured 5.044 ms long
uploads after the first segment were hidden, the counted schedule
would save up to 53 s at 131K and 1.40 h at 1M from long uploads.
These are ceilings. Contention, readback, and model parity still
need to be measured. The scratch bridge SHA-256 is
`f66f94b5dee84255a110a4a84240afb999c1b8c610d6141f7a7b13351b87aa9f`.

The staged Phlegm bridge uses both tiers for older keys outside the relative
extent of all 256 queries. It seeds the host merge with the model's attention
sinks, then evaluates the remaining old keys, relative positions and sliding
attention using the existing bundle. The prototype keeps split Q, BF16 cache
and FP32 partial state. The normal and preview builds passed protocol
preflight. The queued preview compares the same committed prefixes at
5,120 and 5,376 in both orders, checks full logits and subsequent commits,
and records cache packing, dispatch and merge time. The preview bridge SHA-256
is `c719434dd26778a7c9e6c6b7a34b477cef9c4166e97d86075cf8a9907ab48a1f`.

The paired Phlegm previews passed at positions 5,120 and 5,376. Both retained
the control's top token, reproduced repeat logits and left subsequent commits
unchanged. Maximum KL was `0.0001694` and `0.0000330` nats, respectively.
The route used one 4K tile at each of the eight full-attention visits per
enabled block. All recorded windows had zero power or PCI events.

| Committed prefix | Control drift | Savings in the two orders |
| ---: | ---: | ---: |
| 5,120 | 5.009% | 6.92%, 6.51% |
| 5,376 | 1.54% | 4.06%, 5.70% |

The first bracket exceeded the 5% drift limit. The second supports a modest
block-time gain at this depth. The model comparison uses transactional
previews and commits control blocks. A full candidate recall and a Phlegm
comparison that exercises the 16K tier are still open.
`model-comparison/result.json` has SHA-256
`0bc7e4beadfffbdc83885490d96f6653e1548b25d40a51127afcc80c28ecc624`.
An extracted MSVC/AVX2 check executed the same staged C++ wire operations.
It checked the query split and transpose over 786,432 values, reproduced all
8,388,608 and 33,554,432 K/V words at nonzero offsets for the 4K and 16K
tiers, and merged three intervals plus sinks against FP64 across all 6,144
query/head pairs. The merge relative L2 was `5.14e-8`. The Phlegm comparison
will measure device transfers and verify cache state. `.plans/inkling2/dit-old-keys/check-model-wire.log`
has SHA-256 `4ff64483fb83e3d4d7a2b5d44a3103e2bd9adb54c8878261f668b2ebcd0f3b72`.

On the same captured 16-query by 4,096-key tile, no old-key block qualified
for omission at a `5e-4` total softmax mass budget, split evenly over blocks
of 32, 64, 128, 256 or 512 keys. This held for every head and row even with
the exact mass from all keys available to the screening test. The tested
geometric bounds could omit zero blocks. Continue measuring denser tile layouts
for this fixture. `.plans/inkling2/dit-old-keys/skip-bounds.json`
has SHA-256 `0dfba94428a4bf214b629c9c966a24064c7906051906cac9df18d63c5055e3b4`.

| Capture under `.plans/inkling2/f32-phlegm/` | SHA-256 |
| --- | --- |
| `million-131k-handoff-preflight.log` | `18a622687a2d60bfaa2ae0d55f6c49da462abd88158d4f5fafb7b4f6ea7a54f` |
| `million-131k-handoff-simulated.log` | `e024952f00105f3d4c07d5b4428c6f96c30750908663a9c6d577cede08c0a8e6` |
| `million-hold-preflight.log` | `4958d9b5ff1acdbf75fa968f66b19bea010ca59d9bf9c1b360408156f1e6715a` |
| `context-estimate-validation.log` | `2c90b48ce2c5c6dacfc9a0e841d1490a544bcb2acb448d728b9edf9806a80ec4` |
| `context-estimate-handoff.log` | `08f0b524e668171e9c9312f5060ed159665a1df1e7e49174f930992ef78c3eee` |

| Artifact under `.plans/inkling2/f32-phlegm/` | SHA-256 |
| --- | --- |
| `phlegm-bridge/phlegm-reference-bench.exe` | `f6ec20e0c4b47504078664673f220f5f474b2e974959cb5fd1d43dad0b6348ea` |
| `bridge-preflight.log` | `e85032850ad6f7cf917a216c1c0ea633dd21a24b96b9505b40a7c6a613436e44` |
| `old-range-8225-preflight.log` | `fcd79e68124eabfd03bde8bfd47dfce0c99459ba62e2dc2f751481552a1af68b` |
| `old-range-8225-board.log` | `0ae51ea6033331f517eabedd22d8e0cd5785be2e065dfe6b4f1486043da8c6ab` |
| `old-range-17441-preflight.log` | `655d51446cdd2e1070a518b9512ee5ceb77b595d3745a8e573a40193d201839e` |
| `recall-131k-preflight.log` | `d1fe1fd7df05d37406ac621dffa56152d28ee825979577cce5c5f7173d0e6c28` |
| `recall-1m-preflight.log` | `e2af443e5a7ae79c1462b0e9f598655192ba7a6ed69839cb2a85573b74153fcf` |
| `recall-1m-route-preflight.log` | `15593607dffd133231384a9e98034fef37c64a526d56871f0377065524324274` |

| Artifact under `.plans/inkling2/virtual-kv/` | SHA-256 |
| --- | --- |
| `storage-131k-0-before.log` | `d6cce0640c1d2776ad9914817093e7a9752017ca8f61d5ac4e8730ec0731a810` |
| `storage-131k-1-after.log` | `fb0728353bf61004a34c3bf50f703a64681ed85c58b3193f059a150492cb7fd6` |
| `storage-1m-before.log` | `67cd0a906557ffd0a9ba43665351878a8ae6c9e959dd4d8132267da28e4ba928` |
| `storage-1m-after.log` | `2b90d7e5178ababfe803defb13bf1dbc53d342c070aa49e65ab4fef064c0ee65` |
| `storage-1m-reserve.log` | `2f1b020ab53c87619ba54f6df610633a21700b0d237ea4ccf3851edd89019655` |
| `bridge-preflight.log` | `a55a43c4a757b5cd98d80f25bb4fbef72578474457a4556e80894a34730f2c72` |

The current tier schedule can make that run much longer than the original estimate. A 256-query
full-attention visit at position 32,768 took 1.302 s in the Q8_0 synthetic probe. If that visit's time
scales with prefix length and applies to all eight full-attention visits, averaging over prompt
positions gives roughly three hours of full attention for 131K and eight days for 1M. This
projection excludes other stages and assumes the same tier throughput at much larger contexts.
End-to-end wall time must come from the actual recall runs.

An exact transfer census uses the current v11 tier schedule, 256-token prefill blocks and the 48 visits
from the GGUF-backed layer walk. The launch counts match the seven retained operator schedule replays, including
the 1,048,576-position boundary. The table counts logical DMA bytes, including padded chunks. Host
K/V syncs can feed multiple device launches because all query groups in a block share old ranges.

| Prompt tokens | Attention launches | Full old ranges outside the relative extent | Host K/V sync volume | Device K/V transfer volume |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 155,648 | 53,760 (34.5%) | 1.22 TB | 2.27 TB |
| 131,072 | 1,409,024 | 999,936 (71.0%) | 5.98 TB | 25.64 TB |
| 1,048,449 | 69,977,832 | 66,701,880 (95.3%) | 125.00 TB | 1.44 PB |

Complete old ranges carry 81.8% of the scheduled device K/V reads at 131K and 97.4% at 1M.
Those are transfer shares. The next device slot now runs the copy/score/softmax cost breakdown
and the packed QK/PV device comparison after the five short Phlegm checks and before 131K 90%.
Both use the exact saved buffers and compiled streams. Their timing controls apply within
each comparison, and the queue records Windows power and PCI events for both. The two
probes cover 273 dispatches each. In a fresh offline replay, the combined packed matrix
path reached maximum head-output relative L2 of `3.1283e-5` against FP64 at both old-range
positions, reproducing the previous host report byte for byte. Device timings remain open.
`f32-phlegm/priority-preflight-attention-20261123/results.jsonl` has SHA-256
`8fc224442d33678a50e16610426220f270c7d724c1138bad78b2242cafc4f5c0`.

The retained shared Q8_0 control took 4.662 ms for one 4,096-key range lying entirely outside the
relative extent. Applying that single dispatch time to the 66.70 million old ranges alone gives
86.38 hours. This estimate excludes the host, context switches and other ranges. The paired profiles
will measure the effect of copy, score, softmax and value accumulation on identical wire geometry.
The four 128-chunk packages compiled with matching instruction streams, control ELFs and placed
buffers and DMA records. The harness brackets every measurement with full controls and marks timing
pairs with more than 5% control drift. It verifies full output against FP64 and checks the intermediate
fields each variant still computes.

A separate transfer prototype sends 13 frames per old key chunk instead of 16. It keeps K and V bytes
and omits the unused relative-bias strip, reducing each 4,096-key transfer from 20,971,520 to 17,039,360 B.
The normal, compact and expert streams share identical core ELFs and initialization. The compact and
normal attention streams have identical query and result descriptors. The maximum linked program is
7,712 B and linked data ends at 61,532 B with an 8 KiB stack reservation per worker. The old-range
fixture accepts a closest key distance of exactly 1,024 and rejects 1,023. The retained hardware
control agrees with the independent FP64 output at `2.99e-5` relative L2. Device equality and timing
for the compact stream remain open. Using it for every eligible range would remove 262 TB of logical
K/V transfers from the 1M prompt, about 18.3% of the current total.

The native compact packer preserves every K and V byte from the retained fixture, including at the
1,024-position boundary. It rejects relative, partial, pending and sliding-window ranges. Its eligibility
counts match the transfer census at 32K, 131K and both 1M boundaries. Isolated MSVC pack time fell from
1.81–1.96 ms to 1.72–1.88 ms per old tier. Clang pack time rose from 1.28–1.31 ms to 1.41–1.45 ms.
These host measurements exclude device sync and execution.

The staged Phlegm reader selects the compact descriptor within the shared singleton image. It keeps
the same query order, key intervals and attention core arithmetic. The combined normal and compact
streams passed the DMA span check, and the repacked query fixture passed offline preparation for
both. The experimental bridge compiled and passed handshake and provenance. Device parity and
throughput are pending.

The combined bundle finished all 14 streams. Their core ELFs and hardware initialization match. The
native geometry verifier checked each compiled stream and rejected one changed offset per stream.
The maximum linked program is 11,072 B. The native operator and Phlegm 8,192-prompt-token paired
tests passed offline preparation with the assembled bundle.

Three isolated variants use 8×8 BF16 matrix scoring for complete old key chunks. The first gathers
tiles on the core. The other two prepack Q and K so the core can load each tile directly. They hold
either two accumulators or one accumulator per query half. Each keeps the original scoring path for
relative, sliding-window and partial chunks. The matrix variants change floating-point grouping.
Checks on 96 head and chunk samples at each old-range boundary verified the tile indexing against
FP64. The CPU FP32 grouping gave a maximum normalized attention-output relative L2 of `3.32e-7`.
The native tiled K/V packer reproduced the independently prepared bytes. Its host-only median was
1.88–1.95 ms versus 1.70–1.76 ms for compact row-major packing on MSVC, and 1.32–1.37 ms versus
1.28–1.31 ms on Clang.

| Score kernel | Max linked program text | Chunk function stack frame |
| --- | ---: | ---: |
| Scalar control | 11,072 B | 1,216 B |
| On-core gather | 13,168 B | 2,560 B |
| Packed, two accumulators | 12,992 B | 2,880 B |
| Packed, one accumulator | 12,544 B | 1,536 B |

For each variant, the normal attention, compact attention and IQ4 expert streams share the same core
program and initialization. Their instructions, control ELFs and placed DMA geometry match the
corresponding scalar streams. The device fixture passed preparation and will compare FP64 parity at
two old-range positions and check a relative partial range against retained hardware output. It uses
the same K/V transfer size across all score variants and brackets them with controls. The actual
numerical and speed effects remain unmeasured.

Two value-accumulation variants use an 8×8 BF16 matrix product on complete old key chunks.
They transpose the split BF16 probabilities in existing scratch and retain the peak, mass and
three-product numerator rescaling. One gathers V on the core. The other loads prepacked V in
64-word tiles, with the same transfer size. A third image combines prepacked Q, K and V so both
matrix products run on the old range. Relative, sliding-window and partial chunks use the original
arithmetic. The prepacked images use experimental flags that accompany their input layouts.

The host replay covered all 24 heads across 128 chunks at each of the two retained old-range
positions. Using NumPy's exponential, the maximum per-head normalized relative L2 error versus
FP64 was `3.101e-5` for the tiled value path and `3.129e-5` with tiled scoring enabled as well.
Both positions use the same K/V bytes and differ in their distance from the relative cutoff.

| Value kernel | Max linked program text | Chunk function stack frame |
| --- | ---: | ---: |
| Scalar control | 11,072 B | 1,216 B |
| Matrix values, on-core gather | 13,520 B | 1,280 B |
| Matrix values, prepacked | 13,488 B | 1,280 B |
| Matrix scores and values, prepacked | 14,880 B | 1,664 B |

The normal attention, compact attention and IQ4 expert streams built with the same core ELFs and
hardware initialization within each arm. The control instructions, control ELFs and placed DMA
geometry match across arms. The 273-dispatch comparison passed offline preparation, including a
byte-exact round trip of all V tiles and unchanged K and padding. It compares both old positions
against FP64 and a relative partial range against the retained device output. Device parity and
timing are pending while the 131K Phlegm request owns the NPU.

An isolated old-range image retains two independent 16-query groups in each active core. Both
groups consume each packed 32-key K/V chunk before the core loads the next chunk. The one-group
and two-group streams have identical 32-worker core ELFs and hardware initialization. Compiled
DMA spans show that either launch reads exactly 16 MiB of K/V for a 4,096-key range. Two separate
launches read 32 MiB for the same 32 queries. The two-group image uses 8,592 B of program text,
linked data ending at byte 63,036, a 768 B kernel stack frame and a 4,096 B stack reserve. It uses
36 memtile buffer descriptors per column. The 42-dispatch device comparison passed offline
preparation. It uses distinct queries and sink settings for the two groups and requires their
raw device outputs to match single-group output and their normalized outputs to match FP64.

For the rendered 1M prompt, the existing schedule has 69.98 million attention launches and
1.436 PB of logical K/V reads. Reducing the record to its packed K/V planes alone gives 1.157 PB.
Pairing the eligible shared tiers brings the projection to 36.63 million launches and 0.597 PB.
At 131,072 tokens the same calculation reduces launches from 1,409,024 to 909,056 and K/V reads
from 25.64 TB to 13.06 TB. These are counts from the compiled geometry and existing schedule. Device
runtime and the cost of integrating the separate image into Phlegm remain open.

The paired image is available in an experimental Phlegm bridge and v11 bundle. The reader batches the
existing loop over complete old K/V tiers into the dedicated context before returning to relative
attention. It preserves ascending key order per query. It retiles the existing split BF16 query cache
once per block, packs K/V once per tier, and uses the one-group stream for an unpaired final group.
Native MSVC and Clang tests reproduced the independent Q and K/V fixtures byte for byte. They also
checked a nonzero cache offset and rejected relative, partial and wrapped K/V ranges, a changed DMA
offset, a changed repeat count, and truncated streams. The experimental Phlegm protocol handshake
and binary provenance passed.

The model comparison uses 5,153 and 8,192 prompt tokens with 32 fixed decode steps. It records full
prefill and decode logits, prefill wall time, and the old range's query, K/V, dispatch, and host merge
times. The 5,153-token case exercises a paired launch and a single fallback after a partial final
block. For 8,192 tokens, the compiled schedule predicts 768 old-range launches in paired mode and
1,536 in single-group mode over the same 96 stages. Both fixtures passed preparation. Raw parity
between widths, KL against v11, context-switch cost and end-to-end speed await device runs.

The exact host merge function from the compiled Phlegm source passed on MSVC and Clang with four
synthetic key ranges and six query-block lengths from 17 to 256 rows. The paired, shared and
head-major layouts produced bit-identical peak, mass and numerator states after each range,
including an unpaired tail, partial groups and rows with zero mass in intermediate ranges. The
largest normalized relative L2 against an independent FP64 sum was `6.843e-8`.

A host-only pack benchmark rotated six packers on three cached offsets, with 11 timing
samples of six packs per arm. The selected loop completes each 8×8 tile with fixed-size copies.
It reduced the original tiled pack from 5.15–5.35 ms to 1.39–1.51 ms per 4,096 keys on MSVC,
and from 1.81–1.88 ms to 1.56–1.66 ms on Clang. The v11 row-major measurements were 1.80–1.86 ms
and 1.14–1.18 ms. Both native compilers reproduced the independent packed bytes with the selected
loop. The rebuilt experimental Phlegm bridge passed handshake and preparation with this packer.
These are host-buffer timings. Phlegm end-to-end timing includes the mapped buffer, sync, dispatch
and context changes.

After the 131K recall and F32 profile, the 42-dispatch device comparison completed under the
Phlegm device lock. Single and paired launches returned bit-identical raw state in all four
orders, with relative L2 at most 2.98e-5 against FP64. The bracketed control moved by 11.6%,
above the 5% timing threshold. There is no accepted speed estimate from this sample. The log
`.plans/inkling2/old-kv-query-pair/device-queue.log` has SHA-256
`6f7ca76840acaf55bf469f31d2ec101f2b90e7703ccd977dd0071e93f4feeb4a`.

An experimental extension lets the same core image process 4,096, 8,192 or 16,384 old keys
per launch. Each stream sets the chunk count and query group count for every worker. The
reader picks the largest compiled tier that fits the complete old range, then visits keys
in ascending order. Relative and sliding-window attention still use their existing streams.
The six candidate streams share all 32 core ELFs and hardware initialization. They fit in
8,688 B of program text, through byte 63,036 of linked local memory with a 4,096 B stack
reserve, and 36 memtile buffer descriptors per column. The maximum packed K/V buffer grows
from 16 MiB to 64 MiB. Each worker still processes one 32-key chunk at a time.

| Prompt tokens | v11 launches | Paired 4K launches | Paired larger-tier launches |
| ---: | ---: | ---: | ---: |
| 32,768 | 155,648 | 128,768 | 113,408 |
| 131,072 | 1,409,024 | 909,056 | 554,240 |
| 1,048,449 | 69,977,832 | 36,627,912 | 11,777,432 |

These schedule counts use compiled DMA spans. The larger tiers reduce query transfers,
device launches and host merges. Packed K/V reads are the same as the paired 4K route.
MSVC and Clang matched the independent K/V wire for seven ranges with nonzero offsets,
validated all six streams and checked 765 tier schedules. The native tests also rejected a
relative cutoff violation, partial and local ranges, wrong RTP writes and truncated spans.
The board fixture compares all six streams on the same 16,384 keys, with raw equality at
equal tier widths and normalized output against FP64 across tier widths. Its queued Phlegm
follow-up uses 17,441 prompt tokens plus eight fixed decode steps and predicts 6,208 versus
4,112 old-range launches. The official development bridge passed handshake
and source provenance. Wall time and numerical quality are pending on the device.

| Artifact under `.plans/inkling2/old-kv-long-tiers/` | SHA-256 |
| --- | --- |
| `resources.json` | `bf52897b9cb90b5fecd99266668b8ee9050d24cfcde3f848552c56eb02a73f5c` |
| `fixture/plan.json` | `018fe717f8099cf2ac7696b49707f433c3b7ba6c8ed3bbb699699fea12f24a01` |
| `check-wire.log` and `check-wire-clang.log` | `838356e9eb00acc3a8bb092b720b0e47603d2f136326fbd5a174806b2eb7c301` |
| `phlegm-preflight.log` | `7347d139043a0b4a66017dcda8c56408ca410a5d07afff0180ff8d933be57379` |

An isolated two-group 16K build lets workers read each packed K/V tile directly from the
consumer DMA buffer. The existing worker copies 16 one-KiB frames into a 16 KiB tile before
scoring. The experimental worker acquires one complete tile, uses it for both query groups,
and releases it. The shim still transfers 64 MiB of K/V and 576 KiB each of query and output
per launch. The linked program uses 7,728 B of text and data through byte 62,012 with the
existing 4 KiB stack reserve. All 32 consumer buffers are 64-byte aligned. The compiled
image uses 36 memtile buffer descriptors per column.

The pending 56-dispatch board pair uses the retained 16K fixture. It requires bit-identical
worker state against the original route and checks against the retained FP64 reference. The
timing comparison brackets every direct-DMA arm with the original, because holding the
consumer buffer during scoring can change DMA overlap. Offline replay verified the comparison
against captured original output and rejected an injected one-bit mismatch. The compile and
preflight establish the buffer layout. Device parity and speed are still open.
`.plans/inkling2/f32-phlegm/priority-direct-dma-preflight-20261123/old-key-direct-dma.stdout.log`
SHA-256 `29c12134af8bd9ef264975926568516e0343d1f193805908f86d6e56de5481e3`.

The direct-DMA bundle now contains all six streams for one and two query groups at 4K,
8K and 16K keys. All streams share 32 core ELFs and initialization. The stream checks
matched control wire spans and passed the native reader's MSVC and Clang validations,
including 765 selected tier schedules. Both short Phlegm comparison fixtures passed
preparation with this bundle. Its captures carry a separate bundle digest from the
copying version. The first free board slot now runs the attention breakdown, packed
matrix comparison and 16K direct-DMA comparison before the feature checks. All nine
checks passed preparation in the revised order.

The managed runner advances direct-DMA to the two Phlegm comparisons only if the actual
16K probe returns bit-identical state, passes the FP64 check, and saves at least 5% in each
of three brackets with at most 5% control drift. It requires zero PCI and power events in that
short probe and checks the exact image and instruction hashes against the candidate bundle.
The Phlegm comparisons still apply their full speed and logits gates to the selected bundle.
The 131K receipt binds that bundle's complete digest, so a kernel change with identical
manifest bytes cannot qualify another bundle for 1M. Both 1M configurations passed
preparation. Simulated receipts exercised acceptance, drift, weak gain, PCI errors, and
changed captures. The hardware result is pending.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `old-kv-direct/bundle-evidence.json` | `ccea520ca2c8a76a8af0319b403ad4b2a24319d3d5345543314341dbceb510a7` |
| `f32-phlegm/direct-8225-preflight.log` | `803abf3f88c174d8a192dd8cfe6e7d5c5f839d413844dcdbc202141f0d3d2aa6` |
| `f32-phlegm/direct-18465-preflight.log` | `6fdf580482a2c6e076cdf9d48b7ba514fb3365c15aa4cdd747490ff7c4509412` |
| `f32-phlegm/priority-speed-first-preflight/results.jsonl` | `0c1d89695430bb3ecc3789612509f96e20419347f6d745777b15eccf0a5ab83c` |
| `f32-phlegm/direct-handoff-preflight.log` | `393c0f34c44860c6980d4224545678d3d86f54481e1761595e717a9138ea4f95` |
| `f32-phlegm/direct-handoff-validation.log` | `a54b1096c34ea103b895eb047ebfa84dc8a934a9263ba4f809588673f1d42cbf` |

An isolated two-group 16K kernel adds a shortcut for tiles where all 16 running maximum
values stay bit-identical. It skips the extra `exp(0)` call and one zero-product update in
the denominator and numerator rescale. It still splits the previous state into rounded
BF16 high and low parts before adding the new contribution. A conventional IEEE FP32
replay matched 1,000,004 finite inputs bit for bit. An independent FP64 score walk of the
retained synthetic 16K wire found unchanged peaks in 22,774 of 24,576 head-group tiles
(92.7%). The remaining tiles exercise the general rescale path.

Peano compiled 10,848 B of program text with a 960 B maximum frame, a 4 KiB stack reserve,
and linked data through byte 63,036. The control and candidate use identical instruction
streams and control ELFs, DMA spans and 36 memtile descriptors per column. The retained
wire prepares a 56-dispatch board pair with three bracketed timings and raw output parity
against the original kernel. The capture parser passed offline replay. Device arithmetic
and wall time are pending. This experiment uses the copying kernel as its control to
isolate the rescale change.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `old-kv-unit-scale/coverage.log` | `b06cc57fbc5e5a66b2b3e28d995d1705a790b112a6889ee1f5120034a3856e5b` |
| `f32-phlegm/priority-unit-scale-preflight/old-key-unit-scale/prepare.json` | `4fc1197b2a3bf08545b56abfab00b76871de6198597fc1447aec410aa01a91f6` |
| `f32-phlegm/priority-unit-scale-preflight/results.jsonl` | `6f29b22c50a8ce378a63cb5861eac3219be0895c8bb1f6af0f0ae2c9370c7282` |

A separate two-group 16K build shortens the older-key exponential from six to four Horner
steps. It retains the original range reduction and the exact value at zero. In a host
emulation over 1,048,577 points in the clamped interval, the maximum relative change from
the existing exponential was `1.52e-5`. On the retained synthetic 16K wire, a FP64 reduction
using the emulated candidate exponential changed normalized output by at most `3.60e-6`
per head relative to the control and `1.63e-5` relative to the FP64 oracle. That calculation
isolates the exponential and leaves device accumulator rounding to the hardware comparison.

The candidate compiled to 8,688 bytes of program text with a 960-byte maximum frame and
linked data through byte 63,004. The host instruction stream, control ELF and DMA spans
match the control build. Its 56-dispatch comparison is queued. It checks repeatability,
normalized error at most `5e-4` to FP64 and `5e-5` to the control, and bracketed wall time.
The capture parser accepted a synthetic in-tolerance result and rejected a larger change.
Model-level logits and device speed remain open.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `old-kv-exp-polynomial/accuracy.log` | `b71b43c11150e66e6a1f87bfe0122c373468c69d4fc578bd0a8790f7e9c69c52` |
| `f32-phlegm/priority-quartic-preflight/old-key-quartic-exp/prepare.json` | `cf0a17f2a3ac1623330ba21e9ed461d35f3f961bd001b952f9884d42806ac19d` |
| `f32-phlegm/priority-quartic-preflight/results.jsonl` | `5e371d10c1fc7596db561cd8027180e22fcbed25dfbe52a99e06b21bbc32645a` |

A two-group 16K candidate also reuses each packed key tile across the two independent
query groups inside the core. Each group retains its high-then-low accumulation order.
The retained 16K fixture produced 12,582,912 bit-identical FP32 scores in host emulation.
The compiled candidate keeps the control's DMA spans and 36 memory-tile buffer descriptors.
Its largest program is 10,544 bytes. Data ends at byte 65,084 with a 4,096-byte stack
reserved. Disassembly shows 2,304 bytes of nested frames along the score-update call path.
The extra score scratch leaves 452 bytes before the 64 KiB data limit.

This change targets core-local key loads. The compiled schedule still reads 67,108,864
bytes of K/V per dispatch. Its 56-dispatch comparison is queued for after the active 131K
request. Device output parity and timing remain open.

A host check measured the cost of dropping the low BF16 query term on the same synthetic
16K fixture. FP64 attention with both terms reproduced the retained baseline exactly.
Keeping only the high term gave median head-relative L2 of `2.414e-3`, maximum `2.733e-3`,
and maximum per-query-row relative L2 of `4.412e-3`. All 48 head and query-group pairs
exceeded the existing `5e-4` head-relative L2 limit. The paired-key candidate keeps both
terms. This check isolates query precision with the original K/V, sink, and FP64 reduction.

The candidate bundle now contains all six one- and two-group streams for 4K, 8K, and 16K
older-key tiers. The compiled streams share the same core ELFs and hardware initialization.
Both MSVC and Clang readers passed seven exact packed ranges and 765 tier schedules. The
Phlegm preflight passed at 8,225 and 18,465 tokens and prepared the 131K and 1M fixtures.

The managed comparison checks direct DMA and paired keys against the same copying control.
It admits an alternative only when the device output matches bit for bit, passes the FP64
error limit, and saves at least 5% in every stable bracket with no logged power or PCI events.
If both alternatives qualify, the one with the larger minimum bracket gain advances to the
Phlegm comparisons. The simulation verified selection, rejection, and the bundle-bound 131K
receipt. Actual model timing and logits and the exact 131K recall are still required before
the selected bundle can advance to 1M.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `old-kv-paired-keys/host-check.log` | `ef1ed268bdf05107ccf51098ae57513862d0abacda5d70005976326f9746421e` |
| `old-kv-paired-keys/bf16-query-only.json` | `5cdd5d83234e98886b368e6b2764d6faa9179d9eeecb5a66d67c91f79d34c05a` |
| `old-kv-paired-keys/preflight/prepare.json` | `5a437b8f8cf3cefb324f3e1c4f0260e2f21acadc026e19629041334813ed4434` |
| `old-kv-paired-keys/bundle-evidence.json` | `98ea4ac84a9e999851ebbe52437f6f39f5701b86a5095a72e2b0d35e11879995` |
| `old-kv-paired-keys/phlegm-preview/input.json` | `d2a82c25880dd9c132b02988912109ab140d748d5b447d1668546d7d8f9835b0` |
| `f32-phlegm/paired-keys-handoff-preflight.log` | `6e787a35dd85bcd860292fb10633cf897e39554e2c060037113e391734fe719c` |
| `f32-phlegm/paired-keys-handoff-simulated.log` | `67fb3bbbe08e5fc1016f65c3c5ff0e8f345cc8ea594c976ee6bb058ccb511e37` |

The board comparisons ran after the 131K 10% recall. Windows logged no power or PCI
events during the following checks. The results use warm medians bracketed by the copying
control. Brackets with control drift above 5% are omitted from the gain column.

| 16K older-key candidate | Stable bracket gains | Maximum control drift | Output |
| --- | --- | ---: | --- |
| Direct DMA | 2.38%, 2.37%, 2.34% | 0.16% | Bit-identical |
| Unit rescale | 1.27%, 2.55%, 2.58% | 3.10% | Bit-identical |
| Quartic exponential | 2.62%, 2.67% | 28.27% | Maximum normalized head change `1.76e-5` |
| Paired keys | 9.12%, 9.16% | 35.06% | Bit-identical |

All four comparisons had maximum head-relative L2 of `5.842e-5` against the FP64 reference.
The first paired-key control fell from 51.34 ms on its first dispatch to 38.68 ms on its
eighth. Its later controls settled near 29.96 ms and the candidate near 27.22 ms. The
unstable first bracket keeps paired keys below the admission gate. A fresh comparison
is prepared with 24 unmeasured dispatches per arm, followed by the original 56 measured
dispatches. A replay with synthetic warm-up markers preserved the seven actual medians
and all three brackets. The warmed hardware run remains pending. Direct DMA fell below
the 5% gain threshold in all three brackets.

In the stage probe, both directions of the first full-range fixture yielded stable
brackets. Copy-only took 10.8–11.7% of control time, the cumulative score arm took
61.6–62.0%, and the cumulative softmax arm took 79.1–80.5%. The separate combined QK/PV
matrix probe changed the relative or partial-range output and failed its exact equality
check. The v11 host pack comparison matched output across all eight shapes. Its long IQ4
stage improved 6.1–8.1% in both directions with 4.56% control drift. This is a stage result
from the development model bridge. Block-level Phlegm throughput remains open.

The retained combined-matrix output also fails on the two packed older-key fixtures.
Their normalized error is 7.09 relative L2 against FP64. The two-key relative fixture had
both experiment flags clear, kept the same 48 positive-mass rows and differed by `7.68e-7`
normalized relative L2 from the control. The relative raw-state difference and large
packed-older-key failure are separate observations. The combined image remains experimental.
The audit `.plans/inkling2/matrix-value/combined-device-failure-audit.json` has SHA-256
`29889d516d9583e8f53899f8e79d6a252516f6ea3777fbcb28cc5d758741e9d7`.

Decoding the archived scalar, packed-value, and combined outputs isolates the error further.
Packed values alone preserved all 384 peak and mass values bit for bit and reached `2.97e-5`
relative L2 against FP64. In the combined image, 332 of 384 peaks matched an independently
reconstructed query/key score within `1e-5`. All 332 nearest matches came from chunk 127,
the final 32 keys in a 4,096-key dispatch. The scalar peaks were within `3.16e-6` of the
full-range FP64 maxima. This suggests that the combined route loses part of its accumulated
softmax state. The cause is open.

The queued board probe uses the same compiled image and instruction stream for the scalar,
matrix-score, matrix-value, and combined paths. It verifies the exact packed input bytes,
tests both old-range positions, and compares one, two, and four visible chunks to an
independent FP64 reference. The reconstructed 4,096-key output agrees with the retained
reference to `1.81e-15` relative L2. Preparation and archived-state checks passed.

| Capture under `.plans/inkling2/matrix-attention-isolation/` | SHA-256 |
| --- | --- |
| `archived-peak.json` | `1860d2b070bb66c65f657eef4514d0f6fa9889d0a5ab0c79165960c569503660` |
| `archived-state.json` | `d007f3b9c93832931dfaa199cdbd1baece637bbd372999c57b5d9630eab907b7` |
| `preflight-with-prefixes/run.cfg` | `c15b279fa3673d585d54a38c5ef67cc27fa743a2953309c3a43e8a5e3c263961` |

The combined image's disassembly shows a post-increment load from the output base
where the previous peak lives at byte 8,192. The pointer advances to the mass at
byte 8,256 after reading the first 16 numerator values. A small Peano compile of
pre- and post-increment C++ loads confirmed the addressing forms. Replaying with
that numerator slice as the previous peak reproduced the faulty normalized device
output to `0.00137` relative L2 and matched 341 of 384 peaks within `1e-5`.
Replaying only the final chunk gave `0.576` relative L2 to the same capture.

An isolated correction reads the previous peak in a non-inlined helper. The
compiled helper uses an indexed load at `0x2000`. Rebuilding the normal attention,
older-key attention and IQ4 expert streams produced matching 32-worker images and
initialization across the three streams. The control streams and placed DMA geometry
match the prior experiment. The corrected image uses 15,152 B of text, with linked
data through byte 61,724 and a 1,728 B attention frame within the 8 KiB stack reserve.
The 157-dispatch probe passed preparation and is queued after the original image
probe. It checks the older-key outputs against FP64 and the relative fallback against
the saved control, including the active row mask and raw output equality.

The corrected 157-dispatch run passed on the NPU. The maximum normalized relative L2
against FP64 was `2.990e-5` at 4,096 keys and `9.808e-6` for the truncated fixtures.
The relative fallback was bit-identical to the archived control. Power and PCI event
counts were zero. Control drift was 1.87% for the far fixture and 0.44% for the edge fixture.
Packed matrix scoring alone reduced warm dispatch time by 24.1–26.1% across fixtures
and orders. Enabling both packed scoring and packed values reduced it by 5.2–11.3%.
Carry the score-only route forward to a Phlegm comparison that includes host packing
and transfers. These timings cover the 4,096-key operator, with model timing and
longer key tiers still open. The corrected capture `matrix-device-20261123/matrix-attention/result.json`
under `.plans/inkling2/matrix-attention-peak-load/` has SHA-256
`d40e0a359f0e39b66fd8eda1b15f3119f1546047dda6907c9d73450427c33e9c`.

The Phlegm model experiment is staged in `.plans/inkling2/matrix-score-v11/`.
It uses the v11 image geometry and keeps the score and value layouts separate. The
host selects matrix scoring only for complete cached tiers outside the relative
extent for every query group in the block. Rolling, relative and partial ranges
keep the scalar layout. The profiled reader counts packed dispatches and separately
times host K/V packing and sync for the same range in each arm. A native wire test
reconstructed the model cache and matched the operator fixture exactly for both
packed query planes, all 1,024 key and value records, and untouched v11 stripe
padding. The test also used a nonzero cache offset. The shared-image build waits
for the block-size timing bracket and rebuilds all 16 shared streams from one source
set. All 16 shared streams compiled, with identical core and initialization
images within each quant format. Maximum program text was 10,832 B, linked data
ended at byte 61,532, and the 1,664 B attention frame and 64 B worker frame fit
within the 8 KiB stack. Each column used three single-descriptor shim tasks in
every stream. `resources.json` has SHA-256
`c5a037d7e9b48e0b31cfe63436403c0da74b8adefa8d8b60cb1fb107a4412b6c`.
The profiled Phlegm bridge built and passed preflight for both arms. Its SHA-256 is
`5ce34a63fde022d296a0584be031a3694a9a8dc410c2e120a417078295b70811`.

On the 6,145-token Phlegm screening pair, packed scoring issued 512 packed
dispatches. The matrix path preserved the final greedy ID with KL `0.00023833`
nats. There were no power or PCI events. The affected tail blocks took 37.359 s
with scalar scoring and 36.043 s with packed scoring. Attention within those
blocks went from 12.575 to 11.786 s. Old-range packing took 25.85 and 42.91 ms,
and sync took 7.68 and 9.68 ms. Full prefill took 212.354 and 213.361 s. This
one-order measurement does not qualify packed scoring as the model default.
`device-6145/result.json` has SHA-256
`7ce504d63cd03f6b8aafb0d509d4eba2b2004d9f012f6bdb480df5e923f6c548`.

The same Q8 image has a matching 512-chunk instruction stream for 16,384
old keys. Its 32 linked cores and hardware initialization match the 128-chunk
stream byte for byte. Program and stack sizes are unchanged and the stream
uses three shim BDs per column. The operator comparison uses the same Q/K/V
values and sink for one 16K launch and four 4K launches in each scoring mode.
It computes an independent FP64 reference across all 24 query heads. The
fixture passed on the board, with maximum relative L2 `4.89e-5` and no power
or PCI events. On the 16K stream, packed scoring took 13.99 and 13.50 ms versus
18.26 and 18.13 ms for scalar scoring. The 4K control drifted by more than 5%.
A repeat with 48 extra warm-up dispatches also exceeded the drift limit and
reversed which tier was faster in its two orders. Packed scoring within the
16K stream showed a gain in both orders. The longer tier itself remains unqualified.
The receipts under `.plans/inkling2/matrix-score-long/` are
`device/result.json` SHA-256 `085259a4002e908ba6a542721f323148450af320fd4f080a173fe9dfa51b798a`
and `warmed-device/result.json` SHA-256
`0e235a802aa3d9e8f6b30e24a40698b143918b195031287dba516c33ceb5c0ed`.

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `matrix-attention-isolation/aliased-peak-simulation.json` | `5ef35cbc8cc24cec779b5fb95c6baf545f0c911dfcce6121f9a2593395ea9b05` |
| `matrix-attention-peak-load/audit.json` | `0d8349342c263ce15db9cdc82bc5bfee5e599bba26d3e4aba761066491f9ce7a` |
| `matrix-attention-peak-load/probe-preflight/input.json` | `58c348f3947f2f69a542e31889a0970f16158ca7033e2d05a274e6208bb59d5d` |

The engine now uses the measured streaming pack for complete, cached old ranges of at
least 4,096 keys when AVX2, OpenMP, and an aligned XRT buffer are available. The existing
packer handles relative, partial, rolling, and unaligned transfers. The streaming path
leaves the unused relative-position stripe intact and fences each writer before the
existing XRT sync. The production source compiled as `inkling2_npu_check`. Comparison
against the previous scalar implementation matched the entire destination in seven
fixtures on both the MSVC AVX2 and Clang portable builds, including an unaligned buffer,
nonzero starts, a partial range, recent keys, and a rolling window. A Phlegm bridge built
from the canonical source passed the protocol handshake with the expected engine and
adapter hashes. The active Phlegm comparisons still use the earlier pinned bridge. A
model run with the new build remains open.

The separate Phlegm activation-pack pair reduced 256-token prefill from 12.79–12.94 s to
11.47–11.59 s, a 10.3–10.4% reduction in both orders. All four runs produced identical
captured logits. Eight-step decode timings moved in opposite directions in the two orderings.
The AVX2 activation pack was already present in the canonical engine source.

| Capture under `.plans/inkling2/f32-phlegm/priority-device-20261123/` | SHA-256 |
| --- | --- |
| `old-key-direct-dma/result.json` | `b24332cb081c798bcb3432b8fc974ccc7b3bab79759c41e1b9e6f61d88079ab0` |
| `old-key-unit-scale/result.json` | `e828bd5c638733bc4df248171afbe70f063b8c569d79eb05b85c949555177a4a` |
| `old-key-quartic-exp/result.json` | `4684fa835f83224594ef2f99c09f98aa26dfb4d1a4859a0ab1a947d9fa4a9d1b` |
| `old-key-paired-keys/result.json` | `456b6346f8994d554f3a9ecbb4484e7e6dae0973ee6f045fbec31afb8e01ac78` |
| `attention-stage-cost.stdout.log` | `d8a63d51aaede11179a8692fdefb81809377cdf138fd76837d1a068833930109` |
| `attention-matrix.stderr.log` | `00c12cd74614d0ca067f03180c93001116e55b347eea694e5b29f6a95047a6d3` |
| `v11-old-key-pack.stdout.log` | `47b874910d4ea9fb53733bddcae5b0780cabb1d1a7d6d6d670510ae3d44c931f` |
| `activation-pack.stdout.log` | `60795ae6594a75d2a42442f4287955d47df8e4931d5020fd26618549e8b2e0d4` |

| Capture under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `host-build-v11/old-kv-stream-npu-build.log` | `619f5960c4ba2d80ebd7cfe29ac5a20dd20b5e8e0df3c3c643afdc263f14434e` |
| `kv-pack-study/canonical-v11-stream/avx2.log` and `portable.log` | `a763a071ab501b862aa840d777d794bcbd38490e8635b6340431b29a29cfcee4` |
| `kv-pack-study/canonical-v11-stream/phlegm-preflight.json` | `9174a29a4238a3370046499844af69b6dde3fcf708abeddc0fdb13d89d395aaf` |
| `kv-pack-study/canonical-v11-stream/phlegm-build.log` | `5b644eddac6fe04e5a1c21d04235072a037e77a0d7c48a02f1800aad5b2eae9c` |

The warmed comparison under `.plans/inkling2/old-kv-paired-keys/warmed-preflight/prepare.json`
has SHA-256 `74a83af98add094175858cae8c5c117ed814c0d6d2f594211ac50f4130b88da1`.

The dispatch estimate put most of the 1M attention work in the two-group 16K stream.
Three diagnostic builds for that stream now isolate DMA and local copies, those stages
plus QK scores, and those stages plus softmax. All three preserve the full control's
instruction stream, control ELF, 67,108,864-byte K/V span, and 36 memory-tile buffer
descriptors. The largest diagnostic program is 6,480 bytes and linked data reaches
byte 63,036 with 4,096 bytes reserved for stack. The comparison warms both arms before
timing and checks the expected peak, mass, and numerator state from the retained fixture.
An offline replay accepted the staged wire outputs, rejected a modified numerator, and
excluded warm-up samples. Device output and timing for these stages remain open.

| Capture under `.plans/inkling2/old-kv-stage-16k/verified-preflight/` | SHA-256 |
| --- | --- |
| `dma/prepare.json` | `47447d6e69bbc7dcfc26f35313d5012d0aeb1f66a0920d0f9d7a10b62fb92098` |
| `scores/prepare.json` | `6a9c63c38ea6dec38adae1e5a34b43c79e6e2d362846ac9fa3d169e39cbd0d0f` |
| `softmax/prepare.json` | `fef7c46b4e0f4a88249fb839e43b11e56c2b1d0557da755017ef811ca297b6fa` |

A separate build processes up to three 16-query groups across a single old K/V tier. The first
three workers in each column each handle two query groups. The fourth handles the third group's
three heads and reads the K/V tile in the adjacent core's memory. A pair of locks signals when the
tile is ready and when the receiving worker has stopped reading it. Each head still runs the same
32-key attention update and sees its keys in the same order. The one- and two-group streams use the
same core image for partial batches.

All nine streams for one to three groups over 4K, 8K and 16K keys share the 32 core ELFs and hardware
initialization. The compiler audit checked the lock order, linked neighbor addresses, runtime control
writes and per-column DMA spans. The largest program text is 9,808 B. Linked data ends at byte 63,036,
the kernel uses a 768 B maximum stack frame with 4,096 B reserved, and each memtile uses 36 buffer
descriptors. MSVC and portable Clang matched all nine stream contracts, seven packed K/V ranges and
the query records at two offsets. The extracted host merge matched the paired layout byte for byte
through nine block sizes and four synthetic tiers, with relative L2 at most 6.85e-8 against FP64.

| Prompt tokens | Paired old-range launches | Three-group old-range launches | Paired old K/V reads | Three-group old K/V reads |
| ---: | ---: | ---: | ---: | ---: |
| 32,768 | 11,520 | 8,640 | 451 GB | 338 GB |
| 131,072 | 145,152 | 108,864 | 8.39 TB | 6.29 TB |
| 1,048,449 | 8,501,480 | 6,375,720 | 559.55 TB | 419.64 TB |

The counts come from the compiled schedules. The 1M fixture would make 9,651,672 total attention
launches, down from 11,777,432 with two groups. At 4K, 8K and 16K keys, the device fixture matched
two plus one groups bit for bit and kept relative L2 below 5.0e-5 against FP64. The two-group
timing brackets drifted by 11.2%, 28.3% and 25.8%. The measurements establish arithmetic parity.
Moving to three groups still needs a stable performance comparison. The paired Phlegm preflight
uses 8,225 prompt tokens and eight forced decode steps. It predicts 784 versus 584 old-range
launches and checks the final 33-row block. Model-level parity and timing remain open.
The staged bridge combines the convolution change with committed-token telemetry. The earlier
convolution-only bridge omitted that telemetry when its host source was replaced. The restored
trace logged commits at tokens 2, 4 and 5 in a native replay of the staged source. Trace-enabled
and trace-disabled logits matched byte for byte.

| Artifact under `.plans/inkling2/old-kv-third-group/` | SHA-256 |
| --- | --- |
| `resources.json` | `4b07bc375bca84f546c56f379fb2f4bbf12ef165b356665ecb0bf22e894b5980` |
| `fixture/plan.json` | `054397851eac8dfe4385ab5fdf886dde0a05b8ce925c67592f9b0096e1817d3b` |
| `merge.log` and `merge-portable.log` | `b79f5badfbb4a4672d35f8b4b1810cd0a67d29af8aa0e35aec57ca33ae541402` |
| `wire.log` | `9634d7a4ed7d72f167a08a8a7c3cf3dc71964fff082ded26d1a052ba13e41d35` |
| `query.log` | `a9b1f0b7d79a6f7a99d036cfecca8a6ac150f2cf196517bc376c60c57ff3c9bf` |
| `phlegm-preflight.log` | `d5bd37d5dc6469fbb572d7fa9984e111df4a429e29b39db14786299fd415a9ae` |
| `trace-on.stderr` | `71b0bad4aed204a7600853835d772a009b32b4cdcb998986971a620b5d004a34` |
| `board.log` | `2c9c46a33b8784765d78857668caa69111f6b5403e2597bf91f804dceabefbb5` |

A host experiment packed the actual BF16 K/V from all eight full-attention visits over two
128-token CPU prefill blocks of the fixed WikiText-2 prompt. Each device-ordered frame stores
512 sign-and-mantissa bytes, 256 bytes of four-bit exponent offsets, and a 64-byte header and
exception table. The 832-byte format holds up to 20 exponent exceptions. Among 4,096 K frames
and 4,096 V frames, the maximum was two exceptions for K and three for V. The tested format
reconstructed all 4,194,304 BF16 values bit for bit. Native scalar and AVX2 encoders matched an
independent Python encoder. Tests covered signed zeros, nonfinite patterns, a full exception
table and rejection on overflow. Portable Clang also passed.

Across seven trials with the first dropped, MSVC copied 8 MiB of already-tiled input in
0.31 ms and encoded it in 0.56 ms. Clang measured 0.39 and 0.64 ms. The packed frames occupy
6.5 MiB, an 18.75% reduction. If a full-context run fits this format throughout, the compiled
three-group schedule would read 341 TB of older K/V. Coverage beyond the captured prompt,
device decode, raw fallback and end-to-end timing remain open.

| Artifact under `.plans/inkling2/kv-lossless/` | SHA-256 |
| --- | --- |
| `coverage.json` | `e0e44fe4fb06902751e5d5d837561a425a257e81e00abc31bf78764a3d1f87f1` |
| `check-msvc.log` | `7d3c915f3923b380562b138c67949edb845f775f4f072d629e012de7e064d5e6` |
| `check-clang.log` | `9aab7db597c9485342692727dc7b155b545f03e912668bd100d9acf9f9a803fc` |
| `check-portable.log` | `47712b24a0429c3739325639e354af5e6d2da7bd27cab80752dfd03c33e81f29` |

The Peano decoder now compiles into all nine attention streams. It expands each 832-byte
input frame into the original 1,024-byte K/V destination before the unchanged attention update.
The nine streams share 32 core ELFs and initialization. The linked image uses 10,352 B of program
text, compared with 9,808 B for the raw image. Linked data ends at 62,844 B, with the same 768 B
maximum kernel stack frame and 4,096 B stack reservation. The 16K tier uses 52 MiB of packed K/V
versus 64 MiB raw, with the same query and output DMA spans and 36 memtile buffer descriptors.

The 49-dispatch hardware comparison reproduced all 8,192 replay frames bit for bit, including the
exception limit and the raw overflow fallback. Raw and packed attention also matched bit for bit
at 4K, 8K and 16K keys. Relative L2 against FP64 stayed below 6.5e-5. The raw and packed median
wall times were 14.50/16.93 ms, 23.24/32.13 ms and 44.20/44.86 ms. Keep the raw route. Full-context
coverage remains open. The input repeats captured model tiles to fill the longer buffers. The
portable native decoder checked all 122,880 frames across the staged replay and attention tapes.

| Artifact under `.plans/inkling2/kv-lossless/` | SHA-256 |
| --- | --- |
| `device-resources.json` | `58fe354bf7f56719d2bd6352f5d4ab158096aa2435013d68db7aca9d051ec27a` |
| `fixture/plan.json` | `d0231b466ba17c47e311275c87c662773d733a9238d32e26f78d272ab133c217` |
| `device-wire.log` | `77b0df548097ad587629409e598caf0774f5686b542b0dbda025decdc80a9568` |
| `board.log` | `d96a76535b77b2bef024f4a54d1378880695569beb38587afdc2067d5a15da7b` |

Host replay compared six K/V packing variants in 12 rotated trials per case, at 4K, 8K and
16K keys and two starting offsets. All variants reproduced the existing tile layout exactly.
The six-thread MSVC path with streaming stores packs 16K keys in 2.85 ms against 6.31 ms for
the serial copy. The test allocates ordinary aligned host memory and overlapped the active 131K
request. The flush replay uses host CLFLUSH as a proxy. Actual mapped XRT sync and dispatch are
separate measurements.

| Packed K/V range | Serial copy | Six-thread streaming | Copy + flush proxy | Streaming + flush proxy |
| ---: | ---: | ---: | ---: | ---: |
| 16,384 keys, starting at 37 | 6.31 ms | 2.85 ms | 6.92 ms | 3.26 ms |

The isolated runtime selects the streamed pack only for a 32-byte aligned destination with
AVX2 and OpenMP. Every writer fences its stores before the buffer sync. MSVC and a portable
Clang build reproduced all seven packed ranges, including the unaligned fallback. The 72-dispatch
mapped XRT comparison matched the retained raw NPU outputs. The streaming pack-plus-sync stage
was 1.82×, 1.97× and 2.27× faster at 4K, 8K and 16K keys by median paired ratio. At 16K,
median pack time was 6.59 ms for copy and 2.81 ms for streaming, while median sync was about
0.24 ms in both arms. Median dispatch took 36.74 and 34.22 ms respectively. Full Phlegm
performance still needs a pair.

The staged bridge can also log each committed token range. Its native five-token replay observed
commits at 2, 4 and 5 with the same complete logits as the trace-disabled run. The recall observer
accepts a split log line, checks contiguous committed ranges and records the exact prefill boundary
before the final response. The bridge provenance, both remaining 131K fixtures and the complete 1M
fixture passed preparation. Those follow-up requests remain pending.

| Artifact under `.plans/inkling2/old-kv-pack-bandwidth/` | SHA-256 |
| --- | --- |
| `host-summary.json` | `69166298033675633580f3143b754aa203348394de1182c93190f1f828c03b23` |
| `stage-check.log` | `94055dd05b2f37059cf794db34dd844a2498c3a0afed39fb02349650285c73bd` |
| `mapped-preflight.log` | `577cd21ada2baaa32fa89758d2fe8c8f3acd169a1bc732fae7cd5ac29bbca900` |
| `mapped-summary.json` | `539248cfb77b63b39fe3fc8a0dadcd46c69b90f5b2dc8babaa08d3a07aaa4894` |
| `recall-preflight.log` | `8f4639445949ff0bad4d8bcbfd1ea069c08fc37ccf352f363d405afd68536d1e` |

| Artifact under `.plans/inkling2/` | SHA-256 |
| --- | --- |
| `attention-stage-cost/schedule.json` | `e3dbd9a23e8c40d3f470be575452ce3b107a6167ade476f60a546cf797c5283e` |
| `attention-stage-cost/preflight.log` | `5229702ba0ab54e07e131ea22956e6cb744342c2aa29da2b6cc59267dc9943b1` |
| `old-kv-frames/resources.json` | `7d3f9aa2d7f88e03bd24ee9071f5e3e30f2875fe8845c6d607d08b0899d493f7` |
| `old-kv-frames/preflight.log` | `cb114bd4b601917a8d71959224231c59f3c3f31064f89ae592be371adc51ec5d` |
| `compact-singleton/check-compact.log` | `15f354c2cc3fad77024b3b3d9405277bf1b656213c0d1ca35c4be2ee56e05c00` |
| `compact-singleton/check-compact-clang.log` | `b452679bff22e7720acb33309e2a21d1d098139c1828ffd91aa969558474102c` |
| `compact-singleton/wire-preflight.log` | `7218a7f8775978d707caff5c336d8e88896be0266e23c8fa23721e95f49289c0` |
| `compact-singleton/phlegm-bridge/phlegm-reference-bench.exe` | `8ff906b33427a56f6fa5971ab46f28051ea78e130e9dfb4082525955a5489fe9` |
| `compact-singleton/resources-complete.json` | `bc42e98c723c544ade7bedc1a3af41f43192e808557ad58c3ecaf3b6e269e2a8` |
| `compact-singleton/bundle/manifest.json` | `43d3be367bc31d467d8a02678b0145f2333851744378a1aaf76e2c92f9d5360f` |
| `matrix-score/host-geometry.json` | `5569fac58323aafebe54d215023653e2b988e0ca5ab0ba641c058d82e5110c1a` |
| `matrix-score-packed/packing.json` | `8814a0c85f66479ff8d41c78d5d9e4e2bb766c43a014022dd0633b4547bd1fbe` |
| `matrix-score-packed/resources.json` | `ea1786d18c5c4d481e42f8b15dd0312976fe17facfd144c430db0f98de139b9a` |
| `matrix-score-packed/check-matrix.log` | `ec1d2be2e81000a9d9b6e53ad8e96d707b7ca4f24c6d33c0b8a6e91ea5a4fd5e` |
| `matrix-score-packed/check-matrix-clang.log` | `f76763420314f6f578075b482ec811793a6fa25f5fbc933a059e89f35c55f96a` |
| `matrix-value/host-geometry.json` | `ad02ec029f11709a2ee808a889b863f9723799f8e178b2bf60214774091a405b` |
| `matrix-value/host-geometry-combined.json` | `b452cae4b83f562026d4d77ddcd5f716fa69f253b2154add461e8689630b4510` |
| `matrix-value/resources-combined.json` | `64a9e2077cac677de5c03bdabb49703a79200c7672b42a078d1d6d2e4aae084e` |
| `matrix-value/preflight-combined.log` | `ad39a4f3897529ea56a50fa5fc129ef4b3e4a6a273695fdc5b2b0cbd35258e6f` |
| `old-kv-query-pair/preflight.json` | `8ffeccf03ee5be36ce64a6d10279de3287ce99ab6532683aa8ca8b14476f4d3b` |
| `old-kv-query-pair/schedule.json` | `4ba972641f0c151ce85ef19774e387b41ec311d8f097195da71b4d1eb9699bab` |
| `old-kv-query-pair/check-io.log` and `check-io-clang.log` | `6403ac2e7c842d9c85ce15ccec35ff08192dc7fbcb638f7be4a1adc619914f27` |
| `old-kv-query-pair/phlegm-bridge/phlegm-reference-bench.exe` | `fb09f7229b68958a24057ec974d1fa7ae7d02f0ee315b1ec0ac5052171825868` |
| `old-kv-query-pair/bundle/manifest.json` | `3dd0b0f26d05adf30fc8936276a4e3a469aae134ea3b058b181ed9c374e93b77` |
| `old-kv-query-pair/phlegm-preflight.json` | `490f6f2440034b0caf10838bdfe9e4760f0d99e74001259cd90f9795d0616154` |
| `old-kv-query-pair/merge-source.json` | `9100579909ece74d3fb27bab4930531b0279e64ee7aef1735041b4a6b738f7e7` |
| `old-kv-query-pair/check-merge.log` | `4d59076d0bb712d4ef9cb8a94f2de65dfa9d898140e0d637bd1c48f101a41242` |
| `old-kv-query-pair/check-merge-clang.log` | `029c88295bd425100f63741ca7a0d02ba69668951625730020241e7459851f21` |
| `old-kv-query-pair/bench-pack-copy.log` | `309a2000cd6cabeebf3ce0791c8a4c83b924a5c3b78783530740915790fc65e6` |
| `old-kv-query-pair/bench-pack-clang-copy.log` | `f85681d9c29938865262ff1c36a7ff83c1c4eb5b9494917254ae7fd0e36ed365` |
| `old-kv-query-pair/phlegm-bridge-copy/phlegm-reference-bench.exe` | `715c36533ca08191c9b12256a9f2ff138e64d480e82b4d6a045adf963d8b685e` |
| `old-kv-query-pair/phlegm-preflight-copy.json` | `ff8269a2080b845d28bb9c016b75d6bace74388bb52b235f492a7b57450977b2` |

The Phlegm HTTP transport probe used the real server and tokenizer with a stub inference backend.
It accepted the 4,356,753-byte request and delivered 1,048,449 prompt IDs and a 128-token budget to
that backend. With the advertised capacity lowered to one token below the prompt length, the same
request returned HTTP 400 before generation. `.plans/inkling2/needle-1m/http-probe-summary.json`
SHA-256 `7c8bdf16c2859f76358be4ed3552e9420f25b8e1dbd5f11d25ed005621e5e738`.

| Local artifact | SHA-256 |
| --- | --- |
| `.plans/inkling2/phlegm-run-1m.out` | `a12b7cb43c9d9134b5bb1b35e9096b66775d9e92e7611d1cc92b02edd6782a87` |
| `.plans/inkling2/phlegm-inkling-response.json` | `c404b9ebf3be4d573c707cf31fd858683202c17701c4f2bc9ef492c427dde601` |
| `.plans/inkling2/phlegm-inkling-stream.txt` | `23689b4e1c178fcda708719c0a9bd16ed91f1dc3a201c70add86a549fbdf32d9` |

`phlegm bench` rejected the development build at its exact-release-tag gate. The short bridge trial covers
the protocol. Publishable timings and the full context matrix are open. Trace `.plans/inkling2/phlegm-bench-attempt.log`
SHA-256 `0571c7bf55a976c60caec47768e2cd7e34066c5484f13b16b5887246d8e57e76`.

Build the host probe with CMake target `inkling2_host_check`. The focused prefill test runs with
`INKLING2_HOST_CHECK` pointing to that executable and `INKLING2_IQ4_GGUF` pointing to the pinned weights.
For the on-device cases, build `iq4_gemv.py` with K=2,304 or `INKLING2_GEMV_K=1280`, use
`designs/inkling2/make_test.py` to pack the named GGUF expert, run its `run.cfg` through XRT and rerun
`make_test.py --verify`. These separate projection images establish arithmetic on real weights.

### Runtime contract from the reference

- Each visit holds its own state. Repeated visits look up the same physical weight tensors. Passes start from the
  same prelude residual and prelude block list. On passes 2 and 3, each layer mixes the previous pass's two
  block outputs into four messages, then applies `(1 - tanh(forget)) * x + tanh(gain) * message` to the Q, K, V
  and MoE inputs. The rank-64 down projection sees the concatenated RMS-normalized input and message.
- For a query at position `p`, the content logit is `dot(Q, K) / sqrt(128)`. The learned term is
  `dot(R, table[p - key_position])` for distances in `[0, 1024)`. Full attention retains older keys with zero
  learned term. Sliding attention masks older keys. Full attention scales Q and R by
  `1 + 0.1 * log(max(1, (p + 1) / 8192))`. The sink adds a query-dependent logit with a zero value vector.
- Each four-tap convolution reads its three prior inputs. Static taps are stored oldest first. Dynamic K/V taps
  are projected from that token's corresponding normalized K/V input, with the current-token tap first.
  The residual-plus-convolution result is scaled by `1 / sqrt(5)` for static and `1 / sqrt(9)` for dynamic.
- Each routed expert has a fused gate/up source tensor with GGUF dimensions `[2304, 2304, 128]` and a down
  source with dimensions `[1152, 2304, 128]`, fastest dimension first. The fused rows store the gate first.
  IQ4_NL layers 0–19 use nonlinear signed codes and a half-precision scale per 32 values. Layers 20–23 use
  Q8_0. GGUF lookup and NPU packing must preserve the expert dimension when flattening rows.

## What is new to this repo

The closest routes are `gptoss` (sink, sliding window, top-k SwiGLU experts, `attn_fa_gx`) and `gemma4` MoE (128
experts top 8, softcap 30, `mb4` expert batches). These parts have no kernel or engine support here:

1. **The loop.** No family runs a layer more than once. The weight pool stays 24 layers. KV, conv state and the
   manifest's layer table become per visit (48). This touches the recipe/manifest contract, so writer and reader
   change in one commit.
2. **Relative-position logits instead of RoPE.** A learned bias on the scores. Every NPU attention kernel here
   assumes RoPE on q/k. Needs a new score term in decode attention and in `attn_fa`.
3. **Dynamic K/V short convolution** in the full layers (decode state of 3 rows a visit). LFM2 has a static short
   conv; this one is input-dependent.
4. **Loop controller**: rank-64 gates, message queries, the depth attention over residual blocks.
5. **IQ4_NL experts.** Same 18-byte block as Q4_0 with a 16-entry value table. The host GGUF reader reads and
   dequantizes its rows, including the outer expert dimension. The NPU pack and routed decode kernels run on
   IQ4_NL and Q8_0, including the block prefill expert GEMMs. Supporting
   Unsloth's mainline UD-Q4_K_XL would require an NVFP4 expert path.
6. **Expert epilogue**: gate bias, two tanh softcaps, route scale 8.
7. **DSpark**: a block drafter with multi-tap target features, close to DFlash 2 (`recipes/dflash2.py`).
   Capture the pre-visit inputs for chronological visits 5, 23 and 41 (physical layers 5, 11 and 17),
   then the final normalized LM-head input. The pinned mainline llama.cpp builds the target's looped
   graph but does not populate its layer-input outputs for Inkling. Use the Unsloth DSpark build and
   root GGUF for a GGUF drafter oracle, or the SGLang target and drafter capture.
   [Unsloth GGUF layout and DSpark support](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF/blob/main/Mainline_Support/README.md)
8. **The vision tower**: `inkling2_vision` is not one of the towers OPEN-VISION-VIT-CONFIG accepts.

Cost estimate (inferred, not measured): at IQ4_NL one expert is ~4.5 MB, so a decode token reads ~36 MB of experts a
visit, ~1.7 GB over 48 visits before cache reuse across passes. That is about 2x gpt-oss:20b's expert stream, which
decodes at 11 tok/s here. Expect decode under 10 tok/s before a drafter.

## Spec impact

Requirements in [the open-engine spec](../spec.md#open-family-inkling2-fixed-loop-inkling-2-mini-from-gguf):
- **OPEN-FAMILY-INKLING2**: validates fixed-loop geometry from the GGUF and runs on its own recipe;
  greedy agreement with the reference; the `## Production bar` table.
- **OPEN-PREFILL-INKLING2**: block prefill, IQ4_NL expert GEMMs, the relative-position flash attention.
- **OPEN-ATTN-RELPOS**: the relative-position score term and log-n scaling in decode and block attention.
- **OPEN-SPEC-DSPARK**: the DSpark drafter on the verify pass (OPEN-SPEC-VERIFY), if it ships.
- **OPEN-VISION-INKLING2**: the tower and projector on the NPU from the mmproj GGUF, the full 40-4096 px range.

Related requirements **OPEN-VISION-VIT-CONFIG**, **OPEN-VISION-EMBED** and **OPEN-SPEC-DERIVE** point to the
dedicated Inkling projector and fixed-loop metadata contracts.

## Decisions for you

1. **Source GGUF.** Bring up ggml-org IQ4_NL against pinned mainline llama.cpp. Its expert types are IQ4_NL and Q8_0.
   Compare with Unsloth's mainline UD-Q4_K_XL before selecting a shipping quant. Its NVFP4 expert tensors require a
   new pack and kernel path.
2. **Context target.** The model states 1,048,576. Full-attention KV is 8 visits (layers 5 and 23 once, 11 and 17 three
   times) x 4 KiB a token: 4 GiB at 131k, **32 GiB at 1M** in bf16. That fits the 88 GiB HX PRO 370 beside 17.6 GB of
   weights. gpt-oss:20b took 31 minutes to prefill 130k here; 1M is quadratic, so many hours. Recommendation:
   capacity 1M, all gates at 8k / 32k / 131k, and one 1M needle run with its time recorded. Alternative: a q8 KV.
3. **Vision range.** 4096 px gives ~42k patches before the 2x2 merge (~10.5k soft tokens). Recommendation: support the
   full range; gate at a size ladder (40, 224, 448, 896, 1792, 4096) and record the cost of each.
4. **Tracker.** Adding the model to OFLM #240 is OFLM's decision. Recommendation: add the row to
   `RECOMMENDED_MODELS.md` now, marked "not on #240 yet", as Gemma 4E is.

## Steps

0. **Reference and facts.** Download ggml-org IQ4_NL, Q8_0 and BF16, the mmproj and the DSpark GGUF (record revisions
   and sha256). Dump each tensor table. Read transformers' modeling file and llama.cpp's mainline graph for the arch;
   write the layer, loop and attention equations into a family spec section. Build the reference (R-0004): llama.cpp
   mainline on BF16 for logits and greedy text; transformers on CPU for intermediate activations per visit. Gate:
   the two references agree on 64 greedy tokens and per-visit hidden states.
1. **Source and quality.** KLD and perplexity on a long real document for ggml-org IQ4_NL and Q8_0 and Unsloth
   UD-Q4_K_XL (if `Mainline_Support/` gives a mainline file). Record the choice for decision 1.
2. **Host-first route.** `recipes/inkling2.py` and the engine reader: derive the spec, per-visit KV and conv state,
   the loop controller, relative-position attention and the experts, with the expert GEMVs and dense projections on
   the NPU and the new parts on the host. Gate: greedy equal to the reference on the chat prompt; per-visit hidden
   states within tolerance.
3. **IQ4_NL experts on the NPU.** The value table in the decode GEMV and the prefill expert GEMM (`mb4` or the
   `gptoss` MX path as base), with the gate bias, softcaps and route scale in the epilogue. Read R-0001, R-0009,
   R-0012/R-0013 first. Gate: bit identity or KL against the host experts.
4. **Attention on the NPU.** Decode attention with the sink, relative-position term, log-n scale, L2 clip, value norm
   and output gate, at depth (KV windows per core as `gx`). Block prefill attention as an `attn_fa` variant to 1M rows.
   The dynamic K/V convolution and the controller gates next to it. Read R-0003/R-0008 and R-0005/R-0011 first. Gates:
   the accuracy gate at window boundaries (1024 rows, 8192 for the log-n floor), greedy equal to the reference.
5. **Block prefill.** Every op family on the block route, the loop included. Gate: logits equal to the decode route
   within tolerance; greedy equal to the reference.
6. **Context.** Needle at 8k, 32k and 131k, depths 10 / 50 / 90 %, plus one needle at the full 1,048,576-token
   capacity with wall time recorded. Check for finite results and inspect KV and conv states at visit boundaries.
7. **Near optimal.** `dibs run --measure`, performance mode (R-0006). Decode per stage at 0 / 2k / 8k / 32k / 131k;
   prefill per op family at 512 / 2k / 8k / 32k. llama.cpp Vulkan on the same machine as the baseline. Name what is
   left, largest first. Measure expert reuse across the three passes of one token before tuning.
8. **Host stages.** A table of each host stage and its cost: router, controller, sampling, the tokenizer, image resize.
9. **Drafter.** Prompt lookup first on the verify pass (cheap baseline). Then DSpark: dump its GGUF, read Unsloth's
   and SGLang's DSpark code, reuse DFlash 2's taps-and-block machinery with four taps. Matrix as OPEN-SPEC-LOOKUP: SPEC
   IDENTICAL on every run, acceptance and speed at 0 / 2k / 8k. Ship the best, or reject each with the measured reason.

   The target now returns logits for every candidate row while retaining the previous position, per-visit KV and
   convolution histories. Accepting a prefix copies its KV rows and reconstructs the three-row convolution histories
   from the prior cache and the accepted inputs. The CPU regression used the ggml-org IQ4_NL model. Accepting 0/4,
   1/4, 2/4 or 4/4 candidates gave byte-identical follow-up logits to replay, including a five-token follow-up block.
   It also checked rejection of invalid operations and reset during a tentative pass.
   `.plans/inkling2/verify-host.log` SHA-256
   `fe1cda7d945011e3af4bd1abf173308cc9017e47facd3160d46e1aa779a086eb`.

   The Phlegm bridge and CLI compiled with protocol 2. On the block route the bridge offers four verify rows per pass
   (one anchor and up to three lookup proposals) and uses Phlegm's existing lookup policy and sampler. The default
   remains speculation off. The companion patch is in `.plans/inkling2/phlegm-verify/verify.patch` with SHA-256
   `d33bc36a96b2fda2d405aad2b035de7e3feb654b694963907594a8c1b9bf93d2`.
   The first paired v8 NPU run used `phlegm serve` with a 20-token repetition prompt and a short code prompt, 48
   generated tokens each. Output text, streamed token bytes, finish reason and usage matched in both modes.
   The rates measure the time between the first and last streamed token with logprobs enabled.

   | Prompt | Off (tok/s) | Lookup (tok/s) | Proposals accepted |
   | --- | ---: | ---: | ---: |
   | Repetition | 2.09 | 1.91 | 30/32 |
   | Code | 2.16 | 1.72 | 26/35 |

   Reversing the order again matched streamed tokens and reproduced the slowdown. Repetition ran at 2.13 tok/s
   off and 1.94 with lookup (30/32 accepted). Code ran at 2.18 and 1.73 tok/s respectively (26/35 accepted).
   Longer-context runs, exact token IDs, and verification at the sliding-window rollback boundary remain open.
   The two artifacts under `.plans/inkling2/phlegm-verify/` are `lookup-pair.jsonl` SHA-256
   `7c8b0096dbd94a6189c66cc9302110cf96cc41b31a09a8f96143a83f9076a9d3` and
   `lookup-pair-reversed.jsonl` SHA-256 `c3e5f2867083fc4a00f2239593d751b9387ca39f0fa27f21608569178d49f9ae`.

   Target prefill and verify can now return the four DSpark features for every requested row. The real IQ4_NL CPU
   check matched all captured features against direct replay for the accepted rows, checked the three pre-visit
   features against the layer observer, and confirmed that all tentative features were finite.
   `.plans/inkling2/verify-dspark-host.log` SHA-256
   `70ed1f4a9d8bf10316204b5e278b674bee4ecc1edf21f6d92f1c3ffbf87b62c7`.

   A second IQ4_NL CPU check kept two independent committed targets. One verified eight input
   rows and accepted the specified prefix, and the other appended only that prefix. All nine
   acceptance lengths from zero to eight matched all accepted features and the ordinary decode
   logits after the follow-up tokens. Each nonempty prefix also matched the logits at its final
   accepted row. Rejected operations left committed state untouched.
   `.plans/inkling2/verify-matrix/all-accepted.log` SHA-256
   `3b592c9e5f6a950f11d27ba0002f6bff64b937d592cc7a568172bdf4ab915fdb`.

   The same IQ4_NL CPU verifier prefilled both independent targets with 1,023 WikiText tokens
   in four chunks. It accepted 2, 0, 7 and 8 of eight tentative rows at positions 1,023, 1,030,
   1,035 and 1,047. The accepted features, final accepted logit row when present, and next decode
   row remained bit-identical to direct appends after the wrap. This covers target rollback on the
   CPU. The Phlegm NPU comparison below covers the acceptance lengths observed in real drafting.
   `.plans/inkling2/verify-matrix/window.log` SHA-256
   `161202fc3a97e584490c8be07c492851e7ed70e52836d4ba978f1e81812d77e1`.

   The pinned mainline text reference cannot load the released Inkling DSpark export. Its generic DSpark loader
   requires `markov_w2.weight`, while Inkling's recurrent correction uses `dspark_hidden_out.weight` and the
   shared LM head. The official Unsloth release's mixed source asset is commit
   `883286076efe5b85fddb31d9cc7035a6b698ba9b` (archive SHA-256
   `92ee8439f9ae51a40f15c168ac64e1d657bf8b08d301c96f8fb2d58625573607`). It implements four learned
   feature mixtures, distinct K/V projections for committed context, a noncausal eight-position block, and the
   recurrent correction. [Unsloth release](https://github.com/unslothai/llama.cpp/releases/tag/b12588-mix-c0c3de8)

   A standalone Windows CPU probe injected the open-kernels IQ4_NL target's real features into the published
   Unsloth DSpark graph. It proposed seven tokens per cycle and accepted 2, 0 and 0 across three cycles on a
   five-token prompt. The six emitted greedy IDs matched ordinary target replay. Draft calls took 135, 84 and
   83 ms under concurrent NPU load. The oracle borrows the token embeddings and LM head from Unsloth's
   UD-IQ1_S target, which differ from ggml-org IQ4_NL. The local probe and captures are in
   `.plans/inkling2/dspark-oracle/`. `probe.stdout` SHA-256
   `f06def523cb6dfaae5e4de47e2ec6430b5734c7687fe0443175726189b42fa94`. An independent F64 calculation of
   the fourth feature mixture and its normalization matched the oracle's recurrent context within
   `5.5e-8` relative L2 at all three accepted boundaries. All captured proposal logits were finite, and their
   argmax IDs matched the probe's proposals.

   The native C++ CPU drafter matched all 21 greedy proposal IDs at positions 5, 8 and 9 using those same
   IQ4_XS embeddings, output weights and captured target features. Across all 200,058 logits per proposal,
   relative L2 was 0.00466–0.02189. It checked that proposing and committing zero rows leave the context
   position unchanged, and that reset reproduces the first proposals. Drafting seven tokens took 0.60–0.63 s
   with four CPU threads in this run. The native IQ4_NL target also passed its CPU verify and prefix replay
   test after the shared helper changes. The draft check covers one short prompt and the oracle weight set.
   `.plans/inkling2/dspark-oracle/native-check.log` SHA-256
   `4b3cf8d6f1e73039a2c59544c09b9bf61c503c9dc7a9913e80a1376e39f0ab00`.

   The extended native draft check retained all 21 oracle matches and replayed the draft cache
   across position 1,024 using repeated captured target features. Accepting eight of 16 offered
   features from position 1,023 produced the same following draft logits and token IDs as
   directly appending the committed features. This isolates the draft cache. The later Phlegm
   bridge comparison exercises the target and drafter together at this boundary.
   `.plans/inkling2/verify-matrix/draft-window-final.log` SHA-256
   `bf21af1780495cd107da2cf820dde63da2954f6d2ce0695db95524de32e220b4`.

   With the actual ggml-org IQ4_NL target, a native CPU replay committed 3, 1 and 1 input rows over
   three drafts, including each anchor, and matched six plain greedy output IDs. Seven proposals took
   1.18–1.21 s at four threads during a concurrent 32K Phlegm recall test.
   `.plans/inkling2/dspark-oracle/native-replay.log` SHA-256
   `25f2b63061fb15b0f7783850d8bae591136193c129d83851f4b9b66ca731a75d`.
   A separate instrumented replay under the same concurrent workload spent 0.82–0.88 s projecting
   through the shared Q8_0 LM head, 71–73% of draft time. The backbone took 0.29–0.34 s.
   `.plans/inkling2/dspark-oracle/native-profile.log` SHA-256
   `ad7ca83ca18a9c30d5f98aa5bdc7a9843142610f9f1b46678f1432d78f4ff5d2`.

   The AVX2 route reads Q8_0 blocks directly from the mapped GGUF. On a fixed synthetic activation with
   the actual head, the six warm passes took a median 10.05 ms each versus 55.01 ms with row dequantization,
   at four CPU threads during the 32K recall run. A four-lane prototype was bit-exact and took 17.59 ms.
   The chosen eight-lane reduction preserved all 42 proposal IDs and reached at most `3.55e-7` relative L2
   against the scalar graph over every vocabulary row at positions 5, 8, 9, 1,023, 1,024 and 1,025. The
   longer case repeats real saved target features to cross the draft's 1,024-token sliding window. The
   IQ4_XS oracle path still matched all 21 captured proposals. A non-AVX build reproduced the scalar
   logits byte for byte in all 42 proposal rows. The optimized native replay with the actual IQ4_NL
   target preserved 3, 1 and 1 committed input rows and six greedy output IDs. The Phlegm CLI and
   AVX2 bridge compiled and passed handshake and provenance. End-to-end NPU timing remains open.
   `.plans/inkling2/dspark-oracle/head-parity-results.json` SHA-256
   `f79893306fc997c4f8c88904b87d5a0561706b166a074db36f4cc7fda4d4e765`.

   Phlegm's protocol 2 bridge now advertises DSpark when block verify is available and `dspark.gguf`
   sits beside `model.gguf`. `--speculative auto` selects it for text and image requests. The bridge
   captures prefill features and appends only accepted verify rows to the draft cache. The default
   `--speculative off` path skips draft loading and feature capture. The C++ and Rust builds passed,
   as did the bridge handshake, provenance check, six GGUF tokenizer tests and the speculation policy
   tests. The first Phlegm hardware pair used the scalar draft head, greedy sampling, six host threads
   and 32 output tokens per case on the v11 bundle. Off ran first and auto ran in a separate resident
   process. Streamed token bytes, text, finish reason and usage matched in both cases.

   | Case | Prompt tokens | Off tok/s | Scalar DSpark tok/s | Accepted / proposed |
   | --- | ---: | ---: | ---: | ---: |
   | Repetition | 20 | 1.310 | 1.201 | 21 / 29 |
   | Code continuation | 83 | 1.431 | 2.280 | 27 / 27 |

   This pairing measured a slowdown on repetition and a speedup on code.
   `.plans/inkling2/dspark-oracle/phlegm-pair.jsonl`
   SHA-256 `c9c758f04c56ca1ea643c8dfd41d980de9b2d895e386337f4b3c0b050ed265c6`.

   A second off-first pair used the fused AVX2 draft head and optimized target host projections.
   It preserved streamed token bytes, text, finish reason, usage and the scalar draft acceptance counts.

   | Case | Off tok/s | Optimized DSpark tok/s | Accepted / proposed |
   | --- | ---: | ---: | ---: |
   | Repetition | 2.045 | 1.630 | 21 / 29 |
   | Code continuation | 2.455 | 3.164 | 27 / 27 |

   Decode improved 29% on code and fell 20% on repetition. Phlegm keeps DSpark opt-in while broader
   workloads and reversed order are tested.
   `.plans/inkling2/singleton-host/phlegm-dspark-pair.jsonl` SHA-256
   `632f58583eb2394b6529d908ad58517723846e4c11711d8ff7d082611c427584`.

   The v11 Phlegm bridge replayed actual DSpark proposals at prefixes of 5 and 1,023 tokens.
   For each prefix, all 24 greedy output IDs matched plain decode. At 1,023, the trace crossed
   the target's sliding-window boundary on new output. It accepted 14 of 56 draft proposals,
   and the five-token trace accepted 12 of 56. The harness checked that each verification kept
   the committed position and that accepting a prefix advanced it by precisely that length.
   This is an observed-proposal test with the AVX2 drafter and query-cache experiment. Longer
   contexts, the forced acceptance matrix on the NPU, and paired timings remain pending.
   `.plans/inkling2/query-pack-reuse/phlegm-dspark-window.log` SHA-256
   `ea975ed5c24e630ee6b9f9c69b5ce47adbca611799a2ba701ccb36e06d298a83`.

   The host now shares each packed Q8_0 weight vector across four input rows for blocks of up to eight
   rows. The kernel preserves the four accumulation chains per row. Three GGUF drafter matrices matched
   dequantization and scalar dot products byte for byte at widths 1, 2, 5, 7, 8 and 9 under MSVC and Clang.
   The candidate and control produced identical full logits and all 42 proposal IDs at positions 5, 8, 9,
   1,023, 1,024 and 1,025 with the ggml-org target head.

   In one-process testing at four host threads, each position ran eight alternating pairs of seven-token
   drafts. With each first pair discarded, the per-position median ratios were 2.51–3.22× for the
   backbone and 1.16–1.48× for the full draft. A separate cross-process comparison had mixed timings.
   Both tests overlapped the Phlegm vision run. The improvement is measured for the CPU draft and
   its impact on Phlegm decode throughput remains open.

   The rebuilt drafter passed the independent Unsloth oracle and its sliding-window test. The rebuilt
   target passed all nine accepted lengths of an eight-token verify block, and the Phlegm bridge
   passed handshake and provenance. The Phlegm device queue is still using its earlier build.

   | Capture in `.plans/inkling2/q8-draft-projection/` | SHA-256 |
   | --- | --- |
   | `paired.log` | `9ce8199e62dcfeee7b45fb60113c41672e0d8e9942018d816edaff8d5f2ff54d` |
   | `replay-pair.log` | `d1175daec5251354fd5dab56121d89497ce386edda066eefff875ec078c4c3e7` |
   | `matrix.log` | `a10dbdf4de1c75f1f09c7fb5b1101394f086ba66afca8a18e2ab4ac9aea62f46` |
   | `matrix-clang.log` | `53ebc631839b66581267f34d0d8a49e3e8a782a9748bcbb3f22eeb7a1e56ab77` |
   | `native-oracle.log` | `dbecab473abe919703825a9eba2a26df3e0a1bc52ed5a63b4bd835ff91b1b0c3` |
   | `native-target-verify.log` | `3b592c9e5f6a950f11d27ba0002f6bff64b937d592cc7a568172bdf4ab915fdb` |

   The AVX2 projection now unpacks Q8_0 weights once per complete eight-row block. The previous
   implementation unpacked them once per four rows. On the real DSpark backbone, the two routes
   produced bit-identical logits for all 42 proposals at positions 5, 8, 9, 1,023, 1,024 and
   1,025 under both MSVC and Clang. The six-thread MSVC replay measured median paired speedups
   of 1.05–1.16× for the backbone and 1.02–1.13× for the full CPU draft across those positions.
   This was an in-process comparison during concurrent NPU vision QA. The Phlegm throughput
   effect remains unmeasured.

   The canonical build matched all 21 independent Unsloth proposal IDs, passed the draft window
   test, and passed all nine target acceptance lengths. The rebuilt Phlegm bridge differs from the
   traced F32 build only in `inkling2_cpu.hpp`. Its protocol and source checks passed. The queued
   forced NPU acceptance test now uses this rebuilt bridge. The first device attempt stopped
   when the harness sent an invalid command during a live tentative block. The bridge reported
   the error and exited with its documented fatal-request policy. The retry runs the full valid
   matrix first, then checks each rejected command in its own process.

   The retest on the Ryzen AI 9 HX PRO 370 passed 18 rollback cases in Phlegm. At prefix
   lengths 5 and 1,023, changing the rejected suffix preserved every accepted logit row,
   next draft ID and logit row, and subsequent forced decode row bit for bit, for acceptance
   lengths 0–8. Both invalid requests were rejected in fresh processes. The run took
   527.78 s and Windows logged zero power events and zero PCI context errors.

   Comparison with direct decode still found differences. At prefix 5, two intermediate
   verify rows changed the highest-scoring token. All nine cases retained their next draft
   IDs and subsequent greedy IDs. At prefix 1,023, one of the nine cases changed the first
   proposal after accepting three tokens and changed the top ID on the first subsequent
   forced decode row. The latter comparison used identical input tokens and reached KL
   divergence of 0.02361261 nats. The draft comparison after the divergent proposal
   conditions on different tokens. Future captures compare only the common draft prefix
   and record the competing IDs and their logit gaps at every decision change.

   The follow-up replay brings each arm to the same committed state at position 1,032
   and checks the preceding logits byte for byte before comparing the three accepted
   tokens. The archived direct and batched preludes already had different logit hashes
   at this position. Replay each independently, including the earlier draft-and-reject
   calls, and check the accepted logits, draft IDs and logits, and subsequent logits
   against the archive at all three prelude checkpoints. The corresponding original arm
   also checks its position 1,037 endpoint against the archive. On each starting state,
   compare full-block verification with partial acceptance, appending only
   the kept block, direct singleton decode, and singleton verification with full
   acceptance. A trace captures nine stages at each of the 48 visits for each kept row.
   This locates the first differing stage while separating truncation from batched
   arithmetic. The instrumented bridge passed provenance and handshake checks with
   both direct and batched preludes. Compared with the forced-rollback bridge, only
   `inkling2_host.cpp` changed. Mock runs reproduced both original call sequences and
   endpoint receipts, checked the committed token path in all eight arms and rejected
   a changed prelude receipt. Both hardware replays are queued after the 512-token
   prefill profiling. Bridge SHA-256
   `5aea98cfff00801abf38ffda8f3babb7d527ff369bb20c62b37fa823eb2dc87e`.

   The on-device replay from the archived batched prelude reproduced the original
   endpoint. Full-block verification and a block containing only the three kept
   tokens produced identical stage traces, last accepted logits, next draft logits,
   and subsequent decode logits. Singleton verification and direct decode matched
   in all of those comparisons, including all three accepted rows. The first
   batched-versus-singleton mismatch occurred at the raw query projection of visit 0
   for each kept token, with bit-identical normalized query inputs. The first draft
   decision differed even though the two subsequent forced decode top IDs matched.
   The replay recorded no Windows power or PCI events. The replay from the archived
   direct prelude also reproduced its original endpoint and the same initial
   differences in the query projection. Singleton verification again matched direct
   decode bit for bit. On that prelude the batched route changed subsequent logits
   with maximum KL `0.00328128` and kept the same subsequent top IDs. That run also
   recorded zero power or PCI events. Its `aligned-direct/result.json` receipt
   has SHA-256 `d148dc28ee4f83c68ed3edf67db35bdaaf48c3ff85fc8ebae2ee0f50e5fa2f59`.

   An independent FP64 projection dequantized the real `blk.0.attn_q.weight` and
   replayed the three layer inputs. Singleton CPU outputs matched the unrounded
   projection to relative L2 below `4.1e-7`. Batched NPU outputs differed by
   `3.79e-5` to `4.76e-5`. Applying the NPU's per-32-value signed int16 activation
   pack in the reference reduced the error to `2.13e-6` to `2.20e-6`. The packed
   reference reproduced the observed difference vector to relative L2 of 4.7–5.7%.
   That localizes the initial drift to projection arithmetic, mostly activation
   rounding. Later expert and attention differences still need isolation before a
   precision change can be qualified across the draft and direct paths.

   Receipts under `.plans/inkling2/dspark-aligned-trace/aligned-device-20261123/aligned-batched/`
   are `result.json` SHA-256
   `a7c6611027a01367fb9b9996c8e1a564109fe231bd1e59d67e1aad0eb32a348b`
   and `query-rounding.json` SHA-256
   `1faefcf4f7007e7d63dc47794684cf0ed117a471a55ee42b9b119d69c0a009a3`.

   A staged follow-up gives each row of the failing verify block the same host Q8
   projection arithmetic as direct decode, for both attention inputs and output.
   Its earlier calls still replay the archived prelude. The block still uses batched
   attention and experts, so the next differing stage will show whether matching the
   dense projections is sufficient. The bridge built and passed protocol and
   source checks, with only `inkling2_host.cpp` differing from the aligned replay
   build. One device replay on the direct prelude is queued behind the current
   comparisons. It compares against the saved singleton trace and logits and limits
   draft scores to their common token prefix. The staged bridge SHA-256 is
   `16fb8d41dac9f9afc470f1e9ca6c64707e52b48a7be12ca466173bb2ef2a5302` under
   `.plans/inkling2/dspark-verify-dense-parity/`.

   The targeted NPU replay completed with no power or PCI events. It matched all
   six traced input stages through the raw query, convolved key/value, and relative
   query. The first difference then appeared in visit 0's attention output for
   each kept row, with maximum absolute differences `7.53e-5` to `1.57e-4`.
   The two new draft proposals matched direct decode, but the first subsequent
   greedy ID changed from 197045 to 82. The dense-only change therefore misses
   the semantic parity gate. Compare singleton and batched attention on the
   captured same-state inputs before changing the verifier's normal route.
   `device/result.json` under that directory has SHA-256
   `0426913adeeed5237682175902bb52b86f593633942e308dd00c854b90d4e1f7`.

   The operator probes use the captured first-visit queries and pending
   keys and values with a deterministic cache on the v11 Q8 image. For position
   1,032, running the exact singleton key span on either the 32- or 64-chunk
   stream gave identical raw numerator, peak and mass. Extending the span to the
   batched end added one chunk with no visible keys for its first row. The peak
   stayed exact, the mass changed, and the new numerator matched an exact
   BF16 high/low re-split of the previous numerator. Zeroing the future K/V
   payload preserved the same result. Independently, absolute 32-key boundaries
   removed a second-row difference when the visible rows shared the same leading
   and trailing chunks. A second fixture at position 1,054 crossed both a trailing
   chunk and the next oldest-window boundary. It reproduced the no-key state
   update even with fixed boundaries on two rows. Neither run recorded a power
   or PCI event. These are operator fixtures with a synthetic completed cache.

   The correction in `.plans/inkling2/dspark-attention-chunks/` preserves the previous
   peak, mass and numerator for every row with no visible key in the chunk. Both
   32- and 64-chunk Q8 streams built into the same 32-core image. The correction
   made the first row exact with and without future keys in a separate chunk.
   With absolute boundaries, all three singleton and batched rows also matched
   raw state exactly at both tested positions, including the case crossing a
   leading and trailing boundary. The original singleton active row remained
   bit-identical and both runs logged zero power and PCI events. The two corrected
   receipts are `fixed-1032/result.json` SHA-256
   `e8862d536cf7848f42e0fe4b3e5de23b44a16cd5c6eb95b7bb7651cf251279c6`
   and `fixed-1054/result.json` SHA-256
   `ce46b54a74124221bf5938d97e1e358ecea7f8fcc24e62578999c955b0b2e103`.
   The original operator receipts `device/result.json`
   and `device-boundary/result.json` have SHA-256
   `e8901a45de17e3f168d356a298069fc60b4efe22113693ecaf44abbe32a15b91`
   and `6186bab2afe6c96fa697f63cdf706c0b9c0136d43180ab38d2cf3ea679a8392b`.

   The follow-up Phlegm diagnostic keeps the archived prelude on its original
   route and switches the short traced block to the verified Q8 attention image
   with absolute chunk boundaries. The two post-prelude arms both use the new
   attention and the verify arm retains the matching singleton dense arithmetic.
   It checks the full stage trace and subsequent target decisions against a fresh
   direct replay. The build and two short NPU replays follow the packed-score
   timing pair. The extra context in this diagnostic makes it a correctness test.

   The direct-prelude replay passed. All traced stages at the 48 visits, all
   accepted logits, the next draft proposals and logits, and both subsequent
   decode logits matched bit for bit between the two arms. Both reproduced the
   original prelude hash. The run logged no power or PCI events. This closes the
   reproduced failure for the staged diagnostic. The other archived prelude
   also matched the full trace and every accepted, draft and subsequent logit
   bit for bit, with zero power and PCI events. `model-device/result.json` has
   SHA-256 `a2dc45641a198753acbfc89bc79841269c6f5f3b9a5e99e0ca2d3cec1f96c053`.
   `model-device-batched/result.json` has SHA-256
   `8a6459762b4c593079644d3ce91d58f0581b473127b83030d01f82b7f837b61f`.

   The normal shared-image build now exposes the correction via a guarded
   attention entry point. The recipe emits `attention_chunk_alignment: 32`
   when both quant images have that entry point. The reader selects absolute
   chunk boundaries and matching dense arithmetic during verification only
   when the field is present. Bundles without it retain their original mode.
   The normal Phlegm bridge compiled, and all four native reference tests and
   the canonical import audit passed. Repackaging the legacy compiler outputs
   produced the original manifest byte for byte, with no alignment field.
   The new bridge loaded the archived v11 bundle and matched all 200,058
   512-token logits, with zero power or PCI events. Its
   `legacy-512/result.json` has SHA-256
   `8c10b20be75e1f5db643cb62faea8779ff52d889334066cceef9fff0bf65ce1e`.
   All 16 shared streams rebuilt and packaged through the canonical recipe
   under `.plans/inkling2/aligned-attention-v11/`. Maximum program text was
   9,520 B, linked data ended at byte 61,532, the attention frame was 1,408 B
   inside an 8 KiB stack, and every column used three single-descriptor shim
   tasks. The optional alignment field passed the reader and operator
   checks. The two shared images and the manifest were the three changed files
   among the 49 packaged files. All instruction streams were byte-identical
   to the saved v11 source bundle. Both window-boundary operator receipts
   reproduced the earlier exact-state test results byte for byte.
   `bundle-evidence.json` has SHA-256
   `5a1df8453304de9c25ab8a4eeb2ff2a8f41c47c425adaeb038b4fda4332d2194`.
   The normal bridge reproduced all 512-token control logits bit for bit with
   the aligned bundle, with no power or PCI events. `aligned-512/result.json`
   has SHA-256 `9b75c3765bb5abc5e0035e9721c311291ca7f8e268d232ec7d8d4bb9d1136979`.
   The normal route passed all 18 forced acceptance and rollback cases at
   prefixes 5 and 1,023. Every accepted logit, next draft proposal and logit,
   and subsequent decode logit matched direct decoding bit for bit. Changing
   the discarded suffix preserved every comparison. The bridge also rejected
   direct decoding during pending verification and an out-of-range acceptance
   length. There were no power or PCI events. `force-accept/result.json` has SHA-256
   `610973b311ddf7f812b4ae4d97ac4ada9e376530c6792451ef10419e64c4d501` and
   `force-accept/rows.jsonl` has SHA-256
   `1b3876c50755b7923e17374d55f45b2a6a3dda358e2bf1c0548a9489273f4284`.

   A matched short Phlegm HTTP check on the corrected bundle preserved all
   streamed token bytes, text, finish reasons and usage in both workloads.
   Repetition measured 2.67 tok/s off and 1.94 with DSpark, accepting 21 of 29
   proposals. Code continuation measured 3.12 and 3.78 tok/s, accepting all 27
   proposals. The comparison ran off first during the BF16 download, so these
   rates are screening observations. DSpark remains opt-in.
   `app-screen.stdout.log` has SHA-256
   `ceb44caf40cb5c9210d1c3c98d220fdf399b1332702b5c41fd0621637d8dd02c`.

   The follow-up `phlegm serve` comparison used the current AVX2
   drafter and timed both orders on the aligned bundle. All four
   arms matched streamed token bytes, text, finish reasons and usage.
   Windows recorded zero power or PCI events. At 1,320 prompt tokens,
   repetition ran at 2.52 and 1.97 tok/s with speculation off, and
   1.65 and 1.72 tok/s with DSpark. Both DSpark trials accepted 20
   of 29 proposals. The 24.5% control drift limits the speed estimate.
   The 83-token code prompt also produced identical output, with all
   27 proposals accepted. The paired rates were 3.35 to 3.50 tok/s
   and 3.33 to 3.04 tok/s with 0.68% control drift. Acceptance alone
   did not predict throughput. Keep DSpark opt-in on the current
   target. `phlegm-verify/dspark-reversed/result.json` has SHA-256
   `703b9de175840d796c97b82a99c2357268d4980ba71ecba6d508ed5a0a03bb28`.

   Captures in `.plans/inkling2/f32-phlegm/priority-device-20261123/features/` have SHA-256
   `0dbc5b96d32c15492b22e1feda2e64b2d9e45fe21c53bc756caced027894c76e` for
   `dspark-rollback.stdout.log` and `3b107278e10292d2993cf5fe763e2137538800ab64cb5b4805040886052e86e3`
   for `dspark-rollback.windows-events.json`.

   The target scores each verified row with a separate 518,565,888-byte Q8 head weight
   stream. The existing two-row Q8 design compiled for Inkling's 2,304 hidden dimensions,
   200,064 padded head rows and eight cores. Each core has 3,136 B of program text,
   a 4,096 B stack reservation and placed buffers ending at 56,896 B. There is one
   activation fill, eight weight fills and eight output drains. The unmodified row-major
   transfer and an experimental contiguous band-major transfer both built successfully
   using the installed Windows toolchain. Inside the sandbox, both candidates and the
   known one-row control failed at the compiler's `insts.elf` edge.

   The operator fixture pins the same real weight buffer and two saved BF16 inputs to
   both arms. An offline check recovered the archived one-row logits in both layouts
   and rejected a modified score. Both layouts then matched all 400,128 logits bit
   for bit on the board, with no power or PCI context events. The band-major
   stream reduced paired scoring from 21.12 and 21.19 ms to 10.32 and 10.46 ms.
   The row-major stream reduced it from 20.81 and 20.89 ms to 10.52 and 10.63 ms.
   The faster band-major variant uses contiguous output DMA. The optional v11
   `--head-pair` bundle reuses the same resident Q8 weights across one- and two-row
   streams. It uses the singleton stream for an odd tail and ordinary decode.
   The engine, recipe and exporter are wired together. Phlegm verification and
   end-to-end speculative timing are separate gates.
   The original two-row stream SHA-256 is
   `741eb609c41e0024ed4a12da83dd4dec1b24f425025c9345f2d7e7af3f8c4f3e`.
   The operator setup is `.plans/inkling2/target-head-pair/preflight/input.json`
   (SHA-256 `1a1657e2c89e0b55637da6333dbe8891ad073a5cafed53a685b6fbaf1de5940b`).
   The clean board receipts under `target-head-pair/device-67536/` are
   `row/result.json` SHA-256 `f0e81a99d74731be4d9aa775f9246f71db6b6b15f9bf8c8264c5f358b3d80751`
   and `band/result.json` SHA-256 `53f30f29bd7658cd7b4b98afa1a033fe5d6a7ce15e499bf16ca97932114e4170`.

   The paired-head Phlegm bridge reproduced all 20 before and after logit captures
   bit for bit, including one-, two-, three- and eight-row verification, a three-row
   commit and the following decode, at prefixes 5 and 1,023. Windows recorded zero
   power and PCI events. In the control-first screening run, median eight-row
   verification took 1.947 versus 1.774 s at prefix 5 and 2.091 versus 1.695 s
   at prefix 1,023.
   The C++ engine targets compiled and the offline import check matched 55 unchanged
   files and 41 recorded changes. The first Windows build could lower the source and
   stopped when it couldn't find `aiebu-asm` and `xclbinutil` on `PATH`. Adding the
   existing XRT tools built the canonical image. Its eight worker ELFs, hardware
   initialization and both instruction streams matched the board-tested image byte
   for byte. The v11 exporter accepted the build with `--package-only --head-pair
   --wide-old-attention`. The existing XCLBIN verifier found only 78 build-stamp bytes
   different. The remaining 53 files, including the manifest, were identical.
   `target-head-pair/phlegm-pair-171744/result.json` has SHA-256
   `dcfa1b061c9eb55916bc34bfb41603ada406e06a9ccca5d8b6323b1d95c3ff51`.
   `target-head-pair/canonical-rebuild.json` has SHA-256
   `9cfdb8bc5283feb8885d9fb91b4eb3e66ce2c0f4b9dfa3ff6b9497ea0d07c7c7` and
   `target-head-pair/export-equivalence.json` has SHA-256
   `c4f909c575c9131cceb2d5102e7769548d1626254842a7c355eeadb417c875f7`.

   Through normal `phlegm serve` on a matching development CLI, the four-run
   bracket preserved streamed token bytes, text, usage and finish reason in both
   workloads. The 83-token code case went from 2.86 and 3.25 tokens/s with DSpark
   off to 4.02 and 3.97 tokens/s with it on, accepting 27 of 27 proposals in both
   orders. The 1,320-token repetition case went from 1.87 and 2.49 tokens/s to
   2.05 and 2.18, accepting 20 of 30 in both orders. Control drift was 12.9% for
   code and 28.5% for repetition. One PCI context event occurred 11 seconds into
   the first server arm. No arm recorded a power event. Keep DSpark opt-in and
   treat these throughput figures as screening evidence.
   `target-head-pair/serve-175888/result.json` has SHA-256
   `1715c9a07994d712545cee750ab0543644bd3a2c0f958e4348dab207214af7ea`.

   | Capture in `.plans/inkling2/q8-eight-rows/` | SHA-256 |
   | --- | --- |
   | `paired.log` | `c6fbc6dd5f4e19fcbe49554a4f1b628f27588a84e975d1007370b1d47f8a494a` |
   | `clang-parity.log` | `777b4bf99c805a88a3375a191068aaad05b78a6ddb82a168d616bfddc622e114` |
   | `oracle.log` | `5f1084728ed2a94ecc1db6cb804814dd224d6e941c86abbc8fe408fd25e24d2b` |
   | `target-verify.log` | `3b592c9e5f6a950f11d27ba0002f6bff64b937d592cc7a568172bdf4ab915fdb` |
   | `phlegm-bridge/phlegm-reference-bench.exe` | `caab77d880c8761bb814f7fc9c008c9c28bf55808820eca8b7e0f3c812077765` |

   The CPU drafter now uses vector arithmetic for the context attention dot product and value
   accumulation, preserving the previous sum order. It uses hardware half conversion when available
   and applies the final logit cap in the existing parallel Q8_0 head loop. The head scale conversion
   matched the software converter for all 63,488 finite binary16 values. All 14,404,176 scales in
   the measured head were finite.

   Six rotated trials at each draft width and prefix ran on the Ryzen AI 9 HX PRO 370 while the
   131K Phlegm request was active. These are stage and full-draft medians for seven proposals from
   the CPU replay. The longer prefixes repeat saved short-prefix target features and exercise both
   sliding and full-attention draft layers.

   | Prefix | CPU threads | Attention before / after | Full draft before / after |
   | ---: | ---: | ---: | ---: |
   | 1,025 | 4 | 17.17 / 9.04 ms | 199.95 / 194.28 ms |
   | 1,025 | 6 | 15.73 / 8.95 ms | 162.99 / 153.77 ms |
   | 4,097 | 4 | 33.29 / 17.25 ms | 216.57 / 197.92 ms |
   | 4,097 | 6 | 29.83 / 16.26 ms | 179.93 / 159.78 ms |

   Separate MSVC AVX2 builds of the before and after sources matched all 77 proposal rows and
   their full logits byte for byte at widths 1, 3 and 7, from 5 through 4,097 tokens. Proposals with
   and without logit capture also matched. MSVC AVX2 and portable builds matched all 21 IDs in the
   independent Unsloth oracle and passed its exact sliding-window commit test. An initial Clang
   AVX2/FMA build kept the IDs but changed the logits because it contracted the scalar expressions.
   The final build retains those expressions with FMA enabled and matches all 77 rows byte for byte.
   Phlegm decode throughput on the updated build remains open.

   | Report in `.plans/inkling2/dspark-head-profile/` | SHA-256 |
   | --- | --- |
   | `summary-attention-4threads.json` | `f34a08519f85254758c657839f68cad4149302a780b545767db06e34549cd238` |
   | `summary-attention-6threads.json` | `e4d64a6d62f44c120b9865cdd63a2d269283262b22b3a68c7ec0689d8290ebc7` |
   | `canonical-verification.json` | `8a22e7c881690b6ae886cf08aadb5be2723c965863729791b99cde902425e99a` |
   | `clang-verification.json` | `a57ab6b691ca900648a25cac28783bf133e5e51b011a8f5384622d14069470ba` |

   Reset now drops draft K/V allocations above one sliding window and keeps smaller warm capacity.
   An extracted native cache replay filled all three draft visits at 1,048,576 positions. The
   old reset retained 4.008 GiB of capacity. The new reset retained 8 MiB across the sliding caches,
   and a subsequent short reset kept its bounded warm allocation. Reusing the real Q8_0 drafter
   after reset at positions 8 and 1,025 returned the same seven proposals and 1,400,406 logits
   byte for byte in both cases. The portable build also passed the 21-proposal Unsloth oracle and
   the sliding-window commit test. The rebuilt traced Phlegm bridge passed handshake and provenance.
   Both remaining 131K fixtures and the full 1M fixture passed preparation with that bridge.

   | Reset replay under `.plans/inkling2/kv-state-boundary/` | SHA-256 |
   | --- | --- |
   | `dspark-state-1m.log` | `4278c1f89ae48c93057274489f0e7dd577357703d299133c434facdc4b0359f3` |
   | `dspark-reset.log` | `0ed52d49f84b8681976ca3c78b87d285b95012c68fc75263f3cc4f7e694f2459` |
   | `dspark-oracle.log` | `69b3de43608572fcda5fd722a8534c1114a8597b4a00eaee1b15aa2ef8e13cc1` |
   | `bridge-preflight.log` | `2288c1cb9291ce234ddc43fb0ecb89dffab1b6b64285262b845712d2f4d6d69e` |

10. **Vision on the NPU.** Weights from the mmproj GGUF; the patch embed (temporal patch 2 for a still image: confirm
    how the processor fills the second frame); the 32x32 position grid resampled to each size; bidirectional attention
    on the online-attention path (whisper_fa fails at real magnitudes, per the Gemma 4 plan); the 2x2 merge and the
    projector. Gates: tower output against transformers at each ladder size (corr > 0.9999); image QA through
    `phlegm serve`; the cost of each size.
    The pinned llama.cpp reference pads RGB to multiples of 40 and defaults to 4,096 merged image tokens. A
    4,096-pixel square needs 10,609 merged tokens and 42,437 tower rows including CLS after padding. Use
    `--image-max-tokens 10609` for that shape. The reference graph uses bilinear learned positions plus 2D RoPE
    in the tower, then removes CLS before the 2x2 merge and exact GELU projector.

    A CPU oracle at `.plans/inkling2/vision-oracle/` exercises the pinned llama.cpp `mtmd` API using the BF16
    mmproj and an RGB gradient and checker pattern. It confirmed the padded token count at every ladder size
    and produced finite F32 projection captures through 4,096 px. Encode time below uses six CPU threads and
    excludes model load and preprocessing. The 4,096-pixel run overlapped the 32K Phlegm recall. Native NPU
    comparison across the ladder remains open.

    | Square side (px) | Projected rows | Reference encode (s) | SHA-256 of `embeddings-<side>.f32` |
    | ---: | ---: | ---: | --- |
    | 40 | 1 | 0.037 | `056c958138529102e285cf5e428a1275c96ee4559efbfb1041f3d7ec23e3a2e6` |
    | 224 | 36 | 0.327 | `590af7c2a88cd7bd80be648b781a028247d3022a3807c11bcb88de53d54c256b` |
    | 448 | 144 | 1.228 | `80e87d31280a51821bc068bdf45a2f86c9cc6cf46f7c0852f5fba572c2820f77` |
    | 896 | 529 | 5.006 | `c0151828120a6612ec095769525f7768a19bb604875eb0dd19515049018878f9` |
    | 1792 | 2,025 | 29.722 | `68fee24541ee59b7b18c1f6bbaeef21fb0d305e8c2a39e166e8d1bf513a8fdc7` |
    | 4096 | 10,609 | 529.358 | `744637f8b0715aace09ef4e076fbaa449327bfe6fe27a934e243420e12c7d4e2` |

    The native `Inkling2Vision` host route reads BF16 mmproj GGUF, pads RGB without rescaling, and uses
    the GGUF's folded still-image patch projection. It follows the pinned GGML operands with FP16 in the
    patch convolution and vision attention, BF16 inputs to BF16 projections, and FP32 residuals. The
    initial unrounded route scored 0.9998504 at 1,792 px. Accounting for the reference's rounding raised
    that to 0.9999157, with 0.0130 relative L2. This encode ran in 141.59 s with six host threads while
    other local work was active. The native output capture `native-ggml-1792.f32` has SHA-256
    `6f0fa875c5cd0e760ba85c09943bdbcc1f28cbd7e96b703549d7c9c1975e6295`.

    | Input dimensions | Native host correlation with pinned `mtmd` |
    | --- | ---: |
    | 40 × 40 | 0.99999341 |
    | 53 × 82 | 0.99996062 |
    | 224 × 224 | 0.99998368 |
    | 448 × 448 | 0.99997118 |
    | 896 × 896 | 0.99994264 |
    | 1792 × 1792 | 0.99991566 |

    The text engine accepts projected image rows as sequence positions. In the 53 × 82 mixed-input
    check, both engines received the same six projected rows between the chat template's text segments
    and committed 28 positions. The five leading IDs agreed in order at prefill and the next fixed
    decode token. KL(llama.cpp ‖ native) was 6.02e-12 and 4.25e-9 nats, respectively. Empty, partial,
    nonfinite and over-capacity embedding blocks left the native sequence position unchanged. The
    native decode capture at `.plans/inkling2/vision-oracle/mixed-final-53x82/native-decode.f32` has
    SHA-256 `8104946207a211c79f312aa906e8c076d1b8fe2cad3ba5e9929d48544ea25f79`.

    `attn_fa` builds in bidirectional mode with a last-valid-key field. Split probabilities and split Q
    were enabled for the two builds. Each core fit 11,984 B of program memory and 63,584 B of placed L1
    including the 8,192 B stack. Both builds dispatched on the Strix NPU, including a full-length
    42,437-key stream with 59 masked padding rows. The long case repeated captured inputs and checked
    eight queries. Nonzero values in padding made a failure to mask visible in the comparison.

    | NPU probe | Driver dispatch (ms) | Relative L2 against FP64 on packed inputs |
    | --- | ---: | ---: |
    | 224 px capture, first attention layer | 4.436 | 7.46e-6 |
    | 224 px capture, last attention layer | 4.351 | 2.40e-5 |
    | 42,437 attention keys, 42,496-row stream | 89.680 | 4.99e-5 |
    | 224 px capture, fused BF16 QKV matrix | 5.619 | 3.68e-7 |
    | 224 px capture, BF16 projector first matrix | 9.174 | 6.82e-7 |

    Probe captures and hashes are in `.plans/inkling2/vision-attention/*/report.json` and
    `.plans/inkling2/vision-projection/*/report.json`.

    The version-9 writer adds one shared BF16 GEMM context with six projection streams and one shared
    bidirectional attention context with nine row tiers from 256 to 42,496. It compares the compiled
    per-core binaries and initialization data before combining streams in each context. The version-8
    text artifacts carry over byte-for-byte. The native tower uses resident packed weights, one K/V upload
    per layer, and fused QKV projections. Patch convolution, affine norms, GELU, position preparation,
    bias and residual additions, and spatial merging remain host stages. The full NPU tower compiled.
    Device validation of the assembled route and per-stage timing remain open. Local manifest
    `.plans/inkling2/vision-v9-bundle/manifest.json` has SHA-256
    `5e9ca83d1d73580325434a498ce2e39d35e8c861c2fb8a2e4124e1e04936d178`.

    The Phlegm bridge advertises image input for a version-9 bundle with `mmproj.gguf` beside
    `model.gguf`. HTTP chat accepts ordered text and inline PNG `image_url` parts. The app expands the
    template marker to the exact number of projected rows, sends RGB via binary IPC, and commits image
    rows in the same prefill block sizes as text. Image requests clear prompt reuse because the token
    cache does not identify image bytes. The rebuilt HTTP probe used the real GGUF tokenizer and an
    instrumented test backend. It verified a two-image prompt expanded to 27 positions, preserved the
    two RGB values and their ordering, and returned HTTP 200 in JSON and SSE and HTTP 400 for a 39-pixel
    image. The first hardware app launch failed at tokenizer discovery because the model directory contained
    both `model.gguf` and `mmproj.gguf`. Phlegm now excludes the canonical projector sidecar from tokenizer
    discovery and still rejects multiple candidate text GGUFs. All six GGUF tokenizer unit tests passed.
    The hardware app validation restarted with a matching bridge and CLI. The sidecar patch in
    `.plans/inkling2/phlegm-vision/tokenizer-sidecar.patch` has SHA-256
    `6d0fc055f55b9d6f7ddff685fa13fcb55de9940afe3b884578afcfa6f13b2d2c`.
    JPEG and remote URLs are outside the current app route.

    The v9 NPU route through `phlegm serve` returned `red` over JSON and `Blue` over SSE, both with
    stop finishes and 67 prompt tokens. Their end-to-end times were 41.08 s and 32.44 s. The invalid
    PNG request returned HTTP 400, and a subsequent text-only request returned `Hello`. The replies
    are in `.plans/inkling2/phlegm-vision/`. SHA-256 for `red.json` is
    `d64c434c38c610cac8b84ec5c434e6239f09442dfa0331f8ef11b5745227dd26` and for `blue.sse` is
    `010d182197f3cba85933991e01014685093275623fed6c97a887b1cb992c6769`.

    Assembled v9 NPU vision parity against the pinned CPU reference on the same fixtures:

    | Image size | Projected rows | Correlation | Relative L2 | Wall time (s) |
    | --- | ---: | ---: | ---: | ---: |
    | 40×40 | 1 | 0.99998470 | 0.00554 | 1.98 |
    | 53×82 | 6 | 0.99997557 | 0.00699 | 0.81 |
    | 224×224 | 36 | 0.99997176 | 0.00752 | 0.88 |
    | 448×448 | 144 | 0.99995332 | 0.00967 | 1.33 |
    | 896×896 | 529 | 0.99989315 | 0.01462 | 3.38 |
    | 1,792×1,792 | 2,025 | 0.99984126 | 0.01782 | 16.94 |
    | 4,096×4,096 | 10,609 | 0.99982452 | 0.01873 | 253.94 |

    The 896, 1,792 and 4,096 px cases missed the required correlation above 0.9999. At 896 px, CPU attention
    with NPU projections raised correlation to 0.99994135 (relative L2 0.01083, 7.53 s). CPU projections with
    NPU attention reached 0.99989453 (relative L2 0.01453, 6.98 s). The paired stage substitutions identify
    NPU attention as the higher-priority source of drift on this fixture. The source within attention and the
    effect across larger fixtures remain open. Local hashes in `.plans/inkling2/phlegm-vision/`:

    | Artifact | SHA-256 |
    | --- | --- |
    | `vision-parity.json` | `b7c29096a550ff8fc1049f844b11496e6a8b344666d65ecb0e96357d386305dd` |
    | `ablation.json` | `68e619dc0557f9f90d7756a5b17bcc15d1362386ebefe827d40a995d590b10c6` |

    A separate FP64 softmax replay on real 224 px Q/K/V captures compared FP16 Q/K/V with the device format
    (split Q, BF16 K and V). Relative L2 from the input rounding alone was 0.00159 at layer 0 and 0.00357 at
    layer 22. At layer 22, replacing only the keys with BF16 gave 0.00335 and replacing only the values gave
    0.00130.

    Version 10 adds a BF16 residual for keys before the attention mask and softmax. Version 11 also adds a
    BF16 value residual into the numerator before normalization. Both modes keep the existing probability
    split. Each 64-key chunk groups the high and low parts into planes so each shim can read its planes in
    the same descriptor, and each core consumes one plane at a time. The 4,096-row slices retain the
    existing bounded descriptor schedule. The 1,024- and 42,496-row builds used at most 12,896 B of
    program text per core for v10 and 13,648 B for v11. The placed L1 buffers and reserved 8,192 B stack
    end at 63,584 B on both. A v9 control rebuilt after both changes matched all 32 earlier core ELFs
    and the instruction stream byte for byte. A causal split-probability build with 32 query heads and
    eight KV heads also matched the 32 core ELFs and the instruction stream built from repository HEAD.

    On captured 224 px Q/K/V, the v10 attention result at layer 22 reduced relative L2 against the FP64
    calculation on the original inputs from 0.00339 for a single BF16 key to 0.00118 with split keys.
    The synthetic 42,437-valid-key test reached 0.00112, including the masked tail across a descriptor
    boundary. The full v10 tower raised the 896 px result to 0.99991826. It improved the 1,792 and
    4,096 px correlations to 0.99986224 and 0.99986470 respectively. Both missed the gate. The largest
    v10 case took 324.13 s end to end, compared with the earlier v9 run at 253.94 s. These were
    individual runs with different concurrent build load.

    With v11, the layer-22 capture reached relative L2 `2.49e-5` against the original input values.
    The 42,437-valid-key probe reached `5.62e-5`, preserving the masked tail and the split-probability
    result. The assembled v11 tower scored 0.99992920 at 896 px and 0.99989844 at 1,792 px. The latter
    remains below the 0.9999 target. The v11 4,096 px result was 0.999900019 (344.86 s, 10,609 rows),
    just above the gate. A separate host-pack comparison rounded Q, K and V to FP16 before the split,
    following the native CPU attention's operand path. It passed all six smaller fixtures, with
    0.99993541 at 896 px and 0.99991370 at 1,792 px. The reader now uses that rounding for v11.
    The assembled runtime reproduced all six smaller embedding captures byte for byte. At 4,096 px it
    produced 10,609 rows in 345.81 s with correlation 0.99990436 and relative L2 0.01383. All seven
    runtime fixtures cleared the correlation gate.
    All nine attention tiers were rebuilt for each variant. Each bundle retained the 31 v9 text
    artifacts byte for byte. Both Phlegm bridge variants compiled. The unrounded v11 image smoke
    passed JSON, SSE, invalid PNG rejection and the text follow-up. The first FP16-rounded Phlegm request
    timed out. A retry with the same binaries and bundle passed all four cases. The JSON red response
    took 61.28 s, and the streamed blue response took 45.13 s. These are complete request times.

    | Artifact under `.plans/inkling2/` | SHA-256 |
    | --- | --- |
    | `vision-v10-bundle/manifest.json` | `0d87e59a790ca4bffa214b0062c24afacd2f7fee455d4618a7c8f26d9a5397c8` |
    | `vision-v11-bundle/manifest.json` | `d58ffa427f95a5dddbb936ac4b939bf42d87906b1d296061f6734266b3bc47ac` |
    | `vision-split-keys/tower/vision-parity.json` | `ab67fe46ce532b0d8eee67b51a353bf2db2263b63291ca26e1e1e4410cf84174` |
    | `vision-split-kv/tower/vision-parity.json` | `ffffa454e2d043c058abdfb645a2dcdc402909ace8a4d4a562cfe537f969bec3` |
    | `vision-split-kv/half-tower/vision-parity.json` | `f58df8c5a978b72011ce4394278cf2955c86bebfd17fbca0a08c28ad97fb3a00` |
    | `vision-split-kv/fp16-runtime-tower/vision-parity.json` | `6f95c8938dd1e6d7c8268715b9b904ab4313ec1135a5250667c25e8e724fa4f9` |
    | `vision-split-kv/http-fp16-retry.log` | `aebd7144efef0064fd40a78a0634b8eb4a91cf32fccbd9d2c7936dcfc4757d43` |
    | `vision-split-kv/layer-22/result.json` | `2d5e39ffbf29e72d506d06f378a64bb47647c0fcef6db6c301c993c445f6e263` |
    | `vision-split-kv/masked-long/result.json` | `370cd1c7cbd17bb5480b982775dadd308b567157e4e9b3ef13b00f8004c15c77` |

    The focused Phlegm server suite reported 42 passes and one ignored dependency test. Its idle-unload
    assertion failed in both this build and the pre-vision build. The native vision/GGUF smoke suite
    passed three tests, and fused host QKV produced byte-identical 224-pixel embeddings to the previous host path.

    The expanded Phlegm image suite adds counting, code recognition, two-image ordering, arrow direction
    and a 4,096-pixel OCR request. Its seven synthetic PNG inputs are validated and enumerated in
    `.plans/inkling2/phlegm-vision/qa-inputs/qa-fixtures.json` SHA-256
    `e5a3ed5e539340481e3d8607a2341771b752ff66c83fa78fe3c5efdce46b0fa4`.
    All eight requests passed through `phlegm serve` on the v11 NPU bundle using the development
    query-cache bridge with speculation off. Every reply stopped normally and reported zero cached
    prompt tokens. The server also rejected the invalid PNG and answered the text follow-up. The NPU
    image tests with DSpark remain open.

    | Request | Expected answer | Image rows | Prompt tokens | Wall time |
    | --- | --- | ---: | ---: | ---: |
    | Red color | Red | 36 | 67 | 109.28 s |
    | Blue color | Blue | 36 | 67 | 57.44 s |
    | Padded 53×82 image | Green | 6 | 35 | 35.44 s |
    | 448-pixel triangle count | 3 | 144 | 173 | 64.30 s |
    | 896-pixel OCR | LARK-572 | 529 | 559 | 114.86 s |
    | Two 224-pixel images in order | 2 | 72 | 107 | 57.53 s |
    | 1,792-pixel arrow | Left | 2,025 | 2,055 | 346.59 s |
    | Full 4,096-pixel OCR | TERN-938 | 10,609 | 10,640 | 1,787.77 s |

    `.plans/inkling2/query-pack-reuse/phlegm-image-qa-complete.json` SHA-256
    `ff78d5ea9cbc03fcb2fe6f20f4074f5e76bc1df0f4754d2a649b8891b40d2655`.
    The full-resolution streamed reply `.plans/inkling2/phlegm-vision/qa-v11-device/printed-key-4096.sse`
    has SHA-256 `52eb473bbe01ce365fb6381834103327ba91ef4c6bec2c6f8b1048710735ef5b`.

    A CPU-only Phlegm test ran the real BF16 vision tower, IQ4_NL text target and Q8_0 DSpark draft
    through `phlegm serve`. The test CLI used 40 byte-identical Phlegm application source files and
    embedded the CPU adapter source for its bridge provenance check. Both requests used a padded
    53×82 image with six embedding rows. Plain decode and DSpark produced identical answer text,
    token usage, finish reasons and committed token IDs. The trace showed equal target and draft
    positions after image prefill and after every commit. Phlegm's own proposal counters matched
    the bridge trace.

    | CPU request | Prompt tokens | Committed decode tokens | Accepted / proposed | Off / auto wall time |
    | --- | ---: | ---: | ---: | ---: |
    | Color name, JSON | 35 | 33 | 14 / 27 | 81.40 s / 87.84 s |
    | Color repeated six times, SSE | 51 | 76 | 39 / 64 | 157.24 s / 163.48 s |

    These are single observations in off-then-auto order. The native `smoke_vision.py` suite now
    reads Phlegm's proposal and acceptance counters per image request. Its `--require-dspark`
    preflight passed for the seven expanded PNG fixtures through 4,096 px. Measurements with
    DSpark on the NPU are pending, including two-image and full-resolution cases.
    The ordered off/auto suite is queued on the same v11 bundle and Phlegm binary for both modes.
    It compares the HTTP answer, reasoning, stop status and usage for all eight requests and
    requires actual DSpark proposals and acceptance. Its JSON/SSE reader reproduced all eight
    saved plain-decode answers. `.plans/inkling2/phlegm-vision/dspark-pair-preflight.log` SHA-256
    `1c45004f38d7548d8b1390c7af3db34c2b3f7ae23859fbe568c344cfb72cf1ce`.

    | Artifact under `.plans/inkling2/vision-dspark-cpu/` | SHA-256 |
    | --- | --- |
    | `test-cli-sources.json` | `7788e9310ca8c940ad6f6befb4f7e80e2118565e8d703d28802aecaf42fcc0c1` |
    | `phlegm-reference-bench.exe` | `18fc53e3de52b760b2246bb0719e6c7f9f2b9a78c9b395205f4f977aee0f1d51` |
    | `run-20261123-075027/summary.json` | `c5b76df400f1042a6f302f69c2a010b95041121d107c29425434758eb705870c` |
    | `run-20261123-075027/auto/bridge.jsonl` | `4941a95941a3468d109060f7a3bc92c48b6072484148b8559d952f3be1252897` |
    | `run-20261123-075027/auto/serve.log` | `bcbffc7e11aa6f6bb8f3ad17b18b67795d160d81b967df4040b62df68f1d144f` |
    | `run-20261123-075027/off/bridge.jsonl` | `7599d3261d3730dbc4019fc3806c8da9790bd2076252df6715e31057cde372db` |

    The pinned llama.cpp multimodal API read `LARK-572` exactly as the final answer to the 896-pixel
    OCR fixture. The CPU projector encoded 529 image rows and the Vulkan target processed 559 total
    prompt positions. The stock `llama-mtmd-cli` exited with Windows status `0xC0000409` on both
    attempts. The verbose retry reached Inkling's specialized template during startup. The API
    probe rendered the effort-zero prompt and image block directly and completed. This establishes
    the reference result for the same fixture later passed by Phlegm on the NPU.

    | Capture in `.plans/inkling2/verify-matrix/` | SHA-256 |
    | --- | --- |
    | `vulkan-vision-api.stdout` | `fc21028a4a0b8979c787e17efad4e126c86ee51010abb813ffe380d58cefce18` |
    | `vulkan-vision-api.stderr` | `3dbbe27de8350ba3783aa2538b2fc96f068ff809d4e3b00b3bb6c111c1f5db4e` |
    | `vulkan-vision-ocr-retry.stderr` | `c1365ca79e2c1a467ea1cb5ffeb701bd53a8f3cc40985de2b8210f028e691246` |

    With the question and image blocks in the same order as the Phlegm requests, the reference
    passed all six fixtures. It returned `green` for the padded 53×82 image, `3` for the 448-pixel
    triangle count, `LARK-572` for 896-pixel OCR, `2` for the ordered pair of 224-pixel images,
    `left` for the 1,792-pixel arrow, and `TERN-938` for the 4,096-pixel OCR fixture. All emitted
    the sampling stop marker and used exactly 6, 144, 529, 72, 2,025 and 10,609 image rows
    respectively. The final case consumed 10,640 total prompt positions. Its CPU encode took
    815.854 s during other workloads. Summary
    `.plans/inkling2/verify-matrix/vision-ladder/results.jsonl` SHA-256
    `a034b9619af48a5b924067b02d76fbdf07e631f34c209436af796d7f0030e426`.

    A paired Phlegm NPU run used the same v11 bundle, bridge and eight requests with speculation
    off and auto. The off arm passed all eight, including `TERN-938` in 1,346.813 s. The first six
    auto cases matched answer, reasoning, finish reason, usage and image rows. DSpark accepted 138
    of 227 proposals across those six. On the 1,792-pixel arrow, auto returned HTTP 500 with an XRT
    dispatch timeout in `blk.16.ffn_down_exps.weight` (ERT state 8). The auto arm stopped before
    4,096-pixel OCR. The capture identifies the timed-out dispatch. Its prefill or verify phase and
    root cause are unresolved. The focused fixture can replay the arrow at the original 16,384-token
    configured capacity with a fresh server to test whether prior requests are required.

    | Capture in `.plans/inkling2/phlegm-vision/qa-v11-dspark-pair/` | SHA-256 |
    | --- | --- |
    | `off.log` | `d2bb4d58028c3566cd3a13af21ce89361da08b84be89aa2af5a9ac31b18bd7b6` |
    | `auto.log` | `38520412b79fc5dc8862a716ec815de9e10dc3efcff6767d5b0bad9e5cd6d6bc` |
    | `auto/arrow-direction.json` | `a2f13540d4523a6c89261fea1d573a9f55ce9404815ba16b94d306d5be54c51f` |

    The fresh-server NPU retry at 1,792 px returned `left` with DSpark enabled. It
    matched the prior off arm's answer, reasoning, stop status and usage exactly.
    It used 2,025 image rows, 2,055 prompt tokens and 50 completion tokens. DSpark
    accepted 28 of 41 proposals over 22 cycles. The request took 244.50 s and the
    test including server startup and follow-up checks took 336.39 s. The malformed
    image was rejected and the following text request returned `Hello`. Windows
    logged zero power events and zero PCI context errors. This is a single fresh
    DSpark request, so its elapsed time is an observation.

    The 4,096-pixel NPU retry returned `TERN-938` with DSpark enabled. All 47 SSE
    frames matched the prior off arm after comparing the choice and usage payloads.
    It used 10,609 image rows, 10,640 prompt tokens, and 50 completion tokens.
    DSpark accepted 35 of 46 proposals over 15 cycles. The request took 1,265.94 s
    and the full test took 1,607.30 s. The malformed PNG was rejected and the
    following text request returned `Hello`. Windows logged Modern Standby entry
    (Kernel-Power event 506, record 188653, idle timeout) and zero PCI context errors.
    Exclude this window from performance claims. The result covers functionality
    and response parity for this full-resolution fixture.

    | Retry evidence under `.plans/inkling2/f32-phlegm/priority-device-20261123/features/` | SHA-256 |
    | --- | --- |
    | `vision-1792/arrow-direction.json` | `2446e5f929c908dd95d71d8c4a53c9a0805f41d061eb8434803613735b3eb8d2` |
    | `vision-1792-dspark.stdout.log` | `7547ea327f46ee3ebebd8118dbbb355653a2630ca4fd4baf3c627a3971d11a41` |
    | `vision-1792-dspark.windows-events.json` | `79c746b4edcfec2ad7df4fd9a5ba266b79569217bb18eba51c084188c7505b05` |
    | `vision-4096/printed-key-4096.sse` | `1c1506adaa87178c52a8dbb81d77f94940cf08ca858fed3d05d1f16d2d1abc03` |
    | `vision-4096-dspark.stdout.log` | `4dddf358218fdfad64711564ff40feb054fb93b81f4d4e1b1b98eea14b7a2c92` |
    | `vision-4096-dspark.windows-events.json` | `d4466726286ba33177c2093c88f6610db868484aa92b4104fd82be4df410f875` |

11. **End to end.** `phlegm run` / `serve`: reasoning effort at none, low, high and max; a tool call and its result;
    an image request. `phlegm bench` prefill and decode against context length, machine and stack named.
12. **Record.** The audit table into OPEN-FAMILY-INKLING2; the new requirements; `RECOMMENDED_MODELS.md`;
    `benchmarks/reference/engine-changes.json`; archive this plan.

Order: 0-2 first; 3 and 4 can run in parallel; 9 and 10 can start after 5.

## Default-route speed work (2026-11-24)

Evidence lives under `.plans/inkling2/near-attention/`. The machine was shared with another NPU session, so
timings are compared only within one sitting and in both orders.

**Near attention.** `designs/inkling2/near_attention.py` processes one 256-row block, all 24 heads, and 1,280 keys
(the relative range plus the block) in one dispatch. Each eight-core group runs one head per pass, and core `w` owns
rows `32w..32w+31`. Each memtile joins a 16-key K (or V) chunk with four cores' relative strips, so the host packs K/V
once per visit, not once per 16-row tile. The relative table is BF16-exact in the pinned GGUF (all 16,384 values),
so each strip is one BF16 plane. Strip `(chunk c, core i)` covers distances `1024 + 32i - 16c - 15 + x`, which is fixed
per physical layer, so the host builds the strip table once per layer. Older full-attention keys run through the same
kernel in 1,280-key segments, or through the wide older-key route when that is enabled. The engine selects the route
for prefill blocks of 17–256 rows whose position is a multiple of 16; decode, verify and unaligned resumes keep their
existing path. The manifest field is `near_attention: {format: inkling2-near-v1, chunks: 80}`.

- Operator accuracy against FP64 on random fixtures (sliding, full, early block with negative keys, 100-row tail):
  worst-row relative L2 `4.7e-5`, median `1.0e-5`. The quadratic term of the exponential uses two BF16 limbs. With one
  limb the worst row was `1.9e-4`.
- On real model data at 512 tokens, per-visit error against FP64 had median `9e-6` and maximum `2.2e-4` (old
  16-row route: `4e-6` and `1.5e-4`).
- Ablation of the 10.45 ms dispatch: QK 3.6, relative bias 1.85, softmax ~2.5, PV 4.2 and DMA 1.5 ms. Replacing the
  unaligned strip load with aligned loads plus one byte shift and transposing PV, so probabilities are the vector
  operand, left QK 1.6, bias 0.5, softmax 1.6, PV 2.1 and DMA 1.7 ms in a slower sitting. Two keys per QK pass and
  four dims per PV pass were best; four keys spilled.
- Phlegm profiler at 512 tokens, same bridge: attention 2.41/2.23 s (control) against 0.94/0.96 s (near). Prefill
  22.92/19.44 s against 17.56/17.94 s.
- Final-logit KL against the FP32 CPU engine at 512 tokens was 0.0007 (control), 0.0026 (near) and 0.0042 (near,
  precise exponential). Per-visit residuals explain this: both NPU routes already differ from the CPU route by about
  `1.3e-3` at visit 0 and reach a 2–4% median by the late visits (int16 activation quantization in the dense and
  expert kernels). The near route stays within the control's spread at every visit (late-visit median `1.9e-2`
  against `2.1e-2`). Single-position KL is not a usable gate for attention changes on this model.

**Kernel optimization level.** Every Inkling AIE kernel was compiled with `-Os`. The grouped gate/up expert stream
took 1.70 ms. `-O2` gave 1.30 ms, and fully unrolling the row loop gave 1.14 ms, all bit-identical. With arithmetic
removed the stream takes 0.63 ms, so the rest is core compute in the per-32-weight FP32 epilogue that GGUF block
scales require. The expert, dense and shared decode designs now default to `-O2`, with the row unroll in the expert
and dense kernels.

**Decode rows.** For decode and verify (at most eight rows) the shared-image attention workers now take K
transposed per 32-key chunk, with FP32 relative bias and masks in the strip area. Keys are the vector lanes, so no
lane of the 16-row tile is wasted. The query header carries the row count (word 6) and the head within its group
(word 5). Chunk partitions and empty-chunk handling are unchanged, so verify and decode see the same per-row
sequence. The manifest field `attention_decode: keys-as-lanes-v1` is written only when every shared attention build
carries the mode. Over 720 checked decode visits the worst error against FP64 was `3e-5`, and with the vectorized
K transpose `7.1e-5`.

**Measured (2026-11-24).**

| Run | Before | After |
| --- | ---: | ---: |
| Phlegm profiler, 8,225 tokens, both orders | 343.8 / 289.7 s (23.9 / 28.4 tok/s) | 205.4 / 197.7 s (40.1 / 41.6 tok/s) |
| Same, attention / MoE stage | 101.9 / 139.8 s | 20.5 / 90.2 s |
| `inkling2_npu_check`, 4,096-token prefill | 168.9 s | 115.0 s |
| Same, 32 decode steps after 4,096 tokens | 423 ms/token | 271 ms/token (host fixes included) |
| Same, 32 decode steps after 512 tokens | 325 ms/token | 261 ms/token |

Further checks with the same bundle (`.plans/inkling2/near-attention/bundle-o3`, which adds a 36-chunk decode tier):

- `phlegm serve` 32K recall at 90% depth with near attention plus the wide older-key route returned `ZIRCON-1824`
  exactly after 862 s of profiled prefill. The three earlier 32K runs took 49–65 minutes each, though those times
  included the whole request. There were no CPU attention fallbacks and no power or PCI events, and peak bridge
  private memory was 19.6 GB.
- `phlegm serve` 131K recall at 90% depth with the same routes returned `ONYX-3179` exactly. The 131,071-token
  prompt took 4,424 s of profiled prefill (29.6 tok/s) and 4,522 s for the request. The earlier packed-key 131K run
  took 5,367 s of prefill. The 117 decode tokens took 0.82 s each, of which 0.33 s was host K/V packing and 0.22 s
  was full-attention dispatch. There were no CPU attention fallbacks and no power or PCI events. Peak bridge memory
  was 35.04 GiB working set and 22.89 GiB private. Seventeen stalled blocks added 379 s while another session held
  most of the memory. `recall-131k/result.json` has SHA-256
  `82034541ef6a8e24efc71a27f707c3bba984aa9f5e6cae7e235ed2d19f7cef1b`.
- A cost model fitted to that run gives `6.1 s + 21.0 ms x (position / 1K tokens)` for one 256-token block. It
  predicts 864 s at 32K against the measured 862 s, and about 20 hours for a 1M prefill at the steady rate. The
  local plan `.plans/inkling2-long-context-speed.md` has the stage tables and the proposed speed work.
- The DSpark forced acceptance and rollback matrix passed all 18 cases at prefixes 5 and 1,023 with decode-row mode.
  Verify output matched direct decode bit for bit, including the following draft and target logits.
- `phlegm bench --development` (validated, no power or PCI events) measured prefill 49.1 / 38.6 / 39.0 tok/s and
  decode 4.03 / 3.61 / 3.45 tok/s at 512 / 1,024 / 2,048 input tokens. The earlier sweep measured 33.7 / 36.1 /
  29.1 and 2.52 / 2.56 / 2.24. The 4,096 case (28.2 and 1.87 tok/s) had decode steps from 367 to 1,172 ms, against
  208–273 ms in native checks at the same depth, so it is excluded until a quiet rerun.

The 8K final logits kept the top-1 token (KL `1.6e-5` against the old route). At 512 tokens the 33 greedy decode tokens
matched the old route exactly. At 4,096 tokens the last prompt position was a near tie (margin 0.004) and the greedy
paths diverged after it. A 4K decode profile showed MoE wait at 82 ms per token, near the IQ4 weight-DMA floor, and
attention dispatch at about 1 ms per call. Before the host fixes it also showed 24 ms per token in output validation,
42 ms in the controller and 32 ms in K/V packing.

**Host stages.** Threading SiTU and the expert reduction (by column, because one token can occupy two slots of a
group) cut their host time from 3.7 to 1.3 s at 512 tokens, but device wait rose by the same amount. Prefill MoE is
device-bound. Narrow F32 projections (controller down at 64 rows, router at 128) now walk token tiles outside so each
tile stays in cache, with the same dot-product code. The attention gate is threaded.

## Loop controller on the NPU (2026-11-25)

Design and feasibility: `.plans/inkling2/controller-kernel/design.md`. Evidence in `.plans/inkling2/controller-kernel/`.

The host runs four controllers (q, k, v, ffn) for each core-layer visit in loop passes 2 and 3: 24 visits and 96
calls a token. Each call is the rms of x and of the message, a 64 x 4,608 down projection, two 2,304 x 64
projections, tanh and the update.

- Decode stays on the host. The singleton decode image has 1.5 KB of program memory and 3.8 KB of L1 left
  (14,864 of 16,384 B and 61,724 of 65,536 B), and all 32 cores are in use. Controller 3 depends on the attention
  output, so it cannot join the dense input dispatch. Two extra dispatches for each visit (48 a token) at the
  0.08-0.25 ms floor, plus 226 MB of F32 weights a token, cost about as much as the 18 ms host stage.
- Prefill blocks of 64 rows or more use a new image, `controller.xclbin` (`designs/inkling2/loop_controller.py`).
  Each of the 32 cores owns 8 of the 256 token rows end to end, and the weights are broadcast to them. Two
  dispatches for each looped visit: q, k and v together (`controller-qkv`), then ffn (`controller-ffn`).
- All 28.3M controller weights in the GGUF are exact bf16 values stored as F32. The kernel keeps them as one bf16
  plane, which is exact, and splits activations into bf16 high and low halves (two bf16 mmul terms, F32 sums). tanh
  is 2 sigmoid(2y) - 1 with `vecmath.h`; rsqrt uses an integer seed and Newton steps.
- Budgets: program 9,600 B, data top 37,436 B, largest stack frame about 2 KB of 4 KB.

| Check | Result |
| --- | --- |
| Fixture against an F64 reference, real blk.6 weights, 200 and 256 rows | largest row relative L2 error of u 1.6e-5 to 2.3e-5, padded rows exactly 0 |
| Fixture time, 256 rows (not a clean run) | q, k, v 2.85-2.96 ms, null-compute copy 1.94-2.0 ms; ffn 1.0 ms |
| Block-route logits, 512 rows in 64-row passes after 256 tokens | 508 of 512 top tokens equal, mean KL 0.00023, largest 0.024; the 4 flips have margins 0.005-0.045 |
| Real-text NLL of those rows | 1.4703 to 1.4698, top-1 344 and 344 |
| Decode logits after a 512-token prefill, 512 positions | 510 of 512 equal, mean KL 0.00019, largest 0.0062; NLL 1.2920 to 1.2918 |
| Greedy, 512-token prompt, 64 tokens | equal for 9 tokens, then a near tie (0.16 logits); on the current route's sequence 63 of 64 top tokens equal |
| Private memory after the first decode token | 17.98 to 18.12 GiB |

Paired prefill timing, 1,024 tokens in 256-row blocks after a 256-token warm-up, 6 host threads, same bridge.
The current-route arms ran CLEAN. Both controller arms were NOISY (another process held one core for 2-16 s),
so their times are not reliable; that noise only slows them down.

| Arm | Prefill | Steady blocks | Controller a block |
| --- | ---: | ---: | ---: |
| Current route (1) | 28.1 s | 6.97, 7.03 s | 1.08, 1.11 s |
| NPU controller (2, noisy) | 20.2 s | 4.92, 4.82 s | 0.30, 0.30 s |
| NPU controller (3, noisy) | 20.5 s | 5.18, 5.05 s | 0.32, 0.31 s |
| Current route (4) | 23.5 s | 5.80, 5.51 s | 0.62, 0.63 s |

- The host controller moved between 0.62 and 1.11 s a block in two clean arms. The NPU controller takes 0.30 s a
  block in both arms: 6.3 ms a call, of which about 3.9 ms a visit is kernel time. The rest is two context entries a
  visit (the first run after a context load took 3.6 ms longer in the fixture) and the host staging copy, sync,
  finite check and unpack.
- Left: put the controller on the dense or expert image to remove the two context entries a visit (about 0.1 s a
  block, estimated); cut the kernel's 4x weight broadcast and repeated activation read (DMA-bound, null copy 2.0 ms).
  A clean paired timing is still owed.

### SiTU in the singleton expert kernel (2026-11-25)

Evidence lives under `.plans/inkling2/situ-kernel/` (design and budgets in `design.md`). One singleton launch per
decode visit now runs the 8 experts' gate/up, SiTU and down: 48 expert launches per token instead of 96, and no
host SiTU, gate/up readback or down-input pack between them.

- Placement: the gate/up pools use a paired band order, so each core holds three gate tiles and their up tiles
  (`pools.cpp paired_band`). A gate band goes to a core scratch, its up band to the output frame, and the core
  writes three down-input K32 tiles over that frame. The runtime sequence drains those tiles to a DDR region of the
  output buffer, waits, and feeds them back as the down leg's activations. No fifo or channel was added: all
  6 + 6 memtile channels per column were already in use.
- Program memory went from 14,864 to 15,616 B of 16,384. The first build overflowed at 17,072 B. It fit after
  four changes: a bit-pattern reciprocal seed (`aie::inv` on 32 floats expands to 32 scalar ops), one K32 tile per
  SiTU call (a loop spilled every constant across the calls), the 16-lane attention exponential running on
  `vexpN<32>` (lane-identical), and one project kernel for both formats with a runtime flag.
- Two device bugs found on the way: the halves' flag offset leaked into the records-per-frame count (the launch
  completed with B records left over and the next launch timed out), and the last pair waited for the down leg's
  first weight record before draining its own frame (deadlock).
- Fixture (`fused_check.cpp`, `fixture.jsonl`): 192 slot rows over layers 0, 7, 13, 19 (IQ4_NL) and 20, 23 (Q8_0),
  inputs at scale 1 and 4 (gate peaks to 18), against the two-launch route with host SiTU. Mean relative L2
  `2.45e-4`, worst `1.17e-3`.
- Bundle logits at 896 prompt tokens plus 256 decode positions (`compare_logits.py`, same bridge): the new bundle
  with `INKLING2_TEST_SEPARATE_EXPERTS=1` matched the a6b8e3f bundle bit for bit in prefill and decode, which checks
  the paired order, the shared project kernel and the 16-lane exponential. The fused route kept prefill bit-equal and
  matched 251 of 256 decode top-1 tokens. All five flips were near ties (reference margins 0.019 to 0.093). KL mean
  `4.4e-4`, max `4.95e-3`. Real-text NLL 1.2769 against 1.2754, top-1 178 against 180 of 256.
- Five-token prompt with 64 greedy tokens (`inkling2_npu_check`): both bundles matched all 64 native host IDs.
- Timing with the staged profiler bridge, 256 decode tokens after 896, 6 host threads, Arbiter `--measure`, order
  reference, fused, fused, reference (the second fused run was flagged NOISY for a 2 s outside burst): decode median
  179.1 / 180.0 ms per token against 171.4 / 170.8 ms. MoE fell from 68.5 to 57.7 ms per token: host SiTU 3.7 ms
  removed, device wait 57.1 to 52.6 ms, configure 1.3 to 0.7 ms. Remaining MoE host work is about 5 ms (routing
  2.4 ms, one token pack, the 8-row readback, finite check, unpack and weighted sum). Prefill is unchanged (same
  bits; 39.9 to 40.8 s for 896 tokens in all four runs). Each route repeated its decode logits bit for bit.
- Verify rows. With fused decode, partial accept failed 16 of 20 cases: IQ4_NL verify rows used grouped experts with
  host SiTU (Q8_0 rows already verify one by one on the singleton stream). The grouped image now has
  `prefill-{iq4,q8}-moe`: one launch per group of 8 slots x 16 rows with the same core SiTU code (`situ.h`, shared
  with the singleton kernel) and the same down-input pack per row, through the same DDR hand-off. Padded rows need no
  mask: their packed input gives zero gate/up, and the core writes the padding the host would. Manifest key
  `prefill_fused_experts`. Grouped text 11,328 B (IQ4_NL) and 10,560 B (Q8_0) of 16,384; singleton 15,696 B.
- Grouped fixture: 168 of 168 IQ4_NL slot rows (1, 4 and 16 rows per slot) matched the fused singleton launch bit for
  bit. Partial accept 20 of 20 and the DSpark forced-acceptance matrix 18 of 18 (rollback, and verify against direct
  decode including the next draft and target rows), with no environment override. Short greedy parity 64 of 64.
- Prefill: with the fused group in prefill too, 1,024-token prefill took 21.73 / 20.87 s against 20.49 / 20.43 s
  (clean, both orders), so prefill keeps two launches per group and only verify rows use the fused group. On that
  route the 896 + 256 logits match the integration bundle bit for bit in prefill and decode, and a paired timing
  (order new, integration, integration, new) gave prefill 20.59 / 20.98 s against 20.66 / 20.52 s and decode
  168.5 / 171.7 against 169.1 / 168.8 ms per token (the first new run was flagged NOISY for a 2 s burst).
- Open: the verify-row cost of the fused group against two launches per group was not measured as a pair. One clean
  fused arm at 1,024 tokens: verify 2 / 4 / 8 rows took 675 / 914 / 1,355 ms.

## Linux on Strix Halo (2026-11-26)

Second machine: Ryzen AI MAX+ 395 (Strix Halo NPU, 8 columns), 123 GiB, Linux 7.2 (Omarchy), XRT 2.27 user space
from `phlegm/target/npu-dev/xrt`, NPU power mode `default` (left unchanged; the machine is shared). Toolchain
`phlegm/target/npu-dev/env.sh`: mlir-aie 1.4.2, llvm-aie `21.0.0.2026080301+c9c5ecb7` (the first machine's Peano
commit), conda GCC 15.3. The v11 bundle exported unchanged (64 kernels). Bridge: Phlegm `benchmarks/reference/build.py`
(the handoff's `build_bridge.py` is MSVC-only). NPU processes need unlimited memlock (`npu-run.sh`). The machine was
shared with other agent sessions (load average 25-60) for the whole sitting, so every timing below is NOISY.

Two Linux-only defects, both fixed (`d085750`):

- The late-readback overlap route was compiled for Windows only. Without it verify rows took the two-launch grouped
  experts with host SiTU, so partial accept failed 16 of 20 (the accepted rows and the following decode rows differed;
  `--reference trial` showed rollback itself was exact). Overlap now runs on every platform
  (`INKLING2_TEST_NO_OVERLAP=1` restores the synchronous route). Overlap and synchronous routes gave bit-equal prefill
  logits and 256 of 256 bit-equal decode rows after 512 tokens.
- GCC contracts the intrinsic `_mm_add_ps(sum, _mm_mul_ps(a, v))` of the 4-row Q8/F32 host dot tiles into FMA in C++
  (MSVC does not). A 4-row verify pass then rounded differently from the 1-row tail: widths 1-3 matched decode bit for
  bit, width 4 did not, even with four equal tokens. A standalone check gave 4 of 8 mismatching dots with default
  flags and 0 with `-ffp-contract=off`. The Inkling sources now build with `-ffp-contract=off` under GCC (CMake and
  Phlegm's `build.py`).

| Gate (Linux, bridge from `d085750`) | Result |
| --- | --- |
| Partial accept, prefixes 5 and 1,021 | 20 of 20 |
| Short greedy parity, 5-token prompt, 64 tokens (`inkling2_npu_check`) | 64 of 64 native IDs |
| Overlap against synchronous route, 512 + 256 | prefill logits and 256 of 256 decode rows bit-equal |
| `phlegm run`, reasoning none | correct one-sentence answer with a reasoning block |

Decode after a 1,024-token prefix (profiled bridge, NOISY, load average ~30): 355 ms a token (Strix Point Windows:
168 ms clean). Device wait is 191 ms over 193 dispatches (48 each of dense input, attention, dense output and fused
experts, plus the head). Host stages: K/V pack 58 ms and sync 12 ms, controller 40 ms, route 9 ms. The host stages
use OpenMP teams and are inflated by the shared CPUs; compare them only within a sitting.

## Rebuild the v11 kernel bundle

From the repository root, use the IRON Python environment with Peano and the XRT packaging tools
on `PATH`.

```text
python open_kernels/export_inkling2_kernels.py --out .plans/inkling2/rebuilt-v11
```

The exporter compiles 40 streams, then runs the Inkling manifest recipe. It creates a new output
directory after packaging succeeds. Set `--build-dir` to choose where the compiler outputs go, or
add `--package-only` to package the 40 existing stream directories with that layout.

Packaging the saved compiler outputs reproduced all 49 files of the tested v11 bundle byte for
byte. Its manifest SHA-256 is `d58ffa427f95a5dddbb936ac4b939bf42d87906b1d296061f6734266b3bc47ac`.

A fresh Windows IRON build compiled and packaged all 40 streams. Its manifest, all 40 instruction
streams, 1,256 worker ELFs and 120 initialization binaries match the tested captures byte for
byte. The eight XCLBIN containers differ. XRT's section dump matched all six sections outside
the AIE partition and matched the partition descriptions apart from their PDI UUID and filename.
Three embedded PDIs match byte for byte. The other five differ within the first 256 bytes only.
The newly packaged containers were then tested through `phlegm serve` on the NPU.
The red JSON and blue SSE cases matched the prior v11 choice and usage payloads,
and the server rejected a malformed PNG and answered a following text request.
The run took 137.13 s including setup and follow-up. Windows logged no power or
PCI context events. This is smoke coverage for the rebuilt bundle.

| Capture in `.plans/inkling2/export-v11/` | SHA-256 |
| --- | --- |
| `source-audit.json` | `5da5862b96a2fcce9f12cbcd7f8bdaf3b14dbce46c8a4193e7527525c45d548c` |
| `xclbin-sections.json` | `8de970ec1fbd1a36cbffff54ee649eac6fc67b692faca264c575f795f7d3aefb` |

| Smoke capture in `.plans/inkling2/f32-phlegm/priority-device-20261123/features/` | SHA-256 |
| --- | --- |
| `rebuilt-bundle.stdout.log` | `131ee3897b52ef8b3330a1fc356320d16c867abb43b3449190af3da0106a5b3c` |
| `rebuilt-bundle.windows-events.json` | `0b18d62ac72e8cf60ac5f9d6a82d8a115e0e7f983007dbc51f3af3db874be848` |
| `rebuilt-http/red.json` | `db0f2d9e34d94c21f8ec340be5ed1ebea72f725d3fa6bdfcc9880a414ba24c7e` |
| `rebuilt-http/blue.sse` | `0669a2bb5e46dbf8dc296e4774cf2cc7bd1a6fc224382ec0057bd6b8eabb0f21` |

## Windows NPU timeout evidence

Windows System log records 188561–188575 contain 14 `pci` Event ID 3 context reports from
10:32:14 to 10:41:30 local time on November 23. All reported `TxnOp=ffffffff` and zero fatal
error fields. This matches the lost-command signature in xdna2-notes F-0108. The queried window
had no Kernel-Power sleep or wake records. The mechanism that lost the commands remains open.

| Phlegm run | Matching driver evidence |
| --- | --- |
| DSpark 1,792 px vision | Record 188562 names bridge PID 141444 about 38 ms before the captured HTTP 500 |
| 8,225-token old-key control | Records 188566–188567 precede the baseline dense projection failure by about 15 seconds |
| 512-token finer-tier control | Records 188568–188569 precede the baseline projection failure by about nine seconds |
| 131K 10% recall | Record 188575 names bridge PID 109396 about eight milliseconds before the captured HTTP 500 |

The driver named other process IDs during both directly matched failures. Their roles are still
unknown. Reject performance or quality evidence from an interval containing matching context
errors. The mapped pack comparison ran at 10:36:00–10:36:02, the three-group board comparison
at 10:36:06–10:36:08, and the lossless comparison at 10:36:11–10:36:13. None of those sample
intervals contains a matching event. The three-group timing still misses its drift gate.

Phlegm applies its parent-death hook on Linux. On Windows the HTTP test harness now assigns
each dedicated server to a private job before making the first request, then waits for its bridge
processes to exit before clearing its lock file. This closes a possible source of overlap when a
test terminates its server. Normal shutdown, a crashed parent with a live child, and a foreign lock
passed the fixture. The actual Phlegm mock server returned HTTP 200 under the guard. The queued
real recalls and image requests will exercise this cleanup on the NPU. [Windows job objects](https://learn.microsoft.com/en-us/windows/win32/procthread/job-objects)

The filtered driver evidence is `.plans/inkling2/pci-timeout-triage.json`, SHA-256
`e08cb22d89279d8d2b2145b966c236975f30a4249002c54a69e8fcc2e960af5f`.
Use `benchmarks/reference/collect_windows_npu_events.ps1` with the test's start and end times.

## Phlegm release benchmark preparation

The Phlegm Git remote publishes the existing annotated `v0.1.0` tag, object
`e61ffdcfe1dea59d4f5f50852dbb559d2236ebcd`, at commit
`886c0f6cbb86250d8abe0646d6ec156131c828b6`. An untouched detached checkout built the CLI
with `--locked --offline --release --features npu`. Cargo recorded `dirty=false` and optimization
level 3. The actual binary passed the release eligibility check. A deliberately invalid context
capacity was then rejected before device access or report creation.

That tag accepts protocol 2 reference bridges and requires the tokenizer as JSON. The publisher's
sidecar from `7a67ae2314b44ef05c87fd1d4d311c58aaeb18f6` has SHA-256
`7f332727f9a3f093dd452a39cc978e6757627e5dad1b0673aa4e75174f236dd6`. It produced exactly
the same 16,640 token IDs as the pinned llama.cpp vocabulary reader on the IQ4_NL GGUF and the
tagged benchmark's repeated source passage. The 512, 1,024, 2,048 and 4,096-token prompts are
prefixes of those IDs. [Publisher tokenizer](https://huggingface.co/thinkingmachines/Inkling-2-Mini/blob/main/tokenizer.json)

The positive eligibility check reached context validation. With a valid context and the actual
Inkling inputs, `phlegm bench` rejected the current protocol 2 bridge before hashing the model or
opening the NPU. The CLI embeds the SHA-256 of its exact `benchmarks/reference/bridge.cpp`.
The `v0.1.0` adapter hash starts with `cd6d7ef0`, and the current adapter hash starts with
`c55eb53f`. A read-only check of Phlegm's Git remote at approximately 01:10 CST on November 24 still found
`v0.1.0` at `e61ffdcfe1dea59d4f5f50852dbb559d2236ebcd` as the only release-version tag.
Its protocol 1 adapter predates Inkling support. The benchmark waiter was withdrawn. The
preflight checks the adapter SHA before admitting another run.

Phlegm already has a development benchmark mode on its protocol-2
branch. A local staging tree applies the existing `44f812a` development
option and `1dc18d8` Windows path fix to the current Inkling-compatible
Phlegm sources. It preserves the normal adapter provenance check and
records the base commit, both development commits and every staged
source hash. The staged Cargo manifest passed offline metadata
resolution and the selected bridge, bundle and verified tokenizer
passed preflight. A short `phlegm bench --development` sweep is queued
after the 131K recall to keep the NPU and CPU clear for that measurement.
It uses the standard four prompt lengths, one warmup and one measured
run per length, and 32 output tokens. This is development evidence.
The published-tag benchmark remains a release gate.
`development/source-provenance.json` has SHA-256
`292808bbdd8899b0b56bd5d6db95747e51489da22e82374ff8d4d247c6c6632a`.

The development CLI was built before the NPU became free. The
first binary rejected `bench --development` because the staged
source had no Git commit metadata. An isolated Git index now uses
the real Phlegm base commit and records the staged edits as dirty.
The rebuilt binary passed that guard and stopped at the intended
invalid-context check. A second probe using the actual pinned
bridge passed its provenance checks and reached the intentionally
missing model, before NPU access. The queued run has the rebuilt
binary available. The SHA-256 is
`8d67a8d43fcd788fcabfd3a1e454ed7ff3c462798c3ac03d067b138da467f093`.

The development sweep completed all eight trials and `phlegm bench
validate` accepted its four context lengths, tokenizer IDs, hashes,
and NPU backend. Windows recorded seven Modern Standby transitions
and zero PCI context errors. The run is functional evidence only.
The development rerun added system, execution, and display power
requests scoped to the sweep. The machine was
on AC at launch. `development/run-33288/result.json` has SHA-256
`c194ab44b9c5f9a53d5e4190d9d02a5674b96e0b76816828894d4b9904a835f3`.

The rerun passed `phlegm bench validate` with zero power or PCI
context events. It used the Ryzen AI 9 HX PRO 370 on Windows
10.0.26200, NPU driver 32.0.20102.3930, six CPU threads, the
IQ4_NL + Q8_0 model, and the C++ reference bridge. Each case had
one warmup and one measured trial with 32 output tokens. AC was
present at both ends. These are development measurements.

| Input tokens | Prefill tok/s | Decode tok/s |
| ---: | ---: | ---: |
| 512 | 33.70 | 2.52 |
| 1,024 | 36.05 | 2.56 |
| 2,048 | 29.11 | 2.24 |
| 4,096 | 30.18 | 2.38 |

`development/run-151712/result.json` has SHA-256
`63f23cb9e38ec591053540e2fd9910e98cd43b0dfb4dc0ac2dd7f1483336e585`.
Its validated `report.json` has SHA-256
`8ba9165cdabd0e44c147bba371c6187df4f3c0cab88ded28616a82a9d57c1fda`.

The staged configuration uses the GGUF and publisher tokenizer, block prefill, six threads,
context capacity 8,192, one warmup and three measured trials with 128 output tokens at each
length. The result reader checks the prompt IDs, weight and tokenizer hashes, NPU backend, tag
and commit, and runs `phlegm bench validate`. The run timestamps support a post-run Windows PCI
and power-event check. It needs a published Phlegm tag that includes the Inkling adapter.

| Capture under `.plans/inkling2/release-bench/` | SHA-256 |
| --- | --- |
| `cargo-target/release/phlegm.exe` | `7b4ea96454f4ed39bbaa9cc1aa71b301b84ca891e3bea9c8983eb29d30b30649` |
| `canonical-bridge/phlegm-reference-bench.exe` | `b7392f059f65b60ed677f456743e14981ea68cea36840d166cac9ec8e8742695` |
| `tokenizer-preflight.json` | `a520733136ba53a5a00e934d3461d9ae9ec5674fe24624c68c4b22cfe7231320` |
| `expected-ids.json` | `c53b9a202fcb2103c0ce83b554be58dedf56bbcb8401835cd3dc7f80884ff173` |
| `bench-preflight.json` | `5d8babb87fcaaeb0205f06d4ceb81667f3713153a9ba7f163512006c6e3bbec7` |
| `adapter-mismatch.log` | `079298f649410dabbb5681ea9f7e24e28833d10cffd7f57efb1ef6f6b3242497` |

## Production bar

The native reference recheck passed all four Inkling tests on November 23 with the pinned
IQ4_NL GGUF. A single five-token block and 2+2+1 prefill produced all 200,058 logits bit for
bit, with the expected top five token IDs. The native GGUF pack check also passed, including
IQ4_NL and NVFP4 row decoding and the Q8 split-scale packing check. These were host tests.
The captures under `.plans/inkling2/host-build-v11/` are `native-reference-recheck.log`
(`f9ae113ca89f792dd50dc87cb66141fd054938bf900dece31f3ab1ce44db6746`) and
`native-gguf-pack-recheck.log` (`0cc4b3bd2fc0f338870ff61fe15bd3d40ca5e7c21cdf14219562c8a067f0fae8`).

The source import audit passed offline. The saved in-tree S3 import matches all 76 recorded
source hashes and preserves the original license. The audit verified 55 unchanged engine files
and 40 recorded changes after restoring five ledger hashes from the committed GPT-OSS and
Granite merge. On Windows, the source staging step confirmed Git's conversion for 79 CRLF
worktree files and left 10 ignored build outputs outside the audited source tree.
`.plans/inkling2/engine-import-audit/audit-stream-kv.log` SHA-256
`5c3074f3018165ec4f9deded8b25c1ee48df7d628cd34433d2876f53fe987dd1`.
[Offline import check](../in-tree-engine-s3.md#import-and-build-contract).

| Gate | Current evidence | Remaining work |
| --- | --- | --- |
| Block prefill | Phlegm selected 256-token layer-major blocks with IQ4_NL and Q8_0 experts, relative attention, and Q8 dense projections on the NPU. The 8,225-token comparison checked 32 full blocks and a 33-token tail. The 131K recall completed 512 blocks with zero host attention fallbacks | Repeat the route checks on the release bundle |
| Near-optimal prefill and decode | The integrated launch-overlap route cut warm Phlegm prefill from 23.17 to 14.11 s at 512 tokens in both orders and from 510.19 to 295.96 s at 8,225 tokens in one order, with exact logits. The clean development sweep measured 29.1–36.1 tokens/s prefill and 2.2–2.6 tokens/s decode at 512–4,096 tokens | Improve attention and expert throughput and repeat across contexts on the release build |
| Real context | `phlegm serve` passed all three depths at 8K, 32K and 131K on the baseline. The packed-key candidate returned the exact 90% needle at 131K with 89 min 27 s of profiled prefill, no host attention fallback, and zero power or PCI context events | Measure 1M Phlegm memory, prefill, and recall once that run is practical |
| NPU coverage | Routed experts, relative-position attention, block dense attention projections and Q8 logits run on the NPU. The shared singleton candidate cut warm 32-step decode by 10.6–17.7% against the normal Phlegm bundle in a clean bracket and preserved all 33 greedy IDs. The matched verifier passed all 18 acceptance and rollback cases | Qualify the exported image across longer contexts and confirm Phlegm throughput with the tracked build |
| Speculative decoding | The corrected Phlegm bundle matched direct decode bit for bit across all 18 forced acceptance and rollback cases. The optional paired head reproduced all verification logits. Its HTTP bracket preserved streamed bytes and showed a code speedup in both orders, mixed repetition results, and 12.9–28.5% control drift. Prompt lookup was slower | Keep DSpark opt-in and validate behavior on the release build |
| Quantized GGUF | ggml-org IQ4_NL with late Q8_0 layers has mean KL 0.006193 and 96.288% top-ID agreement on the 8K BF16 WikiText comparison. All three quants and Phlegm NPU IQ4_NL matched all 48 BF16 ARC choices. Unsloth had lower WikiText KL and needs native NVFP4 block and decode kernels | Keep the audited IQ4_NL candidate for the release evidence |
| Application and benchmarks | `phlegm run` and `phlegm serve` passed text smoke tests. `phlegm bench --development` produced a validated 512–4,096-token NPU sweep with zero power or PCI context events. The v0.1.0 tag's adapter SHA check rejected the current bridge | Run and validate `phlegm bench` with a published Phlegm tag that includes the Inkling adapter |
| Model features | CLI and HTTP thinking frames, short inference at none, low, high and max, and streamed and plain tool round trips. All eight Phlegm NPU image QA cases passed with speculation off. DSpark matched the first six on the NPU and matched the 1,792-pixel arrow and 4,096-pixel OCR on fresh-server retries | Run broader reasoning, tool calling and image workloads on the release build |

## Risks

- The arch is one day old. llama.cpp mainline and Unsloth's fork disagree on the GGUF layout, and the arch name
  differs (`inkling2_loop` / `inkling2`). Pin a llama.cpp commit for the reference.
- Relative-position logits and the dynamic convolution change every attention kernel's inner loop. They are the
  largest kernel cost and the largest accuracy risk.
- 48 visits double the dispatch count of a 24-layer model. One-context-per-layer designs (R-0003) matter more here.

## OFLM side

- `src/model_list.json`: an entry naming the text GGUF, the mmproj and the DSpark file, with the real quant
  (IQ4_NL + Q8_0), family `inkling2` and context 1,048,576.
- `oflm add` for the three files; `oflm serve` image requests; reasoning effort as a number and its parse; tool calls.
- `oflm bench` on a benchmark page; the README row; adding the model to #240.
