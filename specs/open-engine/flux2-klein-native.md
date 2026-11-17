# FLUX.2 [klein] native NPU route in Phlegm

Checkpoint: [black-forest-labs/FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
revision `e7b7dc27f91deacad38e78976d1f2b499d76a294` (Apache-2.0, step-distilled,
`is_distilled: true`). Reference: the unchanged diffusers `Flux2KleinPipeline`
(0.42.0.dev0, torch 2.14.1+cpu). Weights, packed weights, kernels and captures live in
`/home/cyrus/Documents/ChatGPT/current/work/phlegm-image-models/`, outside Git. The
working log is `.plans/flux2-klein-native.md`.

## Route

- Schedule: `open_kernels/klein_pipeline.py` (text encoder Qwen3-4B layers 1-27 with
  taps 9/18/27, conditioning, the denoising steps, the VAE decoder; an edit adds the VAE
  encoder and the reference tokens after the generated ones). Every op runs on the NPU;
  the host tokenizes, gathers the prompt's embedding rows, writes the noise (and an
  edit's reference pixels), patches `te_attn`'s valid length, and encodes the PNG.
- Kernels: `open_kernels/export_dit_kernels.py --resolutions 512,1024 --edits 512,1024`.
  The dit_gemm marker `dit_kernels.json` lists `resolutions` and `edits`; the server
  accepts exactly those sizes.
- Runner: `open_kernels/flux2_native_generate.py` on the pyxrt host
  (`open_kernels/harness/npu_host.py`), reading the diffusers checkpoint directly (no
  q4nx bundle). It refuses a checkpoint whose transformer/text-encoder geometry differs
  from the schedule and undistilled (Base) checkpoints. Every queued `xrt::run` is kept
  alive until it completes: Linux XRT aborts the process when one is destroyed in flight.
- Dispatch (`--dispatch`, default `elf`): one hardware context from the configuration's
  full ELF (`diffusion_r<key>.elf`, `diffusion_elf.json`), ported from OFLM's
  `src/open_diffusion/engine.cpp`. Each set change opens a stretch: the set's
  configure-only kernel (`main:cfg_<set>_a`/`_b`, alternating) and the ops after it, as
  one `xrt::runlist`, at most 32 in flight. te_attn's valid length is the
  `fa:te_attn_vl<n>` head run before it. Runs are made per image and dropped when
  their stretch completes, so any step count stays under the driver's live-run limit.
  pyxrt cannot set the context's QoS priority (OFLM uses high, 0x180, so another
  process cannot preempt a half-configured set); Phlegm relies on the Dibs `npu`
  lease instead. `sets` keeps one context per kernel set: pixel-identical, 1.45x
  slower, and on Windows its process exits 127 after a good run.
- DiT source (IMAGE-FLUX2-GGUF): `--transformer-gguf` reads the DiT from a published
  GGUF (`open_kernels/klein_gguf.py`); `--model` still supplies the text encoder, VAE
  and tokenizer. `open_kernels/klein_fetch.py` (IMAGE-FLUX2-GGUF-FETCH) installs from a
  GGUF repo and writes the config.
- Server/CLI: `backend=flux2` in `phlegm-core/src/image.rs` (`Flux2ImageConfig`:
  `model_id`, `model`, `kernels`, `python`, `repo_root`, optional `env_script`/`dibs`,
  `transformer_gguf`/`gguf_modulation`, `output_root`). Steps 1-50 (default 4);
  `guidance_scale` and `negative_prompt` are refused. On Windows (IMAGE-FLUX2-WINDOWS)
  the runner starts directly with the environment of the PowerShell `env_script`.
- Noise: without `--noise`, numpy PCG64 from the seed; an edit uses the stream
  `[seed, 1]`. With the text-to-image stream, editing an image generated with the same
  seed reproduced the reference unchanged (measured below).

## Results (2026-11-08, Strix Halo XDNA2 (Ryzen AI MAX+ 395), Fedora 44, XRT 2.27.0 a29919c)

Native runs with the reference's exact initial latents, 4 steps, against the bf16
published pipeline. Pixel PSNR/MAE are on 8-bit RGB.

| case | final latents rel L2 / cosine | MAE / PSNR | NPU time (one run) |
|---|---|---|---|
| 512 t2i "A red fox in fresh snow" | 34.2% / 0.9411 | 10.33 / 21.97 dB | 5.00 s |
| 1024 t2i, same prompt | 36.8% / 0.9327 | 9.15 / 23.04 dB | 12.47 s |
| 512 edit "Make the fox blue" | 8.96% / 0.99598 | 1.73 / 39.20 dB | 7.24 s |
| 1024 edit, same | 8.37% / 0.99656 | 1.58 / 39.63 dB | 25.93 s |
| 512 edit "Replace the noodles with sushi" | 19.7% / — | — / 26.37 dB | — |

All images are coherent and follow their prompts; text-to-image keeps the reference's
composition with small pose/detail drift, which matches the OFLM study of this
arithmetic (all-NPU LPIPS 0.107 mean against bf16, `OPEN-DIFFUSION-QUALITY`; LPIPS
not measured here). NPU times are single runs under Dibs, not benchmarks.

`phlegm image` and `phlegm serve-images` each completed text-to-image and one-reference
edits at 512 and 1024 on hardware (HTTP 200; `/v1/models` lists
`black-forest-labs/FLUX.2-klein-4B`; `guidance_scale` -> 400).

## Strix Point, Windows: GGUF, quality, speed (2026-11-14)

Machine: Ryzen AI 9 HX PRO 370 (Strix Point, XDNA2), Windows 11 26200, XRT 2.21.0,
driver 32.0.20102.3930, NPU firmware 1.1.2.64, mlir-aie 1.4.2. Kernel set built here
(`export_dit_kernels.py --resolutions 512,1024 --edits 512,1024`, 268 streams, ~75 min
with a CPU job beside it). Evidence: `C:\dev\klein-eval\` (not in Git).

**GGUF.** `Atomic-Germ/FLUX.2-Klein-4B-OpenFLM-GGUF` revision `8deec59d`,
`flux-2-klein-4b-Q8_0.gguf` (4.30 GB; 80 Q8_0, 9 BF16, 60 F32 tensors). Its adaLN
tensor is BFL-ordered (detected locally and by range read). Against the BF16
checkpoint: the BF16/F32 tensors and the swapped adaLN tensor are exact; Q8_0 block
weights are 0.56% relative L2 (cosine 0.99998). All 86 schedule weights build.
Q8_0 is the shipped quant: OFLM measured Q4_1 visibly grainy, and the NPU runs
BFP16 whatever the source, so a 4-bit source saves disk only.

**Quality.** Reference: the unchanged diffusers 0.40.0 `Flux2KleinPipeline` on CPU
(torch 2.14.0, bf16), seed 1, 4 steps, its noise injected. The edits edit the
reference's own t2i image. LPIPS is AlexNet.

| case | BF16: latents rel L2 | PSNR dB | LPIPS | Q8_0 GGUF: latents rel L2 | PSNR dB | LPIPS |
|---|---|---|---|---|---|---|
| fox512 | 37.5% | 21.17 | 0.090 | 34.7% | 21.70 | 0.080 |
| lighthouse512 | 25.4% | 26.93 | 0.066 | 25.9% | 26.78 | 0.069 |
| ramen512 | 29.0% | 23.54 | 0.051 | 29.9% | 23.39 | 0.053 |
| foxblue512 (edit) | 9.0% | 39.20 | 0.004 | 9.1% | 38.18 | 0.004 |
| sushi512 (edit) | 11.8% | 32.13 | 0.010 | 12.1% | 31.17 | 0.010 |
| fox1024 | 35.6% | 23.38 | 0.089 | 36.1% | 23.08 | 0.088 |
| foxblue1024 (edit) | 8.2% | 40.75 | 0.004 | 8.4% | 40.09 | 0.004 |

Mean LPIPS: BF16 0.0449, Q8_0 0.0441. The Q8_0 source is equal to BF16 within the
spread. The t2i LPIPS (0.05-0.09) is below the OFLM study's all-NPU 0.107 mean.

**Speed.** Q8_0 GGUF, ELF dispatch, `--repeat`, median, NPU power mode
`performance` (read back before and after), Dibs `npu`+`cpu` lease held.

| request | NPU total | cond / text / encode | per step | VAE | runs |
|---|---|---|---|---|---|
| t2i 512, 4 steps | 3.90 s | 0.035 / 0.342 / - | 0.797 s | 0.334 s | 6 (3.89-3.95) |
| t2i 512, 8 steps | 7.11 s | 0.035 / 0.347 / - | 0.806 s | 0.334 s | 3 |
| t2i 1024, 4 steps | 12.31 s | 0.035 / 0.339 / - | 2.70 s | 1.11 s | 4 (12.27-12.42) |
| edit 512, 4 steps | 6.38 s | 0.035 / 0.340 / 0.204 | 1.37 s | 0.337 s | 4 |
| edit 1024, 4 steps | 27.12 s | 0.036 / 0.342 / 0.661 | 6.26 s | 1.12 s | 3 |
| t2i 512, `--dispatch sets` | 5.66 s | 0.045 / 0.734 / - | 1.10 s | 0.497 s | 6 |

The ELF dispatch is pixel-identical to `sets` (lighthouse512, max difference 0) and
1.45x faster: 822 set changes per image cost ~2 ms each as context switches.

Host stages per image (all small): tokenize and embedding rows 1.5-2 ms, noise
<0.1 ms, reference image 10 ms (512) / 50 ms (1024), RGBA read 2-7 ms, PNG 35-115 ms.
Load: 3.6-7.4 s per process (weights 7.55 GiB from the packed cache in ~2 s, plus ELF
kernels); a first run packs the weights once (~2 min, 6 processes).

**Profile** (`--profile`: one serialised run, a wait after every op; 4.32 s at 512,
12.79 s at 1024). Steps: GEMM 56% / 53%, attention 13% / 28%, elementwise 12% / 7%;
VAE 8%; text 10% / 3%. Per op at 1024: `sgl_in` (M 4608, K 3072, N 27648) 42.1 ms =
18.6 TFLOPS, `sgl_out` 24.4 ms = 14.3 TFLOPS, single-block attention 36.0 ms = 7.2
TFLOPS. At 512: `sgl_in` 14.7 ms = 17.8 TFLOPS, attention 5.7 ms = 5.1 TFLOPS.

What is left, largest first:
1. Attention: 7.2 TFLOPS at 1024 against dit_gemm's 18-19 (28% of the 1024 steps).
2. The resident process: `phlegm image` and `serve-images` start the runner per
   request, so 3.6-7.4 s of load adds to every request (HTTP 512 edit 13.4 s wall for
   6.4 s on the NPU; 1024 t2i 19.7 s for 12.3 s). OFLM's engine keeps the weights.
3. Elementwise ops (292 per image) as separate dispatches; fusing them into the GEMMs
   is the known 1.26-1.40x lever (xdna2-notes F-0314).
4. `sgl_out` reads a gathered A at 14.3 TFLOPS against `sgl_in`'s 18.6.

GEMMs run at the dit_gemm ceiling measured elsewhere (17-19.5 TFLOPS, F-0301).

**CLI and HTTP** (IMAGE-FLUX2-WINDOWS, config from `klein_fetch.py`): `phlegm image`
512 t2i in 9.4 s wall, the prompt `A "quoted" otter; ... $HOME & %PATH%` unchanged in
`summary.json`. `serve-images`: `/v1/models` lists `black-forest-labs/FLUX.2-klein-4B`;
1024 generation 200; edits 512 and 1024 (multipart) 200 with the expected blue fox;
`guidance_scale` 400; 768x768 400.

## Production bar (OFLM `AGENTS.md`, #240)

| Item | Status |
|---|---|
| Block route, context length, speculative decoding | Not applicable (diffusion); prompts truncate at 512 tokens as the published pipeline does |
| Near optimal, profiled | Profiled above; what is left is named |
| All on the NPU | Every op; host stages named and timed above |
| `add` / `serve` | `klein_fetch.py` + `phlegm image` / `serve-images`: Windows with the GGUF config (2026-11-14); Linux with the checkpoint config (2026-11-08), the GGUF config not yet run there |
| Ships quantized | Q8_0 GGUF DiT (Q4_1 rejected: grainy, per OFLM); text encoder and VAE BF16 from the checkpoint, no GGUF of either exists |
| Benchmarks recorded | Here, Strix Point Windows; Strix Halo numbers above are single runs. OFLM's `docs/docs/benchmarks/` page and README row are OFLM-side work |
| Features | t2i and one-reference edits at 512 and 1024, 1-50 steps. Left out: other sizes (kernel sets are per size), multi-reference edits (one reference per edit in the schedule), CFG / negative prompt (distilled model) |

## Swapped text encoder (2026-11-08)

The route takes any text encoder with Qwen3-4B's exact geometry: build a checkpoint
directory whose `text_encoder/` is the replacement and whose other entries link to the
published checkpoint, and point the config's `model` at it. Give it its own `kernels`
directory (links to the kernel set plus its own `packed/`), or the packed-weight cache
repacks every time the two are swapped. `embed_rows` rounds a non-BF16 embedding table to
bf16 (it used to copy the raw bits, which is wrong for an F16 file);
`specs/open-engine/tests/test_flux2_embed_rows.py` covers both.

Measured: `ponpoke/flux2-klein-4b-uncensored-text-encoder` at `633217e5`
(`model.safetensors` sha256 `3d6f27fd...`, matches the Hub). It is an F16 re-save of the
published encoder: same 398 tensors and shapes; layers 0-13 equal the published weights
after an f16 round trip; `o_proj` and `down_proj` changed in layers 14-35 (2.0-3.1%
relative L2 each). Its prompt embeddings for "A red fox in fresh snow" differ from the
published encoder's by 0.79/0.78/0.80% at taps 9/18/27. 512 text-to-image, 4 steps, the
reference's noise, against the unchanged diffusers pipeline with the same encoder:
latents 38.9% rel L2 / cosine 0.9238, pixels MAE 11.04 / 21.56 dB (published encoder:
34.2% / 21.97 dB, table above). `phlegm image` t2i and a "Make the fox blue" edit at 512
both produced the expected images. Edits and 1024 were not compared against the
reference. License per its card: flux-non-commercial-v2.1.

## Windows Python environment (verified 2026-11-11)

Activate the existing IRON environment in each PowerShell session before running
the Python host commands below:

```powershell
. C:\dev\mlir-aie\iron_env.ps1
python -c "import pyxrt; print(pyxrt.device(0).get_info(pyxrt.xrt_info_device.name))"
```

This selects Python 3.13.7 at
`C:\dev\mlir-aie\ironenv\Scripts\python.exe` and the SDK binding at
`C:\Xilinx\XRT\python\pyxrt.pyd`. Device 0 opens as NPU Strix, BDF
`00c5:00:01.1`, on Ryzen AI 9 HX PRO 370. XRT 2.21.0,
driver 32.0.20102.3930, firmware 1.1.2.64, Windows build 26200.8655.
The official IRON SAXPY example passed under `dibs run npu`; the device lease
was released on exit. All 23 conditioning checks pass in this environment.

Added safetensors 0.8.0 and pytest 9.2.0 to the existing environment. Existing
numpy 2.5.3, ml_dtypes 0.6.0, tokenizers 0.23.2 and Pillow 13.0.0 supply the
remaining runner dependencies. The default shell selects Python 3.11, so
activation is required. Evidence lives in ignored `.plans/pyxrt-20261111/`.
The earlier missing-SDK report came from a sandbox access denial. The installed
SDK and activation script work when accessed outside that sandbox.

## Experimental projected conditioning (2026-11-10)

The local CLI accepts `phlegm image --config CONFIG --prompt TEXT --out DIR
--projected-conditioning FILE`. The Python runner accepts the same input flag.
HTTP image JSON does not accept a host conditioning-file selector.

The safetensors file must declare metadata protocol
`klein-projected-conditioning-v1` and contain:

| Tensor | Type and shape | Meaning |
|---|---|---|
| `projected_conditioning` | BF16/F16/F32 `[512,3072]` | All projected rows, including padding |
| `text_ids` | I64 `[512,4]` | `[0,0,0,L]`, L=0..511 |
| `input_ids` | I64 `[512]` | Exact native tokenizer output for `--prompt` |
| `attention_mask` | I64 `[512]` | Nonempty right-padded sequence; padding ID 151643 |

Inputs are finite, bounded to 16 MiB, rounded once to BF16 and hashed from the
consumed byte snapshot. BF16 bits are preserved. Token IDs/mask must match the
normal prompt template/truncation exactly. Current companion targets and
`save_prediction` outputs retain this metadata; older predictions without IDs/mask
are numerical evaluation artifacts only.

This mode omits text weight loading/packing, text dispatches and native `ctx_emb`;
it uses `packed_projected/` alongside the normal cache. Denoising mutates X's text
rows, so the host drains pending operations and restores conditioning at every
original `ctx_emb` position. Buffer and parameter layout remains unchanged, with
unused text slots zero. A compatible full transformer/VAE checkpoint, tokenizer,
model index and compiled kernel sets are still needed. The projected mode does
not require text-encoder weights/config; the normal mode still does.

Offline validation: 23 Python checks pass on Windows and Linux, six core image
tests and four CLI tests pass. All twelve real BF16 Windows targets preserve bits;
all twelve real FP32 Linux targets match Torch's BF16 conversion and prompt IDs.
These results do not execute NPU kernels. No kernel arithmetic or ABI changed.

Live gate, **not yet run**, in the existing pyxrt environment on the NPU host:

```sh
python open_kernels/flux2_conditioning_replay.py --model FULL_CHECKPOINT \
  --kernels KERNEL_SET --out FRESH_CAPTURE_DIR --size 512 --steps 4 --seed 17
```

This requires the normal text encoder too: it captures the native context
projection before later queued operations can mutate it, verifies that capture
preserves outputs, then requires exact replay latents/pixels, a changed output
under zero conditioning and exact restoration on another request. All runs use
the same saved noise; sources, weights, kernels and artifacts are hashed. This
checks the runtime boundary, without claiming PyTorch teacher targets equal
native compensated-BFP conditioning. After this gate, compare those teacher
targets under the existing numerical/image-quality gate. Editing and 1024 remain
additional hardware coverage. No new quality or performance acceptance yet.

## Open

- Separate research: [compact conditioning and VAE decoder distillation](flux-conditioning-distillation.md).
  Encoding precedes optional enrichment; a replaceable output head targets the
  existing denoiser first. No student integration or performance claim yet.

- FLUX.2 [klein] 9B (gated, non-commercial): config unread, so no schedule yet.
- Base (undistilled) variants: CFG, deferred by decision.
- Prompts over 512 tokens are truncated as the published pipeline does.
- The four "what is left" items under the 2026-11-14 profile.
