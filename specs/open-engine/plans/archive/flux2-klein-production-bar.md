# FLUX.2 [klein] 4B to the production bar (OFLM #240)

Spec: [flux2-klein-native.md](../flux2-klein-native.md). Bar: OFLM `AGENTS.md`
"Production bar for OFLM kernel sets" (#239). Started 2026-11-14.

## Spec impact

New requirements:

- **IMAGE-FLUX2-GGUF** (test): the runner takes the DiT from a published GGUF
  (`--transformer-gguf`). The text encoder, VAE and tokenizer still come from the
  diffusers checkpoint, because no GGUF of them exists. BFL tensor names map to the
  diffusers names. Fused q|k|v splits in order. F32, F16, BF16, Q8_0, Q4_0 and
  Q4_1 decode. K-quants are refused. The final adaLN halves swap when the file
  uses BFL order (scale, shift).
- **IMAGE-FLUX2-GGUF-FETCH** (manual): one command fetches a GGUF repo's DiT and
  the base checkpoint parts its README names, then writes a `phlegm image` config.
  This is phlegm's form of `oflm add`.
- **IMAGE-FLUX2-WINDOWS** (manual): `phlegm image` and `phlegm serve-images` run
  the `flux2` backend on native Windows (PowerShell env script, no bash).

Ported from OFLM `c2a9e094` (#232): `utilities/dit-chain/gguf_weights.py`,
`gguf_source.py`, `klein_checkpoint.py`.

## Bar items and how this plan answers them

| Item | Answer |
|---|---|
| All on the NPU | Already true; measure each host stage's cost |
| Near optimal | Profile per phase (text, cond, steps, vae) and per op family; name what is left |
| Quantized | Q8_0 GGUF source (OFLM measured Q4_1 visibly grainy); the NPU runs BFP16 either way |
| Install/serve | `klein_fetch.py` + `phlegm image` / `serve-images` on Windows |
| Benchmarks | Repeated runs, median, machine and stack named |
| Features | t2i + one-reference edits at 512 and 1024; name what is left out |
| Quality | Q8_0 vs BF16 on the NPU; LPIPS against the diffusers reference |

## Steps

1. Build the kernel set on Windows (`export_dit_kernels.py --resolutions 512,1024 --edits 512,1024`).
2. Port the GGUF reader to `open_kernels/klein_gguf.py`; wire `--transformer-gguf`
   into the runner and `transformer_gguf` into `Flux2ImageConfig`.
3. Test IMAGE-FLUX2-GGUF with a synthetic GGUF.
4. Windows launch path for the `flux2` backend.
5. Hardware: BF16 vs Q8_0 on the NPU, profile, benchmark, CLI and HTTP runs, LPIPS.
6. Merge into the spec, archive this plan.
