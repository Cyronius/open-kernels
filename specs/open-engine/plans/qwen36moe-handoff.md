# Qwen3.6-35B-A3B production bar: handoff (2026-11-26, Strix Halo)

For the session that resumes this work. The evidence is in the spec's audit (`spec.md`, "Production bar
(AGENTS.md, OFLM #240): qwen3.6-moe:35b-a3b", with "What is left"), [qwen36moe-production-bar.md](qwen36moe-production-bar.md)
(the plan and its log) and [qwen36moe-attn-fa256.md](qwen36moe-attn-fa256.md). Read the audit first.

## Where the work is

| repo | branch | what it holds |
|---|---|---|
| open-kernels (`Cyronius/open-kernels`) | `feat/qwen36moe-production-bar` | engine, recipe, designs, specs, plans, the helpers in `benchmarks/reference/qwen36moe/` |
| phlegm (`Cyronius/phlegm`) | `feat/qwen36moe-bar` | the bridge (`benchmarks/reference/bridge.cpp`), chat image input (`phlegm-core/src/vision.rs`, SERVER-IMAGE-INPUT), the speculative mode in `init`, the MTP drafter policy, the `{% generation %}` tags |

Neither branch has a PR. In phlegm the open-kernels submodule pointer was not moved; move it after the open-kernels
branch merges. On this box the phlegm branch is the worktree `~/code/phlegm-wt/qwen36moe-bar`; its `open-kernels/`
submodule gets this checkout's engine sources copied in by `rebuild_bridge.sh` (so its tree reads modified).

## State (2026-11-26)

The kernel side passes the bar (RECOMMENDED_MODELS.md): block prefill, needles to 262,000, the MTP drafter, the
Unsloth GGUF at the backends' KL spread, `phlegm run` / `serve`, benchmarks recorded, vision end to end, every host
stage named, near optimal with the remainders named. New today:

- Vision (OPEN-VISION-BLOCK, SERVER-IMAGE-INPUT): image prompts on the block route at M-RoPE positions; the host
  vision tower on packed AVX-512 GEMMs; `phlegm serve` image parts and `phlegm run --image`; answers equal to
  llama.cpp's on the test image; MTP exact after images.
- OPEN-MOE-PIPE (prefill MoE passes pipelined, 1.19-1.29x at 4k, bit-identical) and OPEN-DN-FUSED (DeltaNet host
  1.4x, KL at the backends' spread).
- Linux: shmem device buffers where XRT 2.21 refuses userptr ones; the helpers use phlegm's npu-dev XRT 2.27.
- The timing probes are gone (kernels byte-identical after a re-export).
- One-context decode and verify images for the q8 spec (`ux`, `vux`): decode 94 -> 73.5 ms a step; phlegm drafts only
  while it pays (SERVER-SPEC-COST).

## Open, in the order to take them

1. **Decode** (73.5 ms a step, 1.31x its 56 ms stream floor, on one context since OPEN-DECODE-ONE-CONTEXT-Q8): the
   streams run at 38-43 GB/s against 52; the routed experts' hidden round trip; the tail's two context changes
   (`ln`, `lm`, about 1.6 ms). The verify pass costs 1.67 plain steps (two-row glue / DeltaNet / post serialized,
   twice the experts), which keeps MTP near break-even on Strix Halo.
2. **Prefill** (2.99 ms a token at 8k): the token-batched expert kernel at 16 tokens a visit (OPEN-MOE-DQBUF) is
   core-bound again (2.34 ms for 32 slots against a 1.53 ms stream; native `mmul<4,8,8>` measured slower); the DeltaNet conv and delta rule on the NPU (a chunked kernel); `attn_fa256` at 2.3x
   its transfer floor (hardware bf16 exp2 with an accuracy gate, fewer PV calls, Q to bfp16 once a pass).
3. **Vision on the NPU** only if the tower outgrows the text prefill (3.7-4.0 s for 3,200 patches against about
   4.7 s of text prefill today).
4. **OFLM side** (not this repo): registry entry, `oflm add` / `serve` / `bench`, benchmark page, README row.

## This machine (Strix Halo, Linux)

- Toolchain: `~/ironenv142` (mlir-aie 1.4.2, Peano 21.0.0.2026080301); aiebu-asm and XRT 2.27 from
  `~/code/phlegm/target/npu-dev/xrt` (`paths.sh` sets PATH, XILINX_XRT, LD_LIBRARY_PATH).
- Model: `~/models/qwen36moe/Qwen3.6-35B-A3B-UD-Q4_K_S` (`model.gguf`, `mmproj-BF16.gguf`, Qwen's config and
  tokenizer @ `345d467b`); kernels `~/models/qwen36moe/kernels-fa262k`; KL data `~/models/qwen36moe/kl` (haystack
  `essays.txt` = the first 40 essays of `sgoel9/paul_graham_essays` @ `0c7155a5`; `base_vk_c512.kld` llama.cpp Vulkan,
  `base_cpu_c512.kld` CPU); llama.cpp `~/code/llama.cpp/build-cpu` and `build-vk`.
- The NPU power mode is Default (setting it needs root; the user chose to keep Default). The box is shared: take
  every NPU run through `arbiter` (`npuq.sh` does), and accept a change only on alternating A/B ratios.

```bash
B=~/code/open-kernels/benchmarks/reference/qwen36moe
$B/cli_build.sh; $B/export_fa262k.sh; $B/rebuild_bridge.sh                 # CLI + tests, kernels, phlegm + bridge
$B/needle.sh ~/models/qwen36moe/kernels-fa262k 8704 n8192_0.5             # PASS
$B/gate.sh fa ~/models/qwen36moe/kernels-fa262k 8                         # KL against kl/base_vk_c512.kld
$B/prefill_prof.sh ~/models/qwen36moe/kernels-fa262k n4096_0.5 n8192_0.5
$B/vision_check.sh ~/models/qwen36moe/kernels-fa262k ~/models/qwen36moe/vision/banner_q.txt ~/models/qwen36moe/vision/banner640.ppm
arbiter run npu --eta 15m -- $B/vision_serve.sh                            # phlegm serve image conversation
$B/bench.sh                                                               # phlegm bench --development
arbiter run npu --eta 100m -- $B/needle_serve.sh 262000 263168             # 262k through phlegm serve
```

The Windows (Strix Point) paths still work: `paths.sh` picks them under git-bash, and the `.cmd` files stay.
