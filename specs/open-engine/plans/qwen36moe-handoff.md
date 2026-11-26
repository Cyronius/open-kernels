# Qwen3.6-35B-A3B production bar: handoff (2026-11-26)

For the session that resumes this work on another machine. The detailed evidence is in
[qwen36moe-production-bar.md](qwen36moe-production-bar.md) (the bar, item by item) and
[qwen36moe-attn-fa256.md](qwen36moe-attn-fa256.md) (the fused attention kernel). Read both first.

## Where the work is

| repo | branch | what it holds |
|---|---|---|
| open-kernels (`Cyronius/open-kernels`) | `feat/qwen36moe-production-bar` | engine, recipe, designs, specs, plans, the helper scripts in `benchmarks/reference/qwen36moe/` |
| phlegm (`Cyronius/phlegm`) | `feat/qwen36moe-bar` | the bridge (`benchmarks/reference/bridge.cpp`), the speculative mode in `init`, the MTP drafter policy, the Qwen3.6 chat-template `{% generation %}` tags |

Neither branch has a PR. In phlegm, the open-kernels submodule pointer was not moved; move it after the
open-kernels branch merges.

## Done (with measurements in the two plans)

- Unsloth GGUF `Qwen3.6-35B-A3B-UD-Q4_K_S` with q8 roles, block prefill, needles to 262,000 tokens.
- Two-row verify pass and the MTP drafter (blk.40), through the CLI and `phlegm serve` (`--speculative mtp`):
  1.24-1.40x decode on the rewrite prompt, output byte-identical to `off`. Prompt lookup measured and kept as an
  option; per-cycle MTP / lookup combination rejected at width 2.
- The linear state as a 16-slot ring (commit re-points the view): commit 3.7 ms.
- phlegm: the bridge reads the speculative mode from `init` (no `OFLM_*` variables), no context cap, MTP keeps
  drafting while its running acceptance is at or above 0.5.
- `attn_fa256` (new, 2026-11-25/26): the full-attention layers' block prefill as one NPU dispatch a block.
  KL / perplexity equal to the old route, needles PASS at 4k, 8k, 32k (three depths), 131k and 262k.
  262,000-token prefill 2,457 s against 14,706 s. 32k prefill 5.18 ms per token against 8.59.

## Open, in the order to take them

1. **Attention kernel speed** (`open_kernels/designs/attn_fa256/`). 7.7 us per 64-key chunk and pass against a
   3.35 us transfer floor (L 8192). Ablations and the rejected paired-QK variant are in the fa256 plan. Ideas not
   yet tried: the hardware bf16 exp2 in the softmax (accuracy gate needed), fewer kernel calls per chunk on the PV
   side, Q converted to bfp16 once per pass, deeper A / B fifos if L1 allows (58.5 of 64 KB used).
2. **Linear-attention layers in prefill** (now the larger cost, 3.5 ms per token): routed MoE `moe_batch` (1.5 ms;
   lever: native `mmul<4,8,8>` and `MB_SHARE_NG`, see the bar plan's Phase 4), and the host DeltaNet conv and
   delta rule (0.9 ms; move to the NPU). MB_NT 16 was measured and rejected.
3. **Overlap in the attention route**: Q pack, O read-back and the gate run between dispatches (0.06 ms per token
   at 32k); the GEMM / attention context switch is twice a layer.
4. **262k through `phlegm serve`** (the context cap is gone; untested at that length).
5. **Vision** (waits on phlegm image input), **`phlegm bench`** run and records. The docs page and the README row
   are OFLM's side.
6. **Cleanup**: probe switches `QWEN36MOE_PROBE_NE`, `QWEN36MOE_PROBE_MB_NT`, `X_ROWS2_PROBE`, `GEMV_ROWS2_PROBE`,
   `X_Q8_ROWS2_PROBE`, `X_Q8_TWICE_PROBE`, `OFLM_VERIFY_PROFILE`; the plain-decode variance (7.65-9.06 tokens/s
   between runs, cause not found).
7. After any engine edit: refresh `benchmarks/reference/engine-changes.json` (LF-normalized hashes; the plain
   `check_import.py` fails on a CRLF checkout).

## Set up the new machine

The scripts in `benchmarks/reference/qwen36moe/` assume this layout. Change the paths at the top of each script
if the new machine differs.

- `C:\dev\mlir-aie\iron_env.cmd` (mlir-aie 1.4.2, Peano 21.0.0), `C:\Xilinx\XRT`, `C:\dev\XRT` headers and
  `C:\dev\xrtNPUfromDLL` (see `cli_build.cmd`). Native Windows; no WSL.
- `C:\code\open-kernels-qwen36moe` (this branch), `C:\code\phlegm-qwen36moe` (phlegm's branch).
- `C:\models\qwen36moe\Qwen3.6-35B-A3B-UD-Q4_K_S` (Unsloth GGUF plus its tokenizer.json / config).
- The `arbiter` tool for the NPU and build locks (`arbiter hold npu` / `release`; `arbiter run` cannot make its
  Windows Job Object in an agent session). `npuq.sh` wraps a command in the hold.

Rebuild everything (no artifacts are committed):

```bash
B=/c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe
cmd //c "C:\code\open-kernels-qwen36moe\benchmarks\reference\qwen36moe\cli_build.cmd"        # the CLI and its unit tests
cmd //c "C:\code\open-kernels-qwen36moe\benchmarks\reference\qwen36moe\export_fa262k.cmd"     # kernels-fa262k, about 1 h cold
bash $B/rebuild_bridge.sh                                                                       # bridge, phlegm, sidecar
```

The export's build root is `C:\models\qwen36moe\build`. A probe export (`QWEN36MOE_PROBE_*`) into the same root
overwrites the shared builds; give a probe its own `--build-root`.

## Check that the new machine reproduces

```bash
B=/c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe
K='C:\models\qwen36moe\kernels-fa262k'
python -P $B/needle.py make 8192 0.5 C:/models/qwen36moe/ctx/needle/n8192_0.5.txt     # needs kl/essays.txt
bash $B/needle.sh "$K" 8704 n8192_0.5                    # PASS, about 38 s of prefill
bash $B/prefill_prof.sh "$K" n8192_0.5                   # about 4.6 ms per token
bash $B/gate.sh fa "$K" 8                                # KL mean about 0.0235 (needs kl/ids and base_vk_c512.kld)
```

The KL gate needs the llama.cpp reference logits (`kl/base_vk_c512.kld`) and the chunk ids (`kl/ids/`); copy
`C:\models\qwen36moe\kl` from the old machine or regenerate them with llama.cpp's `--kl-divergence-base`.
`essays.txt` is the needle haystack; copy it too.

Kernel alone (no model):

```bash
B=/c/code/open-kernels-qwen36moe/benchmarks/reference/qwen36moe
mkdir -p /c/models/qwen36moe/fa256 && cd /c/models/qwen36moe/fa256
cmd //c "C:\code\open-kernels-qwen36moe\benchmarks\reference\qwen36moe\fa256_build.cmd 8192 C:\models\qwen36moe\fa256\s8192 8192"
bash $B/fa256_mk.sh s8192 ts8192 8192 && bash $B/fa256_run.sh ts8192    # about 7.8 ms, rel_fro 9.0e-3
```

Do not carry absolute speeds across machines (Strix Point against Strix Halo, other drivers): rerun the
references above first and compare against those.
