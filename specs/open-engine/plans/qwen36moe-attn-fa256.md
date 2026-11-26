# Qwen3.6-35B-A3B: fused block-prefill attention at head dim 256

Status: step 1 done, step 2 in progress (2026-11-25). Design: `open_kernels/designs/attn_fa256/`.

## Results so far (2026-11-25, Strix Halo, quiet NPU under an arbiter hold; not `--measure`)

Kernel on random data (make_test.py: q/k scaled as after a q/k norm, an attention sink on row 0), fp64 reference of
the kernel's rounded inputs, `compare.py`. One dispatch = one 256-token block, 16 heads, against L rows:

| build | L | p0 | ms | rel_fro | worst row cos |
|---|---:|---:|---:|---:|---:|
| K on A, V on B (depth 2 / 1), 8 KB pieces | 1024 | 768 | 1.55 | 8.9e-3 | 0.99993 |
| same | 4096 | 3840 | 4.9 | 9.1e-3 | 0.99994 |
| same | 4096 | 1000 | 3.2-3.4 | 9.3e-3 | 0.99993 |
| same (sliced: 2 x 64 chunks) | 8192 | 7936 | 9.3 | 9.0e-3 | 0.99995 |
| same | 32768 | 32512 | 35.4 | 9.0e-3 | 0.99995 |
| both streams carry half of every K and V chunk, 4 KB pieces, depth 3 / 3 | 8192 | 7936 | 7.84 | 9.0e-3 | 0.99995 |
| same | 1024 | 768 | 1.99 | 9.5e-3 | 0.99994 |

Ablations at L 8192, first build: all 9.3 ms; DMA only (`AF_ABL=7`) 3.9 ms; without QK -1.5, without the softmax
-1.7, without PV -2.0 ms. The parts add: compute and transfer did not overlap (V had one L1 slot and only stream
B). The split-stream build: DMA only 3.43 ms, all 7.84 ms. Per key chunk and pass: 7.7 us against a 3.35 us
transfer floor (32 KB per stream per chunk) and about 5 us of compute.

Rejected: one QK call per A / B piece pair (one pass over G for 8 k blocks instead of two for 4). Under
`arbiter hold --measure`, two rounds of 3 runs at L 8192: best 7.84 ms split against 8.20 ms paired, with spikes to
14-17 ms in both arms; no gain.

L1 per core (tile 0,2, from `input_with_addresses.mlir`): stack 4 KB (af_sm's frame 2.7 KB), O 16 KB, A and B 12 KB
each, Q 8 KB, G 4 KB, state 1.1 KB: 58.5 KB. Program memory 8.8 KB. A build takes 15-30 s.

Model gates (`kernels-fa8k`: the current tree exported at an 8192-row capacity; CLI, block route):

| gate | attn_fa256 | attn_block (`OFLM_OPEN_ATTN_FA=0`, same set) |
|---|---|---|
| KL vs llama.cpp, 8 chunks x 512 (2040 positions): mean / median / p95 / max | 0.0235 / 0.0080 / 0.088 / 1.39 | 0.0235 / 0.0082 / 0.085 / 1.97 |
| same top token | 93.3 % | 92.9 % |
| perplexity (llama.cpp 7.347) | 7.4065 | 7.4073 |
| needle 4096 / 8192 | PASS / PASS | (PASS / PASS earlier) |
| prefill 4096 / 8192 tokens | 19.6 s / 38.3 s | 24.6 s / 50.7 s (profile runs, `OFLM_OPEN_PROFILE` on) |

32k (`kernels-fa262k`, 262,144-row capacity): needles at depth 0.1 / 0.5 / 0.9 PASS (prefill 169.2 / 166.2 /
164.4 s). Prefill profile (`OFLM_OPEN_PROFILE`, ms per token; earlier attn_block numbers in brackets):

| | 4k | 8k | 32k |
|---|---:|---:|---:|
| whole prefill | 4.59 (6.01) | 4.63 (6.19) | 5.18 (8.59) |
| full-attention layers | 1.08 (1.71) | 1.14 (2.16) | 1.59 (4.72) |
| linear-attention layers | 3.51 | 3.48 | 3.58 |

Long context (2026-11-26, same set, CLI needles at depth 0.5): 131,072 tokens PASS, prefill 912.9 s (6.97 ms per
token); 262,000 tokens PASS, prefill 2,456.8 s (9.38 ms per token) against 14,706 s on attn_block (6.0x).
Logs: `C:/models/qwen36moe/ctx/needle/n131072_0.5.log`, `n262000_0.5.log`.

The attention dispatch now sits in the full layers' `gemm` bucket (0.23 / 0.30 / 0.74 ms per token); the host
attention stage is gone (`attn` 0.06 ms per token: Q pack, O read-back, gate).

The attn_block arm's KL equals the earlier `q8s` result to every digit, so the A/B differs in the attention route only.

Projection (inferred from the 7.7 us per chunk and pass): 262k prefill attention about 1,300 s for the 10 layers,
against about 13,000 s on `attn_block`; 32k about 25 s against 60 s.
Parent: [qwen36moe-production-bar.md](qwen36moe-production-bar.md), Phase 4, "Long-context cause".
Spec impact: a new route for an existing requirement (OPEN-CTX-QWEN36MOE: prefill correct to 262,144); no
requirement changes. Engine and recipe change in one commit (manifest contract).

## Why

The 10 full-attention layers prefill through `attn_block` (`core.cpp:5838-5886`): per 256-token block, per KV
head, per 4,096-row chunk, the host re-tiles K and V, reads back a 2048 x 4096 f32 score matrix (32 MB), runs the
softmax and sends P back. Measured stage totals: attention 1.4 s at 4k, 60 s at 32k, 804 s at 131k, 3,316 s at
262k, plus the dispatches inside the GEMM bucket (9,943 s at 262k). About 90 % of the 4.1 h prefill at 262k.

## Starting point

`designs/dit_fa` (head dim 128, causal and GQA modes, bf16 token-major Q/K/V/O, online softmax per core with no
merge, 8.6 TFLOPS at 24 x 4608^2 x 128; README). Two groups of 16 cores; a core owns TQ = 32 query rows; K and V
broadcast per group through one memtile each, in [64 keys, 64 dims] pieces; the head dim is split into 64-wide
chunks inside the core.

L1 at head dim 256 (dit_fa's budget, scaled): A and B fifos 2 x 8 KB each (unchanged: the pieces stay 64 x 64),
output slab 32 x 256 bf16 = 16 KB, q_saved 16 KB, G 4 KB, state, stack 4 KB: about 72 KB, over the 64 KB. With
TQ = 16: output 8 KB, q 8 KB, about 56 KB. A group of 16 cores then covers 256 query rows a pass, which is exactly
one block (T = 256).

## Design

- One dispatch is one 256-token block of one layer against its window of L cached rows (as `attn_fa`): one xclbin,
  one instruction stream per 256 rows of window, one hardware context for the layer's attention.
- Q: the q projection's 16 heads x 256, after the q/k norms and the partial rotary (64 of 256 dims), pre-scaled.
  Host-side today (`block_layer_full`); keep it there for the first version and count its cost.
- K, V: read in place from the layer's KV cache rows (bf16; K at kvh x 256, V after the K heads), as `attn_fa`
  reads its cache. No host tiling.
- GQA: 2 KV heads, 8 query heads each; the two groups take one KV head each, 8 passes a dispatch.
- O: f32 or bf16 token rows [256, 16 x 256]; the sigmoid q gate and the out-projection stay where they are.
- Causal bound per row: p0 + t, p0 from the pass record (as `attn_fa`).

## Steps

1. Kernel at a fixed small shape (L = 1024, 4096): build, L1 and program-memory check, compare against fp64
   attention on random data and on captured Qwen3.6 Q/K/V (rel_fro and row cosine; gate: no worse than `attn_block`
   against the same reference).
2. Recipe: emit `attn_fa256` streams per 256 rows of window up to the export's `--max-ctx`; manifest entry next to
   `attn_block`. Engine: `block_layer_full` uses it when present (`OFLM_OPEN_ATTN_FA256=0` keeps `attn_block`).
3. Model gates at 4k and 8k: first-token logits against the `attn_block` route (KL and top token on the needle
   prompts), `--prefill-logits` positions near the block edges.
4. Timing at 4k, 8k, 32k (minutes each); then 65k and 131k; the 262k needle last.

## Risks

- The window walk at 262k is 1,024 key chunks a pass; dit_fa's longest measured is 72. Check BD and instruction
  stream size per L (one stream per 256 rows: 1,024 streams at 262k, as `attn_fa` does).
- Program memory: dit_fa's kernels at head dim 128; doubling the head-dim chunk loop may not fit 16 KB.
- Accuracy of bf16 scores over 262k keys: fp32 running sums as dit_fa; check the needle, not only random data.

## Estimate

Attention FLOPs at 262k: 10 layers x 16 heads x 262k^2 / 2 x 256 x 2 x 2 = 5.6 PFLOP. At dit_fa's 8.6 TFLOPS that
is about 650 s, against about 13,000 s now (inferred). At 32k about 10 s against 60.
