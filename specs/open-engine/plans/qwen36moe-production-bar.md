# qwen3.6-moe:35b-a3b to the production bar

Specs: `spec.md` OPEN-FAMILY-QWEN36MOE, OPEN-FAMILY-QWEN36MOE-GGUF, OPEN-QUANT-Q8, OPEN-MOE-BATCH. Bar: `AGENTS.md`
"Production bar for OFLM kernel sets". Target list: OFLM #240. Progress: [RECOMMENDED_MODELS.md](../../../RECOMMENDED_MODELS.md).
Written 2026-11-23. Branch `feat/qwen36moe-production-bar`, worktree `C:/code/open-kernels-qwen36moe`.
Machine: Ryzen AI 9 HX PRO 370 (Strix Point), Windows 11, 88 GiB RAM. Local notes and logs: `.plans/qwen36moe/`.

## Where it stands (2026-11-23, read from the code and the specs)

| Bar item | State | Evidence |
|---|---|---|
| Block prefill | Yes: layer-major, stage-major, 256-row GEMMs, token-batched experts (`mb_s*`), NPU attention (`ag_s*` / `ag_pv*`) | `core.cpp` `step_gemm_prompt`; OPEN-MOE-BATCH |
| Near optimal | **Gap.** Oct 21 profile (q4nx, old engine): 8192-token prefill 29.0 s, of which `mb_s256` 16.3 s, host DeltaNet conv + rule 3.2 s, MoE prep 1.2 s, tail 2.5 s; decode 178 ms at 8192. #240 quotes 14.4 tok/s decode, 184-196 tok/s prefill | `C:/models/q35p/out/r35b*`, F-0318, F-0321 |
| Real context | **Gap.** Model states 262,144. Engine default 4096; decode kernels bake `MAX_CTX` 4096 into the KV type and patch the window at run time (6000 rows ran once); prefill attention chunks past 4096 on the host merge. No run past 8192 on this model | `layout.py:55`, `core.cpp:5348-5398`, README:569 |
| All on NPU | Decode: yes (host only reads the router indices and patches fills). Prefill host stages: RMSNorm and residual, DeltaNet conv and delta rule, router, shared-expert gate, q/k-norm + RoPE + KV write, row softmax, attention output gate | agent map 2026-11-23 |
| Drafter | **Gap.** No rows / verify pass for this family; prompt lookup needs one; no MTP; nothing rolls back DeltaNet state | `dense.py` `ROWS_FAMILIES`, `core.cpp:6074` |
| Quantized GGUF | **Gap.** The GGUF route runs (KAT / Ornith / Nex, own pure Q4_0). Unsloth's files keep every non-expert projection at Q8_0; the view requantizes them to q4_1 (13% next-token flips, `qwen36moe.py:1319`) and requantizes Q4_K experts too (`expert_stripes_gguf`; the exact `q35_stripes` op exists but is not emitted) | OPEN-FAMILY-QWEN36MOE-GGUF |
| `phlegm run` / `serve` | KAT, Ornith, Nex on Linux (Strix Halo) via the `qwen35moe` patch bridge. Not on this machine, not on the in-tree engine | spec results 2026-10-31 |
| Benchmarks | None recorded for this model | |
| Features | Reasoning split checked on KAT. Tool calls not checked. Vision: host ViT coded for Qwen3.5/3.6 (`vit.cpp`, mmproj `qwen3vl_merger`), "not run end to end". MTP head: present in Unsloth's MTP GGUF, not read | `vision/vit.cpp`, README:596 |

### The model files (Unsloth, read from the remote GGUF headers 2026-11-23)

`unsloth/Qwen3.6-35B-A3B-MTP-GGUF` carries the MTP layer as `blk.40` (41 blocks, `nextn_predict_layers` 1); the
plain repo does not. Context 262,144. Per role, `UD-Q4_K_S` (21.39 GB with MTP):

- Experts: gate / up Q4_K (41 of 41); down Q4_K (38) and Q6_K (3).
- Attention, DeltaNet (`attn_qkv`, `attn_gate`, `ssm_out`), shared expert, MTP `eh_proj`: Q8_0.
- Head Q6_K; embedding Q8_0; norms, A, dt, conv, alpha, beta, router F32 (one BF16 router).

`UD-IQ4_NL` uses IQ3_S gate / up experts; `MXFP4_MOE` uses MXFP4 gate / up and Q5_K down. `mmproj-BF16.gguf` (0.9 GB)
holds the vision tower.

### Decode byte budget (inferred from the spec; phase 0 measures it)

Per token: 2.43 B weights (experts 1.01 B, non-expert projections 1.41 B) plus the 0.51 B-weight head at q8.
- All projections q4_1 (0.625 B a weight): 2.06 GB, about 46 ms at 45 GB/s.
- Non-expert projections at q8 (Unsloth's Q8_0 kept): 2.67 GB, about 59 ms (+30%).

## Decisions for the user

1. **Disk.** C: has 20 GB free. The Unsloth file is 21.4 GB, a kernel set about 1 GB, and llama.cpp needs the
   same file. Something must be deleted or moved first. Candidates: the deprecated q4nx containers under
   `C:/models/bench/oflm-root/models` (35B 23 GB, 27B, 9B; each hard-linked a second time, so both links must
   go), `C:/models/dit-pad` (13 GB), `C:/models/gate-root` (11 GB), `C:/models/g4e-port` (9.9 GB).
2. **GGUF and quant.** Recommendation: `UD-Q4_K_S` from the MTP repo, with Q8_0 projections kept at q8 (a
   per-role q8 GGUF view) and Q4_K experts kept exactly (`q35_stripes` / `q35_down`). It is Unsloth, it carries
   the MTP head, and nothing is requantized except the 3 Q6_K down tensors and the head. Cost: about +30% decode
   bytes. Phase 1 measures KL and decode for both this and requantized q4_1 projections; the choice follows the data.
3. **Phase 1b (open, 2026-11-23).** The Q8_0 shared expert is requantized to q4_1 today and costs the whole KL
   gap (log). Approve the host-decode shared expert (phase 1b), or accept the q4_1 shared expert at KL 0.040.

### Bar status (2026-11-23)

| item | status |
|---|---|
| block prefill | runs on the GGUF q8 view (20-26 ms a token at 512, development) |
| near optimal, profiled | not done; decode 105 ms at position 1, attention walk compute-bound at depth |
| context 262,144 | needles pass to 262,000 (2026-11-25); prefill there 41 min on attn_fa256 (2026-11-26; 4.1 h on attn_block), decode 1.12-1.19 s |
| everything on the NPU | host stages not yet costed (norms, router, DeltaNet rule and conv, prefill softmax) |
| drafter | not started; MTP acceptance measured on llama.cpp (79-95% for one draft) |
| phlegm run / serve | pass: plain, streamed, reasoning on / off, tool call round trip |
| Unsloth GGUF q4 | UD-Q4_K_S; block-route gate KL 0.0235 against the 0.019 backend spread (phase 1b, `kernels-q8s`) |
| benchmarks | not run |
| features | reasoning and tools pass; vision and MTP not started |

## Spec impact

New requirements (names provisional):
- **OPEN-GGUF-Q8-ROLES-QWEN36MOE:** a GGUF whose non-expert projections are Q8_0 packs them at q8 (`q8_perm`)
  and runs the q8 decode and prefill routes; Q4_K experts keep their codes.
- **OPEN-SPEC-MTP-QWEN36MOE:** the MTP layer drafts; a rows pass verifies; DeltaNet state rolls back on rejection.
- **OPEN-CTX-QWEN36MOE:** decode and prefill correct to 262,144 tokens.
- **OPEN-VISION-QWEN36:** the Qwen3.6 vision tower from the mmproj GGUF (on the NPU, or with a named host cost).

Modified: **OPEN-FAMILY-QWEN36MOE** gets a `## Production bar` audit table, as Granite and gpt-oss have.

## Phases

Every measurement runs under `dibs run --measure` (R-0006). Record machine, OS, XRT, driver, NPU firmware.
Quality gates compare against llama.cpp on the same GGUF, at the level of llama.cpp's own CPU-vs-Vulkan spread
(the gate used for gpt-oss and Granite).

### Phase 0: baseline (no download needed)

1. Build the in-tree CLI (done 2026-11-23: `build.cmd`, all unit tests pass).
2. Export the kernel set from `recipes/specs/qwen36-35b-a3b.json` into `C:/models/qwen36moe/kernels`.
3. Run the local q4nx container (same `spec_hash`) for a profile only: decode at 0 / 1k / 8k / 32k; prefill at
   1k / 8k / 32k with `OFLM_OPEN_DISPATCH_LOG=1`. Name each stage against the byte budget.

### Phase 1: the Unsloth GGUF (bar item: quantized GGUF)

1. Download `UD-Q4_K_S` (MTP repo) and `mmproj-BF16.gguf`; record sha256.
2. Emit `q35_stripes` / `q35_down` for Q4_K experts (exact codes) instead of the requantizing `*_gguf` ops.
3. A per-role q8 view: `quant_for_gguf` for qwen36moe (from GGUF tensor types, `quant_map_from_gguf_types`),
   `gguf_view` accepting q8 roles, a `q35_q8` op (port from `saluki-runtime.patch`) writing `q8_perm` half-tiles.
4. Gates: `qwen35moe_gguf_check.py` exact bounds; teacher-forced KL and argmax against llama.cpp on the same file
   (491-token prompt and an 8k prompt); greedy tokens.

### Phase 1b: an exact shared expert (approved by decision 2, implemented 2026-11-24)

Decision 2 was wrong on one point: the recipe also requantizes the Q8_0 shared expert (`_check` refuses a q8
`shared` role, because the decode cores run it in the routed experts' slot loop). The 8-chunk gate (log below)
puts the whole KL gap to llama.cpp on that requantization. The lx main core has 80 B of program memory free, so a
q8 GEMV entry for the shared slot does not fit.

Proposal (a `shared` role at q8 means this; a q4_1 `shared` role is unchanged):
1. Decode: lx1 / ax1 run the routed slots only (`moe_body` / `moe_sequence` `nx = NE`, as `mx` already does; one
   more `accfin(-1)` call site, which must fit in the 80 B). A new host step `shared_q8` after `moeroute2` reads
   `xm` and the residual from `act[l]`, runs the shared expert from the GGUF's Q8_0 codes (f32 math, AVX2,
   sigmoid gate), adds it to the residual and syncs it back before lx1 starts. Measured host cost on this box:
   71-82 us a layer at 4-12 threads, about 3 ms a token (scratchpad bench, 40 layers of real-size Q8_0 weights).
   lx1 / ax1 stream 1.97 MB less a layer, so the net decode change is expected near zero (inferred; measure it).
2. Block prefill: the shared expert's GEMMs read the Q8_0 source directly (`bf16_gemm` without `via: q4_1`, and
   the `lean` variant's exact hi / lo split), not the pool's requantized q4_1 copy.
3. The pool keeps its shared q4_1 slots (unused by decode), so the layout and `moeroute2` do not change.
4. Writer and reader in one commit: recipe (`_check`, `gemm_route` shared weights, `layer_x` `nx`, the host step
   in the program), manifest parser, `Core::host_step`, a host unit test against the f32 reference.
5. Gate: rerun the 8-chunk KL. Expected (inferred): about 0.026-0.028 mean, the NPU's arithmetic level against
   the same weights, from 0.040.

Host stage named for item 4: `shared_q8`, about 3 ms a token.

### Phase 2: `phlegm run` / `serve` (item 6, features)

1. Build phlegm's bridge from this worktree (`benchmarks/reference/build.ps1 -ReferenceRoot`), family `qwen36moe`.
2. `phlegm run`, `serve` plain and streamed; reasoning on / off; a tool-call round trip.

### Phase 3: context to 262,144 (items 3, 4)

1. KV and position table at 262,144 (512 MiB a full layer, 10 layers). Decode `MAX_CTX` and window patching past
   4096; prefill attention chunks past 4096.
2. Needles at 32k, 128k, 262k (depths 10 / 50 / 90 %), against llama.cpp where it fits in time.
3. Decode and prefill rate against depth.

### Phase 4: near optimal, host stages (items 2, 4)

1. Decode profile against the byte floor; prefill profile by stage.
2. Move a host stage to the NPU when it costs more than about 5% of prefill (DeltaNet rule and conv first).
   Name every remaining host stage with its cost.

**Prefill profile by stage (2026-11-25, measured).** `kernels-q8s`, block route T = 256, layer-major, CLI,
`OFLM_OPEN_PROFILE`, needle prompts (`.plans/qwen36moe/prefill_prof.sh` + `prefill_prof.py`, rows in
`C:/models/qwen36moe/ctx/prof/`). ms a prompt token, summed over the layers of each type:

| stage | 4,096 | 8,192 | 32,768 |
|---|---|---|---|
| whole prefill | 6.01 (166 tok/s) | 6.19 (162) | 8.59 (116) |
| routed MoE, 40 layers (`per_token`; of it `moe_run`) | 2.82 (2.43) | 2.62 (2.24) | 2.48 (2.07) |
| host, 30 DeltaNet layers | 1.81 | 1.83 | 1.83 |
| of it DeltaNet conv + delta rule | 0.98 | 0.99 | 0.98 |
| of it tail, MoE prep + read, sync, shared | 0.63 | 0.64 | 0.65 |
| GEMMs, 30 DeltaNet layers | 0.19 | 0.14 | 0.10 |
| 10 full-attention layers (wall) | 1.71 | 2.16 | 4.72 |
| of it attention (NPU, `attn`) | 0.33 | 0.59 | 1.84 |
| of it host | 0.53 | 0.81 | 2.12 |

- Short prompts: the routed MoE is 47 % of prefill and the host 39 %. The DeltaNet conv + rule alone is 16 %, over
  the 5 % rule above. Neither grows with length.
- The routed MoE is at its own DMA floor already: `moe_block` runs per 256-token block, every expert's tokens cut
  into visits of `MB_NT` = 8, and each visit streams the expert's 1.97 MB (F-0305: `mb_s256` 14.21 ms against an
  11.55 ms floor). A block has Poisson(8) tokens an expert, so the weight bytes a token are fixed by `MB_NT`, not by
  T. To lower them: batch the MoE over the whole prompt (the schedule is layer-major already, but `moe_block` is
  still called a block at a time) and raise `MB_NT` to 16 (builds and passes; at T = 256 it bought 4 % because few
  experts had more than 8 tokens; the kernel turned core-bound there, `recipes/qwen36moe.py` `MB_NT` note).
- Attention grows linearly a token as expected (8x the tokens, 5.6x the ms a token). The host part of the
  full-attention layers grows too (0.53 -> 2.12), not yet split by stage.
- Long context: at 32k the whole is 8.6 ms a token; 131k ran at 15.1 and 262k at 56.1 (needle logs). So the
  superlinear part starts past 32k and grows sharply past 131k (NPU part 10.9x for 2x the tokens). Not found yet.

**Long-context cause (2026-11-25, from the code and the existing logs; no new run).** The 10 full-attention layers
use the `attn_block` route (`core.cpp:5838-5886`, manifest `attn_block`: m 2048, hd 256, l_max 4096; no `attn_fa`
in this set: that kernel is built for hd 64). Per 256-token block and per KV head, for every 4,096-row chunk of the
window: the host re-tiles the chunk's K and V (`tile_rows_as_bt` / `tile_rows_as_b`), the NPU runs Q.K^T, the host
reads back the whole 2048 x 4096 f32 score matrix (32 MB), runs the softmax, sends P back, and the NPU runs P.V.
At 262k that is about 655,000 chunk visits over 10 layers and about 21 TB of score read-back (inferred). The stage
totals agree: `attn` 1.4 s at 4k, 60 s at 32k, 804 s at 131k, 3,316 s at 262k, and the GEMM bucket, which holds
these dispatches, grows from 66 s (32k) to 910 s (131k) and 9,943 s (262k) while the projections alone are about
0.17 ms a token. So about 90 % of the 262k prefill is this attention route. The fix is a fused attention kernel on
the NPU for hd 256 (online softmax on the array, K/V read in place from the cache, no score round trip), the shape
`attn_fa` / `gptoss_attention` have for hd 64.

**Next (needs review before the build):**
1. Long context: an `attn_fa`-style fused attention kernel for hd 256 (Qwen3.6's gated attention: the q gate stays
   where it is). The largest single item for the bar: it decides the 131k and 262k prefill. The 65,536-token
   profile is no longer needed to find the cause; run it after the kernel, to measure it.
2. Routed MoE over the whole prompt with `MB_NT` = 16: `moe_block` takes every block of the layer at once (a
   driver change; the per-token scatter order must stay the same for exact logits), then an `MB_NT` 16 set A/B at
   4k and 8k. Halves the expert visits; the gain depends on how core-bound a 16-token visit is (estimated 0.5-0.8
   ms a token, 10-13 % of short prefill; not measured).
3. DeltaNet conv + delta rule on the NPU (block prefill): a new chunked kernel; the largest host stage (16 %), and
   the bar's "everything on the NPU" item. Days of work; start only after 1 and 2.

**Step 2 result (2026-11-25 17:41, measured).** `moe_block` already took the whole prompt (`block_layer_moe(l, TOT,
N, ...)`; its "three passes serve a block" comment is stale), so only `MB_NT` changed: `kernels-nt16` =
`kernels-q8s` with the six `mb_s*` streams at 16 tokens a visit (`QWEN36MOE_PROBE_MB_NT=16`, `export_nt16.cmd`).
`nt_ab.sh`, A B A B, first-token logits bit-identical at both lengths:

| prompt | q8s (s) | nt16 (s) | expert run, last pair (s) |
|---|---|---|---|
| 4,096 | 20.37, 19.79 | 19.53, 24.24 | 8.76 / 10.28 |
| 8,192 | 44.01, 44.35 | 42.86, 43.32 | 16.88 / 16.75 |

Half the expert visits, the same expert time: the kernel is arithmetic-bound, not weight-bound, at 8 tokens a visit
already. Rejected (`MB_NT` stays 8). Why, from the kernel's own measurements (`moe_batch.h:68-82`): at 8 tokens the
dispatch is within 15 % of its weight stream (16 slots 0.895 ms, the stream alone 0.779), so the stream is the
floor. A 16-token visit halves the stream a token, but it repeats the q4 unpacking for each 8-token sub-tile, so the
core work a token stays the same and the core becomes the limit. The lever the kernel names: the native
`mmul<4, 8, 8>` (aie2p's bf16 shape; `mmul<8, 8, 8>` is emulated, eight `vextbcst` a mac) plus one unpacked weight
fed to both sub-tiles (`MB_SHARE_NG`, which spilled an accumulator when tried). If that brings a 16-token visit's
core under its halved stream, the expert stage drops toward half: about 1 ms a token, 15-20 % of short prefill
(inferred). Kernel work with a known register-file risk.

### Phase 5: drafter (item 5)

1. A rows pass for `qwen36moe` (R rows through `lx*` / `ax*`; experts as the union of the rows' picks), with a
   DeltaNet state snapshot per row so a rejection rolls back.
2. The MTP layer (`blk.40`: `eh_proj`, norms, one full-attention MoE layer, shared head) as the drafter.
3. Prompt lookup on the same rows pass. Ship the better one, or reject both with measurements.

**Cost estimate (2026-11-24, for the go / no-go; needs review before the build).**

Measured on `kernels-q8s` at position 1 (`.plans/qwen36moe/mtp_cost.sh`; the machine was busy, so the whole
step read 160 ms against the usual 107-110 ms, and only the minimums are used):

| dispatch | min ms | MB |
|---|---|---|
| `lx0` (norm, q8 projections, conv, DeltaNet, router) | 1.98 | 48.6 |
| `lx1` (8 routed experts) | 0.78 | 15.3 |
| `ax0` | 1.41 | 33.7 |
| `ax1` | 0.76 | 15.3 |
| `mb_s8` / `mb_s16` / `mb_s32` (batched experts, up to 8 tokens a slot) | 0.60 / 1.36 / 2.60 | |
| lm_head (clean runs) | 12.5 | 516 |

`--bench-kernel lx0` and `--bench-kernel lx1` alone both hang at their second dispatch: each stream needs the
other to run between its dispatches (the same finding as OPEN-ROUTE-REDO). A verify image needs its own pair.

The routed experts are about 25 % of a layer. A two-row verify streams `lx0` / `ax0` once for both rows; the expert
union of two tokens is about 14 slots (`mb_s16` 1.36 ms against 0.78). Estimate, not a measurement:

| | per cycle | tokens a cycle | per token |
|---|---|---|---|
| decode now | 107-117 ms | 1 | 107-117 ms |
| MTP draft (one layer + head) + 2-row verify | about 15 + 150 ms | 1.8-1.9 (llama.cpp acceptance 79-95 % for one draft) | about 87-92 ms |

So about 1.2-1.3x, if the verify pass reaches the estimate. That clears the "generally want one" bar, so the
recommendation is to build it, not to reject it.

**Decision (user, 2026-11-24):** build MTP, combined with prompt lookup on the same verify pass.

**Design (2026-11-24, from a read of the precedents; nothing built yet).**
- Base: fork `designs/verify/vux.py` as `designs/verify/vmx.py` (own contexts `vlx` / `vax` sharing decode's pools,
  consts, KV and state; n at run time; `stateslot` rollback; `attnposv` rows; 2048-wide K slices). The layer body
  comes from `designs/layer_x/xcommon.py` (`moe_body`, `dn_body`), the two-row q8 tile from
  `designs/gptoss/gx_rows_q8.h`, the two-row head from `lm_head_q8` `LMHEAD_ROWS`. Not `dxl` (dense, q4, no state
  rollback). Two verify contexts, as decode has two (`merged_image()` is off for q8 roles).
- DeltaNet: `vdn.h` (S read once, each row's S' in place in decode's op order); R + 1 state slots a linear layer
  (about 6 MB x 30 at R = 2); decode `lx0` and block prefill address the committed slot through `stateslot`.
- Attention: one shared window, every row drained before the fill; `attnv_apply` extended from `rb_win` to `rb`.
- Router per row on the helper core (not the host: other arithmetic flips near ties). Routed experts over the union:
  16 placeholder slots (`moe2_table` takes topk 16), a host `route_union` writing R x 16 weights (0 where a row did
  not pick the expert), one accumulator per row. Each row adds its experts in a fixed order: sort by expert id in
  decode `route()` too, then re-run the KL gate. Not `mb_sN` (its own context, a host gather, other arithmetic).
- Shared expert on the host, both rows in one pass over the q8 codes, overlapped with the expert dispatch.
- Engine: port `verify` / `accept` / `verify_logits` / `verify_slots` from openflowlm-next (not in this repo);
  `SpecMethod::mtp` beside lookup in `spec_lookup.hpp`, priced per cycle; MTP KV kept current every cycle.
- MTP (`blk.40`; the GGUF view skips it today): a manifest `mtp` section packs it as one more full-attention layer
  with its own pool, consts, act and KV. Draft: host embedding row, enorm, hnorm, `eh_proj`; layer 40's `ax0`,
  route, shared expert, `ax1`; `ln` with `shared_head_norm`, head, argmax.

**Milestone 1 result (2026-11-25, measured; timing-only probe builds, output wrong).** `kernels-q8s` against probe
sets, `--bench-kernel mx_linear:80:0` and `--bench-decode 30`, two alternating rounds, min ms, NPU free of other
clients. Today the machine is about 35 % faster than in the table above (`lx0` 1.30 against 1.98; whole step 101-102 ms
submit-ahead), so only today's numbers are compared.

| probe (per layer) | round 1 | round 2 | against one row |
|---|---|---|---|
| expert block, 8 slots, one row (`q8s`, `probe8`) | 0.490 | 0.491 / 0.494 | |
| expert block, 16 slots, one row (`probe16`) | 0.871 | 0.867 | +0.38 |
| expert block, 8 slots, two rows in the mmul A rows (`probe8r2`) | 1.792 | 1.800 | +1.31 (3.7x) |
| expert block, 16 slots, two rows (`probe16r2`) | 3.543 | 3.510 | +3.03 |
| `lx0`, real / DeltaNet arithmetic null (`nd`) / null + two-row q8 GEMVs (`q8r2nd`) | 1.300 / 1.286 / 1.293 | 1.299 / 1.313 / 1.286 | +0 |
| `ax0`, the same three | 0.935 / 0.919 / 0.919 | 0.939 / 0.944 / 0.928 | +0 |
| whole step (submit-ahead), the same three | 102.0 / 100.6 / 102.4 | 101.3 / - / - | +0 |

- The front is stream-bound: the q8 projections at two rows and the DeltaNet arithmetic cost nothing measurable.
- Two rows inside one expert tile are compute-bound (3.7x at 8 slots; the up | gate TU is built `-Oz` on this
  spec, which may add to it, not measured). Two-row tiles are rejected for the experts, as for GPT-OSS.
- Changed design: each routed slot carries one row (16 slots = 2 rows x 8 experts; a shared expert is streamed
  twice). That is the measured `probe16` one-row cost, +0.38 ms a layer. The slot picks its row's x table and
  output accumulator; the host writes the row per slot with the route patch.

Estimate from these (inferred): step 102 + experts 15.2 (40 x 0.38) + DeltaNet second row 0.7 (the null-DeltaNet
difference) + one more state slot written 2.8 (3.3 MB x 30 at 36 GB/s) + head second row 0.7 (+6 %, GPT-OSS) + second
attention walk 0.3 + host shared expert second row up to 2 = about 124 ms, 1.21 steps. Under the 160 ms kill line.
MTP draft about 13 ms (`ax0` + `ax1` + head + `eh_proj`). At 1.79-1.95 tokens a cycle: 70-77 ms a token against 102,
1.33-1.45x. Prompt lookup has no draft cost. **Go for milestone 2.**

Follow-up probe (2026-11-25, `kernels-q8x2`: every q8 half tile called a second time on the same chunk, real
DeltaNet): whole step 104.1 / 105.5 ms against 104.9 (`q8s`; its second round lost an `lx1` in the bench's own walk,
which has no recovery); `lx0` 1.386 / 1.315 against 1.314. No measurable cost. Program memory: the `lx` main core is
full (last symbol ends at 16,272 B; the core program itself is 5,504 B), `ax` has 2,176 B free.

**Design, revised by the probes (2026-11-25).** The verify images reuse decode's one-row arithmetic, so each row is
bit-identical to a decode step by construction and the routing order and KL gate do not change:
- Front: each weight chunk is passed to the one-row q8 half tile once per row, with that row's x table (a second
  9 KB table in L1) and its own y element. DeltaNet, the conv glue, the post core and the norm + router run once a
  row, in order; the host sequence addresses row 0's S' to a slot and row 1 reads it (the rollback slot).
- Experts: decode's `moe_body` in a run-time loop over the rows, 8 slots each, with that row's router output and
  residual; moeroute2 patches 16 slots. Cost as `probe16`.
- Program memory: the `lx` core has no room for unrolled copies; every row loop is a run-time loop (`range_`), and
  the per-row buffers are chosen by the host sequence (DMA addresses), not by core code.

**Milestones.**
1. Done (above). Timing probe that confirms or kills the estimate, before any engine work: probe builds of the expert half at 14
   and 16 slots with two rows, the `lx` front with two-row q8 GEMVs and DeltaNet twice, the `ax` front with two
   walks, `lm` at two rows. Kill if the pass adds up to more than about 160 ms.
2. Done 2026-11-25 (below). Compile-only `vlx` / `vax` main cores against 16,384 B of program memory (the decode `lx` core has 80 B free).
   If they need more images, stop: each extra context switch costs about 0.8 ms a layer.
3. Done 2026-11-25 (below). `vlx` / `vax` and the recipe; per-layer bit identity with two different rows.
4. Engine port; sorted routing plus the KL re-gate; `--verify-check` PASS; `--bench-verify`.
5. Prompt lookup on the verify pass (`--spec`, SPEC IDENTICAL), measured.
6. MTP packing and host pieces; drafts against llama.cpp's; acceptance.
7. Per-cycle choice of MTP or lookup; `phlegm run` / `serve`; then R = 3.

**Milestone 2 result (2026-11-25).** `designs/layer_x/vlx.py`, `vax.py` and the shared `vrows.py` compile for
`kernels-q8s`' spec, both parts each: `vlx` fullest core 15,616 of 16,384 B (768 B free), `vax` 14,304 (2,080 free).
What it took on the `lx` main core: the og elements acquired one at a time (an `acquire(2)` gives the x fifo a third
4 KB buffer, and the second table left the core 1,408 B over its 64 KB data memory); the q8 entry's row split as a
constant (`gen_kernels.q8_gy`: drops a 144 B software divide, also from decode's cores); the band loops rolled
(`X.ROLLED`, the 4B's fix: the core program 5,904 -> 5,024 B). `-Oz` on every main-core unit made it 1 KB larger.
The rolled band loops are not yet timed. Placeholders for the recipe: row r's routed slot j compiles as expert
r * 8 + j (`moe_sequence(slot0=)`), so `moe2_table` at topk 16 tells the rows apart; `vax` row r drains its cache row
to placeholder row r + 1 and reads ptab row r + 1, the multi-row form `attn_table` already parses; `vlx` row r reads
state slot r and writes slot r + 1 (`slot_table`).

**Milestone 3 result (2026-11-25, measured, `kernels-v` = `kernels-q8s` + the four verify images, `OFLM_OPEN_VERIFY=1`,
`--verify-check 6`, 32-token prompt):** every row's logits bit-identical to a plain step from the same checkpoint (6 of 6
cycles, both rows on even cycles, row 0 only on odd ones), and the step after `verify_commit` too. Found on the way:
- `vax.py` hung its first dispatch: four weight transfers a channel before `xn` made the host's DMA throttle wait on q's
  weights, which wait on `xn` (decode's `full_sequence` issues three). Fixed.
- All 40 layers matched a serial plain step, but the logits did not: served decode adds the host shared expert after
  the routed MoE (`shared_overlap_`), the serial path inside it, so the two orders round differently. The verify pass
  now does what served decode does (host shared expert during `vlx1` / `vax1`, added after).

Cost at position 32-45 (min of 6): verify pass 165-171 ms against a 101-105 ms step; part0 (`vlx0` / `vax0`) 99-101,
part1 (MoE, both rows) 41-42, host shared expert 19-21 (overlapped), two heads 25-26; `verify_commit` 5.8-6.2 ms (a
device copy of 30 states). So about 1.66 steps a cycle against the 1.21 estimated. Next, in order: the two-row head
(`LMHEAD_ROWS=2`, about -12 ms); row 1's glue during row 0's DeltaNet and row 0's post during row 1's (the front is
about +0.8 ms a layer over decode's, not +0 as the probes suggested: the probes had no per-row serialization);
`stateslot` re-pointing for the commit (-6 ms).

**After milestone 3 (2026-11-25, measured, same check, every row still bit-identical):**
- The two-row head (`lmv`: `lm_head_q8` at `LMHEAD_ROWS=2`, its tile now taking bf16 scales as the one-row one does; one
  norm per row into `vhn@r`): heads 25.9 -> 12.5-13.9 ms. Row views (`name@r`) take their size from the manifest:
  an XRT buffer can be larger than asked, which put row 1's normed input in the wrong place.
- Row 0's glue during the z projection (decode's `finish_oldest`): `vlx0` 75 -> 69-71 ms over 30 layers. Interleaving
  row 1's glue with row 0's DeltaNet and row 0's post with row 1's (`vrows.interleave`) changed nothing measurable.
- Per kernel over a pass (`OFLM_VERIFY_PROFILE`): `vlx0` 69-71, `vax0` 21.5-22, `vlx1` 30-31, `vax1` 10 ms; the
  pass 145-149 ms against a 101-105 ms step (1.43 steps), plus a 5 ms commit.
- phlegm: the reference bridge runs a recurrent model's `verify_logits` through `step_verify` (width `verify_rows()`,
  accept = `verify_commit`), so `phlegm run --speculative lookup` works on this model; with `OFLM_OPEN_VERIFY=1` and
  `kernels-v`, greedy, its output is byte-identical to `--speculative off` on two code-rewrite prompts.
  Correction (2026-11-25 14:40): that result proves nothing. phlegm removes every `OFLM_*` variable from the bridge's
  environment (`phlegm-core/src/bench/reference.rs`, `Process::spawn`), so the bridge loaded no verify pass, reported
  `verify_width` 0, and phlegm ran plain decode in every mode. `phlegm serve` (`spec_serve.sh`, rewrite / explain
  prompts, 115 / 256 tokens, greedy) confirmed it: off 8.85 / 8.88 tokens/s, lookup 8.83 / 8.65, mtp 8.68 / 8.66,
  outputs identical. Fix: `CoreConfig::verify` / `mtp` load each part when the kernel set and GGUF have it, and the
  bridge sets both. The environment switches still work for the CLI.
- phlegm now sends its `--speculative` mode in the bridge's `init` (`NpuOptions::speculative`; the release bench sends
  `off`); the bridge loads the verify pass for lookup / mtp / auto and the MTP layer for mtp / auto. The bridge's
  context cap (131,072) is gone: the engine's allocations decide.
- `phlegm serve` (2026-11-25 16:06, `arbiter hold --measure`, `kernels-v`, `--max-ctx 4096`, greedy, thinking off,
  `spec_serve.sh`; rewrite prompt 160 tokens in / 115 out, explain 29 in / 256 out; outputs byte-identical to off):

  | mode | rewrite tokens/s | explain tokens/s | bridge counters (rewrite; explain) |
  |---|---|---|---|
  | off | 7.74 | 7.87 | |
  | lookup | 9.70 (1.25x) | 8.63 (1.10x) | 53 passes, 44 both rows, 18 plain; 23 passes, 7 both rows, 225 plain |
  | mtp | 10.87 (1.40x) | 9.94 (1.26x) | 58 passes, 57 both rows; 118 passes, 100 both rows, 37 plain |

  MTP cycle (rewrite): pass 154 ms, draft 15.2, commit + MTP catch-up row 9.7, so 179 ms for 1.98 tokens. Off varies
  between runs more than the speculative modes: 7.65-8.88 tokens/s over five runs today (113-131 ms a token, against
  the CLI's 110), cause not found; mtp gave 10.76-11.32 in the three runs without CPU load from other sessions, 7.82
  in one with it (the verify pass runs the host shared expert for two rows).
- phlegm's `SpecPolicy` ran two plain cycles after every missed draft, right for lookup, wrong for MTP at 85 %
  acceptance (the 37 plain cycles above). `Drafter::policy`: MTP now skips them while its running acceptance stays at
  0.5 or more (`SpecPolicy::for_model_drafts`; a drafted cycle costs about 1.5 plain steps).
  Measured (16:25, `arbiter hold --measure`, same prompts, outputs byte-identical): off 8.91 / 9.06 tokens/s, mtp
  11.08 / 10.23 (1.24x / 1.13x); explain now 139 passes, 113 both rows (81 %), 3 plain (was 37), bridge time
  24.9 -> 24.3 s; mtp explain 9.94 -> 10.23 tokens/s across the two runs.
- The commit re-points instead of copying (2026-11-25 18:00, measured). A linear layer's state is a view
  (`state_[l]`, an XRT sub-buffer) of a 16-slot ring (`state_ring_`, +0.87 GB over 30 layers); `verify_commit` moves
  the view n slots and copies the committed slot back to slot 0 only at the ring's end (about every 7 cycles). No
  kernel or manifest change: every dispatch binds `state` through `buffer()`. `--verify-check 20` PASS (every row
  and the step after each commit bit-identical across two ring wraps); `--spec-mtp 128`: commit + catch-up row
  9.7 -> 3.7 ms a cycle, 92.0 -> 87.3 ms a token against plain 108.8 (1.20x -> 1.25x), SPEC IDENTICAL.
- Per-cycle choice of MTP or lookup (milestone 7), decided from the 16:06 counters: not worth it at verify width 2.
  A width-2 cycle drafts one token, so the choice only picks which drafter supplies it. MTP took 57 of 58 drafts on
  rewrite (98 %) against lookup's 44 of 53 (83 %), and 85 % on explain where lookup drafted in 23 of 248 cycles. A
  lookup draft saves only the 12.5 ms MTP head (about 7 % of a 179 ms cycle), so swapping in a lookup token loses
  whenever its acceptance is more than about 7 points under MTP's. Ship MTP (`auto` uses it when the bridge offers
  it). Combining the two pays at R = 3 (an MTP token, then a lookup continuation), which needs three-row `vlx` /
  `vax` images; the main cores have 768 B (`vlx`) and 2,080 B (`vax`) of program memory left.
  Gap found on the way: a pass with a lookup draft and the MTP layer loaded does not write MTP cache row
  `start - 1`, so `auto` falling back to lookup would leave holes in the MTP cache. Not reachable today (phlegm's
  `auto` takes MTP whenever it is offered).

**MTP drafter design (2026-11-25, from the GGUF; nothing built).** `blk.40` holds exactly a full-attention layer's
tensors (`attn_q|k|v|output`, the two norms, routed and shared experts, router) plus `nextn.eh_proj` [2 x hidden ->
hidden], `nextn.enorm`, `nextn.hnorm`, `nextn.shared_head_norm`; `qwen35moe.nextn_predict_layers` = 1, `block_count`
41. So the MTP layer packs with the full-attention layer type's own ops at layer index 40 (its own pool, consts and
KV) and runs on decode's `ax0` / `ax1` (one row) or the verify pass's `vax0` / `vax1` (two rows). A cycle at committed
position p, seed s (emitted, not yet in the cache) and h (the main model's pre-norm output for the row that gave s):
1. draft d = head(shared_head_norm(MTP(eh_proj[enorm(emb s) | hnorm(h)]))) at MTP position p;
2. verify rows [s, d] at p, p + 1, giving each row's token and pre-norm output;
3. d accepted: commit 2, and the MTP layer runs two rows next cycle (p + 1 catches its cache up with (h_p, d), p + 2
   drafts); rejected: commit 1, one row.
Host parts: enorm, hnorm, the 8 MB `eh_proj` GEMV. Estimated cycle: verify 147 + MTP layer 2-3 + head 12.5 + host 1-2
= about 164 ms for 1.79-1.95 tokens (llama.cpp acceptance 79-95 %): 84-92 ms a token against 102, 1.11-1.21x.

**MTP drafter result (2026-11-25, measured, `OFLM_OPEN_MTP=1`).** `blk.40` packs with the full-attention type's ops
and runs on decode's `ax0` / `ax1`; the input is llama.cpp's (`graph_mtp`): `eh_proj([enorm(emb t) | hnorm(output_norm(h))])`,
h the main model's output at the previous position, then `shared_head_norm` and the shared head.
- `--mtp-accept 64` (32-token prompt, every prompt position through the MTP layer first, greedy): 58 of 63 drafts
  match the next greedy token (92.1 %); MTP step 12.7 ms mean (the layer about 2.5 ms, the head 12.5 when wanted).
- `--spec-mtp 128` (`kernels-v`, same prompt): MTP draft + verify pass 92.0 ms a token against plain 110.2 (1.20x),
  89.6 % of 67 cycles took both rows, tokens identical to plain greedy (SPEC IDENTICAL). Per cycle: draft 15.7,
  verify 149.0, commit + the MTP cache catch-up row 9.7 ms.
- The MTP cache over the prompt matters little (`--spec-mtp 128`, same run, 2026-11-25 12:53): a 1,724-token
  code-explanation prompt 73.0 % of cycles took both rows warmed, 68.4 % with the prompt rows left zero (1.08x / 1.05x
  against plain 117 ms; the verify pass is 162 ms at that depth); the 32-token prompt 89.6 % either way (1.21x).
  Acceptance depends on the text more than on the warm-up.
- Not yet: the MTP cache over a prompt prefilled by the block route (the run above warmed it one plain step a token);
  `stateslot` instead of the 5 ms commit copy; the catch-up row inside the next draft (two rows on `vax`).

Risks: program memory (the gating one); L1 for the two-row tables; the expert union's compute (it was not hidden
for GPT-OSS); shim buffer descriptors (one task group a row for the expert stream).

### Phase 6: vision (item 9)

1. Run the host ViT from `mmproj-BF16.gguf` end to end through `phlegm serve` (image QA).
2. Time it; move its GEMMs to the NPU if it costs more than the text prefill of the same image.

### Phase 7: benchmarks and audit

1. `phlegm bench --development` sweep (prefill and decode against context), machine and stack named.
2. `## Production bar` table in OPEN-FAMILY-QWEN36MOE; RECOMMENDED_MODELS.md row; engine-changes.json.

## OFLM side (not open-kernels work)

Registry entry in `src/model_list.json` (Unsloth `UD-Q4_K_S` MTP GGUF, `qwen35moe`, 262,144, mmproj and MTP
drafter); `oflm add` / `serve` / `bench`; benchmark page; README row; #240 checkbox.

## Log

- 2026-11-23: plan written. Worktree created from `origin/main` `eb72261`. CLI and unit tests built.
- 2026-11-23, decisions (user): the user frees disk space; ship Unsloth `UD-Q4_K_S` (MTP repo, revision
  `5bc3e238`) with Q8_0 projections kept at q8; NPU runs wait while the Inkling agent uses the NPU.
- 2026-11-23, downloads: `C:/models/qwen36moe/Qwen3.6-35B-A3B-UD-Q4_K_S/` holds `model.gguf` (hard link to
  `gguf/Qwen3.6-35B-A3B-UD-Q4_K_S.gguf`), `mmproj-BF16.gguf`, and Qwen's `config.json`, tokenizer and templates
  (`Qwen/Qwen3.6-35B-A3B` revision `345d467b`).
- 2026-11-23, phase 0 (development, the q4nx container, Inkling bridges on the NPU at the same time): 1024-token
  prefill 10.9 s, 8192-token prefill 73.9 s (`mb_s256` 21.8 s, host DeltaNet conv + rule 9.3 s, host attention
  softmax / merge 5.6 s). The first decode dispatch `lx1` timed out (ERT state 8) on the old and the new kernel
  sets, with and without submit-ahead. Not reproduced yet with the NPU quiet; R-0010 names another NPU client as a
  cause (F-0185). Logs `C:/models/qwen36moe/prof/`.
- 2026-11-23, phase 1 code: in-tree engine ops `q35_q8` (Q8_0 codes kept, scale RNE to bf16, in `q8_perm`
  half-tiles), `q35_bf16` (the bf16 GEMM weight from GGUF), `q35_mat` `split`; `op_region` accepts `q35_mat`
  (without it the in-tree engine refuses the GGUF block route). Recipe: `quant_for_gguf` (q8 for a Q8_ROLES role
  whose tensors are not Q4_0 / Q4_1 / Q4_K), `gguf_view` with q8 roles and route weights, experts through
  `q35_stripes` / `q35_down` (Q4_K codes kept). The shared expert stays q4_1 (requantized from Q8_0): this
  family has no q8 shared-expert path. Tests: `gguf_pack_test` q35_q8 layout (0 differ), recipe tests, fixture
  regenerated (expert op names only).
- 2026-11-23, q8 kernel set exported: `C:/models/qwen36moe/kernels-q8` (spec quant attn / linear / linear_out q8;
  view ops `q35_q8` x8, `q35_mat` x6, `q35_stripes`, `q35_down`; route weights `q35_bf16`; kernels `gemmb_*` for the
  bf16 route and `gemm_n*` for the split `lean` variant).
- 2026-11-23, rows-pass survey (for phase 5): the `ux.py` main cores are at 16256 of 16384 B of program memory, so
  L-row work cannot go into the decode image as it stands. The dense `dxl` rows pass (K2, Granite) refuses `rb > 1`;
  `seek()` is the only rollback, so a DeltaNet state would advance with no way back. Prior art: the Qwen3.5 verify
  design (`designs/verify/vux.py`, width + 1 state slots a linear layer, OPEN-DECODE-SPECULATIVE-VERIFY, 4 positions
  at 1.59 steps) and the host `DeltaTape` / `deltanet_rollback` (unit-tested, unused). `plans/spec-decode.md`
  measured an expert union of 34-36 a layer at 8 tokens on this model. No MTP acceptance has been measured.
- 2026-11-23, llama.cpp references (build `bb4caa75`, Vulkan, `C:/models/qwen36moe/llama-vk`; `C:/models/qwen36moe/kl/`):
  - KL base: `llama-perplexity` on 40 Paul Graham essays, `-c 512 --chunks 8`, PPL 7.3515 +/- 0.411
    (`base_vk_c512.kld`). Control, llama.cpp CPU against that Vulkan base: KL mean 0.0190, median 0.0078,
    p99 0.204, same top token 94.3%. This is the bar the NPU must meet (the gpt-oss / Granite gate).
  - MTP acceptance (`llama-server --spec-type draft-mtp`, greedy, thinking off, `mtp_accept.json`): one draft
    token accepted 95% (code), 79% (prose), 79% (edit); with two drafts 2.76 / 2.18 / 2.1 tokens a verify cycle;
    with three 3.3 / 2.34 / 1.9. So a 2-row verify at under about 1.4 decode steps pays on every prompt type.
- 2026-11-23, the first gate run refused the GGUF block route at load: the MoE kinds' host stages (norms, router,
  DeltaNet alpha / beta / A / dt bias / conv / norm, q / k norms) read their tensors only from a q4nx container.
  Fix: `pools::const_values` gives a put or q35 const's values with the decode pack's own row maps (V-head order,
  transposes); `core.cpp` reads the GGUF consts through it by the GGUF names. The old GGUF conv transpose did not
  apply the V-head map, so it was wrong for this file as well.
- 2026-11-23, bvlob's research branch (`abondis/openflowlm-experiments` `perf/qwen36-rewrite` `9a5e5f87`, Linux, a
  Krackan box; their numbers do not carry here). What it gives this plan:
  - No decode speedup landed there (status report, row 210; later rows refute or close each decode lever).
  - Their measured attribution (their box): decode 70 ms at position 1, 74 at 1k, 96 at 8k, 183 at 32k, 298 at 64k;
    decode growth is the ax0 attention, linear in position (0.36 us a position a layer), so about 1 s a token at
    262k (inferred). 32k gate passed there with sets exported at `OFLM_EXPORT_MAX_CTX` 32768 / 65536; trap: the
    attention bound `attn_newrow` is compiled in, not read from the manifest.
  - Block GEMM `T` 1280 -> 1024 gave +7.1% at 1k there from removed padding; this recipe uses `T` 256, which has
    no such padding. A larger `T` is still worth one A/B here.
  - Their tiled AVX2 router (2.8 s -> 0.27 s a 1k prefill there, at -O0 / -O3 GCC) measured no gain here: MSVC
    `/O2 /arch:AVX2` runs the per-token router in 4-6 ms a layer at T 1024 (about 0.2 s a 1k prefill), and the
    tiled port 5.5-6.5 ms (TB 32 / 8 / 4). Not ported.
  - Nothing there on MTP (closed by its owner), on GGUF, or on intermittent lx1 ERT 8 with the stock set.
- 2026-11-23, phase 1 gate, GGUF q8 view on the block route (`kernels-q8`, 8 chunks of 512, 2040 scored positions,
  teacher-forced; `C:/models/qwen36moe/kl/q8/score.json`): prefill 512 tokens in 10.4-13.3 s (20-26 ms a token,
  development runs, not a benchmark). Every chunk ran; logits dumped through the CLI's checked write.

  | pair | KL mean | KL median | same top | PPL |
  |---|---|---|---|---|
  | NPU vs llama.cpp Vulkan, same file | 0.0401 | 0.0173 | 91.0% | 7.487 vs 7.347 |
  | llama.cpp Vulkan, shared expert + Q6_K + head requantized as the view does, vs the file | 0.0393 | 0.0183 | 90.8% | 7.482 |
  | the same, shared expert only | 0.0389 | 0.0178 | 91.1% | 7.468 |
  | the same, the 3 Q6_K `ffn_down_exps` (to q4_1) and the head (to q8) only | 0.0021 | 0.0016 | 97.5% | 7.347 |
  | NPU vs llama.cpp Vulkan on the requantized file (NPU arithmetic alone) | 0.0262 | 0.0101 | 92.5% | 7.487 vs 7.480 |
  | control: llama.cpp CPU vs Vulkan | 0.0190 | 0.0078 | 94.3% | 7.344 |

  The gate fails (0.040 against 0.019). The requantized Q8_0 shared expert explains the whole gap; the NPU's
  arithmetic adds 1.4x the backend spread, as on Nex (0.026 vs 0.020). The requantized copy is made by
  `.plans/qwen36moe/requant_emul.py` (gguf-py Q4_1, fp16 scale and min: the same law up to the scale rounding).
  Proposal: phase 1b.
- 2026-11-23, `phlegm run` failed at template compile: Qwen3.6's chat template uses transformers' `{% generation %}`
  / `{% endgeneration %}` assistant-mask tags, which minijinja lacks. Fixed in phlegm (`hf_generation_tags`, a
  plain `if true` block; unit test `generation_tags_render_their_body`).
- 2026-11-23, `phlegm run` (phlegm `feat/qwen36moe-bar` release build, the bridge built from this worktree by
  `C:/models/qwen36moe/bridge/build.cmd`, its `build.json` written by `.plans/qwen36moe/bridge_sidecar.py` with
  `build.ps1`'s hashing; `kernels-q8`, `--think false --temperature 0 --max-ctx 4096`): a 31-token prompt answered
  "Three prime numbers greater than 50 are 53, 59, and 61. I verified each by confirming they are not divisible by
  any prime numbers less than their square roots (specifically 2, 3, 5, and 7), ...". Correct.
- 2026-11-23, a `--max-ctx 262144` export (`kernels-q8-262k`) rebuilt nothing: every xclbin and instruction stream is
  byte-identical to the 4096 set; only `max_ctx_default` differs. The decode streams compile no context capacity
  (the `attnpos` patch writes the window length and row offsets at run time; bit 31 is the only flag, so 32-bit
  values hold to about 1M rows of 2 KB). bvlob's `attn_newrow` trap does not apply to this tree (no such symbol).
- 2026-11-23, decode step against depth (`kernels-q8`, `--max-ctx 262144 --at-position P --bench-step 5`, zeroed KV:
  dispatch timing only, no values checked; npu + bigmem held, a development run, `.plans/qwen36moe/depth.sh`).
  Wall time a step, submit-ahead (the default schedule), median:

  | position | 1 | 4096 | 32768 | 131072 | 262100 |
  |---|---|---|---|---|---|
  | ms a step | 104.9 | 119.2 | 224.2 | 594.8 | 1084.2 |

  Every dispatch completes at a 262100-row window. The growth is linear: 3.7 us a row a step, 0.37 us a row a full
  layer (bvlob measured 0.36 on Krackan). At 262k the 10 full layers read about 5 GB of KV a step (2 KB a row), which
  streams in about 0.1 s at 50 GB/s, so the walk is about 10x over the memory floor: compute-bound, as gpt-oss's was.
  Phase 4 lever: the ax0 decode attention walk.
- 2026-11-23, `phlegm serve` (same build and bridge, `kernels-q8`, `--max-ctx 8192`, greedy;
  `.plans/qwen36moe/serve_run.sh` + `serve_check.py`, log `C:/models/qwen36moe/phase2/serve_check.log`). All pass:
  - plain, thinking off: "391" for 17 * 23, empty `reasoning_content`, 3 completion tokens, 1.6 s.
  - streamed: one correct sentence, first token at 1.08 s.
  - thinking on: `content` "391", 557 characters of `reasoning_content`, `finish_reason` stop, 26.1 s.
  - tool call: `finish_reason` tool_calls, `get_weather` with `{"city":"Paris"}`; after the tool message, "The
    weather in Paris right now is light rain with a temperature of 14°C." (prompt cache reused 292 of 331 tokens).
  - Trap: phlegm takes the dibs npu lock itself, so under `npuq.sh` (which holds it) the server printed "NPU busy
    (lock held) -- waiting..." repeatedly before it started (the whole check took 14 minutes; the cause of the release
    is not known). Run serve without the outer npu lock next time.
- 2026-11-24, phase 1b implemented (OPEN-SHARED-HOST-Q8): recipe `shared_host` / `shared_steps`, `lx` / `ax`
  `nx = NX_DECODE`, manifest `shared_q8`, `Core::shared_q8`, `host::shared_expert_q8` (unit test against an fp64
  forward: max error 0). Export `C:/models/qwen36moe/kernels-q8s`: lx0 / lx1 fit with 48 B of program memory free;
  both layer programs are `run, moeroute2, shared_q8, run`. `check_import.py` passes on an LF copy of the tree (66
  identical, 27 recorded); on this checkout it fails only from `core.autocrlf`.
- 2026-11-24, phase 1b gate, block route (`kernels-q8s`, 8 chunks, 2040 positions; `kl/q8s/score.json`):

  | pair | KL mean | KL median | p95 | same top | PPL |
  |---|---|---|---|---|---|
  | NPU, shared expert from the Q8_0 codes, vs llama.cpp Vulkan | 0.0235 | 0.0082 | 0.085 | 92.9% | 7.407 vs 7.347 |
  | before (shared expert requantized to q4_1) | 0.0401 | 0.0173 | 0.137 | 91.0% | 7.487 |
  | control: llama.cpp CPU vs Vulkan | 0.0190 | 0.0078 | | 94.3% | 7.344 |

  Better than the 0.026 the plan inferred: the NPU now sits at 1.24x the backend spread in mean and 1.06x in median.
  Three chunks of the first run failed with `Failed to submit command to hw queue (0xc01e0200)` while free RAM was
  about 10 GB; their reruns passed.
- 2026-11-24, decode hangs (sequential route, a decode step per prompt token, 512 tokens a run): `lx1` / `ax1` end in
  ERT state 8 at random positions, on `kernels-q8s` and on `kernels-q8` (before phase 1b) alike: 6 of 8 runs failed on
  `kernels-q8`. Each hang logs `pci` Event ID 3 with `TxnOp=ffffffff`, zero fault fields and `CtxPC=28b060ad`: the
  lost-command signature (F-0108), the same PC as the phase 0 hangs and other models' hangs on 2026-11-13. A fresh hang
  comes after 17,000-29,000 dispatches; a run started right after a hang can fail at position 0.
  `xrt-smi examine -r aie-partitions` shows Windows' `WorkloadsSessionHost.exe` contexts on the NPU at all times
  (one at 188,788 submissions, one still submitting). Hypothesis: F-0185 (a contender hangs a full queue at normal
  priority). Probe: `OFLM_OPEN_CTX_PRIORITY` creates the engine's contexts with an amdxdna priority (0x180 high).
- 2026-11-24, the hang is not the contender: high priority (0x180) hung 7 of 8, serial dispatches 5 of 8, and the
  Windows contexts were near idle at each hang. `lx1` / `ax1` are the only decode streams the host rewrites before
  every dispatch (`moeroute2`); `lx0` / `ax0` never failed. With the routed stream patched into alternate copies
  (`OFLM_OPEN_ROUTE_ALT` probe), 8 of 8 runs completed (about 164,000 MoE dispatches); sequential-route KL 0.0210,
  median 0.0079, same top 93.3% (`kl/q8s_seq_alt/score.json`). Now the default (OPEN-ROUTE-ALT;
  `OFLM_OPEN_ROUTE_ONE=1` restores one copy). CLI and bridge rebuilt.
- 2026-11-24, the residual hangs (1 of 8 with two copies and the overlap; resubmission on the same context also
  ends in state 8, so it was removed) and a hang in the first decode step of a `phlegm bench` warmup. The
  `xrt-smi` samples of 10:24-10:40 show other agents' processes on the NPU while our runs held the dibs npu lock:
  `run_kernel.exe` at 10:27:39, 10:27:44 and 10:28:47, beside our hangs at 10:27:45 and 10:28:51 (Event ID 3
  times); `bench.exe` from 10:32:41 to 10:34:54; `inkling2_npu_check.exe` at the 13:23 bench hang (that one was this work's fault: `phlegm bench` takes only phlegm's own `phlegm-npu.lock` file, not dibs npu, so it started inside the Inkling session's hold; `bench2.sh` now wraps it in dibs). `npuq.sh` only
  looked for named clients, so it did not see them. Now it waits for any non-Windows NPU context and logs every
  foreign one seen during a run. F-0185 (a contender hangs a full queue) is again the lead hypothesis; the A/B
  rows of OPEN-ROUTE-ALT were not taken with the NPU free of other agents' work, so they are not conclusive.
- 2026-11-24, a loss with no foreign client: the `phlegm bench` run under dibs npu lost `lx1` layer 13 at
  position 1081 (dispatch #83,696) while the watcher logged no foreign NPU context. So contention is not the only
  cause; the engine now recovers instead (OPEN-ROUTE-REDO). Order of findings: rerunning `lx1` on a new context
  completed as a no-op with the old instruction BO (the provoked gate's chunk 0 then crashed in `xrt::bo::~bo`
  during the exception unwind), and hung with a new BO; whole layers from `lx0` on a new context match a plain
  run; a new context made after freeing the old one hung, so the old one is kept. Forced-loss probes (`lx1`
  layers 0/4/29/38, `ax1` layers 3/39, serial and submit-ahead) give the plain run's tokens. Copy cost about 1 ms
  a token. `check_import` PASS (66 identical, 27 recorded) on an LF copy.
- 2026-11-25, needle at 262,000 (50 %, `kernels-q8s`, `needle.sh`): PASS, answer 58213. Prefill 14,706 s
  (17.8 tok/s; GEMMs and block attention 9,943 s, host 3,974 s). Against 131k (910 s and 1,066 s) the NPU part grew
  10.9x for 2x the tokens, more than the 4x of the attention area (cause not measured). Decode at 262,004-262,010
  1,120-1,189 ms a step, of which `part0` (the attention walks) 1,057-1,090 ms. Log: `ctx/needle/n262000_0.5.log`.
- 2026-11-26, moved to the Strix Halo box (Ryzen AI MAX+ 395, Linux 7.2.15-omarchy, amdxdna in-tree, NPU firmware
  1.1.2.65, XRT 2.21.75 from the distro and phlegm's npu-dev XRT 2.27.0, mlir-aie 1.4.2 in `~/ironenv142`). Model
  `~/models/qwen36moe/Qwen3.6-35B-A3B-UD-Q4_K_S` (same Unsloth revision `5bc3e238`; sha256 in `gguf/SHA256SUMS`).
  `benchmarks/reference/qwen36moe/paths.sh` holds both machines' paths; every helper has a Linux form.
  - Setup traps: the distro XRT has no `aiebu-asm` (the export takes it from phlegm's npu-dev XRT); XRT 2.21.75
    refuses userptr device buffers at every size on this kernel (`amdxdna_insert_pages: Failed to mmap dma buf
    -22`), so the engine now falls back to shmem buffers, and the helpers use XRT 2.27, which takes them;
    `xrt-smi configure --pmode` needs root here (`DRM_IOCTL_AMDXDNA_SET_STATE ... Permission denied`), so the NPU
    stays in the Default power mode until someone runs it with sudo.
  - Cold export of `kernels-fa262k` (all images, attn_fa256 tiers to 262,144 rows, verify images): about 15 min at -j 8.
  - Reproduced: needle 8192 PASS (prefill 40.1 s on a busy box); `--verify-check 8` PASS; `--spec-mtp 128` SPEC
    IDENTICAL; KL gate against a llama.cpp CPU base on a regenerated haystack (the first 40 essays of
    `sgoel9/paul_graham_essays` @ `0c7155a5`, sha256 `ce2f77f5...`): mean 0.0275, median 0.0080, p95 0.108, same top
    93.4 %, PPL 6.301 against 6.291. The CPU-vs-Vulkan control on this text is not run yet.
  - Vision end to end (OPEN-VISION-BLOCK, SERVER-IMAGE-INPUT): see the spec.
- 2026-11-26, Strix Halo speed context (development, NPU Default power mode by the user's choice; the box is shared:
  another session trains on the iGPU at 93-99 % busy and the CPU load average ran 16-40). The same 4,096-token
  block prefill took 45 s at load 30-40 and 23-24 s at load 16-28, and decode at position 1 read 150-250 ms a step
  (serial 211 ms, submit-ahead 149-198 ms; Strix Point in performance mode: 101-110). Ratios from alternating
  runs are the only numbers used to accept a change here (R-0006).
  - OPEN-MOE-PIPE: 4k prefill 1.19-1.29x, logits byte-identical.
  - OPEN-DN-FUSED: DeltaNet host 1.05-1.21 -> 0.77-0.86 ms a token at 4k; KL gate 0.0278 (0.0275 before).
  - OMP threads 8 / 16 / 32 make no difference to 4k prefill on the busy box (40.2-45.3 s, alternating).
  - 4k profile after both (ms a token, `dn_new2`): linear layers 4.18 (MoE waited 1.60, DeltaNet host 0.77,
    tail 0.30, GEMM 0.22, MoE gather + scatter 0.53 now under the NPU's passes), full layers 1.42 (MoE waited 0.53,
    GEMM 0.27, attention 0.11); whole 5.62.
- 2026-11-26, the KL control on the regenerated text: llama.cpp Vulkan base (`build-vk` with Khronos headers 1.4.363 and
  SPIRV-Headers vulkan-sdk-1.4.335.0 from a local prefix; the box has no vulkan-headers package): PPL 6.2388. llama.cpp
  CPU against it: mean 0.0269, median 0.0077, same top 92.9 %. NPU (`fa2`, after OPEN-DN-FUSED) against it: 0.0241,
  0.0066, 93.4 %. `kl/base_vk_c512.kld` now points at the Vulkan base, as on Strix Point.
- 2026-11-26, phlegm bridge: a request line may now be 16 MB (was 1 MB: a 262,000-token prompt's ids are 1.6 MB).
  `phlegm run --image` PASS; 32,768-token needle through `phlegm serve` PASS (189 s).
- 2026-11-26 12:27-13:03, 262,000-token needle (depth 0.5) through `phlegm serve` (`needle_serve.sh 262000 263168`, MTP
  on, the chat template around it: 262,005 prompt tokens): answer 58213, PASS, 2,163 s wall (prefill and 5 tokens),
  Default power mode, shared box. Against the CLI's 2,457 s on Strix Point (performance mode, before OPEN-MOE-PIPE /
  OPEN-DN-FUSED). This closes handoff item 4 (262k through `phlegm serve`).
- 2026-11-26 13:12, `phlegm serve` features on Strix Halo (`serve_features.sh`, MTP on): thinking off "391" with empty
  reasoning; thinking on "391" with 1,042 characters of reasoning, `stop`; streamed "Red, Orange, Yellow" in 7
  chunks; `get_weather` `{"city":"Paris"}` with `finish_reason` tool_calls, and after the tool message "The weather in
  Paris right now is light rain with a temperature of 14°C." (292 of 325 prompt tokens reused). FEATURES PASS.
- 2026-11-26 14:30, speculative modes through `phlegm serve` on Strix Halo (`spec_serve.sh`, OPEN-DECODE-STREAM-LX set,
  greedy, thinking off; outputs byte-identical to `off`): off 10.43 / 10.42 tokens/s (rewrite / explain), MTP 12.29 /
  11.07 (1.18x / 1.06x; rewrite 57 of 58 passes took both rows, explain 112 of 142), lookup 12.12 / 10.18 (1.16x /
  0.98x). The MTP cycle: verify pass 144.7 ms (1.51 plain steps here against 1.35 on Strix Point), draft 12.2, commit
  3.9. MTP stays the shipped drafter; its gain here is bounded by the verify pass's cost, which has the same context
  switches as decode (`vlx` / `vax`).
- 2026-11-26 14:25, measure unit (`scratch/measure_unit.sh` under `arbiter run --measure`, NOISY only for one other
  process at one core for 407 s): `phlegm bench --development` 212 / 251 / 276 / 287 tok/s prefill and 10.4 / 10.2 /
  9.8 / 9.1 tok/s decode at 512 / 1k / 2k / 4k; CLI 4k and 8k prefill 3.48 / 3.47 ms a token; decode 94 ms a step.
  The busy-box numbers earlier in the day were 1.6-1.9x slower: the box's load, not the engine.
- 2026-11-26 15:00, OPEN-DECODE-ONE-CONTEXT-Q8: with every projection role at q8 the merged decode image needs one
  GEMV entry, the q8 one: `ux` builds for this spec at 16,256 of 16,384 B and removes the 20 lx / ax context
  switches a step (94.0 -> 78.9 ms), with OPEN-DECODE-STREAM-UX's knobs 73.5 ms; `vux` merges the verify pass
  (142.5 -> 121.8 ms). All gates bit-identical. Speculation is then near break-even on this box (plain 73 ms against a
  1.67-step verify pass): phlegm drafts only while it pays (SERVER-SPEC-COST). Measure unit: decode 13.2 / 12.9 /
  12.3 / 11.2 tok/s at 512 / 1k / 2k / 4k, prefill unchanged (212-287 tok/s).
- 2026-11-26 15:40, OPEN-MOE-DQBUF: the expert kernel at 16 tokens a visit with the dequantised k-tile kept in L1
  (4.6 us a token against 6.3); prefill 1.16-1.17x at 4k and 8k (8k 24.6 s, 333 tok/s), logits byte-identical.
  Rejected on the way: native `mmul<4,8,8>` (slower in both shapes), the full mac unroll (program memory).
- 2026-11-26 15:44, measure unit after OPEN-MOE-DQBUF: prefill 227 / 280 / 315 / 334 tok/s, decode 13.3 / 13.0 / 12.4 /
  11.3 tok/s at 512 / 1k / 2k / 4k (`bench/phlegm_bench_measure_1544.json`); 8k CLI prefill 2.99 ms a token. Vision
  tower 2.2 s for 3,200 patches on the quiet box (AVX-512 BF16 linears tried: no faster, 2 % less exact, dropped).
- 2026-11-27, OPEN-DECODE-STREAM-UG + OPEN-LMHEAD-Q6 exported (`kernels-fa262k`): decode 71.3 ms a step (72.2-72.4 with
  `OFLM_OPEN_LMHEAD_Q6=0`). Measure unit (NOISY only for another session's one-core wait loop): prefill 223 / 277 / 313
  / 332 tok/s, decode 13.5 / 13.1 / 12.5 / 11.4 tok/s at 512 / 1k / 2k / 4k (`bench/phlegm_bench_measure_q6.json`,
  sha256 69689673697316fb...). The unit before it, with a 30-core process alongside, is kept as
  `phlegm_bench_q6_NOISY.json` and not used. MemDraft (the Qwen3.5-9B code table) as the draft source: 35 % of
  one-token drafts taken against MTP's 90 %, 0.95-1.05x against MTP's 1.20x on a loaded box; MTP stays the drafter. The CLI hook used for this
  (`OFLM_SPEC_MEMDRAFT`) is not committed: a Qwen3.6-trained table, and a verify pass wider than two rows, would come
  first.
