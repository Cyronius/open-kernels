# Working in open-kernels

## Start and resume

- Local plans and handoffs live in ignored `.plans/`. Keep durable shared evidence in tracked specs.
- Preserve existing uncommitted work. Inspect `git status` and the relevant diff before editing; do not reset,
  overwrite, or commit unrelated changes.
- This repo is often checked out as phlegm's `open-kernels/` submodule. A submodule checkout is a detached HEAD:
  make a branch before you commit, and push here before phlegm moves its pointer.

## Direction

- This repository is the canonical home of `open_kernels/`, `src/open_qwen36/`, `npu_offload/`, `viz/` and
  `specs/open-engine/`. Consult `open_kernels/UPSTREAM.md` before kernel or engine work.
- openflowlm-next has the same paths. Open one upstream PR per finished feature.
- Do not do engine or app work in openflowlm-next. Make open-kernel work, including its engine side, here and
  in phlegm when needed. OFLM's engine, server, registry and `oflm` tools are a separate effort: list what they
  need in the plan, and do not change them.
- The kernel writer (`open_kernels/recipes/`) and the engine's manifest reader (`src/open_qwen36/`) are one contract.
  Change them in the same commit. Preserve the manifest/runtime ABI, the local version-7 DIT mapping, and the Llama
  high/low probability contract.
- Companion patches in `open_kernels/upstream-patches/` have separate source bases; they are not a ready-to-apply
  series.
- After an engine change, update `benchmarks/reference/engine-changes.json`; `check_import.py` checks the engine
  against OFLM trunk.

## Production bar for OFLM kernel sets

Scope here: open-kernels work is separate from OFLM. Both oflm and phlegm are test suites for the items below:
`phlegm run` / `phlegm serve` / `phlegm bench` and `oflm add` / `oflm serve` / `oflm bench` both count as evidence.
OFLM's registry entry, engine, server, docs pages and README row are OFLM's separate effort. List them under an
"OFLM side" heading, not as open-kernels gaps.

Do not add q4nx support for any model: q4nx is being deprecated. New support
loads the model from GGUF.

Every OFLM PR that adds or changes support for a model also updates its entry
in `src/model_list.json`: the entry exists, names the files the engine loads,
and states the real quant, family and context length. A model with no correct
registry entry is not supported, whatever the engine can do.

A top-tier supported model ships finished and production-ready. Before you export
its kernel set as done, or open an OFLM issue or PR that calls it supported,
show each of these items with measurements:

- Prefill uses the block route (usually). State the reason for any exception.
- Decode and prefill are near optimal. Profile them and name what is left.
- No unusual context-length limits. Test up to the model's real context, not
  only the 4096 default.
- Everything runs on the NPU (usually). Name each host stage and its cost.
- Speculative decoding was explored: MTP, DFlash, DFlash2, prompt lookup or
  another drafter. Ship one, or reject it with the measured reason. We
  generally want one.
- `oflm add` / `oflm serve` or `phlegm run` / `phlegm serve` run the model, not only `open_qwen36_cli`.
- The model ships quantized, usually q4. Prefer Unsloth's GGUFs when the
  engine supports their quant types; state the reason for any other source.
- Benchmarks are recorded: `oflm bench` or `phlegm bench` results (prefill and decode against
  context length, with the machine and stack named) on the model's page in
  `docs/docs/benchmarks/`, and a row in the README's "Recommended models"
  table. A development build counts for the bar (`phlegm bench --development`,
  or `oflm bench` from a branch); say which build ran.
- Most of the model's own features are exposed: its vision and audio towers,
  MTP heads (as the drafter), reasoning and tool calling, and for image models
  every supported resolution and image editing. Name each feature left out
  and the reason.

If an item is not done, the PR stays a draft and lists the item.

Reaching the bar is not a release. Release steps (benchmarks on a tagged Phlegm
or OFLM build, published release notes) happen when a release is cut, and are
not open items of the bar.

Track each top-tier model's progress (OFLM #240) in `RECOMMENDED_MODELS.md`.

## Research and optimization

- Before starting or resuming a model port or planning its performance work, read the relevant project notes in
  `specs/open-engine/notes/`. Start with [the Inkling 2 Mini port notes](specs/open-engine/notes/inkling2-mini-port.md)
  for new GGUF layouts, recurrent or looped state, long context, and Phlegm verification.
- Update the project notes when a measurement, failed approach, or tooling issue changes how a future port
  should proceed. Link to the detailed evidence in the model spec or plan.
- For Inkling block-prefill measurements, use the tracked [Phlegm profiler](benchmarks/reference/inkling2-prefill/README.md)
  and the [`phlegm-prefill-profile` skill](skills/phlegm-prefill-profile/SKILL.md).
- Read relevant recommendations in `~/code/xdna2-notes/recommendations/` before proposing kernel optimizations.
  Follow their finding IDs to evidence, stack versions and open questions. If that checkout is absent, record the gap.
- Start with R-0001 (budgets), R-0003/R-0008 (contexts and dispatch), R-0004 (reference), R-0006 (measurement),
  R-0011 (buffers), R-0012/R-0013 (rates).
- Keep measured, documented, inferred and hearsay claims distinct. Do not carry absolute speeds between Strix Point,
  Strix Halo, operating systems or stacks.
- Profile dispatches, context switches, DMA floors, KV traffic and allocations before tuning arithmetic. Record
  hypotheses and failed experiments too.
- Check program memory, L1 plus stack, DMA/BD budgets and compiler output before accepting a new kernel shape.
  Rebuild every stream belonging to an image.

## Verification and evidence

- Run focused regression checks for changed routes; use `pytest.ini` and the relevant spec for prerequisites.
  Missing hardware/native dependencies are untested coverage, not passes. Run `git diff --check` for edits.
- Arithmetic changes need independent numerical and model-quality gates, including long-prefix/window boundaries;
  retain known failures explicitly.
- Never commit weights, generated kernels, captures or bulky logs. Link evidence by path and hash; check in small
  summaries or fixtures where appropriate.
