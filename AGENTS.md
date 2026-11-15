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
- The kernel writer (`open_kernels/recipes/`) and the engine's manifest reader (`src/open_qwen36/`) are one contract.
  Change them in the same commit. Preserve the manifest/runtime ABI, the local version-7 DIT mapping, and the Llama
  high/low probability contract.
- Companion patches in `open_kernels/upstream-patches/` have separate source bases; they are not a ready-to-apply
  series.
- After an engine change, update `benchmarks/reference/engine-changes.json`; `check_import.py` checks the engine
  against OFLM trunk.

## Research and optimization

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
