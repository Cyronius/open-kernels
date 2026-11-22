# Sync from openflowlm-next: main through #234, then #245's model-onboarding tools

Date: 2026-11-21. Upstream: https://github.com/Atomic-Germ/OpenFlowLM-Next, local clone `C:/code/OpenFlowLM-Next3`
(remote `origin`). Pinned refs:

- upstream `main` = `d7c145f5` (#234)
- PR #245 head = `8410c965` (remote-tracking ref `upstream/pr-245`). Do not follow later pushes.

Our last import is described in `open_kernels/UPSTREAM.md` (2026-10-23 and 2026-10-28 rows). We already have
#193, #194, #207, #209, #213, #223 (pre-merge head `0c22864`) and #225 from their branches.

## Goal

1. Bring the merged state of upstream `main` into open-kernels, keeping every local change that `UPSTREAM.md`
   ("Changes made here that upstream does not have") and `benchmarks/reference/engine-changes.json` list.
2. Then bring in #245's spec -> gate -> build -> catalogue -> register tools so that a new checkpoint of a
   supported family needs one command. These tools are the developer half of phlegm's P4 (automatic model
   onboarding).
3. Everything lives in open-kernels at the same paths as upstream. phlegm only bumps the submodule.

## Out of scope

- `utilities/oflm-add/` (oflm's installer), q4nx-build, `release/`, `oflm-test`, packaging (#242), #185 (oflm pull),
  oflm's settings registry, `src/pull/dev_command.cpp` and `src/src/main.cpp`.
- The one-off research tree in `utilities/` (`prefill-*`, `l3-*`, `dit-*`, `sequential-*`, etc.) and their tests,
  unless a kept file imports them.
- `--fit` (reads Linux `MemFree`; phlegm-core owns context sizing). File a phlegm follow-up instead.
- Open upstream PR #247. Wait for it to merge.
- phlegm-side wiring (`phlegm kernels build`, adding a model through the tray). Separate phlegm plan.

## Branch and baseline

- Work in the standalone clone `C:/code/open-kernels-oflm-sync`, branch `sync/oflm-d7c145f5` (already created from `origin/main` `d3a75d7`). Remote `upstream` = `C:/code/OpenFlowLM-Next3`.
- Before any change, run the offline suite and record the count:
  `python -m pytest -q --tb=short` with `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1`.
- Commit each phase separately. Do not push and do not open a PR; Cyrus reviews first.
- NPU runs: take the lock first (`dibs` skill / `dibs` command, resource `npu`).

## Phase 1: #230 (block prefill padded tail)

- `git cherry-pick -x 4a07b3e8` (fetch it from the upstream clone first). Engine files only.
- Resolve against our `core.cpp` / `engine.cpp`. Keep `engine-changes.json` hashes valid: if a file in that list
  changes, update its hash and add no new change entry (this is upstream code).
- Gate: `block_host_test` builds and passes.

## Phase 2: #232 and the rest of main, Python side

Files (upstream path = our path):

- `open_kernels/gguf_pool/` package (`__init__.py`, `gguf_pool.py`, `gguf_scale_width.py`). It replaces our flat
  `open_kernels/gguf_pool.py` and `open_kernels/gguf_scale_width.py`. Update every importer (at least
  `designs/gemv_q4/make_test.py`, `designs/lm_head_q8/make_test.py`, `model/flint_gguf_reference.py`,
  `model/gptoss_gguf_reference.py`, `recipes/dense.py`, `recipes/gguf.py`, `recipes/qwen35.py`,
  `recipes/qwen36moe.py`). Delete the flat files only when nothing imports them.
- `open_kernels/model/gguf_ref.py`; `designs/gguf/{gemv_q4,lm_head_q4,lm_head_q8}/`; `designs/lm_head_q4/{make_test,compare}.py`;
  `designs/ln/probe_nr32.py`.
- `recipes/spec.py`, `pack.py`, `qwen35.py`, `qwen36moe.py`, `catalogue.py` and the changed spec JSONs.
- New spec JSONs: take `lfm2-2.6b`, `llama3-8b`, `nanbeige41-3b`, `qwen3vl-4b`, `qwen3vl-8b` (the vision ones only if
  Phase 3 takes the ViT GGUF route). Skip `gemma-4-12b-it-npu2.json` (an NPU2/q4nx name; we have our own Gemma 4 spec).
  Take `recipes/candidates/*` as they are (candidates are not claims).
- Tools: `utilities/op-gate.py`, `register-spec.py`, `spec-gaps.py`, `clean-kernels.py`, and the newer
  `build-new-spec.py`, `gen-kernel-spec.py`, `export-kernels.py`, `catalogue-add-point.py` (#228's pyxrt lookup included).
- Tests: `specs/open-engine/tests/test_spec_gates.py`, `test_gguf_section.py`, `test_model_registry.py` only if it
  does not need oflm-add, `test_spec_model_names.py` changes, and any test that traces a kept file.

Method: three-way merge per file. Base = the upstream ref we imported that file from (`0c22864` for #223 files,
the refs in the `UPSTREAM.md` table and `utilities/kernel-sync-sources.json` for others). Ours = open-kernels.
Theirs = `d7c145f5`. On conflict, every row of "Changes made here that upstream does not have" wins, in particular:

- manifest version 7 for `bfp16_dit`
- Bonsai t2 packing, merged images, verify passes and tail
- the attention cache key's head width; Qwen3-0.6B 128-wide heads
- `recipes/dense.attn_flags` as the one flag source; `OG_STREAM` condition in `dx.py`
- MiniCPM5-2B kv heads 2 and its catalogue tuple
- Windows path and line-ending normalisation

Windows: every tool must run from Windows Python with `pathlib` paths, as the 2026-10-23 import did.

Gate: offline suite count >= baseline plus the new tests, no new failures. Update `kernel-sync-sources.json`
(ref, commit, sha256) for each tool taken.

## Phase 3: #232 and the rest of main, C++ engine side

`src/open_qwen36/` changed a lot upstream (`core.cpp` +1.8k, `pools.cpp`, `manifest.*`, `block_host.*`, `cli.cpp`,
`gguf_file.*` removed, new `spec_lookup.*`, `prefill_*.hpp`, `buffer_range.hpp`, `diagnostic_range.hpp`,
`layer_major_policy.hpp`, `vision/vit.cpp` + `vit_gguf_test.cpp`).

Method: rebase and replay. Start from upstream `d7c145f5`'s `src/open_qwen36/`, then re-apply each entry in
`benchmarks/reference/engine-changes.json` (pr226 manifest move-only, llama GGUF q/k rows, GGUF lmhead_q8,
dense layer-major rope, Gemma 4 patches, and the rest), then Phase 1's #230 if upstream lacks it (it does not;
it is in `d7c145f5`). Update the hashes and run `benchmarks/reference/check_import.py`. Change `base` in
`engine-changes.json` to `d7c145f5`.

Manifest rule: the engine must still read every manifest version our recipes emit, including local version 7.

Take #234's `omp_policy.*` and #231's per-layer profiling with the rest of main; they are on main.

Gates:

- the native tests build and pass on Windows (`src/open_qwen36/build.cmd` or the CMake route the repo uses)
- phlegm's bridge builds from the in-tree engine (`../bridges` in phlegm; see `16084a3`)
- NPU, under `dibs`: greedy decode of the golden set in `benchmarks/reference/` matches for every model that has a
  reference (at least Qwen3-8B, MiniCPM5-2B, Granite 4.2 3B, Qwen3.5 0.8B). Any mismatch stops the phase; report it,
  do not paper over it.

## Phase 4: #245 tools (pinned SHA)

Take from #245:

- `utilities/build-new-spec.py`, `gen-kernel-spec.py`, `export-kernels.py`, `op-gate.py`, new `probe-family-sizes.py`
- `open_kernels/designs/short_conv/{sc_test,make_test,compare}.py`
- `designs/attn/make_test.py`, `designs/attn/probe_dense.py` (bias fifo for qwen2 q/k/v bias)
- `designs/gemv_q4/gemv_q4.py` (x blocks streamed for large K), `gguf_pool/gguf_scale_width.py`
- `recipes/catalogue.py`, `dense.py`, `families.py` (nanbeige -> llama3 alias), `qwen36moe.py`
- `specs/open-engine/tests/test_lfm2.py`, `test_spec_model_names.py`; the `.gitignore` lines for generated
  `gemv_q4_p*b*r*_k*.cc`, `gemv_q4s32_*`, `prep_blk_k*.cc` (keep checked-in baselines)
- The engine guard that allows GGUF block prefill only for supported kinds and pool-sourced GEMM weights
  (`engine.cpp` / `core.cpp` hunks). Not `--fit`.

Changes to make while porting:

1. **No `oflm dev` requirement.** Add `utilities/devenv.py`: finds Python, the IRON toolchain
   (Windows `C:\dev\mlir-aie\ironenv` via `iron_env.ps1`'s variables, Linux `ironvenv` / `utilities/env.sh`), XRT
   and the NPU; `python utilities/devenv.py check` prints what `oflm dev env-check` prints; a function builds the
   `run_kernel` harness (CMake on `open_kernels/harness`) on demand. Every #245 script imports it instead of
   assuming `oflm dev` set the environment. Reuse `export-kernels.py`'s existing venv logic; do not fork it.
2. **No oflm-add import.** Move `family_from_config_dict` (and the alias tables it needs) from
   `oflm_add/__init__.py` into `open_kernels/recipes/families.py`. `build-new-spec.py` and `register-spec.py` import
   it from there.
3. **Windows.** `pathlib` everywhere; `python` not `python3`; no bash requirement.
4. **Specs.** Take only specs that #245 says were verified end to end and that meet the production bar in
   `AGENTS.md`: candidates are `llama3-2b`, `lfm2-700m`, `qwen3vl-8b`, `nanbeige4.1-3b` and `qwen25-it-1.5b`
   (#245 also reports Gemma 3 12B working, but its head has no spec diff for it). Every other new spec goes to `recipes/candidates/` or is left out. Do not take #245's
   rewrites of `qwen35-27b.json` and `qwen3-8b.json` without a diff review against ours; report the diff.
5. **Spec text.** Apply #245's `OPEN-ATTN-QKV-BIAS` verification change in `specs/open-engine/spec.md`, with
   `utilities/build-new-spec.py` in place of `oflm dev build-new-spec`.

Gates:

- offline suite green
- `python utilities/build-new-spec.py --spec qwen3-4b --dry-run` prints a plan on Windows
- under `dibs`: `python utilities/build-new-spec.py --repo <one small verified repo, e.g. a Qwen2.5-1.5B GGUF>`
  reaches `RC=0, spec ready`, and the resulting set decodes on the NPU

## Spec impact

- Modified: `OPEN-ATTN-QKV-BIAS` (verification: automated gate replaces the manual procedure for new tuples).
- New/modified requirements arriving with #232/#231/#234 text in `spec.md`: merge upstream's text; where an ID
  collides with ours, keep ours and record the rename in `UPSTREAM.md`.
- New: `OPEN-ONBOARD-ONE-COMMAND` (verification: manual). A checkpoint of a family with a recipe, whose shape the
  catalogue covers or the gate can validate, goes from HF repo or GGUF to a registered spec with one command, on
  Windows and Linux. Acceptance: the three outcomes (`RC=0`, parked in candidates with a named constraint,
  `NOT_AUTOMATED`) and the Phase 4 gates above. Write the procedure in `spec.md`.

## Ledger

Update `open_kernels/UPSTREAM.md`: rows under "Pulled in" for `d7c145f5` (#228, #230, #231, #232, #234, #239) and
#245 at its pinned SHA; rows under "Changes made here" for `devenv.py`, the moved `family_from_config_dict`, the
specs held back, and any conflict resolution that differs from upstream.

## After

Move this plan to `specs/open-engine/plans/archive/`. In phlegm: bump the `open-kernels` submodule on its own branch.
## Windows hardware handoff

The Windows build and offline gates cover the engine, bridge, harness and onboarding dry run. The NPU gates remain pending.

### Phase 3 golden decode

Run from `C:/code/phlegm`. Build the bridge from the Phase 3 source before any Phase 4 engine change. Use the matching case JSON and recorded golden JSON for every model that has one, including Qwen3-8B, MiniCPM5-2B, Granite 4.2 3B and Qwen3.5 0.8B.

```powershell
git -C C:/code/open-kernels-oflm-sync worktree add --detach C:/code/open-kernels-oflm-phase3 cd2f13a7
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/reference/build.ps1 -ReferenceRoot C:/code/open-kernels-oflm-phase3 -OutDir target/reference-phase3 -XrtInclude C:/dev/XRT/src/runtime_src/core/include
$goldenDir = 'target/npu-dev/one-runtime/golden'
$casesDir = 'target/npu-dev/engine-intake/hardware-cases'
$bridge = 'target/reference-phase3/phlegm-reference-bench.exe'
$goldens = @(Get-ChildItem -LiteralPath $goldenDir -File -Filter '*.json')
if ($goldens.Count -lt 4) { throw 'The golden directory needs all four reference models.' }
foreach ($golden in $goldens) {
    $cases = Join-Path $casesDir $golden.Name
    if (-not (Test-Path -LiteralPath $cases)) { throw "Missing case list $cases" }
    $out = "target/reference-phase3/$($golden.BaseName)-compare.json"
    dibs run npu -- python -I benchmarks/reference/ab.py compare --bridge $bridge --cases $cases --golden $golden.FullName --rounds 2 --out $out
    if ($LASTEXITCODE -ne 0) { throw "Golden comparison failed for $($golden.BaseName)" }
}
```

The case lists must resolve to the installed models and kernel sets on the machine running the gate. The expected golden assets were absent from the sandbox checkout.

### Phase 4 onboarding

Run the build from `C:/code/open-kernels-oflm-sync` after applying the bundle and checking the Phase 3 results.

```powershell
dibs run npu -- python utilities/build-new-spec.py --repo unsloth/Qwen2.5-1.5B-Instruct-GGUF --spec-name qwen25-it-1.5b
```

Expect `RC=0, spec ready`. From `C:/code/phlegm`, build the Phase 4 bridge and run the decode gate. Set `$env:ONBOARD_MODEL_DIR` to the local GGUF model directory containing `model.gguf` from that repository.

```powershell
powershell -NoProfile -ExecutionPolicy Bypass -File benchmarks/reference/build.ps1 -ReferenceRoot C:/code/open-kernels-oflm-sync -OutDir target/reference-phase4 -XrtInclude C:/dev/XRT/src/runtime_src/core/include
dibs run npu -- python benchmarks/reference/smoke.py --bridge target/reference-phase4/phlegm-reference-bench.exe --model-dir $env:ONBOARD_MODEL_DIR --kernels C:/code/open-kernels-oflm-sync/src/xclbins/Qwen2.5-1.5B-Instruct-GGUF-NPU2/open_kernels
```
