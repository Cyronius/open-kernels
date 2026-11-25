# ArtLite techniques: pages, formats, one weight image, loop variants

2026-11-25. Status: **planned, not started.** Source note: [artlite-speed-techniques.md](../notes/artlite-speed-techniques.md).

## Scope

This plan takes four ArtLite techniques to our stack:

| # | Technique | Phase |
|---|---|---|
| 5 | Page visits and page sizes | 1 |
| 3 | Native GGUF format kernels, then code pages if needed | 2 |
| 2 | One packed weight image for prefill and decode | 3 |
| 1 | Shape-specialized code: a variant search for each loop | 4 |

Whole-line DMA writes (#4) are out of scope. The phase 0 census counts partial lines at no extra cost. If the
count is large, a later plan can take it.

All four items are open-kernels work. Phlegm changes are not in this plan.

## Test bed

- Model: Qwen3-8B. It has both prefill routes (`fast` bfp16 and `lean` q4), the GGUF view, and recorded
  memory peaks (13.3 GiB `fast`, 9.8 GiB `lean`, OPEN-PREFILL-GEMM8).
- Files: our current Q4_1 GGUF, and Unsloth's Q4_K_M. The Q4_K_M file mixes Q4_K and Q6_K tensors.
- Machine: the HX 370 (Strix Point), Windows 11, the native toolchain (`C:\dev\mlir-aie\iron_env.cmd`, `C:\Xilinx\XRT`).
- Context: 512, 1K, 2K and 4K only. Each run takes minutes. Scale-up comes after the optimum is known.
- Measurement: follow xdna2-notes R-0006 (power mode, quiet box, alternated rounds, minimum and median).
- Branch: make a branch from `origin/main` in a separate worktree. The main checkout has uncommitted Inkling 2 work.
  Do not touch it.

## Phase 0: baselines and census tools (no new kernels)

Each step is offline or one short run. Phases 1 to 4 use these numbers as gates.

1. **Baseline.** Qwen3-8B, both files, `fast` and `lean`: prefill tok/s and decode ms/token at 512 to 4K, peak
   RSS, load time, and the time of each dispatch in one prefill block. Record NLL on the existing gate prompts.
2. **Stream census.** A tool reads the instruction streams of one prefill block after the engine patches them.
   - The engine gets a debug switch that writes each patched instruction buffer of the first block to a directory.
   - The tool decodes BD blockwrites (op 1) and DDR patches (op 0x81), as in `harness/stream_patch.hpp` and
     `compose_elf.py:txn_ops`. The shim BD register map is in xdna2-notes F-0041. The BD limits are in F-0004 and F-0051 to F-0053.
   - For each stream and each buffer argument, it counts rows, bytes, rows with a partial 64-byte end line, and
     4 KB page visits. It assumes each XRT buffer starts on a 4 KB boundary. Check that assumption once on the device.
   - Output: one table for each block, ranked by page visits.
3. **Decode byte census.** A tool reads the manifest pool ops and the GGUF header. For each tensor it gives the
   GGUF type, the GGUF bytes, the pool format and the pool bytes that decode streams for each token. No device.
4. **Compiler census.** Port `kernel_remarks.py` to the native Windows toolchain (it now names WSL). Run it on the
   hot kernels from step 1. For each loop, record bundles, MII, spills, code bytes and stack.

Spec impact: none. The debug switch is below the traceability line.

## Phase 1: page visits (#5)

1. **Probe.** On the HX 370, measure with a design that has no compute cores:
   - a read stream and a write stream (16 channels, and 1 channel);
   - 128-byte rows against 132-byte rows (this fills ArtLite's empty Strix Point cell);
   - a gather of short rows, each row on a new 4 KB page. This gives our page-visit rate.

   Use the existing `harness/run_kernel.cpp` path. Record the result as a measured raw note in xdna2-notes.
2. **Estimate.** Page visits from the phase 0 census, divided by the probe's rate, against the block time from the
   phase 0 baseline.
3. **Decision gate.** Continue to step 4 only if the page cost is at least 10% of the block time.
   Otherwise record the numbers in the source note and stop phase 1.
4. **Brick layout.** For the worst stream (ArtLite's worst were GEMM outputs and weight slices), change the layout so the rows of one DMA task share
   4 KB pages. The kernel writer and the manifest reader change in one commit. Add a manifest feature for the
   layout. Keep the version-7 DIT mapping and the older layouts readable.
5. **Linux 2 MB pages (optional).** The engine wraps process memory on transparent huge pages as userptr buffers for
   buffers of 2 MB or more (xdna2-notes F-0118). Do this only when a Linux box is available. Windows cannot get 2 MB
   pages today (amd/xdna-driver#1817).

Gates for step 4: the census shows fewer page visits; prefill is faster at 512 to 4K in alternated rounds; logits
are byte-identical to the old layout (a layout change does not change arithmetic).

Spec impact: steps 1 to 3 none. Step 4 adds a requirement for the layout feature (provisional `OPEN-LAYOUT-BRICK`)
and modifies the manifest feature list.

## Phase 2: native GGUF formats (#3)

Today `std_perm_gguf` dequantizes Q4_K, Q6_K and Q8_0 rows and quantizes them again to the q4 pool layout
(`pools.cpp:1065`). For Q6_K this drops from 6.56 to 5 bits for each weight. For Q4_K the pool streams 7.5% more
bytes than a native chunk (5120 against 4736 bytes, `designs/gemm_q4/NOTES.md`).

1. **Measure the loss.** On the Unsloth Q4_K_M file: our NLL and greedy tokens against llama.cpp CPU on the same
   file. Count the Q6_K tensors and their bytes from the phase 0 decode census.
2. **Q6_K at q8 (interim).** If step 1 shows a loss, route Q6_K tensors to q8 pools. Mixed q4 and q8 cores exist
   (`catalogue.MIXED_CORE_FITS`). Check whether Qwen3 at width 4096 fits in program memory. Gate: NLL against
   llama.cpp, greedy tokens.
3. **Native Q4_K GEMV.** Use the `GEMM_Q4K` method from `designs/gemm_q4` for the decode GEMV: the integer sub-scale
   on the int32 partial, one float epilogue for each 256-wide superblock. Gate: max relative error against a float64
   reference, NLL, greedy tokens, decode ms/token against the q4_1 pool at 512 to 4K.
4. **Native Q6_K GEMV.** It streams 6.56 bits for each weight, against 8.5 for q8. Same gates as step 3.
5. **Code pages (only if needed).** If the Q4_K and Q6_K bodies do not fit in one merged layer image (main cores are
   at 16256 of 16384 bytes, xdna2-notes F-0104), spike code pages first:
   - Can a core write its own program memory through TileControl on our driver? This is open (xdna2-notes raw
     `2026-10-20-artlite-code-paging-for-16kb-program-memory.md`).
   - A context switch to another process or a suspend after 5 s loses the page store. The engine must write the
     pages again before the next step.

   If the spike fails, use one context for each format and record the switch cost.

Spec impact: a new requirement that GGUF Q4_K and Q6_K tensors run without a lossy requantization (provisional
`OPEN-QUANT-GGUF-KQ`). It modifies the `std_perm_gguf` behavior stated under the pool ops. OPEN-QUANT-Q4K (q4nx
containers) does not change; q4nx is deprecated.

## Phase 3: one weight image (#2)

Today the `fast` route makes a bfp16 copy of each projection at load (`bfp16_dit` ops, 9 bits for each weight),
and the q4_1 pools for decode stay. On Qwen3-8B the cost is 3.5 GiB of peak memory, and load takes 19-24 s against 14-16 s for `lean`.

1. **Kernel.** A prefill GEMM that reads the decode pool chunks and unpacks them to bfp16 on the array, then runs
   the `dit_gemm` arithmetic. Two layouts to try, in this order:
   - unpack inside each GEMM core before the mmul;
   - a separate unpack core role that feeds the GEMM cores through the memory tile (ArtLite's method).

   Check program memory, L1 plus stack, and DMA/BD budgets (xdna2-notes R-0001) before each build.
2. **Exact values.** The load-time packer rounds q4_1 values to bfp16 with round to nearest. The unpack must use
   the same rule. Then the logits are byte-identical to `fast`. That is the main gate.
3. **Engine.** Drop the `bfp16_dit` op for this route. Peak memory and load time must match `lean`.
4. **Formats.** If phase 2 changes the pool format, the unpack must read that format too. Do phase 2 steps 1 to 3
   first, so the pool format is known.

Gates: byte-identical logits against `fast`; prefill within 5% of `fast` at 512 to 4K; peak RSS and load time
within 5% of `lean`.

Spec impact: modifies OPEN-PREFILL-GEMM8 (no bfp16 copy, new memory peak) and the manifest contract for the
route (provisional `OPEN-PREFILL-ONE-IMAGE`). Sets exported before the change keep the version-7 DIT mapping.

## Phase 4: loop variant search (#1)

This phase changes kernels only. It can run in parallel with phases 1 to 3.

1. **Tool.** `variant_search.py` takes one kernel TU and a list of loop variants (unroll counts and the other
   loop pragmas that Peano accepts). It compiles each variant through the phase 0 compiler census and ranks
   them by bundles times trip count. It refuses a variant that overflows program memory or the stack.
2. **Targets.** The hot kernels from the phase 0 baseline, first the loops furthest above their MII.
3. **Accept.** Hardware A/B at 512 to 4K in alternated rounds, and byte-identical output. A pragma does not change
   arithmetic order in Peano without fast-math. Check this on the first kernel.

Spec impact: none, if the output is byte-identical.

## Order

1. Phase 0 (all four steps).
2. Phase 1 steps 1 to 3, and phase 2 step 1. These are the two short measurements that decide the most.
3. Phase 2 steps 2 to 4, then phase 3. Phase 1 step 4 only if its gate passes.
4. Phase 4 runs beside the others when a kernel is not under change.

## Verification and evidence

- Every arithmetic change: an independent numerical gate and a model-quality gate (NLL, greedy tokens), as in AGENTS.md.
- After each engine change, update `benchmarks/reference/engine-changes.json`.
- Record each measured result in xdna2-notes `raw/open-kernels/` and in this plan. Record failed approaches too.
- Do not commit weights, generated kernels, captures or bulky logs.

## OFLM side

- If Unsloth Q4_K_M becomes the shipped Qwen3-8B file: its `src/model_list.json` entry (quant, files, context).
- Any new manifest feature (bricks, one image): OFLM's manifest reader must accept it or refuse the set.

## Open questions

- A Linux box for phase 1 step 5. Without one, that step stays open.
- Qwen3-8B as the test bed. Inkling 2 Mini is the active port, but its routes still change. The results apply to it after it settles.
