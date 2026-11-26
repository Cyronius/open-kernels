# Inkling 2 Mini handoff (2026-11-26)

Use this file to continue the Inkling 2 Mini production-bar work on a different machine. The full lab record is
[the production-bar plan](../inkling2-mini-production-bar.md). The port lessons are in
[the port notes](../../notes/inkling2-mini-port.md). Read [AGENTS.md](../../../../AGENTS.md) first.

## Where the work is

Branch `feat/inkling2-integration` on `origin` holds all the work. Nothing is merged into `main`. No PR is open.

| Commit | Change |
| --- | --- |
| `a6b8e3f` | Engine, kernels and evidence. Dense decode projections on the NPU by default (singleton image, shared Q8 prefill pool) |
| `29e366b` | Verify passes of 9 to 64 rows go through the block route (`wide_verify_limit`), for long prompt-lookup drafts |
| `c42e8d1` | Row-mode attention packs and syncs only the K/V chunks before the key end (bit-equal) |
| `8e48f1e`, `26e2c17` | Loop controller on the NPU for prefill blocks of 64 rows or more (`controller.xclbin`) |
| `85ba8fa`, `6dba996` | SiTU in the singleton expert kernel: one launch per decode visit |
| `9f293d6`, `2db70c2` | SiTU in the grouped expert kernel for verify rows, so verify is bit-equal to decode again |

The local branches `feat/inkling2-production-bar`, `feat/inkling2-controller-kernel` and `feat/inkling2-situ-kernel`
are only on the first machine. Their commits are all in the integration branch.

## Measured state (Strix Point, Windows 11 26200, 6 host threads)

These are the last clean numbers. Do not carry them to another machine or stack. Measure again.

| Item | Release bundle (a6b8e3f) | Integration bundle |
| --- | ---: | ---: |
| Decode token, 1,024-token prefix | 180.8 to 185.1 ms | 168.1 to 169.1 ms |
| Prefill, 1,024 tokens (256-row blocks) | 21.6 to 21.7 s | 20.4 to 20.7 s |
| Partial accept (verify against direct decode) | 20 of 20 | 20 of 20 |
| DSpark forced-acceptance matrix | 18 of 18 | 18 of 18 |
| Short greedy parity, 5-token prompt, 64 tokens | 64 of 64 | 64 of 64 |
| Decode logits against the release bundle, 256 positions | reference | 254 of 256 same top token, KL mean 0.00030 |
| Real-text NLL, same positions | 1.2716 | 1.2714 |

- Prefill logits are not bit-equal to the release bundle. The NPU loop controller changes the block arithmetic.
- Verify on Q8_0 layers runs one row at a time on the singleton stream. The grouped Q8 kernel uses a different
  scale scheme, so it cannot match singleton bits. Verify cost is the same as on the 2-launch route: 670 to 675,
  898 to 914 and 1,323 to 1,355 ms for 2, 4 and 8 rows on both routes. These runs were NOISY (2 to 6 s bursts),
  but both orders agree within 2%.
- Prefill keeps 2 launches per group with host SiTU. Fused groups in prefill were 2 to 6% slower.
  `INKLING2_TEST_FUSED_PREFILL_GROUPS=1` selects them for tests.
- `INKLING2_TEST_SEPARATE_EXPERTS=1` selects the old 2-launch expert route everywhere, for A/B runs.
- K/V upload per decode token at 512 tokens: 8.0 ms (pack 5.9, sync 2.1). At 3,584 tokens: 20.0 ms.
- Long prompt-lookup drafts (16 to 64 rows) gain only on real copying (+23 to 25%) and lose up to 14% elsewhere.
  The fixed cost of a block pass (about 1.25 s) is the cause. Keep them opt-in.
- Singleton image program memory is 15,696 of 16,384 B. The grouped image uses 11,328 B.

## Open items, in order

1. A CLEAN repeat of the verify cost pair (the first pair was NOISY; it showed no difference).
2. A profiled stage split of the integration route. The profiler hooks are not on this branch; see
   [the profiler README](../../../../benchmarks/reference/inkling2-prefill/README.md).
3. Prefill SiTU on the NPU: find why fused groups are slower (the kernel calls SiTU one row at a time, 144 calls per
   core per group) before another attempt.
4. Merge the loop controller into the dense or expert image to remove two context entries per looped visit
   (estimated 0.1 s per 256-row block).
5. K/V: a per-visit cache of packed K^T/V saves the copy, but doubles K/V memory and keeps the sync. To remove the
   sync, move the bias out of the K/V buffer (kernel layout change).
6. `benchmarks/reference/check_import.py --upstream <OFLM checkout>`: not run. The first machine had no OFLM checkout
   with the base commit `38ae3d2`.
7. Production-bar items that need the user: the 1M-context run, a Phlegm release tag and `phlegm bench` on it,
   features on the release build, a shipped drafter. Decode is still below the 6 to 7 tok/s target.

## Phlegm side

Phlegm acceptance needs Phlegm changes that are not committed anywhere. They are in
[phlegm-uncommitted.patch](phlegm-uncommitted.patch): the working-tree diff of `C:/code/phlegm` on branch
`feat/flux2-resident-worker` at `4aa1493` (on `origin`), plus the new file `phlegm-core/src/server_media.rs`.

- The patch includes the reference adapter (`benchmarks/reference/bridge.cpp`: `long_verify`, `wide_width`),
  `LongLookup` and `LongPolicy` in `speculative.rs`, the 64 MB logits limit, and earlier Inkling server, tokenizer
  and tool work.
- The patch also moves the `open-kernels` submodule to `874eb89`, a commit that exists only on the first machine.
  Apply without that hunk and point the submodule at `feat/inkling2-integration` instead:

```
git -C phlegm switch -c feat/inkling2-phlegm --no-track 4aa1493
git -C phlegm apply --exclude=open-kernels ../open-kernels/specs/open-engine/plans/inkling2-handoff/phlegm-uncommitted.patch
```

- The patch was checked with `git apply --check` against `4aa1493`.
- Phlegm `cargo test`: the speculation tests pass. `npu_lock` and `embeddings` tests failed before these changes.

## Rebuild on a new machine

Toolchain: `C:\dev\mlir-aie\iron_env.cmd`, XRT headers `C:/dev/XRT/src/runtime_src/core/include`, XRT import library
`C:/dev/xrtNPUfromDLL/xrt_coreutil.lib`, MSVC 2022 Build Tools. Use Arbiter for NPU runs, builds and measurements
(`arbiter run npu|build|--measure ...`).

1. **Model files.** From [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF/tree/e8a2cecf0e78914f07cc27b9be711a9b2c7561d9):
   `Inkling-2-Mini-IQ4_NL.gguf` (SHA-256 `cbdf3fa9…81af4`) and the BF16 mmproj (`e914997f…d9cb`). DSpark:
   Unsloth's `dspark-Inkling-2-Mini-Q8_0.gguf` (`2ec78510…3773`). Make one folder with `model.gguf`, `mmproj.gguf` and
   `dspark.gguf`. The bridge allows 8 verify rows only when `dspark.gguf` is in the folder.
2. **Kernel bundle.** The defaults build the singleton image, the fused experts and the loop controller. A full
   build took about 30 minutes on the first machine.

```
call C:\dev\mlir-aie\iron_env.cmd
cd open_kernels
python export_inkling2_kernels.py --out <bundle> --wide-old-attention --near-attention
```

3. **Bridge.** Build the Phlegm reference bridge from this repo's engine and the patched Phlegm adapter:

```
python -B specs/open-engine/plans/inkling2-handoff/tools/build_bridge.py --reference-root . --out-dir <bridge-dir> \
  --adapter-root <phlegm>/benchmarks/reference --xrt-include C:/dev/XRT/src/runtime_src/core/include \
  --xrt-lib C:/dev/xrtNPUfromDLL/xrt_coreutil.lib
```

4. **Gates.** The tools in [tools/](tools/) talk to the bridge over its protocol 2 pipe. `ids-4096.txt` is the fixed
   prompt; `five.ids` and `native64.utf8` are the short greedy parity prompt and its native-host reference.

Run from the repo root with `T=specs/open-engine/plans/inkling2-handoff/tools`.

| Gate | Command |
| --- | --- |
| Partial accept | `python -B $T/partial_accept.py --bridge <exe> --model-dir <model> --bundle <bundle> --ids $T/ids-4096.txt` |
| Decode logits | `python -B $T/decode_dump.py <out> --bridge <exe> --model-dir <model> --bundle <bundle> --ids $T/ids-4096.txt --prefix 512 --positions 256` |
| Compare two dumps | `python -B $T/bundle_gate.py <out-a> <out-b> $T/ids-4096.txt` (needs a profiled bridge; otherwise compare the `.npy` files) |
| Decode and prefill timing | `python -B $T/verify_cost.py <out> ... --prefix 1024 --threads 6 --max-rows 1 --cycles 12 --blocks` under `arbiter run --measure` |
| Verify cost, 1 to 8 rows | `python -B $T/verify_cost.py <out> ... --prefix 1024 --max-rows 8 --cycles 6 --dspark --blocks` |
| Short greedy parity | `inkling2_npu_check <bundle> <model.gguf> $T/five.ids <out.f32> 64` (CMake target in `src/open_qwen36/`), compare with `$T/native64.utf8` |

- The DSpark forced-acceptance matrix script depends on local files of the first machine. The procedure is in the
  production-bar plan under the verify matrix and OPEN-SPEC-DSPARK.
- Report every measurement as CLEAN or NOISY. Do not quote NOISY numbers as results.
