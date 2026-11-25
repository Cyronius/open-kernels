# Inkling 2 Mini prefill profiler

This profiler runs the layer-major route through Phlegm's protocol-2 reference bridge, with 256-token blocks by default. It times a fixed prompt one block at a time and captures the full final vocabulary. The staged engine adds timing scopes around the existing arithmetic and kernel calls.

## Build and run

From the open-kernels repository on Windows, use Python 3.11 or newer and set the Phlegm and XRT paths to the local checkouts and libraries.

```powershell
$profiler = 'benchmarks/reference/inkling2-prefill'
$scratch = '.plans/inkling2-prefill-profile'
$phlegm = 'C:/code/phlegm'
$xrtInclude = 'C:/dev/XRT/src/runtime_src/core/include'
$xrtLib = 'C:/dev/xrtNPUfromDLL/xrt_coreutil.lib'

python -B "$profiler/stage.py" "$scratch/source"
& "$phlegm/benchmarks/reference/build.ps1" -ReferenceRoot "$scratch/source" `
    -OutDir "$scratch/bridge" -XrtInclude $xrtInclude -XrtLib $xrtLib
```

Use Phlegm's matching `benchmarks/reference/build.py` on Linux. Keep the bridge's `.build.json` sidecar beside the executable. `stage.py` copies the canonical engine into a fresh directory and checks its source anchors before staging. If those anchors drift, update the timing placements and verify the staged source and logits against the current engine.

Pass the model directory containing `model.gguf`, the kernel bundle directory containing `manifest.json`, and a whitespace-separated file of token IDs. Use the same IDs and assets for each comparison arm.

```powershell
$inputs = @(
    '--bridge', "$scratch/bridge/phlegm-reference-bench.exe",
    '--model-dir', 'path/to/inkling-model',
    '--bundle', 'path/to/inkling-v11-bundle',
    '--ids', 'path/to/prompt.ids',
    '--prefix', '512'
)
python -B "$profiler/run.py" "$scratch/preflight-512" @inputs --prepare-only
python -B "$profiler/run.py" "$scratch/device-512" @inputs
```

Preflight validates provenance and the protocol handshake without initializing the model. A run acquires the shared `PHLEGM_NPU_LOCK` or the default `phlegm-npu.lock` in the system temp directory. Check the current owner before retrying after a lock conflict. On Windows the runner holds a scoped idle-sleep request and captures power transitions and PCI context errors. Start at 512 tokens for a quick comparison and use 8,225 tokens to check ordinary longer prefill. Choose fresh output paths for each arm.

For block-size experiments, pass `--block` with the size the experimental bridge advertises in its fast route. The profiler verifies the reported size at initialization and checks each warm-up and timed block in the trace. Use the same `--warmup-tokens` in both arms to replay an equal prefix before timing.

## Inspect a long prefill

For a fresh, single `phlegm serve` request using the instrumented bridge, read the captured server log while prefill is running.

```powershell
python -B "$profiler/summarize.py" "$scratch/serve.log" --prompt-tokens 131071
```

The summary checks contiguous blocks from position zero, excludes decode beyond the submitted prompt, and ignores a line the process is still writing. It reports completed tokens and per-interval timing, stage medians, and CPU attention fallback counts. The default interval is 4,096 tokens. Confirm that the process is still alive and use the Windows event capture to check for time spent in standby. Profiled block time excludes gaps between blocks.

## Read the result

- `input.json` records the bridge, engine, manifest and token IDs, plus model file identity and the thread count. For shared comparisons, retain the source GGUF and complete bundle digests with the result.
- `result.json` records end-to-end prefill wall time, tokens per second, per-block and aggregate stage times and counts, and the logit hash. Warm-up blocks are excluded from totals. `logits.f32` contains the final 200,058 little-endian float32 scores.
- `bridge.stderr.log` preserves the raw stage counters. `block_ms` includes its component stages. `moe_total_ms` includes routing, gather, activation, reduction and grouped launches. `group_total_ms` includes weight preparation, activation packing and sync, instruction patch and sync, configuration, wait, readback sync, finite validation, and unpack. Sum siblings within a stage to find its remaining time. Wall time also includes bridge and request overhead.
- `controller_update_ms` measures the nonlinear update inside `controller_ms`. The difference includes input normalization and projections. `attention_gate_ms` measures the host sigmoid and multiply after attention, separately from the relative-attention and dense projection interfaces.
- `layer_norm_ms` covers attention input and FFN normalization. `qkv_post_ms` covers clipping, query scaling, value normalization and BF16 KV encoding. `kv_commit_ms` covers CPU cache growth and block commit. `embed_ms` covers token gather and embedding normalization.
- `attention_host_calls` counts visits that fall back to CPU attention. `head_calls` counts logit evaluations, and `head_host_calls` records evaluations that fall back to the CPU head. `head_host_ms` is included in `head_ms`.
- `attention_kv_ms` measures key/value packing and upload. `attention_query_ms` measures query packing or header updates and upload for each tile. `attention_sliding_dispatch_ms` and `attention_full_dispatch_ms` include dispatch and readback for their respective attention visits. `attention_merge_ms` times partial-result merges. These counters are included in `attention_ms`, which also covers shared query preparation and final normalization.
- An experimental bundle exported with `--wide-old-attention` adds `attention_old_prepare_ms` for query upload and newly visible key encoding, `attention_old_upload_ms` for K/V copies and upload, and `attention_old_merge_ms` for partial merges. The counts and times for `attention_old_dispatch_4k` and `attention_old_dispatch_16k` show which older-key tiers ran. These counters are also included in `attention_ms`.
- On Windows, `power_status` records the scoped system and execution requests and the AC/battery and battery-saver status at both ends. Use `--keep-display` if display idle is triggering Modern Standby during measurements. `events` covers the bridge lifetime and `prefill_events` counts records within the timed prefill. The event log remains the evidence of standby transitions. If collection fails, `result.json` retains the measurement and `event_interval` with `events` set to `null`. Rerun `collect_windows_npu_events.ps1` over that exact interval before using the timing. [Microsoft power request behavior](https://learn.microsoft.com/en-us/windows/win32/api/winbase/nf-winbase-powersetrequest)
- Bracket performance changes in both orders with a quiet, warm control. Check the matching block sequence, finite logits and quality tolerances, then inspect power events, PCI errors and control drift before quoting a speedup. The production bar still calls for application and release benchmark evidence.
