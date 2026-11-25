---
name: phlegm-prefill-profile
description: Diagnose ordinary Inkling 2 Mini block-prefill performance in Phlegm using the tracked open-kernels profiler. Use for slow prompt ingestion, grouped expert or host-stage attribution, and stage-level comparisons at fixed prompt lengths.
---

# Profile Inkling prefill through Phlegm

Use the open-kernels checkout in the current workspace or locate the user's checkout under `C:/code`. Read `AGENTS.md` and `benchmarks/reference/inkling2-prefill/README.md` in that checkout.

- Use the tracked `stage.py` and `run.py`. Build the instrumented protocol-2 Phlegm bridge in an isolated source directory. Keep the bridge sidecar, kernel bundle, GGUF and fixed prompt IDs tied to the result. Leave generated builds and captures in `.plans/`.
- Start with a short complete run such as 512 tokens, then check the workload the user reported. Separate ordinary prompt ingestion from maximum-context attention timing. With an occupied NPU, run `--prepare-only` and keep the live owner intact.
- Compare `prefill_wall_seconds`, the block sequence, and aggregate stage counters before choosing a change. `block`, `moe_total`, and `group_total` are nested totals. In the expert breakdown, distinguish device wait, activation and instruction sync, readback, finite scan, and unpack. Check host gather, SiTU, reduction, mixing, and convolution separately.
- For relative attention, compare sliding and full dispatches, query and K/V uploads, and partial-result merge. These counters are nested inside `attention_ms`. Keep the prompt position and block count with each stage share.
- For an uncached, single HTTP recall request, use `summarize.py` on the server log to check contiguous progress and compare rates by context interval. Check the live process and Windows power events separately from profiled time.
- For a speed claim, pair warm controls in both orders, verify logits and execution route, and check drift and Windows power or PCI events. Report stage timings as stage timings. Use Phlegm's normal application path and release benchmark for the production gate.
