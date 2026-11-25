# ArtLite speed techniques: kernels, memory and submission

2026-11-24. Read this note before you plan prefill layouts, weight formats or the submission flow. It records what ArtLite's documents state, what we have today, and which layer owns each change.

## Evidence

- Source: ArtLite `8167fe88ca88` (upstream `main`, 2026-11-24), read from a local checkout. Files: `README.md`, `docs/design.md`, `docs/install.md`, `docs/probe.md`, `docs/page-local-bricks.md`, `docs/models/llama3.md`, `docs/models/qwen35.md`.
- Class: documented. Each ArtLite number is ArtLite's own measurement on ArtLite's machines. We ran nothing for this note.
- The first version of this list was a secondhand summary. The check against the source changed two items. They are marked **Corrected**.
- "Our state" entries are from our code and spec at open-kernels `1b9d39c` plus the uncommitted Inkling 2 work. One entry is marked inferred.
- Earlier first-hand notes: [prefill code audit](../artlite-prefill-code-audit.md) (ArtLite `a07a7b88`) and [baseline readiness](../artlite-baseline-readiness.md).
- ArtLite's license is LGPL-2.1-or-later. Use the techniques. Do not copy the code.

## What ArtLite states

### Submission: transaction against ELF

- ArtLite has two backends. One calls the `amdxdna` driver directly (Linux). The other uses XRT. XRT is the only backend on Windows and is optional on Linux (`docs/design.md:43-54`).
- On XRT, the control program of a fused ELF runs as a transaction. The firmware runs a control program more slowly as a transaction than as an ELF. The cost is about 30 us more per dispatch for a small fused ELF. Prefill is 2-6% slower. The two Linux backends are level otherwise.
- Dispatch round trip in `artlite-probe`: 49 us on Strix Halo, 56 us on Krackan Point (both Linux), 94 us on Strix Point through Windows and XRT. The `turbo` power mode cut the Strix Point figure to 58 us (`docs/probe.md`, `docs/install.md:110-114`).

### Shape-specialized code

- Each operator renders C++ for its shapes, data formats and position on the array. It also renders the IRON design for the data movement. Peano optimizes the C++ with hints chosen for each loop (`README.md:22-26`).
- Two tools support the hints. `measure_ii` reports the initiation interval, spills, code bytes and stack of each loop. `variant_search` ranks pragma and rewrite variants for each loop (`docs/development.md:118-132`, `docs/code-map.md:26`).

### One packed weight image

- Decode and prefill read the same weight image (`docs/models/llama3.md:43`, `docs/models/qwen35.md:85`). The weights stay in tile formats that match the GGUF types. The cores unpack them.
- The prefill GEMM cores unpack the tiles into bfp16ebs8 panels on the array. There is no second weight copy in DRAM. The [prefill code audit](../artlite-prefill-code-audit.md) gives the core roles.
- Load time: a page-cached 2 GB image loads in about 34 ms with parallel reads and non-temporal stores, against 124 ms for `read()` (`docs/design.md:116-119`).

### Format kernels and code pages

- There is one tile format for each GGUF type: Q2_K to Q6_K, Q4_0, Q4_1, Q8_0, the IQ1 to IQ4 types, NVFP4 and BFP16 (`docs/design.md:216`).
- A 9280-byte tile holds 16 IQ4_XS rows, 20 IQ3_S rows, 21 Q3_K rows or 23 IQ3_XXS rows (`docs/models/qwen35.md:163-171`). Fewer bits per weight put more rows in each tile.
- For the IQ types, the cores read grid values with gathers from table copies in core memory.
- A model with ten tile formats loads the kernel for each format as a code page. An IQ page brings its lookup table (`docs/models/qwen35.md:173-174`). The method is in ArtLite's `docs/paging.md` and in xdna2-notes `raw/phlegm/2026-10-20-artlite-code-paging-for-16kb-program-memory.md`.
- Hazard: a switch to the context of a different process loses the page store. A device suspend after 5 s of idle time also loses it. The runtime writes the pages again (`docs/design.md:138`, `docs/design.md:320`).

### Whole-line DMA writes

- DRAM is written in 64-byte lines. A row that starts or ends inside a line is a masked write. LPDDR5 runs a masked write as a read-modify-write inside the DRAM.
- Probe result, 16 channels, 128-byte rows (whole lines) against 132-byte rows (`docs/page-local-bricks.md:33-48`):

| Machine | Whole lines (GB/s) | Partial end lines (GB/s) | Ratio |
|---|---:|---:|---:|
| Krackan Point, Linux | 73.2 | 15.6 | 4.7 |
| Strix Halo, Linux | 57.2 | 41.4 | 1.38 |
| Strix Halo, Linux, 4 KB pages | 12.2 | 12.1 | 1.0 |
| Krackan Point, Windows | 6.0 | 5.4 | 1.1 |
| Strix Point, Windows | not measured | not measured | |

- The fix: each GEMM pass writes windows of whole lines. Some columns are written two times. The cost is 3-9% more bytes written.
- **Corrected:** the summary gave only the 4.7 ratio. The source also states that Strix Halo prefill speed did not change on five models after the fix. On Windows the two row types are almost level, because the page cost below is the limit.
- The stated cause is specific to LPDDR5. ArtLite's Strix Point machine has DDR5 SO-DIMMs and has no measured write rows.

### Pages

- The NPU translates the address of each memory page that a DMA row touches. A row on a new page is one page visit. The limit is one for the full NPU, not one for each channel. One translation cache holds about 500 to 1000 pages.
- Linux: ArtLite host buffers of 2 MB or more are process memory on 2 MB pages (transparent huge pages set to `madvise` or `always`), pinned as userptr buffers. A buffer that the driver allocates has 4 KB pages (`docs/design.md:111-116`).
- Windows: the NPU gets 4 KB pages only. Process large pages (`MEM_LARGE_PAGES`) changed nothing on Krackan Point. The limit is 45 to 80 million page visits a second. Strix Halo with 4 KB pages gets 140 to 170 million (`docs/page-local-bricks.md:65-118`).
- Strix Point on Windows: a read stream runs at 53.2 GB/s. A gather of 256-byte rows on different pages runs at 14.0 GB/s (`docs/page-local-bricks.md:33-39`).
- Cost example: a 1024-token prefill chunk of Llama 3.2 1B makes 7.5 million page visits. That is 95 to 150 ms on Windows. The work of the chunk takes about 105 ms on Strix Halo. ArtLite states that this is most of its 2x prefill gap on Windows (`docs/page-local-bricks.md:120-131`).
- Decode reads the weights of each column as one stream and pays little. Weights on 2 MB userptr pages made decode 0.8% slower, and ArtLite rejected that change for decode (`docs/design.md:328`).
- Proposed and not implemented: page-local bricks. A brick is a 4 KB block that holds a small group of tokens across a range of features, so the rows of one DMA task share pages. A driver request for 2 MB pages on Windows is open as amd/xdna-driver#1817.
- **Corrected:** the summary presented 2 MB pages as a technique to adopt. On Windows no layer can get them today. The Windows remedy is a layout change.

## Our state and the owner of each change

Three layers exist. Phlegm is the Rust CLI, server, sampler and bench, plus a JSON bridge over `Core`. The engine is `src/open_qwen36/`. The kernels are `open_kernels/designs/`, `open_kernels/recipes/` and the exporters. The engine and the kernels are both in open-kernels.

| Technique | Our state | Owner |
|---|---|---|
| ELF submission | The engine registers an xclbin and runs `xrt::kernel(ctx, "MLIR_AIE")` with a cacheable instruction buffer ([core.cpp:637-654](../../../src/open_qwen36/core.cpp#L637-L654), `inkling2_npu.cpp`). Inferred: this is the transaction flow. A full ELF ran on Windows XRT 2.21 in a spike (xdna2-notes F-0148, F-0149). | Engine and exporter (`open_kernels/compose_elf.py`) |
| Shape-specialized code | Designs are written by hand with compile-time parameters. Recipes compose them for each family. The designs carry about 470 `#pragma clang loop` unroll hints. `open_kernels/kernel_remarks.py` reads compiler remarks. There is no variant search for each loop. | Kernels |
| One weight image | Decode reads packed pools. The dense `fast` prefill route makes a second bfp16ebs8 copy at load, at 9 bits for each weight. Qwen3-8B peaks at 13.3 GiB against 9.8 GiB for `lean` (OPEN-PREFILL-GEMM8 in [spec.md](../spec.md)). | Kernels and engine, one contract |
| Format kernels | Native kernels exist for q4_1, q8, ternary T2, MXFP4 (GPT-OSS experts) and IQ4_NL (Inkling 2 experts). The reader changes other K-quant and IQ rows to a pool format on the host ([gguf_file.cpp:595](../../../src/open_qwen36/gguf_file.cpp#L595), [pools.cpp:1069](../../../src/open_qwen36/pools.cpp#L1069)). Saluki's i-quant file runs at q8 (OPEN-FAMILY-SALUKI). Decode then streams more bytes for each weight than the file holds. | Kernels and engine, one contract |
| Code pages | None. The main cores of the merged layer image use 16256 of 16384 bytes of program memory (xdna2-notes F-0104). | Kernels and engine |
| Whole-line writes | Not counted. | Kernels and manifest layouts |
| Pages | Each data buffer is `xrt::ext::bo(device, size)`, which is driver memory ([core.cpp:697-763](../../../src/open_qwen36/core.cpp#L697-L763)). The rows and page visits of our prefill streams are not counted. | Linux: engine plus a system setting. Windows: kernels and engine layouts |

All seven rows are open-kernels work. Loop hints change only the kernels. 2 MB pages on Linux change only the engine. Each other row changes the kernel writer and the manifest reader, so the two sides change in one commit ([AGENTS.md](../../../AGENTS.md#direction)).

## What Phlegm can do

Phlegm holds no XRT buffer and no kernel. It cannot change memory traffic. Its part is small:

- On Linux, read `/sys/kernel/mm/transparent_hugepage/enabled` at start and warn if it is `never`. Record the page size in bench provenance. This helps only after the engine wraps its own memory.
- Set the NPU power mode at the start of `run` and `serve`. ArtLite's CLI and server raise `default` to `performance`. `phlegm bench` only reads the mode today. Follow xdna2-notes R-0006 before a change.
- Offer the quant file that has a native kernel in the model catalogue.
- Record the page size, power mode and submission flow with each bench result.

## Checks before any work

Do these in order. Each one is short. None needs a long NPU run.

1. Count the buffer descriptors of one prefill block without the device. For each stream, count rows, rows that start or end inside a 64-byte line, and 4 KB page visits. ArtLite counted the same data from its dispatched control code. This check sets the value of the whole-line and page items.
2. Measure a stream against short rows on Strix Point with Windows. A design with no cores is sufficient. The run takes about one minute. This fills the cells that ArtLite did not measure.
3. Run a paired trial of one decode stream as an ELF and as a transaction at 512 to 4K tokens. Multiply the difference by the dispatch count of a token.
4. Count the bytes that decode streams for each token against the bytes of the same tensors in the GGUF, for each model. This sets the value of native format kernels.

Do not carry ArtLite's speeds to our machines. Record each result in the model plan and add the lesson here.
