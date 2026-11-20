# Granite 4.2 3B: prefill and decode to the NPU's floor

**Done 2026-11-20.** The user accepted the measured bounds below; commit and PR.

Goal (user, 2026-11-19): as fast as the NPU can go, not as fast as the GPU. Spec: OPEN-FAMILY-GRANITE's production bar.
Kernel and engine side only.

## Floors (computed from R-0009 / R-0012 rates; the bar is the floor, not Vulkan)

| Phase | Start (2026-11-19) | Now (`g131072_v4`) | Floor |
|---|---|---|---|
| Decode, depth 0 | 41.8 ms | 41.7 ms | 2.13 GB of weights at ~52 GB/s: ~41 ms |
| Decode at 8k | 73.2 ms | 63.3 ms | ~54 ms (the `ATTN_NULL` stream: 58.4) |
| Decode at 32k | 167.1 ms | 128.2 ms | ~93 ms (the `ATTN_NULL` stream: 107.7) |
| Prefill at 2048 | 248 tok/s | 727 tok/s | NPU-bound: GEMMs at the bfp16 rate (~11 TFLOPS), `attn_fa` ~1.3 TFLOPS |
| Prefill at 32k | 103 tok/s | 162 tok/s | `attn_fa` is 85 %; QK with hi / lo Q at ~70 % of the vector MAC rate, PV at 100 % |

## Done

1. **An accurate bfp16 route as the default** (OPEN-PREFILL-BFP16-QKV): CPU emulation put the bfp16 route's error in
   qkv; qkv runs as one bfp16 GEMM over 3 K ([hi | lo | hi] x [hi | hi | lo]). Needle matrix unchanged.
2. **Decode attention** (OPEN-ATTN-HI-ONLY): bf16 q and p, ATTN_EXPA 3's exponential; `dxl` skips the zero halves.
   The window ring depth and the score-loop unroll were tried and changed nothing or made it slower.
3. **`attn_fa`**: exp2c and a 2-key loop (OPEN-PREFILL-ATTN-FA). Single Q was faster but lost the 64k needle.
4. **Host**: the three-term pack's block decode without `std::ldexp`; norms by reciprocal multiply.
5. Exporter: probe variables now change each build's reuse key.
6. **Short prompts**: a window of 512 rows or fewer takes the host attention (`OFLM_OPEN_ATTN_FA_MIN`): 256 tokens
   494 -> 350 ms, 512 tokens 746 -> ~715 ms; logits corr >= 0.99994 against the NPU attention, greedy identical.

## Measured and not shipped

- **Matrix scorer** (`ATTN_G5_MMUL`): `dx` 120.9 ms at 32k against 128.1; with `dxl`'s matching scorer ROWS PASS and
  SPEC IDENTICAL, but `dxl`'s verify pass is 17-34 % slower (its cores mix two kv heads), so prompt lookup serves
  copy-heavy replies slower. Off by default.
- Single Q in `attn_fa`: 1.14x on the kernel, lost the 64k needle.
- Deeper KV window ring, smaller score unroll: no gain or slower.

## Bounds on what is left (spec: OPEN-FAMILY-GRANITE, "What is left")

1. Decode past 8k: 128.1 ms at 32k = the KV stream (107.7, ~41 GB/s on 4 shim channels: F-0199 puts one channel at
   ~12 GB/s, and only 5 columns have a free MM2S channel) + ~20 ms of arithmetic. A `dxl` whose cores follow kv heads
   would let the matrix scorer ship (~6 % at 32k); the array has too few cores for that layout today.
2. `attn_fa` QK at ~69 % of the vector MAC rate with the hi / lo Q the needle needs.
3. Two context switches a layer in prefill (~200 ms of a 512-token prompt); the programs cannot share one context.

## Spec impact

New: OPEN-PREFILL-BFP16-QKV, OPEN-ATTN-HI-ONLY. Modified: OPEN-PREFILL-ATTN-FA (exp2c, single Q as an option),
OPEN-FAMILY-GRANITE (production bar numbers). `ATTN_G5_MMUL` adds a requirement if it ships.
