# Qwen3.6 decode: stage hand-offs on chip (plan, not started)

Status 2026-11-27: written for a go / no-go. Nothing here is built or measured unless it says so.

## Where the step is

Shipped set `kernels-fa262k` (OPEN-DECODE-STREAM-UG, OPEN-LMHEAD-Q6): 71.3 ms a step against a 56 ms stream floor
(2.9 GB at 52 GB/s). From the audit (spec.md, "What is left", OPEN-FAMILY-QWEN36MOE-GGUF):

| part | per step | over its stream floor | what holds it (measured) |
|---|---|---|---|
| `lx0` x30 | 33.4 ms | ~5.7 ms | DDR hand-offs between dependent stages; arithmetic is hidden (`LX_NULL_*`) |
| `ax0` x10 | 8.6 ms | ~2.2 ms | as `lx0` |
| `lx1` / `ax1` x40 | 14.9 ms | ~3.1 ms | the hidden's DDR round trip between up / gate and down |
| head + tail | ~12 ms | ~3 ms | entering the head's context and the layer context again (both small levers measured: no gain) |

Ruled out on the way (measured, spec.md): deeper weight rings, DeltaNet issued beside the glue, four head cores a
column, an empty dispatch to pre-switch the context, the final norm off the NPU.

## The hand-offs (UX_SEQ issue order, `xlayer.linear_sequence_seq`)

Each is a drain that must land in DDR, a wait in the instruction stream, then a fill:

1. entry norm (`ln`, Tile(0, 3)) -> `xn` in DDR -> `x` fill to the 8 main cores;
2. `qkv` drains -> DDR -> the glue's `gact` fills (Tile(2, 4));
3. the glue's records -> DDR -> each main core's `w` channel, round by round (DeltaNet);
4. per DeltaNet round, `o` -> DDR -> post's `pin` (Tile(1, 3)); post's `og` -> DDR;
5. `og` -> `x` fill for the out projection;
6. out drains -> DDR -> `lni` for the post-norm and router; router -> DDR -> the host (`moeroute2`).

`ax0` has 1, 5, 6 and the attention cores' `ain` / `aout` in place of 2-4. The MoE parts have the hidden: up / gate
`y` drains -> DDR -> `x` fill for down, once per routed slot.

## Constraints (from ux.py and the audit)

- Each main core's two input channels are taken: `w` (memtile ring -> core) and `x` (broadcast from shim (1, 0)).
- Shim budget: 14 of 16 fills, 15 of 16 drains in use.
- Main cores: 128 B of program memory free. The core programs cannot grow; any change has to be in the DMA plumbing
  (memtile BDs, shim BDs) or in the helpers' cores.
- Every core must receive the same bytes in the same order (the logits-unchanged rule every stream knob follows).

## Candidate: the `x` broadcast from a memtile with a fixed source schedule

`x` reaches the main cores as one broadcast from the shim. If it instead left memtile (1, 1)'s MM2S, that memtile could
take its input from more than one S2MM channel: the shim stream as now, plus `lno` from the norm helper and `pout`
from post (and, in the MoE parts, a join of the cores' `y`). The order of `x` elements in a layer is fixed at compile
time, so a static BD chain on the memtile MM2S can pick each element's source in that order. It removes hand-offs
1 and 5 (and the hidden's round trip in the MoE parts) without touching a main core's program.

Open questions, in the order they would be answered:

1. Can IRON place a memtile buffer fed by two producers on different channels in a fixed alternation? ObjectFifo
   `join` concatenates per object, it does not alternate over time; this likely needs the placed dialect
   (`aie.memtile_dma` with hand-written BD chains and locks), which no design in this repo uses yet.
2. Memtile (1, 1)'s channel budget with the weight ring already on it (channels 0 in, 1 out for `wm1`).
3. What one hand-off costs, measured, before building: a probe that removes ONE wait (e.g. hand-off 1 with `xn`
   pre-staged from the previous layer, timing only) gives the per-hand-off price and so the ceiling.

## Estimate

- Ceiling if every hand-off in 1, 5 and the MoE hidden went on chip: on the order of the MoE's ~3 ms plus part of
  `lx0` / `ax0`'s ~8 ms; inferred, not measured.
- Effort: the probe in question 3 about half a day; a placed-dialect memtile schedule, all four streams rebuilt, the
  verify image (`vux`) twin and the logits gates: several days.
- Risk: the placed dialect is new to this repo; the MoE parts' `moeroute2` instruction-stream patching must keep
  working on the changed streams.

## Asks

Go / no-go from the user. If go: question 3 first, and stop if one hand-off costs too little to pay for the rest.
