"""Build and verify a same-context test of the four routed expert streams."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_pool import IQ4_NL_BYTES, IQ4_NL_VALUES, read_gguf  # noqa: E402
from recipes.gptoss_pack import bf16_bits, bf16_value, mx4_tiles, q8_0_codes, q8_tiles  # noqa: E402


EXPERT_IDS = (11, 127, 0, 6, 65, 20, 30, 4)
WIDTH = 2304
STREAMS = (
    ("iq4-up", 0, "ffn_gate_up_exps", "IQ4_NL"),
    ("q8-down", 23, "ffn_down_exps", "Q8_0"),
    ("iq4-down", 0, "ffn_down_exps", "IQ4_NL"),
    ("q8-up", 23, "ffn_gate_up_exps", "Q8_0"),
)


def pack_case(model, offset, info, rng, root, name):
    cols, rows, experts = info["dims"]
    assert rows == WIDTH and experts == 128
    compact = info["type"] == "IQ4_NL"
    k = (cols + 255) // 256 * 256
    block_bytes = IQ4_NL_BYTES if compact else 34
    blocks = np.memmap(model, dtype=np.uint8, mode="r", offset=offset,
                       shape=(experts, rows, cols // 32, block_bytes))
    activation = np.zeros((len(EXPERT_IDS), WIDTH), np.float32)
    activation[:, :cols] = rng.standard_normal((len(EXPERT_IDS), cols), dtype=np.float32)
    bits = bf16_bits(activation)
    expected = []
    weights = root / f"{name}-weights.bin"
    with weights.open("wb") as sink:
        packed_bytes = 0
        for slot, expert in enumerate(EXPERT_IDS):
            plane = blocks[expert]
            if compact:
                scales = np.ascontiguousarray(plane[..., :2]).view("<f2").reshape(rows, -1).astype(np.float32)
                nibbles = plane[..., 2:]
                indices = np.concatenate((nibbles & 15, nibbles >> 4), axis=-1)
                packed = mx4_tiles(indices.reshape(rows, cols), scales, rows, k)
                codes = IQ4_NL_VALUES[indices]
            else:
                codes, scales = q8_0_codes(plane, rows, cols)
                packed = q8_tiles(codes, scales, rows, k)
            packed_bytes = packed.nbytes
            packed.tofile(sink)
            rounded = codes.reshape(rows, -1, 32) * bf16_value(bf16_bits(scales))[..., None]
            expected.append(rounded.reshape(rows, cols).astype(np.float64) @ bf16_value(bits[slot, :cols]))
    bits.tofile(root / f"{name}-input.bin")
    np.save(root / f"{name}-expected.npy", np.stack(expected))
    return packed_bytes * len(EXPERT_IDS), bits.nbytes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("builds", type=Path, help="parent of the four routed-* build directories")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    output_bytes = len(EXPERT_IDS) * WIDTH * 4
    if args.verify:
        failed = False
        for name, *_ in STREAMS:
            expected = np.load(args.builds / f"{name}-expected.npy")
            output = np.fromfile(args.builds / f"{name}-output.bin", dtype="<f4").reshape(expected.shape)
            error = np.linalg.norm(output - expected, axis=1) / np.linalg.norm(expected, axis=1)
            print(f"{name}: maximum relative L2 {np.max(error):.8g}")
            failed |= not np.all(error <= 1e-4)
        if failed:
            raise SystemExit("routed projection differs from the reference")
        return

    _, tensors, base = read_gguf(str(args.model))
    rng = np.random.default_rng(4057)
    config = ["device", "xclbin G routed-iq4-up/final.xclbin", f"buf output {output_bytes}"]
    for name, layer, tensor, quant in STREAMS:
        info = tensors[f"blk.{layer}.{tensor}.weight"]
        assert info["type"] == quant
        w_bytes, x_bytes = pack_case(args.model, base + info["offset"], info, rng, args.builds, name)
        config += [f"kernelx {name} G routed-{name}/insts.bin", f"buf {name}-weights {w_bytes} {name}-weights.bin",
                   f"buf {name}-input {x_bytes} {name}-input.bin"]
    for name, *_ in (*STREAMS, STREAMS[0]):
        config += [f"run {name} {name}-weights {name}-input output", f"dump output {name}-output.bin {output_bytes}"]
    (args.builds / "routed.cfg").write_text("\n".join(config) + "\n", encoding="ascii")


if __name__ == "__main__":
    main()
