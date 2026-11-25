"""Pack one real GGUF expert projection and emit an NPU harness case."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_pool import IQ4_NL_BYTES, IQ4_NL_VALUES, dequant_iq4_nl, read_gguf  # noqa: E402
from recipes.gptoss_pack import bf16_bits, bf16_value, mx4_tiles  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("expert", type=int)
    parser.add_argument("build", type=Path, help="directory containing final.xclbin and insts.bin")
    parser.add_argument("--verify", action="store_true", help="check output.bin after running the XRT harness")
    args = parser.parse_args()
    _, tensors, base = read_gguf(str(args.model))
    info = tensors[args.tensor]
    cols, rows, experts = info["dims"]
    if info["type"] != "IQ4_NL" or not 0 <= args.expert < experts:
        parser.error("expected an IQ4_NL expert plane and an in-range expert ID")
    expert_bytes = rows * (cols // 32) * IQ4_NL_BYTES
    offset = base + info["offset"] + args.expert * expert_bytes
    blocks = np.memmap(args.model, dtype=np.uint8, mode="r", offset=offset,
                       shape=(rows, cols // 32, IQ4_NL_BYTES))
    scales = np.ascontiguousarray(blocks[..., :2]).view("<f2").reshape(rows, -1).astype(np.float32)
    packed = blocks[..., 2:]
    indices = np.concatenate((packed & 15, packed >> 4), axis=-1)
    n, k = (rows + 63) // 64 * 64, (cols + 255) // 256 * 256
    weights = mx4_tiles(indices.reshape(rows, cols), scales, n, k)
    x = np.zeros(k, np.float32)
    x[:cols] = np.random.default_rng(45).standard_normal(cols, dtype=np.float32)
    x_bits = bf16_bits(x)
    x_bf16 = bf16_value(x_bits[:cols]).astype(np.float64)
    scales_bf16 = bf16_value(bf16_bits(scales))
    rounded = (IQ4_NL_VALUES[indices] * scales_bf16[..., None]).reshape(rows, cols)
    expected = rounded.astype(np.float64) @ x_bf16
    source = dequant_iq4_nl(blocks).astype(np.float64) @ x[:cols].astype(np.float64)
    if args.verify:
        output = np.fromfile(args.build / "output.bin", dtype="<f4")
        if len(output) != n or not np.isfinite(output).all() or np.any(output[rows:]):
            raise SystemExit("unexpected output size, non-finite values, or nonzero padding")
        compare = output[:rows].astype(np.float64)
        kernel_error = np.linalg.norm(compare - expected)
        source_error = np.linalg.norm(compare - source)
        kernel_norm, source_norm = np.linalg.norm(expected), np.linalg.norm(source)
        print(f"relative L2 vs packed BF16 inputs {kernel_error / max(kernel_norm, 1e-30):.8g}, "
              f"vs source GGUF and FP32 input {source_error / max(source_norm, 1e-30):.8g}")
        if kernel_error > 1e-4 * kernel_norm or source_error > 1e-2 * source_norm:
            raise SystemExit("IQ4_NL GEMV differs from the reference")
        return
    weights.tofile(args.build / "weights.bin")
    x_bits.tofile(args.build / "activation.bin")
    np.savez(args.build / "expected.npz", rounded=expected, gguf=source, output_rows=n)
    (args.build / "run.cfg").write_text(
        "device\nxclbin G final.xclbin\nkernelx k G insts.bin\n"
        f"buf weights {weights.nbytes} weights.bin\n"
        f"buf activation {x_bits.nbytes} activation.bin\nbuf output {n * 4}\n"
        f"run k weights activation output\ndump output output.bin {n * 4}\n"
    )
    print(f"N={n} K={k} expert {args.expert}, {weights.nbytes} packed bytes")


if __name__ == "__main__":
    main()
