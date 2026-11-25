"""Generate and verify a real-GGUF fixture for the Inkling expert GEMM."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_pool import IQ4_NL_BYTES, IQ4_NL_VALUES, read_gguf  # noqa: E402
from recipes.gptoss_pack import bf16_bits, bf16_value  # noqa: E402
from recipes.inkling2_block import pack_activations, pack_weights, unpack_output  # noqa: E402


def reference_weights(blocks: np.ndarray, iq4: bool) -> np.ndarray:
    """Decode GGUF directly, independent of the packed tile permutation."""
    n = blocks.shape[0]
    scales = np.ascontiguousarray(blocks[..., :2]).view("<f2").astype(np.float32).reshape(n, -1)
    qs = blocks[..., 2:]
    codes = IQ4_NL_VALUES[np.concatenate((qs & 15, qs >> 4), axis=-1)] if iq4 else qs.view(np.int8)
    return (codes.astype(np.float64) * bf16_value(bf16_bits(scales))[..., None]).reshape(n, -1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("expert", type=int)
    parser.add_argument("build", type=Path)
    parser.add_argument("--rows", type=int, default=32, choices=(16, 32, 64))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        ref = np.load(args.build / "expected.npy")
        actual = unpack_output(np.fromfile(args.build / "output.bin", dtype="<f4"), *ref.shape)
        error = np.linalg.norm(actual - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(f"maximum per-token relative L2 {error.max():.8g}")
        if not np.isfinite(error).all() or error.max() > 1e-4:
            raise SystemExit("expert block differs from the independent dot products")
        return

    _, tensors, base = read_gguf(str(args.model))
    info = tensors[args.tensor]
    k, n, experts = info["dims"]
    if n != 2304 or info["type"] not in ("IQ4_NL", "Q8_0") or not 0 <= args.expert < experts:
        parser.error("expected a 2,304-output expert tensor and an in-range expert ID")
    iq4 = info["type"] == "IQ4_NL"
    block_size = IQ4_NL_BYTES if iq4 else 34
    plane_bytes = n * (k // 32) * block_size
    offset = base + info["offset"] + args.expert * plane_bytes
    blocks = np.memmap(args.model, dtype=np.uint8, mode="r", offset=offset, shape=(n, k // 32, block_size))
    w = pack_weights(blocks, iq4)
    x = np.random.default_rng(57).standard_normal((args.rows, k), dtype=np.float32)
    x *= np.exp2(np.linspace(-4, 4, args.rows, dtype=np.float32))[:, None]
    a, values = pack_activations(x)
    np.save(args.build / "expected.npy", values @ reference_weights(blocks, iq4).T)
    w.tofile(args.build / "weights.bin")
    a.tofile(args.build / "activation.bin")
    output_bytes = args.rows * n * 4
    (args.build / "run.cfg").write_text(
        "device\nxclbin G final.xclbin\nkernelx k G insts.bin\n"
        f"buf weights {w.nbytes} weights.bin\nbuf activation {a.nbytes} activation.bin\nbuf output {output_bytes}\n"
        + "run k weights activation output\n" * 6
        + f"dump output output.bin {output_bytes}\n",
        encoding="ascii",
    )
    print(f"{info['type']} M={args.rows} N={n} K={k} expert {args.expert}, {w.nbytes} B")


if __name__ == "__main__":
    main()
