"""Exercise grouped decode and prefill directly from the shared Inkling expert pack."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_pool import IQ4_NL_BYTES, read_gguf  # noqa: E402
from make_block_test import reference_weights  # noqa: E402
from recipes.inkling2_block import pack_activations, pack_weights, unpack_grouped_output  # noqa: E402

EXPERTS = (127, 5, 90, 43, 81, 0, 12, 66)
PREFILL_COUNTS = (16, 9, 4, 1, 13, 2, 7, 0)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("tensor")
    parser.add_argument("build", type=Path)
    parser.add_argument("--rows", type=int, default=4, choices=(1, 2, 4))
    parser.add_argument("--token-rows", type=int, default=4, choices=(4, 16))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    m = args.token_rows
    counts = (1,) * len(EXPERTS) if m == 4 else PREFILL_COUNTS
    present = np.arange(m)[None, :] < np.asarray(counts)[:, None]
    if args.verify:
        ref = np.load(args.build / "expected.npy")
        actual = unpack_grouped_output(np.fromfile(args.build / "output.bin", dtype="<f4"),
                                       len(EXPERTS), args.rows, m, ref.shape[-1])
        ref = ref.reshape(actual[present].shape)
        error = np.linalg.norm(actual[present] - ref, axis=1) / np.linalg.norm(ref, axis=1)
        print(f"maximum relative L2 across {len(error)} active rows {error.max():.8g}")
        if not np.isfinite(error).all() or error.max() > 1e-4 or np.any(actual[~present]):
            raise SystemExit("grouped projection differs from independent dot products or zero padding")
        return

    _, tensors, base = read_gguf(str(args.model))
    info = tensors[args.tensor]
    k, n, experts = info["dims"]
    if n != 2304 or info["type"] not in ("IQ4_NL", "Q8_0") or max(EXPERTS) >= experts:
        parser.error("expected a 2,304-output Inkling expert tensor")
    iq4 = info["type"] == "IQ4_NL"
    block_size = IQ4_NL_BYTES if iq4 else 34
    rng = np.random.default_rng(303)
    x = np.zeros((len(EXPERTS), m, k), np.float32)
    x[present] = rng.standard_normal((sum(counts), k), dtype=np.float32)
    x *= np.exp2(np.arange(len(EXPERTS), dtype=np.float32) - 3)[:, None, None]
    a, values = pack_activations(x.reshape(-1, k), tile_m=m)
    values = values.reshape(x.shape)
    expected = []
    with (args.build / "weights.bin").open("wb") as weights:
        for slot, expert in enumerate(EXPERTS):
            offset = base + info["offset"] + expert * n * (k // 32) * block_size
            blocks = np.memmap(args.model, dtype=np.uint8, mode="r", offset=offset, shape=(n, k // 32, block_size))
            pack_weights(blocks, iq4).tofile(weights)
            if counts[slot]:
                expected.append(values[slot, :counts[slot]] @ reference_weights(blocks, iq4).T)

    np.save(args.build / "expected.npy", np.concatenate(expected))
    a.tofile(args.build / "activation.bin")
    output_bytes = len(EXPERTS) * m * n * 4
    weight_bytes = (args.build / "weights.bin").stat().st_size
    (args.build / "run.cfg").write_text(
        "device\nxclbin G final.xclbin\nkernelx k G insts.bin\n"
        f"buf weights {weight_bytes} weights.bin\nbuf activation {a.nbytes} activation.bin\nbuf output {output_bytes}\n"
        + "run k weights activation output\n" * 6
        + f"dump output output.bin {output_bytes}\n",
        encoding="ascii",
    )


if __name__ == "__main__":
    main()
