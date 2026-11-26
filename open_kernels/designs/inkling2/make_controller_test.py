"""Generate and verify a real-GGUF fixture for the Inkling loop controller kernel."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from gguf_pool import read_gguf  # noqa: E402
from recipes.inkling2_controller import (  # noqa: E402
    BLOCK_ROWS, HIDDEN, OUTPUT_FLOATS, RANK, pack_activations, pack_controller, reference,
)

KINDS = ("q", "k", "v", "ffn")


def controller_weights(model: Path, layer: int):
    _, tensors, base = read_gguf(str(model))

    def load(name, shape):
        info = tensors[name]
        if info["type"] != "F32":
            raise SystemExit(f"{name} is {info['type']}, expected F32")
        return np.memmap(model, dtype="<f4", mode="r", offset=base + info["offset"], shape=shape)

    prefix = f"blk.{layer}.loop_"
    return [(load(f"{prefix}{k}_ctrl_down.weight", (RANK, 2 * HIDDEN)),
             load(f"{prefix}{k}_ctrl_gain.weight", (HIDDEN, RANK)),
             load(f"{prefix}{k}_ctrl_forget.weight", (HIDDEN, RANK))) for k in KINDS]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("build", type=Path, help="loop_controller build (final.xclbin, insts.bin)")
    parser.add_argument("out", type=Path)
    parser.add_argument("--layer", type=int, default=6)
    parser.add_argument("--first", type=int, default=0)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--rows", type=int, default=BLOCK_ROWS)
    parser.add_argument("--repeat", type=int, default=1, help="runs in the config, for timing")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    eps = 1e-6
    weights = controller_weights(args.model, args.layer)
    rng = np.random.default_rng(91)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    if args.verify:
        actual = np.fromfile(out / "output.bin", dtype="<f4").reshape(args.count, BLOCK_ROWS, HIDDEN)
        worst = 0.0
        for n in range(args.count):
            x = np.load(out / f"x{n}.npy")
            m = np.load(out / f"m{n}.npy")
            ref = reference(x, m, *weights[args.first + n], eps)
            got = actual[n, :args.rows].astype(np.float64)
            update = np.linalg.norm(ref - x, axis=1)
            error = np.linalg.norm(got - ref, axis=1)
            rel_u = error / np.maximum(np.linalg.norm(ref, axis=1), 1e-30)
            rel_update = error / np.maximum(update, 1e-30)
            pad = np.abs(actual[n, args.rows:]).max() if args.rows < BLOCK_ROWS else 0.0
            print(f"controller {KINDS[args.first + n]}: max row rel L2 of u {rel_u.max():.3e}, "
                  f"of the update {rel_update.max():.3e} (median {np.median(rel_update):.3e}), "
                  f"max abs {np.abs(got - ref).max():.3e}, padded rows max {pad:.3e}")
            worst = max(worst, rel_u.max())
            if not np.isfinite(got).all() or pad != 0.0:
                raise SystemExit("non-finite output or nonzero padded rows")
        if worst > 1e-4:
            raise SystemExit("controller output differs from the F64 reference")
        print("PASS")
        return

    packed = np.concatenate([pack_controller(*w) for w in weights])
    packed.tofile(out / "weights.bin")
    acts = []
    for n in range(args.count):
        scale = np.exp2(rng.uniform(-3, 3, size=(args.rows, 1))).astype(np.float32)
        x = (rng.standard_normal((args.rows, HIDDEN)) * scale).astype(np.float32)
        m = (rng.standard_normal((args.rows, HIDDEN)) * scale[::-1]).astype(np.float32)
        np.save(out / f"x{n}.npy", x)
        np.save(out / f"m{n}.npy", m)
        acts.append(pack_activations(x, m))
    np.concatenate(acts).tofile(out / "activation.bin")
    b, o = args.build.resolve(), out.resolve()
    lines = ["device", f"xclbin C {b}/final.xclbin", f"kernelx ctrl C {b}/insts.bin",
             f"buf w {packed.nbytes} {o}/weights.bin", f"buf x {args.count * BLOCK_ROWS * 2 * HIDDEN * 4} {o}/activation.bin",
             f"buf y {args.count * OUTPUT_FLOATS * 4}"]
    lines += ["run ctrl w x y"] * args.repeat
    lines += [f"dump y {o}/output.bin {args.count * OUTPUT_FLOATS * 4}"]
    (out / "run.cfg").write_text("\n".join(lines) + "\n")
    print(f"wrote {out / 'run.cfg'}")


if __name__ == "__main__":
    main()
