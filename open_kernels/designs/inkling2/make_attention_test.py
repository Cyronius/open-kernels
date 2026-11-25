"""Test the streamed Inkling attention against a dense FP64 calculation."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from model.inkling2_gguf_reference import Weights  # noqa: E402
from recipes.gptoss_pack import bf16_bits, bf16_value  # noqa: E402
from recipes.inkling2_attention import (  # noqa: E402
    EXTENT, GROUP, HD, HEADS, KV_HEADS, OUT_FLOATS, RD, SHARED_FRAME_FLOATS, SHARED_FRAMES, SHARED_ROWS, TK, TQ,
    pack_keys_values, pack_queries, pack_shared_queries, unpack_shared_attention,
)


def reference(q, k, v, relative, table, sinks, start, window):
    q_pos = np.arange(start, start + len(q))
    distance = q_pos[:, None] - np.arange(len(k))
    allowed = distance >= 0
    if window:
        allowed &= distance < window
    bh = table[np.clip(distance, 0, EXTENT - 1)]
    rel = np.einsum("tnd,tsd->nts", relative.astype(np.float64), bh)
    rel *= (distance >= 0) & (distance < EXTENT)
    kr = np.repeat(k.astype(np.float64), GROUP, axis=1)
    vr = np.repeat(v.astype(np.float64), GROUP, axis=1)
    s = np.einsum("tnd,snd->nts", q.astype(np.float64), kr) / np.sqrt(HD)
    s += rel
    s = np.where(allowed, s, -np.inf)
    peak = np.maximum(s.max(axis=-1), sinks.T)
    prob = np.exp(s - peak[..., None])
    norm = prob.sum(axis=-1) + np.exp(sinks.T - peak)
    out = np.einsum("nts,snd->ntd", prob / norm[..., None], vr)
    return out, peak, norm


def verify(build):
    expect = np.load(build / "expected.npz")
    shared = "shared" in expect and bool(expect["shared"])
    parts = []
    for i in range(int(expect["parts"])):
        out = np.fromfile(build / f"out-{i}.bin", dtype="<f4")
        parts.append(unpack_shared_attention(out) if shared else out.reshape(HEADS, OUT_FLOATS))
    parts = np.stack(parts).astype(np.float64)
    peak = parts[..., -2 * TQ:-TQ]
    mass = parts[..., -TQ:]
    out = parts[..., :-2 * TQ].reshape(-1, HEADS, TQ, HD)
    peak_all = peak.max(axis=0)
    scale = np.exp(peak - peak_all)
    mass_all = (mass * scale).sum(axis=0)
    actual = (out * scale[..., None]).sum(axis=0) / mass_all[..., None]
    err = np.linalg.norm(actual - expect["output"], axis=-1) / np.linalg.norm(expect["output"], axis=-1)
    peak_err = np.abs(peak_all - expect["peak"]).max()
    norm_err = np.abs(mass_all * np.exp(peak_all - expect["peak"]) / expect["norm"] - 1).max()
    print(f"{len(parts)} ranges, worst relative L2 {err.max():.8g}, max peak delta {peak_err:.8g}, "
          f"max relative normalizer delta {norm_err:.8g}")
    if not np.isfinite(parts).all() or not np.isfinite(err).all() or err.max() > 5e-4 or peak_err > 2e-4 or norm_err > 5e-4:
        raise SystemExit("attention output or streaming state differs from the FP64 reference")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("build", type=Path)
    parser.add_argument("--chunks", type=int, default=1)
    parser.add_argument("--case", choices=("short", "sliding", "long"), default="short")
    parser.add_argument("--shared-format", choices=("iq4", "q8"))
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        verify(args.build)
        return

    rows, window, layer = {"short": (21, 1024, 4), "sliding": (1036, 1024, 4), "long": (8194, 0, 5)}[args.case]
    start = rows - TQ
    weights = Weights(args.model)
    table = weights(f"blk.{layer}.attn_rel_bias.weight")
    rng = np.random.default_rng(465)
    q = rng.standard_normal((TQ, HEADS, HD), dtype=np.float32) * 1.5
    relative = rng.standard_normal((TQ, HEADS, RD), dtype=np.float32) * 0.7
    sinks = rng.standard_normal((TQ, HEADS), dtype=np.float32)
    sinks[::3] += 6
    if not window:
        fields = "inkling2_loop.attention.log_scale_"
        positions = (np.arange(start, rows, dtype=np.float32) + 1) / weights.metadata[fields + "floor"]
        tau = 1 + weights.metadata[fields + "alpha"] * np.log(np.maximum(1, positions))
        q *= tau[:, None, None]
        relative *= tau[:, None, None]
    k, v = (bf16_bits(rng.standard_normal((rows, KV_HEADS, HD), dtype=np.float32)) for _ in range(2))
    output, peak, norm = reference(q, bf16_value(k), bf16_value(v), relative, table.astype(np.float64), sinks, start, window)
    shared = bool(args.shared_format)
    output_bytes = (KV_HEADS * SHARED_FRAMES * SHARED_ROWS * SHARED_FRAME_FLOATS if shared else HEADS * OUT_FLOATS) * 4
    cfg = ["device", "xclbin G final.xclbin", "kernelx k G insts.bin", f"buf output {output_bytes}"]
    parts = 0
    for a in range(0, rows, args.chunks * TK):
        z = min(rows, a + args.chunks * TK)
        qry = pack_queries(q, relative, sinks, start, a, z, window, a == 0)
        if shared:
            qry = pack_shared_queries(qry, args.shared_format == "iq4")
        kv = pack_keys_values(k[a:z], v[a:z], table, start, a, args.chunks)
        qry.tofile(args.build / f"q-{parts}.bin")
        kv.tofile(args.build / f"kv-{parts}.bin")
        cfg += [f"buf q{parts} {qry.nbytes} q-{parts}.bin", f"buf kv{parts} {kv.nbytes} kv-{parts}.bin",
                f"run k q{parts} kv{parts} output", f"dump output out-{parts}.bin {output_bytes}"]
        parts += 1
    (args.build / "run.cfg").write_text("\n".join(cfg) + "\n", encoding="ascii")
    np.savez(args.build / "expected.npz", parts=parts, output=output, peak=peak, norm=norm, shared=shared)


if __name__ == "__main__":
    main()
