r"""Test vectors for attn_fa256 at Qwen3.6-35B-A3B's attention shape (16 query heads, 2 kv heads, head dim 256):
one 256-token block at position p0 against the L = p0 + 256 cached rows before and including it.

    python make_test.py --build <build dir> --out <dir> [--L 1024] [--p0 N] [--qk-scale 1] [--seed 0]
    python make_test.py --build <build dir> --out <dir> --capture <npz>   (captured q [T,16,256], k/v [rows,2,256])
    run_kernel.exe <dir>\run.cfg
    python compare.py <dir>

Writes q.bin (per head the record with p0, then the head's 256 rows scaled by log2(e)/16), kv.bin (a KV cache of
--max-rows rows, K | V per row, bf16), ref.bin (fp64 causal attention as f32 [256, 4096]) and run.cfg.
--p0 below L - 256 leaves stream chunks past the block, which the causal bound must skip.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

T, HD, REC = 256, 256, 64
QS = np.log2(np.e) / 16.0


def bf(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float32).astype(bfloat16)


def q_buffer(q: np.ndarray, p0: int) -> np.ndarray:
    """q [T, heads, 256] fp32 in natural units -> the kernel's Q buffer (bf16 [heads][64 + 256][256])."""
    heads = q.shape[1]
    out = np.zeros((heads, REC + T, HD), np.uint16)
    out[:, 0, :2] = np.array([p0], np.int32).view(np.uint16)
    out[:, REC:, :] = bf(q * np.float32(QS)).view(np.uint16).transpose(1, 0, 2)
    return out.reshape(-1)


def reference(q: np.ndarray, k: np.ndarray, v: np.ndarray, p0: int, gqa: int) -> np.ndarray:
    """fp64 causal attention of the kernel's rounded inputs; q [T, heads, 256] (log2 units), k/v [rows, kvh, 256]."""
    heads = q.shape[1]
    out = np.zeros((T, heads, HD))
    pos = p0 + np.arange(T)
    for h in range(heads):
        kh, vh = k[:, h // gqa, :].astype(np.float64), v[:, h // gqa, :].astype(np.float64)
        s = q[:, h, :].astype(np.float64) @ kh.T
        s[np.arange(kh.shape[0])[None, :] > pos[:, None]] = -np.inf
        s -= s.max(1, keepdims=True)
        p = np.exp2(s)
        out[:, h, :] = (p / p.sum(1, keepdims=True)) @ vh
    return out.reshape(T, heads * HD)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--insts", type=Path, help="defaults to <build>/insts.bin")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--L", type=int, default=1024)
    ap.add_argument("--p0", type=int, default=-1, help="defaults to L - 256")
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--kv-ld", type=int, default=1024)
    ap.add_argument("--max-rows", type=int, default=4096)
    ap.add_argument("--qk-scale", type=float, default=1.0)
    ap.add_argument("--capture", type=Path)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    p0 = a.L - T if a.p0 < 0 else a.p0
    kvh, heads = a.kv_heads, a.heads
    k = np.zeros((a.max_rows, kvh, HD), np.float32)
    v = np.zeros((a.max_rows, kvh, HD), np.float32)
    if a.capture:
        c = np.load(a.capture)
        n = min(c["k"].shape[0], p0 + T)
        k[:n], v[:n] = c["k"][:n], c["v"][:n]
        q = c["q"][:T].astype(np.float32)
    else:
        # unit-norm-ish rows scaled so scores spread over a few log2 units, as after a q/k norm
        k[:] = rng.standard_normal(k.shape) * np.sqrt(a.qk_scale)
        v[:] = rng.standard_normal(v.shape)
        k[0] *= 3.0      # an attention sink: row 0 dominates some scores
        q = (rng.standard_normal((T, heads, HD)) * np.sqrt(a.qk_scale) / 16.0 * 4.0).astype(np.float32)
    k, v = bf(k).astype(np.float32), bf(v).astype(np.float32)
    kv = np.zeros((a.max_rows, a.kv_ld), np.float32)
    kv[:, :kvh * HD] = k.reshape(a.max_rows, -1)
    kv[:, kvh * HD:2 * kvh * HD] = v.reshape(a.max_rows, -1)
    qb = q_buffer(q, p0)
    qr = qb.view(bfloat16).astype(np.float32).reshape(heads, REC + T, HD)[:, REC:, :].transpose(1, 0, 2)
    ref = reference(qr, k[:p0 + T], v[:p0 + T], p0, heads // kvh)
    qb.tofile(a.out / "q.bin")
    bf(kv).view(np.uint16).tofile(a.out / "kv.bin")
    ref.astype(np.float32).tofile(a.out / "ref.bin")
    o_bytes = T * heads * HD * 4
    b = a.build.resolve()
    insts = (a.insts or b / "insts.bin").resolve()
    cfg = "\n".join(["device", f"xclbin G {b / 'final.xclbin'}", f"kernelx k G {insts}",
                     f"buf q {qb.size * 2} q.bin", f"buf kv {kv.size * 2} kv.bin", f"buf o {o_bytes}",
                     *(["run k q kv o"] * a.runs), f"dump o o.bin {o_bytes}", ""])
    (a.out / "run.cfg").write_text(cfg)
    flop = 4 * heads * T * (p0 + T // 2) * HD
    print(f"L {a.L} p0 {p0}: Q {qb.size * 2} B, KV {kv.size * 2} B, O {o_bytes} B, {flop / 1e9:.2f} GFLOP (causal)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
