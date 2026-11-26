"""Operand layouts of the Inkling 2 loop controller kernel (designs/inkling2/loop_controller.py)."""
from __future__ import annotations

import numpy as np

from recipes.gptoss_pack import bf16_bits, bf16_value

HIDDEN, RANK = 2304, 64
BLOCK_ROWS = 256                     # token rows a dispatch
COLS, CORE_ROWS, TILE_ROWS = 8, 4, 8  # 32 cores, 8 token rows each
CHUNK = 32                           # hidden indices a frame (x and m of each)
CHUNKS = HIDDEN // CHUNK
FRAME_FLOATS = TILE_ROWS * 2 * CHUNK
WEIGHT_OBJECT = 4096                 # bf16 values (8 KB)
CONTROLLER_VALUES = 2 * CHUNKS * WEIGHT_OBJECT     # bf16 values of one packed controller
CONTROLLER_BYTES = CONTROLLER_VALUES * 2
ACTIVATION_FLOATS = BLOCK_ROWS * 2 * HIDDEN
OUTPUT_FLOATS = BLOCK_ROWS * HIDDEN
assert CONTROLLER_VALUES == 2 * HIDDEN * RANK + 2 * RANK * HIDDEN


def bf16_exact(values: np.ndarray) -> np.ndarray:
    """F32 -> bf16 bits; the kernel keeps only these, so every value must be a bf16 value."""
    v = np.ascontiguousarray(values, np.float32)
    bits = bf16_bits(v)
    if not np.array_equal(bf16_value(bits), v):
        raise ValueError("controller weights are not bf16 values")
    return bits


def pack_controller(down: np.ndarray, gain: np.ndarray, forget: np.ndarray) -> np.ndarray:
    """down [64, 4608], gain and forget [2304, 64] (GGUF row order) -> uint16 stream of one controller."""
    if down.shape != (RANK, 2 * HIDDEN) or gain.shape != (HIDDEN, RANK) or forget.shape != (HIDDEN, RANK):
        raise ValueError("unexpected controller weight shapes")
    # Phase 1, frame j: k tiles 0-3 cover x[32j..], 4-7 cover m[32j..]; [k tile][z tile][8 k][8 z].
    d = bf16_exact(down).reshape(RANK // 8, 8, 2, CHUNKS, 4, 8)   # z tile, z lane, half, j, k tile, k lane
    phase1 = d.transpose(3, 2, 4, 0, 5, 1).reshape(-1)
    # Phase 2, chunk j: gain then forget, each [z tile][h tile][8 z][8 h].
    def chunks(w):
        t = bf16_exact(w).reshape(CHUNKS, 4, 8, RANK // 8, 8)  # j, h tile, h lane, z tile, z lane
        return t.transpose(0, 3, 1, 4, 2).reshape(CHUNKS, -1)
    phase2 = np.stack((chunks(gain), chunks(forget)), axis=1).reshape(-1)
    return np.concatenate((phase1, phase2)).astype("<u2")


def pack_activations(x: np.ndarray, m: np.ndarray) -> np.ndarray:
    """x, m [rows <= 256, 2304] -> [256, 4608] floats with x and m interleaved in runs of 32."""
    rows = x.shape[0]
    if x.shape != m.shape or x.shape[1] != HIDDEN or rows > BLOCK_ROWS:
        raise ValueError("unexpected controller activation shape")
    out = np.zeros((BLOCK_ROWS, CHUNKS, 2, CHUNK), np.float32)
    out[:rows, :, 0] = x.reshape(rows, CHUNKS, CHUNK)
    out[:rows, :, 1] = m.reshape(rows, CHUNKS, CHUNK)
    return out.reshape(-1)


def reference(x: np.ndarray, m: np.ndarray, down, gain, forget, eps: float) -> np.ndarray:
    """F64 controller update of the host engine (Controller::apply)."""
    x = x.astype(np.float64)
    m = m.astype(np.float64)
    def rms(v):
        return v / np.sqrt(np.mean(v * v, axis=1, keepdims=True) + eps)
    z = np.concatenate((rms(x), rms(m)), axis=1) @ down.astype(np.float64).T
    g = z @ gain.astype(np.float64).T
    f = z @ forget.astype(np.float64).T
    return x + np.tanh(g) * m - np.tanh(f) * x
