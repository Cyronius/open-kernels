"""Pack the compact operands and unpack the column-major Inkling expert GEMM."""
from __future__ import annotations

import numpy as np

from gguf_pool import IQ4_NL_BYTES
from recipes.gptoss_pack import bf16_bits, bf16_value

TM, TN, TK, COLS = 16, 96, 32, 8


def pack_activations(values: np.ndarray, *, tile_m: int = TM, from_fp32: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Pack [M,K] float and return its independently dequantized int16 values."""
    m, k = values.shape
    if tile_m not in (4, TM) or m % tile_m or k % TK:
        raise ValueError("activation shape must be tiled by 4 or 16 rows and 32 columns")
    x = np.ascontiguousarray(values, dtype=np.float32).reshape(m, k // TK, TK)
    if from_fp32:
        exp = ((x.view(np.uint32) >> 23) & 255).max(axis=-1).astype(np.int32)
    else:
        bits = bf16_bits(x)
        exp = ((bits >> 7) & 255).max(axis=-1).astype(np.int32)
        x = bf16_value(bits)
    if np.any(exp == 255):
        raise ValueError("non-finite activation")
    shifts = np.minimum(126, 141 - exp)
    fixed = np.clip(np.rint(np.ldexp(x.astype(np.float64), shifts[..., None])), -32768, 32767).astype(np.int16)
    restored = np.ldexp(fixed.astype(np.float64), -shifts[..., None]).reshape(m, k)
    # [array row, K tile] then BF16 row scales (each repeated in eight lanes)
    # and [rows/4, K/8, row, K] codes.
    scales = bf16_bits(np.ldexp(np.ones_like(shifts, dtype=np.float32), -shifts))
    scales = scales.reshape(m // tile_m, tile_m, k // TK).transpose(0, 2, 1)
    s = np.ascontiguousarray(np.repeat(scales, 8, axis=-1), dtype="<u2")
    a = fixed.reshape(m // tile_m, tile_m // 4, 4, k // TK, TK // 8, 8).transpose(0, 3, 1, 4, 2, 5)
    a = np.ascontiguousarray(a, dtype="<i2").view(np.uint8).reshape(m // tile_m, k // TK, -1)
    return np.concatenate((s.view(np.uint8), a), axis=-1).reshape(-1), restored


def pack_weights(blocks: np.ndarray, iq4: bool, *, split_scale: bool = False) -> np.ndarray:
    """Pack GGUF [N,K/32,block bytes], optionally preserving FP16 scales as two BF16 values."""
    n, tiles_k, block_size = blocks.shape
    if n % (COLS * TN) or block_size != (IQ4_NL_BYTES if iq4 else 34) or (iq4 and split_scale):
        raise ValueError("invalid Inkling expert weight plane")
    scales = np.ascontiguousarray(blocks[..., :2]).view("<f2").reshape(n, tiles_k).astype(np.float32)
    hi = bf16_bits(scales)
    scales = np.stack((hi, bf16_bits(scales - bf16_value(hi)))) if split_scale else hi[None]
    raw = blocks[..., 2:]
    if iq4:
        codes = np.concatenate((raw & 15, raw >> 4), axis=-1)
    else:
        codes = raw
    # A tile holds [scale per output column], [N/8,K/8,K lane,N lane] codes.
    groups = n // TN
    s = scales.reshape(-1, groups, TN, tiles_k).transpose(1, 3, 0, 2)
    s = np.ascontiguousarray(s, dtype="<u2").view(np.uint8).reshape(groups, tiles_k, -1)
    b = codes.reshape(groups, TN // 8, 8, tiles_k, TK // 8, 8).transpose(0, 3, 1, 4, 5, 2)
    if iq4:
        b = b[..., ::2] | (b[..., 1::2] << 4)
    tiles = np.concatenate((s, b.reshape(groups, tiles_k, -1)), axis=-1)
    packed = tiles.reshape(groups // COLS, COLS, tiles_k, -1).transpose(1, 0, 2, 3)
    return np.ascontiguousarray(packed).reshape(-1)


def unpack_output(data: np.ndarray, m: int, n: int) -> np.ndarray:
    """[array column, N group, activation row, N lane] -> [M,N]."""
    return data.reshape(COLS, n // (TN * COLS), m, TN).transpose(2, 1, 0, 3).reshape(m, n)


def unpack_grouped_output(data: np.ndarray, slots: int, core_rows: int, m: int, n: int) -> np.ndarray:
    """Undo the interleaved core stripes and each tile's four-row layout."""
    packed = data.reshape(slots, n // (TN * core_rows), core_rows, m // 4, TN // 8, 4, 8)
    stripes = packed.transpose(0, 2, 1, 3, 5, 4, 6).reshape(slots, COLS, n // (TN * COLS), m, TN)
    return stripes.transpose(0, 3, 2, 1, 4).reshape(slots, m, n)
