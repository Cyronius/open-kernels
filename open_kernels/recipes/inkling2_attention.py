"""Buffer layout for the Inkling relative-attention stream."""
from __future__ import annotations

import numpy as np

from recipes.gptoss_pack import bf16_bits, bf16_value
from recipes.inkling2_block import TN

TQ, TK, HD, RD, STRIP = 16, 32, 128, 16, 64
KV_HEADS, GROUP, EXTENT = 8, 3, 1024
HEADS = KV_HEADS * GROUP
QUERY_BYTES = 128 + 4 * (HD + RD) * TQ
KV_BYTES = 4 * (TK * HD + RD * STRIP)
OUT_FLOATS = TQ * (HD + 2)
SHARED_ROWS, SHARED_FRAME_FLOATS = 4, 4 * TN
SHARED_FRAMES = (OUT_FLOATS + SHARED_FRAME_FLOATS - 1) // SHARED_FRAME_FLOATS


def shared_weight_bytes(iq4):
    return TN * (18 if iq4 else 34)


def shared_query_frames(iq4):
    width = shared_weight_bytes(iq4)
    return (QUERY_BYTES + width - 1) // width


def pack_shared_queries(queries, iq4):
    width = shared_weight_bytes(iq4) * shared_query_frames(iq4)
    out = np.zeros((KV_HEADS, SHARED_ROWS, width), np.uint8)
    out[:, :GROUP, :QUERY_BYTES] = queries.reshape(KV_HEADS, GROUP, QUERY_BYTES)
    return out


def unpack_shared_attention(values):
    records = values.reshape(KV_HEADS, SHARED_FRAMES, SHARED_ROWS, SHARED_FRAME_FLOATS)
    records = records[:, :, :GROUP].transpose(0, 2, 1, 3)
    return records.reshape(HEADS, -1)[:, :OUT_FLOATS]


def split_bf16(x):
    hi = bf16_bits(x)
    return hi, bf16_bits(x - bf16_value(hi))


def pack_queries(q, relative, sinks, query_start, key_start, key_end, window, include_sink):
    rows = q.shape[0]
    if not 1 <= rows <= TQ or q.shape != (rows, HEADS, HD):
        raise ValueError("invalid Inkling query shape")
    if relative.shape != (rows, HEADS, RD) or sinks.shape != (rows, HEADS):
        raise ValueError("invalid Inkling relative query or sink shape")
    packed = np.zeros((HEADS, QUERY_BYTES), np.uint8)
    header = packed[:, :64].view("<i4")
    header[:, :5] = (query_start, key_start, key_end, window, include_sink)
    packed[:, 64:128].view("<f4")[:, :rows] = sinks.T
    x = np.zeros((HEADS, HD + RD, TQ), dtype=np.float32)
    x[:, :HD, :rows] = q.transpose(1, 2, 0) * np.float32(1 / np.sqrt(HD))
    x[:, HD:, :rows] = relative.transpose(1, 2, 0)
    hi, lo = split_bf16(x)
    record = packed[:, 128:].view("<u2").reshape(HEADS, 2, HD + RD, TQ)
    record[:, 0] = hi
    record[:, 1] = lo
    return packed


def pack_keys_values(keys, values, table, query_start, key_start, chunks):
    rows = keys.shape[0]
    if keys.shape != (rows, KV_HEADS, HD) or values.shape != keys.shape or rows > chunks * TK:
        raise ValueError("invalid Inkling KV shapes")
    if keys.dtype != np.uint16 or values.dtype != np.uint16 or table.shape != (EXTENT, RD):
        raise ValueError("expected BF16 KV and a full relative table")
    packed = np.zeros((KV_HEADS, chunks, KV_BYTES // 2), dtype="<u2")
    table_hi, table_lo = split_bf16(table)
    matrix_words = TK * HD
    for chunk, row in enumerate(range(0, rows, TK)):
        count = min(rows - row, TK)
        for offset, x in ((0, keys), (matrix_words, values)):
            matrix = packed[:, chunk, offset:offset + matrix_words].reshape(KV_HEADS, TK, HD)
            matrix[:, :count] = x[row:row + count].transpose(1, 0, 2)
        first = query_start - (key_start + row + TK - 1)
        lo, hi = max(0, first), min(EXTENT, first + TK + TQ - 1)
        if lo < hi:
            strip = packed[:, chunk, 2 * matrix_words:].reshape(KV_HEADS, 2, RD, STRIP)
            strip[:, 0, :, lo - first:hi - first] = table_hi[lo:hi].T
            strip[:, 1, :, lo - first:hi - first] = table_lo[lo:hi].T
    return packed
