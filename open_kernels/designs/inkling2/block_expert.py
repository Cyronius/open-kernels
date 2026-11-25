"""Batched Inkling 2 dense and expert projections with compact weights and FP32 sums.

Shapes share an image when they use the same core count and operand layout.
A is int16 with one power-of-two scale per row and 32 columns. B keeps the
original codes and BF16 block scales, with an optional second BF16 plane
to preserve each GGUF FP16 scale exactly. The NPU reuses B across rows of A.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gguf_pool import IQ4_NL_VALUES  # noqa: E402
from ironutil import include_dirs  # noqa: E402

N, TM, TN, TK, COLS = 2304, 16, 96, 32, 8


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def block_expert(w: In, x: In, y: Out, *,
                 k: CompileTime[int], iq4: CompileTime[bool], rows: CompileTime[int],
                 n: CompileTime[int] = N, split_scale: CompileTime[bool] = False, source_hash: CompileTime[int] = 0):
    if k not in (1152, 2304, 3072) or n not in (1536, 2304, 6912, 9216) or rows not in (1, 2, 4):
        raise ValueError("unsupported Inkling projection shape or core count")
    if iq4 and split_scale:
        raise ValueError("split scales are only used for Q8 projections")
    tiles_k, tiles_n = k // TK, n // (TN * COLS)
    tile_a = TM * (8 + TK) * 2
    tile_b = TN * (2 + 2 * int(split_scale) + (TK // 2 if iq4 else TK))
    m = TM * rows
    a_bytes = tile_a * tiles_k * rows
    b_bytes = tile_b * tiles_k * tiles_n * COLS

    def byte_type(count):
        return np.ndarray[(count,), np.dtype[np.uint8]]

    a_ty, b_ty = byte_type(tile_a), byte_type(tile_b)
    c_ty = np.ndarray[(TM * TN,), np.dtype[np.float32]]
    c_joint = np.ndarray[(m * TN,), np.dtype[np.float32]]
    table_ty = np.ndarray[(32,), np.dtype[np.int16]]
    lookup = np.tile(IQ4_NL_VALUES, 2).astype(np.int16)
    rtp_ty = np.ndarray[(1,), np.dtype[np.int32]]
    project = ExternalFunction("inkling_block_expert", source_file=str(HERE / "block_expert.cc"),
                               arg_types=[a_ty, b_ty, table_ty, table_ty, c_ty, np.int32],
                               include_dirs=include_dirs(),
                               compile_flags=[os.environ.get("INKLING2_EXPERT_OPT", "-O2"), f"-DIQ4_WEIGHTS={int(iq4)}",
                                              f"-DSPLIT_SCALE={int(split_scale)}",
                                              f"-DEXPERT_ROW_UNROLL={os.environ.get('INKLING2_EXPERT_UNROLL', 1)}"])

    a_ddr = [ObjectFifo(a_ty, depth=2, name=f"A_ddr_{r}") for r in range(rows)]
    a_cores = [a.cons().split([0], obj_types=[a_ty], names=[f"A_row_{r}"])[0] for r, a in enumerate(a_ddr)]
    b_ddr = [ObjectFifo(b_ty, depth=2, name=f"B_ddr_{c}") for c in range(COLS)]
    b_cores = [b.cons().forward(obj_type=b_ty, name=f"B_col_{c}") for c, b in enumerate(b_ddr)]
    c_ddr = [ObjectFifo(c_joint, depth=2, name=f"C_ddr_{c}",
                       dims_to_stream=[(TM // 4, 4 * TN), (4, 8), (TN // 8, 32), (8, 1)])
             for c in range(COLS)]
    c_cores = [fifo.prod().join([TM * TN * r for r in range(rows)], obj_types=[c_ty] * rows,
                               names=[f"C_tile_{r}_{c}" for r in range(rows)], depths=[1] * rows)
               for c, fifo in enumerate(c_ddr)]
    rtps = [[Buffer(rtp_ty, name=f"chunks_{r}_{c}", initial_value=np.zeros(1, np.int32), use_write_rtp=True)
             for c in range(COLS)] for r in range(rows)]

    def compute(a, b, c, lut0, lut1, chunks, gemm):
        out = c.acquire(1)
        x = a.acquire(1)
        w = b.acquire(1)
        remaining = chunks[0] - 1
        gemm(x, w, lut0, lut1, out, 1)
        a.release(1)
        b.release(1)
        for _ in range_(remaining):
            x = a.acquire(1)
            w = b.acquire(1)
            gemm(x, w, lut0, lut1, out, 0)
            a.release(1)
            b.release(1)
        c.release(1)

    workers = [Worker(compute, stack_size=0x1000, fn_args=[
        a_cores[r].cons(), b_cores[c].cons(), c_cores[c][r].prod(),
        Buffer(table_ty, initial_value=lookup, address=0xA000),
        Buffer(table_ty, initial_value=lookup, address=0xC000), rtps[r][c], project,
    ]) for r in range(rows) for c in range(COLS)]

    def linear(total, offset, length, repeat=1):
        assert length % 256 == 0
        blocks = length // 256
        inner = max(i for i in range(1, 1024) if blocks % i == 0)
        return TensorAccessPattern((total,), offset, [repeat, blocks // inner, inner, 256],
                                   [0, inner * 256, 256, 1])

    def sequence(weights, inputs, output, ap, bp, cp):
        for row in rtps:
            for rtp in row:
                rtp[0] = tiles_k
        group = TaskGroup()
        for c in range(COLS):
            length = m * n // COLS
            cp[c].drain(output, tap=linear(m * n, c * length, length), wait=True, group=group)
            if c < rows:
                length = a_bytes // rows
                ap[c].fill(inputs, tap=linear(a_bytes, c * length, length, tiles_n), group=group)
            length = b_bytes // COLS
            bp[c].fill(weights, tap=linear(b_bytes, c * length, length), group=group)
        group.finish()

    output_ty = np.ndarray[(m * n,), np.dtype[np.float32]]
    runtime = Runtime(sequence, [byte_type(b_bytes), byte_type(a_bytes), output_ty,
                                 [f.prod() for f in a_ddr], [f.prod() for f in b_ddr], [f.cons() for f in c_ddr]])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = block_expert
SPECIALIZE = {
    "k": int(os.environ.get("INKLING2_BLOCK_K", N)),
    "iq4": os.environ.get("INKLING2_BLOCK_FORMAT", "iq4") == "iq4",
    "rows": int(os.environ.get("INKLING2_BLOCK_ARRAY_ROWS", 2)),
    "n": int(os.environ.get("INKLING2_BLOCK_N", N)),
    "split_scale": os.environ.get("INKLING2_BLOCK_SPLIT_SCALE", "0") == "1",
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + (HERE / "iq4_lut.h").read_bytes()
                                      + (HERE / "block_expert.cc").read_bytes()).hexdigest()[:8], 16),
}
