"""Project eight experts directly from the block-prefill weight layout."""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gguf_pool import IQ4_NL_VALUES  # noqa: E402
from ironutil import include_dirs  # noqa: E402

N, TN, TK, SLOTS = 2304, 96, 32, 8
UP_K, DOWN_K, PAIRS = 2304, 1152, 3


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def compact_decode(w: In, x: In, y: Out, *,
                   k: CompileTime[int], iq4: CompileTime[bool], rows: CompileTime[int],
                   token_rows: CompileTime[int] = 4, split_scale: CompileTime[bool] = False,
                   moe: CompileTime[bool] = False, source_hash: CompileTime[int] = 0):
    if moe:
        # Gate/up, SiTU and down of 16 rows per slot in one launch; the down leg sets the band geometry.
        if rows != 4 or token_rows != 16:
            raise ValueError("the fused grouped stream needs 4 core rows and 16 token rows")
        k = DOWN_K
    if k not in (1152, 2304) or rows not in (1, 2, 4):
        raise ValueError("expected the Inkling expert widths and 1, 2 or 4 core rows")
    if token_rows not in (4, 16):
        raise ValueError("expected 4 or 16 token rows per expert")
    if split_scale and iq4:
        raise ValueError("the split-scale weight layout is Q8 only")
    tiles_k = k // TK
    bands = N // TN // rows
    tile_a = token_rows * (8 + TK) * 2
    tile_b = TN * (4 + TK) if split_scale else TN * (2 + (TK // 2 if iq4 else TK))
    tile_c = token_rows * TN
    a_bytes = tile_a * tiles_k
    b_bytes = tile_b * tiles_k * bands * rows
    c_elems = token_rows * N

    def byte_type(count):
        return np.ndarray[(count,), np.dtype[np.uint8]]

    def float_type(count):
        return np.ndarray[(count,), np.dtype[np.float32]]

    a_ty, b_ty, c_ty = byte_type(tile_a), byte_type(tile_b), float_type(tile_c)
    table_ty = np.ndarray[(32,), np.dtype[np.int16]]
    lookup = np.tile(IQ4_NL_VALUES, 2).astype(np.int16)
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]
    project = ExternalFunction("inkling_block_expert", source_file=str(HERE / "grouped_expert.cc"),
                               arg_types=[a_ty, b_ty, table_ty, table_ty, c_ty, np.int32],
                               include_dirs=include_dirs(),
                               compile_flags=[os.environ.get("INKLING2_EXPERT_OPT", "-O2"), f"-DIQ4_WEIGHTS={int(iq4)}", f"-DBLOCK_ROWS={token_rows}",
                                              "-DVECMATH_COMPACT",
                                              f"-DEXPERT_ROW_UNROLL={os.environ.get('INKLING2_EXPERT_UNROLL', 1)}",
                                              f"-DEXPERT_SKIP_EMPTY={os.environ.get('INKLING2_EXPERT_SKIP', 0)}"] +
                                             (["-DEXPERT_NULL"] if os.environ.get("INKLING2_EXPERT_NULL") else []) +
                                             (["-DSPLIT_SCALE=1", "-DSPLIT_SCALE_HIGH_ONLY=1"] if split_scale else []))
    situ = Kernel("inkling_grouped_situ", project.object_file_name, [c_ty, c_ty, c_ty])

    def band(a, b, out, count, gemv, lut0, lut1):
        x = a.acquire(1)
        w = b.acquire(1)
        gemv(x, w, lut0, lut1, out, 1)
        a.release(1)
        b.release(1)
        for _ in range_(count - 1):
            x = a.acquire(1)
            w = b.acquire(1)
            gemv(x, w, lut0, lut1, out, 0)
            a.release(1)
            b.release(1)

    def compute(a, b, c, lut0, lut1, ctrl, gemv, situ_fn, gate, up):
        first_x = a.acquire(1)
        first_w = b.acquire(1)
        # The first records arrive after the RTP writes, so the control words are read after them.
        remaining, fused = ctrl[0] - 1, ctrl[1]
        for _ in range_(1 - fused):
            first_out = c.acquire(1)
            gemv(first_x, first_w, lut0, lut1, first_out, 1)
            a.release(1)
            b.release(1)
            for _ in range_(remaining):
                x = a.acquire(1)
                w = b.acquire(1)
                gemv(x, w, lut0, lut1, first_out, 0)
                a.release(1)
                b.release(1)
            c.release(1)
        for _ in range_(fused):
            frame = c.acquire(1)
            # Pair 0's gate band starts with the records read above.
            gemv(first_x, first_w, lut0, lut1, gate, 1)
            a.release(1)
            b.release(1)
            for _ in range_(UP_K // TK - 1):
                x = a.acquire(1)
                w = b.acquire(1)
                gemv(x, w, lut0, lut1, gate, 0)
                a.release(1)
                b.release(1)
            band(a, b, up, UP_K // TK, gemv, lut0, lut1)
            situ_fn(gate, up, frame)
            c.release(1)
            for _ in range_(PAIRS - 1):
                out = c.acquire(1)
                band(a, b, gate, UP_K // TK, gemv, lut0, lut1)
                band(a, b, up, UP_K // TK, gemv, lut0, lut1)
                situ_fn(gate, up, out)
                c.release(1)
            for _ in range_(N // TN // rows):
                out = c.acquire(1)
                band(a, b, out, DOWN_K // TK, gemv, lut0, lut1)
                c.release(1)

    a_prods, b_prods, c_conss, workers, rtps = [], [], [], [], []
    for col in range(SLOTS):
        mem = Tile(col, 1)
        shim = Tile(col, 0)
        a = ObjectFifo(a_ty, depth=2, name=f"A_{col}")
        a_core = a.cons().forward(tile=mem, obj_type=a_ty)
        b = ObjectFifo(byte_type(tile_b * rows), depth=2, name=f"B_{col}")
        b_core = b.cons().split([tile_b * row for row in range(rows)], tile=mem, obj_types=[b_ty] * rows)
        c = ObjectFifo(float_type(tile_c * rows), depth=2, name=f"C_{col}")
        c_core = c.prod().join([tile_c * row for row in range(rows)], tile=mem, obj_types=[c_ty] * rows)
        for row in range(rows):
            chunks = Buffer(rtp_ty, initial_value=np.zeros(2, np.int32), use_write_rtp=True)
            rtps.append(chunks)
            workers.append(Worker(compute, tile=Tile(col, 2 + row), stack_size=0x2000, fn_args=[
                a_core.cons(), b_core[row].cons(), c_core[row].prod(),
                Buffer(table_ty, initial_value=lookup, address=0xA000),
                Buffer(table_ty, initial_value=lookup, address=0xC000), chunks, project, situ,
                Buffer(c_ty), Buffer(c_ty),
            ]))
        a_prods.append(a.prod(tile=shim))
        b_prods.append(b.prod(tile=shim))
        c_conss.append(c.cons(tile=shim))

    def weight_tap(total, slot, per_slot):
        return TensorAccessPattern((SLOTS, total // SLOTS), slot * per_slot, [1, per_slot // rows // tile_b, rows, tile_b],
                                   [0, tile_b, per_slot // rows, 1])

    # Fused stream buffers: gate/up pool, 16 packed rows per slot, down pool, then [y of all slots | SiTU wire].
    up_bytes = tile_b * (UP_K // TK) * bands * rows
    x_bytes = tile_a * (UP_K // TK)
    wire_elems = rows * PAIRS * tile_c
    out_elems = SLOTS * (c_elems + wire_elems)

    def fused_sequence(up_weights, tokens, down_weights, output, ap, bp, cp):
        for ctrl in rtps:
            ctrl[0] = tiles_k
            ctrl[1] = 1
        group = TaskGroup()
        for slot in range(SLOTS):
            # Frame (pair j, core r) lands in slot 4j + r of this column's wire region.
            cp[slot].drain(output, tap=TensorAccessPattern(
                (out_elems,), SLOTS * c_elems + slot * wire_elems, [1, 1, wire_elems // 256, 256], [0, 0, 256, 1]),
                wait=True, group=group)
            ap[slot].fill(tokens, tap=TensorAccessPattern(
                (SLOTS, x_bytes), slot * x_bytes, [bands, 1, UP_K // TK, tile_a], [0, 0, tile_a, 1]), group=group)
            bp[slot].fill(up_weights, tap=weight_tap(SLOTS * up_bytes, slot, up_bytes), group=group)
        group.finish()
        # The down leg reads the wire back in K order (pair j of core r holds K tiles 3(3r + j)..) after the drain.
        group = TaskGroup()
        for slot in range(SLOTS):
            cp[slot].drain(output, tap=TensorAccessPattern(
                (out_elems,), slot * c_elems, [1, 1, c_elems // 256, 256], [0, 0, 256, 1]), wait=True, group=group)
            ap[slot].fill(output, tap=TensorAccessPattern(
                (out_elems,), SLOTS * c_elems + slot * wire_elems, [bands, rows, PAIRS, 3 * tile_a // 4],
                [0, tile_c, rows * tile_c, 1]), group=group)
            bp[slot].fill(down_weights, tap=weight_tap(SLOTS * b_bytes, slot, b_bytes), group=group)
        group.finish()

    def sequence(weights, activations, output, ap, bp, cp):
        for chunks in rtps:
            chunks[0] = tiles_k
            chunks[1] = 0
        group = TaskGroup()
        for slot in range(SLOTS):
            cp[slot].drain(output, tap=TensorAccessPattern(
                (SLOTS, c_elems), slot * c_elems, [1, 1, c_elems // 256, 256], [0, 0, 256, 1]),
                wait=True, group=group)
            ap[slot].fill(activations, tap=TensorAccessPattern(
                (SLOTS, a_bytes), slot * a_bytes, [bands, 1, tiles_k, tile_a], [0, 0, tile_a, 1]),
                group=group)
            # For each K tile gather one block from each worker's contiguous
            # stripe of the existing expert plane, in memtile split order.
            bp[slot].fill(weights, tap=TensorAccessPattern(
                (SLOTS, b_bytes), slot * b_bytes, [1, bands * tiles_k, rows, tile_b],
                [0, tile_b, b_bytes // rows, 1]), group=group)
        group.finish()

    if moe:
        runtime = Runtime(fused_sequence, [byte_type(SLOTS * up_bytes), byte_type(SLOTS * x_bytes),
                                           byte_type(SLOTS * b_bytes), float_type(out_elems), a_prods, b_prods, c_conss])
    else:
        runtime = Runtime(sequence, [byte_type(SLOTS * b_bytes), byte_type(SLOTS * a_bytes),
                                     float_type(SLOTS * c_elems), a_prods, b_prods, c_conss])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = compact_decode
SPECIALIZE = {
    "k": int(os.environ.get("INKLING2_COMPACT_K", N)),
    "iq4": os.environ.get("INKLING2_COMPACT_FORMAT", "iq4") == "iq4",
    "rows": int(os.environ.get("INKLING2_COMPACT_ROWS", 4)),
    "token_rows": int(os.environ.get("INKLING2_COMPACT_TOKENS", 4)),
    "split_scale": os.environ.get("INKLING2_COMPACT_SPLIT_SCALE", "0") == "1",
    "moe": os.environ.get("INKLING2_COMPACT_MOE", "0") == "1",
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + b"".join((HERE / name).read_bytes() for name in (
        "iq4_lut.h", "block_expert.cc", "grouped_expert.cc", "situ.h")) + (HERE.parents[1] / "include/vecmath.h").read_bytes()
    ).hexdigest()[:8], 16),
}
