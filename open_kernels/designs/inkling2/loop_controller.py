"""Inkling 2 loop controllers for one 256-row prefill block: 32 cores, 8 token rows each, weights broadcast."""
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
from ironutil import include_dirs  # noqa: E402
from recipes.inkling2_controller import (  # noqa: E402
    ACTIVATION_FLOATS, BLOCK_ROWS, CHUNK, CHUNKS, COLS, CONTROLLER_BYTES, CORE_ROWS, FRAME_FLOATS, HIDDEN,
    OUTPUT_FLOATS, TILE_ROWS, WEIGHT_OBJECT,
)

EPS_BITS = int(np.float32(1e-6).view(np.int32))


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def loop_controller(w: In, x: In, y: Out, *, count: CompileTime[int], first: CompileTime[int],
                    eps_bits: CompileTime[int] = EPS_BITS, source_hash: CompileTime[int] = 0):
    if count not in (1, 3) or first + count > 4:
        raise ValueError("expected controllers 0-2 or 3 of a layer")
    column_rows = CORE_ROWS * TILE_ROWS

    def byte_type(n):
        return np.ndarray[(n,), np.dtype[np.uint8]]

    def float_type(n):
        return np.ndarray[(n,), np.dtype[np.float32]]

    w_ty = byte_type(WEIGHT_OBJECT * 2)
    x_ty, y_ty = float_type(FRAME_FLOATS), float_type(TILE_ROWS * CHUNK)
    z_ty, sq_ty, g_ty = float_type(512), float_type(128), float_type(256)
    tiles_ty = np.ndarray[(1024,), np.dtype[np.uint16]]
    params_ty = np.ndarray[(2,), np.dtype[np.int32]]
    null = bool(os.environ.get("INKLING2_CONTROLLER_NULL"))  # timing builds only: all DMA, no arithmetic
    source = ExternalFunction("inkling_ctrl_down", source_file=str(HERE / "loop_controller.cc"),
                              arg_types=[x_ty, w_ty, z_ty, z_ty, sq_ty, tiles_ty, np.int32],
                              include_dirs=include_dirs(),
                              compile_flags=[os.environ.get("INKLING2_CONTROLLER_OPT", "-O2"), "-DVECMATH_COMPACT"]
                                            + (["-DCONTROLLER_NULL"] if null else []))
    obj = source.object_file_name
    down = Kernel("inkling_ctrl_down", obj, [x_ty, w_ty, z_ty, z_ty, sq_ty, tiles_ty, np.int32])
    scale = Kernel("inkling_ctrl_scale", obj, [z_ty, z_ty, sq_ty, tiles_ty, params_ty])
    gate = Kernel("inkling_ctrl_gate", obj, [tiles_ty, w_ty, g_ty, g_ty, x_ty, y_ty])

    def controller(xf, wf, yf, params, zx, zm, sq, tiles, zs, g, f, kernels, start):
        down_fn, scale_fn, gate_fn = kernels
        for j in range_(start, CHUNKS):
            frame = xf.acquire(1)
            weights = wf.acquire(1)
            down_fn(frame, weights, zx, zm, sq, tiles, j)
            xf.release(1)
            wf.release(1)
        scale_fn(zx, zm, sq, zs, params)
        for _ in range_(CHUNKS):
            frame = xf.acquire(1)
            weights = wf.acquire(1)
            out = yf.acquire(1)
            gate_fn(zs, weights, g, f, frame, out)
            xf.release(1)
            wf.release(1)
            yf.release(1)

    def work(xf, wf, yf, params, zx, zm, sq, tiles, zs, g, f, kernels):
        down_fn = kernels[0]
        # The first frame always arrives after the RTP writes, so the count read below is fresh.
        frame = xf.acquire(1)
        weights = wf.acquire(1)
        down_fn(frame, weights, zx, zm, sq, tiles, 0)
        xf.release(1)
        wf.release(1)
        rest = params[0] - 1
        controller(xf, wf, yf, params, zx, zm, sq, tiles, zs, g, f, kernels, 1)
        for _ in range_(rest):
            controller(xf, wf, yf, params, zx, zm, sq, tiles, zs, g, f, kernels, 0)

    w_prods, x_prods, y_conss, workers, controls = [], [], [], [], []
    w_cores = []
    for r in range(CORE_ROWS):
        col = COLS - CORE_ROWS + r
        wf = ObjectFifo(w_ty, depth=2, name=f"W_{r}")
        w_cores.append(wf.cons().forward(tile=Tile(col, 1), obj_type=w_ty, depth=2, name=f"W_core_{r}"))
        w_prods.append(wf.prod(tile=Tile(col, 0)))
    for c in range(COLS):
        mem = Tile(c, 1)
        xf = ObjectFifo(float_type(FRAME_FLOATS * CORE_ROWS), depth=2, name=f"X_{c}")
        x_core = xf.cons().split([FRAME_FLOATS * r for r in range(CORE_ROWS)], tile=mem,
                                 obj_types=[x_ty] * CORE_ROWS, depths=[2] * CORE_ROWS,
                                 names=[f"X_core_{c}_{r}" for r in range(CORE_ROWS)])
        yf = ObjectFifo(float_type(TILE_ROWS * CHUNK * CORE_ROWS), depth=2, name=f"Y_{c}")
        y_core = yf.prod().join([TILE_ROWS * CHUNK * r for r in range(CORE_ROWS)], tile=mem,
                                obj_types=[y_ty] * CORE_ROWS, depths=[2] * CORE_ROWS,
                                names=[f"Y_core_{c}_{r}" for r in range(CORE_ROWS)])
        for r in range(CORE_ROWS):
            params = Buffer(params_ty, initial_value=np.zeros(2, np.int32), use_write_rtp=True, name=f"params_{c}_{r}")
            controls.append(params)
            workers.append(Worker(work, tile=Tile(c, 2 + r), stack_size=0x1000, fn_args=[
                x_core[r].cons(), w_cores[r].cons(), y_core[r].prod(), params,
                Buffer(z_ty), Buffer(z_ty), Buffer(sq_ty), Buffer(tiles_ty), Buffer(tiles_ty),
                Buffer(g_ty), Buffer(g_ty), (down, scale, gate),
            ]))
        x_prods.append(xf.prod(tile=Tile(c, 0)))
        y_conss.append(yf.cons(tile=Tile(c, 0)))

    weight_bytes = 4 * CONTROLLER_BYTES

    def linear(total, offset, length):
        blocks = length // 256
        inner = max(i for i in range(1, 1024) if blocks % i == 0)
        return TensorAccessPattern((total,), offset, [1, blocks // inner, inner, 256], [0, inner * 256, 256, 1])

    def sequence(weights, activations, output, wp, xp, yc):
        for params in controls:
            params[0] = count
            params[1] = eps_bits
        group = TaskGroup()
        for c in range(COLS):
            for n in range(count):
                yc[c].drain(output, tap=TensorAccessPattern(
                    (count * OUTPUT_FLOATS,), n * OUTPUT_FLOATS + c * column_rows * HIDDEN,
                    [1, CHUNKS, column_rows, CHUNK], [0, CHUNK, HIDDEN, 1]), wait=True, group=group)
        for c in range(COLS):
            for n in range(count):
                xp[c].fill(activations, tap=TensorAccessPattern(
                    (count * ACTIVATION_FLOATS,), n * ACTIVATION_FLOATS + c * column_rows * 2 * HIDDEN,
                    [2, CHUNKS, column_rows, 2 * CHUNK], [0, 2 * CHUNK, 2 * HIDDEN, 1]), group=group)
        for prod in wp:
            prod.fill(weights, tap=linear(weight_bytes, first * CONTROLLER_BYTES, count * CONTROLLER_BYTES), group=group)
        group.finish()

    runtime = Runtime(sequence, [byte_type(weight_bytes), float_type(count * ACTIVATION_FLOATS),
                                 float_type(count * OUTPUT_FLOATS), w_prods, x_prods, y_conss])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = loop_controller
SPECIALIZE = {
    "count": int(os.environ.get("INKLING2_CONTROLLER_COUNT", 3)),
    "first": int(os.environ.get("INKLING2_CONTROLLER_FIRST", 0)),
    "eps_bits": int(os.environ.get("INKLING2_CONTROLLER_EPS_BITS", EPS_BITS)),
    "source_hash": int(hashlib.sha256(b"".join(p.read_bytes() for p in (
        HERE / "loop_controller.cc", HERE.parents[1] / "include/vecmath.h",
        HERE.parents[1] / "recipes/inkling2_controller.py",
    )) + os.environ.get("INKLING2_CONTROLLER_NULL", "").encode()).hexdigest()[:8], 16),
}
