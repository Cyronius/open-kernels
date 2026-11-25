"""Project eight routed Inkling 2 experts in one dispatch.

The four streams share a core image. Width and format are programmed by
the sequence. Weight fills initially point at experts 0..7 and are patched
to the actual IDs by the runtime.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
import sys

import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import Buffer, CompileTime, In, ObjectFifo, Out, Program, Runtime, Worker, WorkerRuntimeBarrier
from aie.iron.controlflow import range_
from aie.iron.kernel import ExternalFunction
from ml_dtypes import bfloat16
import numpy as np

HERE = Path(__file__).resolve().parent
GEMV = HERE.parent / "gemv_q4"
sys.path.insert(0, str(HERE.parents[1]))
from gguf_pool import IQ4_NL_VALUES  # noqa: E402
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.gptoss_pack import CH, MX_CH  # noqa: E402

WIDTH, SLOTS = 2304, 8
GROUP_BYTES = 4 * MX_CH
assert GROUP_BYTES == 2 * CH


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def routed_gemv(w: In, x: In, y: Out, *,
                k: CompileTime[int], compact: CompileTime[bool], cores: CompileTime[int],
                source_hash: CompileTime[int] = 0):
    bands = WIDTH // 64
    if cores <= 0 or bands % cores or k not in (WIDTH, 1280):
        raise ValueError("routed GEMV needs equal core bands and a supported padded width")
    per_core = bands // cores
    groups = k // 256 * (1 if compact else 2)
    band_bytes = groups * GROUP_BYTES
    expert_bytes = bands * band_bytes
    weights_type = np.ndarray[(SLOTS * expert_bytes,), np.dtype[np.uint8]]
    group_type = np.ndarray[(GROUP_BYTES,), np.dtype[np.uint8]]
    activation_type = np.ndarray[(WIDTH,), np.dtype[bfloat16]]
    input_type = np.ndarray[(SLOTS * WIDTH,), np.dtype[bfloat16]]
    table_type = np.ndarray[(9 * WIDTH // 4,), np.dtype[np.uint8]]
    control_type = np.ndarray[(16,), np.dtype[np.int32]]
    lut_type = np.ndarray[(32,), np.dtype[np.int16]]
    expanded_type = np.ndarray[(CH,), np.dtype[np.uint8]]
    band_type = np.ndarray[(64,), np.dtype[np.float32]]
    output_type = np.ndarray[(SLOTS * WIDTH,), np.dtype[np.float32]]
    includes = include_dirs() + [str(GEMV)]
    prep = ExternalFunction("inkling_iq4_prep", source_file=str(HERE / "iq4_gemv_prep.cc"),
                            arg_types=[activation_type, table_type, np.int32], include_dirs=includes)
    project_iq4 = ExternalFunction("inkling_iq4_group", source_file=str(HERE / "iq4_gemv_group.cc"),
                                   arg_types=[group_type, table_type, lut_type, lut_type, expanded_type, band_type,
                                              np.int32, np.int32], include_dirs=includes, compile_flags=["-Os"])
    project_q8 = ExternalFunction("inkling_q8_group", source_file=str(HERE / "q8_routed_group.cc"),
                                  arg_types=[group_type, table_type, band_type, np.int32, np.int32],
                                  include_dirs=includes, compile_flags=["-Os"])
    weights = [ObjectFifo(group_type, depth=2, name=f"weights{c}") for c in range(cores)]
    output = [ObjectFifo(band_type, depth=2, name=f"out{c}") for c in range(cores)]
    activation = ObjectFifo(activation_type, depth=1, name="activation")
    modes = [Buffer(control_type, initial_value=np.zeros(16, np.int32), use_write_rtp=True,
                    name=f"mode{c}") for c in range(cores)]
    barriers = [WorkerRuntimeBarrier() for _ in range(cores)]
    table = np.tile(IQ4_NL_VALUES, 2).astype(np.int16)

    def core_body(rx, xin, tx, tab, lut0, lut1, expanded, mode, barrier, setup, iq4, q8):
        barrier.wait_for_value(1)
        for _ in range_(SLOTS):
            a = xin.acquire(1)
            width = mode[0]
            group_count = mode[1]
            setup(a, tab, width)
            xin.release(1)
            for _ in range_(per_core):
                out = tx.acquire(1)
                for _ in range_(mode[2]):
                    for g in range_(group_count):
                        part = rx.acquire(1)
                        iq4(part, tab, lut0, lut1, expanded, out, g, width)
                        rx.release(1)
                for _ in range_(mode[3]):
                    for g in range_(group_count):
                        part = rx.acquire(1)
                        q8(part, tab, out, g, width)
                        rx.release(1)
                tx.release(1)

    workers = [
        Worker(core_body, stack_size=0x1000, fn_args=[
            weights[c].cons(), activation.cons(), output[c].prod(), Buffer(table_type),
            Buffer(lut_type, initial_value=table, address=0xA000),
            Buffer(lut_type, initial_value=table, address=0xC000), Buffer(expanded_type),
            modes[c], barriers[c], prep, project_iq4, project_q8,
        ]) for c in range(cores)
    ]

    def sequence(pool, values, result, weight_prods, x_prod, y_conss):
        for c in range(cores):
            modes[c][0] = k
            modes[c][1] = groups
            modes[c][2] = int(compact)
            modes[c][3] = int(not compact)
        for barrier in barriers:
            barrier.set(1)
        pw, px, py = Pipeline(2), Pipeline(2), Pipeline(2)
        px.fill(x_prod, values, TensorAccessPattern((1, SLOTS * WIDTH), 0, [1, 1, SLOTS, WIDTH],
                                                    [0, 0, WIDTH, 1]))
        for c in range(cores):
            py.drain(y_conss[c], result, TensorAccessPattern(
                (1, SLOTS * WIDTH), c * per_core * 64, [1, 1, SLOTS, per_core * 64], [0, 0, WIDTH, 1]))
        length = per_core * band_bytes
        for slot in range(SLOTS):
            for c in range(cores):
                pw.fill(weight_prods[c], pool, TensorAccessPattern(
                    (1, SLOTS * expert_bytes), slot * expert_bytes + c * length,
                    [1, 1, 1, length], [0, 0, 0, 1]))
        py.finish()
        px.finish()
        pw.finish()

    runtime = Runtime(sequence, [weights_type, input_type, output_type, [f.prod() for f in weights],
                                 activation.prod(), [f.cons() for f in output]])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = routed_gemv
SPECIALIZE = {
    "k": int(os.environ.get("INKLING2_ROUTED_K", WIDTH)),
    "compact": os.environ.get("INKLING2_ROUTED_FORMAT", "iq4") == "iq4",
    "cores": int(os.environ.get("INKLING2_ROUTED_CORES", 12)),
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + b"".join(p.read_bytes() for p in (
        HERE / "iq4_gemv_prep.cc", HERE / "iq4_gemv_group.cc", HERE / "iq4_lut.h", HERE / "q8_routed_group.cc",
        GEMV / "gemv_q8.h", GEMV / "gemv_tab.h",
    ))).hexdigest()[:8], 16),
}
