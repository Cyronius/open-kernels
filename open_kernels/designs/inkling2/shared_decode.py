"""Run Inkling attention and grouped expert decode on two shared input channels."""
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
from ml_dtypes import bfloat16
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from gguf_pool import IQ4_NL_VALUES  # noqa: E402
from ironutil import include_dirs  # noqa: E402
from recipes.inkling2_attention import (  # noqa: E402
    GROUP, KV_BYTES, OUT_FLOATS, QUERY_BYTES, SHARED_FRAME_FLOATS as FRAME_VALUES, SHARED_FRAMES as OUT_FRAMES,
    SHARED_ROWS as ROWS, TK, TQ, shared_query_frames, shared_weight_bytes,
)
from recipes.inkling2_block import TN  # noqa: E402

SLOTS, EXPERT_WIDTH = 8, 2304
ACT_GROUP, ACT_BYTES = 4, 320
BANDS = EXPERT_WIDTH // TN // ROWS
assert OUT_FRAMES == BANDS


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def shared_decode(w: In, x: In, y: Out, *, k: CompileTime[int], iq4: CompileTime[bool],
                  chunks: CompileTime[int], singleton: CompileTime[bool] = False,
                  bands: CompileTime[int] = BANDS, source_hash: CompileTime[int] = 0):
    if k == 0:
        if not 1 <= chunks <= 128 or bands != OUT_FRAMES:
            raise ValueError("invalid shared attention tier")
    elif (chunks or k not in ((1152, 2304, 3072) if singleton else (1152, 2304))
          or not 1 <= bands <= BANDS or (not singleton and bands != BANDS)
          or k % (TK * ACT_GROUP)):
        raise ValueError("invalid shared projection shape")
    tiles = k // TK
    a_frame = ACT_GROUP * ACT_BYTES
    b_frame = TN * 36 if singleton else shared_weight_bytes(iq4)
    a_bytes = tiles * ACT_BYTES if k else chunks * KV_BYTES
    nfq = (QUERY_BYTES + b_frame - 1) // b_frame if singleton else shared_query_frames(iq4)
    records = tiles // 2 if singleton and iq4 else tiles
    stripe = (bands * records if k else nfq) * b_frame
    b_bytes = ROWS * stripe
    c_elems = ROWS * bands * FRAME_VALUES

    def byte_type(count):
        return np.ndarray[(count,), np.dtype[np.uint8]]

    def float_type(count):
        return np.ndarray[(count,), np.dtype[np.float32]]

    aty, bty, cty = byte_type(a_frame), byte_type(b_frame), float_type(FRAME_VALUES)
    q_ty, kv_ty, o_ty = byte_type(QUERY_BYTES), byte_type(KV_BYTES), float_type(OUT_FLOATS)
    scores_ty = float_type(TQ * TK)
    probs_ty = np.ndarray[(TK * 64,), np.dtype[bfloat16]]
    table_ty = np.ndarray[(32,), np.dtype[np.int16]]
    control_size = 5 if singleton else 2
    control_ty = np.ndarray[(control_size,), np.dtype[np.int32]]
    lookup = np.tile(IQ4_NL_VALUES, 2).astype(np.int16)
    ex_args = [aty, bty, table_ty, table_ty, cty, np.int32, np.int32]
    symbol = "inkling_shared_q8" if singleton else "inkling_shared_expert"
    null = bool(os.environ.get("INKLING2_SHARED_NULL"))  # timing builds only: all DMA, no arithmetic
    flags = [os.environ.get("INKLING2_SHARED_OPT", "-O2"), "-DBLOCK_ROWS=4", "-DINKLING2_SINGLETON" if singleton else f"-DIQ4_WEIGHTS={int(iq4)}"]
    flags += ["-DEXPERT_NULL"] if null else []
    external = ExternalFunction(symbol, source_file=str(HERE / "shared_decode.cc"), arg_types=ex_args,
                                include_dirs=include_dirs(), compile_flags=flags)
    eo = external.object_file_name
    expert = Kernel(symbol, eo, ex_args)
    iq4_pair = Kernel("inkling_shared_iq4_pair", eo, ex_args) if singleton else expert
    copy_query = Kernel("inkling_shared_query", eo, [bty, q_ty, np.int32])
    copy_kv = Kernel("inkling_shared_kv", eo, [aty, kv_ty, np.int32])
    emit = Kernel("inkling_shared_output", eo, [o_ty, cty, np.int32])
    emit_zero = Kernel("inkling_shared_zero", eo, [cty])
    external = ExternalFunction("inkling_attn_init", source_file=str(HERE / "relative_attention.cc"),
                                arg_types=[q_ty, o_ty], include_dirs=include_dirs(),
                                compile_flags=[os.environ.get("INKLING2_SHARED_OPT", "-O2"), "-DVECMATH_COMPACT",
                                               "-DINKLING2_ALIGNED_CHUNKS"]
                                              + (["-DINKLING2_ATTENTION_NULL"] if null else []))
    ao = external.object_file_name
    init = Kernel("inkling_attn_init", ao, [q_ty, o_ty])
    step = Kernel("inkling_attn_chunk_aligned", ao, [q_ty, kv_ty, o_ty, scores_ty, probs_ty, np.int32])

    def expert_band(a, b, c, first, count, lut0, lut1, project, pair):
        result = c.acquire(1)
        activations = a.acquire(1)
        project(activations, first, lut0, lut1, result, 0, 1)
        b.release(1)
        for i in range(1, ACT_GROUP // pair):
            weights = b.acquire(1)
            project(activations, weights, lut0, lut1, result, i * pair, 0)
            b.release(1)
        a.release(1)
        for _ in range_(count - 1):
            activations = a.acquire(1)
            for i in range(ACT_GROUP // pair):
                weights = b.acquire(1)
                project(activations, weights, lut0, lut1, result, i * pair, 0)
                b.release(1)
            a.release(1)
        c.release(1)

    def make_worker(is_query_head):
        def work(a, b, c, mode, lut0, lut1, query, kv, result, scores, probs, kernels):
            expert_fn, pair_fn, copy_q, copy_k, init_fn, step_fn, emit_fn, zero_fn = kernels
            first = b.acquire(1)
            # The first B record always arrives after the RTP writes. This orders
            # the mode read on every launch without a core reset or barrier lock.
            if singleton:
                iq4_on, q8_on, kind, count, band_count = mode[0], mode[1], mode[2], mode[3], mode[4]
                for _ in range_(iq4_on):
                    expert_band(a, b, c, first, count, lut0, lut1, pair_fn, 2)
                    for _ in range_(band_count - 1):
                        weights = b.acquire(1)
                        expert_band(a, b, c, weights, count, lut0, lut1, pair_fn, 2)
            else:
                kind, count = mode[0], mode[1]
                q8_on, band_count = 1 - kind, BANDS
            for _ in range_(q8_on):
                expert_band(a, b, c, first, count, lut0, lut1, expert_fn, 1)
                for _ in range_(band_count - 1):
                    weights = b.acquire(1)
                    expert_band(a, b, c, weights, count, lut0, lut1, expert_fn, 1)
            for _ in range_(kind):
                if is_query_head:
                    copy_q(first, query, 0)
                b.release(1)
                for i in range_(1, nfq):
                    frame = b.acquire(1)
                    if is_query_head:
                        copy_q(frame, query, i)
                    b.release(1)
                if is_query_head:
                    init_fn(query, result)
                for chunk in range_(count):
                    for i in range_(KV_BYTES // a_frame):
                        frame = a.acquire(1)
                        if is_query_head:
                            copy_k(frame, kv, i)
                        a.release(1)
                    if is_query_head:
                        step_fn(query, kv, result, scores, probs, chunk)
                for i in range_(OUT_FRAMES):
                    frame = c.acquire(1)
                    if is_query_head:
                        emit_fn(result, frame, i)
                    else:
                        zero_fn(frame)
                    c.release(1)
        return work

    ap, bp, cp, workers, controls = [], [], [], [], []
    for col in range(SLOTS):
        mem, shim = Tile(col, 1), Tile(col, 0)
        a = ObjectFifo(aty, depth=2, name=f"A_{col}")
        a_core = a.cons().forward(tile=mem, obj_type=aty, depth=1)
        b = ObjectFifo(byte_type(ROWS * b_frame), depth=2, name=f"B_{col}")
        b_core = b.cons().split([i * b_frame for i in range(ROWS)], tile=mem, obj_types=[bty] * ROWS,
                                depths=[1] * ROWS)
        c = ObjectFifo(float_type(ROWS * FRAME_VALUES), depth=2, name=f"C_{col}")
        c_core = c.prod().join([i * FRAME_VALUES for i in range(ROWS)], tile=mem, obj_types=[cty] * ROWS,
                               depths=[1] * ROWS)
        for row in range(ROWS):
            has_head = row < GROUP
            mode = Buffer(control_ty, initial_value=np.zeros(control_size, np.int32), use_write_rtp=True)
            controls.append(mode)
            scratch = [Buffer(ty if has_head else byte_type(64)) for ty in (q_ty, kv_ty, o_ty, scores_ty, probs_ty)]
            workers.append(Worker(make_worker(has_head), tile=Tile(col, 2 + row), stack_size=0x2000, fn_args=[
                a_core.cons(), b_core[row].cons(), c_core[row].prod(), mode,
                Buffer(table_ty, initial_value=lookup, address=0xBFC0),
                Buffer(table_ty, initial_value=lookup, address=0xFFC0), *scratch,
                (expert, iq4_pair, copy_query, copy_kv, init, step, emit, emit_zero),
            ]))
        ap.append(a.prod(tile=shim))
        bp.append(b.prod(tile=shim))
        cp.append(c.cons(tile=shim))

    def sequence(first, second, out, a_prods, b_prods, c_conss):
        for mode in controls:
            count = tiles // ACT_GROUP if k else chunks
            if singleton:
                mode[0] = int(bool(k) and iq4)
                mode[1] = int(bool(k) and not iq4)
                mode[2] = int(not k)
                mode[3] = count
                mode[4] = bands
            else:
                mode[0] = int(not k)
                mode[1] = count
        group = TaskGroup()
        for col in range(SLOTS):
            c_conss[col].drain(out, tap=TensorAccessPattern(
                (SLOTS, c_elems), col * c_elems, [1, 1, c_elems // 256, 256], [0, 0, 256, 1]),
                wait=True, group=group)
            if k:
                a_tap = TensorAccessPattern((SLOTS, a_bytes), col * a_bytes, [bands, 1, tiles, ACT_BYTES],
                                            [0, 0, ACT_BYTES, 1])
            else:
                blocks = a_bytes // 256
                inner = max(i for i in range(1, 1024) if blocks % i == 0)
                a_tap = TensorAccessPattern((SLOTS, a_bytes), col * a_bytes, [1, blocks // inner, inner, 256],
                                            [0, inner * 256, 256, 1])
            a_prods[col].fill(second, tap=a_tap, group=group)
            b_prods[col].fill(first, tap=TensorAccessPattern(
                (SLOTS, b_bytes), col * b_bytes, [1, stripe // b_frame, ROWS, b_frame],
                [0, b_frame, stripe, 1]), group=group)
        group.finish()

    runtime = Runtime(sequence, [byte_type(SLOTS * b_bytes), byte_type(SLOTS * a_bytes), float_type(SLOTS * c_elems),
                                 ap, bp, cp])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = shared_decode
SPECIALIZE = {
    "k": int(os.environ.get("INKLING2_SHARED_K", EXPERT_WIDTH)),
    "iq4": os.environ.get("INKLING2_SHARED_FORMAT", "iq4") == "iq4",
    "chunks": int(os.environ.get("INKLING2_SHARED_CHUNKS", "0")),
    "singleton": bool(int(os.environ.get("INKLING2_SHARED_SINGLETON", "0"))),
    "bands": int(os.environ.get("INKLING2_SHARED_BANDS", str(BANDS))),
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + os.environ.get("INKLING2_SHARED_NULL", "").encode()
                                      + b"".join(p.read_bytes() for p in (
        HERE / "shared_decode.cc", HERE / "block_expert.cc", HERE / "iq4_lut.h", HERE / "relative_attention.cc",
        HERE.parents[1] / "include/vecmath.h", HERE.parents[1] / "recipes/inkling2_attention.py",
    ))).hexdigest()[:8], 16),
}
