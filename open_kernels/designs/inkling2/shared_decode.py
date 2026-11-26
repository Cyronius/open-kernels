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
UP_K, DOWN_K, PAIRS = 2304, 1152, BANDS // 2


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def shared_decode(w: In, x: In, y: Out, *, k: CompileTime[int], iq4: CompileTime[bool],
                  chunks: CompileTime[int], singleton: CompileTime[bool] = False,
                  bands: CompileTime[int] = BANDS, moe: CompileTime[bool] = False, source_hash: CompileTime[int] = 0):
    if moe:
        # Gate/up, SiTU and down of one token in one launch; the down leg sets the band geometry.
        if not singleton or chunks or bands != BANDS:
            raise ValueError("the fused expert stream needs the singleton image")
        k = DOWN_K
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
    ex_args = [aty, bty, table_ty, table_ty, cty] + ([cty] if singleton else []) + [np.int32, np.int32]
    symbol = "inkling_shared_project" if singleton else "inkling_shared_expert"
    null = bool(os.environ.get("INKLING2_SHARED_NULL"))  # timing builds only: all DMA, no arithmetic
    flags = [os.environ.get("INKLING2_SHARED_OPT", "-O2"), "-DBLOCK_ROWS=4"]
    flags += ["-DINKLING2_SINGLETON", "-DVECMATH_COMPACT"] if singleton else [f"-DIQ4_WEIGHTS={int(iq4)}"]
    flags += ["-DEXPERT_NULL"] if null else []
    external = ExternalFunction(symbol, source_file=str(HERE / "shared_decode.cc"), arg_types=ex_args,
                                include_dirs=include_dirs(), compile_flags=flags)
    eo = external.object_file_name
    expert = Kernel(symbol, eo, ex_args)
    situ = Kernel("inkling_shared_situ", eo, [cty, cty, np.int32]) if singleton else None
    copy_query = Kernel("inkling_shared_query", eo, [bty, q_ty, np.int32])
    copy_kv = Kernel("inkling_shared_kv", eo, [aty, kv_ty, np.int32])
    emit = Kernel("inkling_shared_output", eo, [o_ty, cty, np.int32])
    emit_zero = Kernel("inkling_shared_zero", eo, [cty])
    external = ExternalFunction("inkling_attn_init", source_file=str(HERE / "relative_attention.cc"),
                                arg_types=[q_ty, o_ty], include_dirs=include_dirs(),
                                compile_flags=[os.environ.get("INKLING2_SHARED_OPT", "-O2"), "-DVECMATH_COMPACT",
                                               "-DINKLING2_ALIGNED_CHUNKS"]
                                              + (["-DINKLING2_EXP32"] if singleton else [])
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

    def singleton_band(a, b, result, scratch, held, count, records, flags, project, lut0, lut1):
        # `held` is the band's first B record, already acquired. Core-side B has depth 1, so every
        # acquire returns this same buffer.
        activations = a.acquire(1)
        project(activations, held, lut0, lut1, result, scratch, 0, flags + 1)
        b.release(1)
        for i in range_(1, records):
            b.acquire(1)
            project(activations, held, lut0, lut1, result, scratch, i, flags)
            b.release(1)
        a.release(1)
        for _ in range_(count - 1):
            activations = a.acquire(1)
            for i in range_(records):
                b.acquire(1)
                project(activations, held, lut0, lut1, result, scratch, i, flags)
                b.release(1)
            a.release(1)

    def make_worker(is_query_head):
        def work(a, b, c, mode, lut0, lut1, query, kv, result, scores, probs, *rest):
            gate, halves, kernels = rest if singleton else (None, None, rest[0])
            expert_fn, situ_fn, copy_q, copy_k, init_fn, step_fn, emit_fn, zero_fn = kernels
            first = b.acquire(1)
            # The first B record always arrives after the RTP writes. This orders
            # the mode read on every launch without a core reset or barrier lock.
            if singleton:
                kind, count, band_count, flags, fused = mode[0], mode[1], mode[2], mode[3], mode[4]
                records = 4 - flags  # 4 Q8_0 records or 2 IQ4_NL records per activation frame
                for _ in range_(fused):
                    for _ in range_(PAIRS):
                        frame = c.acquire(1)
                        # The gate band goes to the scratch (flag 4), its up band to the frame.
                        for half in range_(2):
                            for _ in range_(half):
                                b.acquire(1)
                            singleton_band(a, b, frame, gate, first, UP_K // TK // ACT_GROUP, records,
                                           flags + halves[half], expert_fn, lut0, lut1)
                        for t in range_(3):
                            situ_fn(gate, frame, t)
                        c.release(1)
                        # After the last pair this waits for the down leg, which needs the frame drained first.
                        b.acquire(1)
                for _ in range_(1 - kind):
                    frame = c.acquire(1)
                    singleton_band(a, b, frame, gate, first, count, records, flags, expert_fn, lut0, lut1)
                    c.release(1)
                    for _ in range_(band_count - 1):
                        b.acquire(1)
                        frame = c.acquire(1)
                        singleton_band(a, b, frame, gate, first, count, records, flags, expert_fn, lut0, lut1)
                        c.release(1)
            else:
                kind, count = mode[0], mode[1]
                for _ in range_(1 - kind):
                    expert_band(a, b, c, first, count, lut0, lut1, expert_fn, 1)
                    for _ in range_(BANDS - 1):
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
            if singleton:
                scratch += [Buffer(cty), Buffer(np.ndarray[(16,), np.dtype[np.int32]],
                                                 initial_value=np.array([4] + [0] * 15, np.int32))]
            workers.append(Worker(make_worker(has_head), tile=Tile(col, 2 + row), stack_size=0x2000, fn_args=[
                a_core.cons(), b_core[row].cons(), c_core[row].prod(), mode,
                Buffer(table_ty, initial_value=lookup, address=0xBFC0),
                Buffer(table_ty, initial_value=lookup, address=0xFFC0), *scratch,
                (expert, situ, copy_query, copy_kv, init, step, emit, emit_zero),
            ]))
        ap.append(a.prod(tile=shim))
        bp.append(b.prod(tile=shim))
        cp.append(c.cons(tile=shim))

    def set_modes():
        for mode in controls:
            count = tiles // ACT_GROUP if k else chunks
            if singleton:
                mode[0] = int(not k)
                mode[1] = count
                mode[2] = bands
                mode[3] = 2 * int(iq4)
                mode[4] = int(moe)
            else:
                mode[0] = int(not k)
                mode[1] = count

    def weight_tap(total, col, per_slot, slot_stripe):
        return TensorAccessPattern((SLOTS, total // SLOTS), col * per_slot, [1, slot_stripe // b_frame, ROWS, b_frame],
                                   [0, b_frame, slot_stripe, 1])

    def sequence(first, second, out, a_prods, b_prods, c_conss):
        set_modes()
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
            b_prods[col].fill(first, tap=weight_tap(SLOTS * b_bytes, col, b_bytes, stripe), group=group)
        group.finish()

    # Fused stream buffers: gate/up pool, one packed token, down pool, then [y of all slots | SiTU wire].
    up_tiles = UP_K // TK
    up_stripe = BANDS * (up_tiles // 2 if iq4 else up_tiles) * b_frame
    x_bytes = up_tiles * ACT_BYTES
    wire_elems = ROWS * PAIRS * FRAME_VALUES
    out_elems = SLOTS * (c_elems + wire_elems)

    def fused_sequence(up_weights, token, down_weights, out, a_prods, b_prods, c_conss):
        set_modes()
        group = TaskGroup()
        for col in range(SLOTS):
            # Core r's band pair j holds K32 tiles 3(3r + j) to 3(3r + j) + 2: frame slot 3r + j.
            c_conss[col].drain(out, tap=TensorAccessPattern(
                (out_elems,), SLOTS * c_elems + col * wire_elems, [1, PAIRS, ROWS, FRAME_VALUES],
                [0, FRAME_VALUES, PAIRS * FRAME_VALUES, 1]), wait=True, group=group)
            a_prods[col].fill(token, tap=TensorAccessPattern((x_bytes,), 0, [BANDS, 1, up_tiles, ACT_BYTES],
                                                             [0, 0, ACT_BYTES, 1]), group=group)
            b_prods[col].fill(up_weights, tap=weight_tap(SLOTS * ROWS * up_stripe, col, ROWS * up_stripe, up_stripe),
                              group=group)
        group.finish()
        # The down leg reads the wire back through DDR only after its drain has finished.
        group = TaskGroup()
        for col in range(SLOTS):
            c_conss[col].drain(out, tap=TensorAccessPattern(
                (out_elems,), col * c_elems, [1, 1, c_elems // 256, 256], [0, 0, 256, 1]), wait=True, group=group)
            a_prods[col].fill(out, tap=TensorAccessPattern(
                (out_elems,), SLOTS * c_elems + col * wire_elems, [BANDS, 1, ROWS * PAIRS, 3 * ACT_BYTES // 4],
                [0, 0, FRAME_VALUES, 1]), group=group)
            b_prods[col].fill(down_weights, tap=weight_tap(SLOTS * b_bytes, col, b_bytes, stripe), group=group)
        group.finish()

    if moe:
        runtime = Runtime(fused_sequence, [byte_type(SLOTS * ROWS * up_stripe), byte_type(x_bytes),
                                           byte_type(SLOTS * b_bytes), float_type(out_elems), ap, bp, cp])
    else:
        runtime = Runtime(sequence, [byte_type(SLOTS * b_bytes), byte_type(SLOTS * a_bytes),
                                     float_type(SLOTS * c_elems), ap, bp, cp])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = shared_decode
SPECIALIZE = {
    "k": int(os.environ.get("INKLING2_SHARED_K", EXPERT_WIDTH)),
    "iq4": os.environ.get("INKLING2_SHARED_FORMAT", "iq4") == "iq4",
    "chunks": int(os.environ.get("INKLING2_SHARED_CHUNKS", "0")),
    "singleton": bool(int(os.environ.get("INKLING2_SHARED_SINGLETON", "0"))),
    "bands": int(os.environ.get("INKLING2_SHARED_BANDS", str(BANDS))),
    "moe": bool(int(os.environ.get("INKLING2_SHARED_MOE", "0"))),
    "source_hash": int(hashlib.sha256(IQ4_NL_VALUES.tobytes() + os.environ.get("INKLING2_SHARED_NULL", "").encode()
                                      + b"".join(p.read_bytes() for p in (
        HERE / "shared_decode.cc", HERE / "block_expert.cc", HERE / "iq4_lut.h", HERE / "relative_attention.cc",
        HERE.parents[1] / "include/vecmath.h", HERE.parents[1] / "recipes/inkling2_attention.py",
    ))).hexdigest()[:8], 16),
}
