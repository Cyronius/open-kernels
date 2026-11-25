"""Inkling near-range block attention: 256 query rows x 24 heads x one key window in one dispatch.

Each eight-core group runs one query head per pass; core w owns query rows 32w..32w+31. Two memtiles
join a key (or value) chunk with four per-core relative strips into one broadcast frame, so the host
never repacks K/V per query tile. The host merges each core's FP32 numerator, row maxima and masses.
"""

import os
from pathlib import Path

import numpy as np
import aie.iron as iron
from aie.helpers.taplib import TensorAccessPattern
from aie.iron import (
    Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime, TaskGroup, Worker,
    WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile

HERE = Path(__file__).resolve().parent
ROWS, CHUNK, DIM, REL, STRIP = 32, 16, 128, 16, 48
GROUPS, COLUMNS, COLUMN_CORES = 4, 2, 4
CORES = COLUMNS * COLUMN_CORES
HEADS, KV_HEADS = 24, 8
KEY_BYTES = CHUNK * DIM * 2
STRIP_BYTES = REL * STRIP * 2
PART_BYTES = COLUMN_CORES * STRIP_BYTES
FRAME_BYTES = KEY_BYTES + PART_BYTES
Q_BYTES = (DIM + REL) * ROWS * 2
OUT_WORDS = ROWS * DIM + 2 * ROWS
assert Q_BYTES + 64 <= FRAME_BYTES


def u8(n):
    return np.ndarray[(n,), np.dtype[np.uint8]]


def linear(total, offset, length):
    assert length % 256 == 0
    blocks = length // 256
    inner = max(i for i in range(1, 1024) if blocks % i == 0)
    return TensorAccessPattern((total,), offset, [1, blocks // inner, inner, 256], [0, inner * 256, 256, 1])


@iron.jit(aiecc_flags=['--dynamic-objFifos', '--alloc-scheme=basic-sequential'])
def near_attention(Q: In, KV: In, S: In, O: Out, *, chunks: CompileTime[int]):
    assert 0 < chunks <= 1023
    frame_ty, key_ty, part_ty = u8(FRAME_BYTES), u8(KEY_BYTES), u8(PART_BYTES)
    q_ty = np.ndarray[(Q_BYTES // 2,), np.dtype[np.uint16]]
    score_ty = np.ndarray[(ROWS * CHUNK,), np.dtype[np.float32]]
    state_ty = np.ndarray[(2 * ROWS + 16,), np.dtype[np.float32]]
    out_ty = np.ndarray[(OUT_WORDS,), np.dtype[np.float32]]
    col_ty = np.ndarray[(OUT_WORDS * COLUMN_CORES,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
    obj = ExternalFunction('na_init', source_file=str(HERE / 'near_attention.cc'), arg_types=[out_ty, state_ty],
                           compile_flags=['-Wno-deprecated-declarations', f"-DNA_ABL={os.environ.get('INKLING2_NEAR_ABL', 0)}",
                                          f"-DNA_QK_KEYS={os.environ.get('INKLING2_NEAR_QK_KEYS', 2)}",
                                          f"-DNA_PV_DIMS={os.environ.get('INKLING2_NEAR_PV_DIMS', 4)}",
                                          f"-DNA_EXP_PRECISE={os.environ.get('INKLING2_NEAR_EXP_PRECISE', 1)}"]).object_file_name
    kernels = [Kernel(name, obj, types) for name, types in (
        ('na_capture', [frame_ty, q_ty, state_ty, np.int32]),
        ('na_init', [out_ty, state_ty]),
        ('na_qk', [q_ty, q_ty, frame_ty, frame_ty, score_ty, state_ty, np.int32, np.int32]),
        ('na_sm', [score_ty, state_ty, out_ty, np.int32, np.int32]),
        ('na_pv', [score_ty, frame_ty, out_ty, state_ty]),
        ('na_store', [out_ty, state_ty]),
    )]

    def worker(index):
        def process(a, b, out, barrier, rtp, q_hi, q_lo, scores, state, ks):
            capture, initialize, qk, softmax, pv, store = ks
            barrier.wait_for_value(1)
            for _ in range_(HEADS // GROUPS):
                result = out.acquire(1)
                for i in range(CORES):
                    frame = a.acquire(1)
                    if i == index:
                        capture(frame, q_hi, state, 1)
                    a.release(1)
                for i in range(CORES):
                    frame = b.acquire(1)
                    if i == index:
                        capture(frame, q_lo, state, 0)
                    b.release(1)
                initialize(result, state)
                for c in range_(rtp[0]):
                    key = a.acquire(1)
                    value = b.acquire(1)
                    qk(q_hi, q_lo, key, value, scores, state, index, c)
                    softmax(scores, state, result, index, c)
                    pv(scores, value, result, state)
                    a.release(1)
                    b.release(1)
                store(result, state)
                out.release(1)
            barrier.release_with_value(1)
        return process

    feeds, drains, workers, barriers, counters = [], [], [], [], []
    for group in range(GROUPS):
        column = group * COLUMNS
        paths = []
        for label, c in [('key', column), ('value', column + 1)]:
            pipe = ObjectFifo(frame_ty, name=f'{label}_l2_{c}', depth=2)
            halves = pipe.prod().join([0, KEY_BYTES], tile=Tile(c, 1), obj_types=[key_ty, part_ty], depths=[2, 2],
                                      names=[f'{label}_rows_{c}', f'{label}_strips_{c}'])
            feeds.append([half.prod(tile=Tile(c, 0)) for half in halves])
            paths.append(pipe)
        for c in range(column, column + COLUMNS):
            pipe = ObjectFifo(col_ty, name=f'out_shim_{c}', depth=2)
            drains.append(pipe.cons(tile=Tile(c, 0)))
            parts = pipe.prod().join([r * OUT_WORDS for r in range(COLUMN_CORES)], tile=Tile(c, 1),
                                     obj_types=[out_ty] * COLUMN_CORES, depths=[1] * COLUMN_CORES,
                                     names=[f'out_core_{c}_{r}' for r in range(COLUMN_CORES)])
            for r in range(COLUMN_CORES):
                suffix = f'{c}_{r}'
                barrier = WorkerRuntimeBarrier()
                barriers.append(barrier)
                counter = Buffer(rtp_ty, name=f'chunks_{suffix}', initial_value=np.zeros(8, dtype=np.int32),
                                 use_write_rtp=True)
                counters.append(counter)
                args = [paths[0].cons(depth=1), paths[1].cons(depth=1), parts[r].prod(), barrier, counter,
                        Buffer(q_ty, name=f'qh_{suffix}'), Buffer(q_ty, name=f'ql_{suffix}'),
                        Buffer(score_ty, name=f'scores_{suffix}'), Buffer(state_ty, name=f'state_{suffix}'),
                        kernels]
                workers.append(Worker(worker((c - column) * COLUMN_CORES + r), args, tile=Tile(c, r + 2),
                                      stack_size=0x1400))

    q_total = HEADS * 2 * CORES * FRAME_BYTES
    kv_total = 2 * KV_HEADS * chunks * KEY_BYTES
    s_total = 2 * chunks * PART_BYTES
    o_total = HEADS * CORES * OUT_WORDS

    def sequence(q, kv, s, out, inputs, outputs):
        for counter in counters:
            counter[0] = chunks
        for barrier in barriers:
            barrier.set(1)
        for k in range(HEADS // GROUPS):
            tasks = TaskGroup()
            for g in range(GROUPS):
                h = g * (HEADS // GROUPS) + k
                kvh = h // (HEADS // KV_HEADS)
                for plane in range(2):
                    rows, strips = inputs[g * COLUMNS + plane]
                    base = (h * 2 + plane) * CORES * FRAME_BYTES
                    rows.fill(q, tap=TensorAccessPattern((q_total,), base, [1, CORES, KEY_BYTES // 256, 256],
                                                         [0, FRAME_BYTES, 256, 1]), group=tasks)
                    rows.fill(kv, tap=linear(kv_total, (plane * KV_HEADS + kvh) * chunks * KEY_BYTES,
                                             chunks * KEY_BYTES), group=tasks)
                    strips.fill(q, tap=TensorAccessPattern((q_total,), base + KEY_BYTES, [1, CORES, PART_BYTES // 256, 256],
                                                           [0, FRAME_BYTES, 256, 1]), group=tasks)
                    strips.fill(s, tap=linear(s_total, plane * chunks * PART_BYTES, chunks * PART_BYTES), group=tasks)
                for plane in range(2):
                    offset = (h * CORES + plane * COLUMN_CORES) * OUT_WORDS
                    outputs[g * COLUMNS + plane].drain(out, tap=TensorAccessPattern(
                        (o_total,), offset, [1, 1, COLUMN_CORES * OUT_WORDS // 256, 256], [0, 0, 256, 1]),
                        wait=True, group=tasks)
            tasks.finish()

    runtime = Runtime(sequence, [u8(q_total), u8(kv_total), u8(s_total),
                                np.ndarray[(o_total,), np.dtype[np.float32]], feeds, drains])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = near_attention
SPECIALIZE = dict(chunks=int(os.environ.get('INKLING2_NEAR_CHUNKS', 80)))
