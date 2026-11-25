"""Inkling full attention over older keys with split BFP16 QK and PV.

Each eight-core group computes one head over 256 query rows. The streams
consume complete 4K or 16K older-key segments without a relative-position term.
The host merges each segment's FP32 numerator and softmax state.
"""

import os
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16
import aie.iron as iron
from aie.iron import (
    Buffer, CompileTime, ExternalFunction, In, Kernel, ObjectFifo, Out, Program, Runtime,
    TaskGroup, Worker, WorkerRuntimeBarrier,
)
from aie.iron.controlflow import range_
from aie.iron.device import Tile


HERE = Path(__file__).resolve().parent
rows, chunk, dim = 32, 32, 128
groups, columns, column_cores = 4, 2, 4
cores = columns * column_cores
heads, kv_heads = 24, 8
tile_words = rows * dim
out_words = (rows + 1) * dim
frame_words = tile_words + 512
query_words = (1 + 2 * cores) * frame_words


@iron.jit(aiecc_flags=['--dynamic-objFifos', '--alloc-scheme=basic-sequential'])
def wide_attention(Q: In, KV: In, O: Out, *, length: CompileTime[int],
                   profile: CompileTime[bool], stage: CompileTime[int], matrix: CompileTime[int], bfp_qk: CompileTime[int]):
    assert length >= chunk and length % chunk == 0
    chunks = length // chunk
    assert chunks <= 512
    assert 0 <= stage <= 3 and (profile or stage == 3)
    query_ty = np.ndarray[(tile_words,), np.dtype[bfloat16]]
    input_ty = np.ndarray[(frame_words,), np.dtype[bfloat16]]
    score_ty = np.ndarray[(rows * chunk,), np.dtype[np.float32]]
    prob_ty = np.ndarray[(rows * chunk,), np.dtype[bfloat16]]
    state_ty = np.ndarray[(2 * rows + 16,), np.dtype[np.float32]]
    out_ty = np.ndarray[(out_words,), np.dtype[np.float32]]
    col_ty = np.ndarray[(out_words * column_cores,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(8,), np.dtype[np.int32]]
    flags = ['-DAF_WIDE_OLD=1', '-DAF_BIDIR=1', '-DAF_PSPLIT=1', '-DAF_QSPLIT=1', '-Wno-deprecated-declarations']
    obj = ExternalFunction('af_init', source_file=str(HERE.parent / 'attn_fa/attn_fa.cc'),
                           arg_types=[out_ty, state_ty], include_dirs=[str(HERE)], compile_flags=flags).object_file_name
    kernels = [Kernel(name, obj, types) for name, types in (
        ('af_meta', [input_ty, state_ty]), ('af_capture', [input_ty, query_ty, np.int32]),
        ('af_init', [out_ty, state_ty]), ('af_qk_select', [query_ty, query_ty, input_ty, score_ty, state_ty, np.int32, np.int32, np.int32]),
        ('af_sm', [score_ty, state_ty, out_ty, prob_ty, np.int32, np.int32]),
        ('af_pv_select', [score_ty, input_ty, out_ty, state_ty, prob_ty, np.int32]), ('af_store_state', [out_ty, state_ty]),
    )]

    def worker(index):
        def process(a, b, out, barriers, key_chunks, q_hi, q_lo, scores, state, probabilities, ks):
            meta, capture, initialize, qk, softmax, pv, store = ks
            barriers.wait_for_value(1)
            for _ in range_(heads // groups):
                result = out.acquire(1)
                value = a.acquire(1)
                meta(value, state)
                a.release(1)
                for i in range(cores):
                    value = a.acquire(1)
                    if i == index:
                        capture(value, q_hi, 0)
                    a.release(1)
                for i in range(cores):
                    value = b.acquire(1)
                    if i == index:
                        capture(value, q_lo, 0)
                    b.release(1)
                initialize(result, state)
                for c in range_(key_chunks[0]):
                    key = a.acquire(1)
                    if profile:
                        for _ in range_(key_chunks[1]):
                            qk(q_hi, q_lo, key, scores, state, 0, c, key_chunks[5])
                    else:
                        qk(q_hi, q_lo, key, scores, state, 0, c, key_chunks[5])
                    a.release(1)
                    if profile:
                        for _ in range_(key_chunks[2]):
                            softmax(scores, state, result, probabilities, 0, c)
                    else:
                        softmax(scores, state, result, probabilities, 0, c)
                    value = b.acquire(1)
                    if profile:
                        for _ in range_(key_chunks[3]):
                            pv(scores, value, result, state, probabilities, key_chunks[4])
                    else:
                        pv(scores, value, result, state, probabilities, key_chunks[4])
                    b.release(1)
                store(result, state)
                out.release(1)
            barriers.release_with_value(1)
        return process

    feeds, drains, workers, barriers, counters = [], [], [], [], []
    for group in range(groups):
        column = group * columns
        paths = []
        for label, c in [('key', column), ('value', column + 1)]:
            pipe = ObjectFifo(input_ty, name=f'{label}_shim_{c}', depth=2)
            feeds.append(pipe.prod(tile=Tile(c, 0)))
            paths.append(pipe.cons().forward(tile=Tile(c, 1), obj_type=input_ty, depth=2, name=f'{label}_l2_{c}'))
        for c in range(column, column + columns):
            pipe = ObjectFifo(col_ty, name=f'out_shim_{c}', depth=2)
            drains.append(pipe.cons(tile=Tile(c, 0)))
            parts = pipe.prod().join([r * out_words for r in range(column_cores)], tile=Tile(c, 1),
                                     obj_types=[out_ty] * column_cores, depths=[1] * column_cores,
                                     names=[f'out_core_{c}_{r}' for r in range(column_cores)])
            for r in range(column_cores):
                suffix = f'{c}_{r}'
                barrier = WorkerRuntimeBarrier()
                barriers.append(barrier)
                counter = Buffer(rtp_ty, name=f'key_chunks_{suffix}', initial_value=np.zeros(8, dtype=np.int32),
                                 use_write_rtp=True)
                counters.append(counter)
                args = [paths[0].cons(depth=1), paths[1].cons(depth=1), parts[r].prod(), barrier, counter,
                        Buffer(query_ty, name=f'qh_{suffix}'), Buffer(query_ty, name=f'ql_{suffix}'),
                        Buffer(score_ty, name=f'scores_{suffix}'), Buffer(state_ty, name=f'state_{suffix}'),
                        Buffer(prob_ty, name=f'probs_{suffix}'), kernels]
                workers.append(Worker(worker((c - column) * column_cores + r), args, tile=Tile(c, r + 2), stack_size=0x1800))

    def fill_queries(q, inputs, head, tasks):
        for g in range(groups):
            h = g * (heads // groups) + head
            for plane in range(2):
                channel = inputs[g * columns + plane]
                offset = h * query_words + (0 if plane == 0 else (1 + cores) * frame_words)
                count = (cores + (plane == 0)) * (frame_words // dim)
                channel.fill(q, offset=offset, sizes=[count, dim], strides=[dim, 1], group=tasks)

    def fill_cache(kv, inputs, head, first, wait, tasks):
        count = min(chunks - first, 64)
        for g in range(groups):
            h = g * (heads // groups) + head
            kvh = h // (heads // kv_heads)
            for plane in range(2):
                channel = inputs[g * columns + plane]
                offset = ((plane * kv_heads + kvh) * chunks + first) * frame_words
                channel.fill(kv, offset=offset, sizes=[count, frame_words // dim, dim], strides=[frame_words, dim, 1],
                             wait=wait, group=tasks)

    def drain_state(out, outputs, head, tasks):
        for g in range(groups):
            h = g * (heads // groups) + head
            for plane in range(2):
                offset = (h * cores + plane * column_cores) * out_words
                outputs[g * columns + plane].drain(out, offset=offset, sizes=[column_cores * (rows + 1), dim],
                                                    strides=[dim, 1], wait=True, group=tasks)

    def sequence(q, kv, out, inputs, outputs):
        for counter in counters:
            counter[0] = chunks
            counter[4] = matrix
            counter[5] = bfp_qk
            if profile:
                for index in range(1, 4):
                    counter[index] = int(stage >= index)
        for barrier in barriers:
            barrier.set(1)
        for head in range(heads // groups):
            queries = TaskGroup()
            fill_queries(q, inputs, head, queries)
            if chunks <= 128:
                for first in range(0, chunks, 64):
                    fill_cache(kv, inputs, head, first, False, queries)
                drain_state(out, outputs, head, queries)
                queries.finish()
                continue

            # Bound each input channel to the query fill plus two key slices.
            pending = []
            for first in range(0, chunks, 64):
                window = TaskGroup()
                fill_cache(kv, inputs, head, first, True, window)
                pending.append(window)
                if len(pending) == 2:
                    pending.pop(0).finish()
                    if queries is not None:
                        queries.finish()
                        queries = None
            output = TaskGroup()
            drain_state(out, outputs, head, output)
            for window in pending:
                window.finish()
            if queries is not None:
                queries.finish()
            output.finish()

    runtime = Runtime(sequence, [np.ndarray[(heads * query_words,), np.dtype[bfloat16]],
                                np.ndarray[(2 * kv_heads * chunks * frame_words,), np.dtype[bfloat16]],
                                np.ndarray[(heads * cores * out_words,), np.dtype[np.float32]], feeds, drains])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = wide_attention
SPECIALIZE = dict(length=int(os.environ.get('INKLING2_WIDE_KEYS', 4096)),
                  profile=os.environ.get('INKLING2_WIDE_PROFILE') == '1',
                  stage=int(os.environ.get('INKLING2_WIDE_STAGE', 3)),
                  matrix=int(os.environ.get('INKLING2_TEST_MATRIX_PV', 1)),
                  bfp_qk=int(os.environ.get('INKLING2_TEST_BFP_QK', 1)))
