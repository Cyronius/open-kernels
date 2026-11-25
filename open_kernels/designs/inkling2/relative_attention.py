"""Inkling causal attention with learned relative logits and a per-query sink.

Eight columns stream one KV head each. Three workers per column consume the
same KV records and use independent query heads. Each worker returns its
online numerator, maximum and denominator for a bounded key range.
"""
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
from ironutil import include_dirs  # noqa: E402
from recipes.inkling2_attention import GROUP, KV_BYTES, KV_HEADS, OUT_FLOATS, QUERY_BYTES, TK, TQ  # noqa: E402


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def relative_attention(q: In, kv: In, o: Out, *, chunks: CompileTime[int], source_hash: CompileTime[int] = 0):
    if not 1 <= chunks <= 128:
        raise ValueError("expected 1–128 key chunks; merge multiple ranges on the host")

    def bytes_type(count):
        return np.ndarray[(count,), np.dtype[np.uint8]]

    def floats_type(count):
        return np.ndarray[(count,), np.dtype[np.float32]]

    q_ty, kv_ty, o_ty = bytes_type(QUERY_BYTES), bytes_type(KV_BYTES), floats_type(OUT_FLOATS)
    scores_ty = floats_type(TQ * TK)
    probs_ty = np.ndarray[(TK * 64,), np.dtype[bfloat16]]
    control_ty = np.ndarray[(1,), np.dtype[np.int32]]
    source = str(HERE / "relative_attention.cc")
    flags = ["-Os", "-DVECMATH_COMPACT"]
    external = ExternalFunction("inkling_attn_init", source_file=source, arg_types=[q_ty, o_ty],
                                include_dirs=include_dirs(), compile_flags=flags)
    init = Kernel("inkling_attn_init", external.object_file_name, [q_ty, o_ty])
    step = Kernel("inkling_attn_chunk", external.object_file_name, [q_ty, kv_ty, o_ty, scores_ty, probs_ty, np.int32])

    def work(q_fifo, kv_fifo, out_fifo, count, scores, probs, init_fn, step_fn):
        query = q_fifo.acquire(1)
        output = out_fifo.acquire(1)
        init_fn(query, output)
        for i in range_(count[0]):
            keys = kv_fifo.acquire(1)
            step_fn(query, keys, output, scores, probs, i)
            kv_fifo.release(1)
        out_fifo.release(1)
        q_fifo.release(1)

    q_prods, kv_prods, o_conss, workers, counts = [], [], [], [], []
    for col in range(KV_HEADS):
        mem = Tile(col, 1)
        shim = Tile(col, 0)
        q_ddr = ObjectFifo(bytes_type(GROUP * QUERY_BYTES), depth=2, name=f"Q_{col}")
        q_core = q_ddr.cons().split([i * QUERY_BYTES for i in range(GROUP)], tile=mem,
                                    obj_types=[q_ty] * GROUP, depths=[1] * GROUP)
        kv_ddr = ObjectFifo(kv_ty, depth=2, name=f"KV_{col}")
        kv_core = kv_ddr.cons().forward(tile=mem, obj_type=kv_ty, depth=1)
        o_ddr = ObjectFifo(floats_type(GROUP * OUT_FLOATS), depth=2, name=f"O_{col}")
        o_core = o_ddr.prod().join([i * OUT_FLOATS for i in range(GROUP)], tile=mem,
                                   obj_types=[o_ty] * GROUP, depths=[1] * GROUP)
        for i in range(GROUP):
            count = Buffer(control_ty, initial_value=np.zeros(1, np.int32), use_write_rtp=True)
            counts.append(count)
            workers.append(Worker(work, tile=Tile(col, 2 + i), stack_size=0x2000, fn_args=[
                q_core[i].cons(), kv_core.cons(), o_core[i].prod(), count,
                Buffer(scores_ty), Buffer(probs_ty), init, step,
            ]))
        q_prods.append(q_ddr.prod(tile=shim))
        kv_prods.append(kv_ddr.prod(tile=shim))
        o_conss.append(o_ddr.cons(tile=shim))

    def linear(length, col, quantum):
        assert length % quantum == 0
        blocks = length // quantum
        inner = max(i for i in range(1, 1024) if blocks % i == 0)
        return TensorAccessPattern((KV_HEADS, length), col * length,
                                   [1, blocks // inner, inner, quantum], [0, inner * quantum, quantum, 1])

    def sequence(q_buf, kv_buf, out_buf, qp, kp, op):
        for count in counts:
            count[0] = chunks
        group = TaskGroup()
        for col in range(KV_HEADS):
            op[col].drain(out_buf, tap=linear(GROUP * OUT_FLOATS, col, 32), wait=True, group=group)
            qp[col].fill(q_buf, tap=linear(GROUP * QUERY_BYTES, col, 128), group=group)
            kp[col].fill(kv_buf, tap=linear(chunks * KV_BYTES, col, 128), group=group)
        group.finish()

    runtime = Runtime(sequence, [bytes_type(KV_HEADS * GROUP * QUERY_BYTES), bytes_type(KV_HEADS * chunks * KV_BYTES),
                                 floats_type(KV_HEADS * GROUP * OUT_FLOATS), q_prods, kv_prods, o_conss])
    return Program(iron.get_current_device(), runtime, workers=workers).resolve_program()


DESIGN = relative_attention
SPECIALIZE = {
    "chunks": int(os.environ.get("INKLING2_ATTN_CHUNKS", "1")),
    "source_hash": int(hashlib.sha256((HERE / "relative_attention.cc").read_bytes() +
                                      (HERE.parents[1] / "include/vecmath.h").read_bytes()).hexdigest()[:8], 16),
}
