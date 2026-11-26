r"""attn_fa256: causal flash attention for a 256-token block prefill at head dim 256 (Qwen3.6-35B-A3B).

    O[t, h*256:(h+1)*256] = softmax_causal(Q_h K_kv(h)^T) V_kv(h)     (Q pre-scaled by log2(e)/16)

One dispatch is one 256-token block of one layer against a window of L cached rows; one xclbin serves every L
(one instruction stream per window), so a layer's attention is one hardware context whatever the prompt length.
Scores, mask, softmax and both products stay on the array.

Topology (dit_fa's): 2 groups of 16 cores (columns 0-3, 4-7). A pass is one query head's 256 tokens; each core
owns 16 of them (core i: tokens 16i .. 16i+15) and walks every key chunk of its kv head's window, so there is no
merge. Fifo A (the group's first memtile) carries dims 0-127 of every row block, fifo B (the second) dims
128-255, both broadcast to the 16 cores: a chunk's K half then its V half on each, so both streams work during
both products. L3 -> memtile moves [64, 128] row blocks; the memtile emits each as four [64, 32] head-dim pieces
in mmul block order. The pass's record and Q ride both fifos ahead of the keys; each core keeps its own rows.
Group g takes kv head g (GQA 8): query heads 8g .. 8g+7, one pass each.

Buffers (the engine's argument order):
  Q   bf16 [heads][64 + 256][256]: per head the record rows (p0, the block's first position, as the int32 at
      [h][0][0:2]; token t's causal bound is p0 + t), then the head's 256 token rows, scaled by log2(e)/16
  KV  the layer's KV cache, bf16 rows of kv_ld elements: K at kvh*256, V at kv_heads*256 + kvh*256
  O   fp32 [256 tokens, heads*256] row-major (the gate and out-projection's input)

L1 per core: A and B fifos 3 x 4 KB each, O 16 KB (fp32), Q 8 KB, G 4 KB, state 1 KB, stack.

    AF_L=1024 python open_kernels/build_design.py open_kernels/designs/attn_fa256/attn_fa256.py <out>
"""

from __future__ import annotations

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
KERNEL_SRC = str(HERE / "attn_fa256.cc")

T, TQ, LKP, DC, HD = 256, 16, 64, 32, 256
HALF = HD // 2                         # head dims a fifo carries
NDCH = HD // DC
HCH = HALF // DC                       # L1 elements a fifo delivers per row block
REC = 64                               # record rows ahead of each head's Q
N_GROUPS, GROUP_COLS, N_ROWS = 2, 4, 4
CORES = GROUP_COLS * N_ROWS            # per group
Q_BLOCKS = T // LKP                    # [64, 256] Q blocks a pass: 4 cores' rows each
ADEPTH = int(os.environ.get("AF_ADEPTH", 3))
BDEPTH = int(os.environ.get("AF_BDEPTH", 3))
MAX_INFLIGHT = int(os.environ.get("AF_INFLIGHT", 2))
STACK = int(os.environ.get("AF_STACK", 0x1000))
TAU = int(os.environ.get("AF_TAU", 16))
ABL = int(os.environ.get("AF_ABL", 0))            # attn_fa256.cc AF_ABL, timing ablations only
RTP_PASSES, RTP_CHUNKS = 0, 1
RTP_LEN = 8
ST_LEN = 2 * TQ * 8 + 16

assert CORES * TQ == T


def q_head_elems() -> int:
    return (REC + T) * HD


def check_shape(L: int, heads: int, kv_heads: int, max_rows: int | None = None) -> str | None:
    if L <= 0 or L % 256 or heads % N_GROUPS or heads % kv_heads or kv_heads != N_GROUPS:
        return (f"attn_fa256 needs positive L % 256, heads % {N_GROUPS}, {N_GROUPS} kv heads "
                f"(got L={L} heads={heads} kv_heads={kv_heads})")
    if max_rows is not None and L > max_rows:
        return f"attn_fa256: window L={L} is past the cache's {max_rows} rows"
    return None


def _block_dims(rows, cols, blk=8):
    """Row-major [rows, cols] -> the 8x8-block column-major order the mmul reads ([col block][row block][8][8])."""
    return [(cols // blk, blk), (rows // blk, blk * cols), (blk, cols), (blk, 1)]


@iron.jit(aiecc_flags=["--dynamic-objFifos", "--alloc-scheme=basic-sequential"])
def attn_fa256(Q: In, KV: In, O: Out, *, L: CompileTime[int], heads: CompileTime[int],
               kv_heads: CompileTime[int], kv_ld: CompileTime[int], max_rows: CompileTime[int]):
    why = check_shape(L, heads, kv_heads, max_rows)
    assert why is None, why
    hpg = heads // N_GROUPS
    qw = heads * HD
    qh_elems = q_head_elems()
    n_chunks = L // LKP

    blk_ty = np.ndarray[(LKP, HALF), np.dtype[bfloat16]]     # L2: half the dims of 64 keys (or 4 cores' Q rows)
    elem_ty = np.ndarray[(LKP, DC), np.dtype[bfloat16]]      # L1: one head-dim chunk of it
    q_ty = np.ndarray[(NDCH * TQ * DC,), np.dtype[bfloat16]]
    g_ty = np.ndarray[(TQ * LKP,), np.dtype[np.float32]]
    st_ty = np.ndarray[(ST_LEN,), np.dtype[np.float32]]
    o_l1_ty = np.ndarray[(TQ * HD,), np.dtype[np.float32]]
    o_l2_ty = np.ndarray[(N_ROWS * TQ * HD,), np.dtype[np.float32]]
    rtp_ty = np.ndarray[(RTP_LEN,), np.dtype[np.int32]]

    flags = [f"-DAF_TAU={TAU}", f"-DAF_ABL={ABL}", "-DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16",
             "-Wno-deprecated-declarations"]
    _ef = ExternalFunction("af_init", source_file=KERNEL_SRC, arg_types=[o_l1_ty, st_ty], compile_flags=flags)
    obj = _ef.object_file_name
    k_meta = Kernel("af_meta", obj, [elem_ty, st_ty])
    k_cap = Kernel("af_capture", obj, [elem_ty, q_ty, np.int32, np.int32])
    k_init = Kernel("af_init", obj, [o_l1_ty, st_ty])
    k_qk = Kernel("af_qk", obj, [q_ty, elem_ty, g_ty, st_ty, np.int32, np.int32, np.int32])
    k_sm = Kernel("af_sm", obj, [g_ty, st_ty, o_l1_ty, np.int32, np.int32])
    k_pv = Kernel("af_pv", obj, [g_ty, elem_ty, o_l1_ty, st_ty, np.int32])
    k_fin = Kernel("af_fin", obj, [o_l1_ty, st_ty])
    kernels = [k_meta, k_cap, k_init, k_qk, k_sm, k_pv, k_fin]

    def core_tile(g, i):
        return Tile(g * GROUP_COLS + i // N_ROWS, 2 + i % N_ROWS)

    A_l3l2, B_l3l2, A_l1, B_l1 = [], [], [], []
    for g in range(N_GROUPS):
        fa = ObjectFifo(blk_ty, name=f"A_L3L2_{g}", depth=2)
        fb = ObjectFifo(blk_ty, name=f"B_L3L2_{g}", depth=2)
        A_l3l2.append(fa)
        B_l3l2.append(fb)
        # over a whole [64, 128] block this is [d chunk][d block][row block][8][8]: four chunks back to back
        a2 = fa.cons().forward(tile=Tile(g * GROUP_COLS, 1), obj_type=elem_ty, depth=2, name=f"A_L2L1_{g}",
                               dims_to_stream=_block_dims(LKP, HALF))
        b2 = fb.cons().forward(tile=Tile(g * GROUP_COLS + 1, 1), obj_type=elem_ty, depth=2, name=f"B_L2L1_{g}",
                               dims_to_stream=_block_dims(LKP, HALF))
        A_l1.append([a2.cons(depth=ADEPTH) for _ in range(CORES)])
        B_l1.append([b2.cons(depth=BDEPTH) for _ in range(CORES)])

    # O: a column's 4 cores (16 rows each, block layout) join at its memtile, which streams each part out
    # row-major (a join's dims apply per part), so the column's 64 rows leave in order
    out_dims = [(TQ, 8), (HD // 8, TQ * 8), (8, 1)]
    O_l2l3, O_l1 = [], {}
    for c in range(N_GROUPS * GROUP_COLS):
        f = ObjectFifo(o_l2_ty, name=f"O_L2L3_{c}", depth=2, dims_to_stream=out_dims)
        O_l2l3.append(f)
        parts = f.prod().join([r * TQ * HD for r in range(N_ROWS)], tile=Tile(c, 1),
                              obj_types=[o_l1_ty] * N_ROWS, names=[f"O_L1L2_{c}_{r}" for r in range(N_ROWS)],
                              depths=[1] * N_ROWS)
        for r in range(N_ROWS):
            O_l1[(c, r)] = parts[r]

    rtp, barriers = {}, {}

    def make_core_fn(i):
        tok0 = i * TQ

        def core_fn(a_in, b_in, o_out, kern, q_buf, g_buf, st, my_rtp, barrier):
            (meta, cap, init, qk, sm, pv, fin) = kern
            barrier.wait_for_value(1)
            n_pass = my_rtp[RTP_PASSES]
            n_ch = my_rtp[RTP_CHUNKS]
            for _ in range_(n_pass):
                o = o_out.acquire(1)
                # every row block comes as dims 0-127 on A and 128-255 on B, so both streams work in each phase
                for d in range(HCH):
                    for f in (a_in, b_in):
                        e = f.acquire(1)
                        if d == 0 and f is a_in:
                            meta(e, st)
                        f.release(1)
                for b in range(Q_BLOCKS):
                    for d in range(HCH):
                        for h, f in enumerate((a_in, b_in)):
                            e = f.acquire(1)
                            if b == i // 4:
                                cap(e, q_buf, h * HCH + d, i % 4)
                            f.release(1)
                init(o, st)
                for c in range_(n_ch):
                    for d in range(HCH):
                        for h, f in enumerate((a_in, b_in)):
                            k = f.acquire(1)
                            qk(q_buf, k, g_buf, st, h * HCH + d, tok0, c)
                            f.release(1)
                    sm(g_buf, st, o, tok0, c)
                    for d in range(HCH):
                        for h, f in enumerate((a_in, b_in)):
                            v = f.acquire(1)
                            pv(g_buf, v, o, st, h * HCH + d)
                            f.release(1)
                fin(o, st)
                o_out.release(1)
            barrier.release_with_value(1)
        return core_fn

    workers = []
    for g in range(N_GROUPS):
        for i in range(CORES):
            t = core_tile(g, i)
            c, r = t.col, t.row - 2
            rtp[(g, i)] = Buffer(rtp_ty, name=f"rtp_{g}_{i}", initial_value=np.zeros(RTP_LEN, dtype=np.int32),
                                 use_write_rtp=True)
            barriers[(g, i)] = WorkerRuntimeBarrier()
            args = [A_l1[g][i], B_l1[g][i], O_l1[(c, r)].prod(), kernels,
                    Buffer(q_ty, name=f"q_{g}_{i}"), Buffer(g_ty, name=f"g_{g}_{i}"),
                    Buffer(st_ty, name=f"st_{g}_{i}"), rtp[(g, i)], barriers[(g, i)]]
            workers.append(Worker(make_core_fn(i), args, tile=t, stack_size=STACK))

    Q_ty = np.ndarray[(heads * qh_elems,), np.dtype[bfloat16]]
    KV_ty = np.ndarray[(max_rows * kv_ld,), np.dtype[bfloat16]]
    O_ty = np.ndarray[(T * qw,), np.dtype[np.float32]]

    def kv_fill(prod, kv, col, chunk0, count, tg, wait=False):
        # per chunk the K rows' half, then the V rows' half (V sits kv_heads * HD after K)
        prod.fill(kv, offset=chunk0 * LKP * kv_ld + col, sizes=[count, 2, LKP, HALF],
                  strides=[LKP * kv_ld, kv_heads * HD, kv_ld, 1], wait=wait, group=tg)

    def o_drain(o_conses, o, g, h, tg):
        for cc in range(GROUP_COLS):
            col = g * GROUP_COLS + cc
            row0 = cc * N_ROWS * TQ
            o_conses[col].drain(o, offset=row0 * qw + h * HD, sizes=[N_ROWS * TQ, HD], strides=[qw, 1],
                                wait=True, group=tg)

    def sequence(q, kv, o, a_prods, b_prods, o_conses):
        for buf in rtp.values():
            buf[RTP_PASSES] = hpg
            buf[RTP_CHUNKS] = n_chunks
        for b in barriers.values():
            b.set(1)
        # Each channel queues Q plus at most two 64-chunk slices (R-0001: under 4 BDs a channel); the oldest slice
        # is awaited before the next is queued, so a window of any length fits the BD pool. A shim BD's outer count
        # wraps above 64, hence the slices.
        for k in range(hpg):
            tq = TaskGroup()
            for g in range(N_GROUPS):
                for half, prods in enumerate((a_prods, b_prods)):
                    prods[g].fill(q, offset=(g * hpg + k) * qh_elems + half * HALF, sizes=[REC + T, HALF],
                                  strides=[HD, 1], group=tq)
            pending: list[TaskGroup] = []
            for chunk0 in range(0, n_chunks, 64):
                count = min(64, n_chunks - chunk0)
                ts = TaskGroup()
                for g in range(N_GROUPS):
                    kv_fill(a_prods[g], kv, g * HD, chunk0, count, ts, wait=True)
                    kv_fill(b_prods[g], kv, g * HD + HALF, chunk0, count, ts, wait=True)
                pending.append(ts)
                if len(pending) == 2:
                    pending.pop(0).finish()
                    if tq is not None:      # Q went ahead of the first slice
                        tq.finish()
                        tq = None
            td = TaskGroup()
            for g in range(N_GROUPS):
                o_drain(o_conses, o, g, g * hpg + k, td)
            for ts in pending:
                ts.finish()
            if tq is not None:
                tq.finish()
            td.finish()

    a_prods = [A_l3l2[g].prod(tile=Tile(g * GROUP_COLS, 0)) for g in range(N_GROUPS)]
    b_prods = [B_l3l2[g].prod(tile=Tile(g * GROUP_COLS + 1, 0)) for g in range(N_GROUPS)]
    o_conses = [O_l2l3[c].cons(tile=Tile(c, 0)) for c in range(N_GROUPS * GROUP_COLS)]
    rt = Runtime(sequence, [Q_ty, KV_ty, O_ty, a_prods, b_prods, o_conses])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = attn_fa256
SPECIALIZE = dict(
    L=int(os.environ.get("AF_L", 1024)),
    heads=int(os.environ.get("AF_HEADS", 16)),
    kv_heads=int(os.environ.get("AF_KV_HEADS", 2)),
    kv_ld=int(os.environ.get("AF_KV_LD", 1024)),
    max_rows=int(os.environ.get("AF_MAX_ROWS", 4096)),
)
