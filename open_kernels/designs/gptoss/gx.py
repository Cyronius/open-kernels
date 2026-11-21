r"""gx: one GPT-OSS layer as two dispatches over dx.py's fabric (recipes/gptoss.py).

  GX_PART=att  ln -> q | k | v (+bias) -> YaRN RoPE -> attention with the learned sink -> o (+bias)
               -> res = x + o; xm = post_attention_norm(res)
               args: pool, xres f32[HID] (in), consts [lnw | postln | qn kn sinks], kv, act, ptab
  GX_PART=moe  per routed expert e: up | gate (+bias) -> w_e * clamped SwiGLU -> h_e; down over the
               four experts into each core's accumulator; xres = res + down + bsum
               args: pool, xres f32[HID] (out), act

The host routes between the two (core.cpp's gptoss_route): it reads xm, writes the four routing
weights beside it (act AD_W, inside xm's last x element) and the weighted down biases (AD_BSUM),
and patches this stream's expert fills, compiled as experts 0..3 (moeroute2).

dx.py's fabric: 8 main cores (Tile(c, 2)) on w (10 KB weight elements) / x (4 KB broadcast) /
y (64 floats) streams, the norm helper (Tile(0, 3)), and for attention 4 cores of 16 heads each
(Tile(2 + c, 3)) on one broadcast `ain`. The attention's GEMVs are gemv_q8.h's half-tile with the
bias in its pad (GEMV_Q8_PAD_BIAS); the experts' are gemv_mx4.h's (the GGUF's MXFP4, 9.25 KB w
elements of four half-tiles) unless the recipe keeps them at q8 (GX_MX4=0). The TUs come from
gen_kernels.py.

Build: OPEN_KERNELS_SPEC=<gptoss spec> GX_PART=att|moe python build_design.py designs/gptoss/gx.py <out>
"""
from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np
from ml_dtypes import bfloat16

import aie.iron as iron
from aie.iron import (Buffer, CompileTime, In, InOut, ObjectFifo, Program, Runtime, TaskGroup, Worker,
                      WorkerRuntimeBarrier)
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.iron.kernel import ExternalFunction
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
LN = HERE.parent / "ln"
LINL = HERE.parent / "lin_layer"
sys.path.insert(0, str(HERE.parent.parent))
from ironutil import Pipeline, include_dirs  # noqa: E402
from recipes.attnknobs import kv_slices  # noqa: E402
from recipes.load import current_spec  # noqa: E402
from recipes import gptoss as GR  # noqa: E402
from recipes.qwen36moe import BAND_ROWS, ELEM  # noqa: E402

SPEC = current_spec()
R = GR.recipe(SPEC)
L, G = R.layout, R.geo
PART = os.environ.get("GX_PART", "att")
assert PART in ("att", "moe"), f"GX_PART={PART!r} (att | moe)"
HID, QW, KVW, FF, HD = G.HID, G.QW, G.KVW, G.FF, G.HD
RW = GR.ROWS
assert RW in (1, 2), f"GX_ROWS={RW} (1 | 2)"
N_CORES, ACORES, NHL = G.N_CORES, G.ACORES, G.NHL
ELN, E_A = L.ELN, G.E_A
OS = ["-Os"]
PROBES = sorted(GR.probe_env().items())
ATTN_FLAGS = GR.attn_flags(G, [(k, v) for k, v in PROBES if k.startswith("ATTN_")])
LN_FLAGS = GR.ln_flags(G)
NULL = ["-DGEMV_NULL"] if os.environ.get("GX_NULL") else []
OGH = min(NHL, G.HPO)                  # heads per og element (attn.h kOGH)
N_OG = NHL // OGH
YB = BAND_ROWS * 4
KVSL = G.KVSLICE > 0                   # per-core windows on their own shim channels (OPEN-ATTN-KVSLICE)
assert G.EXPERT_CALL_BYTES == G.CALL_BYTES, "gx.py: one image serves both streams, so a w element is one size"
assert G.XN_ELEMS == G.XM_ELEMS == 2
if KVSL:
    NSHIM = 8
    _mm2s = [0] * NSHIM                # MM2S producers already on each shim column: lni, w_c, x, ain
    for _c in [0, 1, 2] + list(range(N_CORES)):
        _mm2s[_c] += 1
    KW_COLS = [c for c in range(NSHIM) if _mm2s[c] < 2]
    KV_SHARE = 1                       # cores of KV_SHARE kv heads read one stream when channels run short
    while len(set(kv_slices(G.NH, G.KVH, ACORES, KV_SHARE)[1])) > len(KW_COLS) and G.KVH % (2 * KV_SHARE) == 0:
        KV_SHARE *= 2
    KVS, KVB = kv_slices(G.NH, G.KVH, ACORES, KV_SHARE)
    KW_GROUPS = sorted(set(KVB))
    assert len(KW_COLS) >= len(KW_GROUPS), f"gx.py KVSLICE: {len(KW_GROUPS)} window streams, free columns {KW_COLS}"
    ATTN_FLAGS += [f"-DATTN_KVS={KVS}", "-DATTN_INTFP=1", "-DATTN_EXP32=1"]
    if KV_SHARE > 1:
        ATTN_FLAGS.append("-DATTN_KVB_ALIGN=1")
    E_W = G.RB * 2 * KVS * HD * 2      # one window element: RB rows of [K slice | V slice]


def pb(K):
    """half-tiles per 64-row band"""
    return K // 64


def ng(K):
    """w elements per band"""
    return pb(K) // G.PER_CALL


def bt(total, off, n):
    return TensorAccessPattern((1, total), off, [1, 1, 1, n], [0, 0, 0, 1])


BB_H, BB_Q, BB_F = GR.band_bytes(HID), GR.band_bytes(QW), GR.band_bytes(FF)
XBB_H, XBB_F = GR.band_bytes(HID, G.EXPERT_CH), GR.band_bytes(FF, G.EXPERT_CH)   # the experts' bands


def xng(K):
    """expert w elements per band"""
    return pb(K) // G.EXPERT_PER_CALL


def acq(fifo, n):
    e = fifo.acquire(n)
    return [e] if n == 1 else e


def gemv_bands(win, yout, tab, f_gy, nbands, ngroups, per_band):
    for _ in range_(nbands):
        ye = yout.acquire(1)
        for g in range_(ngroups):
            we = win.acquire(1)
            f_gy(we, tab, ye, g, per_band, 4)
            win.release(1)
        yout.release(1)


def gemv_bands2(win, yout, tab, f_gy2, nbands, ngroups, per_band):
    for _ in range_(nbands):
        ye = acq(yout, 2)
        for g in range_(ngroups):
            we = win.acquire(1)
            f_gy2(we, tab, ye[0], ye[1], g, per_band, 4)
            win.release(1)
        yout.release(2)


def _kernels(names, elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty, wv_ty):
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def ef(sym, args, flags=OS):
        return ExternalFunction(sym, source_file=str(HERE / f"{sym}.cc"), arg_types=args, include_dirs=inc,
                                compile_flags=flags)
    mk = {"gy": lambda: ef("gx_gy", [elem, tab_ty, y_ty, i32, i32, i32], OS + NULL),
          "gms": lambda: ef("gx_gms", [elem, tab_ty, ms_ty, i32, i32, i32], OS + NULL),
          "gacc": lambda: ef("gx_gacc", [elem, tab_ty, acc_ty, i32, i32, i32, i32], OS + NULL),
          "act": lambda: ef("gx_act", [ms_ty, y_ty, wv_ty, i32]),
          "wgrab": lambda: ef("gx_wgrab", [x_ty, wv_ty]),
          "copy": lambda: ef("gx_copy", [acc_ty, y_ty, i32]),
          "prep": lambda: ef("gx_prep", [x_ty, tab_ty, i32, i32]),
          "prepf": lambda: ef("gx_prepf", [x_ty, tab_ty, i32, i32]),
          "gy2": lambda: ef("gx_gy2", [elem, tab_ty, y_ty, y_ty, i32, i32, i32], OS + NULL),
          "gms2": lambda: ef("gx_gms2", [elem, tab_ty, ms_ty, ms_ty, i32, i32, i32], OS + NULL),
          "gacc2": lambda: ef("gx_gacc2", [elem, tab_ty, acc_ty, acc_ty, i32, i32, i32, i32], OS + NULL),
          "prep2": lambda: ef("gx_prep2", [x_ty, tab_ty, i32, i32, i32]),
          "prepf2": lambda: ef("gx_prepf2", [x_ty, tab_ty, i32, i32, i32])}
    return [mk[n]() for n in names]


def _image():
    """The cores and fifos of the one image both streams run on (a stream is gx_att's or gx_moe's `sequence`).
    Every core reads a per-dispatch mode (set_mode) after its first data lands and runs the attention or the
    expert body; the two streams then share one hardware context instead of switching 48 times a token."""
    elem = np.ndarray[(G.CALL_BYTES,), np.dtype[np.uint8]]
    x_ty = np.ndarray[(ELEM // 2,), np.dtype[bfloat16]]
    y_ty = np.ndarray[(BAND_ROWS,), np.dtype[np.float32]]
    kmax = max(HID, QW, FF)
    tabn = max(G.TAB_BYTES, 4 * kmax + kmax // 4) if RW == 2 else G.TAB_BYTES   # the 2-token table at the widest K
    tab_ty = np.ndarray[(tabn,), np.dtype[np.uint8]]
    ms_ty = np.ndarray[(G.MS_FLOATS,), np.dtype[np.float32]]
    acc_ty = np.ndarray[(G.DOWN_PC * BAND_ROWS,), np.dtype[np.float32]]
    wv_ty = np.ndarray[(8,), np.dtype[np.float32]]
    u8_ln = np.ndarray[(ELN,), np.dtype[np.uint8]]
    u8_a = np.ndarray[(E_A,), np.dtype[np.uint8]]
    pb_ty = np.ndarray[(8,), np.dtype[np.int32]]              # [pos, nf, seen, -, blocks, remainder]
    bhd = np.ndarray[(HD,), np.dtype[bfloat16]]
    brow = np.ndarray[(KVW,), np.dtype[bfloat16]]
    og_ty = np.ndarray[(OGH * HD,), np.dtype[bfloat16]]
    fcs = np.ndarray[(G.ROT,), np.dtype[np.float32]]
    fhd = np.ndarray[(HD,), np.dtype[np.float32]]
    fq = np.ndarray[(2 * QW,), np.dtype[bfloat16]]          # ATTN_VEXP: q pre-split, [hi | lo]
    fml = np.ndarray[(2 * G.MLS,), np.dtype[np.float32]]
    foacc = np.ndarray[(NHL * HD,), np.dtype[np.float32]]
    fsk = np.ndarray[(NHL,), np.dtype[np.float32]]          # this core's sinks (attn.h ATTN_SINK)
    i32 = np.int32
    inc = include_dirs() + [str(GEMV), str(ATTN), str(LN), str(LINL)]

    def af(sym, src, args, flags):
        return ExternalFunction(sym, source_file=str(src), arg_types=args, include_dirs=inc, compile_flags=flags)

    KF = _kernels(("gy2", "act", "wgrab", "copy", "gms2", "gacc2", "prep2", "prepf2") if RW == 2 else
                  ("gy", "prep", "gms", "gacc", "act", "wgrab", "copy", "prepf"),
                  elem, x_ty, y_ty, tab_ty, ms_ty, acc_ty, wv_ty)
    f_add3 = af("ln_add3", LN / "ln_add3.cc", [u8_ln] * 4, LN_FLAGS)
    f_nr = af("ln_nr", LINL / "ln_nr.cc", [u8_ln] * 4, LN_FLAGS)
    f_lny = af("ln_y", LN / "ln_y.cc", [u8_ln] * 5 + [i32], LN_FLAGS)
    f_lnx = af("ln_xn", LN / "ln_xn.cc", [u8_ln] * 6, LN_FLAGS)
    f_meta = af("attn_meta", ATTN / "attn_meta.cc", [u8_a, u8_a, bhd, bhd, fcs, pb_ty, fsk, i32], ATTN_FLAGS)
    f_q = af("attn_q", ATTN / "attn_q.cc", [u8_a, bhd, fcs, fq, i32], ATTN_FLAGS)
    f_k = af("attn_k", ATTN / "attn_k.cc", [u8_a, bhd, fcs, fhd, brow, i32], ATTN_FLAGS)
    f_v = af("attn_v", ATTN / "attn_v.cc", [u8_a, brow, i32], ATTN_FLAGS)
    f_init = af("attn_init", ATTN / "attn_init.cc", [foacc, fml, fsk], ATTN_FLAGS)
    if KVSL:                                                    # the window from the core's own fifo: blocks, then the tail
        u8_w = np.ndarray[(E_W,), np.dtype[np.uint8]]
        f_step = af("attn_tailw", ATTN / "attn_tailw.cc", [u8_w, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
        f_stepb = af("attn_stepw", ATTN / "attn_stepw.cc", [u8_w, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
    else:
        f_step = af("attn_step", ATTN / "attn_step.cc", [u8_a, u8_a, fq, foacc, fml, pb_ty, i32], ATTN_FLAGS)
        f_stepb = af("attn_stepb", ATTN / "attn_stepb.cc", [u8_a] * (2 * G.RB) + [fq, foacc, fml, pb_ty, i32],
                     ATTN_FLAGS)
    f_stepn = af("attn_step_new", ATTN / "attn_step_new.cc", [brow, brow, fq, foacc, fml, i32], ATTN_FLAGS)
    f_fin = af("attn_fin_ng", ATTN / "attn_fin_ng.cc", [foacc, fml, og_ty, i32], ATTN_FLAGS)

    of_w = [ObjectFifo(elem, name=f"w{c}", depth=2) for c in range(N_CORES)]
    of_y = [ObjectFifo(y_ty, name=f"y{c}", depth=2) for c in range(N_CORES)]
    of_x = ObjectFifo(x_ty, name="x", depth=max(G.XN_ELEMS, G.OG_ELEMS, G.XM_ELEMS))
    of_lni = ObjectFifo(u8_ln, name="lni", depth=5)
    of_lno = ObjectFifo(u8_ln, name="lno", depth=1)
    of_ain = ObjectFifo(u8_a, name="ain", depth=4 if KVSL else max(4, 2 * G.RB + 2))   # a block is acquired at once
    of_kw = [ObjectFifo(u8_w, name=f"kw{g}", depth=2) for g in range(len(KW_GROUPS))] if KVSL else []
    of_aout = ObjectFifo(brow, name="aout", depth=2)
    assert N_OG == 1, "gx.py joins one og element a core"
    OGJ = -(-ACORES // 4)                                       # a memory tile takes at most 6 input channels
    assert ACORES % OGJ == 0
    OGC = ACORES // OGJ
    og_grp_ty = np.ndarray[(OGC * OGH * HD,), np.dtype[bfloat16]]
    of_og_all = [ObjectFifo(og_grp_ty, name=f"ogj{g}", depth=2) for g in range(OGJ)]
    of_og = []
    for g in range(OGJ):
        of_og += of_og_all[g].prod().join([i * OGH * HD for i in range(OGC)], obj_types=[og_ty] * OGC,
                                          names=[f"og{g * OGC + i}" for i in range(OGC)], depths=[2] * OGC)

    # mode[0] runs the attention stream's body, mode[1] the experts'; a runtime trip count of 0 or 1 is the branch.
    # The mode is read after the first data landed: the word the barrier shows can be the previous dispatch's.
    modes = [Buffer(np.ndarray[(16,), np.dtype[np.int32]], name=f"mode{c}", initial_value=np.zeros(16, np.int32),
                    use_write_rtp=True) for c in range(N_CORES + 1)]
    bars = [WorkerRuntimeBarrier() for _ in range(N_CORES + 1)]

    def set_mode(att):
        for c in range(N_CORES + 1):
            modes[c][0] = 1 if att else 0
            modes[c][1] = 0 if att else 1
        for c in range(N_CORES + 1):
            bars[c].set(1)

    def main_body(win, xin, yout, tab, ms, acc, wv, mode, bar, f_gy, f_prep, f_gms, f_gacc, f_act, f_wgrab, f_copy,
                  f_prepf):
        bar.wait_for_value(1)
        xe = acq(xin, 2)                                       # xn (attention) or xm (experts): the same prep
        for i in range(2):
            f_prep(xe[i], tab, HID, i)
        for _ in range_(mode[0]):                              # q | k | v against xn (K = HID)
            gemv_bands(win, yout, tab, f_gy, G.Q_PC + 2 * G.KV_PC, ng(HID), pb(HID))
        for _ in range_(mode[1]):                              # the routing weights ride in xm's last element
            f_wgrab(xe[G.XM_ELEMS - 1], wv)
        xin.release(2)
        for _ in range_(mode[0]):                              # o against og (K = QW)
            oe = acq(xin, G.OG_ELEMS)
            for i in range(G.OG_ELEMS):
                f_prep(oe[i], tab, QW, i)
            gemv_bands(win, yout, tab, f_gy, G.O_PC, ng(QW), pb(QW))
            xin.release(G.OG_ELEMS)
        for _ in range_(mode[1]):
            # per expert: up | gate per band -> h_e band (scaled by w_e)
            for e in range_(G.TOPK):
                for _ in range_(G.UP_PC):
                    for g in range_(xng(HID)):
                        we = win.acquire(1)
                        f_gms(we, tab, ms, g, pb(HID), G.MS_U)
                        win.release(1)
                    for g in range_(xng(HID)):
                        we = win.acquire(1)
                        f_gms(we, tab, ms, g, pb(HID), G.MS_G)
                        win.release(1)
                    ye = yout.acquire(1)
                    f_act(ms, ye, wv, e)
                    yout.release(1)
            # down: h_e's table, then expert e's bands added onto the accumulator
            for e in range_(G.TOPK):
                for i in range_(G.H_ELEMS):
                    he = xin.acquire(1)
                    f_prepf(he, tab, FF, i)
                    xin.release(1)
                for b in range_(G.DOWN_PC):
                    for g in range_(xng(FF)):
                        we = win.acquire(1)
                        f_gacc(we, tab, acc, g, pb(FF), b, e)
                        win.release(1)
            for b in range_(G.DOWN_PC):
                ye = yout.acquire(1)
                f_copy(acc, ye, b)
                yout.release(1)

    def main_body2(win, xin, yout, tab, ms, acc, wv, ms2, acc2, wv2, mode, bar, f_gy2, f_act, f_wgrab, f_copy,
                   f_gms2, f_gacc2, f_prep2, f_prepf2):
        # Two token rows: one 2-token table, each w element read once and decoded once for both rows.
        bar.wait_for_value(1)
        for t in range(2):
            xe = acq(xin, 2)                                   # xn (attention) or xm (experts) of row t
            for i in range(2):
                f_prep2(xe[i], tab, HID, t, i)
            for _ in range_(mode[1]):
                f_wgrab(xe[G.XM_ELEMS - 1], wv if t == 0 else wv2)
            xin.release(2)
        for _ in range_(mode[0]):                              # q | k | v of both rows against xn
            gemv_bands2(win, yout, tab, f_gy2, G.Q_PC + 2 * G.KV_PC, ng(HID), pb(HID))
        for _ in range_(mode[0]):                              # o of both rows against og
            for t in range(2):
                oe = acq(xin, G.OG_ELEMS)
                for i in range(G.OG_ELEMS):
                    f_prep2(oe[i], tab, QW, t, i)
                xin.release(G.OG_ELEMS)
            gemv_bands2(win, yout, tab, f_gy2, G.O_PC, ng(QW), pb(QW))
        for _ in range_(mode[1]):
            for e in range_(G.TOPK):
                for _ in range_(G.UP_PC):
                    for g in range_(xng(HID)):
                        we = win.acquire(1)
                        f_gms2(we, tab, ms, ms2, g, pb(HID), G.MS_U)
                        win.release(1)
                    for g in range_(xng(HID)):
                        we = win.acquire(1)
                        f_gms2(we, tab, ms, ms2, g, pb(HID), G.MS_G)
                        win.release(1)
                    ye = yout.acquire(1)
                    f_act(ms, ye, wv, e)
                    yout.release(1)
                    ye = yout.acquire(1)
                    f_act(ms2, ye, wv2, e)
                    yout.release(1)
            for e in range_(G.TOPK):
                for t in range(2):
                    for i in range_(G.H_ELEMS):
                        he = xin.acquire(1)
                        f_prepf2(he, tab, FF, t, i)
                        xin.release(1)
                for b in range_(G.DOWN_PC):
                    for g in range_(xng(FF)):
                        we = win.acquire(1)
                        f_gacc2(we, tab, acc, acc2, g, pb(FF), b, e)
                        win.release(1)
            for b in range_(G.DOWN_PC):
                ye = yout.acquire(1)
                f_copy(acc, ye, b)
                yout.release(1)
                ye = yout.acquire(1)
                f_copy(acc2, ye, b)
                yout.release(1)

    def ln_body(ain, aout, f_nr, f_lny, f_lnx, f_add3, mode, bar):
        bar.wait_for_value(1)
        # the first 3-element stage: [x0 x1 lnw] -> [xn] (attention, the layer-entry norm) or [res out2 bsum] -> [xres half]
        e = ain.acquire(3)
        o = aout.acquire(1)
        for _ in range_(mode[0]):
            f_nr(e[0], e[1], e[2], o)
        for _ in range_(mode[1]):
            f_add3(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)
        for _ in range_(mode[0]):
            # [x0 x1 postln out0 out1] -> [res0] [res1] [xm]: res = x + out, xm = post_attention_norm(res)
            e = ain.acquire(5)
            for i in range(2):
                o = aout.acquire(1)
                f_lny(e[0], e[1], e[3], e[4], o, i)
                aout.release(1)
            o = aout.acquire(1)
            f_lnx(e[0], e[1], e[3], e[4], e[2], o)
            aout.release(1)
            ain.release(5)
        for _ in range_(mode[1]):                              # the experts' second half
            e = ain.acquire(3)
            o = aout.acquire(1)
            f_add3(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)

    def attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, h0,
             f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, kw=None):
        e = ain.acquire(2)                                     # [qn | kn | sinks], the position record
        f_meta(e[0], e[1], qn, kn, cs, pbuf, sk, h0)
        ain.release(2)
        for h in range_(G.Q_AIN_ELEMS):
            e = ain.acquire(1)
            f_q(e, qn, cs, qs, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_k(e, kn, cs, tmp, kout, h)
            ain.release(1)
        for h in range_(G.K_AIN_ELEMS):
            e = ain.acquire(1)
            f_v(e, vout, h)
            ain.release(1)
        if aout is not None:                                   # core 0 owns the cache row
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = kout[j]
            aout.release(1)
            o = aout.acquire(1)
            for j in range_(KVW):
                o[j] = vout[j]
            aout.release(1)
        f_init(oacc, ml, sk)
        if kw is not None:                                     # KVSLICE: one element a block, then the padded tail block
            for _ in range_(pbuf[4]):
                e = kw.acquire(1)
                f_stepb(e, qs, oacc, ml, pbuf, h0)
                kw.release(1)
            for _ in range_(pbuf[6]):
                e = kw.acquire(1)
                f_step(e, qs, oacc, ml, pbuf, h0)
                kw.release(1)
        else:
            for _ in range_(pbuf[4]):                          # whole blocks of RB cached rows
                e = ain.acquire(2 * G.RB)
                f_stepb(*[e[i] for i in range(2 * G.RB)], qs, oacc, ml, pbuf, h0)
                ain.release(2 * G.RB)
            for _ in range_(pbuf[5]):                          # what did not fill a block
                e = ain.acquire(2)
                f_step(e[0], e[1], qs, oacc, ml, pbuf, h0)
                ain.release(2)
        f_stepn(kout, vout, qs, oacc, ml, h0)
        for hp in range_(N_OG):
            o = ogout.acquire(1)
            f_fin(oacc, ml, o, hp)
            ogout.release(1)

    def walks(call):
        # one walk a query row (row 1's window holds row 0's new K and V); a runtime loop keeps one copy of the body
        if RW == 1:
            call()
        else:
            for _ in range_(RW):
                call()

    if KVSL:
        def attn0(ain, kw, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            walks(lambda: attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, 0,
                               f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, kw))

        def make_attn(c):
            def body(ain, kw, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
                     f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                walks(lambda: attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, c * NHL,
                                   f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb, kw))
            return body
    else:
        def attn0(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
                  f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
            attn(ain, aout, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, 0,
                 f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)

        def make_attn(c):
            def body(ain, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk,
                     f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb):
                attn(ain, None, ogout, qn, kn, cs, qs, tmp, kout, vout, oacc, ml, pbuf, sk, c * NHL,
                     f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb)
            return body

    def ln_body2(ain, aout, f_nr, f_lny, f_lnx, f_add3, mode, bar):
        # the norm core per query row: the entry norm (attention) or the add's two halves (experts) a row, then the
        # post-attention norm a row
        bar.wait_for_value(1)
        for _ in range_(RW):
            e = ain.acquire(3)
            o = aout.acquire(1)
            for _ in range_(mode[0]):
                f_nr(e[0], e[1], e[2], o)
            for _ in range_(mode[1]):
                f_add3(e[0], e[1], e[2], o)
            aout.release(1)
            ain.release(3)
            for _ in range_(mode[1]):
                e = ain.acquire(3)
                o = aout.acquire(1)
                f_add3(e[0], e[1], e[2], o)
                aout.release(1)
                ain.release(3)
        for _ in range_(mode[0]):
            for _ in range_(RW):
                e = ain.acquire(5)
                for i in range(2):
                    o = aout.acquire(1)
                    f_lny(e[0], e[1], e[3], e[4], o, i)
                    aout.release(1)
                o = aout.acquire(1)
                f_lnx(e[0], e[1], e[3], e[4], e[2], o)
                aout.release(1)
                ain.release(5)

    workers = [Worker(ln_body2 if RW == 2 else ln_body, fn_args=[of_lni.cons(), of_lno.prod(), f_nr, f_lny, f_lnx, f_add3, modes[N_CORES],
                                        bars[N_CORES]], tile=Tile(0, 3), stack_size=GR.STACK)]
    for c in range(N_CORES):
        if RW == 2:
            workers.append(Worker(main_body2, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                        Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                        Buffer(acc_ty, name=f"acc{c}"), Buffer(wv_ty, name=f"wv{c}"),
                                                        Buffer(ms_ty, name=f"msb{c}"), Buffer(acc_ty, name=f"accb{c}"),
                                                        Buffer(wv_ty, name=f"wvb{c}"), modes[c], bars[c]] + KF,
                                  tile=Tile(c, 2), stack_size=GR.STACK))
            continue
        workers.append(Worker(main_body, fn_args=[of_w[c].cons(), of_x.cons(), of_y[c].prod(),
                                                   Buffer(tab_ty, name=f"tab{c}"), Buffer(ms_ty, name=f"ms{c}"),
                                                   Buffer(acc_ty, name=f"acc{c}"), Buffer(wv_ty, name=f"wv{c}"),
                                                   modes[c], bars[c]] + KF,
                              tile=Tile(c, 2), stack_size=GR.STACK))

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(bhd, name=f"qn{s}"), Buffer(bhd, name=f"kn{s}"), Buffer(fcs, name=f"cs{s}"),
                Buffer(fq, name=f"qs{s}"), Buffer(fhd, name=f"tmp{s}"), Buffer(brow, name=f"kout{s}"),
                Buffer(brow, name=f"vout{s}"), Buffer(foacc, name=f"oacc{s}"), Buffer(fml, name=f"ml{s}"),
                Buffer(pb_ty, name=f"pb{s}"), Buffer(fsk, name=f"sk{s}")]
    afns = [f_meta, f_q, f_k, f_v, f_init, f_step, f_stepn, f_fin, f_stepb]
    ATILES = ([(2 + c, 3) for c in range(6)] + [(2, 4), (3, 4)] + [(c, 4) for c in (0, 1, 4, 5, 6, 7)]
              + [(c, 5) for c in range(8)])                    # row 3 has columns 2..7; rows 4 and 5 are idle
    assert ACORES <= len(ATILES)

    def atile(c):
        return Tile(*ATILES[c])
    def kwc(c):
        return [of_kw[KW_GROUPS.index(KVB[c])].cons()] if KVSL else []
    workers.append(Worker(attn0, fn_args=[of_ain.cons()] + kwc(0) + [of_aout.prod(), of_og[0].prod()] + abufs(0) + afns,
                          tile=atile(0), stack_size=GR.STACK))
    for c in range(1, ACORES):
        workers.append(Worker(make_attn(c), fn_args=[of_ain.cons()] + kwc(c) + [of_og[c].prod()] + abufs(c) + afns,
                              tile=atile(c), stack_size=GR.STACK))
    return {"workers": workers, "set_mode": set_mode, "OGJ": OGJ, "OGC": OGC, "of_lni": of_lni, "of_lno": of_lno,
            "of_w": of_w, "of_x": of_x, "of_y": of_y, "of_ain": of_ain, "of_aout": of_aout, "of_og_all": of_og_all,
            "of_kw": of_kw}


def _endpoints(im):
    """Every shim endpoint of the image, in the one order both streams declare them: lni, lno, w, x, y, then the
    attention's ain, aout, og joins and window streams (the experts' stream leaves those idle)."""
    return [im["of_lni"].prod(tile=Tile(0, 0)), im["of_lno"].cons(tile=Tile(0, 0)),
            [im["of_w"][c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
            im["of_x"].prod(tile=Tile(1, 0)),
            [im["of_y"][c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
            im["of_ain"].prod(tile=Tile(2, 0)), im["of_aout"].cons(tile=Tile(1, 0)),
            [im["of_og_all"][g].cons(tile=Tile(2 + g, 0)) for g in range(im["OGJ"])]] + \
        ([[im["of_kw"][g].prod(tile=Tile(KW_COLS[g], 0)) for g in range(len(KW_GROUPS))]] if KVSL else [])


def ydrain(py, y_cons, a_act, base, bands, row_stride):
    """A core's `bands` y elements at `base`; with two rows the core emits [band][row], row r at + r * row_stride."""
    if RW == 1:
        return py.drain(y_cons, a_act, bt(L.AD_BYTES, base, bands * YB))
    return py.drain(y_cons, a_act, TensorAccessPattern((1, L.AD_BYTES), base, [1, bands, RW, YB],
                                                       [0, YB, row_stride, 1]))


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gx_att(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, srchash: CompileTime[int] = 0):
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(RW * HID,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(L.CD_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(L.KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(L.PTAB_BYTES,), np.dtype[np.uint8]]
    im = _image()
    workers, set_mode, OGJ, OGC = im["workers"], im["set_mode"], im["OGJ"], im["OGC"]

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c,
                 og_cs, kw_ps=()):
        set_mode(True)
        # 1. layer-entry norm: xn -> act
        tg_ln = TaskGroup()
        for r in range(RW):
            lni.fill(c_xres, tap=bt(RW * HID, r * HID, HID), wait=True, group=tg_ln)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_LNW, ELN), wait=True, group=tg_ln)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XN + r * G.XN_ELEMS * ELEM, ELN), wait=True, group=tg_ln)
        # 2. q | k | v: weights now, xn after the norm
        pw, py = Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_Q + c * G.Q_PC * BB_H, G.Q_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_K + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_V + c * G.KV_PC * BB_H, G.KV_PC * BB_H))
            ydrain(py, y_conss[c], a_act, L.AD_Q + c * G.Q_PC * YB, G.Q_PC, QW * 4)
            ydrain(py, y_conss[c], a_act, L.AD_KVN + c * G.KV_PC * YB, G.KV_PC, 2 * KVW * 4)
            ydrain(py, y_conss[c], a_act, L.AD_KVN + KVW * 4 + c * G.KV_PC * YB, G.KV_PC, 2 * KVW * 4)
        tg_ln.finish()                                            # xn is in DDR
        tg_x = TaskGroup()
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_XN, RW * G.XN_ELEMS * ELEM), wait=True, group=tg_x)
        # 3. attention: meta + record now, q / k / v after the GEMVs, the window, the new row out
        pa_out, pa_in = Pipeline(3), Pipeline(3)
        for r in range(RW):                                       # query row r: compiled for placeholder row r + 1
            if r:
                pa_out.finish()                                   # row r - 1's new K, V row is in DDR before row r's window
            pa_out.drain(aout_c, a_kv, bt(L.KV_BYTES, (r + 1) * L.KV_ROW, L.KV_ROW))   # [k' | v'] -> row pos + r (attnpos)
            for g in range(OGJ):
                pa_out.drain(og_cs[g], a_act, bt(L.AD_BYTES, L.AD_OG + r * G.OG_ELEMS * ELEM + g * OGC * NHL * HD * 2,
                                                 OGC * NHL * HD * 2))
            pa_in.fill(ain_p, a_consts, bt(L.CD_BYTES, L.CD_META, E_A))          # [qn | kn | sinks]
            pa_in.fill(ain_p, a_ptab, bt(L.PTAB_BYTES, (r + 1) * L.PTAB_ROW, L.PTAB_ROW))   # the position record (attnpos)
            for g, kvb in enumerate(KW_GROUPS if KVSL else []):                  # one block compiled, sized per token (attnpos)
                pa_in.fill(kw_ps[g], a_kv, TensorAccessPattern((1, L.KV_BYTES), kvb * HD * 2,
                                                               [1, G.RB, 2, KVS * HD * 2],
                                                               [0, L.KV_ROW, L.KV_ROW // 2, 1]))
            if r == 0:
                py.finish(*y_conss)                               # q, k, v of every row are in DDR
            pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_Q + r * QW * 4, QW * 4))
            pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + r * 2 * KVW * 4, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(L.AD_BYTES, L.AD_KVN + r * 2 * KVW * 4 + KVW * 4, KVW * 4))
        if not KVSL:
            pa_in.fill(ain_p, a_kv, bt(L.KV_BYTES, 0, L.KV_ROW))                 # the window: rows [0, nf) (attnpos)
        # 4. o projection against og
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_O + c * G.O_PC * BB_Q, G.O_PC * BB_Q))
            ydrain(py, y_conss[c], a_act, L.AD_OUT + c * G.O_PC * YB, G.O_PC, HID * 4)
        pa_out.finish()                                           # og (and the new cache row) are in DDR
        x_prod.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OG, RW * G.OG_ELEMS * ELEM), wait=True, group=tg_x)
        # 5. res = x + out; xm = post_attention_norm(res)
        tg_ln2 = TaskGroup()
        if RW == 1:
            lni.fill(c_xres, tap=bt(HID, 0, HID), wait=True, group=tg_ln2)
            lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES, HID * 4), wait=True, group=tg_ln2)
            lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM, ELN), wait=True, group=tg_ln2)
            py.finish()                                           # out is in DDR
            lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT, HID * 4), wait=True, group=tg_ln2)
        else:                                                     # each row's [x, postln, out] arrives together
            for r in range(RW):
                lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_RES + r * HID * 4, HID * 4), wait=True, group=tg_ln2)
                lno.drain(a_act, tap=bt(L.AD_BYTES, L.AD_XM + r * G.XM_ELEMS * ELEM, ELN), wait=True, group=tg_ln2)
            py.finish()                                           # out of every row is in DDR
            for r in range(RW):
                lni.fill(c_xres, tap=bt(RW * HID, r * HID, HID), wait=True, group=tg_ln2)
                lni.fill(a_consts, tap=bt(L.CD_BYTES, L.CD_POSTLN, ELN), wait=True, group=tg_ln2)
                lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT + r * HID * 4, HID * 4), wait=True, group=tg_ln2)
        tg_ln2.finish()
        pw.finish()
        pa_in.finish()
        tg_x.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty] + _endpoints(im))
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def gx_moe(pool: In, xres: InOut, act: InOut, *, srchash: CompileTime[int] = 0):
    pool_ty = np.ndarray[(L.POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(RW * HID,), np.dtype[np.float32]]
    act_ty = np.ndarray[(L.AD_BYTES,), np.dtype[np.uint8]]
    im = _image()
    workers, set_mode = im["workers"], im["set_mode"]

    def sequence(a_pool, c_xres, a_act, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs, kw_ps=()):
        set_mode(False)
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # 1. xm (+ the routing weights) to every main core
        px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_XM, RW * G.XM_ELEMS * ELEM))
        # 2. up | gate: each core's h bands of all four experts in ONE drain (expert e's at AD_H + e * FF * 4),
        #    then expert slot e's fills -- slot e is compiled as expert e, re-pointed by moeroute2
        for c in range(N_CORES):
            if RW == 1:
                py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + c * G.UP_PC * YB,
                                                                [1, 1, G.TOPK, G.UP_PC * YB], [0, 0, FF * 4, 1]))
            else:                                                 # per expert, per band, per row
                py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_H + c * G.UP_PC * YB,
                                                                [G.TOPK, G.UP_PC, RW, YB],
                                                                [FF * 4, YB, G.TOPK * FF * 4, 1]))
        for e in range(G.TOPK):
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, e * L.EXPERT_UG + c * G.UP_PC * 2 * XBB_H,
                                               G.UP_PC * 2 * XBB_H))
        py.finish()                                               # every h_e is in DDR
        # 3. down: h_e, then expert slot e's down bands; the cores hold the sum
        for c in range(N_CORES):
            if RW == 1:
                py.drain(y_conss[c], a_act, bt(L.AD_BYTES, L.AD_OUT2 + c * G.DOWN_PC * YB, G.DOWN_PC * YB))
            else:
                py.drain(y_conss[c], a_act, TensorAccessPattern((1, L.AD_BYTES), L.AD_OUT2 + c * G.DOWN_PC * YB,
                                                                [1, G.DOWN_PC, RW, YB], [0, YB, HID * 4, 1]))
        for e in range(G.TOPK):
            for r in range(RW):
                px.fill(x_prod, a_act, bt(L.AD_BYTES, L.AD_H + r * G.TOPK * FF * 4 + e * FF * 4, G.H_ELEMS * ELEM))
            for c in range(N_CORES):
                pw.fill(w_prods[c], a_pool, bt(L.POOL_BYTES, L.POOL_DOWN + e * L.UP_BYTES + c * G.DOWN_PC * XBB_F,
                                               G.DOWN_PC * XBB_F))
        py.finish()                                               # out2 is in DDR
        # 4. xres = res + out2 + bsum
        for r in range(RW):                                       # a group a row: the shim has 16 buffer descriptors
            tg = TaskGroup()
            for i in range(2):
                lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_RES + r * HID * 4 + i * ELN, ELN), wait=True, group=tg)
                lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_OUT2 + r * HID * 4 + i * ELN, ELN), wait=True, group=tg)
                lni.fill(a_act, tap=bt(L.AD_BYTES, L.AD_BSUM + r * HID * 4 + i * ELN, ELN), wait=True, group=tg)
            lno.drain(c_xres, tap=bt(RW * HID, r * HID, HID), wait=True, group=tg)
            tg.finish()
        pw.finish()
        px.finish()

    rt = Runtime(sequence, [pool_ty, xres_ty, act_ty] + _endpoints(im))
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = gx_att if PART == "att" else gx_moe
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.py"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + [(HERE.parent.parent / "recipes" / "gptoss.py").read_bytes(), (LN / "ln.h").read_bytes(),
                   (LN / "ln_y.cc").read_bytes(), (LN / "ln_xn.cc").read_bytes(), (LN / "ln_add3.cc").read_bytes(),
                   (LINL / "ln_nr.cc").read_bytes(), (GEMV / "gemv_q8.h").read_bytes(), (GEMV / "gemv_q4.h").read_bytes(),
                   (GEMV / "gemv_mx4.h").read_bytes(), (HERE / "gx_rows.h").read_bytes(), str(G.EXPERT_CH).encode(),
                   (GEMV / "gemv_tab.h").read_bytes(), (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   SPEC.spec_hash().encode(), repr(PROBES).encode(), PART.encode(), str(RW).encode()])
SPECIALIZE = {"srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
