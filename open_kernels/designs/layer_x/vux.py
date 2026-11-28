r"""vux: the MoE verify pass's two images (vlx.py, vax.py) on ONE xclbin, so a verify pass runs in one hardware context
instead of switching between two 20 times (OPEN-VERIFY-ONE-CONTEXT). ux.py's construction for the decode pair: the
two layer types' main-core programs are the same sequence with two numbers changed -- the projection's band count
and the DeltaNet heads (0 in a full layer) -- read from RTP words, so the merged core is vlx's program with those two
numbers at run time; the helpers are the union (post at (1, 3), the conv glue moved to (2, 4), the attention cores
at (2..2+ACORES-1, 3), the norm + router at (0, 3)), each fed by one layer type's dispatches only.

    part 0 = vlx's part 0   part 1 = vlx's MoE   part 2 = vax's part 0   part 3 = vax's MoE

Every core and helper runs the calls it ran in vlx / vax in the same order on the same bytes: every row is still
bit-identical to a decode step. Args (both types): pool, xres f32[2][HID], consts, state | kv, act, ptab (the
linear stream never touches ptab).
Build: VUX_PART=<0..3> OPEN_KERNELS_SPEC=<spec.json> python build_design.py designs/layer_x/vux.py designs/layer_x/build_vux<part>
"""

from __future__ import annotations

import hashlib
import os
import sys
from pathlib import Path

import numpy as np

import aie.iron as iron
from aie.iron import Buffer, CompileTime, In, InOut, Program, Runtime, TaskGroup, Worker
from aie.iron.controlflow import range_
from aie.iron.device import Tile
from aie.helpers.taplib import TensorAccessPattern

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
GLUE = HERE.parent / "dn_glue"
POST = HERE.parent / "dn_post"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_HP, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN, A_Z,  # noqa: E402
                    AA_BYTES, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_QG, AA_RES, AA_ROUT, AA_XM, AA_XN,
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT, GLUE_SIDE_BYTES,
                    CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, KV_BYTES, KV_ROW,
                    POOL_BYTES, POOL_QKV, POOL_Z, PTAB_BYTES, PTAB_ROW, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402
import vrows as V  # noqa: E402
from vrows import NR, YB  # noqa: E402

DL, DA = R.linear, R.attn
if DL is None or DA is None or X.KIND != "moe" or DL.CONV_F32 or DL.XN_F32 or DA.XN_F32 or DA.KV_PAIR:
    sys.exit("vux.py: the merged verify image needs the MoE linear and full layer types (not dense, fp32-conv, paired KV)")
if not ({"attn", "linear", "linear_out"} <= set(X.Q8)):
    sys.exit("vux.py: built for q8 projections in every role (one GEMV entry on the main core)")
HID = X.HID
HD = DA.HD
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC, VW, OUT_K = DL.QKV_PC, DL.Z_PC, DL.OUT_PC, DL.VW, DL.OUT_K
Q_PC, KV_PC, O_PC = DA.Q_PC, DA.KV_PC, DA.O_PC
QW, KVW, O_K = DA.QW, DA.KVW, DA.O_K
if (OUT_PC, OUT_K, DL.OG_ELEMS) != (O_PC, O_K, DA.OG_ELEMS):
    sys.exit("vux.py: the two og projections differ; one image cannot share that stage")
if ((X.role_groups("linear", HID), X.role_per_band("linear", HID))
        != (X.role_groups("attn", HID), X.role_per_band("attn", HID))
        or (X.role_groups("linear_out", OUT_K), X.role_per_band("linear_out", OUT_K))
        != (X.role_groups("attn", O_K), X.role_per_band("attn", O_K))):
    sys.exit("vux.py: the q8 roles' band geometries differ between the layer types")
TILE, NT, G, NG, KEY_TILES = XL.TILE, XL.NT, XL.G, XL.NG, XL.KEY_TILES
ACORES, NHL = XL.ACORES, XL.NHL
NBANDS_LIN = QKV_PC + Z_PC                       # RTP word 0, linear stream
NBANDS_FULL = 2 * Q_PC + 2 * KV_PC               # RTP word 0, full stream
DN_HEADS_PC = X.DN_HEADS_PC                      # RTP word 1, linear stream (full: 0)
# one set of buffer-argument totals for all four streams (each stream keeps its own row strides)
ACT_TOT = NR * max(A_BYTES, AA_BYTES)
CONSTS_T = max(C_BYTES, CA_BYTES)
STATE_TOT = max((NR + 1) * STATE_BYTES, KV_BYTES)
XRES_T = NR * HID
PART = int(os.environ.get("VUX_PART", 0))
if PART not in (0, 1, 2, 3):
    sys.exit(f"vux.py: VUX_PART={PART} (0 = vlx0, 1 = vlx1, 2 = vax0, 3 = vax1)")
GLUE_TILE = Tile(2, 4)
bt = X.bt
X.ROLLED = True     # band loops keep one call site, as in vlx.py


def rows3(tt: int, slot: int) -> TensorAccessPattern:
    """XL.rows3 in state slot `slot`."""
    return TensorAccessPattern((1, STATE_TOT), slot * STATE_BYTES + tt * TILE * 2,
                               [1, 1, XL.CONV_ROWS, TILE * 2], [0, 0, XL.NCH * 2, 1])


def twice(body):
    def run(*a):
        for _ in range_(NR):
            body(*a)
    return run


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def vux(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
        srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    g = XL.glue_post_types()
    a = XL.attn_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(XRES_T,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CONSTS_T,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_TOT,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_TOT,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]
    rtp_ty = np.ndarray[(2,), np.dtype[np.int32]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(ATTN), str(X.LN), str(X.LINL), str(X.RT),
                            str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    GK = XL.glue_post_kernels(inc, g)
    afns = XL.attn_kernels(inc, a)

    of_w, of_y, of_x = XL.main_fifos(t)
    w_src = XL.weight_ring(of_w, t)   # one context: no switch for the ring to slow down (OPEN-DECODE-STREAM-LX)
    of_lni, of_lno = XL.ln_fifos(tl)
    of_side, of_gact, of_gout, of_pin, of_pout = XL.glue_post_fifos(g)
    of_ain, of_aout, of_og = XL.attn_fifos(a)
    rtp = [Buffer(rtp_ty, name=f"rtp{c}", initial_value=np.zeros(2, dtype=np.int32), use_write_rtp=True)
           for c in range(N_CORES)]

    def main_body(win, xin, yout, my_rtp, tab1, *args):
        B, K = X.unpack_args(args)
        tab0 = B["tab"]
        # row 0's xn element first: it orders the RTP reads behind the stream's write32s (ux.py)
        xe = xin.acquire(1)
        nbands = my_rtp[0]                                  # linear | full projection bands
        nheads = my_rtp[1]                                  # DeltaNet heads | 0
        K["prep2048"](xe, tab0)
        xin.release(1)
        xe = xin.acquire(1)
        K["prep2048"](xe, tab1)
        xin.release(1)
        V.gemv2_bands(win, yout, tab0, tab1, K["gy8"], nbands, X.role_groups("linear", HID),
                      X.role_per_band("linear", HID), 4)
        for _ in range_(NR):
            X.dn_body(win, yout, B, K, nheads)
        V.prep2(xin, tab0, tab1, [K["prep4096a"], K["prep4096b"]])
        V.role_gemv2(win, yout, tab0, tab1, K, "linear_out", OUT_PC, OUT_K)
        for _ in range_(NR):
            X.moe_body(win, xin, yout, B, K, nx=X.NX_DECODE)

    def glue2(*args):
        for _ in range_(NR):
            XL.glue_body(*args)

    def post2(*args):
        for _ in range_(NR):
            XL.post_body(*args)

    attn_body, make_attn_body = XL._attn_bodies()

    def abufs(c):
        s = "" if c == 0 else str(c)
        return [Buffer(a["b256"], name=f"qn{s}"), Buffer(a["b256"], name=f"kn{s}"), Buffer(a["f64"], name=f"cs{s}"),
                Buffer(a["fq"], name=f"qs{s}"), Buffer(a["f256"], name=f"tmp{s}"), Buffer(a["kv"], name=f"kout{s}"),
                Buffer(a["kv"], name=f"vout{s}"), Buffer(a["foacc"], name=f"oacc{s}"),
                Buffer(a["fml"], name=f"ml{s}"), Buffer(a["pb"], name=f"pb{s}")]

    attn_workers = [Worker(twice(attn_body), fn_args=[of_ain.cons(), of_aout.prod()] + abufs(0) + afns,
                           tile=Tile(2, 3), stack_size=XL.STACK)]
    for c in range(1, ACORES):
        attn_workers.append(Worker(twice(make_attn_body(c)), fn_args=[of_ain.cons(), of_og[c - 1].prod()] + abufs(c) + afns,
                                   tile=Tile(2 + c, 3), stack_size=XL.STACK))
    f32, facc = g["f32"], g["facc"]
    workers = ([Worker(V.ln_router2, fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"),
                                              Buffer(tl["racc"], name="racc"), Buffer(tl["xb"], name="rxs1"),
                                              Buffer(tl["racc"], name="racc1"), L["ln_nr"], L["ln"], L["rcopy"],
                                              L["racc"], L["rfin"]], tile=Tile(0, 3), stack_size=XL.STACK)]
               + XL.main_workers(main_body, of_w, of_x, of_y, t, K,
                                 extra=lambda c: [rtp[c], Buffer(t["tab"], name=f"tabr{c}")])
               + [Worker(post2, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(g["nw"], name="nwb"),
                                         GK["post"], GK["post_copy"]], tile=Tile(1, 3), stack_size=XL.STACK),
                  Worker(glue2, fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                         Buffer(facc, name="acc_a"), Buffer(facc, name="acc_b"),
                                         Buffer(f32, name="decay"), Buffer(f32, name="beta"),
                                         Buffer(g["fqk"], name="qk"), Buffer(g["fvt"], name="vt"),
                                         Buffer(g["fxn"], name="xnb"),
                                         GK["ab"], GK["small"], GK["conv"], GK["emit"], GK["copy"]],
                         tile=GLUE_TILE, stack_size=XL.STACK)]
               + attn_workers)

    BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)

    def lin_front(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
              side_p, gact_p, gout_c, pin_p, pout_c):
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_LIN
            rtp[c][1] = DN_HEADS_PC
        # 1. both rows' entry norms
        tg = TaskGroup()
        for r in range(NR):
            lni.fill(c_xres, tap=bt(XRES_T, r * HID, HID), wait=True, group=tg)
            lni.fill(a_consts, tap=bt(CONSTS_T, C_LNW, ELEM), wait=True, group=tg)
            lno.drain(a_act, tap=bt(ACT_TOT, r * A_BYTES + A_XN, ELEM), wait=True, group=tg)
        # 2. qkv | z for both rows: one weight pass
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_TOT, A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_TOT, A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg.finish()                                      # xn of both rows is in DDR
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_TOT, r * A_BYTES + A_XN, ELEM))
        py.finish_oldest(*y_conss)                       # qkv of both rows is in DDR; row 0's glue runs during z
        # 3-5. glue, DeltaNet and post per row (conv state and S: slot r -> r + 1). Row 1's glue streams while row 0's
        # DeltaNet runs, and row 0's post while row 1's DeltaNet runs: their transfers are issued interleaved, since a
        # throttled channel blocks the host and would otherwise hold back the other core's transfers.
        rec, ohb = DL.RECORD_BYTES, DL.O_HEAD_BYTES

        def glue(r):
            ab = r * A_BYTES
            tg_s = TaskGroup()
            side_p.fill(a_act, tap=bt(ACT_TOT, ab + A_XN, ELEM), wait=True, group=tg_s)
            side_p.fill(a_consts, tap=bt(CONSTS_T, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
            pipe = Pipeline(3)
            for tt in range(NT):
                pipe.drain(gout_c, a_state, rows3(tt, r + 1))
                if tt >= KEY_TILES:
                    pipe.drain(gout_c, a_act, bt(ACT_TOT, ab + A_VEC + (tt - KEY_TILES) * DL.HEADS_PER_TILE * rec,
                                                 DL.HEADS_PER_TILE * rec))
                pipe.fill(gact_p, a_act, bt(ACT_TOT, ab + A_QKV + tt * TILE * 4, TILE * 4))
                pipe.fill(gact_p, a_state, rows3(tt, r))
                yield
            pipe.finish()                                # row r's records and conv state are in DDR
            tg_s.finish()

        def deltanet(r):
            ab = r * A_BYTES
            s_in, s_out = r * STATE_BYTES + STATE_S_OFF, (r + 1) * STATE_BYTES + STATE_S_OFF
            for h in range(X.DN_HEADS_PC):
                for c in range(N_CORES):
                    hd = c * X.DN_HEADS_PC + h
                    pw.fill(w_prods[c], a_act, bt(ACT_TOT, ab + A_VEC + hd * rec, X.CALL_BYTES))
                    pw.fill(w_prods[c], a_state, bt(STATE_TOT, s_in + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    py.drain(y_conss[c], a_state, bt(STATE_TOT, s_out + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    py.drain(y_conss[c], a_act, bt(ACT_TOT, ab + A_O + hd * ohb, ohb))
                    pw.fill(w_prods[c], a_state, bt(STATE_TOT, s_in + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    yield
            py.finish()                                  # row r's S' and o are in DDR

        def post(r):
            ab = r * A_BYTES
            pipe = Pipeline(3)
            pipe.fill(pin_p, a_consts, bt(CONSTS_T, C_NW, ELEM))
            for gg in range(NG):
                pipe.drain(pout_c, a_act, bt(ACT_TOT, ab + A_OG + gg * G * 2, G * 2))
                pipe.fill(pin_p, a_act, bt(ACT_TOT, ab + A_O + gg * G * 4, G * 4))
                pipe.fill(pin_p, a_act, bt(ACT_TOT, ab + A_Z + gg * G * 4, G * 4))
                yield
            pipe.finish()

        V.interleave(glue(0))
        V.interleave(deltanet(0), glue(1))               # glue(1) reads conv slot 1, which glue(0) wrote
        V.interleave(deltanet(1), post(0))
        # 6a. the out projection's weights and both rows' drains
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(CONSTS_T, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_TOT, A_BYTES, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        V.interleave(post(1))
        # 6b. out against both rows' og
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_TOT, r * A_BYTES + A_OG, VW * 2))
        py.finish()                                      # out of both rows is in DDR
        # the residual + post-attention norm of both rows, then one router pass for both
        V.ln2_router2(lni, lno, [((c_xres, bt(XRES_T, r * HID, HID)), (a_act, bt(ACT_TOT, r * A_BYTES + A_OUT, HID * 4)),
                                  (a_act, bt(ACT_TOT, r * A_BYTES + A_RES, HID * 4)), (a_act, bt(ACT_TOT, r * A_BYTES + A_XM, ELEM)),
                                  (a_act, bt(ACT_TOT, r * A_BYTES + A_ROUT, ELEM))) for r in range(NR)],
                      (a_consts, bt(CONSTS_T, C_POSTLN, ELEM)),
                      lambda tg: XL.router_fill(lni, None, a_consts, CONSTS_T, C_RW, tg))
        pw.finish()
        px.finish()

    def lin_moe(a_pool, c_xres, a_consts, a_act, w_prods, x_prod, y_conss):
        # decode's MoE stream once a row: X.moe_sequence on that row's act half and xres row
        for r in range(NR):
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod,
                           y_conss, ACT_TOT, CONSTS_T, r * A_BYTES + A_XM, r * A_BYTES + A_ROUT, r * A_BYTES + A_RES,
                           r * A_BYTES + A_HP, C_SGW, nx=X.NX_DECODE, xres_t=XRES_T, xres_off=r * HID,
                           slot0=r * X.NE, early_down=XL.X_WORDER, ug_bd=XL.X_SEQ, early_ug=XL.X_WORDER and XL.X_WUG)

    def y2(off, n):
        return V.y2_tap(ACT_TOT, AA_BYTES, off, n)

    def full_front(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs):
        for c in range(N_CORES):
            rtp[c][0] = NBANDS_FULL
            rtp[c][1] = 0
        # both rows' entry norms
        tg = TaskGroup()
        for r in range(NR):
            lni.fill(c_xres, tap=bt(XRES_T, r * HID, HID), wait=True, group=tg)
            lni.fill(a_consts, tap=bt(CONSTS_T, CA_LNW, ELEM), wait=True, group=tg)
            lno.drain(a_act, tap=bt(ACT_TOT, r * AA_BYTES + AA_XN, ELEM), wait=True, group=tg)
        # q | gate | k | v for both rows: one weight pass
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        # only three transfers a channel before xn: a fourth would wait (Pipeline) on q's, which waits on xn
        for c in range(N_CORES):
            for off, n in XL.w_regions(c)[:3]:
                pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, off, n))
            for off, n in XL.y_regions(c)[:3]:
                py.drain(y_conss[c], a_act, y2(off, n))
        tg.finish()                                      # xn of both rows is in DDR
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_TOT, r * AA_BYTES + AA_XN, ELEM))
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *XL.w_regions(c)[3]))
            py.drain(y_conss[c], a_act, y2(*XL.y_regions(c)[3]))
        py.finish()                                      # q, gate, k, v of both rows are in DDR
        # o's weights and both rows' drains: they wait only for og
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, *XL.w_regions(c)[4]))
            off, n = XL.y_regions(c)[4]
            py.drain(y_conss[c], a_act, y2(off, n))
        # attention, row 0 then row 1: row r's cache row lands before row r + 1's window is read
        for r in range(NR):
            ab = r * AA_BYTES
            pa_out, pa_in = Pipeline(3), Pipeline(3)
            pa_out.drain(aout_c, a_kv, bt(STATE_TOT, (r + 1) * KV_ROW, KV_ROW))   # row pos + r (attnpos)
            pa_out.drain(aout_c, a_act, bt(ACT_TOT, ab + AA_OG, NHL * HD * 2))
            for c in range(1, ACORES):
                pa_out.drain(og_cs[c - 1], a_act, bt(ACT_TOT, ab + AA_OG + c * NHL * HD * 2, NHL * HD * 2))
            pa_in.fill(ain_p, a_consts, bt(CONSTS_T, CA_META, DA.E_A))
            pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, (r + 1) * PTAB_ROW, PTAB_ROW))
            pa_in.fill(ain_p, a_act, bt(ACT_TOT, ab + AA_QG, QW * 4))
            pa_in.fill(ain_p, a_act, bt(ACT_TOT, ab + AA_KVN, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(ACT_TOT, ab + AA_KVN + KVW * 4, KVW * 4))
            pa_in.fill(ain_p, a_kv, bt(STATE_TOT, 0, KV_ROW))                     # the window (attnpos, row r)
            pa_in.fill(ain_p, a_act, bt(ACT_TOT, ab + AA_QG + QW * 4, QW * 4))
            pa_out.finish()                              # row r's og and cache row are in DDR
            pa_in.finish()
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_TOT, r * AA_BYTES + AA_OG, QW * 2))
        py.finish()                                      # out of both rows is in DDR
        # the residual + post-attention norm of both rows, then one router pass for both
        V.ln2_router2(lni, lno, [((c_xres, bt(XRES_T, r * HID, HID)), (a_act, bt(ACT_TOT, r * AA_BYTES + AA_OUT, HID * 4)),
                                  (a_act, bt(ACT_TOT, r * AA_BYTES + AA_RES, HID * 4)), (a_act, bt(ACT_TOT, r * AA_BYTES + AA_XM, ELEM)),
                                  (a_act, bt(ACT_TOT, r * AA_BYTES + AA_ROUT, ELEM))) for r in range(NR)],
                      (a_consts, bt(CONSTS_T, CA_POSTLN, ELEM)),
                      lambda tg: XL.router_fill(lni, None, a_consts, CONSTS_T, CA_RW, tg))
        pw.finish()
        px.finish()


    def sequence(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                 side_p, gact_p, gout_c, pin_p, pout_c, ain_p, aout_c, og_cs):
        if part == 0:
            lin_front(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                      side_p, gact_p, gout_c, pin_p, pout_c)
        elif part == 1:
            lin_moe(a_pool, c_xres, a_consts, a_act, w_prods, x_prod, y_conss)
        elif part == 2:
            full_front(a_pool, c_xres, a_consts, a_state, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss,
                       ain_p, aout_c, og_cs)
        else:
            for r in range(NR):
                X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod,
                               y_conss, ACT_TOT, CONSTS_T, r * AA_BYTES + AA_XM, r * AA_BYTES + AA_ROUT,
                               r * AA_BYTES + AA_RES, r * AA_BYTES + AA_HP, CA_SGW, nx=X.NX_DECODE, xres_t=XRES_T,
                               xres_off=r * HID, slot0=r * X.NE, early_down=XL.X_WORDER, ug_bd=XL.X_SEQ, early_ug=XL.X_WORDER and XL.X_WUG)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [w_src[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0)),
                            of_ain.prod(tile=Tile(5, 0)), of_aout.cons(tile=Tile(6, 0)),
                            [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = vux
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes(), (HERE / "vrows.py").read_bytes(),
                   Path(__file__).read_bytes()]
                + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.cc"))
                + sorted(f.read_bytes() for f in ATTN.glob("*.h")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_q8.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   (HERE.parent.parent / "include" / "scalar_fp.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
