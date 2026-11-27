r"""vlx: lx.py's twin for two token rows per weight pass (the MoE verify pass, plan qwen36moe-production-bar.md phase 5).

Each row runs decode's own one-row arithmetic, so a row is bit-identical to an lx step by construction:
the q8 half tile is called once per row on the same weight chunk (each row has its own x table and y
element), and the conv glue, DeltaNet, post, the norm + router and the MoE block run once a row, in
order. Row loops are run-time loops: the lx main core has no program memory for unrolled copies.

Args: pool, xres f32[2][HID], consts, state [slot 0 | slot 1 | slot 2] (slot 0 the committed state,
row r reads slot r and writes slot r + 1), act [row 0 | row 1] (lx's act layout each).
Parts: 0 = the attention block for both rows and both routers; 1 = the MoE block, row 0 then row 1
(moeroute2 patches 16 routed slots: row r's slot j compiles as placeholder expert r * 8 + j).
Build: VLX_PART=<0|1> OPEN_KERNELS_SPEC=<spec.json> python build_design.py designs/layer_x/vlx.py designs/layer_x/build_vlx<part>
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
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (A_BYTES, A_HP, A_O, A_OG, A_OUT, A_QKV, A_RES, A_ROUT, A_VEC, A_XM, A_XN, A_Z,  # noqa: E402
                    C_BYTES, C_LNW, C_NW, C_POSTLN, C_RW, C_SGW, C_SIDE, C_WOUT, GLUE_SIDE_BYTES,
                    POOL_BYTES, POOL_QKV, POOL_Z, STATE_BYTES, STATE_S_OFF, S_HEAD_BYTES, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402
import vrows as V  # noqa: E402
from vrows import NR, YB  # noqa: E402

D = R.linear
if D is None or X.KIND != "moe" or D.CONV_F32 or D.XN_F32:
    sys.exit("vlx.py: the verify twin of the MoE linear layer (lx.py, not the dense or fp32-conv paths)")
if not ({"linear", "linear_out"} <= set(X.Q8)):
    sys.exit("vlx.py: built for q8 projections (the probed shape); q4 roles need gemv_q4's call twice too")
HID = X.HID
N_CORES = X.N_CORES
ELEM = X.ELEM
QKV_PC, Z_PC, OUT_PC, VW, OUT_K = D.QKV_PC, D.Z_PC, D.OUT_PC, D.VW, D.OUT_K
TILE, NT, G, NG, KEY_TILES = XL.TILE, XL.NT, XL.G, XL.NG, XL.KEY_TILES
ACT_T = NR * A_BYTES
STATE_T = (NR + 1) * STATE_BYTES
XRES_T = NR * HID
PART = int(os.environ.get("VLX_PART", 0))
bt = X.bt
X.ROLLED = True     # band loops keep one call site (recipes/catalogue.py ROLLED_BANDS); the core is at its limit


def rows3(tt: int, slot: int) -> TensorAccessPattern:
    """XL.rows3 in state slot `slot`."""
    return TensorAccessPattern((1, STATE_T), slot * STATE_BYTES + tt * TILE * 2,
                               [1, 1, XL.CONV_ROWS, TILE * 2], [0, 0, XL.NCH * 2, 1])


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def vlx(pool: In, xres: InOut, consts: In, state: InOut, act: InOut, *, part: CompileTime[int] = 0,
        srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    g = XL.glue_post_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(XRES_T,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(C_BYTES,), np.dtype[np.uint8]]
    state_ty = np.ndarray[(STATE_T,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(GEMV), str(GLUE), str(POST), str(X.LN), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    GK = XL.glue_post_kernels(inc, g)

    of_w, of_y, of_x = XL.main_fifos(t)
    w_src = XL.weight_ring(of_w, t)   # OPEN-DECODE-STREAM-LX
    of_lni, of_lno = XL.ln_fifos(tl)
    of_side, of_gact, of_gout, of_pin, of_pout = XL.glue_post_fifos(g)

    def main_body(win, xin, yout, tab1, *args):
        B, K = X.unpack_args(args)
        tab0 = B["tab"]
        V.prep2(xin, tab0, tab1, [K["prep2048"]])
        V.role_gemv2(win, yout, tab0, tab1, K, "linear", QKV_PC + Z_PC, HID)
        for _ in range_(NR):
            X.dn_body(win, yout, B, K)
        V.prep2(xin, tab0, tab1, [K["prep4096a"], K["prep4096b"]])
        V.role_gemv2(win, yout, tab0, tab1, K, "linear_out", OUT_PC, OUT_K)
        for _ in range_(NR):
            X.moe_body(win, xin, yout, B, K, nx=X.NX_DECODE)

    def glue2(*a):
        for _ in range_(NR):
            XL.glue_body(*a)

    def post2(*a):
        for _ in range_(NR):
            XL.post_body(*a)

    f32, facc = g["f32"], g["facc"]
    workers = ([Worker(V.ln_router2, fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"),
                                         Buffer(tl["racc"], name="racc"), Buffer(tl["xb"], name="rxs1"),
                                              Buffer(tl["racc"], name="racc1"), L["ln_nr"], L["ln"], L["rcopy"],
                                         L["racc"], L["rfin"]], tile=Tile(0, 3), stack_size=XL.STACK)]
               + XL.main_workers(main_body, of_w, of_x, of_y, t, K, extra=lambda c: [Buffer(t["tab"], name=f"tabr{c}")])
               + [Worker(post2, fn_args=[of_pin.cons(), of_pout.prod(), Buffer(g["nw"], name="nwb"),
                                         GK["post"], GK["post_copy"]], tile=Tile(1, 3), stack_size=XL.STACK),
                  Worker(glue2, fn_args=[of_side.cons(), of_gact.cons(), of_gout.prod(),
                                         Buffer(facc, name="acc_a"), Buffer(facc, name="acc_b"),
                                         Buffer(f32, name="decay"), Buffer(f32, name="beta"),
                                         Buffer(g["fqk"], name="qk"), Buffer(g["fvt"], name="vt"),
                                         Buffer(g["fxn"], name="xnb"),
                                         GK["ab"], GK["small"], GK["conv"], GK["emit"], GK["copy"]],
                         tile=Tile(2, 3), stack_size=XL.STACK)])

    BB_HID, BB_OUT = X.role_band_bytes("linear", HID), X.role_band_bytes("linear_out", OUT_K)

    def front(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
              side_p, gact_p, gout_c, pin_p, pout_c):
        # 1. both rows' entry norms
        tg = TaskGroup()
        for r in range(NR):
            lni.fill(c_xres, tap=bt(XRES_T, r * HID, HID), wait=True, group=tg)
            lni.fill(a_consts, tap=bt(C_BYTES, C_LNW, ELEM), wait=True, group=tg)
            lno.drain(a_act, tap=bt(ACT_T, r * A_BYTES + A_XN, ELEM), wait=True, group=tg)
        # 2. qkv | z for both rows: one weight pass
        pw, py, px = Pipeline(3), Pipeline(3), Pipeline(3)
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_QKV + c * QKV_PC * BB_HID, QKV_PC * BB_HID))
            pw.fill(w_prods[c], a_pool, bt(POOL_BYTES, POOL_Z + c * Z_PC * BB_HID, Z_PC * BB_HID))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_T, A_BYTES, A_QKV + c * QKV_PC * YB, QKV_PC * YB))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_T, A_BYTES, A_Z + c * Z_PC * YB, Z_PC * YB))
        tg.finish()                                      # xn of both rows is in DDR
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_T, r * A_BYTES + A_XN, ELEM))
        py.finish_oldest(*y_conss)                       # qkv of both rows is in DDR; row 0's glue runs during z
        # 3-5. glue, DeltaNet and post per row (conv state and S: slot r -> r + 1). Row 1's glue streams while row 0's
        # DeltaNet runs, and row 0's post while row 1's DeltaNet runs: their transfers are issued interleaved, since a
        # throttled channel blocks the host and would otherwise hold back the other core's transfers.
        rec, ohb = D.RECORD_BYTES, D.O_HEAD_BYTES

        def glue(r):
            ab = r * A_BYTES
            tg_s = TaskGroup()
            side_p.fill(a_act, tap=bt(ACT_T, ab + A_XN, ELEM), wait=True, group=tg_s)
            side_p.fill(a_consts, tap=bt(C_BYTES, C_SIDE, GLUE_SIDE_BYTES), wait=True, group=tg_s)
            pipe = Pipeline(3)
            for tt in range(NT):
                pipe.drain(gout_c, a_state, rows3(tt, r + 1))
                if tt >= KEY_TILES:
                    pipe.drain(gout_c, a_act, bt(ACT_T, ab + A_VEC + (tt - KEY_TILES) * D.HEADS_PER_TILE * rec,
                                                 D.HEADS_PER_TILE * rec))
                pipe.fill(gact_p, a_act, bt(ACT_T, ab + A_QKV + tt * TILE * 4, TILE * 4))
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
                    pw.fill(w_prods[c], a_act, bt(ACT_T, ab + A_VEC + hd * rec, X.CALL_BYTES))
                    pw.fill(w_prods[c], a_state, bt(STATE_T, s_in + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    py.drain(y_conss[c], a_state, bt(STATE_T, s_out + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    py.drain(y_conss[c], a_act, bt(ACT_T, ab + A_O + hd * ohb, ohb))
                    pw.fill(w_prods[c], a_state, bt(STATE_T, s_in + hd * S_HEAD_BYTES, S_HEAD_BYTES))
                    yield
            py.finish()                                  # row r's S' and o are in DDR

        def post(r):
            ab = r * A_BYTES
            pipe = Pipeline(3)
            pipe.fill(pin_p, a_consts, bt(C_BYTES, C_NW, ELEM))
            for gg in range(NG):
                pipe.drain(pout_c, a_act, bt(ACT_T, ab + A_OG + gg * G * 2, G * 2))
                pipe.fill(pin_p, a_act, bt(ACT_T, ab + A_O + gg * G * 4, G * 4))
                pipe.fill(pin_p, a_act, bt(ACT_T, ab + A_Z + gg * G * 4, G * 4))
                yield
            pipe.finish()

        V.interleave(glue(0))
        V.interleave(deltanet(0), glue(1))               # glue(1) reads conv slot 1, which glue(0) wrote
        V.interleave(deltanet(1), post(0))
        # 6a. the out projection's weights and both rows' drains
        for c in range(N_CORES):
            pw.fill(w_prods[c], a_consts, bt(C_BYTES, C_WOUT + c * OUT_PC * BB_OUT, OUT_PC * BB_OUT))
            py.drain(y_conss[c], a_act, V.y2_tap(ACT_T, A_BYTES, A_OUT + c * OUT_PC * YB, OUT_PC * YB))
        V.interleave(post(1))
        # 6b. out against both rows' og
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_T, r * A_BYTES + A_OG, VW * 2))
        py.finish()                                      # out of both rows is in DDR
        # the residual + post-attention norm of both rows, then one router pass for both
        V.ln2_router2(lni, lno, [((c_xres, bt(XRES_T, r * HID, HID)), (a_act, bt(ACT_T, r * A_BYTES + A_OUT, HID * 4)),
                                  (a_act, bt(ACT_T, r * A_BYTES + A_RES, HID * 4)), (a_act, bt(ACT_T, r * A_BYTES + A_XM, ELEM)),
                                  (a_act, bt(ACT_T, r * A_BYTES + A_ROUT, ELEM))) for r in range(NR)],
                      (a_consts, bt(C_BYTES, C_POSTLN, ELEM)),
                      lambda tg: XL.router_fill(lni, None, a_consts, C_BYTES, C_RW, tg))
        pw.finish()
        px.finish()

    def moe(a_pool, c_xres, a_consts, a_act, w_prods, x_prod, y_conss):
        # decode's MoE stream once a row: X.moe_sequence on that row's act half and xres row
        for r in range(NR):
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod,
                           y_conss, ACT_T, C_BYTES, r * A_BYTES + A_XM, r * A_BYTES + A_ROUT, r * A_BYTES + A_RES,
                           r * A_BYTES + A_HP, C_SGW, nx=X.NX_DECODE, xres_t=XRES_T, xres_off=r * HID,
                           slot0=r * X.NE, early_down=XL.X_WORDER, ug_bd=XL.X_SEQ, early_ug=XL.X_WORDER and XL.X_WUG)

    def sequence(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                 side_p, gact_p, gout_c, pin_p, pout_c):
        if part == 0:
            front(a_pool, c_xres, a_consts, a_state, a_act, lni, lno, w_prods, x_prod, y_conss,
                  side_p, gact_p, gout_c, pin_p, pout_c)
        else:
            moe(a_pool, c_xres, a_consts, a_act, w_prods, x_prod, y_conss)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, state_ty, act_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [w_src[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_side.prod(tile=Tile(2, 0)), of_gact.prod(tile=Tile(3, 0)), of_gout.cons(tile=Tile(2, 0)),
                            of_pin.prod(tile=Tile(4, 0)), of_pout.cons(tile=Tile(1, 0))])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = vlx
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc")) + sorted(f.read_bytes() for f in HERE.glob("*.h"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes(), (HERE / "vrows.py").read_bytes(),
                   Path(__file__).read_bytes()]
                + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in GLUE.glob("*.cc")) + sorted(f.read_bytes() for f in GLUE.glob("*.h"))
                + sorted(f.read_bytes() for f in POST.glob("*.cc")) + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_q8.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
