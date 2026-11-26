r"""vax: ax.py's twin for two token rows per weight pass (the MoE verify pass, plan qwen36moe-production-bar.md phase 5).

The full-attention layer of vlx.py: each weight element goes to decode's one-row q8 tile once per row, and the
attention cores, the norm + router and the MoE block run once a row. Row r is position pos + r: its new K / V
row drains to placeholder row r + 1 and its position record is ptab row r + 1 (stream_patch attn_table's
multi-row form); row 1's window fill waits for row 0's cache row, so row 1 attends to row 0.

Args: pool, xres f32[2][HID], consts, kv, act [row 0 | row 1] (ax's act layout each), ptab.
Parts: 0 = attention block for both rows and both routers; 1 = MoE, row 0 then row 1 (placeholder experts r * 8 + j).
Build: VAX_PART=<0|1> OPEN_KERNELS_SPEC=<spec.json> python build_design.py designs/layer_x/vax.py designs/layer_x/build_vax<part>
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

HERE = Path(__file__).parent
GEMV = HERE.parent / "gemv_q4"
ATTN = HERE.parent / "attn"
sys.path.insert(0, str(HERE.parent.parent))
sys.path.insert(0, str(HERE))
from ironutil import Pipeline, include_dirs  # noqa: E402
from layout import (AA_BYTES, AA_HP, AA_KVN, AA_OG, AA_OUT, AA_QG, AA_RES, AA_ROUT, AA_XM, AA_XN,  # noqa: E402
                    CA_BYTES, CA_LNW, CA_META, CA_POSTLN, CA_RW, CA_SGW, KV_BYTES, KV_ROW, POOL_BYTES,
                    PTAB_BYTES, PTAB_ROW, R, SPEC)
import xcommon as X  # noqa: E402
import xlayer as XL  # noqa: E402
import vrows as V  # noqa: E402
from vrows import NR  # noqa: E402

D = R.attn
if D is None or X.KIND != "moe" or D.XN_F32 or D.KV_PAIR:
    sys.exit("vax.py: the verify twin of the MoE full-attention layer (ax.py, not the dense or paired-KV paths)")
if "attn" not in X.Q8:
    sys.exit("vax.py: built for q8 projections (the probed shape)")
HID = X.HID
HD = D.HD
N_CORES = X.N_CORES
ELEM = X.ELEM
Q_PC, KV_PC, O_PC = D.Q_PC, D.KV_PC, D.O_PC
QW, KVW, O_K = D.QW, D.KVW, D.O_K
ACORES, NHL = XL.ACORES, XL.NHL
ACT_T = NR * AA_BYTES
XRES_T = NR * HID
PART = int(os.environ.get("VAX_PART", 0))
bt = X.bt
X.ROLLED = True     # band loops keep one call site, as in vlx.py


def twice(body):
    def run(*a):
        for _ in range_(NR):
            body(*a)
    return run


@iron.jit(aiecc_flags=["--alloc-scheme=basic-sequential"])
def vax(pool: In, xres: InOut, consts: In, kv: InOut, act: InOut, ptab: In, *, part: CompileTime[int] = 0,
        srchash: CompileTime[int] = 0):
    t = X.types()
    tl = X.ln_types()
    a = XL.attn_types()
    pool_ty = np.ndarray[(POOL_BYTES,), np.dtype[np.uint8]]
    xres_ty = np.ndarray[(XRES_T,), np.dtype[np.float32]]
    consts_ty = np.ndarray[(CA_BYTES,), np.dtype[np.uint8]]
    kv_ty = np.ndarray[(KV_BYTES,), np.dtype[np.uint8]]
    act_ty = np.ndarray[(ACT_T,), np.dtype[np.uint8]]
    ptab_ty = np.ndarray[(PTAB_BYTES,), np.dtype[np.uint8]]

    inc = include_dirs() + [str(GEMV), str(ATTN), str(X.LN), str(X.LINL), str(X.RT), str(HERE.parent / "moe_experts")]
    K = X.kernels(inc, t)
    L = X.ln_kernels(inc, tl)
    afns = XL.attn_kernels(inc, a)

    of_w, of_y, of_x = XL.main_fifos(t)
    w_src = XL.weight_ring(of_w, t, XL.X_WRING_AX)   # OPEN-DECODE-STREAM-LX
    of_lni, of_lno = XL.ln_fifos(tl)
    of_ain, of_aout, of_og = XL.attn_fifos(a)

    def main_body(win, xin, yout, tab1, *args):
        B, K = X.unpack_args(args)
        tab0 = B["tab"]
        V.prep2(xin, tab0, tab1, [K["prep2048"]])
        V.role_gemv2(win, yout, tab0, tab1, K, "attn", 2 * Q_PC + 2 * KV_PC, HID)   # q | gate | k | v
        V.prep2(xin, tab0, tab1, [K["prep4096a"], K["prep4096b"]])
        V.role_gemv2(win, yout, tab0, tab1, K, "attn", O_PC, O_K)
        for _ in range_(NR):
            X.moe_body(win, xin, yout, B, K, nx=X.NX_DECODE)

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
    workers = ([Worker(V.ln_router2, fn_args=[of_lni.cons(), of_lno.prod(), Buffer(tl["xb"], name="rxs"),
                                              Buffer(tl["racc"], name="racc"), Buffer(tl["xb"], name="rxs1"),
                                              Buffer(tl["racc"], name="racc1"), L["ln_nr"], L["ln"], L["rcopy"],
                                              L["racc"], L["rfin"]], tile=Tile(0, 3), stack_size=XL.STACK)]
               + XL.main_workers(main_body, of_w, of_x, of_y, t, K, extra=lambda c: [Buffer(t["tab"], name=f"tabr{c}")])
               + attn_workers)

    def y2(off, n):
        return V.y2_tap(ACT_T, AA_BYTES, off, n)

    def front(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs):
        # both rows' entry norms
        tg = TaskGroup()
        for r in range(NR):
            lni.fill(c_xres, tap=bt(XRES_T, r * HID, HID), wait=True, group=tg)
            lni.fill(a_consts, tap=bt(CA_BYTES, CA_LNW, ELEM), wait=True, group=tg)
            lno.drain(a_act, tap=bt(ACT_T, r * AA_BYTES + AA_XN, ELEM), wait=True, group=tg)
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
            px.fill(x_prod, a_act, bt(ACT_T, r * AA_BYTES + AA_XN, ELEM))
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
            pa_out.drain(aout_c, a_kv, bt(KV_BYTES, (r + 1) * KV_ROW, KV_ROW))   # row pos + r (attnpos)
            pa_out.drain(aout_c, a_act, bt(ACT_T, ab + AA_OG, NHL * HD * 2))
            for c in range(1, ACORES):
                pa_out.drain(og_cs[c - 1], a_act, bt(ACT_T, ab + AA_OG + c * NHL * HD * 2, NHL * HD * 2))
            pa_in.fill(ain_p, a_consts, bt(CA_BYTES, CA_META, D.E_A))
            pa_in.fill(ain_p, a_ptab, bt(PTAB_BYTES, (r + 1) * PTAB_ROW, PTAB_ROW))
            pa_in.fill(ain_p, a_act, bt(ACT_T, ab + AA_QG, QW * 4))
            pa_in.fill(ain_p, a_act, bt(ACT_T, ab + AA_KVN, KVW * 4))
            pa_in.fill(ain_p, a_act, bt(ACT_T, ab + AA_KVN + KVW * 4, KVW * 4))
            pa_in.fill(ain_p, a_kv, bt(KV_BYTES, 0, KV_ROW))                     # the window (attnpos, row r)
            pa_in.fill(ain_p, a_act, bt(ACT_T, ab + AA_QG + QW * 4, QW * 4))
            pa_out.finish()                              # row r's og and cache row are in DDR
            pa_in.finish()
        for r in range(NR):
            px.fill(x_prod, a_act, bt(ACT_T, r * AA_BYTES + AA_OG, QW * 2))
        py.finish()                                      # out of both rows is in DDR
        # the residual + post-attention norm of both rows, then one router pass for both
        V.ln2_router2(lni, lno, [((c_xres, bt(XRES_T, r * HID, HID)), (a_act, bt(ACT_T, r * AA_BYTES + AA_OUT, HID * 4)),
                                  (a_act, bt(ACT_T, r * AA_BYTES + AA_RES, HID * 4)), (a_act, bt(ACT_T, r * AA_BYTES + AA_XM, ELEM)),
                                  (a_act, bt(ACT_T, r * AA_BYTES + AA_ROUT, ELEM))) for r in range(NR)],
                      (a_consts, bt(CA_BYTES, CA_POSTLN, ELEM)),
                      lambda tg: XL.router_fill(lni, None, a_consts, CA_BYTES, CA_RW, tg))
        pw.finish()
        px.finish()

    def sequence(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs):
        if part == 0:
            front(a_pool, c_xres, a_consts, a_kv, a_act, a_ptab, lni, lno, w_prods, x_prod, y_conss, ain_p, aout_c, og_cs)
            return
        for r in range(NR):
            X.moe_sequence(Pipeline(3), Pipeline(3), Pipeline(3), a_pool, a_consts, a_act, c_xres, w_prods, x_prod,
                           y_conss, ACT_T, CA_BYTES, r * AA_BYTES + AA_XM, r * AA_BYTES + AA_ROUT, r * AA_BYTES + AA_RES,
                           r * AA_BYTES + AA_HP, CA_SGW, nx=X.NX_DECODE, xres_t=XRES_T, xres_off=r * HID,
                           slot0=r * X.NE, early_down=XL.X_WORDER, ug_bd=XL.X_SEQ)

    rt = Runtime(sequence, [pool_ty, xres_ty, consts_ty, kv_ty, act_ty, ptab_ty,
                            of_lni.prod(tile=Tile(0, 0)), of_lno.cons(tile=Tile(0, 0)),
                            [w_src[c].prod(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_x.prod(tile=Tile(1, 0)),
                            [of_y[c].cons(tile=Tile(c, 0)) for c in range(N_CORES)],
                            of_ain.prod(tile=Tile(2, 0)), of_aout.cons(tile=Tile(1, 0)),
                            [of_og[c].cons(tile=Tile(3 + c, 0)) for c in range(ACORES - 1)]])
    return Program(iron.get_current_device(), rt, workers=workers).resolve_program()


DESIGN = vax
_src = b"".join(sorted(f.read_bytes() for f in HERE.glob("*.cc"))
                + [(HERE / "xcommon.py").read_bytes(), (HERE / "xlayer.py").read_bytes(), (HERE / "vrows.py").read_bytes(),
                   Path(__file__).read_bytes()]
                + X.source_hash_inputs()
                + sorted(f.read_bytes() for f in ATTN.glob("*.cc")) + sorted(f.read_bytes() for f in ATTN.glob("*.h"))
                + sorted(f.read_bytes() for f in X.RT.glob("*.cc"))
                + [(X.LN / "ln.cc").read_bytes(), (X.LN / "ln.h").read_bytes(), (X.LINL / "ln_nr.cc").read_bytes(),
                   (GEMV / "gemv_q4.h").read_bytes(), (GEMV / "gemv_q8.h").read_bytes(), (GEMV / "gemv_tab.h").read_bytes(),
                   (HERE.parent.parent / "include" / "vecmath.h").read_bytes(),
                   (HERE.parent.parent / "include" / "scalar_fp.h").read_bytes(), SPEC.spec_hash().encode()])
SPECIALIZE = {"part": PART, "srchash": int(hashlib.sha1(_src).hexdigest()[:8], 16)}
