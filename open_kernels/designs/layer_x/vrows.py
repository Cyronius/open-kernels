"""Two-row helpers shared by the verify images vlx.py and vax.py."""

from __future__ import annotations

from aie.iron.controlflow import range_
from aie.helpers.taplib import TensorAccessPattern

import xcommon as X

NR = 2                                        # token rows a pass
YB = X.BAND_ROWS * 4


def gemv2_bands(win, yout, tab0, tab1, gy, nbands, ngroups, per_band, rs):
    """X.gemv_bands for two rows: each weight element goes to the one-row tile once per row."""
    for _ in range_(nbands):
        ye = yout.acquire(2)
        for g in X.band_range(ngroups):
            we = win.acquire(1)
            gy(we, tab0, ye[0], g, per_band, rs)
            gy(we, tab1, ye[1], g, per_band, rs)
            win.release(1)
        yout.release(2)


def role_gemv2(win, yout, tab0, tab1, K, role, nbands, kk):
    gemv2_bands(win, yout, tab0, tab1, K["gy8"], nbands, X.role_groups(role, kk), X.role_per_band(role, kk), 4)


def prep2(xin, tab0, tab1, preps):
    """Each row's activation into its own table, one 4 KB element per call (an acquire(2) would make the x fifo
    three buffers deep)."""
    for tb in (tab0, tab1):
        for prep in preps:
            xe = xin.acquire(1)
            prep(xe, tb)
            xin.release(1)


def y2_tap(act_t: int, row_bytes: int, off: int, nbytes: int) -> TensorAccessPattern:
    """A core's band results as [band][row][64 f32], each row into its own act half."""
    return TensorAccessPattern((1, act_t), off, [1, nbytes // YB, NR, YB], [0, YB, row_bytes, 1])


def ln_router2(ain, aout, xs0, acc0, xs1, acc1, f_nr, f_ln, f_rc, f_ra, f_rf):
    """X.ln_router_body's stages for two rows: both entry norms, both residual norms, then one pass over the
    router's W for both rows (each row's sums in its own accumulator, in decode's order)."""
    for _ in range_(NR):
        e = ain.acquire(3)
        o = aout.acquire(1)
        f_nr(e[0], e[1], e[2], o)
        aout.release(1)
        ain.release(3)
    for xs in (xs0, xs1):
        e = ain.acquire(5)
        oo = aout.acquire(3)
        f_ln(e[0], e[1], e[3], e[4], e[2], oo[0], oo[1], oo[2])
        f_rc(oo[2], xs)
        aout.release(3)
        ain.release(5)
    for rb in range_(X.W_ELEMS):
        e = ain.acquire(1)
        f_ra(e, xs0, acc0, rb)
        f_ra(e, xs1, acc1, rb)
        ain.release(1)
    for acc in (acc0, acc1):
        o = aout.acquire(1)
        f_rf(acc, o)
        aout.release(1)


def ln2_router2(lni, lno, rows, postln, router):
    """Host side of ln_router2's second half. rows: per row (xres tap, out tap, res tap, xm tap, rout tap);
    postln: the norm weight's tap; router(group) fills W once."""
    from ironutil import Pipeline
    from aie.iron import TaskGroup
    p = Pipeline(3)
    for xres, out, res, xm, _ in rows:
        p.fill(lni, *xres)
        p.fill(lni, *postln)
        p.fill(lni, *out)
        p.drain(lno, *res)
        p.drain(lno, *xm)
    p.finish()                                       # res and xm of both rows are in DDR
    tg = TaskGroup()
    router(tg)
    for *_, rout in rows:
        lno.drain(rout[0], tap=rout[1], wait=True, group=tg)
    tg.finish()


def interleave(*gens):
    """Issue several host-sequence generators' transfers round-robin (each yields after a few transfers)."""
    live = list(gens)
    while live:
        for g in list(live):
            try:
                next(g)
            except StopIteration:
                live.remove(g)
