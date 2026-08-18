"""FV3 ``tracer_2d_1L`` for the six-face duo lane -- the large-time-step
tracer transport on the acoustic loop's accumulated mass fluxes.

ORACLE (published, checksum-pinned tree; ``model/fv_tracer2d.F90``):

    fv_dynamics.F90:472-478   dp1 = delp            (BEFORE dyn_core)
    dyn_core.F90:313-316      mfx/mfy/cx/cy capacitors zeroed
    sw_core.F90:903-920       d_sw1 accumulates crx_adv/fx into them,
                              EVERY acoustic sub-step, BEFORE the
                              inter-panel allflux averaging
    fv_dynamics.F90:534       tracer_2d_1L(q, dp1, mfx, mfy, cx, cy, ...)

THE LIVE ARM IS ``tracer_2d_1L``.  The resolved deck echoes
``Z_TRACER=T`` and the duo runs have ``gridstruct%dg%is_initialized``,
so ``fv_dynamics.F90:528-540`` takes the ``z_tracer`` branch, not
``tracer_2d`` and not ``tracer_2d_nested``.  The other arms are refused,
not silently substituted.

THE CAPACITORS ARE THE UN-AVERAGED FLUXES, ON PURPOSE.  ``d_sw1``
accumulates ``xflux = xflux + fx`` (sw_core.F90:910) from ITS OWN
``fv_tp_2d(delp)`` fluxes, and dyn_core's inter-panel averaging block
(dyn_core.F90:853-900) then edits ``allflux`` -- a different array --
so the capacitor keeps the pre-average value at panel edges.  The
tracer fluxes computed HERE get their own seam averaging (``flux_adj``,
fv_tracer2d.F90:46-108), but ``dp2 = dp1 + div(mfx)`` uses the
un-averaged capacitor.  The port reproduces exactly that; "fixing" the
inconsistency would break oracle parity.

DECK FLAGS (resolved from the run logfile echo, not the namelist text,
because several keys appear twice in ``input.nml``):

    HORD_TR=6  KORD_TR=9  Q_SPLIT=0  Z_TRACER=T  DNATS=1  NCNST(field
    table)=3 -> nr=2 advected (sphum, liq_wat); rainwat is inert.
    TRDM2=0.0  NORD_TR=0  LIM_FAC=1.0  INLINE_Q=F

``q_split`` is accepted for signature parity but THIS TREE'S
``tracer_2d_1L`` never reads it -- its body computes
``nsplt = int(1.+cmax(k))`` unconditionally (fv_tracer2d.F90:247;
the ``q_split == 0`` branch exists only in ``tracer_2d`` at :510-522).
A non-zero value is refused so a deck that expects the fixed-split
behaviour fails loudly instead of silently getting the dynamic one.

SUBCYCLING IS PER LEVEL AND GLOBAL ACROSS FACES.  ``cmax(k)`` is
reduced with ``mp_reduce_max`` over ranks (fv_tracer2d.F90:249); the
six faces are this port's ranks, so the per-level max is taken over
all six BEFORE ``nsplt`` -- a per-face nsplt would subcycle the same
level differently on different faces, which the oracle never does.
On this deck (npz=5) the ``k < npz/6`` arm of the cmax formula is
dead (5/6 == 0 in integer division) and every level uses
``max(|cx|,|cy|) + 1 - sin_sg(i,j,5)``.
"""
from __future__ import annotations

import numpy as np

from legoesm.core.fv3_native_d_sw import fl_limiter as _fl, fv_tp_2d
from legoesm.grids.fv3_native_gridstruct import (
    average_shared_edge_cgrid,
    fort,
)

# fv_tracer2d.F90:243 -- `if (trdm > 1.e-4)` gates the dp1 halo
# pre-exchange (and nothing else in tracer_2d_1L: unlike upstream FV3,
# this tree's fv_tp_2d call at :302-304 carries no nord/damp arguments).
TRDM_ACTIVE_MIN = 1.0e-4


def alloc_flux_capacitors(n: int, ng: int, km: int) -> list:
    """The mfx/mfy/cx/cy capacitors, zeroed -- dyn_core.F90:313-316.

    Shapes are the per-level arrays ``d_sw1_duo`` already takes (its
    ``xflux``/``yflux``/``cx``/``cy`` dummies), with a trailing level
    axis:

        mfx (npx, m_a, km)  origin (is,  js)   used i=is..ie+1, j=js..je
        mfy (m_a, npx, km)  origin (is,  js)   used i=is..ie,   j=js..je+1
        cx  (npx, m_a, km)  origin (is,  jsd)  used i=is..ie+1, j=jsd..jed
        cy  (m_a, npx, km)  origin (isd, js)   used i=isd..ied, j=js..je+1

    (``npx = n+1``, ``m_a = n+2*ng``; the mfx columns beyond ``je`` and
    mfy rows beyond ``ie`` exist only because the caller-side arrays are
    allocated rectangular -- they stay zero.)
    """
    npx = n + 1
    m_a = n + 2 * ng
    z = np.zeros
    return [{"mfx": z((npx, m_a, km), dtype=np.float64),
             "mfy": z((m_a, npx, km), dtype=np.float64),
             "cx": z((npx, m_a, km), dtype=np.float64),
             "cy": z((m_a, npx, km), dtype=np.float64)}
            for _ in range(6)]


def require_tracer_2d_1l_lane(*, z_tracer: bool, q_split: int,
                              nord_tr: int, trdm: float,
                              inline_q: bool = False) -> None:
    """Refuse every fv_tracer2d arm this port does not carry."""
    if inline_q:
        raise NotImplementedError(
            "inline_q=True advects tracers INSIDE d_sw1 (sw_core.F90:"
            "976-996) and skips tracer_2d entirely (fv_dynamics.F90:517); "
            "d_sw1_duo refuses that lane and so does this one. The "
            "resolved deck echoes INLINE_Q=F.")
    if not z_tracer:
        raise NotImplementedError(
            "z_tracer=False selects tracer_2d (fv_tracer2d.F90:391), the "
            "GLOBAL-split arm with its own q_split/ksplt logic -- a "
            "different routine, not a parameter of this one. The resolved "
            "deck echoes Z_TRACER=T, which selects tracer_2d_1L "
            "(fv_dynamics.F90:533-536).")
    if q_split != 0:
        raise NotImplementedError(
            f"q_split={q_split}: this tree's tracer_2d_1L IGNORES q_split "
            f"(its nsplt is always int(1+cmax), fv_tracer2d.F90:247); a "
            f"deck that sets q_split expects the fixed-split behaviour of "
            f"tracer_2d and must not get the dynamic one silently. The "
            f"resolved deck echoes Q_SPLIT=0.")
    if nord_tr != 0:
        raise NotImplementedError(
            f"nord_tr={nord_tr}: tracer del-n damping. This tree's "
            f"tracer_2d_1L fv_tp_2d call (fv_tracer2d.F90:302-304) carries "
            f"no nord/damp arguments, so nord_tr is dead there; refusing a "
            f"non-zero value rather than silently dropping it. The "
            f"resolved deck echoes NORD_TR=0.")
    if trdm > TRDM_ACTIVE_MIN:
        raise NotImplementedError(
            f"trdm={trdm} > {TRDM_ACTIVE_MIN}: the dp1 halo pre-exchange "
            f"(fv_tracer2d.F90:243-256) is not ported (it feeds nothing in "
            f"this tree's tracer_2d_1L, but a deck that sets trdm2 expects "
            f"upstream's damped-flux variant). The resolved deck echoes "
            f"TRDM2=0.0.")


def _q_halo_exchange_sixface(ctx: dict, f6: list) -> None:
    """``ext_scalar(q, gridstruct%dg, bd, domain, 0, 0)`` analog
    (fv_tracer2d.F90:276-281 for q; :371-377 for qn2 between subcycles).

    Same authoritative path and the same fail-closed contract as the
    delp/pt exchange in ``fv3_native_acoustic_3d``: a context without
    the ext bundle must declare the substitution with
    ``ext_exclude=('ascalar',)`` rather than silently get the
    edge-strip index-copy helper, whose corner-diagonal gap is exactly
    what the tracer PPM stencils read near panel corners.

    Write-back through the caller's list entries: the certified delp/pt
    site passes level views and relies on in-place writes, but the
    D-wind site documents that an exchange MAY replace list entries, so
    the caller copies ``f6[t]`` back into its slabs after this returns.
    """
    n, ng = ctx["n"], ctx["ng"]
    use_ext = bool(ctx.get("use_ext_bundle")) and ctx.get("ectx") is not None
    excluded = "ascalar" in tuple(ctx.get("ext_exclude", ()))
    if use_ext and not excluded:
        from legoesm.grids.fv3_native_ext_vector import ext_scalar_sixface
        ext_scalar_sixface(f6, "A", ctx["ectx"])
        return
    if not use_ext and not excluded:
        raise ValueError(
            "_q_halo_exchange_sixface: fv_tracer2d.F90:276-281 fills the "
            "tracer halos with ext_scalar on the duo lane, and this "
            "context has no ext bundle. Build it with use_ext_bundle="
            "True, or declare the substitution with "
            "ext_exclude=('ascalar',).")
    from legoesm.grids.fv3_native_gridstruct import (
        exchange_agrid_scalar_halos,
    )
    for t in range(1, 7):
        exchange_agrid_scalar_halos(f6, t, n, ng)


def _tp_gsf(gs: dict, bd) -> dict:
    """fort-wrapped gridstruct dict for fv_tp_2d's callees -- the same
    keys ``d_sw1_duo`` hands them, for the same reason (raw numpy would
    wrap negative Fortran indices silently)."""
    isd, jsd = bd.isd, bd.jsd
    return {
        "dxa": fort(gs["dxa"], isd, jsd),
        "dya": fort(gs["dya"], isd, jsd),
        "area": fort(gs["area"], isd, jsd),
        "rarea": fort(gs["rarea"], isd, jsd),
        "del6_v": fort(gs["del6_v"], isd, jsd),
        "del6_u": fort(gs["del6_u"], isd, jsd),
        "da_min": float(gs["da_min"]),
        "bounded_domain": bool(gs.get("bounded_domain", False)),
        "grid_type": int(gs.get("grid_type", 0)),
        "sw_corner": bool(gs.get("sw_corner", True)),
        "se_corner": bool(gs.get("se_corner", True)),
        "nw_corner": bool(gs.get("nw_corner", True)),
        "ne_corner": bool(gs.get("ne_corner", True)),
    }


def tracer_2d_1l_sixface(ctx: dict, q6: list, dp1_6: list, flux_cap: list,
                         *, km: int, nq: int, hord_tr: int, dt: float,
                         q_split: int = 0, nord_tr: int = 0,
                         trdm: float = 0.0, lim_fac: float = 1.0,
                         z_tracer: bool = True,
                         inline_q: bool = False) -> None:
    """``tracer_2d_1L`` (fv_tracer2d.F90:114-389), all six faces, in place.

    ``q6``       [face][iq] tracer arrays (m_a, m_a, km), padded like
                 delp; ADVECTED IN PLACE.
    ``dp1_6``    [face] (m_a, m_a, km) -- delp as it was BEFORE dyn_core
                 (fv_dynamics.F90:472-478).  Mutated: the subcycled
                 branch writes ``dp1 = dp2`` between subcycles (:361-365),
                 exactly as the Fortran dummy is; the caller passes a
                 per-n_map copy.
    ``flux_cap`` [face] dicts from :func:`alloc_flux_capacitors`, as
                 accumulated by the acoustic loop.  Mutated: the frac
                 rescale at :262-291 divides them by nsplt in place.
    ``dt``       mdt = bdt/k_split (fv_dynamics.F90:413) -- accepted for
                 signature parity with the Fortran; the body never uses
                 it (the capacitors already carry the time integral),
                 which is also true upstream.

    The seam treatment is the duo arm throughout: q halos via
    ``ext_scalar`` and the per-tracer fluxes averaged across panel
    edges by ``flux_adj`` (ported as
    :func:`...fv3_native_gridstruct.average_shared_edge_cgrid`, the
    same CGRID_NE boundary-blend dyn_core's barrier 1 uses -- flux_adj
    is that operation on a single flux pair).
    """
    require_tracer_2d_1l_lane(z_tracer=z_tracer, q_split=q_split,
                              nord_tr=nord_tr, trdm=trdm,
                              inline_q=inline_q)
    del dt  # signature parity only; see docstring.
    if len(q6) != 6 or len(dp1_6) != 6 or len(flux_cap) != 6:
        raise ValueError(
            f"q6/dp1_6/flux_cap must each have 6 faces, got "
            f"{len(q6)}/{len(dp1_6)}/{len(flux_cap)}")
    for t in range(6):
        if len(q6[t]) != nq:
            raise ValueError(
                f"face {t + 1}: {len(q6[t])} tracer arrays, expected "
                f"nq={nq}")

    n, ng, bd = ctx["n"], ctx["ng"], ctx["bd"]
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    npx = n + 1
    m_a = n + 2 * ng
    for t in range(6):
        if not bool(ctx["gs6"][t].get("bounded_domain", False)):
            raise ValueError(
                f"face {t + 1}: gridstruct is not bounded_domain -- the "
                f"duo tracer lane (ext_scalar halos + flux_adj seam "
                f"averaging) is only certified on the bounded duo "
                f"gridstructs; the plain-conventions lane would need "
                f"copy_corners/edge branches this port does not take.")
        for iq in range(nq):
            if q6[t][iq].shape != (m_a, m_a, km):
                raise ValueError(
                    f"face {t + 1} tracer {iq}: shape "
                    f"{q6[t][iq].shape}, expected {(m_a, m_a, km)}")
        if dp1_6[t].shape != (m_a, m_a, km):
            raise ValueError(
                f"face {t + 1}: dp1 shape {dp1_6[t]. shape}, expected "
                f"{(m_a, m_a, km)}")

    gsf6 = [_tp_gsf(ctx["gs6"][t], bd) for t in range(6)]

    # ---- xfx/yfx from the Courant capacitors (:191-215) + cmax (:217-231)
    xfx6 = [[None] * km for _ in range(6)]
    yfx6 = [[None] * km for _ in range(6)]
    cmax_face = np.zeros((6, km), dtype=np.float64)
    for t in range(6):
        gs = ctx["gs6"][t]
        DXA = fort(gs["dxa"], isd, jsd)
        DYA = fort(gs["dya"], isd, jsd)
        DX = fort(gs["dx"], isd, jsd)
        DY = fort(gs["dy"], isd, jsd)
        SIN_SG = fort(gs["sin_sg"], isd, jsd)
        for k in range(km):
            cxf = fort(flux_cap[t]["cx"][:, :, k], is_, jsd)
            cyf = fort(flux_cap[t]["cy"][:, :, k], isd, js)
            xfx = _fl(is_, ie + 1, jsd, jed)
            yfx = _fl(isd, ied, js, je + 1)
            for j in range(jsd, jed + 1):
                for i in range(is_, ie + 1 + 1):
                    if cxf[i, j] > 0.0:
                        xfx[i, j] = (cxf[i, j] * DXA[i - 1, j] * DY[i, j]
                                     * SIN_SG[i - 1, j, 3 - 1])
                    else:
                        xfx[i, j] = (cxf[i, j] * DXA[i, j] * DY[i, j]
                                     * SIN_SG[i, j, 1 - 1])
            for j in range(js, je + 1 + 1):
                for i in range(isd, ied + 1):
                    if cyf[i, j] > 0.0:
                        yfx[i, j] = (cyf[i, j] * DYA[i, j - 1] * DX[i, j]
                                     * SIN_SG[i, j - 1, 4 - 1])
                    else:
                        yfx[i, j] = (cyf[i, j] * DYA[i, j] * DX[i, j]
                                     * SIN_SG[i, j, 2 - 1])
            xfx6[t][k] = xfx
            yfx6[t][k] = yfx

            # :217-231. Fortran `if (k < npz/6)` with 1-based k and
            # INTEGER division -- on npz=5 the first arm is dead.
            cm = 0.0
            if (k + 1) < km // 6:
                for j in range(js, je + 1):
                    for i in range(is_, ie + 1):
                        cm = max(cm, abs(cxf[i, j]), abs(cyf[i, j]))
            else:
                for j in range(js, je + 1):
                    for i in range(is_, ie + 1):
                        cm = max(cm, max(abs(cxf[i, j]), abs(cyf[i, j]))
                                 + 1.0 - SIN_SG[i, j, 5 - 1])
            cmax_face[t, k] = cm

    # trdm > 1e-4 would exchange dp1 halos here (:243-256) -- refused
    # at entry, so nothing to do.

    # :258 mp_reduce_max(cmax, npz) -- the six faces are the ranks.
    cmax = cmax_face.max(axis=0)

    # ---- frac rescale of the capacitors (:262-291), per level ----------
    nsplt_k = [int(1.0 + cmax[k]) for k in range(km)]
    for k in range(km):
        nsplt = nsplt_k[k]
        if nsplt > 1:
            frac = 1.0 / float(nsplt)
            for t in range(6):
                # cx over (is..ie+1, jsd..jed) == the whole slab;
                # mfx over (is..ie+1, js..je) -- the live columns only
                # (the rest are structurally zero, but mirror the
                # Fortran ranges exactly).
                flux_cap[t]["cx"][:, :, k] *= frac
                xfx6[t][k].a[:, :] *= frac
                flux_cap[t]["mfx"][:, 0:je - js + 1, k] *= frac
                flux_cap[t]["cy"][:, :, k] *= frac
                yfx6[t][k].a[:, :] *= frac
                flux_cap[t]["mfy"][0:ie - is_ + 1, :, k] *= frac

    # ---- q halo update: ext_scalar(q, ..., 0, 0) (:276-281) ------------
    # The Fortran call exchanges the whole 4-D array at once; the
    # operator is horizontal, so per-(k, iq) six-face slabs are the same
    # exchange.
    for k in range(km):
        for iq in range(nq):
            f6 = [q6[t][iq][:, :, k] for t in range(6)]
            _q_halo_exchange_sixface(ctx, f6)
            for t in range(6):
                q6[t][iq][:, :, k] = f6[t]

    # ---- the k loop (:284-388) ------------------------------------------
    for k in range(km):
        nsplt = nsplt_k[k]

        # ra_x/ra_y (:286-296)
        ra_x6, ra_y6 = [], []
        for t in range(6):
            AREA = fort(ctx["gs6"][t]["area"], isd, jsd)
            xfx, yfx = xfx6[t][k], yfx6[t][k]
            ra_x = _fl(is_, ie, jsd, jed)
            ra_y = _fl(isd, ied, js, je)
            for j in range(jsd, jed + 1):
                for i in range(is_, ie + 1):
                    ra_x[i, j] = AREA[i, j] + xfx[i, j] - xfx[i + 1, j]
            for j in range(js, je + 1):
                for i in range(isd, ied + 1):
                    ra_y[i, j] = AREA[i, j] + yfx[i, j] - yfx[i, j + 1]
            ra_x6.append(ra_x)
            ra_y6.append(ra_y)

        def _dp2_face(t: int) -> object:
            """dp2 = dp1 + div(mfx, mfy)*rarea (:302-306 / :341-345)."""
            RAREA = fort(ctx["gs6"][t]["rarea"], isd, jsd)
            dp1f = fort(dp1_6[t][:, :, k], isd, jsd)
            mfxf = fort(flux_cap[t]["mfx"][:, :, k], is_, js)
            mfyf = fort(flux_cap[t]["mfy"][:, :, k], is_, js)
            dp2 = _fl(is_, ie, js, je)
            for j in range(js, je + 1):
                for i in range(is_, ie + 1):
                    dp2[i, j] = dp1f[i, j] + (
                        mfxf[i, j] - mfxf[i + 1, j]
                        + mfyf[i, j] - mfyf[i, j + 1]) * RAREA[i, j]
            return dp2

        def _tp_and_adj(src6: list) -> tuple:
            """fv_tp_2d on every face, then flux_adj across the seams.

            The oracle's flux_adj (mpp_get_boundary CGRID_NE + 0.5
            blend, :85-107) runs per face against its neighbours; the
            six-face analog needs all faces' fluxes present, so the
            per-face transport is completed first -- the same
            gather-then-apply the certified barrier-1 helper does.
            """
            fx6, fy6 = [], []
            for t in range(6):
                qf = fort(src6[t], isd, jsd)
                cxf = fort(flux_cap[t]["cx"][:, :, k], is_, jsd)
                cyf = fort(flux_cap[t]["cy"][:, :, k], isd, js)
                mfxf = fort(flux_cap[t]["mfx"][:, :, k], is_, js)
                mfyf = fort(flux_cap[t]["mfy"][:, :, k], is_, js)
                fx = _fl(is_, ie + 1, js, je)
                fy = _fl(is_, ie, js, je + 1)
                fv_tp_2d(qf, cxf, cyf, npx, npx, hord_tr, fx, fy,
                         xfx6[t][k], yfx6[t][k], gsf6[t], bd,
                         ra_x6[t], ra_y6[t], lim_fac,
                         mfx=mfxf, mfy=mfyf, duogrid=True)
                fx6.append(fx)
                fy6.append(fy)
            # flux_adj (:310-312): fx (n+1, n) / fy (n, n+1) at origin
            # (1, 1) -- exactly the _fl arrays' raw layout.
            average_shared_edge_cgrid([f.a for f in fx6],
                                      [f.a for f in fy6], n, ng)
            return fx6, fy6

        if nsplt == 1:
            dp2_6 = [_dp2_face(t) for t in range(6)]
            for iq in range(nq):
                fx6, fy6 = _tp_and_adj([q6[t][iq][:, :, k]
                                        for t in range(6)])
                for t in range(6):
                    RAREA = fort(ctx["gs6"][t]["rarea"], isd, jsd)
                    dp1f = fort(dp1_6[t][:, :, k], isd, jsd)
                    qf = fort(q6[t][iq][:, :, k], isd, jsd)
                    fx, fy = fx6[t], fy6[t]
                    dp2 = dp2_6[t]
                    for j in range(js, je + 1):
                        for i in range(is_, ie + 1):
                            qf[i, j] = (qf[i, j] * dp1f[i, j]
                                        + (fx[i, j] - fx[i + 1, j]
                                           + fy[i, j] - fy[i, j + 1])
                                        * RAREA[i, j]) / dp2[i, j]
        else:
            # :332-388.  qn2 holds the full padded slab per tracer;
            # copied at it == 1 (:348-354 copies isd..ied, jsd..jed).
            qn2 = [[np.array(q6[t][iq][:, :, k], copy=True)
                    for iq in range(nq)] for t in range(6)]
            for it in range(1, nsplt + 1):
                dp2_6 = [_dp2_face(t) for t in range(6)]
                for iq in range(nq):
                    fx6, fy6 = _tp_and_adj([qn2[t][iq] for t in range(6)])
                    for t in range(6):
                        RAREA = fort(ctx["gs6"][t]["rarea"], isd, jsd)
                        dp1f = fort(dp1_6[t][:, :, k], isd, jsd)
                        qnf = fort(qn2[t][iq], isd, jsd)
                        fx, fy = fx6[t], fy6[t]
                        dp2 = dp2_6[t]
                        if it < nsplt:
                            dst = qnf
                        else:
                            dst = fort(q6[t][iq][:, :, k], isd, jsd)
                        for j in range(js, je + 1):
                            for i in range(is_, ie + 1):
                                dst[i, j] = (qnf[i, j] * dp1f[i, j]
                                             + (fx[i, j] - fx[i + 1, j]
                                                + fy[i, j] - fy[i, j + 1])
                                             * RAREA[i, j]) / dp2[i, j]
                if it < nsplt:
                    # :361-365 dp1 = dp2 (compute window), then the qn2
                    # halo refresh (:367-379, the duo ext_scalar arm).
                    for t in range(6):
                        dp1_6[t][ng:ng + n, ng:ng + n, k] = dp2_6[t].a
                    for iq in range(nq):
                        f6 = [qn2[t][iq] for t in range(6)]
                        _q_halo_exchange_sixface(ctx, f6)
                        for t in range(6):
                            qn2[t][iq][:, :] = f6[t]
