"""FV3 non-hydrostatic core — the LIVE routines on the pinned deck.

Loop-faithful NumPy ports from the pinned oracle
(``/burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean``):

  ============  =============================  ==========================
  here          oracle                         lines
  ============  =============================  ==========================
  update_dz_c   nh_utils.F90 update_dz_c       :49-191
  riem_solver_c nh_utils.F90 Riem_Solver_c     :313-420
  sim1_solver   nh_utils.F90 SIM1_solver       :1193-1324
  ============  =============================  ==========================

WHY ONLY THESE THREE (the live/dead partition, deck-resolved):

* ``a_imp = 1.`` on the NH deck (``run_c48_nh13`` resolved namelist echo;
  the deck file sets several keys twice, so the echo is the authority).
  ``Riem_Solver3``'s dispatch (``nh_core.F90:138-154``) sends
  ``a_imp > 0.999`` to **SIM1_solver**; SIM3p0/SIM3/RIM_2D/SIM_solver are
  all dead.  ``Riem_Solver_c``'s dispatch (``nh_utils.F90:392-401``) has
  only three arms and likewise lands in SIM1_solver.
* Build defines: the oracle build uses
  ``-DSPMD -Duse_libMPI -Duse_netCDF -DINTERNAL_FILE_NML`` — so
  ``USE_COND``, ``MOIST_CAPPA`` and ``DZ_MIN_6`` are ALL undefined.  The
  non-moist branches below are the compiled ones, and
  ``dz_min = 2.`` (``nh_utils.F90:43``), not 6.

CONVENTIONS (each verified in source, not assumed — see the NH spec's
section 5 and the oracle lines cited there):

* ``delz < 0``, ``delp > 0``; ``dz2(i,k) = gz(k+1) - gz(k)`` with gz
  decreasing upward in k-index (k=1 is the TOP).
* ``gz`` enters ``riem_solver_c`` as height*grav-like geopotential and
  leaves rebuilt from ``hs`` upward: ``gz(k) = gz(k+1) - dz2*grav``
  (``nh_utils.F90:407-415``).
* ``ws`` is DIAGNOSED, not imposed: ``ws = (zs - gz_bottom) * rdt``
  (``nh_utils.F90:173-178``).
* ``pef`` is FULL interface pressure on the C stage: ``pe2 + pem``
  (``nh_utils.F90:404``); the D stage's ``Riem_Solver3`` writes the
  PERTURBATION instead — do not conflate them when porting unit 6.

Arrays are Fortran-indexed via ``fort`` views exactly like the certified
2-D stage kernels; loops are translated literally, vectorised only over
the innermost ``i`` where the Fortran loop is itself independent in ``i``.
The tridiagonal solves in ``sim1_solver`` carry sequential k-dependencies
and stay as explicit k loops (vector in i).
"""
from __future__ import annotations

import numpy as np

from legoesm.grids.fv3_native_gridstruct import (
    FV3_GRAV,
    FV3_RDGAS,
    fort,
)

# nh_utils.F90:40-44 -- DZ_MIN_6 is not in either build's DEFS.
DZ_MIN = 2.0
# nh_utils.F90:45
_R3 = 1.0 / 3.0


def update_dz_c(bd, km: int, dt: float, dp0: np.ndarray,
                zs: fort, area: fort, ut: np.ndarray, vt: np.ndarray,
                gz: np.ndarray, ws: fort, npx: int, npy: int, *,
                sw_corner: bool, se_corner: bool,
                ne_corner: bool, nw_corner: bool,
                grid_type: int = 0) -> None:
    """``nh_utils.F90:49-191`` verbatim.

    ``ut/vt/gz`` are raw padded arrays with axes (i, j, k), i/j starting
    at ``is-ng`` — the caller passes the SAME layout the 3-D phases use.
    ``gz`` (km+1 interfaces) and ``ws`` are updated in place through the
    fort views built here.  ``dp0`` is the 1-based reference thickness
    (``dp_ref``), passed as a plain 1-D array of length km.

    The two flux windows are exactly the oracle's:
    x: ``(is-1:ie+2, js-1:je+1)``; y: ``(is-1:ie+1, js-1:je+2)``.
    """
    from legoesm.core.fv3_native_sw_core import fill_4corners

    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng
    if dp0.shape != (km,):
        raise ValueError(f"update_dz_c: dp0 must be ({km},), got {dp0.shape}")
    if gz.shape[2] != km + 1:
        raise ValueError(f"update_dz_c: gz needs {km + 1} interfaces, "
                         f"got {gz.shape[2]}")

    rdt = 1.0 / dt
    top_ratio = dp0[0] / (dp0[0] + dp0[1])            # :75  (dp0 is 0-based)
    bot_ratio = dp0[km - 1] / (dp0[km - 2] + dp0[km - 1])   # :76

    is1, js1 = is_ - 1, js - 1
    ie1, je1 = ie + 1, je + 1
    ie2, je2 = ie + 2, je + 2
    # SEPARATE array origins (codex NH r1 #1: reusing the i origin for j
    # is silently wrong the moment a caller has is != js).
    ilo = is_ - ng
    jlo = js - ng

    def _f2(a2d):
        return fort(a2d, ilo, jlo)

    gzf = fort(gz, ilo, jlo)          # 3-D view; k passed 0-based below

    for k1 in range(1, km + 2):       # do 6000 k=1,km+1
        k = k1 - 1                    # 0-based level index into ut/vt/gz
        xfx = np.zeros((ie2 - is1 + 1, je1 - js1 + 1))
        yfx = np.zeros((ie1 - is1 + 1, je2 - js1 + 1))
        xf = fort(xfx, is1, js1)
        yf = fort(yfx, is1, js1)
        utf = fort(ut, ilo, jlo)
        vtf = fort(vt, ilo, jlo)

        if k1 == 1:                                            # :94-106
            for j in range(js1, je1 + 1):
                for i in range(is1, ie2 + 1):
                    xf[i, j] = utf[i, j, 0] + (utf[i, j, 0]
                                               - utf[i, j, 1]) * top_ratio
            for j in range(js1, je2 + 1):
                for i in range(is1, ie1 + 1):
                    yf[i, j] = vtf[i, j, 0] + (vtf[i, j, 0]
                                               - vtf[i, j, 1]) * top_ratio
        elif k1 == km + 1:                                     # :107-121
            for j in range(js1, je1 + 1):
                for i in range(is1, ie2 + 1):
                    xf[i, j] = utf[i, j, km - 1] + (
                        utf[i, j, km - 1] - utf[i, j, km - 2]) * bot_ratio
            for j in range(js1, je2 + 1):
                for i in range(is1, ie1 + 1):
                    yf[i, j] = vtf[i, j, km - 1] + (
                        vtf[i, j, km - 1] - vtf[i, j, km - 2]) * bot_ratio
        else:                                                  # :122-133
            int_ratio = 1.0 / (dp0[k - 1] + dp0[k])
            for j in range(js1, je1 + 1):
                for i in range(is1, ie2 + 1):
                    xf[i, j] = (dp0[k] * utf[i, j, k - 1]
                                + dp0[k - 1] * utf[i, j, k]) * int_ratio
            for j in range(js1, je2 + 1):
                for i in range(is1, ie1 + 1):
                    yf[i, j] = (dp0[k] * vtf[i, j, k - 1]
                                + dp0[k - 1] * vtf[i, j, k]) * int_ratio

        gz2 = np.array(gz[:, :, k], copy=True)                 # :136-140
        g2 = _f2(gz2)

        if grid_type < 3:                                      # :142
            fill_4corners(g2, 1, npx, npy, sw=sw_corner, se=se_corner,
                          ne=ne_corner, nw=nw_corner)
        fx = np.zeros_like(xfx)
        fxf = fort(fx, is1, js1)
        for j in range(js1, je1 + 1):                          # :143-152
            for i in range(is1, ie2 + 1):
                v = g2[i - 1, j] if xf[i, j] > 0.0 else g2[i, j]
                fxf[i, j] = xf[i, j] * v

        if grid_type < 3:                                      # :154
            fill_4corners(g2, 2, npx, npy, sw=sw_corner, se=se_corner,
                          ne=ne_corner, nw=nw_corner)
        fy = np.zeros_like(yfx)
        fyf = fort(fy, is1, js1)
        for j in range(js1, je2 + 1):                          # :155-164
            for i in range(is1, ie1 + 1):
                v = g2[i, j - 1] if yf[i, j] > 0.0 else g2[i, j]
                fyf[i, j] = yf[i, j] * v

        af = fort(area.a if isinstance(area, fort) else area, ilo, jlo)
        for j in range(js1, je1 + 1):                          # :166-171
            for i in range(is1, ie1 + 1):
                gzf[i, j, k] = (
                    (g2[i, j] * af[i, j] + fxf[i, j] - fxf[i + 1, j]
                     + fyf[i, j] - fyf[i, j + 1])
                    / (af[i, j] + xf[i, j] - xf[i + 1, j]
                       + yf[i, j] - yf[i, j + 1]))

    # :173-189  ws diagnosis + monotone-height limiter (dz_min = 2.)
    zsf = fort(zs.a if isinstance(zs, fort) else zs, ilo, jlo)
    wsf = fort(ws.a if isinstance(ws, fort) else ws, ilo, jlo)
    for j in range(js1, je1 + 1):
        for i in range(is1, ie1 + 1):
            wsf[i, j] = (zsf[i, j] - gzf[i, j, km]) * rdt
        for k in range(km - 1, -1, -1):        # do k=km,1,-1 (0-based)
            for i in range(is1, ie1 + 1):
                gzf[i, j, k] = max(gzf[i, j, k], gzf[i, j, k + 1] + DZ_MIN)


def edge_profile(q1: np.ndarray, q2: np.ndarray, j_lo: int, km: int,
                 dp0: np.ndarray, uniform_grid: bool,
                 limiter: int) -> tuple[np.ndarray, np.ndarray]:
    """``nh_utils.F90:1535-1641`` verbatim -- one j-row at a time is the
    Fortran shape, but every i is independent, so this port takes the
    row PAIR ``q1, q2`` as (ni, km) arrays for one j and returns the
    (ni, km+1) edge profiles.  ``j_lo`` is unused arithmetic-wise and
    kept for call-site readability against the oracle.

    Only the ``uniform_grid = .false.`` branch is exercised on the duo
    lane (``update_dz_d`` hardcodes ``uniform_grid = .false.`` at
    nh_utils.F90:229); the uniform branch is ported anyway because it is
    ten lines and a deck flag away.

    ``limiter = 0`` on the update_dz_d call (:241-246); the nonzero
    branch (zero-crossing clamp of top/bottom winds) is ported for the
    same reason.
    """
    del j_lo
    ni = q1.shape[0]
    qe1 = np.zeros((ni, km + 1))
    qe2 = np.zeros((ni, km + 1))

    if uniform_grid:                                      # :1552-1581
        r2o3 = 2.0 / 3.0
        r4o3 = 4.0 / 3.0
        gak = np.zeros(km)
        qe1[:, 0] = r4o3 * q1[:, 0] + r2o3 * q1[:, 1]
        qe2[:, 0] = r4o3 * q2[:, 0] + r2o3 * q2[:, 1]
        gak[0] = 7.0 / 3.0
        for k in range(1, km):
            gak[k] = 1.0 / (4.0 - gak[k - 1])
            qe1[:, k] = (3.0 * (q1[:, k - 1] + q1[:, k])
                         - qe1[:, k - 1]) * gak[k]
            qe2[:, k] = (3.0 * (q2[:, k - 1] + q2[:, k])
                         - qe2[:, k - 1]) * gak[k]
        bet = 1.0 / (1.5 - 3.5 * gak[km - 1])
        qe1[:, km] = (4.0 * q1[:, km - 1] + q1[:, km - 2]
                      - 3.5 * qe1[:, km - 1]) * bet
        qe2[:, km] = (4.0 * q2[:, km - 1] + q2[:, km - 2]
                      - 3.5 * qe2[:, km - 1]) * bet
        for k in range(km - 1, -1, -1):
            qe1[:, k] = qe1[:, k] - gak[k] * qe1[:, k + 1]
            qe2[:, k] = qe2[:, k] - gak[k] * qe2[:, k + 1]
    else:                                                 # :1583-1618
        gam = np.zeros((ni, km))
        g0 = dp0[1] / dp0[0]
        xt1 = 2.0 * g0 * (g0 + 1.0)
        bet = g0 * (g0 + 0.5)
        qe1[:, 0] = (xt1 * q1[:, 0] + q1[:, 1]) / bet
        qe2[:, 0] = (xt1 * q2[:, 0] + q2[:, 1]) / bet
        gam[:, 0] = (1.0 + g0 * (g0 + 1.5)) / bet
        gk = 0.0
        for k in range(1, km):
            gk = dp0[k - 1] / dp0[k]
            bet_v = 2.0 + 2.0 * gk - gam[:, k - 1]
            qe1[:, k] = (3.0 * (q1[:, k - 1] + gk * q1[:, k])
                         - qe1[:, k - 1]) / bet_v
            qe2[:, k] = (3.0 * (q2[:, k - 1] + gk * q2[:, k])
                         - qe2[:, k - 1]) / bet_v
            gam[:, k] = gk / bet_v
        # :1602-1609 -- gk here is the LAST loop value (Fortran leaves the
        # do-variable's final value in scope; a fresh dp0(km)/dp0(km-1)
        # would be the same number, but the reuse is the literal source).
        a_bot = 1.0 + gk * (gk + 1.5)
        xt1 = 2.0 * gk * (gk + 1.0)
        xt2 = gk * (gk + 0.5) - a_bot * gam[:, km - 1]
        qe1[:, km] = (xt1 * q1[:, km - 1] + q1[:, km - 2]
                      - a_bot * qe1[:, km - 1]) / xt2
        qe2[:, km] = (xt1 * q2[:, km - 1] + q2[:, km - 2]
                      - a_bot * qe2[:, km - 1]) / xt2
        for k in range(km - 1, -1, -1):
            qe1[:, k] = qe1[:, k] - gam[:, k] * qe1[:, k + 1]
            qe2[:, k] = qe2[:, k] - gam[:, k] * qe2[:, k + 1]

    if limiter != 0:                                      # :1623-1633
        qe1[:, 0] = np.where(q1[:, 0] * qe1[:, 0] < 0.0, 0.0, qe1[:, 0])
        qe2[:, 0] = np.where(q2[:, 0] * qe2[:, 0] < 0.0, 0.0, qe2[:, 0])
        qe1[:, km] = np.where(q1[:, km - 1] * qe1[:, km] < 0.0,
                              0.0, qe1[:, km])
        qe2[:, km] = np.where(q2[:, km - 1] * qe2[:, km] < 0.0,
                              0.0, qe2[:, km])
    return qe1, qe2


def update_dz_d(ndif: np.ndarray, damp: np.ndarray, hord: int, bd, km: int,
                npx: int, npy: int, area: np.ndarray, rarea: np.ndarray,
                dp0: np.ndarray, zs: np.ndarray, zh: np.ndarray,
                crx: np.ndarray, cry: np.ndarray, xfx: np.ndarray,
                yfx: np.ndarray, ws: np.ndarray, rdt: float,
                gridstruct: dict, lim_fac: float = 1.0) -> None:
    """``nh_utils.F90:194-311`` verbatim.

    Reuses the CERTIFIED transport primitives -- ``fv_tp_2d`` and
    ``del6_vt_flux`` from ``fv3_native_d_sw`` -- exactly as the oracle
    reuses its own; no new horizontal numerics are written here.

    PRECONDITION on ``gridstruct['area']`` (measured, probe job
    9355001): the CORNER-DIAGONAL halo cells must hold REAL areas, as
    the oracle's mpp-exchanged ``gridstruct%area`` does.  The single-
    tile builder leaves BIG_NUMBER sentinels there, and fv_tp_2d's
    y-intermediates read them: with sentinels, the flux of a CONSTANT
    field deviates from ``xfx*q`` by 35% near corners; with real areas
    the deviation is pure rounding (1.5e-05 of 1.2e11).  Supplying
    corner-diagonal areas is owned by the six-face NH integration
    (port-plan unit 7), not by this routine.

    Layouts (Fortran dummies -> here):
      zh   (isd:ied, jsd:jed, km+1)    padded, in place
      crx/xfx (is:ie+1, jsd:jed, km)   x-edge, NOT padded in i
      cry/yfx (isd:ied, js:je+1, km)   y-edge, NOT padded in j
      ws   (is:ie, js:je)              compute-window output
      damp (km+1), ndif (km+1)         MUTATED: level km+1 copies km
                                       (:231-232) -- the caller's arrays
                                       must arrive with that slot free.

    ``uniform_grid = .false.`` (:229) and the edge_profile limiter is 0.
    """
    from legoesm.core.fv3_native_d_sw import del6_vt_flux, fv_tp_2d
    from legoesm.core.fv3_native_sw_core import Bounds

    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng
    isd, ied = is_ - ng, ie + ng
    jsd, jed = js - ng, je + ng

    if damp.shape != (km + 1,) or ndif.shape != (km + 1,):
        raise ValueError("update_dz_d: damp/ndif must have km+1 slots "
                         f"(got {damp.shape}, {ndif.shape})")
    damp[km] = damp[km - 1]                               # :231
    ndif[km] = ndif[km - 1]                               # :232

    # fv_tp_2d / del6_vt_flux dereference the gridstruct dict with
    # FORTRAN indices (AREA[i, j] at i = isd < 0).  The certified d_sw
    # caller wraps every 2-D metric in a fort view first
    # (fv3_native_d_sw.py:2862-2913); passing a raw numpy array instead
    # makes negative indices WRAP and positive ones sit ng cells off --
    # measured (probe job 9355103): q_i off by 24% of a constant field,
    # zh invariance broken at 3e-6 relative.  Mirror the certified
    # recipe for exactly the keys the two callees read here.
    def _wrap2(name):
        v = gridstruct[name]
        return v if isinstance(v, fort) else fort(np.asarray(v), isd, jsd)

    gsf = dict(gridstruct)
    for _k2 in ("area", "rarea", "dxa", "dya", "del6_u", "del6_v"):
        if _k2 in gridstruct:
            gsf[_k2] = _wrap2(_k2)
    gsf["area"] = (area if isinstance(area, fort)
                   else fort(np.asarray(area), isd, jsd))
    gsf["rarea"] = (rarea if isinstance(rarea, fort)
                    else fort(np.asarray(rarea), isd, jsd))
    gridstruct = gsf

    ni_x = ie + 1 - is_ + 1          # crx i-extent  (is..ie+1)
    nj_y = je + 1 - js + 1           # cry j-extent  (js..je+1)
    nid = ied - isd + 1
    njd = jed - jsd + 1

    # :238-246  vertical edge reconstruction, row by row as upstream
    crx_adv = np.zeros((ni_x, njd, km + 1))
    xfx_adv = np.zeros((ni_x, njd, km + 1))
    cry_adv = np.zeros((nid, nj_y, km + 1))
    yfx_adv = np.zeros((nid, nj_y, km + 1))
    for jj in range(njd):                     # j = jsd..jed
        crx_adv[:, jj, :], xfx_adv[:, jj, :] = edge_profile(
            crx[:, jj, :], xfx[:, jj, :], jsd + jj, km, dp0, False, 0)
        j_f = jsd + jj
        if js <= j_f <= je + 1:
            j2 = j_f - js
            cry_adv[:, j2, :], yfx_adv[:, j2, :] = edge_profile(
                cry[:, j2, :], yfx[:, j2, :], j_f, km, dp0, False, 0)

    afort = fort(area, isd, jsd)
    rfort = fort(rarea, isd, jsd)
    zhf = fort(zh, isd, jsd)
    bd2 = bd if isinstance(bd, Bounds) else Bounds.single_tile(
        ie - is_ + 1, ng)

    for k in range(km + 1):                   # :254-291
        ra_x = np.zeros((ie - is_ + 1, njd))
        rax = fort(ra_x, is_, jsd)
        for jj in range(njd):
            j = jsd + jj
            for i in range(is_, ie + 1):
                rax[i, j] = (afort[i, j] + xfx_adv[i - is_, jj, k]
                             - xfx_adv[i - is_ + 1, jj, k])
        ra_y = np.zeros((nid, je - js + 1))
        ray = fort(ra_y, isd, js)
        for j in range(js, je + 1):
            for i in range(isd, ied + 1):
                ray[i, j] = (afort[i, j] + yfx_adv[i - isd, j - js, k]
                             - yfx_adv[i - isd, j - js + 1, k])

        crx_k = fort(np.ascontiguousarray(crx_adv[:, :, k]), is_, jsd)
        xfx_k = fort(np.ascontiguousarray(xfx_adv[:, :, k]), is_, jsd)
        cry_k = fort(np.ascontiguousarray(cry_adv[:, :, k]), isd, js)
        yfx_k = fort(np.ascontiguousarray(yfx_adv[:, :, k]), isd, js)
        fx = fort(np.zeros((ni_x, je - js + 1)), is_, js)
        fy = fort(np.zeros((ie - is_ + 1, nj_y)), is_, js)

        if damp[k] > 1.0e-5:                              # :266-283
            z2 = np.array(zh[:, :, k], copy=True)
            z2f = fort(z2, isd, jsd)
            fv_tp_2d(z2f, crx_k, cry_k, npx, npy, hord, fx, fy,
                     xfx_k, yfx_k, gridstruct, bd2, rax, ray, lim_fac)
            wk2 = fort(np.zeros((nid, njd)), isd, jsd)
            fx2 = fort(np.zeros((nid + 1, njd)), isd, jsd)
            fy2 = fort(np.zeros((nid, njd + 1)), isd, jsd)
            del6_vt_flux(int(ndif[k]), npx, npy, float(damp[k]), z2f,
                         wk2, fx2, fy2, gridstruct, bd2)
            for j in range(js, je + 1):
                for i in range(is_, ie + 1):
                    zhf[i, j, k] = (
                        (z2f[i, j] * afort[i, j] + fx[i, j] - fx[i + 1, j]
                         + fy[i, j] - fy[i, j + 1])
                        / (rax[i, j] + ray[i, j] - afort[i, j])
                        + (fx2[i, j] - fx2[i + 1, j] + fy2[i, j]
                           - fy2[i, j + 1]) * rfort[i, j])
        else:                                             # :284-289
            # The oracle passes zh ITSELF here (zh(isd,jsd,k)), so
            # fv_tp_2d's copy_corners mutates zh's ghost corners in
            # place -- only the damp branch works on a copy.  A NumPy
            # slice view preserves exactly that aliasing; the per-cell
            # update below reads only its own (i,j), as upstream.
            z2f = fort(zh[:, :, k], isd, jsd)
            fv_tp_2d(z2f, crx_k, cry_k, npx, npy, hord, fx, fy,
                     xfx_k, yfx_k, gridstruct, bd2, rax, ray, lim_fac)
            for j in range(js, je + 1):
                for i in range(is_, ie + 1):
                    zhf[i, j, k] = (
                        (z2f[i, j] * afort[i, j] + fx[i, j] - fx[i + 1, j]
                         + fy[i, j] - fy[i, j + 1])
                        / (rax[i, j] + ray[i, j] - afort[i, j]))

    # :294-309  ws diagnosis + monotone limiter (ws BEFORE the limiter)
    wsf = fort(ws, is_, js)
    zsf = fort(zs, isd, jsd)
    for j in range(js, je + 1):
        for i in range(is_, ie + 1):
            wsf[i, j] = (zsf[i, j] - zhf[i, j, km]) * rdt
        for k in range(km - 1, -1, -1):
            for i in range(is_, ie + 1):
                zhf[i, j, k] = max(zhf[i, j, k], zhf[i, j, k + 1] + DZ_MIN)


def sim1_solver(dt: float, is_: int, ie: int, km: int, rgas: float,
                gama: float, kappa: float, pe: np.ndarray,
                dm2: np.ndarray, pm2: np.ndarray, pem: np.ndarray,
                w2: np.ndarray, dz2: np.ndarray, pt2: np.ndarray,
                ws: np.ndarray, p_fac: float) -> None:
    """``nh_utils.F90:1193-1324`` verbatim, non-MOIST_CAPPA branches.

    All arrays are PLAIN 0-based NumPy with i as axis 0 (length
    ``ie-is_+1``) and k as axis 1 (km or km+1): the (i,k) work arrays of
    the oracle, whose i window the CALLER has already selected.  ``pe``
    (km+1) is output; ``w2`` and ``dz2`` are updated in place.  ``gm2``
    and ``cp2`` of the Fortran signature are MOIST_CAPPA-only reads and
    are deliberately not parameters.

    Vectorised over i everywhere (every Fortran i-loop is independent);
    k-recurrences stay explicit.
    """
    for name, a in (("pe", pe), ("w2", w2), ("dz2", dz2), ("dm2", dm2),
                    ("pm2", pm2), ("pem", pem), ("pt2", pt2)):
        if np.asarray(a).dtype != np.float64:
            raise TypeError(
                f"sim1_solver: {name} must be float64 (got "
                f"{np.asarray(a).dtype}); a float32 column silently "
                f"truncates the solve by ~1e-3 in dz2 (codex NH r1 #4)")
    t1g = gama * 2.0 * dt * dt          # :1211 (non-moist)
    rdt = 1.0 / dt
    capa1 = kappa - 1.0

    w1 = np.array(w2, copy=True)        # :1224 w1 = w2
    # :1219-1226  pe = (-dm/dz * rgas * pt)^gama - pm
    pe[:, :km] = np.exp(
        gama * np.log(-dm2 / dz2 * rgas * pt2)) - pm2
    # (pe's km+1-th column is set below; the Fortran fills pe(:,1:km) here)

    g_rat = np.zeros((dm2.shape[0], km))
    bb = np.zeros_like(g_rat)
    dd = np.zeros_like(g_rat)
    for k in range(km - 1):             # :1228-1234  k=1,km-1
        g_rat[:, k] = dm2[:, k] / dm2[:, k + 1]
        bb[:, k] = 2.0 * (1.0 + g_rat[:, k])
        dd[:, k] = 3.0 * (pe[:, k] + g_rat[:, k] * pe[:, k + 1])

    pp = np.zeros((dm2.shape[0], km + 1))
    gam = np.zeros_like(g_rat)
    bet = np.array(bb[:, 0], copy=True)             # :1237
    pp[:, 0] = 0.0
    pp[:, 1] = dd[:, 0] / bet
    bb[:, km - 1] = 2.0                             # :1240
    dd[:, km - 1] = 3.0 * pe[:, km - 1]

    for k in range(1, km):              # :1244-1250  k=2,km
        gam[:, k] = g_rat[:, k - 1] / bet
        bet = bb[:, k] - gam[:, k]
        pp[:, k + 1] = (dd[:, k] - pp[:, k]) / bet

    for k in range(km - 1, 0, -1):      # :1252-1256  k=km,2,-1
        pp[:, k] = pp[:, k] - gam[:, k] * pp[:, k + 1]

    # w solver :1259-1291
    aa = np.zeros_like(pp[:, :km])
    for k in range(1, km):              # k=2,km
        aa[:, k] = (t1g / (dz2[:, k - 1] + dz2[:, k])
                    * (pem[:, k] + pp[:, k]))
    bet = dm2[:, 0] - aa[:, 1]
    w2[:, 0] = (dm2[:, 0] * w1[:, 0] + dt * pp[:, 1]) / bet
    for k in range(1, km - 1):          # k=2,km-1
        gam[:, k] = aa[:, k] / bet
        bet = dm2[:, k] - (aa[:, k] + aa[:, k + 1] + aa[:, k] * gam[:, k])
        w2[:, k] = (dm2[:, k] * w1[:, k]
                    + dt * (pp[:, k + 1] - pp[:, k])
                    - aa[:, k] * w2[:, k - 1]) / bet
    p1 = t1g / dz2[:, km - 1] * (pem[:, km] + pp[:, km])       # :1282
    gam_km = aa[:, km - 1] / bet
    bet = dm2[:, km - 1] - (aa[:, km - 1] + p1 + aa[:, km - 1] * gam_km)
    w2[:, km - 1] = (dm2[:, km - 1] * w1[:, km - 1]
                     + dt * (pp[:, km] - pp[:, km - 1])
                     - p1 * ws - aa[:, km - 1] * w2[:, km - 2]) / bet
    gam[:, km - 1] = gam_km
    for k in range(km - 2, -1, -1):     # :1288-1291 k=km-1,1,-1
        w2[:, k] = w2[:, k] - gam[:, k + 1] * w2[:, k + 1]

    # :1293-1300  pe rebuild as the time-tendency integral
    pe[:, 0] = 0.0
    for k in range(km):
        pe[:, k + 1] = pe[:, k] + dm2[:, k] * (w2[:, k] - w1[:, k]) * rdt

    # :1302-1319  back-out dz2 (bottom-up), the p_fac floor is here
    p1 = (pe[:, km - 1] + 2.0 * pe[:, km]) * _R3
    dz2[:, km - 1] = -dm2[:, km - 1] * rgas * pt2[:, km - 1] * np.exp(
        capa1 * np.log(np.maximum(p_fac * pm2[:, km - 1],
                                  p1 + pm2[:, km - 1])))
    for k in range(km - 2, -1, -1):     # k=km-1,1,-1
        p1 = (pe[:, k] + bb[:, k] * pe[:, k + 1]
              + g_rat[:, k] * pe[:, k + 2]) * _R3 - g_rat[:, k] * p1
        dz2[:, k] = -dm2[:, k] * rgas * pt2[:, k] * np.exp(
            capa1 * np.log(np.maximum(p_fac * pm2[:, k], p1 + pm2[:, k])))


def riem_solver3(ms: int, dt: float, bd, km: int, akap: float, cp: float,
                 ptop: float, zs: np.ndarray, w: np.ndarray,
                 delz: np.ndarray, pt: np.ndarray, delp: np.ndarray,
                 zh: np.ndarray, pe: np.ndarray, ppe: np.ndarray,
                 pk3: np.ndarray, pk: np.ndarray, peln: np.ndarray,
                 ws: np.ndarray, p_fac: float, a_imp: float, *,
                 scale_m: float = 0.0, use_logp: bool = False,
                 last_call: bool = True, fp_out: bool = False) -> None:
    """``nh_core.F90:42-206`` verbatim, non-USE_COND / non-MOIST_CAPPA.

    The D-stage Riemann driver.  Differences from :func:`riem_solver_c`
    that matter and are easy to conflate (each read from source):

    * the j window is ``js..je`` (:87), NOT ``js-1..je+1``;
    * the i window is ``is..ie`` -- no ring;
    * ``ppe`` is the PERTURBATION ``pe2`` when ``fp_out`` is false
      (:173-185); on the pinned deck ``beta = 0`` so
      ``fp_out = (beta < -0.1)`` is FALSE and the perturbation is what
      downstream ``nh_p_grad`` receives -- the C stage's full-pressure
      convention does not apply here;
    * ``dz2`` comes from ``zh`` (:96), and ``zh`` is rebuilt from
      ``zs`` WITHOUT a grav factor (:195-202): zh is HEIGHT, where the
      C stage's gz is geopotential;
    * ``pm2`` divides by ``peln2`` differences (:104), not by
      ``log(pem_ratio)`` directly -- same value, DIFFERENT rounding;
      both are kept literal;
    * ``pk3(:, :, 1) = ptk = exp(akap*log(ptop))`` and interior levels
      ``exp(akap*peln2)`` (:80, :91); with ``use_logp`` false the
      log-overwrite at :187-193 is dead on this deck;
    * ``last_call`` copies peln/pk/pe out in the oracle's (i,k,j)
      layouts for pe/peln -- the CALLER owns those transposed arrays
      exactly as ``fv3_native_state_3d.field_shape`` declares them.

    Layouts here: padded (i,j,k) for zs/w/pt/delp/zh/ppe/pk3;
    ``pe``/``peln`` in the oracle's (i,k,j) with their own windows
    (pe: is-1..ie+1; peln: is..ie); ``delz``/``pk``/``ws`` compute-window
    arrays with axes (i,j,k)/(i,j).
    """
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng
    ilo = is_ - ng
    jlo = js - ng
    gama = 1.0 / (1.0 - akap)
    rgrav = 1.0 / FV3_GRAV
    peln1 = float(np.log(ptop))
    ptk = float(np.exp(akap * peln1))

    if not (a_imp > 0.999):
        raise NotImplementedError(
            f"riem_solver3: a_imp={a_imp} selects a dead arm on the "
            f"pinned deck (a_imp=1. -> SIM1); nh_core.F90:138-154.")

    ni = ie - is_ + 1
    o = is_ - ilo                   # offset of i=is in the padded axis
    for j in range(js, je + 1):
        jj = j - jlo
        jc = j - js                 # compute-window j (delz, pk, ws)
        dm = np.array(delp[o:o + ni, jj, :km], dtype=np.float64)

        pem = np.zeros((ni, km + 1))
        peln2 = np.zeros((ni, km + 1))
        pem[:, 0] = ptop                                    # :99-104
        peln2[:, 0] = peln1
        pk3[o:o + ni, jj, 0] = ptk
        for k in range(1, km + 1):                          # :108-118
            pem[:, k] = pem[:, k - 1] + dm[:, k - 1]
            peln2[:, k] = np.log(pem[:, k])
            pk3[o:o + ni, jj, k] = np.exp(akap * peln2[:, k])

        pm2 = np.empty((ni, km))
        dz2 = np.empty((ni, km))
        w2 = np.empty((ni, km))
        for k in range(km):                                 # :121-135
            pm2[:, k] = dm[:, k] / (peln2[:, k + 1] - peln2[:, k])
            dm[:, k] = dm[:, k] * rgrav
            dz2[:, k] = zh[o:o + ni, jj, k + 1] - zh[o:o + ni, jj, k]
            w2[:, k] = w[o:o + ni, jj, k]

        if ws.shape != (ni, je - js + 1):
            raise ValueError(
                f"riem_solver3: ws must be the compute window "
                f"({ni}, {je - js + 1}) (nh_core.F90:60 ws(is:ie,js:je)), "
                f"got {ws.shape}")
        pe2 = np.zeros((ni, km + 1))
        sim1_solver(dt, is_, ie, km, FV3_RDGAS, gama, akap, pe2, dm,
                    pm2, pem, w2, dz2,
                    np.array(pt[o:o + ni, jj, :km], dtype=np.float64),
                    np.array(ws[:, jc], dtype=np.float64),
                    p_fac)                                  # :147-149

        for k in range(km):                                 # :157-162
            w[o:o + ni, jj, k] = w2[:, k]
            delz[:, jc, k] = dz2[:, k]

        if last_call:                                       # :164-172
            # pe is the oracle's (is-1:ie+1, km+1, js-1:je+1): the copy
            # writes i=is..ie only, i.e. rows 1..ni of the first axis,
            # and column j maps to slot jc+1.  The i=is-1/ie+1 rows and
            # j ring slots keep whatever the caller carried (upstream
            # fills them via pe_halo, a separate routine).
            for k in range(km + 1):
                peln[:, k, jc] = peln2[:, k]
                pk[:, jc, k] = pk3[o:o + ni, jj, k]
                pe[1:1 + ni, k, jc + 1] = pem[:, k]

        if fp_out:                                          # :173-185
            for k in range(km + 1):
                ppe[o:o + ni, jj, k] = pe2[:, k] + pem[:, k]
        else:
            for k in range(km + 1):
                ppe[o:o + ni, jj, k] = pe2[:, k]

        if use_logp:                                        # :187-193
            for k in range(1, km + 1):
                pk3[o:o + ni, jj, k] = peln2[:, k]

        zh[o:o + ni, jj, km] = zs[o:o + ni, jj]             # :195-197
        for k in range(km - 1, -1, -1):                     # :198-202
            zh[o:o + ni, jj, k] = zh[o:o + ni, jj, k + 1] - dz2[:, k]


def riem_solver_c(ms: int, dt: float, bd, km: int, akap: float, cp: float,
                  ptop: float, hs: np.ndarray, w3: np.ndarray,
                  pt: np.ndarray, delp: np.ndarray, gz: np.ndarray,
                  pef: np.ndarray, ws: np.ndarray, p_fac: float,
                  a_imp: float, scale_m: float = 0.0) -> None:
    """``nh_utils.F90:313-420`` verbatim, non-USE_COND branch.

    Padded (i,j,k) arrays as in :func:`update_dz_c`; ``gz`` (km+1) is
    updated in place, ``pef`` (km+1) is output.  The j loop runs
    ``js-1..je+1`` and the i window is ``is-1..ie+1`` (:334, :339-345).
    Dispatch: on the pinned deck ``a_imp = 1.`` so only the SIM1 arm is
    implemented; the two dead arms RAISE rather than fall through —
    a deck change must be heard, not absorbed.
    """
    is_, ie, js, je, ng = bd.is_, bd.ie, bd.js, bd.je, bd.ng
    ilo = is_ - ng
    jlo = js - ng
    gama = 1.0 / (1.0 - akap)
    rgrav = 1.0 / FV3_GRAV

    if not (a_imp > 0.5):
        raise NotImplementedError(
            f"riem_solver_c: a_imp={a_imp} selects a dead arm on the "
            f"pinned deck (a_imp=1. -> SIM1); SIM3p0/RIM_2D are not "
            f"ported. nh_utils.F90:392-401.")

    is1, ie1 = is_ - 1, ie + 1
    ni = ie1 - is1 + 1
    o = is1 - ilo                    # offset of i=is-1 in the padded axis

    for j in range(js - 1, je + 2):
        jj = j - jlo
        dm = np.array(delp[o:o + ni, jj, :km], dtype=np.float64)   # :347-351
        pem = np.zeros((ni, km + 1))
        pem[:, 0] = ptop                                           # :354-356
        pef[o:o + ni, jj, 0] = ptop
        for k in range(1, km + 1):                                 # :362-366
            pem[:, k] = pem[:, k - 1] + dm[:, k - 1]

        dz2 = np.empty((ni, km))
        pm2 = np.empty((ni, km))
        w2 = np.empty((ni, km))
        for k in range(km):                                        # :370-385
            dz2[:, k] = gz[o:o + ni, jj, k + 1] - gz[o:o + ni, jj, k]
            pm2[:, k] = dm[:, k] / np.log(pem[:, k + 1] / pem[:, k])
            dm[:, k] = dm[:, k] * rgrav
            w2[:, k] = w3[o:o + ni, jj, k]

        pe2 = np.zeros((ni, km + 1))
        sim1_solver(dt, is1, ie1, km, FV3_RDGAS, gama, akap, pe2, dm,
                    pm2, pem, w2, dz2,
                    np.array(pt[o:o + ni, jj, :km], dtype=np.float64),
                    np.array(ws[o:o + ni, jj], dtype=np.float64),
                    p_fac)                                         # :398-400

        for k in range(1, km + 1):                                 # :403-407
            pef[o:o + ni, jj, k] = pe2[:, k] + pem[:, k]

        gz[o:o + ni, jj, km] = hs[o:o + ni, jj]                    # :410-412
        for k in range(km - 1, -1, -1):                            # :414-418
            gz[o:o + ni, jj, k] = (gz[o:o + ni, jj, k + 1]
                                   - dz2[:, k] * FV3_GRAV)
