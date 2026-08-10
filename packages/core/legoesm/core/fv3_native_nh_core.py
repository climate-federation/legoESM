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
    lo = is_ - ng                     # array origin in i and j

    def _f2(a2d):
        return fort(a2d, lo, lo)

    gzf = fort(gz, lo, lo)            # 3-D view; k passed 0-based below

    for k1 in range(1, km + 2):       # do 6000 k=1,km+1
        k = k1 - 1                    # 0-based level index into ut/vt/gz
        xfx = np.zeros((ie2 - is1 + 1, je1 - js1 + 1))
        yfx = np.zeros((ie1 - is1 + 1, je2 - js1 + 1))
        xf = fort(xfx, is1, js1)
        yf = fort(yfx, is1, js1)
        utf = fort(ut, lo, lo)
        vtf = fort(vt, lo, lo)

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

        af = fort(area.a if isinstance(area, fort) else area, lo, lo)
        for j in range(js1, je1 + 1):                          # :166-171
            for i in range(is1, ie1 + 1):
                gzf[i, j, k] = (
                    (g2[i, j] * af[i, j] + fxf[i, j] - fxf[i + 1, j]
                     + fyf[i, j] - fyf[i, j + 1])
                    / (af[i, j] + xf[i, j] - xf[i + 1, j]
                       + yf[i, j] - yf[i, j + 1]))

    # :173-189  ws diagnosis + monotone-height limiter (dz_min = 2.)
    zsf = fort(zs.a if isinstance(zs, fort) else zs, lo, lo)
    wsf = fort(ws.a if isinstance(ws, fort) else ws, lo, lo)
    for j in range(js1, je1 + 1):
        for i in range(is1, ie1 + 1):
            wsf[i, j] = (zsf[i, j] - gzf[i, j, km]) * rdt
        for k in range(km - 1, -1, -1):        # do k=km,1,-1 (0-based)
            for i in range(is1, ie1 + 1):
                gzf[i, j, k] = max(gzf[i, j, k], gzf[i, j, k + 1] + DZ_MIN)


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
    lo = is_ - ng
    gama = 1.0 / (1.0 - akap)
    rgrav = 1.0 / FV3_GRAV

    if not (a_imp > 0.5):
        raise NotImplementedError(
            f"riem_solver_c: a_imp={a_imp} selects a dead arm on the "
            f"pinned deck (a_imp=1. -> SIM1); SIM3p0/RIM_2D are not "
            f"ported. nh_utils.F90:392-401.")

    is1, ie1 = is_ - 1, ie + 1
    ni = ie1 - is1 + 1
    o = is1 - lo                     # offset of i=is-1 in the padded axis

    for j in range(js - 1, je + 2):
        jj = j - lo
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
