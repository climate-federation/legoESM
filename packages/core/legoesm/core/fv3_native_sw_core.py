"""FV3-native C-grid half step (c_sw) — loop-faithful port, phase 4a.

Index-exact python transcription of ``model/sw_core.F90`` @ 6f658bd0
(GFDL_atmos_cubed_sphere, production branch — no SW_DYNAMICS) for the
single-tile cubed-sphere case:

- ``c_sw`` (C-grid forward step: 1st-order upwind delp/pt transport,
  upwind KE, corner absolute vorticity, vorticity flux + KE-gradient
  update of the C winds),
- ``d2a2c_vect`` (D->A->C covariant/contravariant wind chain with the
  cube-corner utmp/vtmp/ua/va fixes and panel-edge one-sided stencils),
- ``divergence_corner`` (B-node divergence for the del-4/6 damping),
- ``fill2_4corners``/``fill_4corners`` (scalar corner ghost fills),
- ``edge_interpolate4``.

Certified field-by-field against the verbatim Fortran extraction run on
identical inputs (``tests/grids/test_fv3_native_swcore_phase4.py``; oracle
harness under ``scripts/validate/fv3_native/``).  This module is the
FIDELITY REFERENCE: plain-python loops mirroring the Fortran statement
order (fast enough for oracle grids).  The vectorized/JAX production
implementation of the native forward-backward core must reproduce THIS
module (and hence the Fortran) before it may claim FV3 fidelity — that is
the phase-4 naming-tripwire condition (see test_fv3_naming_phase5).

Array convention: plain numpy, float64, every axis' row 0 == Fortran
``isd = 1 - ng`` (node-type axes are one slot longer).  Winds follow FV3
staggering: ``u`` on x-edges (isd:ied, jsd:jed+1), ``v`` on y-edges
(isd:ied+1, jsd:jed) in Fortran bounds.
"""

from __future__ import annotations

from typing import NamedTuple

import numpy as np
from legoesm.grids.fv3_native_gridstruct import fort

# sw_core.F90 module constants (verbatim; big_number is the production
# non-OVERLOAD_R4 branch value — codex r1 P1-1)
BIG_NUMBER = 1.0e30
A1 = 0.5625
A2 = -0.0625
C1 = -2.0 / 14.0
C2 = 11.0 / 14.0
C3 = 5.0 / 14.0


class Bounds(NamedTuple):
    """fv_grid_bounds_type for one whole-face tile."""

    is_: int
    ie: int
    js: int
    je: int
    isd: int
    ied: int
    jsd: int
    jed: int
    ng: int

    @classmethod
    def single_tile(cls, n: int, ng: int) -> Bounds:
        return cls(1, n, 1, n, 1 - ng, n + ng, 1 - ng, n + ng, ng)


def _fa(bd: Bounds, ihi_extra: int, jhi_extra: int,
        fill: float = np.nan) -> fort:
    """New Fortran-indexed data-domain array (cell + extra node slots)."""
    ni = bd.ied - bd.isd + 1 + ihi_extra
    nj = bd.jed - bd.jsd + 1 + jhi_extra
    return fort(np.full((ni, nj), fill), bd.isd, bd.jsd)


def _fl(ilo: int, ihi: int, jlo: int, jhi: int,
        fill: float = np.nan) -> fort:
    """New Fortran-indexed local array with explicit bounds."""
    return fort(np.full((ihi - ilo + 1, jhi - jlo + 1), fill), ilo, jlo)


def edge_interpolate4(ua4, dxa4) -> float:
    """sw_core.F90 edge_interpolate4 (verbatim)."""
    t1 = dxa4[0] + dxa4[1]
    t2 = dxa4[2] + dxa4[3]
    return 0.5 * (((t1 + dxa4[1]) * ua4[1] - dxa4[1] * ua4[0]) / t1
                  + ((t2 + dxa4[2]) * ua4[2] - dxa4[2] * ua4[3]) / t2)


def fill2_4corners(q1: fort, q2: fort, direction: int, npx: int, npy: int,
                   sw: bool = True, se: bool = True,
                   ne: bool = True, nw: bool = True) -> None:
    """sw_core.F90 fill2_4corners (verbatim index mapping)."""
    if direction == 1:
        if sw:
            q1[-1, 0] = q1[0, 2]
            q1[0, 0] = q1[0, 1]
            q2[-1, 0] = q2[0, 2]
            q2[0, 0] = q2[0, 1]
        if se:
            q1[npx + 1, 0] = q1[npx, 2]
            q1[npx, 0] = q1[npx, 1]
            q2[npx + 1, 0] = q2[npx, 2]
            q2[npx, 0] = q2[npx, 1]
        if nw:
            q1[0, npy] = q1[0, npy - 1]
            q1[-1, npy] = q1[0, npy - 2]
            q2[0, npy] = q2[0, npy - 1]
            q2[-1, npy] = q2[0, npy - 2]
        if ne:
            q1[npx, npy] = q1[npx, npy - 1]
            q1[npx + 1, npy] = q1[npx, npy - 2]
            q2[npx, npy] = q2[npx, npy - 1]
            q2[npx + 1, npy] = q2[npx, npy - 2]
    elif direction == 2:
        if sw:
            q1[0, 0] = q1[1, 0]
            q1[0, -1] = q1[2, 0]
            q2[0, 0] = q2[1, 0]
            q2[0, -1] = q2[2, 0]
        if se:
            q1[npx, 0] = q1[npx - 1, 0]
            q1[npx, -1] = q1[npx - 2, 0]
            q2[npx, 0] = q2[npx - 1, 0]
            q2[npx, -1] = q2[npx - 2, 0]
        if nw:
            q1[0, npy] = q1[1, npy]
            q1[0, npy + 1] = q1[2, npy]
            q2[0, npy] = q2[1, npy]
            q2[0, npy + 1] = q2[2, npy]
        if ne:
            q1[npx, npy] = q1[npx - 1, npy]
            q1[npx, npy + 1] = q1[npx - 2, npy]
            q2[npx, npy] = q2[npx - 1, npy]
            q2[npx, npy + 1] = q2[npx - 2, npy]
    else:  # pragma: no cover - guard
        raise ValueError(f"fill2_4corners: dir={direction}")


def fill_4corners(q: fort, direction: int, npx: int, npy: int,
                  sw: bool = True, se: bool = True,
                  ne: bool = True, nw: bool = True) -> None:
    """sw_core.F90 fill_4corners (verbatim index mapping)."""
    if direction == 1:
        if sw:
            q[-1, 0] = q[0, 2]
            q[0, 0] = q[0, 1]
        if se:
            q[npx + 1, 0] = q[npx, 2]
            q[npx, 0] = q[npx, 1]
        if nw:
            q[0, npy] = q[0, npy - 1]
            q[-1, npy] = q[0, npy - 2]
        if ne:
            q[npx, npy] = q[npx, npy - 1]
            q[npx + 1, npy] = q[npx, npy - 2]
    elif direction == 2:
        if sw:
            q[0, 0] = q[1, 0]
            q[0, -1] = q[2, 0]
        if se:
            q[npx, 0] = q[npx - 1, 0]
            q[npx, -1] = q[npx - 2, 0]
        if nw:
            q[0, npy] = q[1, npy]
            q[0, npy + 1] = q[2, npy]
        if ne:
            q[npx, npy] = q[npx - 1, npy]
            q[npx, npy + 1] = q[npx - 2, npy]
    else:  # pragma: no cover - guard
        raise ValueError(f"fill_4corners: dir={direction}")


def d2a2c_vect(u: np.ndarray, v: np.ndarray, gs: dict, bd: Bounds,
               npx: int, npy: int, dord4: bool = True,
               grid_type: int = 0, bounded_domain: bool = False):
    """sw_core.F90 d2a2c_vect — returns (ua, va, uc, vc, ut, vt).

    Cubed-sphere whole-face tile (all four corner flags true).  Output
    arrays are zero-initialized (matching the oracle driver), so slots the
    Fortran never writes hold 0.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COSA_U = fort(gs["cosa_u"], isd, jsd)
    COSA_V = fort(gs["cosa_v"], isd, jsd)
    COSA_S = fort(gs["cosa_s"], isd, jsd)
    RSIN_U = fort(gs["rsin_u"], isd, jsd)
    RSIN_V = fort(gs["rsin_v"], isd, jsd)
    RSIN2 = fort(gs["rsin2"], isd, jsd)
    DXA = fort(gs["dxa"], isd, jsd)
    DYA = fort(gs["dya"], isd, jsd)

    UA = _fa(bd, 0, 0, 0.0)
    VA = _fa(bd, 0, 0, 0.0)
    UT = _fa(bd, 0, 0, 0.0)
    VT = _fa(bd, 0, 0, 0.0)
    UC = _fa(bd, 1, 0, 0.0)
    VC = _fa(bd, 0, 1, 0.0)

    id_ = 1 if dord4 else 0
    npt = 4 if (grid_type < 3 and not bounded_domain) else -2

    utmp = _fl(isd, ied, jsd, jed, BIG_NUMBER)
    vtmp = _fl(isd, ied, jsd, jed, BIG_NUMBER)

    if bounded_domain:  # pragma: no cover - oracle is cubed-sphere only
        raise NotImplementedError("bounded_domain d2a2c not ported")

    # ---------- interior ----------
    for j in range(max(npt, js - 1), min(npy - npt, je + 1) + 1):
        for i in range(max(npt, isd), min(npx - npt, ied) + 1):
            utmp[i, j] = A2 * (U[i, j - 1] + U[i, j + 2]) \
                + A1 * (U[i, j] + U[i, j + 1])
    for j in range(max(npt, jsd), min(npy - npt, jed) + 1):
        for i in range(max(npt, is_ - 1), min(npx - npt, ie + 1) + 1):
            vtmp[i, j] = A2 * (V[i - 1, j] + V[i + 2, j]) \
                + A1 * (V[i, j] + V[i + 1, j])

    # ---------- edges ----------
    if grid_type < 3:
        if js == 1 or jsd < npt:
            for j in range(jsd, npt - 1 + 1):
                for i in range(isd, ied + 1):
                    utmp[i, j] = 0.5 * (U[i, j] + U[i, j + 1])
                    vtmp[i, j] = 0.5 * (V[i, j] + V[i + 1, j])
        if (je + 1) == npy or jed >= (npy - npt):
            for j in range(npy - npt + 1, jed + 1):
                for i in range(isd, ied + 1):
                    utmp[i, j] = 0.5 * (U[i, j] + U[i, j + 1])
                    vtmp[i, j] = 0.5 * (V[i, j] + V[i + 1, j])
        if is_ == 1 or isd < npt:
            for j in range(max(npt, jsd), min(npy - npt, jed) + 1):
                for i in range(isd, npt - 1 + 1):
                    utmp[i, j] = 0.5 * (U[i, j] + U[i, j + 1])
                    vtmp[i, j] = 0.5 * (V[i, j] + V[i + 1, j])
        if (ie + 1) == npx or ied >= (npx - npt):
            for j in range(max(npt, jsd), min(npy - npt, jed) + 1):
                for i in range(npx - npt + 1, ied + 1):
                    utmp[i, j] = 0.5 * (U[i, j] + U[i, j + 1])
                    vtmp[i, j] = 0.5 * (V[i, j] + V[i + 1, j])

    # contra-variant components at cell centre
    for j in range(js - 1 - id_, je + 1 + id_ + 1):
        for i in range(is_ - 1 - id_, ie + 1 + id_ + 1):
            UA[i, j] = (utmp[i, j] - vtmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]
            VA[i, j] = (vtmp[i, j] - utmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]

    # ---------- A -> C: fix the edges, Xdir ----------
    for i in range(-2, 0 + 1):                          # sw corner
        utmp[i, 0] = -vtmp[0, 1 - i]
    for i in range(0, 2 + 1):                           # se corner
        utmp[npx + i, 0] = vtmp[npx, i + 1]
    for i in range(0, 2 + 1):                           # ne corner
        utmp[npx + i, npy] = -vtmp[npx, je - i]
    for i in range(-2, 0 + 1):                          # nw corner
        utmp[i, npy] = vtmp[0, je + i]

    if grid_type < 3 and not bounded_domain:
        ifirst = max(3, is_ - 1)
        ilast = min(npx - 2, ie + 2)
    else:  # pragma: no cover - cubed-sphere oracle
        ifirst = is_ - 1
        ilast = ie + 2

    # 4th-order interpolation for interior points
    for j in range(js - 1, je + 1 + 1):
        for i in range(ifirst, ilast + 1):
            UC[i, j] = A2 * (utmp[i - 2, j] + utmp[i + 1, j]) \
                + A1 * (utmp[i - 1, j] + utmp[i, j])
            UT[i, j] = (UC[i, j] - V[i, j] * COSA_U[i, j]) * RSIN_U[i, j]

    if grid_type < 3:
        # Xdir ua corner overrides
        UA[-1, 0] = -VA[0, 2]                           # sw
        UA[0, 0] = -VA[0, 1]
        UA[npx, 0] = VA[npx, 1]                         # se
        UA[npx + 1, 0] = VA[npx, 2]
        UA[npx, npy] = -VA[npx, npy - 1]                # ne
        UA[npx + 1, npy] = -VA[npx, npy - 2]
        UA[-1, npy] = VA[0, npy - 2]                    # nw
        UA[0, npy] = VA[0, npy - 1]

        if is_ == 1 and not bounded_domain:
            for j in range(js - 1, je + 1 + 1):
                UC[0, j] = C1 * utmp[-2, j] + C2 * utmp[-1, j] \
                    + C3 * utmp[0, j]
                UT[1, j] = edge_interpolate4(
                    [UA[-1, j], UA[0, j], UA[1, j], UA[2, j]],
                    [DXA[-1, j], DXA[0, j], DXA[1, j], DXA[2, j]])
                if UT[1, j] > 0.0:
                    UC[1, j] = UT[1, j] * SIN_SG[0, j, 3 - 1]
                else:
                    UC[1, j] = UT[1, j] * SIN_SG[1, j, 1 - 1]
                UC[2, j] = C1 * utmp[3, j] + C2 * utmp[2, j] \
                    + C3 * utmp[1, j]
                UT[0, j] = (UC[0, j] - V[0, j] * COSA_U[0, j]) * RSIN_U[0, j]
                UT[2, j] = (UC[2, j] - V[2, j] * COSA_U[2, j]) * RSIN_U[2, j]

        if (ie + 1) == npx and not bounded_domain:
            for j in range(js - 1, je + 1 + 1):
                UC[npx - 1, j] = C1 * utmp[npx - 3, j] \
                    + C2 * utmp[npx - 2, j] + C3 * utmp[npx - 1, j]
                UT[npx, j] = edge_interpolate4(
                    [UA[npx - 2, j], UA[npx - 1, j],
                     UA[npx, j], UA[npx + 1, j]],
                    [DXA[npx - 2, j], DXA[npx - 1, j],
                     DXA[npx, j], DXA[npx + 1, j]])
                if UT[npx, j] > 0.0:
                    UC[npx, j] = UT[npx, j] * SIN_SG[npx - 1, j, 3 - 1]
                else:
                    UC[npx, j] = UT[npx, j] * SIN_SG[npx, j, 1 - 1]
                UC[npx + 1, j] = C3 * utmp[npx, j] \
                    + C2 * utmp[npx + 1, j] + C1 * utmp[npx + 2, j]
                UT[npx - 1, j] = (UC[npx - 1, j]
                                  - V[npx - 1, j] * COSA_U[npx - 1, j]) \
                    * RSIN_U[npx - 1, j]
                UT[npx + 1, j] = (UC[npx + 1, j]
                                  - V[npx + 1, j] * COSA_U[npx + 1, j]) \
                    * RSIN_U[npx + 1, j]

    # ---------- Ydir ----------
    for j in range(-2, 0 + 1):                          # sw corner
        vtmp[0, j] = -utmp[1 - j, 0]
    for j in range(0, 2 + 1):                           # nw corner
        vtmp[0, npy + j] = utmp[j + 1, npy]
    for j in range(-2, 0 + 1):                          # se corner
        vtmp[npx, j] = utmp[ie + j, 0]
    for j in range(0, 2 + 1):                           # ne corner
        vtmp[npx, npy + j] = -utmp[ie - j, npy]
    VA[0, -1] = -UA[2, 0]                               # sw
    VA[0, 0] = -UA[1, 0]
    VA[npx, 0] = UA[npx - 1, 0]                         # se
    VA[npx, -1] = UA[npx - 2, 0]
    VA[npx, npy] = -UA[npx - 1, npy]                    # ne
    VA[npx, npy + 1] = -UA[npx - 2, npy]
    VA[0, npy] = UA[1, npy]                             # nw
    VA[0, npy + 1] = UA[2, npy]

    if grid_type < 3:
        for j in range(js - 1, je + 2 + 1):
            if j == 1 and not bounded_domain:
                for i in range(is_ - 1, ie + 1 + 1):
                    VT[i, j] = edge_interpolate4(
                        [VA[i, -1], VA[i, 0], VA[i, 1], VA[i, 2]],
                        [DYA[i, -1], DYA[i, 0], DYA[i, 1], DYA[i, 2]])
                    if VT[i, j] > 0.0:
                        VC[i, j] = VT[i, j] * SIN_SG[i, j - 1, 4 - 1]
                    else:
                        VC[i, j] = VT[i, j] * SIN_SG[i, j, 2 - 1]
            elif j == 0 or (j == (npy - 1) and not bounded_domain):
                for i in range(is_ - 1, ie + 1 + 1):
                    VC[i, j] = C1 * vtmp[i, j - 2] + C2 * vtmp[i, j - 1] \
                        + C3 * vtmp[i, j]
                    VT[i, j] = (VC[i, j] - U[i, j] * COSA_V[i, j]) \
                        * RSIN_V[i, j]
            elif j == 2 or (j == (npy + 1) and not bounded_domain):
                for i in range(is_ - 1, ie + 1 + 1):
                    VC[i, j] = C1 * vtmp[i, j + 1] + C2 * vtmp[i, j] \
                        + C3 * vtmp[i, j - 1]
                    VT[i, j] = (VC[i, j] - U[i, j] * COSA_V[i, j]) \
                        * RSIN_V[i, j]
            elif j == npy and not bounded_domain:
                for i in range(is_ - 1, ie + 1 + 1):
                    VT[i, j] = edge_interpolate4(
                        [VA[i, j - 2], VA[i, j - 1], VA[i, j], VA[i, j + 1]],
                        [DYA[i, j - 2], DYA[i, j - 1],
                         DYA[i, j], DYA[i, j + 1]])
                    if VT[i, j] > 0.0:
                        VC[i, j] = VT[i, j] * SIN_SG[i, j - 1, 4 - 1]
                    else:
                        VC[i, j] = VT[i, j] * SIN_SG[i, j, 2 - 1]
            else:
                # 4th-order interpolation for interior points
                for i in range(is_ - 1, ie + 1 + 1):
                    VC[i, j] = A2 * (vtmp[i, j - 2] + vtmp[i, j + 1]) \
                        + A1 * (vtmp[i, j - 1] + vtmp[i, j])
                    VT[i, j] = (VC[i, j] - U[i, j] * COSA_V[i, j]) \
                        * RSIN_V[i, j]
    else:  # pragma: no cover - cubed-sphere oracle
        for j in range(js - 1, je + 2 + 1):
            for i in range(is_ - 1, ie + 1 + 1):
                VC[i, j] = A2 * (vtmp[i, j - 2] + vtmp[i, j + 1]) \
                    + A1 * (vtmp[i, j - 1] + vtmp[i, j])
                VT[i, j] = VC[i, j]

    return UA.a, VA.a, UC.a, VC.a, UT.a, VT.a


def divergence_corner(u: np.ndarray, v: np.ndarray,
                      ua: np.ndarray, va: np.ndarray,
                      gs: dict, bd: Bounds, npx: int, npy: int,
                      grid_type: int = 0) -> np.ndarray:
    """sw_core.F90 divergence_corner — returns divg_d (B-node array).

    Never-written halo slots hold NaN (the oracle driver holds a sentinel
    there; the reconciliation test compares written slots only).
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    UA = fort(ua, isd, jsd)
    VA = fort(va, isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COS_SG = fort(gs["cos_sg"], isd, jsd)
    DXC = fort(gs["dxc"], isd, jsd)
    DYC = fort(gs["dyc"], isd, jsd)
    RAREA_C = fort(gs["rarea_c"], isd, jsd)

    DIVG = _fa(bd, 1, 1)

    is2 = max(2, is_)
    ie1 = min(npx - 1, ie + 1)

    if grid_type > 3:  # pragma: no cover - cubed-sphere oracle
        raise NotImplementedError("grid_type > 3 not ported")

    uf = _fl(is_ - 2, ie + 2, js - 1, je + 2)
    vf = _fl(is_ - 1, ie + 2, js - 2, je + 2)

    for j in range(js, je + 1 + 1):
        if j == 1 or j == npy:
            for i in range(is_ - 1, ie + 1 + 1):
                uf[i, j] = U[i, j] * DYC[i, j] * 0.5 \
                    * (SIN_SG[i, j - 1, 4 - 1] + SIN_SG[i, j, 2 - 1])
        else:
            for i in range(is_ - 1, ie + 1 + 1):
                uf[i, j] = (U[i, j] - 0.25 * (VA[i, j - 1] + VA[i, j])
                            * (COS_SG[i, j - 1, 4 - 1]
                               + COS_SG[i, j, 2 - 1])) \
                    * DYC[i, j] * 0.5 \
                    * (SIN_SG[i, j - 1, 4 - 1] + SIN_SG[i, j, 2 - 1])

    for j in range(js - 1, je + 1 + 1):
        for i in range(is2, ie1 + 1):
            vf[i, j] = (V[i, j] - 0.25 * (UA[i - 1, j] + UA[i, j])
                        * (COS_SG[i - 1, j, 3 - 1] + COS_SG[i, j, 1 - 1])) \
                * DXC[i, j] * 0.5 \
                * (SIN_SG[i - 1, j, 3 - 1] + SIN_SG[i, j, 1 - 1])
        if is_ == 1:
            vf[1, j] = V[1, j] * DXC[1, j] * 0.5 \
                * (SIN_SG[0, j, 3 - 1] + SIN_SG[1, j, 1 - 1])
        if (ie + 1) == npx:
            vf[npx, j] = V[npx, j] * DXC[npx, j] * 0.5 \
                * (SIN_SG[npx - 1, j, 3 - 1] + SIN_SG[npx, j, 1 - 1])

    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            DIVG[i, j] = vf[i, j - 1] - vf[i, j] + uf[i - 1, j] - uf[i, j]

    # remove the extra term at the corners
    DIVG[1, 1] = DIVG[1, 1] - vf[1, 0]
    DIVG[npx, 1] = DIVG[npx, 1] - vf[npx, 0]
    DIVG[npx, npy] = DIVG[npx, npy] + vf[npx, npy]
    DIVG[1, npy] = DIVG[1, npy] + vf[1, npy]

    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            DIVG[i, j] = RAREA_C[i, j] * DIVG[i, j]

    return DIVG.a


def c_sw(delp: np.ndarray, pt: np.ndarray, w: np.ndarray,
         u: np.ndarray, v: np.ndarray, gs: dict, bd: Bounds,
         npx: int, npy: int, dt2: float, nord: int = 1,
         hydrostatic: bool = True, dord4: bool = True,
         grid_type: int = 0) -> dict:
    """sw_core.F90 c_sw (production branch) — one C-grid forward step.

    Mutates nothing; returns a dict with delpc, ptc, wc, uc, vc, ua, va,
    ut, vt, divg_d as full-size numpy arrays (Fortran bounds convention).
    ``delp``/``pt``/``w`` corner ghosts are filled internally
    (fill2_4corners / fill_4corners) exactly as upstream — pass arrays
    with any deterministic corner-region content.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, jsd = bd.isd, bd.jsd
    iep1, jep1 = ie + 1, je + 1
    bounded_domain = False

    delp = np.array(delp, dtype=np.float64, copy=True)
    pt = np.array(pt, dtype=np.float64, copy=True)
    w = np.array(w, dtype=np.float64, copy=True)

    DELP = fort(delp, isd, jsd)
    PT = fort(pt, isd, jsd)
    W = fort(w, isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COS_SG = fort(gs["cos_sg"], isd, jsd)
    COSA_U = fort(gs["cosa_u"], isd, jsd)
    COSA_V = fort(gs["cosa_v"], isd, jsd)
    SINA_U = fort(gs["sina_u"], isd, jsd)
    SINA_V = fort(gs["sina_v"], isd, jsd)
    DX = fort(gs["dx"], isd, jsd)
    DY = fort(gs["dy"], isd, jsd)
    DXC = fort(gs["dxc"], isd, jsd)
    DYC = fort(gs["dyc"], isd, jsd)
    RDXC = fort(gs["rdxc"], isd, jsd)
    RDYC = fort(gs["rdyc"], isd, jsd)
    RAREA = fort(gs["rarea"], isd, jsd)
    RAREA_C = fort(gs["rarea_c"], isd, jsd)
    FC = fort(gs["fC"], isd, jsd)

    UF = fort(u, isd, jsd)   # D winds (read-only here)
    VF = fort(v, isd, jsd)

    ua_a, va_a, uc_a, vc_a, ut_a, vt_a = d2a2c_vect(
        u, v, gs, bd, npx, npy, dord4=dord4, grid_type=grid_type,
        bounded_domain=bounded_domain)
    UA = fort(ua_a, isd, jsd)
    VA = fort(va_a, isd, jsd)
    UC = fort(uc_a, isd, jsd)
    VC = fort(vc_a, isd, jsd)
    UT = fort(ut_a, isd, jsd)
    VT = fort(vt_a, isd, jsd)

    if nord > 0:
        divg_a = divergence_corner(u, v, ua_a, va_a, gs, bd, npx, npy,
                                   grid_type=grid_type)
    else:  # pragma: no cover - production uses nord > 0
        divg_a = _fa(bd, 1, 1).a

    # ---- scale the transport winds ----
    for j in range(js - 1, jep1 + 1):
        for i in range(is_ - 1, iep1 + 1 + 1):
            if UT[i, j] > 0.0:
                UT[i, j] = dt2 * UT[i, j] * DY[i, j] * SIN_SG[i - 1, j, 3 - 1]
            else:
                UT[i, j] = dt2 * UT[i, j] * DY[i, j] * SIN_SG[i, j, 1 - 1]
    for j in range(js - 1, je + 2 + 1):
        for i in range(is_ - 1, iep1 + 1):
            if VT[i, j] > 0.0:
                VT[i, j] = dt2 * VT[i, j] * DX[i, j] * SIN_SG[i, j - 1, 4 - 1]
            else:
                VT[i, j] = dt2 * VT[i, j] * DX[i, j] * SIN_SG[i, j, 2 - 1]

    # ---- transport delp (+ pt, and w if non-hydrostatic) ----
    fx = _fl(is_ - 1, ie + 2, js - 1, je + 1)
    fx1 = _fl(is_ - 1, ie + 2, js - 1, je + 1)
    fx2 = _fl(is_ - 1, ie + 2, js - 1, je + 1)
    fy = _fl(is_ - 1, ie + 1, js - 1, je + 2)
    fy1 = _fl(is_ - 1, ie + 1, js - 1, je + 2)
    fy2 = _fl(is_ - 1, ie + 1, js - 1, je + 2)

    DELPC = _fa(bd, 0, 0)
    PTC = _fa(bd, 0, 0)
    WC = _fa(bd, 0, 0)

    # Xdir
    if grid_type < 3 and not bounded_domain:
        fill2_4corners(DELP, PT, 1, npx, npy)
    if hydrostatic:
        for j in range(js - 1, jep1 + 1):
            for i in range(is_ - 1, ie + 2 + 1):
                if UT[i, j] > 0.0:
                    fx1[i, j] = DELP[i - 1, j]
                    fx[i, j] = PT[i - 1, j]
                else:
                    fx1[i, j] = DELP[i, j]
                    fx[i, j] = PT[i, j]
                fx1[i, j] = UT[i, j] * fx1[i, j]
                fx[i, j] = fx1[i, j] * fx[i, j]
    else:
        if grid_type < 3:
            fill_4corners(W, 1, npx, npy)
        for j in range(js - 1, je + 1 + 1):
            for i in range(is_ - 1, ie + 2 + 1):
                if UT[i, j] > 0.0:
                    fx1[i, j] = DELP[i - 1, j]
                    fx[i, j] = PT[i - 1, j]
                    fx2[i, j] = W[i - 1, j]
                else:
                    fx1[i, j] = DELP[i, j]
                    fx[i, j] = PT[i, j]
                    fx2[i, j] = W[i, j]
                fx1[i, j] = UT[i, j] * fx1[i, j]
                fx[i, j] = fx1[i, j] * fx[i, j]
                fx2[i, j] = fx1[i, j] * fx2[i, j]

    # Ydir
    if grid_type < 3 and not bounded_domain:
        fill2_4corners(DELP, PT, 2, npx, npy)
    if hydrostatic:
        for j in range(js - 1, jep1 + 1 + 1):
            for i in range(is_ - 1, iep1 + 1):
                if VT[i, j] > 0.0:
                    fy1[i, j] = DELP[i, j - 1]
                    fy[i, j] = PT[i, j - 1]
                else:
                    fy1[i, j] = DELP[i, j]
                    fy[i, j] = PT[i, j]
                fy1[i, j] = VT[i, j] * fy1[i, j]
                fy[i, j] = fy1[i, j] * fy[i, j]
        for j in range(js - 1, jep1 + 1):
            for i in range(is_ - 1, iep1 + 1):
                DELPC[i, j] = DELP[i, j] + (fx1[i, j] - fx1[i + 1, j]
                                            + fy1[i, j] - fy1[i, j + 1]) \
                    * RAREA[i, j]
                PTC[i, j] = (PT[i, j] * DELP[i, j]
                             + (fx[i, j] - fx[i + 1, j]
                                + fy[i, j] - fy[i, j + 1])
                             * RAREA[i, j]) / DELPC[i, j]
    else:
        if grid_type < 3:
            fill_4corners(W, 2, npx, npy)
        for j in range(js - 1, je + 2 + 1):
            for i in range(is_ - 1, ie + 1 + 1):
                if VT[i, j] > 0.0:
                    fy1[i, j] = DELP[i, j - 1]
                    fy[i, j] = PT[i, j - 1]
                    fy2[i, j] = W[i, j - 1]
                else:
                    fy1[i, j] = DELP[i, j]
                    fy[i, j] = PT[i, j]
                    fy2[i, j] = W[i, j]
                fy1[i, j] = VT[i, j] * fy1[i, j]
                fy[i, j] = fy1[i, j] * fy[i, j]
                fy2[i, j] = fy1[i, j] * fy2[i, j]
        for j in range(js - 1, je + 1 + 1):
            for i in range(is_ - 1, ie + 1 + 1):
                DELPC[i, j] = DELP[i, j] + (fx1[i, j] - fx1[i + 1, j]
                                            + fy1[i, j] - fy1[i, j + 1]) \
                    * RAREA[i, j]
                PTC[i, j] = (PT[i, j] * DELP[i, j]
                             + (fx[i, j] - fx[i + 1, j]
                                + fy[i, j] - fy[i, j + 1])
                             * RAREA[i, j]) / DELPC[i, j]
                WC[i, j] = (W[i, j] * DELP[i, j]
                            + (fx2[i, j] - fx2[i + 1, j]
                               + fy2[i, j] - fy2[i, j + 1])
                            * RAREA[i, j]) / DELPC[i, j]

    # ---- compute KE (cubed-sphere branch) ----
    ke = _fl(is_ - 1, ie + 1, js - 1, je + 1)
    vort = _fl(is_ - 1, ie + 1, js - 1, je + 1)
    if bounded_domain or grid_type >= 3:  # pragma: no cover
        raise NotImplementedError
    for j in range(js - 1, jep1 + 1):
        for i in range(is_ - 1, iep1 + 1):
            if UA[i, j] > 0.0:
                if i == 1:
                    ke[1, j] = UC[1, j] * SIN_SG[1, j, 1 - 1] \
                        + VF[1, j] * COS_SG[1, j, 1 - 1]
                elif i == npx:
                    ke[i, j] = UC[npx, j] * SIN_SG[npx, j, 1 - 1] \
                        + VF[npx, j] * COS_SG[npx, j, 1 - 1]
                else:
                    ke[i, j] = UC[i, j]
            else:
                if i == 0:
                    ke[0, j] = UC[1, j] * SIN_SG[0, j, 3 - 1] \
                        + VF[1, j] * COS_SG[0, j, 3 - 1]
                elif i == (npx - 1):
                    ke[i, j] = UC[npx, j] * SIN_SG[npx - 1, j, 3 - 1] \
                        + VF[npx, j] * COS_SG[npx - 1, j, 3 - 1]
                else:
                    ke[i, j] = UC[i + 1, j]
    for j in range(js - 1, jep1 + 1):
        for i in range(is_ - 1, iep1 + 1):
            if VA[i, j] > 0.0:
                if j == 1:
                    vort[i, 1] = VC[i, 1] * SIN_SG[i, 1, 2 - 1] \
                        + UF[i, 1] * COS_SG[i, 1, 2 - 1]
                elif j == npy:
                    vort[i, j] = VC[i, npy] * SIN_SG[i, npy, 2 - 1] \
                        + UF[i, npy] * COS_SG[i, npy, 2 - 1]
                else:
                    vort[i, j] = VC[i, j]
            else:
                if j == 0:
                    vort[i, 0] = VC[i, 1] * SIN_SG[i, 0, 4 - 1] \
                        + UF[i, 1] * COS_SG[i, 0, 4 - 1]
                elif j == (npy - 1):
                    vort[i, j] = VC[i, npy] * SIN_SG[i, npy - 1, 4 - 1] \
                        + UF[i, npy] * COS_SG[i, npy - 1, 4 - 1]
                else:
                    vort[i, j] = VC[i, j + 1]

    dt4 = 0.5 * dt2
    for j in range(js - 1, jep1 + 1):
        for i in range(is_ - 1, iep1 + 1):
            ke[i, j] = dt4 * (UA[i, j] * ke[i, j] + VA[i, j] * vort[i, j])

    # ---- circulation on the C grid ----
    fxc = _fl(is_ - 1, ie + 2, js - 1, je + 1)
    fyc = _fl(is_ - 1, ie + 1, js - 1, je + 2)
    for j in range(js - 1, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            fxc[i, j] = UC[i, j] * DXC[i, j]
    for j in range(js, je + 1 + 1):
        for i in range(is_ - 1, ie + 1 + 1):
            fyc[i, j] = VC[i, j] * DYC[i, j]

    vortc = _fl(is_, ie + 1, js, je + 1)
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            vortc[i, j] = fxc[i, j - 1] - fxc[i, j] \
                - fyc[i - 1, j] + fyc[i, j]

    # remove the extra term at the corners
    vortc[1, 1] = vortc[1, 1] + fyc[0, 1]
    vortc[npx, 1] = vortc[npx, 1] - fyc[npx, 1]
    vortc[npx, npy] = vortc[npx, npy] - fyc[npx, npy]
    vortc[1, npy] = vortc[1, npy] + fyc[0, npy]

    # absolute vorticity
    for j in range(js, je + 1 + 1):
        for i in range(is_, ie + 1 + 1):
            vortc[i, j] = FC[i, j] + RAREA_C[i, j] * vortc[i, j]

    # ---- transport absolute vorticity (cubed-sphere branch) ----
    for j in range(js, je + 1):
        for i in range(is_, iep1 + 1):
            if i == 1 or i == npx:
                fy1[i, j] = dt2 * VF[i, j]
            else:
                fy1[i, j] = dt2 * (VF[i, j] - UC[i, j] * COSA_U[i, j]) \
                    / SINA_U[i, j]
            if fy1[i, j] > 0.0:
                fy[i, j] = vortc[i, j]
            else:
                fy[i, j] = vortc[i, j + 1]
    for j in range(js, jep1 + 1):
        if j == 1 or j == npy:
            for i in range(is_, ie + 1):
                fx1[i, j] = dt2 * UF[i, j]
                if fx1[i, j] > 0.0:
                    fx[i, j] = vortc[i, j]
                else:
                    fx[i, j] = vortc[i + 1, j]
        else:
            for i in range(is_, ie + 1):
                fx1[i, j] = dt2 * (UF[i, j] - VC[i, j] * COSA_V[i, j]) \
                    / SINA_V[i, j]
                if fx1[i, j] > 0.0:
                    fx[i, j] = vortc[i, j]
                else:
                    fx[i, j] = vortc[i + 1, j]

    # ---- update the time-centred C-grid winds ----
    for j in range(js, je + 1):
        for i in range(is_, iep1 + 1):
            UC[i, j] = UC[i, j] + fy1[i, j] * fy[i, j] \
                + RDXC[i, j] * (ke[i - 1, j] - ke[i, j])
    for j in range(js, jep1 + 1):
        for i in range(is_, ie + 1):
            VC[i, j] = VC[i, j] - fx1[i, j] * fx[i, j] \
                + RDYC[i, j] * (ke[i, j - 1] - ke[i, j])

    return {
        "delpc": DELPC.a, "ptc": PTC.a, "wc": WC.a,
        "uc": UC.a, "vc": VC.a, "ua": UA.a, "va": VA.a,
        "ut": UT.a, "vt": VT.a, "divg_d": divg_a,
    }
