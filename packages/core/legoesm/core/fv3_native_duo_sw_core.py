"""FV3-native DUO-GRID c_sw pieces — loop-faithful port, phase 4c.

The duo-grid FV3 (Mouallem/Xi-Chen, Zenodo 8327578, the authoritative
``atmos_cubed_sphere-symmetryclean`` tree) takes DIFFERENT c_sw branches
from the plain FV3 certified in phase 4a (``fv3_native_sw_core``): with
``flagstruct%duogrid`` it calls ``divergence_corner_duo`` instead of
``divergence_corner``, skips the ``fill2/fill_4corners`` corner fills
(the duo halos carry real cross-face data), and takes the
``bounded_domain .or. grid_type>=3 .or. duogrid`` KE/vorticity branch (no
``sin_sg`` panel-edge special-case).  legoESM's PRODUCTION solver runs
these duo branches, so full duo-grid FV3 fidelity certifies THESE, not the
plain ones.

This module ports the duo-specific routines loop-faithfully (index-exact,
plain numpy, Fortran statement order).  ``divergence_corner_duo`` is a
verified-faithful TRANSLATION of the authoritative Fortran — bit-exact
(uint64 words) on the ONE stored C12, ``grid_type=0`` cubed-sphere fixture
(``grid_type>3`` is not ported), with the fixture's ``input_sha256``
ENFORCED against its stored arrays (``test_fv3_native_duo_phase4c``), and
its extraction pinned byte-identical to sw_core.F90:2345-2447 (SHA-256).

SCOPE: the duo c_sw feeds ``divergence_corner_duo`` (and the KE/vorticity
path) with ua/va from ``d2a2c_vect``'s dg-initialized cross-face branch,
which differs from the plain ``c_sw`` ua/va at panel edges/corners.  That
duo ``d2a2c_vect`` is now ported + certified here as ``d2a2c_vect_duo``
(bit-exact vs the authoritative Fortran DUO branch,
``test_fv3_native_d2a2c_duo``) — closing the ua/va scope caveat.  The
remaining phase-4c work is threading these certified duo leaves
(``d2a2c_vect_duo`` + ``divergence_corner_duo``) into a full no-corner-fill
/ no-``sin_sg``-edge duo c_sw body for end-to-end pipeline certification.

Conventions match ``fv3_native_sw_core`` / ``fv3_native_d_sw`` (``fort``
views, ``Bounds``).
"""

from __future__ import annotations

import numpy as np
from legoesm.core.fv3_native_sw_core import Bounds, _fa
from legoesm.grids.fv3_native_gridstruct import fort

# --- d2a2c 4th-order interpolation coefficients (sw_core.F90:53-54) ---
_A1_D2A2C = 0.5625     # 9/16   coeff-ok: FV3 published d2a2c/PPM interpolant
_A2_D2A2C = -0.0625    # -1/16  coeff-ok: FV3 published d2a2c/PPM interpolant
_BIG_NUMBER = 1.0e30   # sw_core.F90:43 corner-init sentinel (guard)


def divergence_corner_duo(u: np.ndarray, v: np.ndarray,
                          ua: np.ndarray, va: np.ndarray,
                          gs: dict, bd: Bounds, npx: int, npy: int,
                          grid_type: int = 0) -> np.ndarray:
    """sw_core.F90 divergence_corner_duo (verbatim; symmetryclean tree).

    Returns divg_d (B-node array, isd:ied+1, jsd:jed+1).  Like
    divergence_corner_nest's interior formula (no is2/ie1 clamp) PLUS the
    duo panel-edge zeroing/quartering that removes the cube-seam
    divergence the duo halos would otherwise double-count.

    u (isd:ied, jsd:jed+1); v (isd:ied+1, jsd:jed); ua/va (isd:ied,
    jsd:jed).  FV3 cos_sg/sin_sg positions 1..4 map to python 0..3
    (W,S,E,N): cos_sg(.,.,4)->[3] N, (.,.,2)->[1] S, (.,.,3)->[2] E,
    (.,.,1)->[0] W.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    UA = fort(ua, isd, jsd)
    VA = fort(va, isd, jsd)
    SIN_SG = fort(gs["sin_sg"], isd, jsd)
    COS_SG = fort(gs["cos_sg"], isd, jsd)
    DXC = fort(gs["dxc"], isd, jsd)
    DYC = fort(gs["dyc"], isd, jsd)
    RAREA_C = fort(gs["rarea_c"], isd, jsd)

    DIVG = _fa(bd, 1, 1, 1.0e25)   # divg_d = 1.e25 initial

    if grid_type > 3:  # pragma: no cover - cubed-sphere oracle
        raise NotImplementedError("grid_type > 3 not ported")

    uf = fort(np.full((ied - isd + 1, jed + 1 - jsd + 1), np.nan), isd, jsd)
    vf = fort(np.full((ied + 1 - isd + 1, jed - jsd + 1), np.nan), isd, jsd)

    for j in range(jsd + 1, jed + 1):
        for i in range(isd, ied + 1):
            uf[i, j] = (U[i, j] - 0.25 * (VA[i, j - 1] + VA[i, j])
                        * (COS_SG[i, j - 1, 4 - 1] + COS_SG[i, j, 2 - 1])) \
                * DYC[i, j] * 0.5 \
                * (SIN_SG[i, j - 1, 4 - 1] + SIN_SG[i, j, 2 - 1])

    for j in range(jsd, jed + 1):
        for i in range(isd + 1, ied + 1):
            vf[i, j] = (V[i, j] - 0.25 * (UA[i - 1, j] + UA[i, j])
                        * (COS_SG[i - 1, j, 3 - 1] + COS_SG[i, j, 1 - 1])) \
                * DXC[i, j] * 0.5 \
                * (SIN_SG[i - 1, j, 3 - 1] + SIN_SG[i, j, 1 - 1])

    for j in range(jsd + 1, jed + 1):
        for i in range(isd + 1, ied + 1):
            DIVG[i, j] = (vf[i, j - 1] - vf[i, j]
                          + uf[i - 1, j] - uf[i, j]) * RAREA_C[i, j]
            # duo panel-edge zeroing (the seam B-nodes)
            if is_ == 1 and i == is_:
                DIVG[i, j] = 0.0
            if (ie + 1) == npx and i == ie + 1:
                DIVG[i, j] = 0.0
            if js == 1 and j == 1:
                DIVG[i, j] = 0.0
            if je + 1 == npx and j == je + 1:
                DIVG[i, j] = 0.0
            # next-to-seam quartering (verbatim; note the upstream j-tests
            # use npx not npy — harmless on the square single tile)
            if is_ == 1 and i == is_ + 1:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if (ie + 1) == npx and i == ie:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if js == 1 and j == 1 + 1:
                DIVG[i, j] = 0.25 * DIVG[i, j]
            if je + 1 == npx and j == je:
                DIVG[i, j] = 0.25 * DIVG[i, j]

    return DIVG.a


def d2a2c_vect_duo(u: np.ndarray, v: np.ndarray, gs: dict, bd: Bounds,
                   npx: int, npy: int, dord4: bool = True,
                   grid_type: int = 0) -> dict:
    """sw_core.F90 d2a2c_vect, DUO branch (``gridstruct%dg%is_initialized``).

    D->A->C vector reconstruction: D-grid (u, v) -> A-grid (ua, va) ->
    C-grid (uc, vc) + contravariant (ut, vt).  The duo/bounded branch runs
    the INTERIOR 4th-order formulas over the full data domain and SKIPS
    every panel-edge / cube-corner special-case (the duo halos carry real
    cross-face winds) — exactly like ``divergence_corner_duo``.

    This is the ua/va source the duo c_sw actually feeds
    ``divergence_corner_duo`` (the plain ``c_sw`` used by the phase-4a
    reference takes d2a2c_vect's NON-duo branch, so its ua/va differ at
    panel edges/corners — the scope caveat this routine closes).

    Returns ``{ua, va, uc, vc, ut, vt}`` (numpy).  u (isd:ied, jsd:jed+1);
    v (isd:ied+1, jsd:jed).  ``dord4``/``id`` is UNUSED on the duo branch
    (its interior covers the full domain id-independently; the ``id``
    stencil widening only applies to the non-duo else-branch); kept for
    signature parity.  FV3 cosa/rsin fields are read from ``gs``.
    """
    is_, ie, js, je = bd.is_, bd.ie, bd.js, bd.je
    isd, ied, jsd, jed = bd.isd, bd.ied, bd.jsd, bd.jed
    a1, a2 = _A1_D2A2C, _A2_D2A2C

    if grid_type >= 3:  # pragma: no cover - cubed-sphere oracle
        raise NotImplementedError("grid_type >= 3 not ported")

    U = fort(u, isd, jsd)
    V = fort(v, isd, jsd)
    COSA_S = fort(gs["cosa_s"], isd, jsd)
    RSIN2 = fort(gs["rsin2"], isd, jsd)
    COSA_U = fort(gs["cosa_u"], isd, jsd)
    RSIN_U = fort(gs["rsin_u"], isd, jsd)
    COSA_V = fort(gs["cosa_v"], isd, jsd)
    RSIN_V = fort(gs["rsin_v"], isd, jsd)

    utmp = _fa(bd, 0, 0, _BIG_NUMBER)   # (isd:ied, jsd:jed)
    vtmp = _fa(bd, 0, 0, _BIG_NUMBER)
    UA = _fa(bd, 0, 0, _BIG_NUMBER)     # (isd:ied, jsd:jed)
    VA = _fa(bd, 0, 0, _BIG_NUMBER)
    UC = _fa(bd, 1, 0, _BIG_NUMBER)     # (isd:ied+1, jsd:jed)
    VC = _fa(bd, 0, 1, _BIG_NUMBER)     # (isd:ied,   jsd:jed+1)
    UT = _fa(bd, 0, 0, _BIG_NUMBER)     # (isd:ied, jsd:jed)
    VT = _fa(bd, 0, 0, _BIG_NUMBER)

    # ---- D -> A (duo interior; sw_core.F90:3421-3454) ----
    for i in range(isd, ied + 1):
        for j in range(jsd + 1, jed):          # do j=jsd+1,jed-1
            utmp[i, j] = a2 * (U[i, j - 1] + U[i, j + 2]) \
                + a1 * (U[i, j] + U[i, j + 1])
    for i in range(isd, ied + 1):
        utmp[i, jsd] = U[i, jsd + 1]           # 0.5*(u(j+1)+u(j+1))
        utmp[i, jed] = U[i, jed + 1]
    for j in range(jsd, jed + 1):
        for i in range(isd + 1, ied):          # do i=isd+1,ied-1
            vtmp[i, j] = a2 * (V[i - 1, j] + V[i + 2, j]) \
                + a1 * (V[i, j] + V[i + 1, j])
        vtmp[isd, j] = V[isd + 1, j]           # 0.5*(v(i+1)+v(i+1))
        vtmp[ied, j] = V[ied + 1, j]
    for i in range(isd, ied + 1):
        for j in range(jsd, jed + 1):
            UA[i, j] = (utmp[i, j] - vtmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]
            VA[i, j] = (vtmp[i, j] - utmp[i, j] * COSA_S[i, j]) * RSIN2[i, j]

    # ---- A -> C  X-dir (duo ifirst=is-1, ilast=ie+2; 3558-3563) ----
    for j in range(js - 1, je + 2):            # do j=js-1,je+1
        for i in range(is_ - 1, ie + 3):       # do i=is-1,ie+2
            UC[i, j] = a2 * (utmp[i - 2, j] + utmp[i + 1, j]) \
                + a1 * (utmp[i - 1, j] + utmp[i, j])
            UT[i, j] = (UC[i, j] - V[i, j] * COSA_U[i, j]) * RSIN_U[i, j]

    # ---- A -> C  Y-dir (duo interior for every j; 3690-3693) ----
    for j in range(js - 1, je + 3):            # do j=js-1,je+2
        for i in range(is_ - 1, ie + 2):       # do i=is-1,ie+1
            VC[i, j] = a2 * (vtmp[i, j - 2] + vtmp[i, j + 1]) \
                + a1 * (vtmp[i, j - 1] + vtmp[i, j])
            VT[i, j] = (VC[i, j] - U[i, j] * COSA_V[i, j]) * RSIN_V[i, j]

    return {"ua": UA.a, "va": VA.a, "uc": UC.a, "vc": VC.a,
            "ut": UT.a, "vt": VT.a}
