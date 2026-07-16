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
verified-faithful TRANSLATION of the authoritative Fortran — bit-exact on
identical inputs (``test_fv3_native_duo_phase4c``), and its extraction is
byte-identical to sw_core.F90:2345-2447 (SHA-256).

SCOPE CAVEAT: this is not yet a full-DUO-PIPELINE certification.  The duo
c_sw feeds ``divergence_corner_duo`` (and the KE/vorticity path) with
ua/va from ``d2a2c_vect``'s dg-initialized cross-face branch, which
differs from the plain ``c_sw`` ua/va the current oracle uses at panel
edges/corners.  Porting the duo ``d2a2c_vect`` (and the no-corner-fill /
no-``sin_sg``-edge duo c_sw body) is the remaining phase-4c work before
the production duo path can claim end-to-end FV3 fidelity.

Conventions match ``fv3_native_sw_core`` / ``fv3_native_d_sw`` (``fort``
views, ``Bounds``).
"""

from __future__ import annotations

import numpy as np
from legoesm.core.fv3_native_sw_core import Bounds, _fa
from legoesm.grids.fv3_native_gridstruct import fort


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
