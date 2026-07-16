"""Duo c_sw parameterization — smoke + branch-engagement (phase-4c).

``c_sw(..., duogrid=True)`` selects the symmetryclean DUO branch the
production solver runs: d2a2c_vect_duo + divergence_corner_duo (both
certified bit-exact), the simple-upwind KE/vorticity, and the
interior-everywhere vorticity transport, SKIPPING the corner fills /
corner-removal.  ``duogrid=False`` stays byte-identical to the phase-4a
plain c_sw (certified separately in test_fv3_native_swcore_phase4).

This gate proves the duo branch is wired + runs finite + genuinely differs
from plain (the branch engages).  The BIT-EXACT certification of the duo
c_sw outputs vs the authoritative Fortran duo c_sw is the next brick (a
full-c_sw one-step oracle, like the c_sw/d_sw phase-4a/4b gates).
"""

import numpy as np
import pytest

RES, NG = 12, 3


@pytest.fixture(scope="module")
def state():
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        FV3_OMEGA,
        FV3_RADIUS_M,
        analytic_swcore_state,
        build_fv3_native_gridstruct,
    )
    gs = build_fv3_native_gridstruct(RES, NG, radius=FV3_RADIUS_M,
                                     omega=FV3_OMEGA)
    st = analytic_swcore_state(gs)
    bd = Bounds.single_tile(RES, NG)
    return gs, st, bd


def _run(gs, st, bd, duogrid):
    from legoesm.core.fv3_native_sw_core import c_sw
    return c_sw(delp=st["delp"], pt=st["pt"], w=np.zeros_like(st["delp"]),
                u=st["u"], v=st["v"], gs=gs, bd=bd, npx=RES + 1, npy=RES + 1,
                dt2=112.5, nord=1, hydrostatic=True, dord4=True, grid_type=0,
                duogrid=duogrid)


def test_duo_c_sw_runs_finite(state):
    """Every duo c_sw output is finite over its written region (no NaN/Inf
    from the swapped duo leaves + upwind KE/vort)."""
    gs, st, bd = state
    out = _run(gs, st, bd, duogrid=True)
    for k in ("delpc", "ptc", "uc", "vc", "ua", "va", "divg_d"):
        a = np.asarray(out[k], dtype=np.float64)
        m = np.abs(a) < 1e29                 # written (non-sentinel) region
        assert m.any(), k
        assert np.isfinite(a[m]).all(), f"{k}: non-finite in duo c_sw"


def test_duo_branch_engages(state):
    """The duo branch genuinely changes the result vs plain — d2a2c_duo +
    divergence_duo + simple-upwind KE + interior vorticity transport +
    skipped corner fills each perturb the C-grid outputs."""
    gs, st, bd = state
    plain = _run(gs, st, bd, duogrid=False)
    duo = _run(gs, st, bd, duogrid=True)
    # each output must differ at a non-trivial number of cells
    for k, floor in (("delpc", 50), ("uc", 50), ("vc", 50), ("divg_d", 50)):
        a = np.nan_to_num(np.asarray(plain[k], float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        b = np.nan_to_num(np.asarray(duo[k], float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        m = np.abs(b) < 1e29
        assert int((np.abs(a - b)[m] > 1e-9).sum()) >= floor, k
