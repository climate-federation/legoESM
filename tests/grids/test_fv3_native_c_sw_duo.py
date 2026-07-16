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


def _interior_mask(bd, shape):
    """Strict compute interior (is..ie, js..je) — a BOUNDS-derived region
    every c_sw output writes, so a NaN/Inf there is a real defect (not
    excluded by a value-derived mask; codex p4c c_sw P2)."""
    lo = 1 - bd.ng
    m = np.zeros(shape, dtype=bool)
    m[bd.is_ - lo:bd.ie - lo + 1, bd.js - lo:bd.je - lo + 1] = True
    return m


@pytest.mark.parametrize("hydrostatic", [True, False])
def test_duo_c_sw_runs_finite(state, hydrostatic):
    """Every duo c_sw output is FINITE over the strict compute interior
    (bounds-derived, so NaN/Inf is genuinely detected).  Runs both
    hydrostatic AND non-hydrostatic (the latter exercises the two
    fill_4corners(W) skip gates)."""
    gs, st, bd = state
    from legoesm.core.fv3_native_sw_core import c_sw
    w = np.zeros_like(st["delp"]) if hydrostatic else \
        (0.1 * np.asarray(st["pt"], float))
    out = c_sw(delp=st["delp"], pt=st["pt"], w=w, u=st["u"], v=st["v"],
               gs=gs, bd=bd, npx=RES + 1, npy=RES + 1, dt2=112.5, nord=1,
               hydrostatic=hydrostatic, dord4=True, grid_type=0, duogrid=True)
    keys = ("delpc", "ptc", "uc", "vc", "ua", "va", "divg_d")
    if not hydrostatic:
        keys = keys + ("wc",)
    for k in keys:
        a = np.asarray(out[k], dtype=np.float64)
        mask = _interior_mask(bd, a.shape)
        assert np.isfinite(a[mask]).all(), f"{k}: non-finite in duo c_sw"


def test_duo_branch_engages(state):
    """The duo branch genuinely changes the result vs plain — count only
    GENUINE differences (both sides non-sentinel; codex p4c c_sw P2 — the
    1e25 divg_d init must not inflate the count)."""
    gs, st, bd = state
    plain = _run(gs, st, bd, duogrid=False)
    duo = _run(gs, st, bd, duogrid=True)
    for k, floor in (("delpc", 50), ("uc", 50), ("vc", 50), ("divg_d", 50)):
        a = np.asarray(plain[k], float)
        b = np.asarray(duo[k], float)
        # both finite AND non-sentinel (|.|<1e24 rejects the 1e25 divg init)
        m = (np.abs(a) < 1e24) & (np.abs(b) < 1e24) \
            & np.isfinite(a) & np.isfinite(b)
        assert int((np.abs(a - b)[m] > 1e-9).sum()) >= floor, k
