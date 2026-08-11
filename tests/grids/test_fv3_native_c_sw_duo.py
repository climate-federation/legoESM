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
    """PLAIN-conventions gridstruct + state: the duogrid=False arm.

    km=1 corpus migration (2026-08-11): duogrid=True on this plain gs is
    the upstream-impossible pair (fv_arrays.F90:1512) and c_sw refuses
    it; the duo arms below use the ``bounded_state`` fixture instead.
    """
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


@pytest.fixture(scope="module")
def bounded_state():
    """BOUNDED-conventions gridstruct + state: the lane the duo runs
    execute (bounded_domain=True, four corner flags False — upstream
    sets them only at .not.bounded; fv_grid_utils.F90:224)."""
    from legoesm.core.fv3_native_sw_core import Bounds
    from legoesm.grids.fv3_native_gridstruct import (
        analytic_swcore_state,
        build_fv3_native_gridstruct_bounded,
    )
    gs = build_fv3_native_gridstruct_bounded(RES, NG)
    gs["bounded_domain"] = True
    for k in ("sw_corner", "se_corner", "ne_corner", "nw_corner"):
        gs[k] = False
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
def test_duo_c_sw_runs_finite(bounded_state, hydrostatic):
    """Every duo c_sw output is FINITE over the strict compute interior
    (bounds-derived, so NaN/Inf is genuinely detected).  Runs both
    hydrostatic AND non-hydrostatic (the latter exercises the two
    fill_4corners(W) skip gates).  Bounded fixture: duogrid=True
    requires bounded_domain=True (fv_arrays.F90:1512)."""
    gs, st, bd = bounded_state
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


def test_duo_branch_engages(bounded_state, monkeypatch):
    """The duo branch genuinely EXECUTES and its result is CONSUMED.

    km=1 migration rewrite (GLM r1 finding 1): the original plain-vs-duo
    difference count became a two-variable comparison once the lanes
    split (duo only runs on bounded metrics, upstream ties them via
    fv_arrays.F90:1512) — the metrics change alone satisfied the floor,
    so a dead `if duogrid:` branch could pass.  This version pins the
    branch directly: (a) SPY — c_sw(duogrid=True) must call
    d2a2c_vect_duo and divergence_corner_duo exactly once (c_sw imports
    them function-scope, so patching the module attribute intercepts
    the real call); (b) CONSUMPTION — perturbing the spy's ut/vt/uc
    return by 1e-3 must move >=50 genuine (non-sentinel) cells of
    delpc and uc, so the branch's output demonstrably feeds the
    kernel's outputs.  Deleting or short-circuiting the duo branch
    fails (a); routing it to a dead copy fails (b)."""
    import legoesm.core.fv3_native_duo_sw_core as duo_mod

    gs, st, bd = bounded_state
    base = _run(gs, st, bd, duogrid=True)

    calls = {"d2a2c": 0, "divg": 0}
    orig_d2a2c = duo_mod.d2a2c_vect_duo
    orig_divg = duo_mod.divergence_corner_duo

    def spy_d2a2c(*a, **k):
        calls["d2a2c"] += 1
        out = dict(orig_d2a2c(*a, **k))
        for key in ("ut", "vt", "uc"):
            out[key] = np.asarray(out[key], dtype=np.float64) + 1.0e-3
        return out

    def spy_divg(*a, **k):
        calls["divg"] += 1
        return orig_divg(*a, **k)

    monkeypatch.setattr(duo_mod, "d2a2c_vect_duo", spy_d2a2c)
    monkeypatch.setattr(duo_mod, "divergence_corner_duo", spy_divg)
    pert = _run(gs, st, bd, duogrid=True)
    assert calls == {"d2a2c": 1, "divg": 1}
    for k, floor in (("delpc", 50), ("uc", 50)):
        a = np.asarray(base[k], float)
        b = np.asarray(pert[k], float)
        # both finite AND non-sentinel (|.|<1e24 rejects the 1e25 divg init)
        m = (np.abs(a) < 1e24) & (np.abs(b) < 1e24) \
            & np.isfinite(a) & np.isfinite(b)
        assert int((np.abs(a - b)[m] > 1e-9).sum()) >= floor, k


def test_duo_on_unbounded_metrics_is_refused(state):
    """Dispatch-hardening lock for the lane guard: duogrid=True on a
    PLAIN-conventions gridstruct (no bounded_domain key at all) must
    fail LOUDLY, citing fv_arrays.F90:1512.  Distinct contract from
    boundedgs' test_csw_refuses_duo_without_bounded, which flips ONLY
    the flag on an otherwise-bounded gs: this one pins the default-path
    refusal (a context that never heard of the flag), that one pins the
    flag-only refusal.  Direct c_sw call — no wrapper hop is claimed
    (codex km=1 r2 finding 4)."""
    gs, st, bd = state
    _run(gs, st, bd, duogrid=False)          # plain arm runs
    with pytest.raises(ValueError, match=r"fv_arrays\.F90:1512"):
        _run(gs, st, bd, duogrid=True)
