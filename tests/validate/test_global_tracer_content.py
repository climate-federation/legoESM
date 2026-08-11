"""Direct tests for scripts/validate/ocean_fidelity/global_tracer_content.py.

The probe's job is to discriminate a conservation defect (total salt falls)
from vertical redistribution (total fixed, bins trade).  Both behaviours are
exercised on synthetic columns with a KNOWN answer -- the probe-validation
rule: run the instrument on a case whose real value you already know before
using it on the unknown one.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "global_tracer_content",
    _ROOT / "scripts/validate/ocean_fidelity/global_tracer_content.py")
m = importlib.util.module_from_spec(_SPEC)
sys.modules["global_tracer_content"] = m
_SPEC.loader.exec_module(m)


def _mesh(nlev=4, nj=3, ni=2, dz=10.0):
    e1t = np.full((nj, ni), 2.0)
    e2t = np.full((nj, ni), 3.0)
    e3t = np.full((nlev, nj, ni), dz)
    tmask = np.ones((nlev, nj, ni))
    return e1t, e2t, e3t, tmask


def test_constant_salinity_gives_exact_content():
    e1t, e2t, e3t, tmask = _mesh()
    S = np.full(e3t.shape, 35.0)
    T = np.full(e3t.shape, 10.0)
    eta = np.zeros(e1t.shape)
    r = m.tracer_content(T, S, eta, e1t, e2t, e3t, tmask)
    vol = 2.0 * 3.0 * 10.0 * 4 * 3 * 2
    assert r["volume_m3"] == pytest.approx(vol, rel=1e-14)
    assert r["salt_content_psu_m3"] == pytest.approx(35.0 * vol, rel=1e-14)
    assert r["mean_S_psu"] == pytest.approx(35.0, abs=1e-12)


def test_vertical_redistribution_conserves_total_but_moves_bins():
    e1t, e2t, e3t, tmask = _mesh(nlev=4)
    eta = np.zeros(e1t.shape)
    T = np.zeros(e3t.shape)
    S0 = np.full(e3t.shape, 35.0)
    # Move 1 psu worth from level 0 to level 3 in every column: total fixed.
    S1 = S0.copy()
    S1[0] -= 1.0
    S1[3] += 1.0
    gdept = np.array([5.0, 15.0, 25.0, 35.0])
    edges = (0.0, 10.0, 30.0, np.inf)
    old_edges = m._DEPTH_BIN_EDGES_M
    m._DEPTH_BIN_EDGES_M = edges
    try:
        r0 = m.tracer_content(T, S0, eta, e1t, e2t, e3t, tmask, gdept)
        r1 = m.tracer_content(T, S1, eta, e1t, e2t, e3t, tmask, gdept)
    finally:
        m._DEPTH_BIN_EDGES_M = old_edges
    assert r1["salt_content_psu_m3"] == pytest.approx(
        r0["salt_content_psu_m3"], rel=1e-14)
    b0, b1 = r0["salt_by_depth_bin_psu_m3"], r1["salt_by_depth_bin_psu_m3"]
    assert b1["0-10m"] < b0["0-10m"]          # surface bin lost salt
    assert b1["30-infm"] > b0["30-infm"]      # deep bin gained it


def test_salt_loss_is_visible_in_the_total():
    e1t, e2t, e3t, tmask = _mesh()
    eta = np.zeros(e1t.shape)
    T = np.zeros(e3t.shape)
    S0 = np.full(e3t.shape, 35.0)
    S1 = S0 * 0.9
    r0 = m.tracer_content(T, S0, eta, e1t, e2t, e3t, tmask)
    r1 = m.tracer_content(T, S1, eta, e1t, e2t, e3t, tmask)
    assert r1["salt_content_psu_m3"] == pytest.approx(
        0.9 * r0["salt_content_psu_m3"], rel=1e-14)


def test_zstar_dilation_scales_volume_not_mean_salinity():
    e1t, e2t, e3t, tmask = _mesh(nlev=4, dz=10.0)   # H = 40 m everywhere
    T = np.zeros(e3t.shape)
    S = np.full(e3t.shape, 35.0)
    eta = np.full(e1t.shape, 4.0)                    # +10% column dilation
    r = m.tracer_content(T, S, eta, e1t, e2t, e3t, tmask)
    assert r["volume_m3"] == pytest.approx(1.1 * r["volume_ref_m3"], rel=1e-14)
    assert r["mean_S_psu"] == pytest.approx(35.0, abs=1e-12)


def test_dry_cells_are_excluded():
    e1t, e2t, e3t, tmask = _mesh(nlev=2, nj=1, ni=2, dz=10.0)
    tmask[:, 0, 1] = 0.0                             # column 1 is land
    T = np.zeros(e3t.shape)
    S = np.full(e3t.shape, 35.0)
    S[:, 0, 1] = 999.0                               # must not leak in
    eta = np.zeros(e1t.shape)
    r = m.tracer_content(T, S, eta, e1t, e2t, e3t, tmask)
    vol = 2.0 * 3.0 * 10.0 * 2                       # one wet column, 2 levels
    assert r["volume_m3"] == pytest.approx(vol, rel=1e-14)
    assert r["mean_S_psu"] == pytest.approx(35.0, abs=1e-12)


def test_nan_in_wet_cell_is_fatal():
    e1t, e2t, e3t, tmask = _mesh()
    T = np.zeros(e3t.shape)
    S = np.full(e3t.shape, 35.0)
    S[1, 1, 1] = np.nan
    eta = np.zeros(e1t.shape)
    with pytest.raises(SystemExit):
        m.tracer_content(T, S, eta, e1t, e2t, e3t, tmask)


def test_native_slice_strips_ghost_row_and_overlap_columns():
    a = np.arange(332 * 362, dtype=np.float64).reshape(332, 362)
    n = m._native(a)
    assert n.shape == (331, 360)
    assert n[0, 0] == a[0, 1]
    assert n[-1, -1] == a[330, 360]
