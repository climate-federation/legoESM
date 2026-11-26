"""ORCA1 oracle: chlorophyll is floored AFTER time interpolation, as in NEMO.

NEMO 5.0.1 ``tra_qsr`` reads the monthly chlorophyll with ``fld_read``
(linear in time between month centres, raw values) and only then clamps
``zchl = MAX(0.03, MIN(fnow, 10))`` (traqsr.F90:322; the ORCA1 profile
branch CASE(1) uses the same clamp inside LOG at :331).  The OMIP runner floored
every remapped MONTH at 0.03 before interpolating, which changes the result
wherever one bracketing month is below 0.03 and the other above (99 of the
65,087 wet ORCA1 cells at the recorded step differ by more than 1e-9 relative,
up to 2.7 %).

Fixture: 121 wet cells from all 32 ranks of the tier-3 QSR record (57 where
the floor order matters), NEMO's two bracketing raw months and its interpolated
``fnow`` (provenance in ``fixtures/orca1_t3/qsr_chl_floor_order.json``).
The runner's step-START sampling time is a separate defect (Q4 timing, not
this PR): the oracle call is made at NEMO's own evaluation time,
``isecsbc = 2592075 s`` (step 34561 of 75 s), so only the floor order is tested.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
RUNNER = ROOT / "scripts" / "run" / "run_omip_core2.py"
FIX = ROOT / "tests" / "ocean" / "unit" / "fixtures" / "orca1_t3" / "qsr_chl_floor_order"


def _mod():
    spec = importlib.util.spec_from_file_location("_omip_runner_chlfloor", RUNNER)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def fx():
    d = dict(np.load(str(FIX) + ".npz"))
    d["meta"] = json.loads(pathlib.Path(str(FIX) + ".json").read_text())
    return d


def _clim(fx):
    t = fx["meta"]["time"]
    chl12 = np.zeros((12, 1, fx["fnow"].size))
    chl12[t["month_bb"] - 1, 0] = fx["fdta_bb"]
    chl12[t["month_aa"] - 1, 0] = fx["fdta_aa"]
    return chl12, t


def _rel(a, b):
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


def test_chl_matches_nemo_clamped_fnow(fx):
    m = _mod()
    chl12, t = _clim(fx)
    dt = 75.0
    step = int(t["isecsbc_s"] / dt)
    assert step * dt == t["isecsbc_s"]
    i0, i1, w = m._month_interp_weights(step, dt)
    assert (i0 + 1, i1 + 1) == (t["month_bb"], t["month_aa"])
    assert abs(w - t["ztinta"]) <= 1e-15
    assert np.all(fx["fnow"] < 10.0)            # upper clamp inactive here
    got = m.chl_at_step(chl12, step, dt, "linear")[0]
    ref = np.maximum(fx["fnow"], 0.03)          # traqsr.F90:322
    assert _rel(got, ref).max() <= 1e-12, _rel(got, ref).max()


def test_fixture_discriminates_floor_order(fx):
    """Flooring the months first misses NEMO on the binding cells."""
    _, t = _clim(fx)
    w = t["ztinta"]
    pre = np.maximum((1 - w) * np.maximum(fx["fdta_bb"], 0.03)
                     + w * np.maximum(fx["fdta_aa"], 0.03), 0.03)
    bind = np.array([c[3] for c in fx["meta"]["cells_rank_i_j_floorbinds"]])
    assert bind.sum() >= 30
    assert _rel(pre, np.maximum(fx["fnow"], 0.03))[bind].min() > 1e-9


def test_loader_keeps_raw_months_below_floor(tmp_path):
    """The loader hands month_sample the RAW remap (no per-month floor)."""
    xr = pytest.importorskip("xarray")
    lat = np.linspace(-10.0, 10.0, 6)
    lon = np.linspace(0.0, 20.0, 8)
    LON, LAT = np.meshgrid(lon, lat)
    chla = np.full((12,) + LAT.shape, 0.01)
    p = tmp_path / "chl.nc"
    xr.Dataset({"nav_lat": (("y", "x"), LAT), "nav_lon": (("y", "x"), LON),
                "CHLA": (("t", "y", "x"), chla)}).to_netcdf(p)
    tl, tn = np.meshgrid(np.array([-5.0, 5.0]), np.array([5.0, 15.0]), indexing="ij")
    out = _mod().load_nemo_chl_monthly(None, "latlon", tl, tn, chl_file=str(p))
    np.testing.assert_allclose(out, 0.01, rtol=1e-12)


def test_scrip_loader_keeps_raw_months_below_floor(tmp_path, monkeypatch):
    """Same for the oracle-weights branch (codex r1: IDW-only coverage gap)."""
    xr = pytest.importorskip("xarray")
    import legoesm.ocean.coupler.omip2_applicator as app
    monkeypatch.setattr(app, "load_scrip_weights", lambda path: (None, None, 4))
    monkeypatch.setattr(app, "apply_scrip_weights",
                        lambda field, src0, wgt, nw, shape: np.full((331, 360), 0.01))
    p = tmp_path / "chl.nc"
    LAT, LON = np.meshgrid(np.linspace(-80, 80, 5), np.linspace(0, 350, 4), indexing="ij")
    xr.Dataset({"nav_lat": (("y", "x"), LAT), "nav_lon": (("y", "x"), LON),
                "CHLA": (("t", "y", "x"), np.full((12,) + LAT.shape, 0.01))}).to_netcdf(p)
    z = np.zeros((332, 362))
    out = _mod().load_nemo_chl_monthly(None, "tripole", z, z, chl_file=str(p),
                                       chl_remap="nemo_scrip")
    np.testing.assert_allclose(out, 0.01, rtol=1e-12)


def test_none_passes_through():
    assert _mod().chl_at_step(None, 0, 150.0, "linear") is None
