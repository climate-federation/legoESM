"""window_diff: paired-arm window means from a partial-month CMOR file and the
restart sidecar.

Synthetic inputs with a known window mean must be reproduced exactly; a
sidecar/file orientation mismatch, a wrong cadence and a two-month bucket must
be refused rather than produce a number.
"""
from __future__ import annotations

import importlib.util
import json
import pathlib

import numpy as np
import pytest
import xarray as xr

_P = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias" / "window_diff.py"
LAT = np.array([-45.0, 45.0])
LON = np.array([0.0, 120.0, 240.0])


@pytest.fixture
def wd():
    spec = importlib.util.spec_from_file_location("window_diff", _P)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _sidecar(path, sums, counts, months=((1979, 3),)):
    data_2d, arrays = [], {}
    for i, (var, s) in enumerate(sums.items()):
        for y, m in months:
            key = f"arr_{i}_{m}"
            data_2d.append([y, m, var, counts[var], key])
            arrays[f"monthly.{key}"] = s
    man = {"version": 1, "type": "SpatialMonthlyAccumulator", "nlat": 2,
           "nlon": 3, "nlev": 1, "data_2d": data_2d, "data_3d": []}
    np.savez(path, **{"monthly.__manifest__": np.array(json.dumps(man))}, **arrays)


def _amon(run_dir, var, mean, tag="197903-197903"):
    d = run_dir / "cmor" / "Amon"
    d.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset({var: (("time", "lat", "lon"), mean[None])},
                    coords={"time": [74.5], "lat": LAT, "lon": LON})
    ds.to_netcdf(d / f"{var}_Amon_legoESM-1-0_amip_r1i1p1f1_gn_{tag}.nc")


def _pattern(a, b):
    """(2, 3) field with a hemispheric asymmetry so a flipped lat is visible."""
    return np.array([[a, a, a], [b, b, b]], dtype=float)


def test_window_mean_recovers_the_arm_days(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    # 20 days of history at (100, 200) per day, then 6 arm days at (130, 260)
    hist = {"rsut": 20 * _pattern(100.0, 200.0), "rlut": 20 * _pattern(10.0, 20.0)}
    _sidecar(run / "cmor_accum_day_0080.npz", hist, {"rsut": 20, "rlut": 20})
    _amon(run, "rsut", (20 * _pattern(100.0, 200.0) + 6 * _pattern(130.0, 260.0)) / 26)
    _amon(run, "rlut", (20 * _pattern(10.0, 20.0) + 6 * _pattern(40.0, 50.0)) / 26)
    m, lat = wd.window_means("arm", 80, 86)
    np.testing.assert_allclose(m["rsut"], _pattern(130.0, 260.0), rtol=1e-12)
    np.testing.assert_allclose(m["rlut"], _pattern(40.0, 50.0), rtol=1e-12)
    assert m["rsut__n"] == 6
    np.testing.assert_allclose(lat, LAT)
    # cos-weighted band mean of a symmetric pair is the plain mean
    assert wd.band_mean(m["rsut"], lat, -90, 90) == pytest.approx(195.0)


def test_orientation_mismatch_is_refused(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0080.npz", {"rsut": 20 * _pattern(100.0, 200.0)}, {"rsut": 20})
    _amon(run, "rsut", _pattern(200.0, 100.0))          # lat flipped
    with pytest.raises(SystemExit, match="orientation"):
        wd.window_means("arm", 80, 86)


def test_cadence_is_asserted(wd, tmp_path, monkeypatch):
    monkeypatch.setattr(wd, "ROOT", tmp_path)
    run = tmp_path / "parent"; run.mkdir()
    s = {"rsut": np.ones((2, 3))}
    _sidecar(run / "cmor_accum_day_0070.npz", s, {"rsut": 10})
    _sidecar(run / "cmor_accum_day_0080.npz", s, {"rsut": 20})
    wd.check_cadence("parent", (70, 80))                 # 1/day: OK
    _sidecar(run / "cmor_accum_day_0080.npz", s, {"rsut": 30})
    with pytest.raises(SystemExit, match="cadence"):
        wd.check_cadence("parent", (70, 80))


def test_two_open_months_are_refused(wd, tmp_path, monkeypatch):
    run = tmp_path / "arm"; run.mkdir()
    _sidecar(run / "cmor_accum_day_0080.npz", {"rsut": np.ones((2, 3))}, {"rsut": 10},
             months=((1979, 3), (1979, 4)))
    with pytest.raises(SystemExit, match="months in the bucket"):
        wd.sidecar_sums(run / "cmor_accum_day_0080.npz")
