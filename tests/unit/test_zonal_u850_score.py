"""Pure helpers of the #1712 zonal-mean u850 scorer."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SCRIPT = (Path(__file__).resolve().parents[2] / "scripts" / "validate" / "amip_bias"
           / "zonal_u850_score.py")


def _mod():
    spec = importlib.util.spec_from_file_location("zonal_u850_score", _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


LAT = np.arange(-87.5, 90.0, 5.0)


def test_band_rmse_selects_30_70_and_weights_by_cos_lat():
    m = _mod()
    d = np.zeros_like(LAT)
    d[LAT == 32.5] = 3.0
    d[LAT == 67.5] = 1.0
    d[LAT == 27.5] = 100.0  # outside the band: must not count
    d[LAT == 72.5] = 100.0
    band = (LAT >= 30) & (LAT <= 70)
    w = np.cos(np.deg2rad(LAT[band]))
    expect = np.sqrt((9 * np.cos(np.deg2rad(32.5)) + np.cos(np.deg2rad(67.5))) / w.sum())
    assert m.band_rmse(d, LAT, (30.0, 70.0)) == pytest.approx(expect, rel=1e-12)
    assert band.sum() == 8


def test_band_rmse_2d_equals_1d_for_zonally_uniform_field():
    m = _mod()
    d = np.sin(np.deg2rad(LAT))
    assert m.band_rmse(np.tile(d[:, None], (1, 72)), LAT, (20.0, 90.0)) == pytest.approx(
        m.band_rmse(d, LAT, (20.0, 90.0)), rel=1e-12)


def test_verdict_thresholds():
    m = _mod()
    # exact 20 % / 5 % are not representable boundaries in float; test either side
    assert m.verdict(1.0, 0.799)[1] == "CONFIRM"
    assert m.verdict(1.0, 0.801)[1] == "INCONCLUSIVE"
    assert m.verdict(1.0, 0.949)[1] == "INCONCLUSIVE"
    assert m.verdict(1.0, 0.951)[1] == "REFUTE"
    assert m.verdict(1.0, 1.20)[1] == "REFUTE"


def test_global_mean_is_cos_weighted():
    m = _mod()
    x = np.tile(np.where(LAT > 0, 1.0, 0.0)[:, None], (1, 72))
    assert m.global_mean(x, LAT) == pytest.approx(0.5, rel=1e-12)


def test_window_requires_days_1_to_30(tmp_path):
    netCDF4 = pytest.importorskip("netCDF4")
    m = _mod()
    f = tmp_path / "ua.nc"
    with netCDF4.Dataset(f, "w") as d:
        d.createDimension("time", 29)
        d.createDimension("bnds", 2)
        d.createDimension("plev", 1)
        d.createDimension("lat", 2)
        d.createDimension("lon", 3)
        d.createVariable("time_bnds", "f8", ("time", "bnds"))[:] = np.stack(
            [np.arange(29), np.arange(1, 30)], 1)
        d.createVariable("plev", "f8", ("plev",))[:] = [85000.0]
        d.createVariable("lat", "f8", ("lat",))[:] = [0.0, 5.0]
        d.createVariable("lon", "f8", ("lon",))[:] = [0.0, 5.0, 10.0]
        d.createVariable("ua", "f4", ("time", "plev", "lat", "lon"))[:] = 1.0
    with pytest.raises(SystemExit, match="days 1-30"):
        m.window_mean_daily(str(f), "ua")
