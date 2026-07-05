"""Tests for scripts/plot/plot_amip_cmor_diagnostics.py — the systematic AMIP
CMOR-Amon diagnostics plotter.  Verifies the pure compute core (area-weighted
global means + TOA budget from synthetic Amon files) and that the figure
renders, so the plot path can't silently rot."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

_MOD_PATH = (pathlib.Path(__file__).resolve().parents[2]
             / "scripts" / "plot" / "plot_amip_cmor_diagnostics.py")
_spec = importlib.util.spec_from_file_location("plot_amip_cmor_diagnostics", _MOD_PATH)
plotmod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(plotmod)


def _write_amon(cmor_amon: pathlib.Path, var: str, field2d: np.ndarray, lat, lon):
    """Write a minimal CMOR-Amon-style NetCDF: (time=1, lat, lon)."""
    cmor_amon.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(field2d[None, :, :], dims=("time", "lat", "lon"),
                      coords={"time": [0.0], "lat": lat, "lon": lon}, name=var)
    da.to_dataset().to_netcdf(cmor_amon / f"{var}_Amon_test_gn.nc")


def test_weights_sum_to_one_and_are_nonuniform():
    w = plotmod._sinlat_area_weights(18, 36)
    assert w.shape == (18, 36)
    np.testing.assert_allclose(w.sum(), 1.0, atol=1e-12)
    # polar rows carry LESS area than equatorial rows (sin-latitude bands).
    assert w[0, 0] < w[9, 0]


def test_constant_field_global_mean_is_the_constant(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", np.full((nlat, nlon), 288.0), lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    gm, ref, unit = diag["global_means"]["tas"]
    assert gm == pytest.approx(288.0, abs=1e-9)   # weights sum to 1
    assert unit == "K"


def test_hemispheric_field_area_weighted_mean_is_half(tmp_path):
    """A field that is 1 in the SH and 0 in the NH integrates to 0.5 under the
    (hemispherically symmetric) sin-latitude weights — the exactness check."""
    nlat, nlon = 40, 20
    lat = np.linspace(-88.0, 88.0, nlat)
    lon = np.linspace(0, 342, nlon)
    field = np.where(lat[:, None] < 0.0, 1.0, 0.0) * np.ones((1, nlon))
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", field, lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    assert diag["global_means"]["tas"][0] == pytest.approx(0.5, abs=1e-2)


def test_toa_budget_and_albedo(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    _write_amon(cmor, "tas", np.full((nlat, nlon), 288.0), lat, lon)
    _write_amon(cmor, "rsdt", np.full((nlat, nlon), 340.0), lat, lon)
    _write_amon(cmor, "rsut", np.full((nlat, nlon), 100.0), lat, lon)
    _write_amon(cmor, "rlut", np.full((nlat, nlon), 239.0), lat, lon)
    diag = plotmod.compute_amip_diagnostics(tmp_path)
    b = diag["budget"]
    assert b["R_TOA"] == pytest.approx(340.0 - 100.0 - 239.0, abs=1e-6)
    assert b["albedo"] == pytest.approx(100.0 / 340.0, abs=1e-6)


def test_missing_tas_raises(tmp_path):
    (tmp_path / "cmor" / "Amon").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        plotmod.compute_amip_diagnostics(tmp_path)


def test_figure_renders(tmp_path):
    nlat, nlon = 18, 36
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(5, 355, nlon)
    cmor = tmp_path / "cmor" / "Amon"
    for v, val in [("tas", 288.0), ("pr", 3.0e-5), ("rsut", 100.0),
                   ("rsdt", 340.0), ("rlut", 239.0), ("clt", 60.0)]:
        _write_amon(cmor, v, np.full((nlat, nlon), val), lat, lon)
    out = plotmod.plot_amip_cmor_diagnostics(tmp_path, "unit", tmp_path / "fig.png")
    assert out.exists() and out.stat().st_size > 0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
