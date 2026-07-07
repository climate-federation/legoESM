"""Direct tests for scripts/data/prep_subgrid_orography.py (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "prep_subgrid_orography.py")
_spec = importlib.util.spec_from_file_location("prep_subgrid_orography", _SCRIPT)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _synthetic_elevation(n_lat=180, n_lon=360, var="elevation"):
    """1-deg global field: ocean, a flat plateau, and a rough checkerboard range.

    - ocean (-4000 m) everywhere by default
    - flat plateau: 1000 m over lat 20..60N, lon 60..120E  (subgrid stddev 0)
    - checkerboard range: alternating 0/1600 m over lat 20..60N, lon 200..260E
      (population stddev exactly 800 m in any fully-interior block)
    """
    lat = np.linspace(-89.5, 89.5, n_lat)
    lon = np.linspace(0.5, 359.5, n_lon)
    elev = np.full((n_lat, n_lon), -4000.0)
    plateau = ((lat[:, None] >= 20) & (lat[:, None] <= 60)
               & (lon[None, :] >= 60) & (lon[None, :] <= 120))
    elev = np.where(plateau, 1000.0, elev)
    rough = ((lat[:, None] >= 20) & (lat[:, None] <= 60)
             & (lon[None, :] >= 200) & (lon[None, :] <= 260))
    ii = np.arange(n_lat)[:, None] + np.arange(n_lon)[None, :]
    checker = np.where(ii % 2 == 0, 1600.0, 0.0)
    elev = np.where(rough, checker, elev)
    return xr.Dataset({var: (("lat", "lon"), elev)},
                      coords={"lat": lat, "lon": lon})


def _sso(block_deg=4.0):
    return prep.subgrid_orography_stddev(
        _synthetic_elevation(), fine_res_deg=1.0, block_deg=block_deg)


def test_output_layout_and_nonnegative():
    out = _sso()
    assert "SSO_STDH" in out and out["SSO_STDH"].dims == ("lat", "lon")
    assert out.sizes["lat"] == 45 and out.sizes["lon"] == 90   # 4-deg global
    lat = np.asarray(out["lat"].values)
    lon = np.asarray(out["lon"].values)
    assert lat[0] < lat[-1] and float(lon.min()) >= 0.0 and float(lon.max()) < 360.0
    assert float(out["SSO_STDH"].min()) >= 0.0
    assert out["SSO_STDH"].attrs["units"] == "m"


def _block_value(out, lat0, lon0):
    da = out["SSO_STDH"]
    return float(da.sel(lat=lat0, lon=lon0, method="nearest").values)


def test_ocean_and_flat_plateau_have_zero_stddev():
    out = _sso()
    # weighted-moment roundoff: a constant block yields O(1e-13), not exact 0
    assert _block_value(out, -40.0, 300.0) == pytest.approx(0.0, abs=1e-9)
    assert _block_value(out, 40.0, 90.0) == pytest.approx(0.0, abs=1e-9)


def test_checkerboard_range_has_exact_stddev():
    out = _sso()
    # interior block of the 0/1600 m checkerboard: population stddev == 800 m
    assert _block_value(out, 40.0, 230.0) == pytest.approx(800.0, abs=1e-9)


def test_coastal_block_between_land_and_ocean_is_intermediate():
    out = _sso()
    # block straddling the plateau's southern coast: mix of 1000 m and 0 m
    v = _block_value(out, 20.0, 90.0)
    assert 0.0 < v < 800.0


def test_stddev_is_area_weighted_by_cos_lat():
    # A polar block whose terrain varies ONLY with latitude: the poleward
    # (low cos(lat), low weight) rows carry the anomaly, so the area-weighted
    # stddev must be strictly below the unweighted np.std of the same block.
    n_lat, n_lon = 180, 360
    lat = np.linspace(-89.5, 89.5, n_lat)
    lon = np.linspace(0.5, 359.5, n_lon)
    elev = np.zeros((n_lat, n_lon))
    elev[-2:, :] = 1600.0                      # two most-poleward rows (88-90N)
    ds = xr.Dataset({"elevation": (("lat", "lon"), elev)},
                    coords={"lat": lat, "lon": lon})
    out = prep.subgrid_orography_stddev(ds, fine_res_deg=1.0, block_deg=4.0)
    v_weighted = float(out["SSO_STDH"].values[-1, 0])   # northernmost block
    block = elev[-4:, :4]
    v_unweighted = float(np.std(block))
    assert 0.0 < v_weighted < v_unweighted


def test_rejects_non_integer_or_degenerate_block_factor():
    ds = _synthetic_elevation()
    with pytest.raises(ValueError, match="integer multiple"):
        prep.subgrid_orography_stddev(ds, fine_res_deg=1.0, block_deg=2.5)
    with pytest.raises(ValueError, match="integer multiple"):
        prep.subgrid_orography_stddev(ds, fine_res_deg=1.0, block_deg=1.0)


def test_cli_roundtrip_writes_loader_compatible_file(tmp_path):
    src = tmp_path / "elev.nc"
    _synthetic_elevation().to_netcdf(src)
    out_path = tmp_path / "sso.nc"
    rc = prep.main(["--input", str(src), "--out", str(out_path),
                    "--fine-res-deg", "1.0", "--block-deg", "4.0"])
    assert rc == 0 and out_path.exists()
    with xr.open_dataset(out_path) as ds:
        assert "SSO_STDH" in ds.data_vars          # load_subgrid_orography default
        assert set(ds["SSO_STDH"].dims) == {"lat", "lon"}
        assert float(ds["SSO_STDH"].max()) > 0.0
