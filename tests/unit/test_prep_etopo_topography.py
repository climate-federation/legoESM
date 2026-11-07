"""Direct tests for scripts/data/prep_etopo_topography.py (offline)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

_SCRIPT = (Path(__file__).resolve().parents[2]
           / "scripts" / "data" / "prep_etopo_topography.py")
_spec = importlib.util.spec_from_file_location("prep_etopo_topography", _SCRIPT)
prep = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(prep)


def _synthetic_elevation(var="elevation", n_lat=180, n_lon=360, with_time=False):
    """A fine global elevation field: a NH land mass (>0) over part of the grid."""
    lat = np.linspace(-89.5, 89.5, n_lat)
    lon = np.linspace(0.5, 359.5, n_lon)
    # land (positive elevation) in 20-60N and lon 60-180; ocean (negative) elsewhere
    land = ((lat[:, None] >= 20) & (lat[:, None] <= 60)
            & (lon[None, :] >= 60) & (lon[None, :] <= 180))
    elev = np.where(land, 1500.0, -4000.0).astype(np.float64)
    coords = {"lat": lat, "lon": lon}
    if with_time:
        return xr.Dataset({var: (("time", "lat", "lon"), elev[None])},
                          coords={**coords, "time": [0]})
    return xr.Dataset({var: (("lat", "lon"), elev)}, coords=coords)


def test_regrids_to_target_resolution_with_elevation_latlon():
    out = prep.regrid_elevation_to_latlon(_synthetic_elevation(), target_res_deg=2.0)
    assert "elevation" in out and out["elevation"].dims == ("lat", "lon")
    assert out.sizes["lat"] == 90 and out.sizes["lon"] == 180     # 2-deg global
    lat = np.asarray(out["lat"].values)
    assert lat[0] < lat[-1]                                       # ascending
    lon = np.asarray(out["lon"].values)
    assert float(lon.min()) >= 0.0 and float(lon.max()) < 360.0   # [0,360)


def test_land_fraction_is_preserved_roughly():
    # the synthetic land band is 40/180 lat * 120/360 lon ~ 0.074 of the globe
    out = prep.regrid_elevation_to_latlon(_synthetic_elevation(), target_res_deg=1.0)
    land_frac = float(np.mean(np.asarray(out["elevation"].values) > 0.0))
    assert 0.03 < land_frac < 0.15


def test_handles_altitude_var_and_duplicate_lon_endpoint():
    # a global grid with BOTH lon 0 and 360 (the 361-lon ETOPO layout) and the
    # CF name 'altitude' — must auto-detect + dedup the wrapped endpoint, not raise.
    lat = np.linspace(-90.0, 90.0, 181)
    lon = np.linspace(0.0, 360.0, 361)                 # inclusive endpoints -> 0==360 mod
    band = np.where((lat[:, None] >= 20) & (lat[:, None] <= 60), 1000.0, -3000.0)
    elev = np.broadcast_to(band, (181, 361)).copy()
    ds = xr.Dataset({"altitude": (("latitude", "longitude"), elev)},
                    coords={"latitude": lat, "longitude": lon})
    out = prep.regrid_elevation_to_latlon(ds, target_res_deg=2.0)
    assert out["elevation"].dims == ("lat", "lon")
    assert float(np.asarray(out["lon"].values).max()) < 360.0   # endpoint deduped


def test_squeezes_extra_dims_and_explicit_var():
    out = prep.regrid_elevation_to_latlon(
        _synthetic_elevation(var="z", with_time=True), var_name="z", target_res_deg=4.0)
    assert out["elevation"].dims == ("lat", "lon")


def test_missing_elevation_var_raises():
    ds = xr.Dataset({"sst": (("lat", "lon"), np.zeros((4, 8)))},
                    coords={"lat": np.linspace(-80, 80, 4), "lon": np.linspace(0, 350, 8)})
    with pytest.raises(KeyError):
        prep.regrid_elevation_to_latlon(ds)


def test_bad_resolution_raises():
    with pytest.raises(ValueError):
        prep.regrid_elevation_to_latlon(_synthetic_elevation(), target_res_deg=0.0)


def test_build_writes_file(tmp_path):
    src = tmp_path / "src.nc"
    _synthetic_elevation(n_lat=90, n_lon=180).to_netcdf(src)
    out = tmp_path / "topo.nc"
    ret = prep.build_topography(str(src), str(out), target_res_deg=2.0)
    assert ret == str(out) and out.exists()
    with xr.open_dataset(out) as chk:
        assert "elevation" in chk and chk["elevation"].dims == ("lat", "lon")


def test_roundtrip_through_load_real_topography(tmp_path):
    """The written file must be consumable by the model's real-topography loader,
    yielding a genuine land/ocean MIX on the model grid."""
    from legoesm.grids.gaussian import create_gaussian_grid
    from legoesm.grids.topography import TopographyConfig, load_real_topography

    topo = prep.regrid_elevation_to_latlon(_synthetic_elevation(), target_res_deg=1.0)
    path = tmp_path / "topo.nc"
    topo.to_netcdf(path)

    grid = create_gaussian_grid(8)
    phis, f_land = load_real_topography(
        grid, config=TopographyConfig(source="file", path=str(path)))
    f_land = np.asarray(f_land)
    assert f_land.shape == grid.lat2d.shape
    assert float(f_land.min()) >= 0.0 and float(f_land.max()) <= 1.0
    assert 0.0 < float(f_land.mean()) < 1.0                       # real land/ocean mix
    assert np.all(np.isfinite(np.asarray(phis)))


def test_global_source_is_periodic_across_the_lon_seam():
    """#1712: identity at the source resolution must hold in the LAST column
    too, and a seam target interpolates between the last and first columns."""
    import numpy as np
    import xarray as xr
    res = 0.5                                  # cell centres, like ETOPO
    lat = -90 + res / 2 + res * np.arange(int(180 / res))
    lon = -180 + res / 2 + res * np.arange(int(360 / res))
    z = np.random.default_rng(0).normal(0, 1000, (lat.size, lon.size))
    ds = xr.Dataset({"z": (("lat", "lon"), z)}, coords={"lat": lat, "lon": lon})
    out = prep.regrid_elevation_to_latlon(ds, target_res_deg=res)
    raw = ds["z"].assign_coords(lon=lon % 360).sortby("lon").values
    np.testing.assert_allclose(out["elevation"].values, raw, atol=1e-6)

    # lon-only field: the first target of a finer grid (lon res/4) lies
    # between the last source column (359.75 = -0.25) and the first (0.25).
    zl = np.broadcast_to(np.arange(lon.size, dtype=float), (lat.size, lon.size))
    ds2 = xr.Dataset({"z": (("lat", "lon"), zl.copy())},
                     coords={"lat": lat, "lon": lon % 360})
    out2 = prep.regrid_elevation_to_latlon(ds2, target_res_deg=res / 2)
    srt = ds2.sortby("lon")["z"].values[0]
    first, last = srt[0], srt[-1]
    t = float(out2["lon"].values[0])                       # 0.125
    w_first = (t + res / 2) / res
    expect = w_first * first + (1 - w_first) * last
    np.testing.assert_allclose(out2["elevation"].values[10:-10, 0], expect,
                               rtol=0, atol=1e-9)
