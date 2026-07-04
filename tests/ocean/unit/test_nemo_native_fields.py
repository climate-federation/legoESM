"""nemo_native_fields loaders: eORCA halo embed + native/regrid paths."""

from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")

from legoesm.ocean.forcing.nemo_native_fields import (
    embed_orca_interior,
    load_nemo_monthly_init_ts,
    load_nemo_sss_restoring_climatology,
)


def _model_coords(ny_i=6, nx_i=8):
    """Model tripole-like coords (interior + ORCA halos)."""
    lat_i = np.linspace(-70, 89, ny_i)[:, None] * np.ones((1, nx_i))
    lon_i = np.ones((ny_i, 1)) * np.linspace(0, 350, nx_i)[None, :]
    lat = np.zeros((ny_i + 1, nx_i + 2))
    lon = np.zeros_like(lat)
    lat[:-1, 1:-1] = lat_i
    lon[:-1, 1:-1] = lon_i
    lat[:-1, 0] = lat_i[:, -1]; lon[:-1, 0] = lon_i[:, -1]
    lat[:-1, -1] = lat_i[:, 0]; lon[:-1, -1] = lon_i[:, 0]
    lat[-1] = lat[-2]; lon[-1] = lon[-2]
    return lat, lon, lat_i, lon_i


def test_embed_orca_interior_layout():
    src = np.arange(6 * 8, dtype=float).reshape(6, 8)
    out = embed_orca_interior(src, 7, 10)
    np.testing.assert_array_equal(out[:-1, 1:-1], src)
    np.testing.assert_array_equal(out[:-1, 0], src[:, -1])
    np.testing.assert_array_equal(out[:-1, -1], src[:, 0])
    np.testing.assert_array_equal(out[-1], out[-2])
    with pytest.raises(ValueError, match="does not match"):
        embed_orca_interior(src, 8, 10)


def _write_sss_nc(path, lat_i, lon_i, ny_i, nx_i):
    with netCDF4.Dataset(path, "w") as ds:
        ds.createDimension("time_counter", 12)
        ds.createDimension("y", ny_i)
        ds.createDimension("x", nx_i)
        v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
        v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
        v = ds.createVariable("presalt", "f8", ("time_counter", "y", "x"))
        v[:] = 30.0 + np.arange(12)[:, None, None] \
            + np.arange(ny_i)[None, :, None] * 0.1


def test_sss_climatology_native_embed(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    p = tmp_path / "sss.nc"
    _write_sss_nc(p, lat_i, lon_i, 6, 8)
    out = load_nemo_sss_restoring_climatology(
        str(p), lat, lon, np.ones(lat.shape, bool))
    assert out.shape == (12, 7, 10)
    # native embed: interior equals the file, months indexed correctly
    np.testing.assert_allclose(out[3][:-1, 1:-1][2, 4], 30 + 3 + 0.2)
    assert np.isfinite(out).all()


def test_sss_climatology_regrid_path(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    p = tmp_path / "sss.nc"
    # source on a DIFFERENT (coarser) grid -> NearestWetRegridder path
    _write_sss_nc(p, lat_i[::2, ::2], lon_i[::2, ::2], 3, 4)
    out = load_nemo_sss_restoring_climatology(
        str(p), lat, lon, np.ones(lat.shape, bool))
    assert out.shape == (12, 7, 10)
    assert np.isfinite(out).all()


def test_monthly_init_native(tmp_path):
    lat, lon, lat_i, lon_i = _model_coords()
    nlev = 5
    tp, sp = tmp_path / "t.nc", tmp_path / "s.nc"
    for path, var, base in ((tp, "contemp", 10.0), (sp, "presalt", 34.0)):
        with netCDF4.Dataset(path, "w") as ds:
            ds.createDimension("time_counter", 12)
            ds.createDimension("deptht", nlev)
            ds.createDimension("y", 6)
            ds.createDimension("x", 8)
            v = ds.createVariable("nav_lat", "f8", ("y", "x")); v[:] = lat_i
            v = ds.createVariable("nav_lon", "f8", ("y", "x")); v[:] = lon_i
            v = ds.createVariable(var, "f8",
                                  ("time_counter", "deptht", "y", "x"))
            fld = (base + np.arange(12)[:, None, None, None]
                   + np.arange(nlev)[None, :, None, None] * 0.01)
            fld = np.broadcast_to(fld, (12, nlev, 6, 8)).copy()
            fld[:, -1, 0, 0] = np.nan          # a below-bathy hole
            v[:] = fld
    T, S = load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                     n_levels=nlev, month=2)
    assert T.shape == (7, 10, nlev) and S.shape == T.shape
    assert np.isfinite(T).all() and np.isfinite(S).all()
    np.testing.assert_allclose(T[:-1, 1:-1][3, 3, 0], 10.0 + 1)   # month 2
    # column fill: the hole inherited the level above
    np.testing.assert_allclose(T[0, 1, nlev - 1], T[0, 1, nlev - 2])
    with pytest.raises(ValueError, match="levels"):
        load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                  n_levels=nlev + 1)
    with pytest.raises(ValueError, match="month"):
        load_nemo_monthly_init_ts(str(tp), str(sp), lat, lon,
                                  n_levels=nlev, month=0)
