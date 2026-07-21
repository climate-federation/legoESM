"""Unit tests for the ERA5 land-forcing climatology builder
(scripts/data/build_land_forcing_climatology.py).

Exercises the PURE ``reduce_arco_to_climatology`` (all unit conversions +
``netrad = ssr + str`` + Dataset assembly) on synthetic arrays -- the networked
``_fetch_arco`` GCS wrapper is NOT exercised offline -- and proves a NetCDF
write -> read round-trip that the global-carbon-IC driver's ``_load_monthly_climatology``
(``build_global_carbon_ic.py --climatology``) actually accepts with its DEFAULT
``--clim-*-var`` flags.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

from legoesm import constants

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PY = _REPO / "scripts" / "data" / "build_land_forcing_climatology.py"
_DRV = _REPO / "scripts" / "data" / "build_global_carbon_ic.py"

_SEC_PER_HOUR = 3600.0  # test-local reference for the ERA5 accumulation conversion


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _mod():
    return _load(_PY, "build_land_forcing_climatology")


def _drv():
    return _load(_DRV, "build_global_carbon_ic")


def _synthetic_raw(nlat=4, nlon=5):
    """Distinct, nonzero per-cell synthetic monthly-mean ARCO raw arrays.

    Native ERA5 units: 2m_temperature [K]; total_precipitation [m] per hourly
    accumulation; the three radiation fields [J/m^2] per hourly accumulation.
    Every entry is distinct so a swapped variable / axis would be caught.  ERA5's
    descending latitude (90..-90) and 0..360 longitude are used deliberately.
    """
    m = _mod()
    r = np.arange(1, 12 * nlat * nlon + 1, dtype=np.float64).reshape(12, nlat, nlon)
    raw = {
        m._VAR_T2M: 250.0 + 0.1 * r,          # [K]
        m._VAR_TP: 1.0e-4 * r,                # [m/hr accum]
        m._VAR_SSRD: 1.0e6 * r,               # [J/m^2/hr]  down SW
        m._VAR_SSR: 0.8e6 * r,                # [J/m^2/hr]  net SW (>0)
        m._VAR_STR: -1.0e5 * r,               # [J/m^2/hr]  net LW (<0)
    }
    lat = np.array([67.5, 22.5, -22.5, -67.5])[:nlat]   # descending, ERA5-like
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)  # 0,72,...,288
    return m, raw, lat, lon, r


def _to_cell_time(field_12_lat_lon):
    """(12, nlat, nlon) -> (nlat*nlon, 12) row-major, as the driver returns."""
    a = np.asarray(field_12_lat_lon)
    n12, nlat, nlon = a.shape
    return a.transpose(1, 2, 0).reshape(nlat * nlon, n12)


def test_reduce_units_netrad_and_structure():
    m, raw, lat, lon, r = _synthetic_raw()
    ds = m.reduce_arco_to_climatology(raw, lat, lon)

    # --- variable names, dims, coords match the driver contract ---
    assert set(ds.data_vars) == {"tas", "pr", "rsds", "netrad"}
    for v in ("tas", "pr", "rsds", "netrad"):
        assert ds[v].dims == ("time", "lat", "lon")
        assert ds[v].shape == (12, lat.size, lon.size)
    assert ds.sizes["time"] == 12
    np.testing.assert_allclose(np.asarray(ds["lat"].values), lat)
    np.testing.assert_allclose(np.asarray(ds["lon"].values), lon)
    np.testing.assert_array_equal(np.asarray(ds["time"].values),
                                  np.arange(1, 13))

    # --- unit conversions + net-radiation sum ---
    np.testing.assert_allclose(ds["tas"].values, raw[m._VAR_T2M])          # K passthrough
    np.testing.assert_allclose(
        ds["pr"].values, raw[m._VAR_TP] * constants.rho_water / _SEC_PER_HOUR)
    np.testing.assert_allclose(ds["rsds"].values, raw[m._VAR_SSRD] / _SEC_PER_HOUR)
    np.testing.assert_allclose(
        ds["netrad"].values, (raw[m._VAR_SSR] + raw[m._VAR_STR]) / _SEC_PER_HOUR)


def test_reduce_netrad_is_ssr_plus_str_exactly():
    """netrad must be ERA5's own (net SW + net LW)/3600 -- NOT an albedo proxy."""
    m, raw, lat, lon, r = _synthetic_raw()
    ds = m.reduce_arco_to_climatology(raw, lat, lon)
    expected = (raw[m._VAR_SSR] + raw[m._VAR_STR]) / _SEC_PER_HOUR
    np.testing.assert_allclose(ds["netrad"].values, expected)
    # net LW is negative here, so netrad must sit BELOW net SW / 3600.
    assert np.all(ds["netrad"].values < raw[m._VAR_SSR] / _SEC_PER_HOUR)


def test_reduce_validates_shape_and_missing_var():
    m, raw, lat, lon, r = _synthetic_raw()

    bad = dict(raw)
    bad[m._VAR_T2M] = raw[m._VAR_T2M][:, :, :-1]      # wrong lon extent
    try:
        m.reduce_arco_to_climatology(bad, lat, lon)
        raise AssertionError("expected ValueError on wrong-shaped raw array")
    except ValueError:
        pass

    missing = dict(raw)
    del missing[m._VAR_SSR]
    try:
        m.reduce_arco_to_climatology(missing, lat, lon)
        raise AssertionError("expected KeyError on missing ARCO variable")
    except KeyError:
        pass

    lat2d = np.stack([lat, lat])                      # 2-D coord -> reject
    try:
        m.reduce_arco_to_climatology(raw, lat2d, lon)
        raise AssertionError("expected ValueError on 2-D lat")
    except ValueError:
        pass


def test_netcdf_roundtrip_driver_accepts(tmp_path):
    """Write the NetCDF and read it back THROUGH the driver's own
    ``_load_monthly_climatology`` with DEFAULT flags -- the real contract."""
    m, raw, lat, lon, r = _synthetic_raw()
    ds = m.reduce_arco_to_climatology(raw, lat, lon)
    nc = tmp_path / "clim.nc"
    ds.to_netcdf(nc)

    # Reload straight through the driver: default --clim-*-var flags must name
    # tas/pr/rsds/netrad and coords lat/lon, so NO extra flags are supplied.
    drv = _drv()
    args = drv.build_arg_parser().parse_args(["--climatology", str(nc)])
    # Regrid onto the SAME grid the file carries -> conservative regrid is identity.
    t, pr, sw, nr = drv._load_monthly_climatology(args, lat, lon)

    ncell = lat.size * lon.size
    for arr in (t, pr, sw, nr):
        assert arr.shape == (ncell, 12)

    np.testing.assert_allclose(t, _to_cell_time(raw[m._VAR_T2M]), rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(
        pr, _to_cell_time(raw[m._VAR_TP] * constants.rho_water / _SEC_PER_HOUR),
        rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(
        sw, _to_cell_time(raw[m._VAR_SSRD] / _SEC_PER_HOUR), rtol=1e-9, atol=1e-6)
    np.testing.assert_allclose(
        nr, _to_cell_time((raw[m._VAR_SSR] + raw[m._VAR_STR]) / _SEC_PER_HOUR),
        rtol=1e-9, atol=1e-6)


def test_reduce_samples_to_months():
    """The production month-averaging collapses year/day/hour and keeps months."""
    m = _mod()
    ny, nd, nh, nlat, nlon = 2, 3, 4, 2, 2
    base = np.arange(12 * nlat * nlon, dtype=np.float64).reshape(12, nlat, nlon) * 7.0 + 5.0
    yoff = np.array([-1.0, 1.0])                    # zero-mean over years
    doff = np.array([-2.0, 0.0, 2.0])              # zero-mean over days
    hoff = np.array([-3.0, -1.0, 1.0, 3.0])        # zero-mean over hours
    full = (base[None, :, None, None, :, :]
            + yoff[:, None, None, None, None, None]
            + doff[None, None, :, None, None, None]
            + hoff[None, None, None, :, None, None])
    flat = full.reshape(ny * 12 * nd * nh, nlat, nlon)   # fetch loop order (y,m,d,h)
    out = m._reduce_samples_to_months(flat, ny, nd, nh, nlat, nlon)
    assert out.shape == (12, nlat, nlon)
    np.testing.assert_allclose(out, base)          # offsets average away -> base


def test_drop_pole_rows():
    m = _mod()
    lat = np.array([90.0, 60.0, 0.0, -60.0, -90.0])
    raw = {name: (np.arange(12 * lat.size * 3, dtype=np.float64)
                  .reshape(12, lat.size, 3) + k)
           for k, name in enumerate(m._FETCH_VARS)}
    raw2, lat2 = m._drop_pole_rows(raw, lat)
    np.testing.assert_array_equal(lat2, np.array([60.0, 0.0, -60.0]))
    for name in m._FETCH_VARS:
        np.testing.assert_array_equal(raw2[name], raw[name][:, 1:4, :])
    # no-op when no row sits on a pole (interior-only grid unchanged)
    interior = np.array([60.0, 30.0, 0.0, -30.0, -60.0])
    raw3, lat3 = m._drop_pole_rows(raw, interior)
    np.testing.assert_array_equal(lat3, interior)


def test_pole_rows_are_the_regrid_hazard_and_trim_fixes_it(tmp_path):
    """Non-vacuous self-test for the pole fix: a pole-INCLUSIVE field regridded
    with the driver's own conservative_regrid_latlon yields NaN pole rows; after
    _drop_pole_rows the produced file regrids cleanly (all finite) THROUGH the
    driver's _load_monthly_climatology."""
    from legoesm.grids.regridding import conservative_regrid_latlon

    m = _mod()
    lat = np.array([90.0, 60.0, 30.0, 0.0, -30.0, -60.0, -90.0])   # ERA5-like, poles in
    lon = np.linspace(0.0, 360.0, 5, endpoint=False)
    r = np.arange(1, 12 * lat.size * lon.size + 1, dtype=np.float64).reshape(
        12, lat.size, lon.size)
    raw = {m._VAR_T2M: 250.0 + 0.1 * r, m._VAR_TP: 1.0e-4 * r,
           m._VAR_SSRD: 1.0e6 * r, m._VAR_SSR: 0.8e6 * r, m._VAR_STR: -1.0e5 * r}

    # (1) hazard: pole rows regrid to NaN (identity regrid onto the same grid).
    field = raw[m._VAR_T2M][0]                                     # (nlat, nlon)
    out = conservative_regrid_latlon(field, lat, lon, lat, lon)
    assert not np.isfinite(out[0]).any() and not np.isfinite(out[-1]).any()
    assert np.isfinite(out[1:-1]).all()                           # interior is fine

    # (2) fix: drop poles, assemble, and read back THROUGH the driver -> all finite.
    raw2, lat2 = m._drop_pole_rows(raw, lat)
    assert lat2.size == lat.size - 2 and not np.any(np.abs(lat2) >= 90.0 - 1e-6)
    ds = m.reduce_arco_to_climatology(raw2, lat2, lon)
    nc = tmp_path / "clim_polefix.nc"
    ds.to_netcdf(nc)

    drv = _drv()
    args = drv.build_arg_parser().parse_args(["--climatology", str(nc)])
    t, pr, sw, nr = drv._load_monthly_climatology(args, lat2, lon)
    for arr in (t, pr, sw, nr):
        assert arr.shape == (lat2.size * lon.size, 12)
        assert np.isfinite(arr).all()                            # no pole NaNs


def test_driver_celsius_flag_offset_matches_freeze_point(tmp_path):
    """A sanity cross-check that the driver reads our Kelvin tas as-is by default
    (the --clim-t-in-celsius path would instead add constants.T_freeze)."""
    m, raw, lat, lon, r = _synthetic_raw()
    ds = m.reduce_arco_to_climatology(raw, lat, lon)
    nc = tmp_path / "clim.nc"
    ds.to_netcdf(nc)

    drv = _drv()
    args_k = drv.build_arg_parser().parse_args(["--climatology", str(nc)])
    t_k, *_ = drv._load_monthly_climatology(args_k, lat, lon)
    args_c = drv.build_arg_parser().parse_args(
        ["--climatology", str(nc), "--clim-t-in-celsius"])
    t_c, *_ = drv._load_monthly_climatology(args_c, lat, lon)

    # Our file is Kelvin: the default read passes through; the celsius flag would
    # (wrongly, for this file) add T_freeze -> the offset must equal T_freeze.
    np.testing.assert_allclose(t_c - t_k, constants.T_freeze, rtol=0, atol=1e-9)
