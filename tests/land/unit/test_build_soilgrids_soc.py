"""Unit tests for the gridded-SOC obs builder
(``scripts/data/build_soilgrids_soc.py``).

Exercises the PURE reduction / unit-conversion / assembly / regrid helpers offline on
synthetic arrays -- the networked ISRIC WCS wrapper (``fetch_soilgrids_coverage``) is NOT
exercised -- plus the conservative-regrid GLOBAL-INTEGRAL conservation property, the WCS
raster -> lat/lon grid mapping, and a NetCDF write->read round-trip in the schema the
validator consumes.
"""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np

_REPO = pathlib.Path(__file__).resolve().parents[3]
_PY = _REPO / "scripts" / "data" / "build_soilgrids_soc.py"


def _mod():
    spec = importlib.util.spec_from_file_location("build_soilgrids_soc", _PY)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


# ---------------------------------------------------------------------------
# SoilGrids mapped-unit -> stock conversion
# ---------------------------------------------------------------------------
def test_soilgrids_layer_stock_matches_hand_calc():
    m = _mod()
    # soc 200 dg/kg = 20 gC/kg; bdod 130 cg/cm3 = 1300 kg/m3; cfvo 100 cm3/dm3 = 0.1 frac.
    # stock = 20[gC/kg] * 1300[kg/m3] * (1-0.1) * 0.05[m] = 1170 gC/m2 = 1.17 kgC/m2.
    got = m.soilgrids_layer_stock(200.0, 130.0, 100.0, 0.05)
    np.testing.assert_allclose(got, 1.17, rtol=1e-12)


def test_soilgrids_layer_stock_scales_linearly_and_with_depth():
    m = _mod()
    base = m.soilgrids_layer_stock(200.0, 130.0, 0.0, 0.10)
    # double the depth -> double the stock; double soc -> double the stock.
    np.testing.assert_allclose(m.soilgrids_layer_stock(200.0, 130.0, 0.0, 0.20), 2 * base)
    np.testing.assert_allclose(m.soilgrids_layer_stock(400.0, 130.0, 0.0, 0.10), 2 * base)


def test_soilgrids_cfvo_reduces_stock_and_full_gravel_is_zero():
    m = _mod()
    no_rock = m.soilgrids_layer_stock(200.0, 130.0, 0.0, 0.10)
    some_rock = m.soilgrids_layer_stock(200.0, 130.0, 300.0, 0.10)     # 30% coarse
    np.testing.assert_allclose(some_rock, no_rock * 0.7, rtol=1e-12)
    # 1000 cm3/dm3 = 100% coarse fragments -> no fine-earth carbon.
    np.testing.assert_allclose(m.soilgrids_layer_stock(200.0, 130.0, 1000.0, 0.10), 0.0)


def test_soilgrids_layer_stock_nan_propagates():
    m = _mod()
    out = m.soilgrids_layer_stock(np.array([200.0, np.nan]), 130.0, 0.0, 0.05)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_assemble_ocs_sums_layers_nan_aware():
    m = _mod()
    # 3 cells x 2 depth layers.
    l0 = np.array([1.0, np.nan, np.nan])
    l1 = np.array([2.0, 3.0, np.nan])
    ocs = m.assemble_ocs_0_100([l0, l1])
    np.testing.assert_allclose(ocs[0], 3.0)          # 1 + 2
    np.testing.assert_allclose(ocs[1], 3.0)          # only the finite layer
    assert np.isnan(ocs[2])                          # all layers missing -> NaN


# ---------------------------------------------------------------------------
# surfdata organic-column integral (reuses shared column_soc)
# ---------------------------------------------------------------------------
def test_integrate_organic_column_matches_manual():
    m = _mod()
    organic = np.array([[[10.0, 6.0, 2.0]]])          # (1 lat, 1 lon, 3 lev) kg OM/m3
    dz = np.array([0.1, 0.2, 0.3])
    col = m.integrate_organic_column(organic, dz, 0.58)
    expect = 0.58 * (10.0 * 0.1 + 6.0 * 0.2 + 2.0 * 0.3)   # kgC/m2
    np.testing.assert_allclose(col[0, 0], expect, rtol=1e-12)


# ---------------------------------------------------------------------------
# conservative regrid conserves the GLOBAL area-integral
# ---------------------------------------------------------------------------
def _area_integral(field, lat, lon):
    """Reference sin(lat)-weighted, periodic-lon area integral (test's own measure)."""
    def edges(c):
        mid = 0.5 * (c[:-1] + c[1:])
        return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])
    le = edges(lat)
    dsin = np.abs(np.sin(np.deg2rad(le[1:])) - np.sin(np.deg2rad(le[:-1])))
    lo = edges(lon)
    dlon = np.abs(lo[1:] - lo[:-1])
    return float(np.sum(field * dsin[:, None] * dlon[None, :]))


def test_regrid_constant_field_is_exact():
    m = _mod()
    src_lat = np.linspace(-88.0, 88.0, 45)
    src_lon = np.linspace(1.25, 358.75, 72)
    tgt_lat = np.linspace(-85.0, 85.0, 18)
    tgt_lon = np.linspace(5.0, 355.0, 24)
    field = np.full((src_lat.size, src_lon.size), 7.3)
    out = m.regrid_to_target(field, src_lat, src_lon, tgt_lat, tgt_lon)
    np.testing.assert_allclose(out, 7.3, rtol=1e-10)   # constant preserved everywhere


def test_regrid_conserves_area_weighted_mean():
    m = _mod()
    # Source and target cover the SAME near-global lat span and full 0-360 lon, so
    # they tile one domain and only the discretization differs.  A first-order
    # CONSERVATIVE regrid preserves the area-weighted GLOBAL MEAN (the quantity the
    # SOC bias/RMSE depend on) -- this is the conservation guard codex asks for.
    src_lat = np.linspace(-89.0, 89.0, 90)
    src_lon = np.linspace(2.0, 358.0, 90)          # edges span [0, 360]
    tgt_lat = np.linspace(-89.0, 89.0, 30)         # SAME span, coarser
    tgt_lon = np.linspace(6.0, 354.0, 30)          # edges span [0, 360]
    rng = np.random.default_rng(0)
    field = (2.0 + np.cos(np.deg2rad(src_lat))[:, None]
             + 0.3 * rng.standard_normal((src_lat.size, src_lon.size)) ** 2)
    out = m.regrid_to_target(field, src_lat, src_lon, tgt_lat, tgt_lon)

    def _wmean(f, la, lo):
        return _area_integral(f, la, lo) / _area_integral(np.ones_like(f), la, lo)

    np.testing.assert_allclose(_wmean(out, tgt_lat, tgt_lon),
                               _wmean(field, src_lat, src_lon), rtol=1e-3)


# ---------------------------------------------------------------------------
# WCS raster -> (lat, lon) grid mapping (no rasterio; grid from request bbox)
# ---------------------------------------------------------------------------
def test_raster_to_grid_orientation_and_nodata():
    m = _mod()
    # north-up raster (top row = +lat), value increasing downward (south).  Tests the
    # pure bbox->lat/lon mapping + nodata + N->S flip (the imageio decode is a thin
    # wrapper exercised for real by the WCS fetch).
    ny, nx = 4, 6
    raster = np.arange(ny * nx, dtype=np.float64).reshape(ny, nx)
    raster[0, 0] = -9999.0                                    # nodata sentinel
    fld, lat, lon = m._raster_to_grid(
        raster, (-180.0, 180.0, -90.0, 90.0), nodata=-9999.0)
    # returned ascending lat; lon within [-180,180]; nodata -> NaN.
    assert lat[0] < lat[-1] and lon.min() > -180.0 and lon.max() < 180.0
    assert lat.size == ny and lon.size == nx
    assert np.isnan(fld[-1, 0])                               # top-left nodata after flip
    # the top (north) row had the smallest values -> after flip it is the LAST row.
    assert np.nanmean(fld[-1, :]) < np.nanmean(fld[0, :])


# ---------------------------------------------------------------------------
# NetCDF write -> read round-trip in the validator's schema
# ---------------------------------------------------------------------------
def test_write_obs_netcdf_roundtrip(tmp_path):
    import xarray as xr
    m = _mod()
    lat = np.linspace(-88.0, 88.0, 8)
    lon = np.linspace(2.5, 357.5, 10)
    soc = np.abs(np.cos(np.deg2rad(lat)))[:, None] * np.ones((1, lon.size)) * 12.0
    out = tmp_path / "soc.nc"
    m.write_obs_netcdf(out, soc, lat, lon, dict(product="surfdata_organic", depth_cm=300.0))
    ds = xr.open_dataset(out)
    try:
        assert ds["soc"].dims == ("lat", "lon")
        assert ds["soc"].attrs["units"] == "kgC/m2"
        assert ds.attrs["product"] == "surfdata_organic"
        np.testing.assert_allclose(ds["soc"].values, soc)
    finally:
        ds.close()


def test_write_obs_netcdf_rejects_shape_mismatch(tmp_path):
    m = _mod()
    lat = np.linspace(-80.0, 80.0, 5)
    lon = np.linspace(0.0, 350.0, 6)
    try:
        m.write_obs_netcdf(tmp_path / "bad.nc", np.zeros((4, 4)), lat, lon, {})
        raise AssertionError("expected ValueError on grid/shape mismatch")
    except ValueError:
        pass
