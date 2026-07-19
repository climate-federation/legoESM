"""Unit tests for the gridded-obs -> per-archetype SOC target reducer.

Pure NumPy (the ``load_gridded_soc`` NetCDF round-trip needs ``xarray``); the reducer
is a FIXED calibration target built once, never inside a JAX-traced model step.

The headline check (``test_per_archetype_reduces_to_known_value``) is the one the task
requires: a synthetic gridded field reduces to a HAND-COMPUTED per-archetype value.
"""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
import pytest

from legoesm.land.carbon.soc_gridded_target import (
    per_archetype_soc_from_grid,
    sample_grid_to_cells,
)
from legoesm.land.carbon.soc_observations import per_archetype_cover_weighted_mean


# --- a small synthetic obs grid shared by several tests ------------------------
# 2x2 grid; field[i_lat, i_lon]:
#           lon=100  lon=200
#   lat=10    1.0      2.0
#   lat=20    3.0      4.0
_OBS_LAT = np.array([10.0, 20.0])
_OBS_LON = np.array([100.0, 200.0])
_FIELD = np.array([[1.0, 2.0], [3.0, 4.0]])
# The four grid-centre cover cells (in a deliberately NON-row-major order to prove the
# lookup is by coordinate, not by flatten order).
_CELL_LAT = np.array([20.0, 10.0, 10.0, 20.0])
_CELL_LON = np.array([200.0, 100.0, 200.0, 100.0])
_CELL_SOC = np.array([4.0, 1.0, 2.0, 3.0])   # field sampled at those centres


def test_sample_grid_exact_centres():
    out = sample_grid_to_cells(_FIELD, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON)
    npt.assert_allclose(out, _CELL_SOC, rtol=1e-12)


def test_sample_grid_robust_to_descending_lat():
    # Store the field north->south (descending lat); the reducer sorts internally.
    lat_desc = _OBS_LAT[::-1]
    field_desc = _FIELD[::-1, :]
    out = sample_grid_to_cells(field_desc, lat_desc, _OBS_LON, _CELL_LAT, _CELL_LON)
    npt.assert_allclose(out, _CELL_SOC, rtol=1e-12)


def test_sample_grid_longitude_frame_offset_is_circular():
    # obs lon in 0..360; a cover cell given in the -180..180 frame must still match.
    obs_lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.array([[10.0, 20.0, 30.0, 40.0]])       # 1 lat row
    obs_lat = np.array([0.0])
    cell_lat = np.array([0.0, 0.0, 0.0])
    # -90 -> 270 (col 3 = 40); -180/180 -> 180 (col 2 = 30); 355 -> seam -> 0 (col 0 = 10)
    cell_lon = np.array([-90.0, -180.0, 355.0])
    # 1-row (singleton-lat) grid needs an explicit tol_deg (no derivable lat spacing);
    # 45 deg > the 5 deg seam distance for the 355->0 wrap, < the 90 deg lon spacing.
    out = sample_grid_to_cells(field, obs_lat, obs_lon, cell_lat, cell_lon, tol_deg=45.0)
    npt.assert_allclose(out, [40.0, 30.0, 10.0], rtol=1e-12)


def test_sample_grid_offgrid_cell_is_nan():
    # A cell far from any obs centre (lat 50 vs grid {10,20}, spacing 10, tol 6) -> NaN.
    out = sample_grid_to_cells(
        _FIELD, _OBS_LAT, _OBS_LON, np.array([50.0]), np.array([100.0]))
    assert np.isnan(out[0])


def test_sample_grid_nan_obs_propagates():
    field = _FIELD.copy()
    field[0, 0] = np.nan                                # ocean / missing at (10,100)
    out = sample_grid_to_cells(field, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON)
    # _CELL at (10,100) is index 1 in the non-row-major cell ordering.
    assert np.isnan(out[1])
    npt.assert_allclose(out[[0, 2, 3]], _CELL_SOC[[0, 2, 3]], rtol=1e-12)


def test_per_archetype_reduces_to_known_value():
    # Membership (ncell=4, npft=2), cells ordered as _CELL_* (soc 4,1,2,3):
    #   arch0 <- cell1 (soc 1, w0.6) + cell2 (soc 2, w0.4)
    #   arch1 <- cell0 (soc 4, w1.0) + cell3 (soc 3, w1.0)
    cid = np.array([[1, -1], [0, -1], [0, -1], [1, -1]])
    cw = np.array([[1.0, 0.0], [0.6, 0.0], [0.4, 0.0], [1.0, 0.0]])
    out = per_archetype_soc_from_grid(
        _FIELD, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON, cid, cw, n_arch=2)
    # arch0 = (0.6*1 + 0.4*2) / (0.6+0.4) = 1.4 ; arch1 = (1*4 + 1*3)/2 = 3.5
    npt.assert_allclose(out, [1.4, 3.5], rtol=1e-12)


def test_reduction_matches_cover_weighted_mean_of_sampled_cells():
    # Consistency: per_archetype_soc_from_grid == per_archetype_cover_weighted_mean on
    # the directly-sampled per-cell SOC (no duplicated reduction numerics).
    cid = np.array([[1, -1], [0, 2], [0, -1], [1, -1]])
    cw = np.array([[1.0, 0.0], [0.6, 0.3], [0.4, 0.0], [1.0, 0.0]])
    got = per_archetype_soc_from_grid(
        _FIELD, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON, cid, cw, n_arch=3)
    cell_soc = sample_grid_to_cells(_FIELD, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON)
    ref = per_archetype_cover_weighted_mean(cell_soc, cid, cw, n_arch=3)
    npt.assert_allclose(got, ref, rtol=1e-12, equal_nan=True)


def test_archetype_with_no_finite_member_is_nan():
    # A cell that lands off-grid -> NaN obs -> its archetype (if it has no other finite
    # member) is NaN, never a fabricated 0.
    cell_lat = np.array([10.0, 50.0])       # cell1 off-grid (lat 50)
    cell_lon = np.array([100.0, 100.0])
    cid = np.array([[0, -1], [1, -1]])
    cw = np.array([[1.0, 0.0], [1.0, 0.0]])
    out = per_archetype_soc_from_grid(
        _FIELD, _OBS_LAT, _OBS_LON, cell_lat, cell_lon, cid, cw, n_arch=2)
    npt.assert_allclose(out[0], 1.0, rtol=1e-12)   # cell0 on-grid (10,100) -> soc 1
    assert np.isnan(out[1])                         # cell1 off-grid -> arch1 NaN


def test_bad_shapes_raise():
    with pytest.raises(ValueError):
        sample_grid_to_cells(np.ones(4), _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON)
    with pytest.raises(ValueError):
        per_archetype_soc_from_grid(
            _FIELD, _OBS_LAT, _OBS_LON, _CELL_LAT, _CELL_LON,
            np.array([0, 1]), np.array([[1.0, 0.0]]))   # cid not 2-D


def test_load_gridded_soc_roundtrip(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "obs.nc"
    ds = xr.Dataset(
        {"soc": (("lat", "lon"), _FIELD)},
        coords={"lat": ("lat", _OBS_LAT), "lon": ("lon", _OBS_LON)})
    ds.to_netcdf(path)
    from legoesm.land.carbon.soc_gridded_target import load_gridded_soc

    soc, lat, lon = load_gridded_soc(str(path))
    npt.assert_allclose(soc, _FIELD, rtol=1e-12)
    npt.assert_allclose(lat, _OBS_LAT, rtol=1e-12)
    npt.assert_allclose(lon, _OBS_LON, rtol=1e-12)


def test_load_gridded_soc_enforces_lat_lon_dim_order(tmp_path):
    # A field stored (lon, lat) must load as (lat, lon) by dim NAME, not silently
    # transposed by shape (guards a square-grid geographic flip).
    xr = pytest.importorskip("xarray")
    path = tmp_path / "lonlat.nc"
    # soc.T is (lon, lat) = (2, 2) here; stored with dims ("lon", "lat").
    xr.Dataset(
        {"soc": (("lon", "lat"), _FIELD.T)},
        coords={"lat": ("lat", _OBS_LAT), "lon": ("lon", _OBS_LON)}).to_netcdf(path)
    from legoesm.land.carbon.soc_gridded_target import load_gridded_soc

    soc, lat, lon = load_gridded_soc(str(path))
    npt.assert_allclose(soc, _FIELD, rtol=1e-12)          # restored to (lat, lon)
    # and it samples correctly onto the cover cells.
    out = sample_grid_to_cells(soc, lat, lon, _CELL_LAT, _CELL_LON)
    npt.assert_allclose(out, _CELL_SOC, rtol=1e-12)


def test_sample_grid_singleton_axis_requires_explicit_tol():
    # A degenerate 1-row obs has no derivable spacing -> default-tol lookup raises,
    # but an explicit tol_deg still works.
    field = np.array([[1.0, 2.0]])                        # 1 lat x 2 lon
    obs_lat = np.array([10.0]); obs_lon = np.array([100.0, 200.0])
    with pytest.raises(ValueError):
        sample_grid_to_cells(field, obs_lat, obs_lon,
                             np.array([10.0]), np.array([100.0]))
    out = sample_grid_to_cells(field, obs_lat, obs_lon,
                               np.array([10.0]), np.array([200.0]), tol_deg=1.0)
    npt.assert_allclose(out, [2.0], rtol=1e-12)


def test_load_gridded_soc_missing_var_raises(tmp_path):
    xr = pytest.importorskip("xarray")
    path = tmp_path / "bad.nc"
    xr.Dataset(
        {"not_soc": (("lat", "lon"), _FIELD)},
        coords={"lat": ("lat", _OBS_LAT), "lon": ("lon", _OBS_LON)}).to_netcdf(path)
    from legoesm.land.carbon.soc_gridded_target import load_gridded_soc

    with pytest.raises(ValueError):
        load_gridded_soc(str(path))
