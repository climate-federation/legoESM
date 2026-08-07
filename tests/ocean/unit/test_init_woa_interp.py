"""Leaf tests for the untested pure helpers in ``ocean.init_woa``.

``init_woa`` is LIVE in production (imported by ``scripts/run/run_omip.py``,
``run_omip_core2.py``, and the centennial-spinup driver) to build a stratified
initial condition from WOA18 climatology, with an analytical fallback when the
NetCDF files are absent. ``_bilinear_2d`` is already covered by
``tests/unit/test_omip2_io.py``; the three pure helpers exercised here had no
direct test, so a regression in the IC interpolation / fallback profiles would
silently corrupt every realistic spin-up.

These are deterministic NumPy functions — no JAX, no x64 requirement.
"""

import numpy as np
import pytest

from legoesm.ocean.init_woa import (
    _analytical_woa_profiles,
    _interp_profile_to_z_coord,
    _nearest_neighbor_2d,
)
from legoesm.ocean.vertical import create_ocean_z_star


@pytest.fixture
def z_coord():
    return create_ocean_z_star(
        n_levels=20, H_max=4000.0, dz_surface=10.0, dz_deep=400.0,
    )


# ---------------------------------------------------------------------------
# _analytical_woa_profiles — latitude-dependent fallback T/S
# ---------------------------------------------------------------------------


def test_analytical_profiles_shape_and_finite(z_coord):
    lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    T, S = _analytical_woa_profiles(lat, z_coord)
    assert T.shape == (lat.size, z_coord.n_levels)
    assert S.shape == (lat.size, z_coord.n_levels)
    assert np.all(np.isfinite(T))
    assert np.all(np.isfinite(S))


def test_analytical_profiles_sst_warmest_at_equator(z_coord):
    """Surface temperature must peak at the equator and fall toward the poles
    (T_surf ~ 28*cos^2(lat)), the defining feature of the fallback."""
    lat = np.array([-60.0, -30.0, 0.0, 30.0, 60.0])
    T, _ = _analytical_woa_profiles(lat, z_coord)
    sst = T[:, 0]  # shallowest level
    assert np.argmax(sst) == 2  # equator is the middle entry
    # Symmetric about the equator and monotonic toward each pole.
    assert sst[2] > sst[1] > sst[0]
    assert sst[2] > sst[3] > sst[4]
    assert np.allclose(sst, sst[::-1])  # hemispheric symmetry


def test_analytical_profiles_temperature_decreases_with_depth(z_coord):
    """Warm surface relaxing to ~1.5 degC deep water: T must be monotonically
    non-increasing downward and bounded by [T_deep, SST]."""
    lat = np.array([0.0])
    T, S = _analytical_woa_profiles(lat, z_coord)
    col = T[0]
    assert np.all(np.diff(col) <= 1e-9)  # non-increasing with depth
    assert col[-1] >= 1.5 - 1e-6  # approaches but does not undershoot T_deep
    assert col[0] <= 28.0 + 1e-6  # equatorial SST cap
    # Salinity stays in a physical open-ocean band.
    assert np.all(S > 33.0)
    assert np.all(S < 36.0)


# ---------------------------------------------------------------------------
# _interp_profile_to_z_coord — WOA depth levels -> model z-star
# ---------------------------------------------------------------------------


def test_interp_linear_profile_is_exact(z_coord):
    """A profile linear in depth must interpolate exactly onto the model
    full-level depths (np.interp is exact for piecewise-linear inputs)."""
    woa_depths = np.array([0.0, 100.0, 500.0, 1000.0, 4000.0])
    # value = 20 - 0.004 * depth (linear), so interp == analytic everywhere.
    profile = 20.0 - 0.004 * woa_depths
    out = _interp_profile_to_z_coord(profile, woa_depths, z_coord)
    model_depths = np.abs(np.asarray(z_coord.z_full_ref))
    expected = 20.0 - 0.004 * np.clip(
        model_depths, woa_depths[0], woa_depths[-1])
    assert out.shape == (z_coord.n_levels,)
    assert np.allclose(out, expected)


def test_interp_too_few_valid_returns_surface_fill(z_coord):
    """With <2 valid (non-NaN) samples the helper returns the single valid
    value everywhere rather than producing NaNs."""
    woa_depths = np.array([0.0, 100.0, 500.0])
    profile = np.array([12.5, np.nan, np.nan])
    out = _interp_profile_to_z_coord(profile, woa_depths, z_coord)
    assert out.shape == (z_coord.n_levels,)
    assert np.allclose(out, 12.5)


def test_interp_all_nan_returns_nan_so_the_caller_fill_applies(z_coord):
    """No valid samples -> NaN, NOT 0.0.

    This test previously asserted ``np.allclose(out, 0.0)`` and called it a
    "documented degenerate fallback".  That pinned a real defect in place:
    0.0 is not NaN, so ``init_ocean_from_woa``'s
    ``np.where(np.isnan(T_out), T_fill, T_out)`` never replaced it and a WET
    cell with no source data entered the model as T = 0 degC / S = 0 PSU --
    fresh water at 0 degC (rho = 999.8) beside ~1027 sea water, a density step
    larger than the ocean's entire range.  Returning NaN routes the column to
    the caller's documented fill instead.
    """
    woa_depths = np.array([0.0, 100.0, 500.0])
    profile = np.full(3, np.nan)
    out = _interp_profile_to_z_coord(profile, woa_depths, z_coord)
    assert np.all(np.isnan(out))
    assert not np.any(out == 0.0)


def test_interp_skips_nan_samples(z_coord):
    """A NaN in the middle of an otherwise-linear profile must be dropped, and
    interpolation must still recover the underlying line."""
    woa_depths = np.array([0.0, 100.0, 500.0, 1000.0])
    profile = np.array([20.0, np.nan, 18.0, 16.0])  # line: 20 - 0.004*depth
    out = _interp_profile_to_z_coord(profile, woa_depths, z_coord)
    model_depths = np.abs(np.asarray(z_coord.z_full_ref))
    expected = 20.0 - 0.004 * np.clip(model_depths, 0.0, 1000.0)
    assert np.allclose(out, expected)


# ---------------------------------------------------------------------------
# _nearest_neighbor_2d — regular lat-lon -> target points
# ---------------------------------------------------------------------------


def test_nearest_neighbor_exact_grid_points():
    """Targets that coincide with source nodes return those exact values."""
    src_lat = np.array([-45.0, 0.0, 45.0])
    src_lon = np.array([0.0, 90.0, 180.0, 270.0])
    # field[i, j] encodes its indices so we can assert which node was picked.
    field = (np.arange(3)[:, None] * 10 + np.arange(4)[None, :]).astype(float)
    target_lat = np.array([0.0, 45.0])
    target_lon = np.array([90.0, 270.0])
    out = _nearest_neighbor_2d(target_lat, target_lon, src_lat, src_lon, field)
    assert out.shape == target_lat.shape
    assert out[0] == 11.0  # lat idx 1, lon idx 1
    assert out[1] == 23.0  # lat idx 2, lon idx 3


def test_nearest_neighbor_rounds_to_closest_and_wraps_lon():
    """Off-node targets snap to the closest node; longitude is matched modulo
    360 so a negative target longitude maps to the right source column."""
    src_lat = np.array([-45.0, 0.0, 45.0])
    src_lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = (np.arange(3)[:, None] * 10 + np.arange(4)[None, :]).astype(float)
    # lat 5 -> nearest 0 (idx 1); lon -85 == 275 mod 360 -> nearest 270 (idx 3)
    out = _nearest_neighbor_2d(
        np.array([5.0]), np.array([-85.0]), src_lat, src_lon, field)
    assert out[0] == 13.0


def test_nearest_neighbor_preserves_trailing_dims():
    """A source field with a trailing (depth) axis returns
    target.shape + field.shape[2:]."""
    src_lat = np.array([-45.0, 0.0, 45.0])
    src_lon = np.array([0.0, 180.0])
    field = np.arange(3 * 2 * 4, dtype=float).reshape(3, 2, 4)  # (lat, lon, z)
    target_lat = np.array([[0.0, 45.0]])  # 2-D target
    target_lon = np.array([[0.0, 180.0]])
    out = _nearest_neighbor_2d(target_lat, target_lon, src_lat, src_lon, field)
    assert out.shape == (1, 2, 4)
    assert np.allclose(out[0, 0], field[1, 0])
    assert np.allclose(out[0, 1], field[2, 1])


# ---------------------------------------------------------------------------
# no-data columns must reach the caller's FILL, never 0.0
#
# Returning 0.0 for a column with no valid source data was a real defect: 0.0
# is not NaN, so ``init_ocean_from_woa``'s
# ``np.where(np.isnan(T_out), T_fill, T_out)`` never replaced it, and a WET
# cell whose source column had no data entered the model as T = 0 degC,
# S = 0 PSU.  Fresh water at 0 degC is rho = 999.8 against ~1027 for sea
# water, so the cell sat beside normal ocean with a density jump of up to
# 30 kg/m^3 -- more than the entire ocean's density range -- which on the
# FESOM2-matched CORE2 config produced ~42 m/s in one 2400 s step.
# ---------------------------------------------------------------------------

def test_all_nan_profile_returns_nan_not_zero():
    """The value that reaches the caller must be NaN so the fill applies.

    Asserting ``not 0.0`` is the point: 0.0 is a PLAUSIBLE-looking number that
    silently survives the NaN fill and becomes fresh 0 degC water.
    """
    z = create_ocean_z_star(n_levels=6, H_max=1000.0)
    depths = np.array([0.0, 100.0, 500.0, 1000.0])
    out = _interp_profile_to_z_coord(np.full(4, np.nan), depths, z)
    assert out.shape == (6,)
    assert np.all(np.isnan(out)), f"expected all-NaN, got {out}"
    assert not np.any(out == 0.0)


def test_single_valid_entry_is_propagated_down_the_column():
    """One valid entry is still usable — it fills the column with that value,
    and must NOT be turned into NaN by the fix above."""
    z = create_ocean_z_star(n_levels=6, H_max=1000.0)
    depths = np.array([0.0, 100.0, 500.0, 1000.0])
    prof = np.array([np.nan, 7.25, np.nan, np.nan])
    out = _interp_profile_to_z_coord(prof, depths, z)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, 7.25)


def test_two_valid_entries_still_interpolate():
    """The >= 2 path is untouched by the fix."""
    z = create_ocean_z_star(n_levels=6, H_max=1000.0)
    depths = np.array([0.0, 1000.0])
    out = _interp_profile_to_z_coord(np.array([10.0, 2.0]), depths, z)
    assert np.all(np.isfinite(out))
    assert out[0] > out[-1]


def test_no_data_column_reaches_the_fill_through_init_ocean_from_woa(tmp_path):
    """End-to-end: a WET cell whose source column is all-NaN must come out at
    the FILL value (1.5 degC / 34.7 PSU), never 0/0 -- which would be fresh
    0 degC water and a ~30 kg/m^3 density step against its neighbour."""
    xr = pytest.importorskip("xarray")
    from legoesm import constants
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.ocean.init_woa import init_ocean_from_woa

    src_lat = np.arange(-89.5, 90.0, 1.0)
    src_lon = np.arange(0.5, 360.0, 1.0)
    depths = np.array([0.0, 100.0, 1000.0])
    T = np.full((1, depths.size, src_lat.size, src_lon.size), 12.0)
    S = np.full_like(T, 35.0)
    # A no-data hole: every source cell that could feed target (lat 0.5, lon
    # 0.5) is NaN, so the horizontal interp yields NaN and the column has no
    # valid entries at all.
    T[:, :, 88:92, 0:4] = np.nan
    S[:, :, 88:92, 0:4] = np.nan
    ds = xr.Dataset(
        {"t_an": (("time", "depth", "lat", "lon"), T),
         "s_an": (("time", "depth", "lat", "lon"), S)},
        coords={"time": [0.0], "depth": depths,
                "lat": src_lat, "lon": src_lon},
    )
    path = tmp_path / "woa_hole.nc"
    ds.to_netcdf(path)

    grid = create_latlon_grid(180, 360)
    z = create_ocean_z_star(n_levels=8, H_max=1000.0)
    T_out, S_out = init_ocean_from_woa(grid, z, path, path)
    T_out = np.asarray(T_out)
    S_out = np.asarray(S_out)

    assert np.all(np.isfinite(T_out)) and np.all(np.isfinite(S_out))
    # NOTHING may be exactly 0 -- that is the defect's signature.
    assert not np.any(S_out == 0.0), "S == 0 PSU leaked into the IC"
    assert not np.any(T_out == 0.0)
    # The INTERIOR of the hole must carry the documented fill.  Only the
    # interior is asserted: the hole's edge cells legitimately interpolate
    # from valid neighbours outside it (the bilinear stencil reweights around
    # NaN), so asserting the full block would be asserting the wrong thing.
    hole = (slice(89, 91), slice(1, 3))
    np.testing.assert_allclose(
        T_out[hole], float(constants.T_deep_ocean_ref_C))
    np.testing.assert_allclose(
        S_out[hole], float(constants.S_deep_ocean_ref_psu))
