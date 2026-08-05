"""Direct leaf tests for the pure helpers in ``legoesm.ocean.init_woa`` (#501).

These three NumPy helpers are LIVE in the OMIP / centennial init drivers but
had no direct unit coverage — a sign/interp/wrap regression would only surface
in a full spin-up run.  Pure functions, so the tests need no model state.
"""

from __future__ import annotations

import numpy as np

from legoesm.ocean.init_woa import (
    _analytical_woa_profiles,
    _interp_profile_to_z_coord,
    _nearest_neighbor_2d,
)
from legoesm.ocean.vertical import create_ocean_z_star


def _z():
    return create_ocean_z_star(n_levels=10, H_max=5000.0)


# --- _interp_profile_to_z_coord ---------------------------------------------

def test_interp_matches_numpy_interp():
    z = _z()
    model_depths = np.abs(np.asarray(z.z_full_ref))
    woa_depths = np.array([0.0, 100.0, 1000.0, 5000.0])
    profile = np.array([20.0, 15.0, 4.0, 1.5])
    got = _interp_profile_to_z_coord(profile, woa_depths, z)
    expected = np.interp(model_depths, woa_depths, profile)
    assert np.allclose(got, expected)
    assert got.shape == (z.n_levels,)


def test_interp_monotone_profile_stays_monotone():
    z = _z()
    woa_depths = np.linspace(0.0, 5000.0, 6)
    profile = np.linspace(25.0, 2.0, 6)  # cooling with depth
    got = _interp_profile_to_z_coord(profile, woa_depths, z)
    # Model depths increase with level → interpolated T must be non-increasing.
    assert np.all(np.diff(got) <= 1e-9)


def test_interp_handles_nan_and_too_few_valid():
    z = _z()
    woa_depths = np.array([0.0, 100.0, 1000.0])
    # Only one valid sample → fall back to that surface value everywhere.
    profile = np.array([18.0, np.nan, np.nan])
    got = _interp_profile_to_z_coord(profile, woa_depths, z)
    assert np.allclose(got, 18.0)
    assert got.shape == (z.n_levels,)
    # All-NaN → NaN, so ``init_ocean_from_woa``'s T_fill/S_fill applies.
    #
    # This previously asserted zeros and called NaN the leak.  It is the other
    # way round: NaN is CAUGHT by the caller's
    # ``np.where(np.isnan(T_out), T_fill, T_out)``, while 0.0 sails straight
    # through it and enters the model as T = 0 degC / S = 0 PSU -- fresh water
    # at 0 degC (rho = 999.8) beside ~1027 sea water.  On the FESOM2-matched
    # CORE2 config that gave 1569 wet columns a density step of up to
    # 30 kg/m^3 and ~42 m/s in one 2400 s step.
    got0 = _interp_profile_to_z_coord(np.full(3, np.nan), woa_depths, z)
    assert np.all(np.isnan(got0))
    assert not np.any(got0 == 0.0)


# --- _analytical_woa_profiles -----------------------------------------------

def test_analytical_profiles_shape_and_equator_warmer():
    z = _z()
    lat = np.array([0.0, 45.0, 90.0])
    T, S = _analytical_woa_profiles(lat, z)
    assert T.shape == (3, z.n_levels)
    assert S.shape == (3, z.n_levels)
    # SST: equator warmest, pole coldest.
    assert T[0, 0] > T[1, 0] > T[2, 0]
    # Deep water converges toward ~1.5 degC regardless of latitude.
    assert abs(T[0, -1] - T[2, -1]) < abs(T[0, 0] - T[2, 0])
    # Temperature decreases with depth at the equator.
    assert np.all(np.diff(T[0]) <= 1e-9)


def test_analytical_profiles_physical_ranges():
    z = _z()
    lat = np.linspace(-90.0, 90.0, 19)
    T, S = _analytical_woa_profiles(lat, z)
    assert np.all(np.isfinite(T)) and np.all(np.isfinite(S))
    assert T.min() > -2.0 and T.max() < 32.0     # ocean potential temp [degC]
    assert 33.0 < S.min() and S.max() < 37.0     # ocean salinity [PSU]


# --- _nearest_neighbor_2d ---------------------------------------------------

def test_nearest_neighbor_exact_at_source_points():
    src_lat = np.array([-30.0, 0.0, 30.0])
    src_lon = np.array([0.0, 90.0, 180.0, 270.0])
    field = np.arange(12, dtype=np.float64).reshape(3, 4)
    # Targets exactly on source nodes → exact field values.
    tlat = np.array([0.0, 30.0])
    tlon = np.array([90.0, 270.0])
    got = _nearest_neighbor_2d(tlat, tlon, src_lat, src_lon, field)
    assert got.shape == (2,)
    assert got[0] == field[1, 1]   # (lat=0, lon=90)
    assert got[1] == field[2, 3]   # (lat=30, lon=270)


def test_nearest_neighbor_longitude_wrap_and_extra_dims():
    src_lat = np.array([0.0, 10.0])
    src_lon = np.array([0.0, 120.0, 240.0])
    field = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)  # trailing dim
    # The helper normalises lon by %360 then does a LINEAR (not circular)
    # argmin.  lon=365 -> 5 -> nearest of {0,120,240} is 0 (col 0); the
    # trailing field dim is preserved in the output shape.
    tlat = np.array([0.0])
    tlon = np.array([365.0])
    got = _nearest_neighbor_2d(tlat, tlon, src_lat, src_lon, field)
    assert got.shape == (1, 4)            # target.shape + field.shape[2:]
    assert np.allclose(got[0], field[0, 0])
