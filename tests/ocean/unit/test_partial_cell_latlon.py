"""Unit tests for the WOA-cold-start geometry helpers promoted into the ocean
package: make_partial_cell_latlon (bathy smooth + thin-cell snap + min_levels ->
OceanPartialCellCoordinate) and woa_ocean_bathymetry.

These are the OMIP-validated pieces that make the coupled 3D-ocean WOA cold start
stable (the flat-bottom z-star coord blows up).
"""
from __future__ import annotations

import os
import jax
jax.config.update("jax_enable_x64", True)

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.vertical import (
    create_ocean_z_star, OceanPartialCellCoordinate,
)
from legoesm.ocean.init_latlon_cgrid import make_partial_cell_latlon

_WOA_T = "data/woa18/woa18_decav_t00_01.nc"


def _synthetic_bathy(grid):
    """A basin deepening toward the centre + a steep shelf, with a polar land
    cap, so smoothing / snapping / shallow-masking all have something to do."""
    lat = np.asarray(grid.lat2d) * 180.0 / np.pi
    lon = np.asarray(grid.lon2d) * 180.0 / np.pi
    H = 200.0 + 5000.0 * np.cos(np.radians(lat)) ** 2          # deep tropics
    H = H + 1500.0 * np.sin(np.radians(2 * lon))               # zonal ripples
    land = np.abs(lat) > 70.0                                  # polar caps
    H = np.where(land, 0.0, np.clip(H, 50.0, 5500.0))
    mask = (~land).astype(np.float64)
    return H, mask


def test_make_partial_cell_latlon_basic():
    grid = create_latlon_grid(n_lat=24, n_lon=48)
    z = create_ocean_z_star(20, H_max=5500.0)
    H, mask = _synthetic_bathy(grid)
    from legoesm.ocean.bathymetry import compute_max_r_factor
    r0 = float(compute_max_r_factor(H, mask))

    zc, H_snap, lm = make_partial_cell_latlon(
        z, H, mask, smoothing_passes=4, min_levels=2)

    assert isinstance(zc, OceanPartialCellCoordinate)
    # Smoothing reduces the bathymetric roughness (r-factor).
    assert float(compute_max_r_factor(H_snap, lm)) <= r0 + 1e-9
    # Output shapes + land convention preserved/extended.
    assert H_snap.shape == H.shape and lm.shape == mask.shape
    assert np.all((lm == 0.0) | (lm == 1.0))
    # Snapped depth is 0 exactly where the mask says land.
    assert np.all(H_snap[lm < 0.5] == 0.0)
    # min_levels: every remaining wet column has >= 2 active reference levels.
    abs_zh = np.abs(np.asarray(z.z_half_ref))
    n_active = (abs_zh[None, None, :z.n_levels] < H_snap[..., None]).sum(axis=2)
    wet = lm > 0.5
    assert np.all(n_active[wet] >= 2)
    # Polar land caps stay land.
    lat = np.asarray(grid.lat2d) * 180.0 / np.pi
    assert np.all(lm[np.abs(lat) > 70.0] == 0.0)


def test_make_partial_cell_no_smoothing_identity_rfactor():
    grid = create_latlon_grid(n_lat=16, n_lon=32)
    z = create_ocean_z_star(15, H_max=5500.0)
    H, mask = _synthetic_bathy(grid)
    zc, H_snap, lm = make_partial_cell_latlon(
        z, H, mask, smoothing_passes=0, min_levels=1)
    assert isinstance(zc, OceanPartialCellCoordinate)
    # No smoothing, min_levels=1: only the thin-cell snap can change H.
    assert np.all(H_snap <= H + 1e-9)


@pytest.mark.skipif(not os.path.exists(_WOA_T),
                    reason="WOA18 file not present (data/woa18/)")
def test_woa_ocean_bathymetry_physical():
    from legoesm.ocean.init_woa import woa_ocean_bathymetry, woa_ocean_mask
    grid = create_latlon_grid(n_lat=90, n_lon=180)         # 2 deg
    H = woa_ocean_bathymetry(grid, _WOA_T, H_max=5500.0, min_depth_m=200.0)
    mask = woa_ocean_mask(grid, _WOA_T) > 0.5
    assert H.shape == (90, 180)
    assert np.all(np.isfinite(H))
    # Land cells dry; wet cells within [min_depth, H_max]; mean a few km.
    assert np.all(H >= 0.0) and np.all(H <= 5500.0)
    wet = H > 0.0
    assert wet.any()
    assert np.all(H[wet] >= 200.0 - 1e-6)
    assert 1500.0 < float(H[wet].mean()) < 5000.0
    # Bathy wet domain is a subset of (close to) the surface ocean mask.
    assert wet.sum() <= mask.sum() + 5
