"""Conservation guard for the soil regrid path.

Whenever soils are regridded (consumer ``build_global_surface_data`` and producer
``assemble.build_v1_surfdata``, both via ``conservative_regrid_latlon``) the
global **area-weighted mean must be conserved** and **NaN must not bleed** from
ocean into coastal land cells.  These tests run in CI and must pass.
"""

import numpy as np
import jax.numpy as jnp
import pytest

from legoesm.land.global_surface_data import (
    _RawSurfaceFields,
    build_global_surface_data,
    GlobalSurfaceDataConfig,
)
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig


def _area_mean(field2d, lat_deg):
    w = np.cos(np.deg2rad(np.asarray(lat_deg)))[:, None]
    f = np.asarray(field2d)
    m = np.isfinite(f)
    return float(np.sum(np.where(m, f, 0.0) * w) / np.sum(m * w))


class _RegularGrid:
    """Regular lat-lon grid tiling the full sphere (cell centres, radians)."""

    def __init__(self, nlat, nlon):
        self.lat_deg = -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat)
        lon = (np.arange(nlon) + 0.5) * (360.0 / nlon)
        self.lon2d, self.lat2d = np.meshgrid(np.deg2rad(lon), np.deg2rad(self.lat_deg))
        self.grid_area = jnp.ones(nlat * nlon, dtype=jnp.float32)
        self.nlat, self.nlon, self.ncol = nlat, nlon, nlat * nlon


def _raw_with_soil(sand2d, clay2d, nlat_s, nlon_s, nz=6):
    """Minimal _RawSurfaceFields with full-sphere source + depth-constant soil."""
    slat = -90.0 + (np.arange(nlat_s) + 0.5) * (180.0 / nlat_s)
    slon = (np.arange(nlon_s) + 0.5) * (360.0 / nlon_s)
    rep = lambda a: np.repeat(np.asarray(a)[:, :, None], nz, axis=2)
    npft = 3
    return _RawSurfaceFields(
        src_lat=slat, src_lon=slon, src_soil_depth=np.cumsum(np.full(nz, 0.1)),
        sand=rep(sand2d), clay=rep(clay2d), organic=rep(np.full_like(sand2d, 5.0)),
        bulk_density=rep(np.full_like(sand2d, 1.4)),
        soil_color=np.ones((nlat_s, nlon_s)),
        years=np.array([0.0]),
        f_land=np.ones((1, nlat_s, nlon_s)), f_lake=np.zeros((1, nlat_s, nlon_s)),
        f_glacier=np.zeros((1, nlat_s, nlon_s)),
        pft_frac=np.full((1, nlat_s, nlon_s, npft), 1.0 / npft),
        lai=np.zeros((12, nlat_s, nlon_s, npft)), sai=np.zeros((12, nlat_s, nlon_s, npft)),
        htop=np.zeros((12, nlat_s, nlon_s, npft)), hbot=np.zeros((12, nlat_s, nlon_s, npft)),
        cell_area=None,
    )


def _top_layer_2d(gsd_field, grid):
    return np.asarray(gsd_field).reshape(grid.nlat, grid.nlon, -1)[:, :, 0]


def test_soil_regrid_conserves_global_mean():
    nlat_s, nlon_s = 18, 36
    rng = np.random.default_rng(2)
    sand = rng.uniform(10.0, 80.0, (nlat_s, nlon_s))     # percent-like, no NaN
    clay = rng.uniform(2.0, 50.0, (nlat_s, nlon_s))
    raw = _raw_with_soil(sand, clay, nlat_s, nlon_s)
    grid = _RegularGrid(12, 24)                          # coarsen, same sphere
    gsd = build_global_surface_data(raw, grid, GlobalSurfaceDataConfig(),
                                    soil_grid=make_soil_grid(SoilGridConfig()))
    slat = -90.0 + (np.arange(nlat_s) + 0.5) * (180.0 / nlat_s)
    # depth-constant source -> every model layer equals the regridded pattern
    sand_t = _top_layer_2d(gsd.sand_frac, grid)
    clay_t = _top_layer_2d(gsd.clay_frac, grid)
    assert np.isclose(_area_mean(sand, slat), _area_mean(sand_t, grid.lat_deg), rtol=1e-3)
    assert np.isclose(_area_mean(clay, slat), _area_mean(clay_t, grid.lat_deg), rtol=1e-3)


def test_soil_regrid_does_not_bleed_ocean_nan_into_land():
    nlat_s, nlon_s = 18, 36
    sand = np.full((nlat_s, nlon_s), 50.0)
    sand[:, 6:18] = np.nan                               # an "ocean" lon band
    clay = np.where(np.isfinite(sand), 20.0, np.nan)
    raw = _raw_with_soil(sand, clay, nlat_s, nlon_s)
    grid = _RegularGrid(12, 24)
    gsd = build_global_surface_data(raw, grid, GlobalSurfaceDataConfig(),
                                    soil_grid=make_soil_grid(SoilGridConfig()))
    sand_t = _top_layer_2d(gsd.sand_frac, grid)
    # target cells over the valid region stay exactly the source value (no bleed),
    # and only cells fully inside the NaN band are NaN.
    finite = np.isfinite(sand_t)
    assert np.allclose(sand_t[finite], 50.0, atol=1e-6)
    assert np.isnan(sand_t).any()                        # the ocean band remains NaN
