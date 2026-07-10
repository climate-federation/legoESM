"""Surfdata-driven spatial land albedo in the AMIP driver.

Exercises the new ``ModelDriver._surfdata_land_albedo`` path (wired into
``_create_physics``): the static land albedo field is taken from the harmonized
surface data (per-column soil-colour + PFT-vegetation blend, glacier override)
instead of the latitude-only ``land_vegetation_albedo`` curve, while ocean /
mask-non-land cells keep the latitude fallback.

The method is tested via a duck-typed ``self`` (real ``LatLonGrid`` + synthetic
surfdata) so the assertions target the new mapping/blend logic without standing
up a full ``ModelDriver.setup()`` (which would need a land-mask NetCDF + the
whole physics pipeline).
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import jax.numpy as jnp
import pytest

xr = pytest.importorskip("xarray")

from legoesm.grids.latlon import create_latlon_grid
from legoesm.surface_albedo import land_vegetation_albedo
from legoesm.land.surface_params import N_PFT_CLM5
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.driver.model_driver import ModelDriver


def _write_spatially_varying_surfdata(path, nlat=8, nlon=16, nlev=7):
    """Synthetic surfdata with a soil-colour gradient + a high-latitude glacier
    band, so the resulting albedo varies in space and exercises the glacier
    override."""
    lat = np.linspace(-85, 85, nlat)
    lon = np.linspace(0, 337.5, nlon)

    # Soil-colour gradient over the 20 CLM classes -> spatially varying albedo.
    soil_color = np.zeros((nlat, nlon))
    for i in range(nlat):
        for j in range(nlon):
            soil_color[i, j] = 1.0 + ((i * nlon + j) % 20)

    # All-bare-soil PFT (finite everywhere -> surfdata-covered).
    pft = np.zeros((1, N_PFT_CLM5, nlat, nlon))
    pft[0, 0] = 100.0
    lai = np.zeros((12, N_PFT_CLM5, nlat, nlon))

    # Cover: land everywhere except a glacier band on the top two rows.
    f_land = np.full((1, nlat, nlon), 100.0)
    f_glacier = np.zeros((1, nlat, nlon))
    f_land[0, -2:, :] = 0.0
    f_glacier[0, -2:, :] = 100.0

    soil = lambda v: np.full((nlev, nlat, nlon), v)
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=soil(40.0), clay_pct=soil(20.0), organic=soil(5.0),
        bulk_density=soil(1300.0), soil_color=soil_color,
        year=np.array([2015.0]),
        f_land=f_land, f_lake=np.zeros((1, nlat, nlon)), f_glacier=f_glacier,
        pft_frac=pft, monthly_lai=lai, monthly_sai=np.zeros_like(lai),
        monthly_height_top=np.zeros_like(lai), monthly_height_bot=np.zeros_like(lai),
    )


def _fake_driver(grid, f_land, start_day=196.0, start_year=2015.0):
    """Minimal duck-typed ModelDriver for calling _surfdata_land_albedo unbound."""
    return SimpleNamespace(grid=grid, _f_land=f_land,
                           config=SimpleNamespace(start_day=start_day,
                                                  start_year=start_year))


def test_surfdata_albedo_is_spatial_finite_and_bounded(tmp_path):
    p = str(tmp_path / "sd.nc")
    _write_spatially_varying_surfdata(p)
    grid = create_latlon_grid(8, 16)
    f_land = jnp.ones(grid.grid_shape_2d)            # all land

    lat_albedo = land_vegetation_albedo(grid.grid_lat)
    field = ModelDriver._surfdata_land_albedo(
        _fake_driver(grid, f_land), p, lat_albedo)
    field = np.asarray(field)

    assert field.shape == grid.grid_shape_2d
    assert np.isfinite(field).all()
    # Spatially varying — strictly more structure than the smooth lat curve.
    assert float(np.std(field)) > float(np.std(np.asarray(lat_albedo)))
    # CLM soil-colour albedos through glacier-ice (0.6) stay in a physical band.
    assert field.min() >= 0.05 and field.max() <= 0.7
    # The glacier band (top two rows) is the bright ice override.
    assert float(field[-2:, :].min()) > 0.5


def test_surfdata_albedo_falls_back_to_lat_over_ocean(tmp_path):
    """Cells the driver mask marks ocean (f_land == 0) keep the lat curve."""
    p = str(tmp_path / "sd.nc")
    _write_spatially_varying_surfdata(p)
    grid = create_latlon_grid(8, 16)

    f_land = np.ones(grid.grid_shape_2d)
    f_land[0, :] = 0.0                                # bottom row = ocean
    f_land = jnp.asarray(f_land)

    lat_albedo = land_vegetation_albedo(grid.grid_lat)
    field = np.asarray(ModelDriver._surfdata_land_albedo(
        _fake_driver(grid, f_land), p, lat_albedo))

    # Ocean row exactly equals the latitude fallback.
    np.testing.assert_allclose(field[0, :], np.asarray(lat_albedo)[0, :], rtol=0, atol=0)
    # A land row differs from the lat curve (surfdata took over).
    assert not np.allclose(field[3, :], np.asarray(lat_albedo)[3, :])


def test_surfdata_albedo_column_grid_mismatch_raises(tmp_path):
    """A surfdata column count that disagrees with the grid is a loud error,
    not a silent mis-mapping."""
    p = str(tmp_path / "sd.nc")
    _write_spatially_varying_surfdata(p)
    grid = create_latlon_grid(8, 16)                 # loader yields 128 columns
    lat_albedo = land_vegetation_albedo(grid.grid_lat)

    # Real grid geometry (so the loader runs) but a mismatched reshape target:
    # grid_lat is (4, 4) = 16, while the loader produces 128-column params.
    # _f_land matches the loader's 128 columns so the f_land reconciliation
    # passes and the failure surfaces at the albedo->grid reshape guard.
    bad_grid = SimpleNamespace(
        lat2d=grid.lat2d, lon2d=grid.lon2d, grid_area=grid.grid_area,
        grid_lat=jnp.zeros((4, 4)),
    )
    bad = SimpleNamespace(grid=bad_grid, _f_land=jnp.ones((8, 16)),
                          config=SimpleNamespace(start_day=196.0))
    with pytest.raises(ValueError, match="column/grid layout mismatch"):
        ModelDriver._surfdata_land_albedo(bad, p, lat_albedo)
