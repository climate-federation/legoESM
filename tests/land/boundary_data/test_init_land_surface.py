"""Integration test: init_land_surface_data (the simulation-start entry).

Writes a synthetic harmonized surfdata, then loads+regrids+adapts it for each
land scheme (multilayer-canopy, multilayer-SEB, slab) — confirming the loader
runs at sim start on a chosen grid and yields scheme-correct land params.
"""

import numpy as np
import jax.numpy as jnp
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.boundary_data import init_land_surface_data
from legoesm.land.surface_params import N_PFT_CLM5, LandSurfaceParams
from legoesm.land.canopy.config import CanopyLandParams
from legoesm.land.config import MultiLayerLandConfig, LandConfig
from legoesm.land.surface_scheme import SimpleSEBConfig, TwoLeafCanopyConfig


class _ToyGrid:
    def __init__(self, nlat=6, nlon=12):
        latc = np.linspace(-80, 80, nlat); lonc = np.linspace(0, 330, nlon)
        self.lat2d, self.lon2d = np.meshgrid(np.deg2rad(latc), np.deg2rad(lonc), indexing="ij")
        self.grid_area = jnp.ones(nlat * nlon, dtype=jnp.float32)
        self.ncol = nlat * nlon


def _write_surfdata(path, nlat=8, nlon=16, nlev=7):
    lat = np.linspace(-85, 85, nlat); lon = np.linspace(0, 337.5, nlon)
    pft = np.zeros((1, N_PFT_CLM5, nlat, nlon)); pft[0, 4] = 100.0   # all BE-tropical
    lai = np.zeros((12, N_PFT_CLM5, nlat, nlon)); lai[:, 4] = 4.0
    soil = lambda v: np.full((nlev, nlat, nlon), v)
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=soil(40.0), clay_pct=soil(20.0), organic=soil(5.0), bulk_density=soil(1300.0),
        soil_color=np.full((nlat, nlon), 8.0),
        year=np.array([2015.0]),
        f_land=np.full((1, nlat, nlon), 100.0),
        f_lake=np.zeros((1, nlat, nlon)), f_glacier=np.zeros((1, nlat, nlon)),
        pft_frac=pft, monthly_lai=lai, monthly_sai=np.zeros_like(lai),
        monthly_height_top=np.zeros_like(lai), monthly_height_bot=np.zeros_like(lai),
    )


@pytest.mark.parametrize("land_cfg, param_type, is_multilayer", [
    (MultiLayerLandConfig(surface_scheme=TwoLeafCanopyConfig()), CanopyLandParams, True),
    (MultiLayerLandConfig(surface_scheme=SimpleSEBConfig()), LandSurfaceParams, True),
    (LandConfig(), CanopyLandParams, False),
    (LandConfig(surface_scheme=SimpleSEBConfig()), LandSurfaceParams, False),
])
def test_init_land_surface_data_per_scheme(tmp_path, land_cfg, param_type, is_multilayer):
    p = str(tmp_path / "sd.nc"); _write_surfdata(p)
    grid = _ToyGrid()
    cfg, params, gsd = init_land_surface_data(p, grid, land_cfg, day_of_year=196.0)

    assert isinstance(params, param_type)                 # scheme-correct params
    # every per-column field has the grid's column count
    leaf = params.LAI if param_type is CanopyLandParams else params.albedo_veg
    assert np.asarray(leaf).shape == (grid.ncol,)
    # multilayer config gets surfdata-derived (Cosby) soil hydraulics
    if is_multilayer:
        assert cfg.hydraulics.retention_curve == "clapp_hornberger"
    assert gsd.pft_frac.shape[1] == grid.ncol             # regridded to the grid
