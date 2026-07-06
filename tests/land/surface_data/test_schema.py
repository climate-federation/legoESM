"""Schema writer unit test + full producer->consumer roundtrip.

Exercises the previously-untested NetCDF *reader* path of
``global_surface_data.load_global_surface_data`` by writing a harmonized
``legoesm_surfdata`` file with :func:`write_surfdata` and reading it back through
the ``"legoesm_surfdata"`` preset onto a toy model grid.
"""

import numpy as np
import jax.numpy as jnp
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_data.sources.hwsd2 import HWSD2_LAYER_DZ
from legoesm.land.global_surface_data import (
    get_surfdata_preset,
    load_global_surface_data,
)
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig


class _ToyGaussianGrid:
    def __init__(self, n_lat=8, n_lon=16):
        lat = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, n_lat)
        lon = np.linspace(0.0, 2 * np.pi, n_lon, endpoint=False)
        self.lon2d, self.lat2d = np.meshgrid(lon, lat)
        self.grid_area = jnp.asarray(np.full(n_lat * n_lon, 1.0e9), dtype=jnp.float32)
        self.ncol = n_lat * n_lon


def _write_complete(path, nlat=12, nlon=24, npft=5, nyear=2, area_value=2.5e8):
    rng = np.random.default_rng(1)
    lat = np.linspace(-85.0, 85.0, nlat)
    lon = np.linspace(0.0, 360.0, nlon, endpoint=False)
    nlev = HWSD2_LAYER_DZ.shape[0]
    soil = lambda hi: rng.uniform(0.0, hi, size=(nlev, nlat, nlon))
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=HWSD2_LAYER_DZ,
        sand_pct=soil(60.0), clay_pct=soil(40.0), organic=soil(50.0),
        bulk_density=soil(1500.0),
        soil_color=rng.integers(1, 20, size=(nlat, nlon)).astype(np.float64),
        cell_area=np.full((nlat, nlon), area_value),
        year=np.arange(2000.0, 2000.0 + nyear),
        f_land=rng.uniform(0, 80, size=(nyear, nlat, nlon)),
        f_lake=rng.uniform(0, 20, size=(nyear, nlat, nlon)),
        f_glacier=rng.uniform(0, 10, size=(nyear, nlat, nlon)),
        pft_frac=rng.uniform(0, 100, size=(nyear, npft, nlat, nlon)),
        monthly_lai=rng.uniform(0, 6, size=(12, npft, nlat, nlon)),
        monthly_sai=rng.uniform(0, 2, size=(12, npft, nlat, nlon)),
        monthly_height_top=rng.uniform(0, 30, size=(12, npft, nlat, nlon)),
        monthly_height_bot=rng.uniform(0, 1, size=(12, npft, nlat, nlon)),
    )
    return npft, nyear, area_value


def test_write_soil_only_units_and_dims(tmp_path):
    p = str(tmp_path / "soil.nc")
    nlev = HWSD2_LAYER_DZ.shape[0]
    write_surfdata(
        p, lat=np.linspace(-80, 80, 6), lon=np.linspace(0, 350, 8),
        soil_dz=HWSD2_LAYER_DZ, sand_pct=np.zeros((nlev, 6, 8)),
    )
    ds = xr.open_dataset(p)
    assert ds["sand_pct"].dims == ("soil_layer", "lat", "lon")
    assert ds["sand_pct"].attrs["units"] == "percent"
    assert "clay_pct" not in ds          # group omitted -> not written
    assert ds["lat"].attrs["units"] == "degrees_north"
    ds.close()


def test_producer_to_consumer_roundtrip(tmp_path):
    p = str(tmp_path / "legoesm_surfdata.nc")
    npft, nyear, area_value = _write_complete(p)
    grid = _ToyGaussianGrid()
    cfg = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=p)
    soil_grid = make_soil_grid(SoilGridConfig())     # 8 model layers

    gsd = load_global_surface_data(cfg, grid, soil_grid=soil_grid)

    ncol = grid.ncol
    assert gsd.sand_frac.shape == (ncol, soil_grid.n_layers)
    assert gsd.pft_frac.shape == (nyear, ncol, npft)
    assert gsd.lai_monthly.shape == (12, ncol, npft)
    # percent -> fraction conversion applied
    assert jnp.all(gsd.sand_frac >= 0.0) and jnp.all(gsd.sand_frac <= 1.0)
    # PFT partition-of-unity and cover <= 1 after renorm
    assert jnp.allclose(jnp.sum(gsd.pft_frac, axis=-1), 1.0, atol=1e-5)
    assert jnp.all(gsd.f_land + gsd.f_lake + gsd.f_glacier <= 1.0 + 1e-5)
    # cell_area must NOT be km^2->m^2 rescaled (area_scale=1.0): stays ~area_value
    assert jnp.allclose(gsd.cell_area, area_value, rtol=1e-3)


def test_legoesm_preset_names():
    cfg = get_surfdata_preset("legoesm_surfdata")
    assert cfg.sand_var == "sand_pct" and cfg.lat_var == "lat"
    assert cfg.area_scale == 1.0
