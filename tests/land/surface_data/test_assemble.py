"""Integration test for v1 surfdata assembly (CLM cover/PFT/LAI + HWSD soil).

Writes tiny synthetic CLM + HWSD-soil NetCDFs, runs build_v1_surfdata, and loads
the result through the runtime loader to confirm the full producer->consumer
chain works on the v1 (CLM + HWSD) composition.
"""

import numpy as np
import jax.numpy as jnp
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.surface_data.assemble import build_v1_surfdata
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5
from legoesm.land.global_surface_data import get_surfdata_preset, load_global_surface_data
from legoesm.land.soil_grid import make_soil_grid, SoilGridConfig


def _write_clm(path, nlat=4, nlon=8):
    pct_nat_pft = np.zeros((15, nlat, nlon))
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    pct_nat_pft[be] = 100.0                         # all natural veg is BE-tropical
    natveg = np.full((nlat, nlon), 100.0)           # all land natural veg
    lai = np.zeros((12, 17, nlat, nlon)); lai[:, be] = 4.0
    latixy = np.broadcast_to(np.linspace(-80, 80, nlat)[:, None], (nlat, nlon)).copy()
    longxy = np.broadcast_to(np.linspace(0, 315, nlon)[None, :], (nlat, nlon)).copy()
    z = lambda: np.zeros((12, 17, nlat, nlon))
    zz = lambda: np.zeros((nlat, nlon))
    xr.Dataset({
        "PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), pct_nat_pft),
        "PCT_CFT": (("cft", "lsmlat", "lsmlon"), np.zeros((2, nlat, nlon))),
        "PCT_NATVEG": (("lsmlat", "lsmlon"), natveg),
        "PCT_CROP": (("lsmlat", "lsmlon"), zz()),
        "PCT_LAKE": (("lsmlat", "lsmlon"), zz()),
        "PCT_GLACIER": (("lsmlat", "lsmlon"), zz()),
        "LANDFRAC_PFT": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),  # all land
        "SOIL_COLOR": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),
        "MONTHLY_LAI": (("time", "lsmpft", "lsmlat", "lsmlon"), lai),
        "MONTHLY_SAI": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_TOP": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_BOT": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "LATIXY": (("lsmlat", "lsmlon"), latixy),
        "LONGXY": (("lsmlat", "lsmlon"), longxy),
    }).to_netcdf(path)


def _write_hwsd_soil(path, nlev=7, nlat=8, nlon=16, bulk=1300.0):
    lat = np.linspace(-85, 85, nlat); lon = np.linspace(0, 337.5, nlon)
    write_surfdata(
        path, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=np.full((nlev, nlat, nlon), 40.0),
        clay_pct=np.full((nlev, nlat, nlon), 20.0),
        organic=np.full((nlev, nlat, nlon), 5.0),
        bulk_density=np.full((nlev, nlat, nlon), bulk),
    )


class _ToyGrid:
    def __init__(self, nlat=6, nlon=12):
        latc = np.linspace(-80, 80, nlat); lonc = np.linspace(0, 330, nlon)
        self.lat2d, self.lon2d = np.meshgrid(np.deg2rad(latc), np.deg2rad(lonc), indexing="ij")
        self.grid_area = jnp.asarray(np.full(nlat * nlon, 1e9), jnp.float32)
        self.ncol = nlat * nlon


def test_build_v1_and_load(tmp_path):
    clm = str(tmp_path / "clm.nc"); soil = str(tmp_path / "soil.nc")
    out = str(tmp_path / "v1.nc")
    _write_clm(clm)
    _write_hwsd_soil(soil, bulk=1300.0)
    build_v1_surfdata(clm, soil, out)

    grid = _ToyGrid()
    cfg = get_surfdata_preset("legoesm_surfdata")._replace(surf_path=out)
    gsd = load_global_surface_data(cfg, grid, soil_grid=make_soil_grid(SoilGridConfig()))

    assert gsd.pft_frac.shape == (1, grid.ncol, N_PFT_CLM5)        # CLM5 17-PFT axis
    assert gsd.lai_monthly.shape == (12, grid.ncol, N_PFT_CLM5)
    # HWSD soil came through (CLM had no bulk density): ~1300 kg/m3 everywhere
    assert jnp.allclose(gsd.bulk_density, 1300.0, rtol=1e-2)
    assert jnp.allclose(gsd.sand_frac, 0.40, atol=1e-2)            # 40% -> fraction
    # land cells: PFT composition sums to 1 and BE-tropical dominates
    psum = jnp.sum(gsd.pft_frac[0], axis=-1)
    land = psum > 0.5
    assert jnp.allclose(psum[land], 1.0, atol=1e-4)
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    assert jnp.all(jnp.argmax(gsd.pft_frac[0][land], axis=-1) == be)
