"""Unit test for the single-point surfdata extractor."""

import numpy as np
import pytest

xr = pytest.importorskip("xarray")

from legoesm.land.boundary_data import surface_params_at_point
from legoesm.land.surface_data.schema import write_surfdata
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5


def _write(tmp_path, nlat=4, nlon=8):
    lat = np.linspace(-80, 80, nlat)        # ascending
    lon = np.linspace(0, 315, nlon)
    nlev = 7
    pft = np.zeros((1, N_PFT_CLM5, nlat, nlon))
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    pft[0, be, 1, 2] = 80.0                  # cell (1,2): BE-tropical dominant
    lai = np.zeros((12, N_PFT_CLM5, nlat, nlon)); lai[:, be, 1, 2] = 5.0
    p = str(tmp_path / "sd.nc")
    write_surfdata(
        p, lat=lat, lon=lon, soil_dz=np.full(nlev, 0.2),
        sand_pct=np.full((nlev, nlat, nlon), 40.0),
        clay_pct=np.full((nlev, nlat, nlon), 20.0),
        organic=np.full((nlev, nlat, nlon), 5.0),
        bulk_density=np.full((nlev, nlat, nlon), 1300.0),
        soil_color=np.full((nlat, nlon), 7.0),
        year=np.array([2015.0]),
        f_land=np.full((1, nlat, nlon), 100.0),
        f_lake=np.zeros((1, nlat, nlon)), f_glacier=np.zeros((1, nlat, nlon)),
        pft_frac=pft, monthly_lai=lai,
        monthly_sai=np.zeros_like(lai), monthly_height_top=np.zeros_like(lai),
        monthly_height_bot=np.zeros_like(lai),
    )
    return p, lat, lon


def test_point_extract_nearest_and_fields(tmp_path):
    p, lat, lon = _write(tmp_path)
    # query near cell (1,2)
    pp = surface_params_at_point(p, lat[1] + 1.0, lon[2] - 1.0)
    assert pp.dominant_pft == "broadleaf_evergreen_tropical"
    assert np.isclose(pp.lai_monthly.max(), 5.0)
    assert np.isclose(pp.sand_pct, 40.0) and np.isclose(pp.clay_pct, 20.0)
    assert pp.soil_color == 7


def test_point_longitude_wrap(tmp_path):
    p, lat, lon = _write(tmp_path)
    # lon 315 (file) should be found when querying -45 (== 315 mod 360)
    pp = surface_params_at_point(p, lat[1], -45.0)
    assert np.isclose(pp.lon % 360.0, 315.0 % 360.0, atol=lon[1] - lon[0])
