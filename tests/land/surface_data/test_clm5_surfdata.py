"""Unit test for the CLM5 surfdata cover/PFT/LAI reader (landunit reconstruction)."""

import numpy as np
import pytest

from legoesm.land.surface_data.sources.clm5_surfdata import read_clm5_cover_veg
from legoesm.land.surface_params import CLM5_PFT_NAMES, N_PFT_CLM5


def _synthetic_clm(nlat=3, nlon=4):
    xr = pytest.importorskip("xarray")
    # one fully-described cell (0,0): 80% natveg (all broadleaf-evergreen-tropical),
    # 20% crop (all cft0 -> crop_c3); rest zero.
    natpft, cft = 15, 2
    pct_nat_pft = np.zeros((natpft, nlat, nlon))
    be_trop = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")   # within natpft (0..14)
    pct_nat_pft[be_trop, 0, 0] = 100.0
    pct_cft = np.zeros((cft, nlat, nlon)); pct_cft[0, 0, 0] = 100.0
    natveg = np.zeros((nlat, nlon)); natveg[0, 0] = 80.0
    crop = np.zeros((nlat, nlon)); crop[0, 0] = 20.0
    lai = np.zeros((12, 17, nlat, nlon)); lai[:, be_trop, 0, 0] = 5.0
    latixy = np.broadcast_to(np.linspace(-60, 60, nlat)[:, None], (nlat, nlon)).copy()
    longxy = np.broadcast_to(np.linspace(0, 270, nlon)[None, :], (nlat, nlon)).copy()
    z = lambda: np.zeros((12, 17, nlat, nlon))
    return xr.Dataset(
        {
            "PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), pct_nat_pft),
            "PCT_CFT": (("cft", "lsmlat", "lsmlon"), pct_cft),
            "PCT_NATVEG": (("lsmlat", "lsmlon"), natveg),
            "PCT_CROP": (("lsmlat", "lsmlon"), crop),
            "PCT_LAKE": (("lsmlat", "lsmlon"), np.zeros((nlat, nlon))),
            "PCT_GLACIER": (("lsmlat", "lsmlon"), np.zeros((nlat, nlon))),
            "SOIL_COLOR": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),
            "MONTHLY_LAI": (("time", "lsmpft", "lsmlat", "lsmlon"), lai),
            "MONTHLY_SAI": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
            "MONTHLY_HEIGHT_TOP": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
            "MONTHLY_HEIGHT_BOT": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
            "LATIXY": (("lsmlat", "lsmlon"), latixy),
            "LONGXY": (("lsmlat", "lsmlon"), longxy),
        }
    )


def test_clm5_reader_landunit_reconstruction():
    d = read_clm5_cover_veg(dataset=_synthetic_clm())
    assert d["pft_frac"].shape == (N_PFT_CLM5, 3, 4)
    assert d["monthly_lai"].shape == (12, N_PFT_CLM5, 3, 4)
    # 1-D coords recovered from 2-D LATIXY/LONGXY
    np.testing.assert_allclose(d["lat"][0], -60.0)
    np.testing.assert_allclose(d["lon"][0], 0.0)
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    cc3 = CLM5_PFT_NAMES.index("crop_c3")
    # natveg 80% * 100% BE-trop -> 80; crop 20% * 100% cft0 (crop_c3) -> 20
    np.testing.assert_allclose(d["pft_frac"][be, 0, 0], 80.0)
    np.testing.assert_allclose(d["pft_frac"][cc3, 0, 0], 20.0)
    # sum of 17-PFT % equals f_land (= natveg + crop)
    np.testing.assert_allclose(d["pft_frac"].sum(axis=0), d["f_land"])
    np.testing.assert_allclose(d["f_land"][0, 0], 100.0)


def test_clm5_reader_pft_axis_matches_surface_params():
    d = read_clm5_cover_veg(dataset=_synthetic_clm())
    assert list(d["pft_names"]) == list(CLM5_PFT_NAMES)
