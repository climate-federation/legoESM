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
    landfrac = np.zeros((nlat, nlon)); landfrac[0, 0] = 1.0    # cell (0,0) fully land
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
            "LANDFRAC_PFT": (("lsmlat", "lsmlon"), landfrac),
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


def _clm_with_ocean_coast(nlat=1, nlon=3):
    """CLM cells (all 100% natveg, sum-to-100 landunits) differing only in LANDFRAC:
    [0]=land (LF 1.0), [1]=coastal (LF 0.2), [2]=ocean (LF 0.0). Exercises the gate:
    the raw landunit % (natveg=100) is identical, so only LANDFRAC distinguishes them."""
    xr = pytest.importorskip("xarray")
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    pct_nat_pft = np.zeros((15, nlat, nlon)); pct_nat_pft[be] = 100.0   # 100% BE-trop everywhere
    pct_cft = np.zeros((2, nlat, nlon))
    natveg = np.full((nlat, nlon), 100.0)                              # 100% of LAND is natveg
    crop = np.zeros((nlat, nlon))
    landfrac = np.array([[1.0, 0.2, 0.0]])                             # land / coast / ocean
    z = lambda: np.zeros((12, 17, nlat, nlon))
    latixy = np.zeros((nlat, nlon))
    longxy = np.broadcast_to(np.linspace(0, 240, nlon)[None, :], (nlat, nlon)).copy()
    return xr.Dataset({
        "PCT_NAT_PFT": (("natpft", "lsmlat", "lsmlon"), pct_nat_pft),
        "PCT_CFT": (("cft", "lsmlat", "lsmlon"), pct_cft),
        "PCT_NATVEG": (("lsmlat", "lsmlon"), natveg),
        "PCT_CROP": (("lsmlat", "lsmlon"), crop),
        "PCT_LAKE": (("lsmlat", "lsmlon"), np.zeros((nlat, nlon))),
        "PCT_GLACIER": (("lsmlat", "lsmlon"), np.zeros((nlat, nlon))),
        "LANDFRAC_PFT": (("lsmlat", "lsmlon"), landfrac),
        "SOIL_COLOR": (("lsmlat", "lsmlon"), np.ones((nlat, nlon))),
        "MONTHLY_LAI": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_SAI": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_TOP": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "MONTHLY_HEIGHT_BOT": (("time", "lsmpft", "lsmlat", "lsmlon"), z()),
        "LATIXY": (("lsmlat", "lsmlon"), latixy),
        "LONGXY": (("lsmlat", "lsmlon"), longxy),
    })


def test_clm5_reader_gates_cover_on_landfrac():
    """f_land / f_lake / f_glacier and pft_frac must be gated by LANDFRAC_PFT so the
    percent-of-LAND landunit values become percent-of-GRIDCELL. Ocean/coastal cells
    carry the SAME raw natveg=100 as land; without the gate they'd all read 100% land."""
    d = read_clm5_cover_veg(dataset=_clm_with_ocean_coast())
    # land: 100*1.0=100 | coast: 100*0.2=20 | ocean: 100*0.0=0  (NOT 100/100/100)
    np.testing.assert_allclose(d["f_land"][0], [100.0, 20.0, 0.0])
    # invariant preserved: pft_frac (also gated) still sums to f_land
    np.testing.assert_allclose(d["pft_frac"].sum(axis=0)[0], [100.0, 20.0, 0.0])
    # ocean cell has zero land -> zero PFT cover
    np.testing.assert_allclose(d["pft_frac"][:, 0, 2], 0.0)


def test_assert_cover_within_landfrac_tripwire():
    """The tripwire fires on UN-gated (percent-of-land) cover -- proves it is not
    vacuous: a coastal cell with f_land=100 but LANDFRAC=0.2 must raise."""
    from legoesm.land.surface_data.sources.clm5_surfdata import (
        assert_cover_within_landfrac)
    z = np.zeros((1, 3))
    landfrac = np.array([[1.0, 0.2, 0.0]])
    gated = np.array([[100.0, 20.0, 0.0]])       # correctly gated -> passes
    assert_cover_within_landfrac(gated, z, z, landfrac)
    ungated = np.array([[100.0, 100.0, 100.0]])  # un-gated: land in coast/ocean cells
    with pytest.raises(ValueError, match="exceeds land fraction"):
        assert_cover_within_landfrac(ungated, z, z, landfrac)


def test_clm5_reader_pft_axis_matches_surface_params():
    d = read_clm5_cover_veg(dataset=_synthetic_clm())
    assert list(d["pft_names"]) == list(CLM5_PFT_NAMES)


def test_reconstruct_clm5_pft_frac_crop_split():
    """The shared landunit→17-PFT reconstruction (used by BOTH the surfdata reader
    and the coupled-AMIP LAI loader) splits crop cover across cft (crop_c3/crop_c4)
    via PCT_CFT and weights natural PFTs by PCT_NATVEG."""
    from legoesm.land.surface_data.sources.clm5_surfdata import reconstruct_clm5_pft_frac

    natveg = np.array([[80.0]]); crop = np.array([[20.0]])   # (1, 1) percent gridcell
    be = CLM5_PFT_NAMES.index("broadleaf_evergreen_tropical")
    nat_pft = np.zeros((15, 1, 1)); nat_pft[be, 0, 0] = 100.0
    cft = np.zeros((2, 1, 1)); cft[0, 0, 0] = 50.0; cft[1, 0, 0] = 50.0  # 50/50 c3/c4
    pf = reconstruct_clm5_pft_frac(natveg, crop, nat_pft, cft)
    assert pf.shape == (N_PFT_CLM5, 1, 1)
    np.testing.assert_allclose(pf[be, 0, 0], 80.0)                       # 80% * 100%
    np.testing.assert_allclose(pf[CLM5_PFT_NAMES.index("crop_c3"), 0, 0], 10.0)  # 20%*50%
    np.testing.assert_allclose(pf[CLM5_PFT_NAMES.index("crop_c4"), 0, 0], 10.0)  # 20%*50%
    np.testing.assert_allclose(pf.sum(), 100.0)                          # = f_land
    # PFT-count guard (wrong natpft count) raises rather than silently mis-aligning.
    with pytest.raises(ValueError, match="expected"):
        reconstruct_clm5_pft_frac(natveg, crop, np.zeros((14, 1, 1)), cft)
