"""Unit tests for the Luo et al. (2024) C4 source."""

import numpy as np
import pytest

from legoesm.land.surface_data.sources.c4_fraction import (
    load_c4_areas,
    c4_fraction_of_grass,
)


def _synthetic_c4(nlat=6, nlon=12):
    xr = pytest.importorskip("xarray")
    lat = -90.0 + (np.arange(nlat) + 0.5) * (180.0 / nlat)   # ascending (like Luo)
    lon = -180.0 + (np.arange(nlon) + 0.5) * (360.0 / nlon)
    years = np.array([2009.0, 2010.0, 2011.0])
    # file layout: dims (years, lon, lat)
    grass = np.zeros((3, nlon, nlat), np.float32)
    grass[1] = 50.0                                          # only 2010 = 50%
    crop = np.full((3, nlon, nlat), 10.0, np.float32)
    return xr.Dataset(
        {"C4_grass_area": (("years", "lon", "lat"), grass),
         "C4_crop_area": (("years", "lon", "lat"), crop)},
        coords={"years": years, "lat": lat, "lon": lon},
    )


def test_load_c4_areas_year_select_and_regrid():
    ds = _synthetic_c4()
    tgt_lat = 90.0 - (np.arange(4) + 0.5) * 45.0             # 0.25-style descending
    tgt_lon = -180.0 + (np.arange(8) + 0.5) * 45.0
    out = load_c4_areas(None, tgt_lat, tgt_lon, year=2010, dataset=ds)
    assert out["c4_grass_area"].shape == (4, 8)
    # constant 50% in 2010 -> fraction 0.5 everywhere after IDW of a constant
    np.testing.assert_allclose(out["c4_grass_area"], 0.5, atol=1e-4)
    np.testing.assert_allclose(out["c4_crop_area"], 0.1, atol=1e-4)


def test_load_c4_areas_picks_correct_year():
    ds = _synthetic_c4()
    tgt_lat = np.array([45.0, -45.0]); tgt_lon = np.array([-90.0, 90.0])
    # year 2009 had zero grass C4
    out = load_c4_areas(None, tgt_lat, tgt_lon, year=2009, dataset=ds)
    np.testing.assert_allclose(out["c4_grass_area"], 0.0, atol=1e-4)


def test_c4_fraction_of_grass():
    c4 = np.array([0.2, 0.5, 0.0, 0.3])
    grass = np.array([0.4, 0.4, 0.0, 0.1])     # cell3: more C4 area than grass cover
    frac = c4_fraction_of_grass(c4, grass)
    np.testing.assert_allclose(frac[0], 0.5)   # 0.2/0.4
    np.testing.assert_allclose(frac[1], 1.0)   # 0.5/0.4 -> clipped
    np.testing.assert_allclose(frac[2], 0.0)   # no grass
    np.testing.assert_allclose(frac[3], 1.0)   # 0.3/0.1 -> clipped
