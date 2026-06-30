"""Unit coverage for the WOA18 -> AMIP SST/SIC forcing builder."""

import numpy as np

from legoesm import constants
from scripts.data.build_amip_sst_from_woa import (
    fill_land_nearest,
    regrid_to_amip,
    sst_to_sic,
)

_FREEZE_C = float(constants.T_freeze_ocean - constants.T_freeze)  # -1.8 degC


def test_sst_to_sic_freezing_ramp_monotone_and_bounded():
    """Below freezing -> full ice; warm -> open water; monotone ramp in [0,1]."""
    s = sst_to_sic(np.array([_FREEZE_C - 1.0, _FREEZE_C, _FREEZE_C + 0.25,
                             _FREEZE_C + 0.5, _FREEZE_C + 5.0]), ramp_C=0.5)
    assert s[0] == 1.0          # colder than freezing -> pack ice
    assert s[1] == 1.0          # at freezing -> still ice
    assert 0.0 < s[2] < 1.0     # mid-ramp
    assert s[3] == 0.0          # top of ramp -> open water
    assert s[4] == 0.0          # warm -> open water
    assert np.all((s >= 0.0) & (s <= 1.0))
    assert np.all(np.diff(s) <= 0.0)   # non-increasing with SST


def test_regrid_constant_field_no_pole_blowup():
    """A constant WOA field regrids to the same constant on the AMIP grid,
    including the +-90 pole rows that lie outside WOA's -89.5..89.5 (edge-pad,
    not extrapolation)."""
    woa_lat = np.linspace(-89.5, 89.5, 60)
    woa_lon = np.linspace(-179.5, 179.5, 120)
    field = np.full((60, 120), 14.0)
    tgt_lat = np.linspace(-90.0, 90.0, 37)
    tgt_lon = np.linspace(0.0, 357.5, 72)
    out = regrid_to_amip(field, woa_lat, woa_lon, tgt_lat, tgt_lon)
    assert out.shape == (37, 72)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 14.0)


def test_regrid_preserves_latitudinal_structure_within_range():
    """A cos(lat) field stays within [min,max] (no interpolation overshoot) and
    keeps its equator-warm structure after regridding."""
    woa_lat = np.linspace(-89.5, 89.5, 60)
    woa_lon = np.linspace(-179.5, 179.5, 120)
    field = (30.0 * np.cos(np.deg2rad(woa_lat)))[:, None] * np.ones((60, 120))
    tgt_lat = np.linspace(-90.0, 90.0, 37)
    tgt_lon = np.linspace(0.0, 357.5, 72)
    out = regrid_to_amip(field, woa_lat, woa_lon, tgt_lat, tgt_lon)
    assert out.min() >= -1e-6 and out.max() <= 30.0 + 1e-6
    eq = int(np.argmin(np.abs(tgt_lat)))
    assert out[eq].mean() > out[0].mean()
    assert out[eq].mean() > out[-1].mean()


def test_regrid_rolls_longitude_into_0_360():
    """A lon-varying field is carried correctly when WOA's -180..180 is rolled
    to 0..360: a feature placed at WOA lon=-90 must land near AMIP lon=270."""
    woa_lat = np.linspace(-89.5, 89.5, 40)
    woa_lon = np.linspace(-179.5, 179.5, 120)
    # warm spike at lon=-90 (=> 270 in 0..360), cool elsewhere
    field = np.full((40, 120), 5.0)
    j = int(np.argmin(np.abs(woa_lon - (-90.0))))
    field[:, j] = 25.0
    tgt_lat = np.linspace(-89.0, 89.0, 40)
    tgt_lon = np.linspace(0.0, 357.5, 144)
    out = regrid_to_amip(field, woa_lat, woa_lon, tgt_lat, tgt_lon)
    i270 = int(np.argmin(np.abs(tgt_lon - 270.0)))
    i90 = int(np.argmin(np.abs(tgt_lon - 90.0)))
    assert out[:, i270].mean() > out[:, i90].mean() + 5.0


def test_fill_land_nearest_fills_holes_periodic():
    """NaN land cells are replaced with the nearest valid value; a constant
    ocean field stays constant after filling, and the dateline fills across the
    lon seam."""
    f = np.full((10, 20), 5.0)
    v = np.ones((10, 20), dtype=bool)
    f[3:6, 4:8] = np.nan
    v[3:6, 4:8] = False
    # a hole straddling the seam (cols 19 and 0)
    f[0, 19] = np.nan
    v[0, 19] = False
    out = fill_land_nearest(f, v)
    assert np.all(np.isfinite(out))
    assert np.allclose(out, 5.0)
