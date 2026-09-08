"""Direct tests for scripts/validate/compare_lmip_biophys_clm.py.

The figure functions need Derecho data; these tests pin the pure numerics the
figures stand on — spherical cell areas, the zonal-sum/global-total identity,
the local-season composite, and the biome partition — each against an answer
known in closed form.
"""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest
import xarray as xr

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "validate" / "compare_lmip_biophys_clm.py")


@pytest.fixture(scope="module")
def mod():
    spec = importlib.util.spec_from_file_location("compare_lmip_biophys_clm",
                                                  _SCRIPT)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _regular_grid(nlat=45, nlon=90):
    lat = np.linspace(-90 + 180 / (2 * nlat), 90 - 180 / (2 * nlat), nlat)
    lon = np.linspace(180 / nlon, 360 - 180 / nlon, nlon)
    return lat, lon


def test_constants_come_from_legoesm(mod):
    from legoesm import constants
    assert mod.LV == constants.L_v
    assert mod.R_EARTH == constants.R_earth


def test_cell_area_sums_to_sphere(mod):
    lat, lon = _regular_grid()
    a = mod.cell_area(lat, lon)
    sphere = 4.0 * np.pi * mod.R_EARTH ** 2
    assert float(a.sum()) == pytest.approx(sphere, rel=1e-12)


def test_bounds_1d_midpoints_and_arity(mod):
    c = np.array([0.0, 2.0, 4.0, 8.0])
    b = mod._bounds_1d(c)
    assert b.shape == (5,)
    np.testing.assert_allclose(b[1:-1], [1.0, 3.0, 6.0])
    # end cells extend symmetrically about their centers
    assert b[0] == pytest.approx(-1.0) and b[-1] == pytest.approx(10.0)


def test_zonal_gpp_area_under_curve_is_global_total(mod):
    """A uniform 1 gC/m2/day field: global_total(zonal_gpp(...)) must equal
    the direct per-cell sum  sum(w) * 365 / 1e15  PgC/yr."""
    lat, lon = _regular_grid(nlat=30, nlon=60)
    w = mod.cell_area(lat, lon)
    clim = xr.DataArray(np.ones((12, lat.size, lon.size)),
                        coords={"month": np.arange(1, 13),
                                "lat": lat, "lon": lon},
                        dims=("month", "lat", "lon"))
    z = mod.zonal_gpp(clim, w)
    expected = float(w.sum()) * mod.DAYS_YR / 1e15
    assert mod.global_total(z) == pytest.approx(expected, rel=1e-12)


def test_local_season_shifts_only_southern_hemisphere(mod):
    """Value == month everywhere; after compositing, an NH cell still reads
    its own month while an SH cell reads month+6 (mod 12)."""
    lat = np.array([-45.0, 45.0])
    lon = np.array([0.0, 180.0])
    months = np.arange(1, 13)
    clim = xr.DataArray(
        np.tile(months[:, None, None], (1, 2, 2)).astype(float),
        coords={"month": months, "lat": lat, "lon": lon},
        dims=("month", "lat", "lon"))
    out = mod.local_season(clim)
    nh = out.sel(lat=45.0, lon=0.0).values
    sh = out.sel(lat=-45.0, lon=0.0).values
    np.testing.assert_allclose(nh, months)
    np.testing.assert_allclose(sh, (months - 1 + 6) % 12 + 1)


def test_biome_pfts_partition_all_17_clm5_pfts(mod):
    """The 8 biome classes cover CLM5 PFT indices 0..16 exactly once —
    a double-assigned or dropped PFT silently corrupts every biome mean."""
    all_pfts = sorted(i for v in mod.BIOME_PFTS.values() for i in v)
    assert all_pfts == list(range(17))
    assert list(mod.BIOME_PFTS) == mod.BIOMES
