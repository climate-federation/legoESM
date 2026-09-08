"""The land-IC regridder: column order fidelity and identity round-trip."""
from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, _REPO / path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_column_order_matches_the_spinup_driver():
    """The regridder addresses columns exactly as the spin-up driver did.

    The source state's column order was DEFINED by run_lmip_biophys's
    grid_latlon_rad; the regridder duplicates that logic (run scripts are not
    importable at package level), so this pins the two against drift — a
    silent divergence would scramble every soil column geographically while
    keeping all shapes valid.
    """
    rg = _load("scripts/data/regrid_land_ic.py", "_rg")
    lb = _load("scripts/run/run_lmip_biophys.py", "_lb")

    class FakeLatLon:
        lat2d = np.linspace(-1.4, 1.4, 12).reshape(3, 4)
        lon2d = np.linspace(0.0, 6.2, 12).reshape(3, 4)

    class FakeMesh:
        latCell = np.linspace(-1.5, 1.5, 7)
        lonCell = np.linspace(0.0, 6.0, 7)

    for g in (FakeLatLon(), FakeMesh()):
        la, lo = rg._cols_rad(g)
        lb_la, lb_lo = lb.grid_latlon_rad(g)
        np.testing.assert_array_equal(la, np.asarray(lb_la))
        np.testing.assert_array_equal(lo, np.asarray(lb_lo))


def test_nearest_neighbour_identity():
    """Each source point is its own nearest neighbour — regrid to the same
    points returns the identity map, so state passes through untouched."""
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    rng = np.random.default_rng(0)
    lat = rng.uniform(-1.4, 1.4, 40)
    lon = rng.uniform(0.0, 6.2, 40)
    idx = nn(lat, lon, lat, lon)
    np.testing.assert_array_equal(idx, np.arange(40))


def test_nearest_neighbour_ignores_non_land_sources():
    """A target next to an excluded source gets the nearest INCLUDED one."""
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    near = nn(np.array([0.0, 0.5]), np.array([0.0, 0.0]),
              np.array([0.01]), np.array([0.0]),
              src_valid=np.array([False, True]))
    assert near[0] == 1


def test_regridding_latitude_returns_latitude():
    """Regrid the latitude field itself; a registration error is visible.

    The order pin and identity test above share the reconstruction logic with
    the script, so a CONSISTENTLY wrong convention (longitude 0-360 versus
    +/-180, pole ordering) would pass both.  Regridding a field that IS the
    coordinate catches that: each target column must receive a source latitude
    within one source cell of its own.
    """
    from legoesm.coupler.grid_remap import nearest_column_map as nn
    src_lat, src_lon = np.meshgrid(
        np.deg2rad(np.arange(-89.0, 90.0, 2.0)),
        np.deg2rad(np.arange(0.0, 360.0, 2.0)), indexing="ij")
    sl, so = src_lat.ravel(), src_lon.ravel()
    rng = np.random.default_rng(1)
    tl = rng.uniform(-1.4, 1.4, 500)
    to = rng.uniform(-3.1, 6.2, 500)      # spans BOTH longitude conventions
    idx = nn(sl, so, tl, to)
    err = np.abs(sl[idx] - tl)
    assert err.max() < np.deg2rad(2.0), (
        f"a target column received a latitude {np.rad2deg(err.max()):.1f} deg "
        "away: the regridder's geometry is mis-registered")
