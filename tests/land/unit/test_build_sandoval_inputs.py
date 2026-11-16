"""Sandoval phi0 inputs: the 1-D -> lat-lon builder
(scripts/data/build_sandoval_inputs.py) and the model-grid mapping in
scripts/run/run_lmip_biophys.py (``sandoval_inputs_on_grid``), chained through a
NetCDF file exactly as a global run reads it."""

from __future__ import annotations

import importlib.util
import pathlib

import numpy as np
import pytest

_REPO = pathlib.Path(__file__).resolve().parents[3]


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, _REPO / rel)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


@pytest.fixture(scope="module")
def builder():
    return _load("scripts/data/build_sandoval_inputs.py", "build_sandoval_inputs")


@pytest.fixture(scope="module")
def driver():
    return _load("scripts/run/run_lmip_biophys.py", "run_lmip_biophys")


def test_scatter_places_points_on_their_cells_and_refuses_bad_layouts(builder):
    lat, lon, g = builder.scatter_land_points(
        [-89.75, 0.25, 0.25], [-179.75, 10.25, 190.25],   # 190.25 E == -169.75
        {"x": [1.0, 2.0, 3.0]}, res_deg=0.5)
    assert lat.size == 360 and lon.size == 720
    x = g["x"]
    assert np.isfinite(x).sum() == 3
    assert x[0, 0] == 1.0
    assert x[np.searchsorted(lat, 0.25), np.searchsorted(lon, 10.25)] == 2.0
    assert x[np.searchsorted(lat, 0.25), np.searchsorted(lon, -169.75)] == 3.0
    with pytest.raises(ValueError, match="cell centres"):
        builder.scatter_land_points([0.25, 0.75, 0.3], [0.25, 0.25, 0.25],
                                    {"x": [1.0, 2.0, 3.0]}, res_deg=0.5)
    with pytest.raises(ValueError, match="same grid cell"):
        builder.scatter_land_points([0.25, 0.25, 0.75], [0.25, 0.25, 0.25],
                                    {"x": [1.0, 2.0, 3.0]}, res_deg=0.5)


def test_grid_mapping_geometric_ai_arithmetic_mgdd0_and_nearest_fill(
        builder, driver, tmp_path):
    """Model cell = area mean over its land source cells; AI in log space.
    Two equal-area source cells with AI 1 and 100 must give 10 (geometric),
    not 50.5 (arithmetic); their mGDD0 5 and 15 must give 10.  A model cell
    with no land takes its nearest filled neighbour's values."""
    import xarray as xr
    grid = driver.make_grid("latlon", 4)               # 4 x 8, 45 deg cells
    tlat = np.rad2deg(np.asarray(grid.lat))
    tlon = np.rad2deg(np.asarray(grid.lon))
    # Land points (0.5 deg cell centres) inside ONE model cell, same latitude
    # row so their areas are equal.
    i0, j0 = 2, 2
    y = tlat[i0] + 0.25
    xs = [tlon[j0] + 0.25, tlon[j0] + 0.75]
    lat, lon, g = builder.scatter_land_points(
        [y, y], xs, {"aridity_index": [1.0, 100.0], "mgdd0": [5.0, 15.0]}, res_deg=0.5)
    f = tmp_path / "sv.nc"
    xr.Dataset({k: (("lat", "lon"), v) for k, v in g.items()},
               coords={"lat": lat, "lon": lon}).to_netcdf(f)
    ai, mg = driver.sandoval_inputs_on_grid(str(f), grid)
    assert ai.shape == (32,) and np.isfinite(ai).all() and np.isfinite(mg).all()
    k = i0 * tlon.size + j0
    np.testing.assert_allclose(ai[k], 10.0, rtol=1e-12)
    np.testing.assert_allclose(mg[k], 10.0, rtol=1e-12)
    # Every other cell has no land and is filled from the one filled cell.
    np.testing.assert_allclose(ai, 10.0, rtol=1e-12)


def test_grid_mapping_refuses_non_latlon_grids_and_nonpositive_ai(
        builder, driver, tmp_path):
    import xarray as xr
    lat, lon, g = builder.scatter_land_points(
        [0.25], [0.25], {"aridity_index": [0.0], "mgdd0": [5.0]}, res_deg=0.5)
    f = tmp_path / "sv.nc"
    xr.Dataset({k: (("lat", "lon"), v) for k, v in g.items()},
               coords={"lat": lat, "lon": lon}).to_netcdf(f)
    with pytest.raises(ValueError, match="aridity_index must be > 0"):
        driver.sandoval_inputs_on_grid(str(f), driver.make_grid("latlon", 4))
    with pytest.raises(ValueError, match="regular lat-lon"):
        driver.sandoval_inputs_on_grid(str(f), driver.make_grid("cubed_sphere", 4))
