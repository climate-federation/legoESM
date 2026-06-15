"""Unit test for ``build_local_latlon_band_grid`` — the staggered lat-band slice
of a ``LatLonCGridGeometry`` (foundation of the multi-GPU SPMD ocean step).

Verifies the C-grid stagger is respected: cell/u-row metrics (leading dim
n_lat) slice ``[a:b]``; v/q-row metrics (leading dim n_lat+1) slice ``[a:b+1]``
(shared bounding face). Values must equal the global metrics at those rows, and
a band partition must reassemble the global grid (cells tile, v-faces overlap by
one row at each interior band boundary). Runs on a single (CPU) device.

Run: ``JAX_ENABLE_X64=1 pytest tests/parallel/test_latlon_band_grid_slice.py``
"""
from __future__ import annotations

import os

os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import pytest

from legoesm.grids.latlon import create_latlon_grid
from legoesm.ocean.dynamics.sharded_ocean_step import (
    build_local_latlon_band_grid,
    _BAND_CELL_ROW_FIELDS,
    _BAND_V_ROW_FIELDS,
    _BAND_KEEP_FIELDS,
)


def _grid(n_lat=48, n_lon=96):
    return create_latlon_grid(n_lat=n_lat, n_lon=n_lon)


def test_band_shapes_and_values():
    """Grid-class-agnostic: iterate the ACTUAL grid fields and verify each is
    sliced per its stagger row (cell-row [a:b], v-row [a:b+1]) with values equal
    to the global rows. ``create_latlon_grid`` returns the regular ``LatLonGrid``
    (lat2d/f/dx/dy/area + lat_v); the curvilinear ``LatLonCGridGeometry`` shares
    the same classification sets and is covered by the union in the slicer."""
    n_lat, n_lon = 48, 96
    grid = _grid(n_lat, n_lon)
    a, b = 12, 24
    band = build_local_latlon_band_grid(grid, a, b)

    assert int(band.n_lat) == b - a
    assert int(band.n_lon) == n_lon

    n_cell = n_v = 0
    for nm in grid._fields:
        if nm in ("n_lat", "total_area"):
            continue
        g = np.asarray(getattr(grid, nm))
        bd = np.asarray(getattr(band, nm))
        if nm in _BAND_CELL_ROW_FIELDS:
            assert bd.shape[0] == b - a, f"{nm}: leading {bd.shape[0]} != {b-a}"
            np.testing.assert_allclose(bd, g[a:b], err_msg=f"{nm} band != rows")
            n_cell += 1
        elif nm in _BAND_V_ROW_FIELDS:
            assert bd.shape[0] == b - a + 1, f"{nm}: lead {bd.shape[0]}!={b-a+1}"
            np.testing.assert_allclose(bd, g[a:b + 1], err_msg=f"{nm} band!=rows")
            n_v += 1
        else:  # kept field — unchanged
            np.testing.assert_allclose(bd, g, err_msg=f"{nm} kept-field changed")
    assert n_cell > 0 and n_v > 0, "expected both cell-row and v-row fields"

    assert float(band.radius) == float(grid.radius)
    # total_area recomputed over the band's cells (LatLonGrid uses "area").
    # rtol is float32-summation-order tolerance (jnp.sum vs np.sum reduction
    # order differ; the grid metrics are float32).
    np.testing.assert_allclose(
        float(band.total_area), float(np.asarray(grid.area)[a:b].sum()),
        rtol=1e-5)


def test_every_field_classified():
    """No geometry field falls through unclassified (would mis-slice silently)."""
    grid = _grid()
    classified = _BAND_CELL_ROW_FIELDS | _BAND_V_ROW_FIELDS | _BAND_KEEP_FIELDS
    classified = set(classified) | {"n_lat", "total_area"}
    missing = set(grid._fields) - classified
    assert not missing, f"unclassified LatLonCGridGeometry fields: {missing}"


def test_band_partition_reassembles_global():
    """A contiguous band partition tiles the cell rows exactly and the v-rows
    overlap by one at every interior boundary (the shared face)."""
    n_lat, n_lon = 48, 96
    grid = _grid(n_lat, n_lon)
    bounds = [(0, 16), (16, 32), (32, 48)]
    # cell rows concatenate to the global
    area_cat = np.concatenate(
        [np.asarray(build_local_latlon_band_grid(grid, a, b).area)
         for a, b in bounds], axis=0)
    np.testing.assert_allclose(area_cat, np.asarray(grid.area))
    # interior v-row boundary is shared: band0's last v-row == band1's first
    b0 = build_local_latlon_band_grid(grid, *bounds[0])
    b1 = build_local_latlon_band_grid(grid, *bounds[1])
    np.testing.assert_allclose(np.asarray(b0.lat_v)[-1], np.asarray(b1.lat_v)[0],
                               err_msg="shared v-face mismatch at band boundary")


def test_out_of_range_raises():
    grid = _grid()
    with pytest.raises(ValueError):
        build_local_latlon_band_grid(grid, 0, 0)
    with pytest.raises(ValueError):
        build_local_latlon_band_grid(grid, 0, grid.n_lat + 1)


def test_unknown_field_raises():
    """Synthetic-violation self-test: an unclassified field must RAISE, proving
    the no-silent-skip guard is non-vacuous. Uses a minimal stand-in NamedTuple
    carrying a deliberately-unclassified field."""
    from collections import namedtuple

    FakeGrid = namedtuple(
        "FakeGrid", ["n_lat", "n_lon", "area_T", "lat_T", "dx_v",
                     "bogus_metric"])
    fake = FakeGrid(
        n_lat=16, n_lon=8,
        area_T=np.ones((16, 8)), lat_T=np.zeros((16, 8)),
        dx_v=np.ones((17, 8)), bogus_metric=np.zeros((16,)))
    with pytest.raises(KeyError, match="bogus_metric"):
        build_local_latlon_band_grid(fake, 0, 8)
