"""Unit tests for legoesm.ocean.fidelity.report_grid."""

from __future__ import annotations

import math

import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.fidelity import report_grid


@pytest.mark.parametrize(
    "res,shape",
    [
        ("0p5deg", (360, 720)),
        ("1deg", (180, 360)),
        ("2deg", (90, 180)),
    ],
)
def test_get_report_grid_known_resolutions(res, shape):
    g = report_grid.get_report_grid(res)
    assert (g.n_lat, g.n_lon) == shape


def test_get_report_grid_default_is_1deg():
    g = report_grid.get_report_grid()
    assert (g.n_lat, g.n_lon) == (180, 360)


def test_get_report_grid_unknown_raises():
    with pytest.raises(ValueError, match="Unknown report-grid resolution"):
        report_grid.get_report_grid("3deg")  # type: ignore[arg-type]


def test_lat_edges_span_pole_to_pole():
    g = report_grid.get_report_grid("1deg")
    assert float(g.lat_edges[0]) == pytest.approx(-math.pi / 2)
    assert float(g.lat_edges[-1]) == pytest.approx(math.pi / 2)
    assert g.lat_edges.shape == (g.n_lat + 1,)


def test_lon_edges_span_full_circle():
    g = report_grid.get_report_grid("1deg")
    assert float(g.lon_edges[0]) == pytest.approx(0.0)
    assert float(g.lon_edges[-1]) == pytest.approx(2.0 * math.pi)
    assert g.lon_edges.shape == (g.n_lon + 1,)


def test_centers_between_edges():
    g = report_grid.get_report_grid("2deg")
    assert g.lat_centers.shape == (g.n_lat,)
    assert g.lon_centers.shape == (g.n_lon,)
    assert np.all(g.lat_centers > g.lat_edges[:-1])
    assert np.all(g.lat_centers < g.lat_edges[1:])
    assert np.all(g.lon_centers > g.lon_edges[:-1])
    assert np.all(g.lon_centers < g.lon_edges[1:])


def test_cell_area_sum_matches_sphere_surface():
    g = report_grid.get_report_grid("1deg")
    total = float(g.cell_area.sum())
    sphere = 4.0 * math.pi * constants.R_earth ** 2
    np.testing.assert_allclose(total, sphere, rtol=1e-10)


def test_cell_area_strictly_positive():
    g = report_grid.get_report_grid("1deg")
    assert np.all(g.cell_area > 0)
    assert g.cell_area.shape == (g.n_lat, g.n_lon)


def test_cell_area_smaller_at_pole_than_equator():
    g = report_grid.get_report_grid("1deg")
    equator_idx = g.n_lat // 2
    assert g.cell_area[equator_idx, 0] > g.cell_area[0, 0]
    assert g.cell_area[equator_idx, 0] > g.cell_area[-1, 0]


def test_report_grid_is_frozen():
    g = report_grid.get_report_grid("2deg")
    with pytest.raises(Exception):
        g.n_lat = 99  # type: ignore[misc]
