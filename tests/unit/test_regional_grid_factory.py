"""Uniform regional-grid instantiation — any limited-area grid via one entry."""

from __future__ import annotations

import pytest

from legoesm.grids.factory import REGIONAL_GRID_TYPES, create_regional_grid


def test_regional_grid_types() -> None:
    assert REGIONAL_GRID_TYPES == (
        "latlon", "latlon_stretched", "mercator", "mpas", "cubed_sphere",
    )


def test_regional_latlon_returns_grid_and_mask() -> None:
    grid, mask = create_regional_grid(
        "latlon", n_lat=8, n_lon=8, lat_south=20.0, lat_north=60.0
    )
    assert type(grid).__name__ == "LatLonGrid"
    # wall_mask spans the walled domain (interior + N/S, and E/W when not periodic)
    assert mask.shape == grid.grid_shape_2d


def test_regional_latlon_periodic_channel() -> None:
    """periodic_x gives a zonal channel (no E/W walls) — kwargs pass through."""
    grid, _mask = create_regional_grid(
        "latlon", n_lat=8, n_lon=12, lat_south=-60.0, lat_north=-30.0, periodic_x=True
    )
    assert type(grid).__name__ == "LatLonGrid"


def test_regional_mercator_returns_bare_grid() -> None:
    grid = create_regional_grid("mercator", n_lon=16, lat_max_deg=60.0)
    assert type(grid).__name__ == "LatLonGrid"


def test_regional_mpas_returns_mesh() -> None:
    grid = create_regional_grid(
        "mpas", lon_range=(0.0, 30.0), lat_range=(20.0, 50.0), resolution_km=600.0
    )
    assert type(grid).__name__ == "VoronoiMesh"


def test_regional_cubed_sphere_panel() -> None:
    grid = create_regional_grid("cubed_sphere", n=8, face_id=2)
    assert type(grid).__name__ == "CubedSphereGrid"
    # a single-face panel's coordinate arrays have a leading face dim of 1 (not 6)
    assert grid.grid_lat.shape[0] == 1


def test_unknown_regional_grid_raises() -> None:
    with pytest.raises(ValueError, match="Unknown regional grid_type"):
        create_regional_grid("flat_earth", n=4)
