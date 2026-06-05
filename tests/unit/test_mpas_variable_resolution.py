"""MPAS variable-resolution mesh via density-weighted Lloyd (refinement gap)."""

from __future__ import annotations

import numpy as np
import pytest

from legoesm.grids.voronoi import create_voronoi_mesh


def _greatcircle(lat_c, lon_c, lat0, lon0):
    return np.arccos(np.clip(
        np.sin(lat_c) * np.sin(lat0)
        + np.cos(lat_c) * np.cos(lat0) * np.cos(lon_c - lon0), -1.0, 1.0))


def test_density_none_reproduces_uniform_mesh() -> None:
    """density_fn=None is byte-identical to the legacy uniform SCVT."""
    a = create_voronoi_mesh(3, lloyd_iterations=8)
    b = create_voronoi_mesh(3, lloyd_iterations=8, density_fn=None)
    assert np.array_equal(np.asarray(a.latCell), np.asarray(b.latCell))
    assert np.array_equal(np.asarray(a.areaCell), np.asarray(b.areaCell))


def test_density_refines_the_high_density_region() -> None:
    """A density bump concentrates cells there -> smaller cells in-region than out."""
    lat0, lon0 = 0.4, 0.0

    def density(lat, lon):
        d = _greatcircle(np.asarray(lat), np.asarray(lon), lat0, lon0)
        return 1.0 + 9.0 * np.exp(-(d ** 2) / (0.35 ** 2))  # 10x finer at centre

    mesh = create_voronoi_mesh(4, lloyd_iterations=60, density_fn=density)
    lat_c = np.asarray(mesh.latCell)
    lon_c = np.asarray(mesh.lonCell)
    area = np.asarray(mesh.areaCell)

    dist = _greatcircle(lat_c, lon_c, lat0, lon0)
    near = dist < 0.4
    far = dist > 1.2
    assert near.sum() > 5 and far.sum() > 5  # both samples populated
    # (1) refinement DIRECTION: the high-density region has smaller cells
    assert area[near].mean() < area[far].mean()
    # (2) genuine VARIABLE resolution: cells span a real size range (not uniform).
    # Density-weighted Lloyd converges linearly, so the achieved contrast grows
    # with lloyd_iterations; for strong/precise refinement use a JIGSAW-built mesh
    # via load_mpas_mesh.  At 60 iters the smallest cell is well under the largest.
    assert area.max() / area.min() > 1.5


def test_density_must_be_positive_and_finite() -> None:
    with pytest.raises(ValueError, match="positive"):
        create_voronoi_mesh(2, lloyd_iterations=5,
                            density_fn=lambda lat, lon: -1.0)


def test_density_requires_lloyd_relaxation() -> None:
    with pytest.raises(ValueError, match="density_fn requires"):
        create_voronoi_mesh(2, lloyd_iterations=0,
                            density_fn=lambda lat, lon: 1.0)


def test_variable_resolution_via_create_grid_factory() -> None:
    """The uniform create_grid factory forwards density_fn -> MPAS variable-res."""
    from legoesm.grids.factory import create_grid

    mesh = create_grid("mpas", 3, lloyd_iterations=20,
                       density_fn=lambda lat, lon: 1.0 + 5.0 * np.exp(-(lat ** 2) / 0.2))
    # cells cluster near the equator (lat=0) -> smaller there than near the poles
    lat_c = np.asarray(mesh.latCell)
    area = np.asarray(mesh.areaCell)
    eq = np.abs(lat_c) < 0.3
    polar = np.abs(lat_c) > 1.0
    assert eq.sum() > 3 and polar.sum() > 3
    assert area[eq].mean() < area[polar].mean()
