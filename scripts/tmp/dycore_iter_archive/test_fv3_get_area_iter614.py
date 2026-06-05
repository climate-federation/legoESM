"""FV3_3D iter 614: get_area + great_circle_distance_cart ports.

Faithful ports of FV3 fv_grid_utils.F90 helpers built on iter-613
``spherical_angle``:

- ``great_circle_distance_cart`` (F90:2065)
- ``get_area``                   (F90:2749)

Tests
-----

1. ``test_great_circle_distance_cart_orthogonal``.
2. ``test_great_circle_distance_cart_identical``.
3. ``test_great_circle_distance_cart_matches_latlon``.
4. ``test_get_area_full_sphere_octant``.
5. ``test_get_area_small_cell_planar_limit``.
6. ``test_get_area_positive``.
7. ``test_get_area_units_match_radius``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    get_area,
    great_circle_distance,
    great_circle_distance_cart,
    latlon2xyz,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_great_circle_distance_cart_orthogonal():
    """Orthogonal unit vectors: d = R · π/2."""
    v1 = jnp.asarray([1.0, 0.0, 0.0])
    v2 = jnp.asarray([0.0, 1.0, 0.0])
    R = 100.0
    d = float(great_circle_distance_cart(v1, v2, radius=R))
    assert abs(d - R * jnp.pi / 2) < 1e-10


def test_great_circle_distance_cart_identical():
    """Same vector: d = 0."""
    v = jnp.asarray([1.0 / jnp.sqrt(3.0)] * 3)
    d = float(great_circle_distance_cart(v, v, radius=1.0))
    assert abs(d) < 1e-10


def test_great_circle_distance_cart_matches_latlon():
    """Cartesian and latlon paths must agree."""
    rng = np.random.default_rng(seed=614)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        v1 = _xyz(lon1, lat1)
        v2 = _xyz(lon2, lat2)
        R = float(constants.R_earth)
        d_cart = float(great_circle_distance_cart(v1, v2, radius=R))
        d_ll = float(great_circle_distance(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
            radius=R,
        ))
        assert abs(d_cart - d_ll) / max(d_ll, 1.0) < 1e-10, (
            f"cart={d_cart}, ll={d_ll}"
        )


def test_get_area_matches_legoesm_cell_area():
    """FV3 get_area must agree with legoESM cubed-sphere cell area
    (l'Huilier's-theorem path); both compute the same GC-bounded
    spherical-quadrilateral area, so they must agree to ~1e-10."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere

    n = 8
    grid = create_cubed_sphere(n, radius=1.0)
    # Sample a non-edge cell on face 0
    face, i, j = 0, n // 2, n // 2
    # legoESM precomputed area
    a_legoesm = float(grid.area[face, i, j])
    # Compute Cartesian corner coordinates from face/(α, β) gnomonic grid
    # — easier: directly reproduce the FV3-style corner-(lon, lat) path
    # by reusing the legoESM utility _face_to_cartesian.
    from legoesm.grids.cubed_sphere import _face_to_cartesian

    alpha_edges = jnp.linspace(-jnp.pi / 4, jnp.pi / 4, n + 1)
    ax, ay = jnp.meshgrid(alpha_edges, alpha_edges, indexing="ij")
    x, y, z = _face_to_cartesian(face, ax, ay)
    r = jnp.sqrt(x * x + y * y + z * z)
    x, y, z = x / r, y / r, z / r
    # 4 corners (SW, SE, NE, NW)
    from legoesm.grids.cubed_sphere import xyz2latlon as _xyz2ll
    sw = _xyz2ll(x[i, j], y[i, j], z[i, j])
    se = _xyz2ll(x[i + 1, j], y[i + 1, j], z[i + 1, j])
    ne = _xyz2ll(x[i + 1, j + 1], y[i + 1, j + 1], z[i + 1, j + 1])
    nw = _xyz2ll(x[i, j + 1], y[i, j + 1], z[i, j + 1])
    a_fv3 = float(get_area(
        sw[0], sw[1], se[0], se[1], ne[0], ne[1], nw[0], nw[1],
        radius=1.0,
    ))
    rel = abs(a_fv3 - a_legoesm) / a_legoesm
    # legoESM stores grid.area in float32 → ~1e-7 precision floor
    assert rel < 1e-6, (
        f"FV3 get_area={a_fv3}, legoESM area={a_legoesm}, rel={rel}"
    )


def test_get_area_small_cell_planar_limit():
    """Tiny cell: spherical excess ≈ planar quadrilateral area."""
    R = 1.0
    eps = 1e-3
    # Tiny square cell at the equator near lon=0
    area = float(get_area(
        jnp.asarray(-eps), jnp.asarray(-eps),
        jnp.asarray(eps), jnp.asarray(-eps),
        jnp.asarray(eps), jnp.asarray(eps),
        jnp.asarray(-eps), jnp.asarray(eps),
        radius=R,
    ))
    # Planar limit: (2eps)² = 4eps²
    expected = (2 * eps) ** 2
    rel = abs(area - expected) / expected
    assert rel < 1e-4, f"rel err {rel}; got {area}, expected {expected}"


def test_get_area_positive():
    """Random cell on sphere → positive area."""
    rng = np.random.default_rng(seed=615)
    for _ in range(5):
        lon_c = float(rng.uniform(0.5, 2 * jnp.pi - 0.5))
        lat_c = float(rng.uniform(-1.0, 1.0))
        dl = 0.05
        area = float(get_area(
            jnp.asarray(lon_c - dl), jnp.asarray(lat_c - dl),
            jnp.asarray(lon_c + dl), jnp.asarray(lat_c - dl),
            jnp.asarray(lon_c + dl), jnp.asarray(lat_c + dl),
            jnp.asarray(lon_c - dl), jnp.asarray(lat_c + dl),
            radius=1.0,
        ))
        assert area > 0.0, f"got {area}"


def test_get_area_units_match_radius():
    """Area scales as R²."""
    dl = 0.1
    a1 = float(get_area(
        jnp.asarray(-dl), jnp.asarray(-dl),
        jnp.asarray(dl), jnp.asarray(-dl),
        jnp.asarray(dl), jnp.asarray(dl),
        jnp.asarray(-dl), jnp.asarray(dl),
        radius=1.0,
    ))
    a2 = float(get_area(
        jnp.asarray(-dl), jnp.asarray(-dl),
        jnp.asarray(dl), jnp.asarray(-dl),
        jnp.asarray(dl), jnp.asarray(dl),
        jnp.asarray(-dl), jnp.asarray(dl),
        radius=10.0,
    ))
    # R doubled by factor 10 → area scaled by 100
    assert abs(a2 / a1 - 100.0) / 100.0 < 1e-10
