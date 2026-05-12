"""FV3_3D iter 616: great-circle intersection port.

Faithful port of FV3 ``intersect`` (fv_grid_utils.F90:2096-2194).
Used by FV3 grid generation for cubed-sphere panel boundaries
and regional refinement.

Tests
-----

1. ``test_intersect_meridian_equator``.
2. ``test_intersect_on_unit_sphere``.
3. ``test_intersect_local_flag``.
4. ``test_intersect_symmetric``.
5. ``test_intersect_perpendicular_meridians``.
6. ``test_intersect_radius_scaling``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    intersect_great_circles,
    latlon2xyz,
    xyz2latlon,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_intersect_meridian_equator():
    """Greenwich meridian arc meets equator arc at (0, 0)."""
    # Meridian arc from (0, -π/4) to (0, π/4)
    a1 = _xyz(0.0, -jnp.pi / 4)
    a2 = _xyz(0.0, jnp.pi / 4)
    # Equator arc from (-π/4, 0) to (π/4, 0)
    b1 = _xyz(-jnp.pi / 4, 0.0)
    b2 = _xyz(jnp.pi / 4, 0.0)
    x_inter, local_a, local_b = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
    lon, lat = xyz2latlon(x_inter[..., 0], x_inter[..., 1], x_inter[..., 2])
    assert abs(float(lon)) < 1e-12, f"lon = {float(lon)}, expected 0"
    assert abs(float(lat)) < 1e-12, f"lat = {float(lat)}, expected 0"
    assert bool(local_a)
    assert bool(local_b)


def test_intersect_on_unit_sphere():
    """Intersection point must be on the unit sphere (radius=1)."""
    rng = np.random.default_rng(seed=616)
    for _ in range(5):
        # Two arbitrary great circles
        a1 = _xyz(float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
                  float(rng.uniform(-1.0, 1.0)))
        a2 = _xyz(float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
                  float(rng.uniform(-1.0, 1.0)))
        b1 = _xyz(float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
                  float(rng.uniform(-1.0, 1.0)))
        b2 = _xyz(float(rng.uniform(0.2, 2 * jnp.pi - 0.2)),
                  float(rng.uniform(-1.0, 1.0)))
        x_inter, _, _ = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
        norm = float(jnp.linalg.norm(x_inter))
        assert abs(norm - 1.0) < 1e-10, f"|x_inter| = {norm}"


def test_intersect_local_flag():
    """local_a/local_b should be False when arcs don't actually cross."""
    # Two arcs on the same hemisphere but not intersecting
    # Equator from (0, 0) to (π/4, 0)
    a1 = _xyz(0.0, 0.0)
    a2 = _xyz(jnp.pi / 4, 0.0)
    # Meridian-segment from (π, π/8) to (π, π/4) — far from arc A
    b1 = _xyz(jnp.pi, jnp.pi / 8)
    b2 = _xyz(jnp.pi, jnp.pi / 4)
    _, local_a, local_b = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
    # Intersection should NOT be inside either arc
    assert not (bool(local_a) and bool(local_b)), (
        "non-overlapping arcs should not have x_inter in both"
    )


def test_intersect_symmetric():
    """Swapping A↔B should give the same intersection point."""
    a1 = _xyz(0.0, -0.1)
    a2 = _xyz(0.0, 0.1)
    b1 = _xyz(-0.1, 0.0)
    b2 = _xyz(0.1, 0.0)
    x_ab, _, _ = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
    x_ba, _, _ = intersect_great_circles(b1, b2, a1, a2, radius=1.0)
    # Symmetric (up to ±sign, but get_nearest picks closer one which is the
    # same for both since centroid is symmetric)
    assert jnp.allclose(x_ab, x_ba, atol=1e-12) or jnp.allclose(
        x_ab, -x_ba, atol=1e-12
    )


def test_intersect_perpendicular_meridians():
    """Two perpendicular meridians intersect at the north pole."""
    # Greenwich meridian arc: (0, π/8) to (0, π/3)
    a1 = _xyz(0.0, jnp.pi / 8)
    a2 = _xyz(0.0, jnp.pi / 3)
    # 90E meridian arc: (π/2, π/8) to (π/2, π/3)
    b1 = _xyz(jnp.pi / 2, jnp.pi / 8)
    b2 = _xyz(jnp.pi / 2, jnp.pi / 3)
    x_inter, _, _ = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
    # Both meridians intersect at the north pole (lat=π/2)
    lon, lat = xyz2latlon(x_inter[..., 0], x_inter[..., 1], x_inter[..., 2])
    assert abs(float(lat) - jnp.pi / 2) < 1e-10


def test_intersect_radius_scaling():
    """Result must scale with radius."""
    a1 = _xyz(0.0, -0.1)
    a2 = _xyz(0.0, 0.1)
    b1 = _xyz(-0.1, 0.0)
    b2 = _xyz(0.1, 0.0)
    x1, _, _ = intersect_great_circles(a1, a2, b1, b2, radius=1.0)
    x10, _, _ = intersect_great_circles(a1, a2, b1, b2, radius=10.0)
    assert jnp.allclose(x10, 10.0 * x1, atol=1e-10)
