"""FV3_3D iter 611: Cartesian grid primitives port.

Faithful ports of FV3 ``fv_grid_utils.F90`` helpers used as
building blocks of grid generation, halo corner code, and many
spherical geometry routines:

- ``latlon2xyz``      (F90:1639)
- ``xyz2latlon`` / ``cart_to_latlon`` (F90:1739)
- ``inner_prod``      (F90:984)
- ``vect_cross``      (F90:1781)
- ``normalize_vect``  (F90:1880)
- ``mid_pt3_cart``    (F90:1996)
- ``mid_pt_cart``     (F90:2026)
- ``get_unit_vect2``  (F90:1848)

Tests
-----

1. ``test_latlon2xyz_unit_norm``: output on unit sphere.
2. ``test_latlon2xyz_xyz2latlon_roundtrip``.
3. ``test_xyz2latlon_pole_safe``: |x|+|y|<eps → lon=0.
4. ``test_inner_prod_orthogonal``.
5. ``test_inner_prod_parallel_unit_vectors``.
6. ``test_vect_cross_orthogonal``.
7. ``test_vect_cross_anticommutes``.
8. ``test_normalize_vect_unit_norm``.
9. ``test_normalize_vect_zero_safe``.
10. ``test_mid_pt3_cart_equidistant``.
11. ``test_mid_pt_cart_matches_mid_pt_sphere``.
12. ``test_get_unit_vect2_unit_norm``.
13. ``test_get_unit_vect2_tangent_to_pc``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_unit_vect2,
    inner_prod,
    latlon2xyz,
    mid_pt3_cart,
    mid_pt_cart,
    mid_pt_sphere,
    normalize_vect,
    vect_cross,
    xyz2latlon,
)


def test_latlon2xyz_unit_norm():
    """latlon2xyz output must be on the unit sphere."""
    rng = np.random.default_rng(seed=611)
    lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=10))
    lat = jnp.asarray(rng.uniform(-jnp.pi / 2, jnp.pi / 2, size=10))
    x, y, z = latlon2xyz(lon, lat)
    norm = jnp.sqrt(x * x + y * y + z * z)
    assert jnp.allclose(norm, 1.0, atol=1e-12)


def test_latlon2xyz_xyz2latlon_roundtrip():
    """Roundtrip latlon → xyz → latlon must recover input."""
    rng = np.random.default_rng(seed=612)
    for _ in range(10):
        lon_in = float(rng.uniform(0.05, 2 * jnp.pi - 0.05))
        lat_in = float(rng.uniform(-jnp.pi / 2 + 0.05, jnp.pi / 2 - 0.05))
        x, y, z = latlon2xyz(jnp.asarray(lon_in), jnp.asarray(lat_in))
        lon_out, lat_out = xyz2latlon(x, y, z)
        assert abs(float(lon_out) - lon_in) < 1e-12
        assert abs(float(lat_out) - lat_in) < 1e-12


def test_xyz2latlon_pole_safe():
    """At z=1 (north pole), |x|+|y|<eps should set lon=0."""
    lon, lat = xyz2latlon(jnp.asarray(0.0), jnp.asarray(0.0), jnp.asarray(1.0))
    assert abs(float(lon) - 0.0) < 1e-12
    assert abs(float(lat) - jnp.pi / 2) < 1e-12


def test_inner_prod_orthogonal():
    """Orthogonal unit vectors → inner_prod = 0."""
    v1 = jnp.asarray([1.0, 0.0, 0.0])
    v2 = jnp.asarray([0.0, 1.0, 0.0])
    assert abs(float(inner_prod(v1, v2))) < 1e-14


def test_inner_prod_parallel_unit_vectors():
    """Same unit vector → inner_prod = 1."""
    v = jnp.asarray([1.0 / jnp.sqrt(3.0)] * 3)
    assert abs(float(inner_prod(v, v)) - 1.0) < 1e-14


def test_vect_cross_orthogonal():
    """e × e = 0; (1,0,0) × (0,1,0) = (0,0,1)."""
    v = jnp.asarray([1.0, 2.0, 3.0])
    assert jnp.allclose(vect_cross(v, v), 0.0, atol=1e-14)
    e1 = jnp.asarray([1.0, 0.0, 0.0])
    e2 = jnp.asarray([0.0, 1.0, 0.0])
    e3 = vect_cross(e1, e2)
    assert jnp.allclose(e3, jnp.asarray([0.0, 0.0, 1.0]), atol=1e-14)


def test_vect_cross_anticommutes():
    """p1 × p2 = -(p2 × p1)."""
    rng = np.random.default_rng(seed=613)
    p1 = jnp.asarray(rng.normal(size=3))
    p2 = jnp.asarray(rng.normal(size=3))
    assert jnp.allclose(vect_cross(p1, p2), -vect_cross(p2, p1), atol=1e-14)


def test_normalize_vect_unit_norm():
    """Normalized vector has unit norm."""
    rng = np.random.default_rng(seed=614)
    v = jnp.asarray(rng.normal(size=3))
    u = normalize_vect(v)
    assert abs(float(jnp.linalg.norm(u)) - 1.0) < 1e-14


def test_normalize_vect_zero_safe():
    """Zero input must not produce NaN."""
    z = jnp.zeros(3)
    u = normalize_vect(z)
    assert jnp.all(jnp.isfinite(u))


def test_mid_pt3_cart_equidistant():
    """mid_pt3_cart of two unit vectors is on unit sphere and equidistant."""
    rng = np.random.default_rng(seed=615)
    lon1, lat1 = float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1))
    lon2, lat2 = float(rng.uniform(0, 2 * jnp.pi)), float(rng.uniform(-1, 1))
    x1, y1, z1 = latlon2xyz(jnp.asarray(lon1), jnp.asarray(lat1))
    x2, y2, z2 = latlon2xyz(jnp.asarray(lon2), jnp.asarray(lat2))
    p1 = jnp.stack([x1, y1, z1], axis=-1)
    p2 = jnp.stack([x2, y2, z2], axis=-1)
    pc = mid_pt3_cart(p1, p2)
    # On unit sphere
    assert abs(float(jnp.linalg.norm(pc)) - 1.0) < 1e-12
    # Equidistant: |pc - p1| = |pc - p2|
    d1 = float(jnp.linalg.norm(pc - p1))
    d2 = float(jnp.linalg.norm(pc - p2))
    assert abs(d1 - d2) < 1e-12


def test_mid_pt_cart_matches_mid_pt_sphere():
    """mid_pt_cart → xyz2latlon should match mid_pt_sphere (lon, lat)."""
    rng = np.random.default_rng(seed=616)
    for _ in range(5):
        lon1 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat1 = float(rng.uniform(-jnp.pi / 2 + 0.2, jnp.pi / 2 - 0.2))
        lon2 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat2 = float(rng.uniform(-jnp.pi / 2 + 0.2, jnp.pi / 2 - 0.2))
        pc = mid_pt_cart(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        lon_cart, lat_cart = xyz2latlon(pc[..., 0], pc[..., 1], pc[..., 2])
        lon_sph, lat_sph = mid_pt_sphere(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        assert abs(float(lon_cart) - float(lon_sph)) < 1e-12, (
            f"lon mismatch: cart={float(lon_cart)} sph={float(lon_sph)}"
        )
        assert abs(float(lat_cart) - float(lat_sph)) < 1e-12, (
            f"lat mismatch: cart={float(lat_cart)} sph={float(lat_sph)}"
        )


def test_get_unit_vect2_unit_norm():
    """Returned tangent vector must be unit-length."""
    rng = np.random.default_rng(seed=617)
    for _ in range(5):
        lon1 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat2 = float(rng.uniform(-1.0, 1.0))
        uc = get_unit_vect2(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        assert abs(float(jnp.linalg.norm(uc)) - 1.0) < 1e-10


def test_get_unit_vect2_tangent_to_pc():
    """Tangent at pc must be ⊥ pc (tangent to sphere at midpoint)."""
    rng = np.random.default_rng(seed=618)
    for _ in range(5):
        lon1 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.1, 2 * jnp.pi - 0.1))
        lat2 = float(rng.uniform(-1.0, 1.0))
        pc = mid_pt_cart(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        uc = get_unit_vect2(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        # Tangent ⊥ pc → dot product near zero
        dot = float(inner_prod(uc, pc))
        assert abs(dot) < 1e-10, f"uc·pc = {dot}, expected ~0"
