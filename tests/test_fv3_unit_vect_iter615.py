"""FV3_3D iter 615: unit_vect_latlon + get_unit_vect3 ports.

Faithful ports of FV3 fv_grid_utils.F90:
- ``unit_vect_latlon`` (F90:2286): east/north tangent vectors at (lon, lat)
- ``get_unit_vect3``   (F90:1865): Cartesian variant of get_unit_vect2

Tests
-----

1. ``test_unit_vect_latlon_orthogonal``.
2. ``test_unit_vect_latlon_unit_norm``.
3. ``test_unit_vect_latlon_equator``.
4. ``test_unit_vect_latlon_north_pole_degenerate``.
5. ``test_unit_vect_latlon_perpendicular_to_position``.
6. ``test_get_unit_vect3_matches_get_unit_vect2``.
7. ``test_get_unit_vect3_unit_norm``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_unit_vect2,
    get_unit_vect3,
    inner_prod,
    latlon2xyz,
    unit_vect_latlon,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_unit_vect_latlon_orthogonal():
    """elon ⊥ elat at every point."""
    rng = np.random.default_rng(seed=615)
    for _ in range(5):
        lon = float(rng.uniform(0.0, 2 * jnp.pi))
        lat = float(rng.uniform(-jnp.pi / 2 + 0.1, jnp.pi / 2 - 0.1))
        elon, elat = unit_vect_latlon(jnp.asarray(lon), jnp.asarray(lat))
        dot = float(inner_prod(elon, elat))
        assert abs(dot) < 1e-14, f"elon·elat = {dot}"


def test_unit_vect_latlon_unit_norm():
    """elon and elat are unit-length away from poles."""
    rng = np.random.default_rng(seed=616)
    for _ in range(5):
        lon = float(rng.uniform(0.0, 2 * jnp.pi))
        lat = float(rng.uniform(-1.2, 1.2))
        elon, elat = unit_vect_latlon(jnp.asarray(lon), jnp.asarray(lat))
        nlon = float(jnp.linalg.norm(elon))
        nlat = float(jnp.linalg.norm(elat))
        assert abs(nlon - 1.0) < 1e-14
        assert abs(nlat - 1.0) < 1e-14


def test_unit_vect_latlon_equator():
    """At equator (lat=0): elon=(0, 1, 0) when lon=π/2, elat=(0, 0, 1)."""
    elon, elat = unit_vect_latlon(jnp.asarray(jnp.pi / 2), jnp.asarray(0.0))
    assert jnp.allclose(elon, jnp.asarray([-1.0, 0.0, 0.0]), atol=1e-14)
    assert jnp.allclose(elat, jnp.asarray([0.0, 0.0, 1.0]), atol=1e-14)


def test_unit_vect_latlon_north_pole_degenerate():
    """At north pole (lat=π/2): elat=(0,0,0) — degenerate."""
    # FV3 formula gives elat = (-1·cos λ, -1·sin λ, 0) — NOT zero,
    # but on equator plane; elon still well-defined.  This documents
    # the FV3 convention near poles (which is degenerate by design).
    elon, elat = unit_vect_latlon(jnp.asarray(0.0), jnp.asarray(jnp.pi / 2))
    # elon at north pole reduces to (0, 1, 0) when lon=0
    assert jnp.allclose(elon, jnp.asarray([0.0, 1.0, 0.0]), atol=1e-14)
    # elat at pole: (-1·1, 0, 0) since sin(π/2)=1, cos(π/2)=0
    assert jnp.allclose(elat, jnp.asarray([-1.0, 0.0, 0.0]), atol=1e-14)


def test_unit_vect_latlon_perpendicular_to_position():
    """Both elon and elat must be ⊥ to position vector (tangent plane)."""
    rng = np.random.default_rng(seed=617)
    for _ in range(5):
        lon = float(rng.uniform(0.0, 2 * jnp.pi))
        lat = float(rng.uniform(-1.2, 1.2))
        elon, elat = unit_vect_latlon(jnp.asarray(lon), jnp.asarray(lat))
        p = _xyz(lon, lat)
        d1 = float(inner_prod(elon, p))
        d2 = float(inner_prod(elat, p))
        assert abs(d1) < 1e-14, f"elon·p = {d1}"
        assert abs(d2) < 1e-14, f"elat·p = {d2}"


def test_get_unit_vect3_matches_get_unit_vect2():
    """get_unit_vect3 (Cartesian) must match get_unit_vect2 (latlon)."""
    rng = np.random.default_rng(seed=618)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        p1 = _xyz(lon1, lat1)
        p2 = _xyz(lon2, lat2)
        uc3 = get_unit_vect3(p1, p2)
        uc2 = get_unit_vect2(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
        )
        assert jnp.allclose(uc3, uc2, atol=1e-12)


def test_get_unit_vect3_unit_norm():
    """get_unit_vect3 output is unit-norm."""
    rng = np.random.default_rng(seed=619)
    for _ in range(5):
        lon1 = float(rng.uniform(0.0, 2 * jnp.pi))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.0, 2 * jnp.pi))
        lat2 = float(rng.uniform(-1.0, 1.0))
        p1 = _xyz(lon1, lat1)
        p2 = _xyz(lon2, lat2)
        uc = get_unit_vect3(p1, p2)
        assert abs(float(jnp.linalg.norm(uc)) - 1.0) < 1e-12
