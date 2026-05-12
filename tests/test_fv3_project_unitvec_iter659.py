"""FV3_3D iter 659: project_sphere_v + get_unit_vector_fv3 ports.

Faithful JAX ports:
- ``project_sphere_v``    (fv_grid_utils.F90:3345)
- ``get_unit_vector_fv3`` (test_cases.F90:8366)

Tests
-----

1. ``test_project_orthogonal_to_e``.
2. ``test_project_radial_zero``.
3. ``test_get_unit_vector_unit_norm``.
4. ``test_get_unit_vector_tangent``.
5. ``test_get_unit_vector_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    get_unit_vector_fv3,
    inner_prod,
    latlon2xyz,
    project_sphere_v,
)


def _xyz(lon, lat):
    x, y, z = latlon2xyz(jnp.asarray(lon), jnp.asarray(lat))
    return jnp.stack([x, y, z], axis=-1)


def test_project_orthogonal_to_e():
    """Projected vector must be orthogonal to e."""
    rng = np.random.default_rng(seed=659)
    for _ in range(5):
        lon = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat = float(rng.uniform(-1.0, 1.0))
        e = _xyz(lon, lat)
        f = jnp.asarray(rng.normal(size=3))
        f_proj = project_sphere_v(f, e)
        dot = float(inner_prod(f_proj, e))
        assert abs(dot) < 1e-13


def test_project_radial_zero():
    """If f is parallel to e, projection is zero."""
    e = _xyz(0.5, 0.3)
    f = 2.5 * e            # radial
    f_proj = project_sphere_v(f, e)
    assert jnp.allclose(f_proj, 0.0, atol=1e-12)


def test_get_unit_vector_unit_norm():
    """Returned tangent vector must be unit-length."""
    rng = np.random.default_rng(seed=660)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        lon3 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat3 = float(rng.uniform(-1.0, 1.0))
        uvec = get_unit_vector_fv3(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
            jnp.asarray(lon3), jnp.asarray(lat3),
        )
        assert abs(float(jnp.linalg.norm(uvec)) - 1.0) < 1e-12


def test_get_unit_vector_tangent():
    """Returned vector is in tangent plane at p2 (⊥ to p2 position)."""
    rng = np.random.default_rng(seed=661)
    for _ in range(5):
        lon1 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat1 = float(rng.uniform(-1.0, 1.0))
        lon2 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat2 = float(rng.uniform(-1.0, 1.0))
        lon3 = float(rng.uniform(0.2, 2 * jnp.pi - 0.2))
        lat3 = float(rng.uniform(-1.0, 1.0))
        p2 = _xyz(lon2, lat2)
        uvec = get_unit_vector_fv3(
            jnp.asarray(lon1), jnp.asarray(lat1),
            jnp.asarray(lon2), jnp.asarray(lat2),
            jnp.asarray(lon3), jnp.asarray(lat3),
        )
        dot = float(inner_prod(uvec, p2))
        assert abs(dot) < 1e-12, f"not tangent: dot = {dot}"


def test_get_unit_vector_finite():
    """No NaN/Inf on random inputs."""
    rng = np.random.default_rng(seed=662)
    lon1 = float(rng.uniform(0.0, 2 * jnp.pi))
    lat1 = float(rng.uniform(-1.0, 1.0))
    lon2 = float(rng.uniform(0.0, 2 * jnp.pi))
    lat2 = float(rng.uniform(-1.0, 1.0))
    lon3 = float(rng.uniform(0.0, 2 * jnp.pi))
    lat3 = float(rng.uniform(-1.0, 1.0))
    uvec = get_unit_vector_fv3(
        jnp.asarray(lon1), jnp.asarray(lat1),
        jnp.asarray(lon2), jnp.asarray(lat2),
        jnp.asarray(lon3), jnp.asarray(lat3),
    )
    assert jnp.all(jnp.isfinite(uvec))
