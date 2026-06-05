"""FV3_3D iter 626: init_cubed_to_latlon port.

Faithful port of FV3 ``init_cubed_to_latlon`` (fv_grid_utils.F90:
2321-2384, grid_type<4 branch).  Computes D-grid → latlon wind
rotation matrices a11/a12/a21/a22.

Tests
-----

1. ``test_init_c2l_shapes``.
2. ``test_init_c2l_finite``.
3. ``test_init_c2l_z_inner_products``.
4. ``test_init_c2l_a_formula``.
5. ``test_init_c2l_identity_when_ec_is_vlon_vlat``.
6. ``test_init_c2l_zero_sin_sg5_safe``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    init_cubed_to_latlon,
    inner_prod,
    unit_vect_latlon,
)


def test_init_c2l_shapes():
    """Output arrays match input cell-grid shape."""
    n = 6
    rng = np.random.default_rng(seed=626)
    agrid_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    ec1 = jnp.asarray(rng.normal(size=(n, n, 3)))
    ec2 = jnp.asarray(rng.normal(size=(n, n, 3)))
    sin_sg5 = jnp.full((n, n), 0.9)
    out = init_cubed_to_latlon(agrid_lon, agrid_lat, ec1, ec2, sin_sg5)
    assert len(out) == 10
    for arr in out[:8]:
        assert arr.shape == (n, n)
    # vlon, vlat have shape (n, n, 3)
    assert out[8].shape == (n, n, 3)
    assert out[9].shape == (n, n, 3)


def test_init_c2l_finite():
    """No NaN/Inf in any output."""
    n = 4
    rng = np.random.default_rng(seed=627)
    agrid_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    ec1 = jnp.asarray(rng.normal(size=(n, n, 3)))
    ec2 = jnp.asarray(rng.normal(size=(n, n, 3)))
    sin_sg5 = jnp.full((n, n), 0.85)
    out = init_cubed_to_latlon(agrid_lon, agrid_lat, ec1, ec2, sin_sg5)
    for arr in out:
        assert jnp.all(jnp.isfinite(arr))


def test_init_c2l_z_inner_products():
    """z11 = ec1·vlon, z12 = ec1·vlat, z21 = ec2·vlon, z22 = ec2·vlat."""
    n = 4
    rng = np.random.default_rng(seed=628)
    agrid_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    ec1 = jnp.asarray(rng.normal(size=(n, n, 3)))
    ec2 = jnp.asarray(rng.normal(size=(n, n, 3)))
    sin_sg5 = jnp.full((n, n), 0.9)
    (a11, a12, a21, a22, z11, z12, z21, z22, vlon, vlat) = init_cubed_to_latlon(
        agrid_lon, agrid_lat, ec1, ec2, sin_sg5,
    )
    assert jnp.allclose(z11, inner_prod(ec1, vlon), atol=1e-14)
    assert jnp.allclose(z12, inner_prod(ec1, vlat), atol=1e-14)
    assert jnp.allclose(z21, inner_prod(ec2, vlon), atol=1e-14)
    assert jnp.allclose(z22, inner_prod(ec2, vlat), atol=1e-14)


def test_init_c2l_a_formula():
    """a11 = 0.5·z22/sin_sg5; a12 = -0.5·z12/sin_sg5; etc."""
    n = 4
    rng = np.random.default_rng(seed=629)
    agrid_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    ec1 = jnp.asarray(rng.normal(size=(n, n, 3)))
    ec2 = jnp.asarray(rng.normal(size=(n, n, 3)))
    sin_sg5 = jnp.full((n, n), 0.87)
    (a11, a12, a21, a22, z11, z12, z21, z22, _, _) = init_cubed_to_latlon(
        agrid_lon, agrid_lat, ec1, ec2, sin_sg5,
    )
    assert jnp.allclose(a11, 0.5 * z22 / sin_sg5, atol=1e-14)
    assert jnp.allclose(a12, -0.5 * z12 / sin_sg5, atol=1e-14)
    assert jnp.allclose(a21, -0.5 * z21 / sin_sg5, atol=1e-14)
    assert jnp.allclose(a22, 0.5 * z11 / sin_sg5, atol=1e-14)


def test_init_c2l_identity_when_ec_is_vlon_vlat():
    """If ec1 = vlon and ec2 = vlat at every cell, then z is the
    identity inner-product (z11 = z22 = 1; z12 = z21 = 0)."""
    n = 4
    rng = np.random.default_rng(seed=630)
    agrid_lon = jnp.asarray(rng.uniform(0.5, 2 * jnp.pi - 0.5, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    vlon_ref, vlat_ref = unit_vect_latlon(agrid_lon, agrid_lat)
    sin_sg5 = jnp.full((n, n), 0.9)
    (_, _, _, _, z11, z12, z21, z22, _, _) = init_cubed_to_latlon(
        agrid_lon, agrid_lat, vlon_ref, vlat_ref, sin_sg5,
    )
    # vlon·vlon = 1, vlon·vlat = 0
    assert jnp.allclose(z11, 1.0, atol=1e-14)
    assert jnp.allclose(z22, 1.0, atol=1e-14)
    assert jnp.allclose(z12, 0.0, atol=1e-14)
    assert jnp.allclose(z21, 0.0, atol=1e-14)


def test_init_c2l_zero_sin_sg5_safe():
    """sin_sg5 = 0 → no NaN/Inf (safe-divide branch)."""
    n = 4
    rng = np.random.default_rng(seed=631)
    agrid_lon = jnp.asarray(rng.uniform(0, 2 * jnp.pi, size=(n, n)))
    agrid_lat = jnp.asarray(rng.uniform(-1.0, 1.0, size=(n, n)))
    ec1 = jnp.asarray(rng.normal(size=(n, n, 3)))
    ec2 = jnp.asarray(rng.normal(size=(n, n, 3)))
    sin_sg5 = jnp.zeros((n, n))
    out = init_cubed_to_latlon(agrid_lon, agrid_lat, ec1, ec2, sin_sg5)
    for arr in out:
        assert jnp.all(jnp.isfinite(arr))
