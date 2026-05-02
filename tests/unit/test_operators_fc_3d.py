"""Smoke tests for the 3D FC operator wrappers (core/operators_fc_3d.py).

The 3D wrappers are thin pass-throughs to the 2D operators in
``operators_fc.py`` (which are ndim-aware).  These tests just verify
that the 3D entry points produce the expected shape and values that
match the 2D version applied per-level.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from legoesm.core.operators_fc import build_fc_config, fc_gradient_x, fc_divergence
from legoesm.core.operators_fc_3d import (
    fc_gradient_x_3d,
    fc_gradient_y_3d,
    fc_divergence_3d,
    fc_curl_z_3d,
    fc_laplacian_3d,
    fc_hyperdiffusion_3d,
    fc_flux_divergence_3d,
    fc_scalar_advection_3d,
    fc_divergence_damping_3d,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


@pytest.fixture
def grid_and_config():
    grid = create_cubed_sphere(12)
    fc_config = build_fc_config(d=2, C=4, degree=5)
    return grid, fc_config


def test_gradient_x_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 4
    field = jnp.ones((6, n, n, nlev), dtype=jnp.float32) * 42.0
    gx = fc_gradient_x_3d(field, grid, fc_config)
    assert gx.shape == (6, n, n, nlev)
    # Gradient of a constant ≈ 0
    assert float(jnp.max(jnp.abs(gx))) < 0.1


def test_gradient_y_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 4
    field = jnp.ones((6, n, n, nlev), dtype=jnp.float32) * 7.0
    gy = fc_gradient_y_3d(field, grid, fc_config)
    assert gy.shape == (6, n, n, nlev)
    assert float(jnp.max(jnp.abs(gy))) < 0.1


def test_divergence_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    u = jnp.ones((6, n, n, nlev), dtype=jnp.float32)
    v = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    div = fc_divergence_3d(u, v, grid, fc_config)
    assert div.shape == (6, n, n, nlev)
    assert jnp.all(jnp.isfinite(div))


def test_curl_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    u = jnp.ones((6, n, n, nlev), dtype=jnp.float32)
    v = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    curl = fc_curl_z_3d(u, v, grid, fc_config)
    assert curl.shape == (6, n, n, nlev)
    assert jnp.all(jnp.isfinite(curl))


def test_laplacian_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    field = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    lap = fc_laplacian_3d(field, grid, fc_config)
    assert lap.shape == (6, n, n, nlev)
    assert jnp.allclose(lap, 0.0)


def test_hyperdiffusion_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    field = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    hd = fc_hyperdiffusion_3d(field, grid, fc_config, coeff=1e15)
    assert hd.shape == (6, n, n, nlev)


def test_flux_divergence_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    q = jnp.ones((6, n, n, nlev), dtype=jnp.float32)
    u = jnp.ones((6, n, n, nlev), dtype=jnp.float32)
    v = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    fd = fc_flux_divergence_3d(q, u, v, grid, fc_config)
    assert fd.shape == (6, n, n, nlev)
    assert jnp.all(jnp.isfinite(fd))


def test_scalar_advection_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    q = jnp.ones((6, n, n, nlev), dtype=jnp.float32)
    u = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    v = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    adv = fc_scalar_advection_3d(q, u, v, grid, fc_config)
    assert adv.shape == (6, n, n, nlev)
    # Zero advecting velocity → zero advection
    assert float(jnp.max(jnp.abs(adv))) < 1e-5


def test_divergence_damping_3d_shape(grid_and_config):
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    u = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    v = jnp.zeros((6, n, n, nlev), dtype=jnp.float32)
    du, dv = fc_divergence_damping_3d(u, v, grid, fc_config)
    assert du.shape == (6, n, n, nlev)
    assert dv.shape == (6, n, n, nlev)


def test_3d_matches_2d_per_level(grid_and_config):
    """The 3D wrapper should produce the same per-level result as the 2D op."""
    grid, fc_config = grid_and_config
    n, nlev = grid.n, 3
    field_2d_per_level = [
        jnp.array([[(i + 1) * 0.1] * n] * n)[None, ...].repeat(6, axis=0).astype(jnp.float32)
        for i in range(nlev)
    ]
    field_3d = jnp.stack(field_2d_per_level, axis=-1)  # (6, n, n, nlev)

    gx_3d = fc_gradient_x_3d(field_3d, grid, fc_config)
    for k in range(nlev):
        gx_2d_k = fc_gradient_x(field_3d[..., k], grid, fc_config)
        # 3D version uses native 4D halo; should be exactly the same per level
        assert jnp.allclose(gx_3d[..., k], gx_2d_k, atol=1e-5)
