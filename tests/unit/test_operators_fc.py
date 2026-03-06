"""Tests for FC-Gram A-grid operators on the cubed-sphere."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.operators_fc import (
    build_fc_config,
    fc_gradient_x,
    fc_gradient_y,
    fc_divergence,
    fc_curl_z,
    fc_laplacian,
    fc_flux_divergence,
    fc_scalar_advection,
    fc_divergence_damping,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


@pytest.fixture
def grid_and_config():
    grid = create_cubed_sphere(12)
    fc_config = build_fc_config(d=2, C=4, degree=5)
    return grid, fc_config


def test_gradient_constant_is_zero(grid_and_config):
    """Gradient of a constant field should be zero."""
    grid, fc_config = grid_and_config
    n = grid.n
    q = jnp.ones((6, n, n), dtype=jnp.float32) * 42.0

    gx = fc_gradient_x(q, grid, fc_config)
    gy = fc_gradient_y(q, grid, fc_config)

    assert float(jnp.max(jnp.abs(gx))) < 0.1
    assert float(jnp.max(jnp.abs(gy))) < 0.1


def test_divergence_outputs_correct_shape(grid_and_config):
    """Divergence produces correct shape."""
    grid, fc_config = grid_and_config
    n = grid.n
    u = jnp.ones((6, n, n), dtype=jnp.float32)
    v = jnp.zeros((6, n, n), dtype=jnp.float32)

    div = fc_divergence(u, v, grid, fc_config)
    assert div.shape == (6, n, n)
    assert jnp.all(jnp.isfinite(div))


def test_curl_outputs_correct_shape(grid_and_config):
    """Curl produces correct shape."""
    grid, fc_config = grid_and_config
    n = grid.n
    u = jnp.ones((6, n, n), dtype=jnp.float32)
    v = jnp.zeros((6, n, n), dtype=jnp.float32)

    vort = fc_curl_z(u, v, grid, fc_config)
    assert vort.shape == (6, n, n)
    assert jnp.all(jnp.isfinite(vort))


def test_flux_divergence_conserves(grid_and_config):
    """FC flux divergence: global sum should be approximately zero."""
    grid, fc_config = grid_and_config
    n = grid.n
    q = jnp.ones((6, n, n), dtype=jnp.float32) * 1000.0
    u = jnp.ones((6, n, n), dtype=jnp.float32) * 10.0
    v = jnp.zeros((6, n, n), dtype=jnp.float32)

    fd = fc_flux_divergence(q, u, v, grid, fc_config)
    global_sum = float(jnp.sum(fd * grid.area))
    # Not exactly zero (FC is not telescoping), but should be small
    assert abs(global_sum) < 1e6, f"Global sum = {global_sum}"


def test_scalar_advection_shape(grid_and_config):
    """Scalar advection produces correct shape."""
    grid, fc_config = grid_and_config
    n = grid.n
    q = jnp.ones((6, n, n), dtype=jnp.float32)
    u = jnp.ones((6, n, n), dtype=jnp.float32)
    v = jnp.zeros((6, n, n), dtype=jnp.float32)

    adv = fc_scalar_advection(q, u, v, grid, fc_config)
    assert adv.shape == (6, n, n)


def test_divergence_damping_uniform(grid_and_config):
    """Divergence damping produces finite results."""
    grid, fc_config_base = grid_and_config
    fc_config = build_fc_config(d=2, C=4, degree=5, div_damp_2=1e5)
    n = grid.n

    u = jnp.ones((6, n, n), dtype=jnp.float32) * 10.0
    v = jnp.ones((6, n, n), dtype=jnp.float32) * 5.0

    du_damp, dv_damp = fc_divergence_damping(u, v, grid, fc_config)
    assert du_damp.shape == (6, n, n)
    assert dv_damp.shape == (6, n, n)
    assert jnp.all(jnp.isfinite(du_damp))
    assert jnp.all(jnp.isfinite(dv_damp))


def test_divergence_damping_zero_when_disabled(grid_and_config):
    """Divergence damping returns zeros when coefficients are zero."""
    grid, fc_config = grid_and_config
    n = grid.n
    u = jnp.ones((6, n, n), dtype=jnp.float32) * 10.0
    v = jnp.ones((6, n, n), dtype=jnp.float32) * 5.0

    du_damp, dv_damp = fc_divergence_damping(u, v, grid, fc_config)
    assert float(jnp.max(jnp.abs(du_damp))) == 0.0
    assert float(jnp.max(jnp.abs(dv_damp))) == 0.0


def test_laplacian_shape(grid_and_config):
    """Laplacian produces correct shape."""
    grid, fc_config = grid_and_config
    n = grid.n
    q = jnp.ones((6, n, n), dtype=jnp.float32)

    lap = fc_laplacian(q, grid, fc_config)
    assert lap.shape == (6, n, n)
    assert jnp.all(jnp.isfinite(lap))


def test_operators_differentiable(grid_and_config):
    """FC operators are differentiable with jax.grad."""
    grid, fc_config = grid_and_config
    n = grid.n

    def loss(q):
        gx = fc_gradient_x(q, grid, fc_config)
        return jnp.sum(gx**2)

    q = jnp.ones((6, n, n), dtype=jnp.float32)
    grad_q = jax.grad(loss)(q)
    assert grad_q.shape == (6, n, n)
    assert jnp.all(jnp.isfinite(grad_q))
