"""Unit tests for discrete operators."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.operators import (
    gradient_x, gradient_y, gradient,
    divergence, curl_z, laplacian,
    global_integral, global_mean,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere


class TestOperators:
    """Tests for discrete differential operators."""

    def test_gradient_constant_field(self, small_grid):
        """Gradient of a constant field should be zero."""
        f = Field(data=jnp.ones((6, 8, 8)) * 5.0, name="const",
                  dims=("face", "x", "y"), units="m")
        gx, gy = gradient(f, small_grid)
        assert jnp.allclose(gx.data, 0.0, atol=1e-10)
        assert jnp.allclose(gy.data, 0.0, atol=1e-10)

    def test_divergence_zero_field(self, small_grid):
        """Divergence of zero vector field should be zero."""
        zero = Field(data=jnp.zeros((6, 8, 8)), name="zero",
                     dims=("face", "x", "y"), units="m/s")
        div = divergence(zero, zero, small_grid)
        assert jnp.allclose(div.data, 0.0, atol=1e-10)

    def test_laplacian_constant(self, small_grid):
        """Laplacian of a constant field should be zero."""
        f = Field(data=jnp.ones((6, 8, 8)) * 3.0, name="const",
                  dims=("face", "x", "y"), units="m")
        lap = laplacian(f, small_grid)
        assert jnp.allclose(lap.data, 0.0, atol=1e-6)

    def test_global_integral_ones(self, small_grid):
        """Global integral of ones should equal total area."""
        f = Field(data=jnp.ones((6, 8, 8)), name="ones",
                  dims=("face", "x", "y"), units="1")
        integral = global_integral(f, small_grid)
        assert jnp.allclose(integral, small_grid.total_area, rtol=1e-6)

    def test_global_mean_constant(self, small_grid):
        """Global mean of a constant field should be that constant."""
        val = 42.0
        f = Field(data=jnp.ones((6, 8, 8)) * val, name="const",
                  dims=("face", "x", "y"), units="1")
        mean = global_mean(f, small_grid)
        assert jnp.allclose(mean, val, rtol=1e-6)

    def test_operators_differentiable(self, small_grid):
        """All operators should be differentiable with jax.grad."""
        def loss(data):
            f = Field(data=data, name="test", dims=("face", "x", "y"), units="m")
            gx = gradient_x(f, small_grid)
            return jnp.sum(gx.data ** 2)

        data = jax.random.normal(jax.random.PRNGKey(0), (6, 8, 8))
        grad_fn = jax.grad(loss)
        grads = grad_fn(data)
        assert jnp.all(jnp.isfinite(grads))

    def test_divergence_differentiable(self, small_grid):
        """Divergence should be differentiable."""
        def loss(u_data, v_data):
            u = Field(data=u_data, name="u", dims=("face", "x", "y"), units="m/s")
            v = Field(data=v_data, name="v", dims=("face", "x", "y"), units="m/s")
            div = divergence(u, v, small_grid)
            return jnp.sum(div.data ** 2)

        key = jax.random.PRNGKey(1)
        u_data = jax.random.normal(key, (6, 8, 8))
        v_data = jax.random.normal(jax.random.split(key)[0], (6, 8, 8))
        grads = jax.grad(loss, argnums=(0, 1))(u_data, v_data)
        assert all(jnp.all(jnp.isfinite(g)) for g in grads)
