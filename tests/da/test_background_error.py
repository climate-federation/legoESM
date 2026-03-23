"""Tests for background error covariance."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.da.background_error import DiagonalB, DiffusionB, _gaspari_cohn


class TestDiagonalB:
    def test_round_trip(self):
        """B_inv @ B @ x ≈ x."""
        sigma = jnp.array([1.0, 2.0, 3.0, 0.5])
        B = DiagonalB(sigma=sigma)
        x = jnp.array([1.0, -2.0, 0.5, 3.0])
        Bx = B.sqrt_multiply(B.sqrt_multiply(x))  # B = B^{1/2} @ B^{1/2}
        recovered = B.inv_multiply(Bx)
        assert jnp.allclose(recovered, x, atol=1e-6)

    def test_positive_definite(self):
        """x^T B^{-1} x > 0 for x != 0."""
        sigma = jnp.array([1.0, 2.0, 3.0])
        B = DiagonalB(sigma=sigma)
        key = jax.random.PRNGKey(0)
        x = jax.random.normal(key, (3,))
        quad = jnp.sum(x * B.inv_multiply(x))
        assert quad > 0

    def test_sqrt_multiply_differentiable(self):
        sigma = jnp.array([1.0, 2.0, 3.0])
        B = DiagonalB(sigma=sigma)
        x = jnp.array([1.0, -1.0, 0.5])
        grad = jax.grad(lambda v: jnp.sum(B.sqrt_multiply(v) ** 2))(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_inv_multiply_differentiable(self):
        sigma = jnp.array([1.0, 2.0, 3.0])
        B = DiagonalB(sigma=sigma)
        x = jnp.array([1.0, -1.0, 0.5])
        grad = jax.grad(lambda v: jnp.sum(B.inv_multiply(v) ** 2))(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_symmetry(self):
        """<Bx, y> = <x, By>."""
        sigma = jnp.array([1.0, 2.0, 3.0, 0.5])
        B = DiagonalB(sigma=sigma)
        key = jax.random.PRNGKey(42)
        x = jax.random.normal(key, (4,))
        key, subkey = jax.random.split(key)
        y = jax.random.normal(subkey, (4,))

        Bx = B.sqrt_multiply(B.sqrt_multiply(x))
        By = B.sqrt_multiply(B.sqrt_multiply(y))
        assert jnp.allclose(jnp.sum(Bx * y), jnp.sum(x * By), atol=1e-5)


class TestDiffusionB:
    @pytest.fixture
    def grid(self):
        from legoesm.grids import create_latlon_grid
        return create_latlon_grid(8, 16)

    def test_sqrt_multiply_differentiable(self, grid):
        n = grid.grid_n_columns
        sigma = jnp.ones(n) * 2.0
        B = DiffusionB(grid, sigma, horizontal_length_scale=1000e3, n_diffusion_iter=4)
        x = jax.random.normal(jax.random.PRNGKey(0), (n,))
        grad = jax.grad(lambda v: jnp.sum(B.sqrt_multiply(v)))(x)
        assert jnp.all(jnp.isfinite(grad))

    def test_smoothing_effect(self, grid):
        """Smoothing should reduce high-frequency variance."""
        n = grid.grid_n_columns
        sigma = jnp.ones(n)
        B = DiffusionB(grid, sigma, horizontal_length_scale=5000e3, n_diffusion_iter=10)
        x = jax.random.normal(jax.random.PRNGKey(1), (n,))
        # Internally _smooth_field should damp departures from global mean
        smoothed_raw = B._smooth_field(x, B.n_iter // 2 + 1)
        # Smoothed field should have less variance than the original
        assert jnp.var(smoothed_raw) < jnp.var(x)


class TestGaspariCohn:
    def test_at_zero(self):
        assert jnp.allclose(_gaspari_cohn(jnp.array(0.0), 1.0), 1.0)

    def test_at_2c(self):
        assert jnp.allclose(_gaspari_cohn(jnp.array(2.0), 1.0), 0.0, atol=1e-6)

    def test_beyond_2c(self):
        assert jnp.allclose(_gaspari_cohn(jnp.array(3.0), 1.0), 0.0, atol=1e-14)

    def test_monotonically_decreasing(self):
        r = jnp.linspace(0, 2.0, 100)
        vals = _gaspari_cohn(r, 1.0)
        assert jnp.all(jnp.diff(vals) <= 1e-6)

    def test_non_negative(self):
        r = jnp.linspace(0, 3.0, 200)
        vals = _gaspari_cohn(r, 1.0)
        assert jnp.all(vals >= -1e-10)
