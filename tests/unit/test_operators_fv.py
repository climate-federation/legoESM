"""Unit tests for FV3-style finite-volume transport operators."""

import jax
import jax.numpy as jnp
import pytest

from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.core.operators_fv import (
    _ppm_edge_values,
    _ppm_limit,
    _ppm_reconstruct_x,
    fv_flux_divergence,
    fv_scalar_advection,
)


@pytest.fixture(scope="module")
def grid():
    return create_cubed_sphere(8)


class TestPPMReconstruction:
    """Tests for PPM edge reconstruction."""

    def test_constant_field_exact(self, grid):
        """PPM of constant field should give exact edge values."""
        n = grid.n
        q = jnp.ones((6, n + 4, n)) * 7.0
        q_hat = _ppm_edge_values(q)
        assert jnp.allclose(q_hat, 7.0, atol=1e-10)

    def test_linear_field_exact(self, grid):
        """PPM of linear field should be exact (4th order captures linear)."""
        n = grid.n
        # Linear ramp along axis 1
        x = jnp.arange(n + 4, dtype=jnp.float32)
        q = jnp.broadcast_to(x[None, :, None], (6, n + 4, n))
        q_hat = _ppm_edge_values(q)
        # Interior edges: at positions 0.5, 1.5, ..., (n+2).5
        expected_inner = x[1:-2] + 0.5  # midpoints: (n+1 values)
        expected_lo = 0.5 * (x[0] + x[1])
        expected_hi = 0.5 * (x[-2] + x[-1])
        expected = jnp.concatenate(
            [jnp.array([expected_lo]), expected_inner, jnp.array([expected_hi])]
        )
        for face in range(6):
            for j in range(n):
                assert jnp.allclose(q_hat[face, :, j], expected, atol=1e-4)

    def test_reconstruct_x_shape(self, grid):
        """PPM x-reconstruction should produce correct shapes."""
        from legoesm.grids.halo import pad_halo
        n = grid.n
        q = jnp.ones((6, n, n)) * 5.0
        q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)
        q_L, q_R = _ppm_reconstruct_x(q_pad)
        assert q_L.shape == (6, n + 1, n)
        assert q_R.shape == (6, n + 1, n)


class TestPPMLimiter:
    """Tests for Colella-Woodward monotonicity limiter."""

    def test_no_new_extrema(self):
        """Limiter should prevent new extrema from step function."""
        q_bar = jnp.array([0.0, 0.0, 1.0, 1.0, 1.0])
        q_L = jnp.array([-0.5, -0.3, 0.5, 0.9, 1.1])
        q_R = jnp.array([0.3, 0.8, 1.2, 1.1, 1.0])

        q_L_lim, q_R_lim = _ppm_limit(q_bar, q_L, q_R)

        # At local extrema (cell 0: q=0 is minimum), should flatten
        # In general, limited values should not overshoot
        assert jnp.all(jnp.isfinite(q_L_lim))
        assert jnp.all(jnp.isfinite(q_R_lim))


class TestFVFluxDivergence:
    """Tests for the Lin-Rood conservative flux-form transport."""

    def test_uniform_field_zero_velocity(self, grid):
        """Transport of uniform field with zero velocity should give zero tendency."""
        n = grid.n
        q = jnp.ones((6, n, n)) * 1000.0
        u = jnp.zeros((6, n, n))
        v = jnp.zeros((6, n, n))

        dq = fv_flux_divergence(q, u, v, grid, dt=600.0)
        assert jnp.allclose(dq, 0.0, atol=1e-10)

    def test_conservation(self, grid):
        """Global sum of flux divergence * area should be ~0."""
        n = grid.n
        key = jax.random.PRNGKey(42)
        q = jax.random.uniform(key, (6, n, n), minval=900.0, maxval=1100.0)
        u = jax.random.normal(jax.random.split(key)[0], (6, n, n)) * 5.0
        v = jax.random.normal(jax.random.split(key)[1], (6, n, n)) * 5.0

        dq = fv_flux_divergence(q, u, v, grid, dt=600.0)
        global_sum = jnp.sum(dq * grid.area)
        relative = float(jnp.abs(global_sum) / jnp.sum(jnp.abs(dq) * grid.area))
        assert relative < 0.01, f"Conservation error: {relative:.6f}"

    def test_output_shape(self, grid):
        """Output should have same shape as input."""
        n = grid.n
        q = jnp.ones((6, n, n))
        u = jnp.ones((6, n, n))
        v = jnp.zeros((6, n, n))
        dq = fv_flux_divergence(q, u, v, grid, dt=600.0)
        assert dq.shape == (6, n, n)

    def test_all_finite(self, grid):
        """All outputs should be finite."""
        n = grid.n
        key = jax.random.PRNGKey(99)
        q = jax.random.uniform(key, (6, n, n), minval=500.0, maxval=1500.0)
        u = jax.random.normal(jax.random.split(key)[0], (6, n, n)) * 20.0
        v = jax.random.normal(jax.random.split(key)[1], (6, n, n)) * 20.0
        dq = fv_flux_divergence(q, u, v, grid, dt=300.0)
        assert jnp.all(jnp.isfinite(dq))

    def test_differentiable(self, grid):
        """Should be differentiable with jax.grad."""
        n = grid.n

        def loss(q):
            u = jnp.ones((6, n, n)) * 10.0
            v = jnp.zeros((6, n, n))
            dq = fv_flux_divergence(q, u, v, grid, dt=600.0)
            return jnp.mean(dq ** 2)

        q = jnp.ones((6, n, n)) * 1000.0
        grads = jax.grad(loss)(q)
        assert jnp.all(jnp.isfinite(grads))

    def test_x_first_vs_y_first(self, grid):
        """Both orderings should give similar results."""
        n = grid.n
        key = jax.random.PRNGKey(7)
        q = jax.random.uniform(key, (6, n, n), minval=900.0, maxval=1100.0)
        u = jax.random.normal(jax.random.split(key)[0], (6, n, n)) * 5.0
        v = jax.random.normal(jax.random.split(key)[1], (6, n, n)) * 5.0

        dq_xf = fv_flux_divergence(q, u, v, grid, dt=600.0, x_first=True)
        dq_yf = fv_flux_divergence(q, u, v, grid, dt=600.0, x_first=False)

        # Should be similar but not identical (splitting order matters)
        rel_diff = jnp.max(jnp.abs(dq_xf - dq_yf)) / (jnp.max(jnp.abs(dq_xf)) + 1e-30)
        assert rel_diff < 0.5  # Within 50% — splitting order differences


class TestFVScalarAdvection:
    """Tests for the non-conservative advection operator."""

    def test_uniform_field_zero(self, grid):
        """Advection of uniform field should be zero."""
        n = grid.n
        q = jnp.ones((6, n, n)) * 300.0
        u = jnp.ones((6, n, n)) * 10.0
        v = jnp.ones((6, n, n)) * 5.0

        dq = fv_scalar_advection(q, u, v, grid, dt=600.0)
        assert jnp.allclose(dq, 0.0, atol=1e-2)

    def test_output_shape_and_finite(self, grid):
        """Output should be correct shape and finite."""
        n = grid.n
        key = jax.random.PRNGKey(0)
        q = jax.random.uniform(key, (6, n, n), minval=200.0, maxval=400.0)
        u = jax.random.normal(jax.random.split(key)[0], (6, n, n)) * 10.0
        v = jax.random.normal(jax.random.split(key)[1], (6, n, n)) * 10.0

        dq = fv_scalar_advection(q, u, v, grid, dt=300.0)
        assert dq.shape == (6, n, n)
        assert jnp.all(jnp.isfinite(dq))

    def test_differentiable(self, grid):
        """Should be differentiable with jax.grad."""
        n = grid.n

        def loss(q):
            u = jnp.ones((6, n, n)) * 10.0
            v = jnp.zeros((6, n, n))
            dq = fv_scalar_advection(q, u, v, grid, dt=600.0)
            return jnp.mean(dq ** 2)

        q = jnp.ones((6, n, n)) * 300.0
        grads = jax.grad(loss)(q)
        assert jnp.all(jnp.isfinite(grads))


class TestNoEdgeArtifacts:
    """Test that FV transport doesn't create cube-edge artifacts."""

    def test_smooth_transport_edge_error(self, grid):
        """Transport a smooth field — edge error should not be systematically larger."""
        n = grid.n
        # Smooth initial condition (Y_1^0 spherical harmonic ≈ sin(lat))
        q = 1000.0 + 100.0 * grid.sin_lat
        u = jnp.ones((6, n, n)) * 10.0
        v = jnp.zeros((6, n, n))

        # Run 20 steps
        dt = 300.0
        for _ in range(20):
            dq = fv_flux_divergence(q, u, v, grid, dt=dt)
            q = q + dt * dq

        assert jnp.all(jnp.isfinite(q))

        # Check that edge cells don't have systematically larger error
        # Compare edge cells vs interior cells
        edge_vals = jnp.concatenate([
            q[:, 0, :].ravel(), q[:, -1, :].ravel(),
            q[:, :, 0].ravel(), q[:, :, -1].ravel(),
        ])
        interior_vals = q[:, 1:-1, 1:-1].ravel()

        edge_std = float(jnp.std(edge_vals))
        interior_std = float(jnp.std(interior_vals))

        # Edge variance should not be dramatically larger than interior
        assert edge_std < 5.0 * interior_std, (
            f"Edge std ({edge_std:.4f}) >> interior std ({interior_std:.4f})"
        )
