"""Unit tests for PPM vertical remapping."""

import jax.numpy as jnp
import pytest

from legoesm.core._future.vertical_remap import (
    _ppm_edge_values_vertical,
    _ppm_limit_vertical,
    vertical_remap_ppm,
)


class TestPPMEdgeValuesVertical:
    """Test the 1D PPM edge reconstruction along the vertical axis."""

    def test_output_shape(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_hat = _ppm_edge_values_vertical(q)
        assert q_hat.shape == (6,)  # nlev + 1

    def test_output_shape_batched(self):
        q = jnp.ones((4, 8, 10))
        q_hat = _ppm_edge_values_vertical(q)
        assert q_hat.shape == (4, 8, 11)

    def test_uniform_field(self):
        """Uniform q → all edges = q."""
        q = jnp.full((10,), 3.0)
        q_hat = _ppm_edge_values_vertical(q)
        assert jnp.allclose(q_hat, 3.0)

    def test_boundary_values(self):
        """Top and bottom edges should equal the adjacent cell value."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        q_hat = _ppm_edge_values_vertical(q)
        assert float(q_hat[0]) == 1.0   # top
        assert float(q_hat[-1]) == 5.0  # bottom

    def test_few_levels(self):
        """nlev=3 should use 2nd-order everywhere."""
        q = jnp.array([1.0, 3.0, 2.0])
        q_hat = _ppm_edge_values_vertical(q)
        assert q_hat.shape == (4,)
        assert jnp.all(jnp.isfinite(q_hat))

    def test_monotonicity_clamp(self):
        """Edges should be clamped between flanking cell values."""
        q = jnp.array([1.0, 5.0, 2.0, 4.0, 3.0])
        q_hat = _ppm_edge_values_vertical(q)
        # Each interior edge k should satisfy min(q[k-1],q[k]) <= q_hat[k] <= max(q[k-1],q[k])
        for k in range(1, len(q)):
            lo = min(float(q[k - 1]), float(q[k]))
            hi = max(float(q[k - 1]), float(q[k]))
            assert float(q_hat[k]) >= lo - 1e-10
            assert float(q_hat[k]) <= hi + 1e-10


class TestPPMLimitVertical:

    def test_limiter_preserves_constant(self):
        q_bar = jnp.full((5,), 2.0)
        q_L = jnp.full((5,), 2.0)
        q_R = jnp.full((5,), 2.0)
        q_L_lim, q_R_lim = _ppm_limit_vertical(q_bar, q_L, q_R)
        assert jnp.allclose(q_L_lim, 2.0)
        assert jnp.allclose(q_R_lim, 2.0)


class TestVerticalRemapPPM:

    def test_identity_remap(self):
        """Remapping to the same grid should be identity."""
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dp = jnp.full((5,), 200.0)
        q_new = vertical_remap_ppm(q, dp, dp)
        assert jnp.allclose(q_new, q, atol=1e-10)

    def test_conservation(self):
        """∫ q dp should be conserved after remapping."""
        q = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0])
        dp_old = jnp.array([100.0, 200.0, 150.0, 250.0, 300.0])
        dp_new = jnp.array([200.0, 200.0, 200.0, 200.0, 200.0])
        q_new = vertical_remap_ppm(q, dp_old, dp_new)
        integral_old = float(jnp.sum(q * dp_old))
        integral_new = float(jnp.sum(q_new * dp_new))
        rel_err = abs(integral_new - integral_old) / abs(integral_old)
        assert rel_err < 1e-10, f"Conservation error: {rel_err}"

    def test_output_shape_batched(self):
        """Batched input should preserve batch dims."""
        q = jnp.ones((4, 8, 5))
        dp = jnp.full((4, 8, 5), 200.0)
        q_new = vertical_remap_ppm(q, dp, dp)
        assert q_new.shape == (4, 8, 5)

    def test_finite_output(self):
        q = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        dp_old = jnp.full((5,), 200.0)
        dp_new = jnp.full((5,), 200.0)
        q_new = vertical_remap_ppm(q, dp_old, dp_new)
        assert jnp.all(jnp.isfinite(q_new))
