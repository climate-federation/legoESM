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

    def test_conservation_coarsen_nlev(self):
        """PR B #3: ∫ q dp is conserved when the target grid has FEWER layers
        (nlev_new < nlev_old). The old-grid pointer was clamped to nlev_new-1,
        so the deepest source layers were unreachable and their mass was lost.
        The equal-length test_conservation above never exercised this."""
        q = jnp.array([1.0, 3.0, 2.0, 5.0, 4.0, 6.0])                  # nlev_old=6
        dp_old = jnp.array([100.0, 200.0, 150.0, 250.0, 150.0, 150.0])  # Σ=1000
        dp_new = jnp.array([250.0, 250.0, 250.0, 250.0])                # nlev_new=4, Σ=1000
        q_new = vertical_remap_ppm(q, dp_old, dp_new)
        assert q_new.shape == (4,)
        integral_old = float(jnp.sum(q * dp_old))
        integral_new = float(jnp.sum(q_new * dp_new))
        rel_err = abs(integral_new - integral_old) / abs(integral_old)
        assert rel_err < 1e-9, f"Conservation error (coarsen 6->4): {rel_err}"

    def test_conservation_refine_nlev(self):
        """PR B #3: refining 4 -> 7 layers (nlev_new > nlev_old). Clamping the
        old-grid pointer to nlev_old-1 (not nlev_new-1) also avoids the OOB
        read XLA would otherwise clamp."""
        q = jnp.array([2.0, 4.0, 1.0, 5.0])                            # nlev_old=4
        dp_old = jnp.array([300.0, 200.0, 250.0, 250.0])               # Σ=1000
        dp_new = jnp.full((7,), 1000.0 / 7.0)                          # nlev_new=7, Σ=1000
        q_new = vertical_remap_ppm(q, dp_old, dp_new)
        assert q_new.shape == (7,)
        integral_old = float(jnp.sum(q * dp_old))
        integral_new = float(jnp.sum(q_new * dp_new))
        rel_err = abs(integral_new - integral_old) / abs(integral_old)
        assert rel_err < 1e-9, f"Conservation error (refine 4->7): {rel_err}"

    def test_no_mass_fabrication_when_target_exceeds_source(self):
        """PR B #3 (codex): when the target column has MORE total pressure than
        the source (Σdp_new > Σdp_old), the sweep must NOT re-consume the last
        source layer and invent mass. All source mass is distributed once and
        the excess target space stays empty, so ∫q_new dp_new must NOT exceed
        ∫q_old dp_old (the old code fabricated mass here)."""
        q = jnp.array([2.0, 4.0, 6.0])                     # nlev_old=3
        dp_old = jnp.array([100.0, 200.0, 100.0])          # Σ=400
        dp_new = jnp.array([150.0, 150.0, 150.0, 150.0])   # nlev_new=4, Σ=600>400
        q_new = vertical_remap_ppm(q, dp_old, dp_new)
        assert q_new.shape == (4,)
        assert jnp.all(jnp.isfinite(q_new))
        integral_old = float(jnp.sum(q * dp_old))
        integral_new = float(jnp.sum(q_new * dp_new))
        # No fabrication: the remapped mass must not exceed the source mass.
        assert integral_new <= integral_old * (1.0 + 1e-9), (
            f"mass fabricated: new={integral_new} > old={integral_old}"
        )
        # All source mass IS distributed once (excess cells just stay empty).
        rel_err = abs(integral_new - integral_old) / abs(integral_old)
        assert rel_err < 1e-9, f"source mass lost (excess-target): {rel_err}"
