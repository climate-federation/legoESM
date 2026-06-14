"""Phase 1 of the density-Jacobian PGF (Shchepetkin & McWilliams 2003).

Tests the harmonic-mean monotonized slope reconstruction
``reconstruct_harmonic_slopes`` in
``legoesm.ocean.dynamics.latlon_cgrid_operators``.

The harmonic-mean slope is the linchpin of S&M03: for linear ρ(z),
``σ_k = a`` (the slope) **exactly** in every column, regardless of
where the centroid sits.  This is what lets adjacent columns with
shifted centroids agree on ρ at intermediate depths so the
horizontal pressure gradient vanishes at rest.

Four tests:

1. ``TestLinearProfileUniformCells``: linear ρ(z) on uniform cells →
   exact slope at all interior cells, one-sided at boundaries.
2. ``TestLinearProfilePartialCells``: linear ρ(z) on a column with
   partial bottom cell → exact slope; two columns with different
   ``bottom_level`` agree on σ in their shared full-cell region.
3. ``TestSignChangingProfileMonotonization``: at a local extremum the
   harmonic-mean limiter clamps σ to zero.
4. ``TestADSmoothness``: ``jax.grad`` through the harmonic mean is
   finite (no NaN through the safe-divide / sign-test branches).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    reconstruct_harmonic_slopes,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


# ---------------------------------------------------------------------------
# 1. Linear ρ(z) on uniform cells
# ---------------------------------------------------------------------------


class TestLinearProfileUniformCells:
    """Linear ρ(z) → σ exact at every cell.

    For ``ρ(z) = a · z + b`` with uniform cell spacing,
    ``Δρ_top = Δρ_bot = a`` and the harmonic mean returns ``a``
    exactly.  At boundaries the one-sided fall-back also gives ``a``.
    """

    @pytest.mark.parametrize("slope", [0.0, 1.0e-3, -2.5e-3, 5.0])
    def test_uniform_column_exact_slope(self, slope):
        nlev = 10
        dz = 100.0
        z = (jnp.arange(nlev) + 0.5) * dz                 # centroids 50, 150, ..., 950
        rho = slope * z + 1027.0
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        sigma = reconstruct_harmonic_slopes(rho, z, is_active)
        np.testing.assert_allclose(np.array(sigma), np.full((nlev,), slope), atol=1e-12)

    def test_one_sided_at_top_and_bottom(self):
        """At k=0 only Δρ_bot exists; at k=nlev-1 only Δρ_top exists.
        For a linear profile both give the same slope."""
        nlev = 5
        dz = 10.0
        z = (jnp.arange(nlev) + 0.5) * dz
        slope = 0.4
        rho = slope * z + 1027.0
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        sigma = reconstruct_harmonic_slopes(rho, z, is_active)
        # σ_0 must equal Δρ_bot_0 = (ρ_0 − ρ_1)/(z_0 − z_1) = slope
        # σ_{nlev-1} must equal Δρ_top = slope
        np.testing.assert_allclose(float(sigma[0]), slope, atol=1e-12)
        np.testing.assert_allclose(float(sigma[-1]), slope, atol=1e-12)


# ---------------------------------------------------------------------------
# 2. Linear ρ(z) on partial-cell column
# ---------------------------------------------------------------------------


class TestLinearProfilePartialCells:
    """For linear ρ(z), σ is exact even when a column has a partial
    bottom cell — and adjacent columns with different
    ``bottom_level`` agree on σ wherever they have full cells in
    common.  This is the property that lets the horizontal Jacobian
    cancel at rest."""

    def test_partial_bottom_one_sided_slope(self):
        """A column with bottom_level < nlev−1: at the partial-bottom
        cell the slope is one-sided (Δρ_top) — for a linear profile
        this still equals the true slope."""
        nlev = 8
        dz_full = 100.0
        slope = 1.0e-3
        # Bottom cell is partial at depth 600m (half the reference dz_full).
        # Full cells: levels 0..5 with centroids at 50, 150, 250, 350, 450, 550.
        # Partial cell at level 6 with thickness 50m → centroid 625.
        # Levels 7 inactive.
        z = jnp.array([50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 625.0, 700.0])
        rho = slope * z + 1027.0
        is_active = jnp.array([True] * 7 + [False])

        sigma = reconstruct_harmonic_slopes(rho, z, is_active)
        # Interior cells (1..5) and partial bottom (6) and top (0):
        # all equal slope for a linear profile.
        np.testing.assert_allclose(
            np.array(sigma[:7]), np.full((7,), slope), atol=1e-12,
        )
        # Inactive cell forced to zero.
        assert float(sigma[7]) == 0.0

    def test_two_columns_agree_in_shared_region(self):
        """Two adjacent columns with same analytic ρ(z) but different
        ``bottom_level``: σ must agree at every cell where both are
        active and have both top + bottom neighbours active.  The
        harmonic-slope formula collapses to the linear slope in both
        columns."""
        nlev = 8
        slope = 2.5e-3

        # Column W: full cells, all 8 levels active.
        # Centroids at uniform 100m spacing.
        z_W = (jnp.arange(nlev) + 0.5) * 100.0
        rho_W = slope * z_W + 1027.0
        active_W = jnp.ones((nlev,), dtype=jnp.bool_)

        # Column E: full cells 0..5, partial cell at level 6 (thickness 30m,
        # so centroid at 600 + 15 = 615), inactive at 7.
        z_E = jnp.array([50.0, 150.0, 250.0, 350.0, 450.0, 550.0, 615.0, 700.0])
        rho_E = slope * z_E + 1027.0
        active_E = jnp.array([True] * 7 + [False])

        sigma_W = reconstruct_harmonic_slopes(rho_W, z_W, active_W)
        sigma_E = reconstruct_harmonic_slopes(rho_E, z_E, active_E)

        # Cells 1..5 in both columns are interior full cells with full
        # neighbours; both should give exactly ``slope``.
        np.testing.assert_allclose(
            np.array(sigma_W[1:6]), np.full((5,), slope), atol=1e-12,
        )
        np.testing.assert_allclose(
            np.array(sigma_E[1:6]), np.full((5,), slope), atol=1e-12,
        )
        # Critically: sigma_W and sigma_E agree at every shared
        # full-neighbour cell.
        np.testing.assert_allclose(
            np.array(sigma_W[1:6]), np.array(sigma_E[1:6]), atol=1e-12,
        )


# ---------------------------------------------------------------------------
# 3. Sign-changing ρ(z) — monotonization
# ---------------------------------------------------------------------------


class TestSignChangingProfileMonotonization:
    """At a local extremum where ``Δρ_top`` and ``Δρ_bot`` have
    opposite signs, the harmonic limiter must clamp σ = 0."""

    def test_local_extremum_clamps_to_zero(self):
        nlev = 5
        dz = 100.0
        z = (jnp.arange(nlev) + 0.5) * dz
        # ρ profile with a local maximum at level 2:
        # values 1, 2, 5, 2, 1
        rho = jnp.array([1.0, 2.0, 5.0, 2.0, 1.0])
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        sigma = reconstruct_harmonic_slopes(rho, z, is_active)
        # At level 2: Δρ_top = (2−5)/(150−250) = 0.03, Δρ_bot = (5−2)/(250−350) = −0.03.
        # Opposite signs → σ_2 = 0.
        np.testing.assert_allclose(float(sigma[2]), 0.0, atol=1e-15)

    def test_constant_profile_zero_slope(self):
        """Constant ρ(z) → σ = 0 everywhere (Δρ_top · Δρ_bot = 0,
        same_sign mask is False, falls through to 0)."""
        nlev = 6
        z = (jnp.arange(nlev) + 0.5) * 100.0
        rho = jnp.full((nlev,), 1027.0)
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        sigma = reconstruct_harmonic_slopes(rho, z, is_active)
        np.testing.assert_allclose(np.array(sigma), np.zeros((nlev,)), atol=1e-15)


# ---------------------------------------------------------------------------
# 4. AD smoothness
# ---------------------------------------------------------------------------


class TestADSmoothness:
    """``jax.grad`` through the harmonic mean must give finite
    gradients — the safe-divide pattern keeps both branches of every
    ``jnp.where`` finite."""

    def test_grad_wrt_rho_finite(self):
        nlev = 6
        z = (jnp.arange(nlev) + 0.5) * 100.0
        rho = 0.5e-3 * z + 1027.0
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        def loss(rho_in):
            sigma = reconstruct_harmonic_slopes(rho_in, z, is_active)
            return jnp.sum(sigma ** 2)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_with_partial_cells_finite(self):
        nlev = 6
        z = jnp.array([50.0, 150.0, 250.0, 350.0, 425.0, 500.0])
        rho = 0.5e-3 * z + 1027.0
        is_active = jnp.array([True, True, True, True, True, False])

        def loss(rho_in):
            sigma = reconstruct_harmonic_slopes(rho_in, z, is_active)
            return jnp.sum(sigma ** 2)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))

    def test_grad_at_extremum_finite(self):
        """Pathological case: the limiter discontinuity at sign change
        produces a piecewise gradient.  We require *finite* values
        only — not smoothness — at the boundary."""
        nlev = 5
        z = (jnp.arange(nlev) + 0.5) * 100.0
        # Profile with a near-extremum at level 2 (slope sign nearly flips).
        rho = jnp.array([1.0, 2.0, 5.0, 4.99, 1.0])
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)

        def loss(rho_in):
            sigma = reconstruct_harmonic_slopes(rho_in, z, is_active)
            return jnp.sum(sigma ** 2)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))


# ---------------------------------------------------------------------------
# 6. 2nd-order backward bottom-cell slope (curved-EOS cube cold-start fix)
# ---------------------------------------------------------------------------


class TestBottomSlope2ndOrder:
    """``bottom_slope_2nd_order`` replaces the O(Δz)-biased one-sided
    bottom-cell slope with a 3-point 2nd-order backward derivative AT the
    bottom centroid — exact for linear AND quadratic ρ(z), removing the
    curvature bias that seeds the cubed-sphere rest-state PGF residual under
    a pressure-dependent EOS, while staying bit-exact for linear ρ."""

    def _col(self):
        # 6 active cells + 1 inactive; non-uniform bottom spacing (partial).
        z = jnp.array([50.0, 150.0, 250.0, 350.0, 450.0, 525.0, 600.0])
        active = jnp.array([True] * 6 + [False])
        return z, active

    def test_default_off_bit_exact(self):
        """The DEFAULT (flag off) path is BITWISE-identical to not passing the
        kwarg at all — the new branch is never entered, so the proven lat-lon /
        tripole / MPAS callers (which never pass it) are unchanged."""
        z, active = self._col()
        rho = jnp.asarray(1.0e-6 * np.asarray(z) ** 2 + 1027.0)  # curved
        s_a = reconstruct_harmonic_slopes(rho, z, active)
        s_b = reconstruct_harmonic_slopes(
            rho, z, active, bottom_slope_2nd_order=False)
        np.testing.assert_array_equal(np.array(s_a), np.array(s_b))

    def test_linear_close_to_legacy(self):
        """Linear ρ: the 2nd-order curvature term vanishes algebraically, so the
        flag-on result equals the one-sided slope to round-off (NOT bitwise — it
        is a different FP op sequence; the delta-form keeps the diff ~1e-13)."""
        z, active = self._col()
        rho = -2.0e-3 * z + 1027.0
        s_legacy = reconstruct_harmonic_slopes(rho, z, active)
        s_2nd = reconstruct_harmonic_slopes(
            rho, z, active, bottom_slope_2nd_order=True)
        np.testing.assert_allclose(
            np.array(s_2nd), np.array(s_legacy), rtol=0, atol=1e-11)

    def test_quadratic_bottom_slope_more_accurate(self):
        """Curved (quadratic) ρ(z): the 2nd-order bottom slope matches the
        analytic dρ/dz at the bottom centroid; the one-sided legacy slope is
        biased away from it."""
        z, active = self._col()
        c = 1.0e-6
        rho = c * z * z + 1027.0                 # dρ/dz = 2c·z (analytic)
        z_bot = float(z[5])
        analytic = 2.0 * c * z_bot
        s_legacy = reconstruct_harmonic_slopes(rho, z, active)
        s_2nd = reconstruct_harmonic_slopes(
            rho, z, active, bottom_slope_2nd_order=True)
        np.testing.assert_allclose(float(s_2nd[5]), analytic, rtol=1e-9)
        err_2nd = abs(float(s_2nd[5]) - analytic)
        err_legacy = abs(float(s_legacy[5]) - analytic)
        assert err_legacy > 10.0 * err_2nd, (
            f"legacy err {err_legacy:.3e} not >> 2nd-order err {err_2nd:.3e}"
        )

    def test_fallback_when_too_shallow(self):
        """A column with only 2 active cells (no k−2) falls back to the
        one-sided slope under the 2nd-order flag (no NaN, no change)."""
        z = jnp.array([50.0, 150.0, 250.0])
        active = jnp.array([True, True, False])
        rho = jnp.array([1027.0, 1028.0, 1027.5])
        s_legacy = reconstruct_harmonic_slopes(rho, z, active)
        s_2nd = reconstruct_harmonic_slopes(
            rho, z, active, bottom_slope_2nd_order=True)
        np.testing.assert_array_equal(np.array(s_legacy), np.array(s_2nd))

    def test_grad_finite_2nd_order(self):
        """jax.grad through the 2nd-order bottom-slope path is finite."""
        z, active = self._col()
        rho = jnp.asarray(1.0e-6 * np.asarray(z) ** 2 + 1027.0)

        def loss(rho_in):
            sigma = reconstruct_harmonic_slopes(
                rho_in, z, active, bottom_slope_2nd_order=True)
            return jnp.sum(sigma ** 2)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))
