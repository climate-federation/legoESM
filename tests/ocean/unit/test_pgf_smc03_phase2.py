"""Phase 2 of the density-Jacobian PGF (Shchepetkin & McWilliams 2003).

Tests ``compute_pressure_at_target_smc03`` in
``legoesm.ocean.dynamics.latlon_cgrid_operators``.

The reconstruction must satisfy four properties:

1. **Constant ρ**: ``P(z) = g · ρ · z`` (depth positive downward).
2. **Linear ρ(z) = a·z + b**: ``P(z) = g · (a/2 · z² + b · z)`` exact
   inside every cell — this is where the harmonic-slope σ becomes
   essential, since ``σ = a`` exactly.
3. **Continuity across cell interfaces**: evaluating at ``z_top_k``
   from cell k must equal evaluating at ``z_bot_{k-1}`` from cell
   k−1 to machine precision.
4. **Cross-column consistency**: two columns with the same analytic
   ρ(z) but different cell discretizations (one full cells, one
   partial-bottom) must give identical P at any common z.  This is
   the property that makes the horizontal Jacobian vanish at rest.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    compute_pressure_at_target_smc03,
    reconstruct_harmonic_slopes,
)


@pytest.fixture(autouse=True)
def _enable_x64():
    orig = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    yield
    jax.config.update("jax_enable_x64", orig)


G = constants.g


def _build_uniform_column(nlev, dz, slope, rho_ref=1027.0):
    """Helper: uniform-cell column with linear ρ(z)."""
    h = jnp.full((nlev,), dz)
    z_top = jnp.arange(nlev, dtype=jnp.float64) * dz
    z_centroid = z_top + 0.5 * dz
    rho = slope * z_centroid + rho_ref
    is_active = jnp.ones((nlev,), dtype=jnp.bool_)
    sigma = reconstruct_harmonic_slopes(rho, z_centroid, is_active)
    return h, z_centroid, rho, sigma


# ---------------------------------------------------------------------------
# 1. Constant density
# ---------------------------------------------------------------------------


class TestConstantDensity:
    """``ρ(z) = ρ_const`` → ``P(z) = g · ρ · z``."""

    def test_constant_rho(self):
        nlev = 5
        dz = 100.0
        rho_const = 1030.0
        h = jnp.full((nlev,), dz)
        z_top = jnp.arange(nlev, dtype=jnp.float64) * dz
        z_centroid = z_top + 0.5 * dz
        rho = jnp.full((nlev,), rho_const)
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)
        sigma = reconstruct_harmonic_slopes(rho, z_centroid, is_active)

        # Sample at multiple depths inside different cells.
        z_target = jnp.array([0.0, 25.0, 100.0, 175.0, 250.0, 425.0, 500.0])
        P = compute_pressure_at_target_smc03(rho, h, z_centroid, sigma, z_target, G)
        expected = G * rho_const * z_target
        np.testing.assert_allclose(np.array(P), np.array(expected), atol=1e-9, rtol=1e-12)


# ---------------------------------------------------------------------------
# 2. Linear ρ(z) = a·z + b
# ---------------------------------------------------------------------------


class TestLinearDensity:
    """``ρ(z) = a·z + b`` → ``P(z) = g·(a·z²/2 + b·z)``.  This is the
    most stringent in-cell test of the harmonic-slope reconstruction:
    only when ``σ_k = a`` exactly is the in-cell integral analytic."""

    @pytest.mark.parametrize("slope", [1.0e-3, -2.0e-3, 5.0])
    def test_linear_rho_exact(self, slope):
        nlev = 8
        dz = 50.0
        rho_ref = 1027.0
        h, z_c, rho, sigma = _build_uniform_column(nlev, dz, slope, rho_ref)

        # Sample at fine grid covering all cells.
        z_total = nlev * dz
        z_target = jnp.linspace(0.0, z_total, 41)
        P = compute_pressure_at_target_smc03(rho, h, z_c, sigma, z_target, G)
        expected = G * (0.5 * slope * z_target ** 2 + rho_ref * z_target)
        # Tolerance scales with the column total pressure ≈ G·rho·z.
        np.testing.assert_allclose(
            np.array(P), np.array(expected), atol=1e-6, rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# 3. Continuity across cell interfaces
# ---------------------------------------------------------------------------


class TestInterfaceContinuity:
    """``P`` evaluated at a cell-interface depth must match whether
    we approach from the cell above or the cell below.  This is
    enforced by the cell-mean cumulative-sum identity for ``P_top``."""

    def test_continuity_at_each_interface(self):
        nlev = 6
        dz = 80.0
        # Non-trivial profile so σ ≠ 0 — but harmonic slope still gives
        # σ = slope exactly for linear ρ.
        h, z_c, rho, sigma = _build_uniform_column(nlev, dz, slope=1.5e-3)

        # Each cell-bottom is also the next cell's top — argmax with
        # interface ties deterministically picks the FIRST matching
        # cell (the upper one), so what we actually verify is that
        # P at the interface matches the analytic value in BOTH the
        # upper-cell limit and the lower-cell limit (offset by a
        # tiny ε down).
        eps = 1.0e-6
        z_interfaces = jnp.arange(1, nlev) * dz
        # Slight offsets above and below the interface so argmax picks
        # the upper cell vs the lower cell respectively.
        P_from_above = compute_pressure_at_target_smc03(
            rho, h, z_c, sigma, z_interfaces - eps, G,
        )
        P_from_below = compute_pressure_at_target_smc03(
            rho, h, z_c, sigma, z_interfaces + eps, G,
        )
        # As eps → 0 both must converge to the same value.  At eps=1e-6m
        # the gap is ~ G·ρ·eps ≈ 0.01 Pa — so we accept atol 0.1 Pa.
        np.testing.assert_allclose(
            np.array(P_from_above), np.array(P_from_below), atol=0.1,
        )


# ---------------------------------------------------------------------------
# 4. Cross-column consistency on partial cells
# ---------------------------------------------------------------------------


class TestCrossColumnConsistency:
    """Two columns with the same analytic ρ(z) but different cell
    discretizations must return identical P at common depths.  This
    is the load-bearing property for the horizontal Jacobian: the
    rest-state PGF will vanish iff the columns agree on P at the
    face-reference depth."""

    def test_full_vs_partial_bottom(self):
        """Column W has 6 full cells of dz=100m → bottom 600m.
        Column E has 5 full cells of dz=100m + 1 partial cell of
        h=50m → bottom 550m.  Both have the same linear ρ(z).
        Compare P at depths in their shared range (0..500m)."""
        slope = 2.0e-3
        rho_ref = 1027.0

        # Column W: full cells.
        nlev_W = 6
        dz = 100.0
        h_W = jnp.full((nlev_W,), dz)
        z_top_W = jnp.arange(nlev_W, dtype=jnp.float64) * dz
        z_c_W = z_top_W + 0.5 * dz
        rho_W = slope * z_c_W + rho_ref
        active_W = jnp.ones((nlev_W,), dtype=jnp.bool_)
        sigma_W = reconstruct_harmonic_slopes(rho_W, z_c_W, active_W)

        # Column E: 5 full + 1 partial (50m), then 1 inactive cell padded
        # so both shapes match (matters for batched usage).
        nlev_E = 7
        h_E = jnp.array([100.0, 100.0, 100.0, 100.0, 100.0, 50.0, 0.0])
        z_top_E = jnp.cumsum(h_E) - h_E
        z_bot_E = jnp.cumsum(h_E)
        z_c_E = z_top_E + 0.5 * h_E
        # Inactive cell's z_centroid is just the seafloor — ρ there is
        # set to the analytic continuation; sigma will be zero anyway.
        rho_E = slope * z_c_E + rho_ref
        active_E = jnp.array([True] * 6 + [False])
        sigma_E = reconstruct_harmonic_slopes(rho_E, z_c_E, active_E)

        # Shared depths: anywhere in [0, 500m] is in active range of both.
        z_target = jnp.linspace(0.0, 500.0, 21)
        P_W = compute_pressure_at_target_smc03(rho_W, h_W, z_c_W, sigma_W, z_target, G)
        # Column E uses padded shape:
        P_E = compute_pressure_at_target_smc03(rho_E, h_E, z_c_E, sigma_E, z_target, G)

        np.testing.assert_allclose(np.array(P_W), np.array(P_E), atol=1e-6, rtol=1e-12)

    def test_cross_column_two_partial_columns(self):
        """Two columns with **different** partial-cell thicknesses
        but the same linear ρ(z): P at any common depth still
        agrees.  This is the configuration the BH seamount creates:
        adjacent columns with different ``bottom_level`` and
        different partial thicknesses."""
        slope = 1.5e-3
        rho_ref = 1027.0

        # Column A: 4 full + partial 75m + 2 inactive
        nlev = 7
        h_A = jnp.array([100.0, 100.0, 100.0, 100.0, 75.0, 0.0, 0.0])
        # Column B: 4 full + partial 30m + 2 inactive
        h_B = jnp.array([100.0, 100.0, 100.0, 100.0, 30.0, 0.0, 0.0])

        def build(h):
            z_top = jnp.cumsum(h) - h
            z_c = z_top + 0.5 * h
            rho = slope * z_c + rho_ref
            active = h > 0.0
            sigma = reconstruct_harmonic_slopes(rho, z_c, active)
            return h, z_c, rho, sigma

        hA, zcA, rhoA, sigmaA = build(h_A)
        hB, zcB, rhoB, sigmaB = build(h_B)

        # Common active depth range: 0..400m (both have 4 full cells).
        z_target = jnp.linspace(0.0, 400.0, 17)
        P_A = compute_pressure_at_target_smc03(rhoA, hA, zcA, sigmaA, z_target, G)
        P_B = compute_pressure_at_target_smc03(rhoB, hB, zcB, sigmaB, z_target, G)
        np.testing.assert_allclose(np.array(P_A), np.array(P_B), atol=1e-6, rtol=1e-12)


# ---------------------------------------------------------------------------
# 5. AD smoothness sanity check
# ---------------------------------------------------------------------------


class TestADSmokeTest:
    """``jax.grad`` of a sum-of-pressures w.r.t. ρ must be finite.
    Notes from the plan: ``argmax`` is piecewise-constant in z_target,
    so we differentiate w.r.t. ρ (smooth) — not z_target."""

    def test_grad_wrt_rho_finite(self):
        nlev = 5
        h = jnp.full((nlev,), 100.0)
        z_top = jnp.arange(nlev, dtype=jnp.float64) * 100.0
        z_c = z_top + 50.0
        rho = 1.0e-3 * z_c + 1027.0
        is_active = jnp.ones((nlev,), dtype=jnp.bool_)
        z_target = jnp.array([25.0, 175.0, 325.0])

        def loss(rho_in):
            sigma = reconstruct_harmonic_slopes(rho_in, z_c, is_active)
            P = compute_pressure_at_target_smc03(rho_in, h, z_c, sigma, z_target, G)
            return jnp.sum(P)

        grad = jax.grad(loss)(rho)
        assert jnp.all(jnp.isfinite(grad))


def test_enclosing_cell_search_matches_the_dense_table():
    """The sorted search must pick the SAME cell the comparison table did.

    The dense form built ``(..., nlev, n_t)`` booleans and took the first
    True.  On a fine mesh that table is tens of gigabytes -- it is why the
    scheme could not run on the ico7 Voronoi grid -- so the search is now a
    ``searchsorted``.  Same index, including the interface tie: a target
    sitting exactly on an interface belongs to the cell ABOVE it under both
    rules.  Reverting the search to the old table must leave this passing,
    which is the point: this test pins the INDEX, not the implementation.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np

    rng = np.random.default_rng(20260916)
    n_col, nlev = 37, 12
    h = jnp.asarray(rng.uniform(5.0, 120.0, size=(n_col, nlev)))
    z_bot = jnp.cumsum(h, axis=-1)
    z_top = z_bot - h

    # Targets: cell centroids, exact interfaces (the tie), the surface and
    # the seafloor (the clamped ends).
    z_centroid = z_bot - 0.5 * h
    targets = jnp.concatenate(
        [z_centroid, z_bot, z_top,
         jnp.zeros((n_col, 1)), z_bot[:, -1:]], axis=-1)
    z_t = jnp.clip(targets, min=0.0, max=z_bot[:, -1:])

    in_cell = (z_t[:, None, :] >= z_top[:, :, None]) & (
        z_t[:, None, :] <= z_bot[:, :, None])
    k_dense = jnp.argmax(in_cell.astype(jnp.int32), axis=-2)

    k_search = jax.vmap(
        lambda b, t: jnp.searchsorted(b, t, side="left"))(z_bot, z_t)

    np.testing.assert_array_equal(np.asarray(k_dense), np.asarray(k_search))
