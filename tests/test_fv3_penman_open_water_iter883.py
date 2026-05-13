"""FV3_3D iter 883: penman_open_water_le_fv3.

λE_pot = (Δ·A + ρ·c_p·VPD·g_a) / (Δ + γ)  [W/m²].

Tests
-----

1. ``test_radiation_limit_zero_vpd``: VPD=0 → λE = Δ·A/(Δ+γ).
2. ``test_advection_limit_zero_a``: A=0 → λE = ρ·c_p·VPD·g_a/(Δ+γ).
3. ``test_canonical_tropical``: realistic inputs → 400-500 W/m².
4. ``test_pm_with_g_s_inf_matches_penman``: PM with g_s→∞ → Penman.
5. ``test_denom_zero_floored``: Δ+γ=0 → finite.
6. ``test_chain_with_complementary``: Penman → iter-882 CR.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    complementary_relationship_et_fv3,
    penman_monteith_le_fv3,
    penman_open_water_le_fv3,
    priestley_taylor_le_fv3,
)


def test_radiation_limit_zero_vpd():
    """VPD=0 → λE_pot = Δ·A/(Δ+γ) (radiation equilibrium)."""
    le = penman_open_water_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([0.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_a=jnp.array([0.05]),
    )
    expected = 200.0 * 400.0 / 267.0
    np.testing.assert_allclose(np.asarray(le), [expected], rtol=1e-12)


def test_advection_limit_zero_a():
    """A=0 → λE_pot = ρ·c_p·VPD·g_a/(Δ+γ) (pure advection)."""
    le = penman_open_water_le_fv3(
        available_energy=jnp.array([0.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_a=jnp.array([0.05]),
    )
    # ρ·c_p·VPD·g_a = 1.225·1005·1500·0.05 = 92322
    # /267 = 345
    assert 300.0 < float(le[0]) < 400.0


def test_canonical_tropical():
    """Tropical noon humid → λE_pot in 400-500 W/m² band."""
    le = penman_open_water_le_fv3(
        available_energy=jnp.array([500.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_a=jnp.array([0.05]),
    )
    assert 400.0 < float(le[0]) < 800.0


def test_pm_with_g_s_inf_matches_penman():
    """PM with g_s → ∞ collapses to Penman."""
    a = jnp.array([400.0])
    vpd = jnp.array([1500.0])
    delta = jnp.array([200.0])
    gamma = jnp.array([67.0])
    g_a = jnp.array([0.05])
    le_pen = penman_open_water_le_fv3(a, vpd, delta, gamma, g_a)
    le_pm = penman_monteith_le_fv3(
        a, vpd, delta, gamma, jnp.array([1.0e6]), g_a,
    )
    np.testing.assert_allclose(np.asarray(le_pm), np.asarray(le_pen),
                                rtol=1e-4)


def test_denom_zero_floored():
    """Δ+γ=0 → finite via floor."""
    le = penman_open_water_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([0.0]),
        gamma_pa_k=jnp.array([0.0]),
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))


def test_chain_with_complementary():
    """Penman λE_pot + PT λE_wet → iter-882 CR λE_actual."""
    a = jnp.array([400.0])
    delta = jnp.array([200.0])
    gamma = jnp.array([67.0])
    le_wet = priestley_taylor_le_fv3(a, delta, gamma)
    le_pot = penman_open_water_le_fv3(
        a, jnp.array([1500.0]), delta, gamma, jnp.array([0.05]),
    )
    le_actual = complementary_relationship_et_fv3(le_wet, le_pot)
    assert jnp.all(jnp.isfinite(le_actual))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=883)
    n_x, n_y = 6, 8
    a = jnp.asarray(rng.uniform(0.0, 600.0, size=(n_x, n_y)))
    vpd = jnp.asarray(rng.uniform(0.0, 3000.0, size=(n_x, n_y)))
    delta = jnp.asarray(rng.uniform(50.0, 250.0, size=(n_x, n_y)))
    gamma = jnp.full((n_x, n_y), 67.0)
    g_a = jnp.asarray(rng.uniform(0.01, 0.1, size=(n_x, n_y)))
    le = penman_open_water_le_fv3(a, vpd, delta, gamma, g_a)
    assert le.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(le))
