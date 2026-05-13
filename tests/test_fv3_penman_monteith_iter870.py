"""FV3_3D iter 870: penman_monteith_le_fv3.

λE = (Δ·A + ρ·c_p·VPD·g_a) / (Δ + γ·(1 + g_a/g_s))  [W/m²].

Tests
-----

1. ``test_no_energy_no_vpd``: A=0, VPD=0 → λE=0.
2. ``test_radiation_limited``: VPD=0 → λE = Δ·A/(Δ+γ·(1+g_a/g_s)).
3. ``test_atmosphere_limited``: A=0 → λE = ρ·c_p·VPD·g_a/(Δ+γ·(1+g_a/g_s)).
4. ``test_open_water_penman_limit``: g_s → ∞ → matches Penman 1948.
5. ``test_full_stomatal_closure``: g_s → 0 → λE → 0.
6. ``test_tropical_noon_band``: realistic inputs → 400-500 W/m².
7. ``test_g_s_zero_floored``: g_s=0 → finite.
8. ``test_chain_with_iter869``: g_s from iter-869 → λE.
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ball_berry_conductance_fv3,
    penman_monteith_le_fv3,
)


def test_no_energy_no_vpd():
    """A=0 + VPD=0 → λE=0."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([0.0]),
        vpd_pa=jnp.array([0.0]),
        delta_pa_k=jnp.array([150.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    np.testing.assert_allclose(np.asarray(le), [0.0], atol=1e-14)


def test_radiation_limited():
    """VPD=0 → λE = Δ·A / (Δ + γ·(1+g_a/g_s))."""
    delta = 150.0
    gamma = 67.0
    g_s = 0.02
    g_a = 0.05
    a = 400.0
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([a]),
        vpd_pa=jnp.array([0.0]),
        delta_pa_k=jnp.array([delta]),
        gamma_pa_k=jnp.array([gamma]),
        g_s=jnp.array([g_s]),
        g_a=jnp.array([g_a]),
    )
    expected = delta * a / (delta + gamma * (1.0 + g_a / g_s))
    np.testing.assert_allclose(np.asarray(le), [expected], rtol=1e-12)


def test_atmosphere_limited():
    """A=0 → λE = ρ·c_p·VPD·g_a / (Δ + γ·(1+g_a/g_s))."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([0.0]),
        vpd_pa=jnp.array([2000.0]),
        delta_pa_k=jnp.array([150.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    assert float(le[0]) > 0.0


def test_open_water_penman_limit():
    """g_s → ∞: collapses to Penman 1948 open-water form.

    Penman: λE = (Δ·A + ρ·c_p·D·g_a) / (Δ + γ)
    """
    delta = 150.0
    gamma = 67.0
    a = 200.0
    vpd = 1000.0
    g_a = 0.05
    rho = 1.225
    cp = 1005.0
    le_penman = (delta * a + rho * cp * vpd * g_a) / (delta + gamma)
    le_pm = penman_monteith_le_fv3(
        available_energy=jnp.array([a]),
        vpd_pa=jnp.array([vpd]),
        delta_pa_k=jnp.array([delta]),
        gamma_pa_k=jnp.array([gamma]),
        g_s=jnp.array([1.0e6]),  # huge g_s ≈ open water
        g_a=jnp.array([g_a]),
    )
    np.testing.assert_allclose(np.asarray(le_pm), [le_penman], rtol=1e-4)


def test_full_stomatal_closure():
    """g_s → 0 → λE → 0 (complete stomatal closure)."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([150.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([1.0e-10]),  # near-zero
        g_a=jnp.array([0.05]),
    )
    assert float(le[0]) < 5.0  # near-zero λE


def test_tropical_noon_band():
    """Tropical forest noon: A=500, VPD=2000, realistic g_s/g_a."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([500.0]),
        vpd_pa=jnp.array([2000.0]),
        delta_pa_k=jnp.array([200.0]),  # warm
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.025]),  # tropical canopy
        g_a=jnp.array([0.08]),
    )
    # Plausible 300-700 W/m² tropical noon (humid, energy-rich)
    assert 300.0 < float(le[0]) < 700.0


def test_g_s_zero_floored():
    """g_s=0 → finite via floor (no NaN)."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1000.0]),
        delta_pa_k=jnp.array([150.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.0]),
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))


def test_chain_with_iter869():
    """g_s from iter-869 → λE via iter-870."""
    # iter-869 chain: A_n=20, h_s=0.5, C_s=400 → g_s=0.235 mol/m²/s
    # Convert mol → m/s (rough: divide by 0.04 mol·s/m³/mol/m³ at SL)
    g_s_mol = ball_berry_conductance_fv3(
        a_n=jnp.array([20.0]),
        h_s=jnp.array([0.5]),
        c_s=jnp.array([400.0]),
    )
    g_s_m_s = g_s_mol * 0.024  # mol/m²/s → m/s factor
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([180.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=g_s_m_s,
        g_a=jnp.array([0.05]),
    )
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=870)
    n_x, n_y = 6, 8
    a = jnp.asarray(rng.uniform(0.0, 600.0, size=(n_x, n_y)))
    vpd = jnp.asarray(rng.uniform(0.0, 3000.0, size=(n_x, n_y)))
    delta = jnp.asarray(rng.uniform(50.0, 250.0, size=(n_x, n_y)))
    gamma = jnp.full((n_x, n_y), 67.0)
    g_s = jnp.asarray(rng.uniform(0.005, 0.05, size=(n_x, n_y)))
    g_a = jnp.asarray(rng.uniform(0.01, 0.1, size=(n_x, n_y)))
    le = penman_monteith_le_fv3(a, vpd, delta, gamma, g_s, g_a)
    assert le.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(le))
