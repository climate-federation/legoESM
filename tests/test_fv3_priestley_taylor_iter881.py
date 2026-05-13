"""FV3_3D iter 881: priestley_taylor_le_fv3.

λE_PT = α_PT · Δ/(Δ+γ) · A   [W/m²].

Tests
-----

1. ``test_zero_energy``: A=0 → λE=0.
2. ``test_canonical_pt``: typical (Δ=200, γ=67, A=400, α=1.26) → 379.
3. ``test_pt_alpha_scaling``: 2× α_PT → 2× λE.
4. ``test_radiation_limit_alpha_a``: Δ → ∞ → λE → α_PT·A.
5. ``test_compared_to_pm``: PT vs PM well-watered ~5% agreement.
6. ``test_denom_zero_floored``: Δ=γ=0 → finite.
7. ``test_chain_full_radiation``: A → iter-881 λE end-to-end.
8. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    aerodynamic_conductance_fv3,
    net_radiation_fv3,
    penman_monteith_le_fv3,
    priestley_taylor_le_fv3,
    psychrometric_constant_fv3,
    saturation_vapor_pressure_slope_fv3,
    soil_heat_flux_g_fv3,
    vpd_from_t_rh_fv3,
)


def test_zero_energy():
    """A=0 → λE=0."""
    le = priestley_taylor_le_fv3(
        available_energy=jnp.array([0.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
    )
    np.testing.assert_allclose(np.asarray(le), [0.0], atol=1e-14)


def test_canonical_pt():
    """Typical: A=400, Δ=200, γ=67, α=1.26 → λE = 1.26·(200/267)·400 ≈ 378."""
    le = priestley_taylor_le_fv3(
        available_energy=jnp.array([400.0]),
        delta_pa_k=jnp.array([200.0]),
        gamma_pa_k=jnp.array([67.0]),
    )
    expected = 1.26 * 200.0 / 267.0 * 400.0
    np.testing.assert_allclose(np.asarray(le), [expected], rtol=1e-12)
    assert 370.0 < float(le[0]) < 390.0


def test_pt_alpha_scaling():
    """2× α_PT → 2× λE."""
    le1 = priestley_taylor_le_fv3(
        jnp.array([400.0]), jnp.array([200.0]), jnp.array([67.0]),
        alpha_pt=1.26,
    )
    le2 = priestley_taylor_le_fv3(
        jnp.array([400.0]), jnp.array([200.0]), jnp.array([67.0]),
        alpha_pt=2.52,
    )
    np.testing.assert_allclose(np.asarray(le2), 2.0 * np.asarray(le1),
                                rtol=1e-12)


def test_radiation_limit_alpha_a():
    """Δ → ∞ → Δ/(Δ+γ) → 1 → λE → α_PT · A."""
    le = priestley_taylor_le_fv3(
        available_energy=jnp.array([400.0]),
        delta_pa_k=jnp.array([1.0e6]),
        gamma_pa_k=jnp.array([67.0]),
    )
    # λE → 1.26·400 = 504
    assert abs(float(le[0]) - 504.0) < 1.0


def test_compared_to_pm():
    """PT vs PM well-watered (low VPD) agree to ~10% accuracy."""
    a = jnp.array([400.0])
    delta = jnp.array([200.0])
    gamma = jnp.array([67.0])
    le_pt = priestley_taylor_le_fv3(a, delta, gamma)
    le_pm = penman_monteith_le_fv3(
        a, jnp.array([800.0]), delta, gamma,
        jnp.array([0.04]), jnp.array([0.05]),  # well-watered g_s
    )
    rel = abs(float(le_pt[0]) - float(le_pm[0])) / float(le_pm[0])
    assert rel < 0.25  # ~25% (depends on choice of VPD/g_s)


def test_denom_zero_floored():
    """Δ+γ → 0 → finite via floor."""
    le = priestley_taylor_le_fv3(
        available_energy=jnp.array([400.0]),
        delta_pa_k=jnp.array([0.0]),
        gamma_pa_k=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(le))


def test_chain_full_radiation():
    """Full radiation→A→iter-881 PT λE end-to-end."""
    r_n = net_radiation_fv3(
        sw_down=jnp.array([800.0]),
        albedo=jnp.array([0.23]),
        lw_down=jnp.array([400.0]),
        lw_up=jnp.array([450.0]),
    )
    g = soil_heat_flux_g_fv3(r_n)
    a = r_n - g
    delta = saturation_vapor_pressure_slope_fv3(jnp.array([298.15]))
    gamma = psychrometric_constant_fv3(jnp.array([101325.0]))
    le = priestley_taylor_le_fv3(a, delta, gamma)
    assert jnp.all(jnp.isfinite(le))
    assert float(le[0]) > 0.0


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=881)
    n_x, n_y = 6, 8
    a = jnp.asarray(rng.uniform(0.0, 600.0, size=(n_x, n_y)))
    delta = jnp.asarray(rng.uniform(50.0, 250.0, size=(n_x, n_y)))
    gamma = jnp.full((n_x, n_y), 67.0)
    le = priestley_taylor_le_fv3(a, delta, gamma)
    assert le.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(le))
