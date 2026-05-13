"""FV3_3D iter 806: lapse_rate_moist_fv3 (Γ_m).

Γ_m = g·(1+L_v·q_sat/(R_d·T)) / (c_p + L_v²·q_sat/(R_v·T²))

Composes thermo.saturation_specific_humidity.

Tests
-----

1. ``test_dry_limit``: q_sat=0 → Γ_m = g/c_p = Γ_d.
2. ``test_tropical_moist``: T=298, q_sat~17 g/kg → Γ_m ≈ 3-5 K/km.
3. ``test_monotonic_q``: ↑q_sat → ↓Γ_m.
4. ``test_monotonic_T``: warmer column → ↓Γ_m at same q_sat.
5. ``test_compose_thermo``: full T,p → q_sat → Γ_m.
6. ``test_shapes_3d_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants, thermo
from legoesm.grids.cubed_sphere import lapse_rate_moist_fv3


def test_dry_limit():
    """q_sat = 0 → Γ_m = g/c_p (dry adiabat)."""
    t = jnp.array([288.0])
    q_sat = jnp.array([0.0])
    gamma = lapse_rate_moist_fv3(t, q_sat)
    expected = constants.g / constants.c_pd
    np.testing.assert_allclose(np.asarray(gamma), [expected], rtol=1e-12)
    # ~9.76 K/km
    assert 9.5e-3 < float(gamma[0]) < 1.0e-2


def test_tropical_moist():
    """T=298 K, q_sat ≈ 17 g/kg → Γ_m ≈ 3-5 K/km (textbook tropical)."""
    t = jnp.array([298.0])
    q_sat = jnp.array([0.017])
    gamma = lapse_rate_moist_fv3(t, q_sat)
    # ~3-5 K/km
    assert 3.0e-3 < float(gamma[0]) < 5.5e-3


def test_monotonic_q():
    """↑q_sat → ↓Γ_m (more latent release damps cooling)."""
    t = jnp.full((4,), 288.0)
    q_lo = jnp.array([1e-4, 1e-3, 5e-3, 1e-2])
    q_hi = q_lo * 2.0
    gamma_lo = lapse_rate_moist_fv3(t, q_lo)
    gamma_hi = lapse_rate_moist_fv3(t, q_hi)
    assert jnp.all(gamma_hi < gamma_lo)


def test_monotonic_T_via_qsat():
    """Realistic: ↑T → ↑q_sat (CC) → ↓Γ_m. (Going through q_sat,
    not at fixed q_sat — the fixed-q_sat case shows ↑T → ↑Γ_m
    since the T² in denominator dominates the 1/T in numerator.)"""
    p = jnp.full((4,), 90_000.0)
    t_lo = jnp.array([270.0, 280.0, 290.0, 300.0])
    t_hi = t_lo + 5.0
    q_lo = thermo.saturation_specific_humidity(t_lo, p)
    q_hi = thermo.saturation_specific_humidity(t_hi, p)
    gamma_lo = lapse_rate_moist_fv3(t_lo, q_lo)
    gamma_hi = lapse_rate_moist_fv3(t_hi, q_hi)
    assert jnp.all(gamma_hi < gamma_lo)


def test_compose_thermo():
    """Full chain T,p → q_sat → Γ_m."""
    t = jnp.array([288.0])
    p = jnp.array([90_000.0])
    q_sat = thermo.saturation_specific_humidity(t, p)
    gamma = lapse_rate_moist_fv3(t, q_sat)
    # Realistic mid-lat moist adiabat: ~5-7 K/km
    assert 4.0e-3 < float(gamma[0]) < 7.5e-3


def test_shapes_3d_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=806)
    n_x, n_y, km = 4, 5, 20
    t = jnp.asarray(rng.uniform(220.0, 305.0, size=(n_x, n_y, km)))
    q_sat = jnp.asarray(rng.uniform(0.0, 0.025, size=(n_x, n_y, km)))
    gamma = lapse_rate_moist_fv3(t, q_sat)
    assert gamma.shape == (n_x, n_y, km)
    assert jnp.all(jnp.isfinite(gamma))
    assert jnp.all(gamma > 0.0)
