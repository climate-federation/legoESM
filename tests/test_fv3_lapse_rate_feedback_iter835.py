"""FV3_3D iter 835: lapse_rate_feedback_fv3.

λ_LR = λ_Planck · (ΔT̄_atm − ΔT_sfc) / ΔT_sfc.

Tests
-----

1. ``test_tropical_amplification_negative``: ΔT̄/ΔT_sfc=1.4 →
   λ_LR ≈ −1.5 W/m²/K.
2. ``test_uniform_zero``: ΔT̄ = ΔT_sfc → λ_LR = 0.
3. ``test_polar_surface_amplified_positive``: ΔT̄ < ΔT_sfc →
   λ_LR > 0.
4. ``test_chain_with_iter831``: λ_LR/λ_Planck = (ΔT̄−ΔT)/ΔT exactly.
5. ``test_zero_dT_sfc_floored``: ΔT_sfc=0 → finite (no NaN).
6. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    lapse_rate_feedback_fv3,
    planck_feedback_fv3,
)


def test_tropical_amplification_negative():
    """ΔT̄/ΔT_sfc=1.4 (moist adiabat) → λ_LR ≈ −1.5 W/m²/K."""
    t_eff = jnp.array([255.0])
    dT_sfc = jnp.array([1.0])
    dT_atm = jnp.array([1.4])
    lam_lr = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    # 0.4 × λ_Planck(255) ≈ 0.4 × −3.76 = −1.50
    assert -1.6 < float(lam_lr[0]) < -1.4


def test_uniform_zero():
    """ΔT̄ = ΔT_sfc (uniform Planck warming) → λ_LR = 0."""
    t_eff = jnp.array([255.0])
    dT_sfc = jnp.array([2.0])
    dT_atm = jnp.array([2.0])
    lam_lr = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    np.testing.assert_allclose(np.asarray(lam_lr), [0.0], atol=1e-13)


def test_polar_surface_amplified_positive():
    """Polar (ΔT̄ < ΔT_sfc): λ_LR > 0 (positive feedback)."""
    t_eff = jnp.array([255.0])
    dT_sfc = jnp.array([2.0])
    dT_atm = jnp.array([1.0])  # column warms less than surface
    lam_lr = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    assert float(lam_lr[0]) > 0.0


def test_chain_with_iter831():
    """λ_LR/λ_Planck = (ΔT̄ − ΔT_sfc) / ΔT_sfc exactly."""
    t_eff = jnp.array([255.0, 270.0])
    dT_sfc = jnp.array([1.0, 2.0])
    dT_atm = jnp.array([1.3, 1.8])
    lam_planck = planck_feedback_fv3(t_eff)
    lam_lr = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    expected_ratio = (dT_atm - dT_sfc) / dT_sfc
    np.testing.assert_allclose(
        np.asarray(lam_lr),
        np.asarray(lam_planck * expected_ratio),
        rtol=1e-12,
    )


def test_zero_dT_sfc_floored():
    """ΔT_sfc=0 → finite via floor (no NaN)."""
    t_eff = jnp.array([255.0])
    dT_sfc = jnp.array([0.0])
    dT_atm = jnp.array([0.5])
    lam_lr = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    assert jnp.all(jnp.isfinite(lam_lr))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=835)
    n_x, n_y = 6, 8
    t_eff = jnp.asarray(rng.uniform(220.0, 310.0, size=(n_x, n_y)))
    dT_sfc = jnp.asarray(rng.uniform(0.5, 4.0, size=(n_x, n_y)))
    dT_atm = jnp.asarray(rng.uniform(0.2, 5.0, size=(n_x, n_y)))
    lam = lapse_rate_feedback_fv3(t_eff, dT_atm, dT_sfc)
    assert lam.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(lam))
