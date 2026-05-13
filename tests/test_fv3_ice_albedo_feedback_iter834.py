"""FV3_3D iter 834: ice_albedo_feedback_fv3.

λ_α = − S_in · dα/dT  (W/m²/K).

Tests
-----

1. ``test_global_mean``: S=340.25, dα/dT=−0.001 → λ_α ≈ +0.34.
2. ``test_positive_for_melt``: dα/dT<0 → λ_α>0 (destabilizing).
3. ``test_zero_no_change``: dα/dT=0 → λ_α=0.
4. ``test_negative_for_cooling_brightening``: dα/dT>0 → λ_α<0.
5. ``test_snowball_runaway_exceeds_planck``: dα/dT=−0.05 →
   |λ_α| > |λ_Planck| at T_eff=255.
6. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    ice_albedo_feedback_fv3,
    planck_feedback_fv3,
)


def test_global_mean():
    """S=340.25, dα/dT=−0.001 → λ_α ≈ +0.34 W/m²/K."""
    s = jnp.array([340.25])
    dadt = jnp.array([-0.001])
    lam = ice_albedo_feedback_fv3(s, dadt)
    np.testing.assert_allclose(np.asarray(lam), [0.34025], rtol=1e-12)


def test_positive_for_melt():
    """dα/dT < 0 (ice melt darkens) → λ_α > 0 (positive feedback)."""
    s = jnp.array([340.25])
    dadt = jnp.array([-0.01, -0.005, -0.002])
    s_b = jnp.broadcast_to(s, dadt.shape)
    lam = ice_albedo_feedback_fv3(s_b, dadt)
    assert jnp.all(lam > 0.0)


def test_zero_no_change():
    """dα/dT=0 → λ_α=0 (no feedback)."""
    s = jnp.array([340.25])
    dadt = jnp.array([0.0])
    lam = ice_albedo_feedback_fv3(s, dadt)
    np.testing.assert_allclose(np.asarray(lam), [0.0], atol=1e-14)


def test_negative_for_cooling_brightening():
    """dα/dT > 0 → λ_α < 0 (stabilizing, brightening with T)."""
    s = jnp.array([340.25])
    dadt = jnp.array([0.001])
    lam = ice_albedo_feedback_fv3(s, dadt)
    assert float(lam[0]) < 0.0


def test_snowball_runaway_exceeds_planck():
    """dα/dT=−0.05 (snowball regime) → λ_α > |λ_Planck| at T=255."""
    s = jnp.array([340.25])
    dadt = jnp.array([-0.05])
    lam_alpha = ice_albedo_feedback_fv3(s, dadt)
    lam_planck = planck_feedback_fv3(jnp.array([255.0]))
    assert float(lam_alpha[0]) > float(jnp.abs(lam_planck[0]))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=834)
    n_x, n_y = 6, 8
    s = jnp.asarray(rng.uniform(100.0, 400.0, size=(n_x, n_y)))
    dadt = jnp.asarray(rng.uniform(-0.05, 0.05, size=(n_x, n_y)))
    lam = ice_albedo_feedback_fv3(s, dadt)
    assert lam.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(lam))
