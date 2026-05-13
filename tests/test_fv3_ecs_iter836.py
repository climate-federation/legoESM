"""FV3_3D iter 836: equilibrium_climate_sensitivity_fv3.

ECS = ΔF / |λ_net|.

Tests
-----

1. ``test_ar6_canonical``: ΔF=3.7, λ=−1.4 → ECS ≈ 2.64 K.
2. ``test_no_feedback_planck_only``: λ=λ_Planck only → ECS ≈ 0.98 K.
3. ``test_ar5ar6_quartet_pre_cloud``: Σ_4 = −2.22 → ECS ≈ 1.67 K.
4. ``test_high_cloud_feedback``: λ_cloud=+1.5 → ECS jumps to ~5 K.
5. ``test_chain_with_full_quartet``: end-to-end primitive chain
   gives consistent ECS.
6. ``test_lam_zero_floored``: λ=0 → finite via floor.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    clausius_clapeyron_dqdt_fv3,
    equilibrium_climate_sensitivity_fv3,
    ice_albedo_feedback_fv3,
    lapse_rate_feedback_fv3,
    planck_feedback_fv3,
)


def test_ar6_canonical():
    """ΔF=3.7, λ=−1.4 → ECS ≈ 2.64 K (AR6 central estimate)."""
    f = jnp.array([3.7])
    lam = jnp.array([-1.4])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    np.testing.assert_allclose(np.asarray(ecs), [3.7 / 1.4], rtol=1e-12)
    assert 2.5 < float(ecs[0]) < 2.8


def test_no_feedback_planck_only():
    """λ = λ_Planck only → ECS ≈ 0.98 K (no-feedback ECS)."""
    f = jnp.array([3.7])
    lam_planck = planck_feedback_fv3(jnp.array([255.0]))
    ecs = equilibrium_climate_sensitivity_fv3(f, lam_planck)
    assert 0.9 < float(ecs[0]) < 1.1


def test_ar5ar6_quartet_pre_cloud():
    """Σ AR5/AR6 quartet = −2.22 → ECS ≈ 1.67 K pre-cloud."""
    lam_quartet = jnp.array([-2.22])
    f = jnp.array([3.7])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam_quartet)
    np.testing.assert_allclose(np.asarray(ecs), [3.7 / 2.22], rtol=1e-12)
    assert 1.6 < float(ecs[0]) < 1.7


def test_high_cloud_feedback():
    """λ_cloud=+1.5 → ECS jumps ~5.1 K (high climate sensitivity)."""
    lam_quartet = -2.22
    lam_cloud = 1.5
    lam_net = jnp.array([lam_quartet + lam_cloud])
    f = jnp.array([3.7])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam_net)
    assert 4.5 < float(ecs[0]) < 5.5


def test_chain_with_full_quartet():
    """Full primitive chain: AR5/AR6 quartet → ECS."""
    t_eff = jnp.array([255.0])
    p = jnp.array([1.0e5])
    t_sfc = jnp.array([288.0])
    s_in = jnp.array([340.25])
    lam_planck = planck_feedback_fv3(t_eff)
    # λ_WV via CC × kernel proxy (1.8/0.6 ratio for sanity)
    dqdt = clausius_clapeyron_dqdt_fv3(t_sfc, p)
    lam_wv = jnp.array([1.8])  # canonical CMIP, set directly
    lam_lr = lapse_rate_feedback_fv3(
        t_eff, jnp.array([1.4]), jnp.array([1.0])
    )
    lam_alpha = ice_albedo_feedback_fv3(s_in, jnp.array([-0.001]))
    lam_net = lam_planck + lam_wv + lam_lr + lam_alpha
    f = jnp.array([3.7])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam_net)
    assert jnp.isfinite(ecs[0])
    assert 0.5 < float(ecs[0]) < 5.0  # plausible Earth band
    # Sanity: dqdt physically positive (used in upstream chain)
    assert float(dqdt[0]) > 0.0


def test_lam_zero_floored():
    """λ=0 → finite via lam_floor (no NaN)."""
    f = jnp.array([3.7])
    lam = jnp.array([0.0])
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    assert jnp.all(jnp.isfinite(ecs))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=836)
    n_x, n_y = 6, 8
    f = jnp.asarray(rng.uniform(1.0, 8.0, size=(n_x, n_y)))
    lam = jnp.asarray(rng.uniform(-3.0, -0.5, size=(n_x, n_y)))
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    assert ecs.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(ecs))
    assert jnp.all(ecs > 0.0)
