"""FV3_3D iter 837: transient_climate_response_fv3.

TCR = ΔF / (|λ_net| + γ).

Tests
-----

1. ``test_ar6_canonical``: ΔF=3.7, |λ|=1.4, γ=0.7 → TCR ≈ 1.76 K.
2. ``test_tcr_le_ecs``: TCR < ECS whenever γ > 0.
3. ``test_no_ocean_uptake_equals_ecs``: γ=0 → TCR = ECS.
4. ``test_high_gamma_suppresses_warming``: ↑γ → ↓TCR.
5. ``test_tcr_ecs_ratio``: ratio = |λ|/(|λ|+γ) exactly.
6. ``test_denom_zero_floored``: |λ|+γ=0 → finite (no NaN).
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    equilibrium_climate_sensitivity_fv3,
    transient_climate_response_fv3,
)


def test_ar6_canonical():
    """ΔF=3.7, |λ|=1.4, γ=0.7 → TCR ≈ 1.76 K (AR6 central)."""
    f = jnp.array([3.7])
    lam = jnp.array([-1.4])
    gamma = jnp.array([0.7])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    np.testing.assert_allclose(np.asarray(tcr), [3.7 / 2.1], rtol=1e-12)
    assert 1.7 < float(tcr[0]) < 1.8


def test_tcr_le_ecs():
    """TCR < ECS for γ > 0 (ocean uptake delays equilibration)."""
    f = jnp.array([3.7])
    lam = jnp.array([-1.4])
    gamma = jnp.array([0.7])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    assert float(tcr[0]) < float(ecs[0])


def test_no_ocean_uptake_equals_ecs():
    """γ=0 → TCR = ECS exactly."""
    f = jnp.array([3.7])
    lam = jnp.array([-1.4])
    gamma = jnp.array([0.0])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    np.testing.assert_allclose(np.asarray(tcr), np.asarray(ecs), rtol=1e-12)


def test_high_gamma_suppresses_warming():
    """↑γ → ↓TCR (more ocean uptake = less surface warming)."""
    f = jnp.array([3.7])
    lam = jnp.array([-1.4])
    tcr_low = transient_climate_response_fv3(f, lam, jnp.array([0.3]))
    tcr_high = transient_climate_response_fv3(f, lam, jnp.array([1.2]))
    assert float(tcr_high[0]) < float(tcr_low[0])


def test_tcr_ecs_ratio():
    """TCR/ECS = |λ|/(|λ|+γ) — realized-warming-fraction identity."""
    f = jnp.array([3.7])
    lam_abs = 1.4
    gamma_val = 0.7
    lam = jnp.array([-lam_abs])
    gamma = jnp.array([gamma_val])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    ecs = equilibrium_climate_sensitivity_fv3(f, lam)
    ratio_actual = float(tcr[0] / ecs[0])
    ratio_expected = lam_abs / (lam_abs + gamma_val)
    np.testing.assert_allclose(ratio_actual, ratio_expected, rtol=1e-12)


def test_denom_zero_floored():
    """|λ|+γ=0 → finite via denom_floor (no NaN)."""
    f = jnp.array([3.7])
    lam = jnp.array([0.0])
    gamma = jnp.array([0.0])
    tcr = transient_climate_response_fv3(f, lam, gamma)
    assert jnp.all(jnp.isfinite(tcr))


def test_shapes_finite():
    """3-D shapes preserved, finite, positive."""
    rng = np.random.default_rng(seed=837)
    n_x, n_y = 6, 8
    f = jnp.asarray(rng.uniform(1.0, 8.0, size=(n_x, n_y)))
    lam = jnp.asarray(rng.uniform(-3.0, -0.5, size=(n_x, n_y)))
    gamma = jnp.asarray(rng.uniform(0.1, 1.5, size=(n_x, n_y)))
    tcr = transient_climate_response_fv3(f, lam, gamma)
    assert tcr.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(tcr))
    assert jnp.all(tcr > 0.0)
