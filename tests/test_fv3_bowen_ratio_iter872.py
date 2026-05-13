"""FV3_3D iter 872: bowen_ratio_fv3.

β = H / λE  (dimensionless).

Tests
-----

1. ``test_tropical_forest_low``: H=50, λE=400 → β ≈ 0.125.
2. ``test_desert_high``: H=200, λE=20 → β = 10.
3. ``test_ocean_low``: H=20, λE=150 → β ≈ 0.13.
4. ``test_zero_h_zero``: H=0 → β=0.
5. ``test_le_zero_floored``: λE=0 → finite (no NaN).
6. ``test_negative_inversion``: H<0 (downward) → β<0.
7. ``test_partition_identity``: λE = A/(1+β), H = β·A/(1+β).
8. ``test_chain_with_iter870``: λE from PM → β.
9. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm.grids.cubed_sphere import (
    bowen_ratio_fv3,
    penman_monteith_le_fv3,
)


def test_tropical_forest_low():
    """H=50, λE=400 (humid forest) → β = 50/400 = 0.125."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([50.0]),
        le_latent=jnp.array([400.0]),
    )
    np.testing.assert_allclose(np.asarray(beta), [0.125], rtol=1e-12)


def test_desert_high():
    """H=200, λE=20 (arid desert) → β = 10."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([200.0]),
        le_latent=jnp.array([20.0]),
    )
    np.testing.assert_allclose(np.asarray(beta), [10.0], rtol=1e-12)


def test_ocean_low():
    """H=20, λE=150 (ocean) → β ≈ 0.13."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([20.0]),
        le_latent=jnp.array([150.0]),
    )
    expected = 20.0 / 150.0
    np.testing.assert_allclose(np.asarray(beta), [expected], rtol=1e-12)


def test_zero_h_zero():
    """H=0 → β=0."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([0.0]),
        le_latent=jnp.array([400.0]),
    )
    np.testing.assert_allclose(np.asarray(beta), [0.0], atol=1e-14)


def test_le_zero_floored():
    """λE=0 → finite via floor (no NaN)."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([100.0]),
        le_latent=jnp.array([0.0]),
    )
    assert jnp.all(jnp.isfinite(beta))


def test_negative_inversion():
    """H<0 (downward) → β<0 (inversion / dew)."""
    beta = bowen_ratio_fv3(
        h_sensible=jnp.array([-30.0]),
        le_latent=jnp.array([200.0]),
    )
    assert float(beta[0]) < 0.0


def test_partition_identity():
    """A = H + λE = (1+β)·λE; verify partition closure."""
    h = 50.0
    le = 400.0
    a = h + le
    beta = float(bowen_ratio_fv3(jnp.array([h]), jnp.array([le]))[0])
    # λE = A / (1 + β)
    le_reconstructed = a / (1.0 + beta)
    np.testing.assert_allclose(le_reconstructed, le, rtol=1e-12)
    # H = β · A / (1 + β)
    h_reconstructed = beta * a / (1.0 + beta)
    np.testing.assert_allclose(h_reconstructed, h, rtol=1e-12)


def test_chain_with_iter870():
    """λE from iter-870 PM → β = H / λE."""
    le = penman_monteith_le_fv3(
        available_energy=jnp.array([400.0]),
        vpd_pa=jnp.array([1500.0]),
        delta_pa_k=jnp.array([180.0]),
        gamma_pa_k=jnp.array([67.0]),
        g_s=jnp.array([0.02]),
        g_a=jnp.array([0.05]),
    )
    h_sens = jnp.array([100.0])
    beta = bowen_ratio_fv3(h_sens, le)
    assert jnp.all(jnp.isfinite(beta))


def test_shapes_finite():
    """3-D shapes preserved, finite."""
    rng = np.random.default_rng(seed=872)
    n_x, n_y = 6, 8
    h = jnp.asarray(rng.uniform(-50.0, 300.0, size=(n_x, n_y)))
    le = jnp.asarray(rng.uniform(5.0, 500.0, size=(n_x, n_y)))
    beta = bowen_ratio_fv3(h, le)
    assert beta.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(beta))
