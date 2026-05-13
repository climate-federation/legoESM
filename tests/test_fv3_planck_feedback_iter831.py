"""FV3_3D iter 831: planck_feedback_fv3.

λ_Planck = −4 · ε · σ · T_eff³  (W/m²/K).

Tests
-----

1. ``test_earth_planck``: T_eff=255 K, ε=1 → λ ≈ −3.76 W/m²/K.
2. ``test_negative_sign``: Earth-like T → λ < 0 (stabilizing).
3. ``test_monotone_in_t``: ↑T_eff → ↑|λ_Planck|.
4. ``test_lower_emissivity_reduces_magnitude``: ε<1 → smaller |λ|.
5. ``test_chain_with_iter829``: T_eff from OLR → λ_Planck chained.
6. ``test_no_feedback_sensitivity``: 2×CO₂ ΔT estimate ≈ 0.98 K.
7. ``test_shapes_finite``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.grids.cubed_sphere import (
    effective_radiating_temperature_fv3,
    planck_feedback_fv3,
)


def test_earth_planck():
    """T_eff=255, ε=1 → λ ≈ −4·σ·255³ ≈ −3.76 W/m²/K."""
    t_eff = jnp.array([255.0])
    lam = planck_feedback_fv3(t_eff)
    expected = -4.0 * constants.sigma_sb * 255.0 ** 3
    np.testing.assert_allclose(np.asarray(lam), [expected], rtol=1e-12)
    assert -3.8 < float(lam[0]) < -3.7


def test_negative_sign():
    """Earth-like T_eff range → all λ < 0 (stabilizing feedback)."""
    t_eff = jnp.array([200.0, 250.0, 300.0])
    lam = planck_feedback_fv3(t_eff)
    assert jnp.all(lam < 0.0)


def test_monotone_in_t():
    """↑T_eff → ↑|λ_Planck| (T³ scaling)."""
    cold = planck_feedback_fv3(jnp.array([240.0]))
    warm = planck_feedback_fv3(jnp.array([270.0]))
    assert float(jnp.abs(warm[0])) > float(jnp.abs(cold[0]))


def test_lower_emissivity_reduces_magnitude():
    """ε=0.5 → halves |λ_Planck| at fixed T_eff."""
    t_eff = jnp.array([255.0])
    lam_full = planck_feedback_fv3(t_eff, emissivity=1.0)
    lam_half = planck_feedback_fv3(t_eff, emissivity=0.5)
    np.testing.assert_allclose(
        np.asarray(lam_half),
        0.5 * np.asarray(lam_full),
        rtol=1e-12,
    )


def test_chain_with_iter829():
    """Compose iter-829: λ_Planck(T_eff(OLR)) from raw OLR."""
    olr = jnp.array([240.0])  # Earth global mean
    t_eff = effective_radiating_temperature_fv3(olr)
    lam = planck_feedback_fv3(t_eff)
    # OLR/T_eff = ε·σ·T_eff³ → λ = −4·OLR/T_eff
    expected = -4.0 * olr / t_eff
    np.testing.assert_allclose(np.asarray(lam), np.asarray(expected), rtol=1e-12)


def test_no_feedback_sensitivity():
    """No-feedback ΔT for 2×CO₂ forcing (3.7 W/m²) ≈ 0.98 K."""
    t_eff = jnp.array([255.0])
    lam = planck_feedback_fv3(t_eff)
    forcing_2xco2 = 3.7  # W/m²
    delta_t = forcing_2xco2 / float(jnp.abs(lam[0]))
    assert 0.9 < delta_t < 1.1


def test_shapes_finite():
    """3-D shapes preserved, finite, negative."""
    rng = np.random.default_rng(seed=831)
    n_x, n_y = 6, 8
    t_eff = jnp.asarray(rng.uniform(200.0, 320.0, size=(n_x, n_y)))
    eps = jnp.asarray(rng.uniform(0.5, 1.0, size=(n_x, n_y)))
    lam = planck_feedback_fv3(t_eff, eps)
    assert lam.shape == (n_x, n_y)
    assert jnp.all(jnp.isfinite(lam))
    assert jnp.all(lam < 0.0)
