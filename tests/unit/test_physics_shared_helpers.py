"""Direct tests for the shared physics-formula helpers in
:mod:`legoesm.atmosphere.physics._shared` — the canonical homes the
``test_no_formula_reimpl`` ratchet enforces (so a re-derived ``(p/p₀)^κ`` /
``g/θ`` must call these instead).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from legoesm.atmosphere.physics._shared import (
    brunt_vaisala_n_squared_from_gradient,
    buoyancy_coefficient,
    exner_function,
    exner_to_pressure,
)

from legoesm import constants


def test_exner_to_pressure_is_inverse_of_exner_function():
    """``exner_to_pressure`` (iter 94) is the exact inverse of ``exner_function``
    (``p = p_ref·Π^{1/κ}`` ↔ ``Π = (p/p_ref)^κ``) — a clean round trip."""
    p = jnp.array([1.0e5, 8.5e4, 5.0e4, 2.0e4])
    np.testing.assert_allclose(
        np.asarray(exner_to_pressure(exner_function(p))), np.asarray(p), rtol=1e-12)
    # Analytic anchors: Π=1 ⇒ p_ref; the explicit p_ref·Π^{1/κ} form.
    assert float(exner_to_pressure(jnp.asarray(1.0))) == \
        float(jnp.asarray(constants.p_ref))
    pi = jnp.array([0.9, 0.7])
    np.testing.assert_allclose(
        np.asarray(exner_to_pressure(pi)),
        constants.p_ref * np.asarray(pi) ** (1.0 / constants.kappa), rtol=1e-12)


def test_buoyancy_coefficient_is_g_over_theta():
    """``buoyancy_coefficient(θ) = g/θ`` (the canonical N²/buoyancy factor)."""
    theta = jnp.array([280.0, 300.0, 320.0])
    np.testing.assert_allclose(
        np.asarray(buoyancy_coefficient(theta)),
        constants.g / np.asarray(theta), rtol=1e-12)


def test_brunt_vaisala_n_squared_from_gradient_bit_identical():
    """``brunt_vaisala_n_squared_from_gradient(θ, ∂θ/∂z)`` (iter 94) preserves the
    ``g·∂θ/∂z/θ`` arithmetic ORDER, so it is BIT-IDENTICAL to the inline form a
    caller replaces (Codex: ``(g/θ)·∂θ/∂z`` would reassociate by 1 ULP)."""
    theta = jnp.array([300.5, 301.0, 305.0])
    dtheta_dz = jnp.array([1.0e-3, 3.0e-3, 8.0e-3])
    np.testing.assert_array_equal(            # EXACT, not just close
        np.asarray(brunt_vaisala_n_squared_from_gradient(theta, dtheta_dz)),
        np.asarray(constants.g * dtheta_dz / theta))


def test_exner_helpers_differentiable():
    """Both Exner helpers are smooth + AD-safe (used in JIT'd diagnosis paths)."""
    g = jax.grad(lambda p: jnp.sum(exner_function(p)))(jnp.array([9.0e4, 1.0e5]))
    assert bool(jnp.all(jnp.isfinite(g)))
    g2 = jax.grad(lambda x: jnp.sum(exner_to_pressure(x)))(jnp.array([0.8, 0.95]))
    assert bool(jnp.all(jnp.isfinite(g2)))
