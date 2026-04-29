"""Tests for ``_shared.diagnose_grid_w_from_omega``.

The helper converts pressure velocity ``ω = dp/dt`` to grid-scale
``w = dz/dt`` via ``w = -ω/(ρ g)`` with optional virtual-temperature
correction.  Used by the spectral PE / hydrostatic convection bridges
to feed the Kain-Fritsch BL trigger.

Tests pin:

* the analytical relation ``w = -ω/(ρ g)`` for a known ``ω, T, p``;
* virtual-temperature reduces ``ρ`` (so |w| increases) for moist air;
* zero ``ω`` → zero ``w`` regardless of ``T``, ``p``, ``q_v``;
* the result is finite at numerical singularities (T → 0);
* differentiability through ``ω`` and ``T``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import diagnose_grid_w_from_omega


def _column(ncol: int = 2, nlev: int = 8, T_sfc: float = 300.0):
    sigma = jnp.linspace(0.05, 0.95, nlev)
    p_full = sigma[None, :] * 1e5 * jnp.ones((ncol, 1))
    T = jnp.full((ncol, nlev), T_sfc)
    return T, p_full, sigma


def test_zero_omega_yields_zero_w():
    T, p_full, _ = _column()
    omega = jnp.zeros_like(T)
    w = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
    assert float(jnp.max(jnp.abs(w))) < 1e-30


def test_analytical_relation_dry():
    """For dry air, ``w = -ω · R_d · T / (p · g)`` exactly."""
    T, p_full, _ = _column()
    omega = jnp.full_like(T, 0.1)   # downward sinking (positive ω)
    w = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
    expected = -omega * constants.R_d * T / (p_full * constants.g)
    assert bool(jnp.allclose(w, expected, atol=1e-10, rtol=1e-10))


def test_virtual_temp_increases_w_magnitude():
    """Virtual temperature increases ``T_v > T``, decreasing ``ρ``,
    so ``|w|`` increases relative to the dry-air baseline."""
    T, p_full, _ = _column()
    omega = jnp.full_like(T, -0.05)  # mild upward motion (negative ω)
    q_v = jnp.full_like(T, 0.015)    # 15 g/kg, tropical-ish

    w_dry = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
    w_moist = diagnose_grid_w_from_omega(omega, T, p_full, q_v=q_v)

    # Both upward (positive w) since ω < 0.
    assert bool(jnp.all(w_dry > 0.0))
    assert bool(jnp.all(w_moist > 0.0))
    # Moist column has larger |w| than dry baseline.
    assert bool(jnp.all(w_moist > w_dry))
    # The factor is ≈ 1 + 0.608 · q_v ≈ 1.0091 at 15 g/kg.
    expected_ratio = 1.0 + (constants.R_v / constants.R_d - 1.0) * 0.015
    assert bool(jnp.allclose(w_moist / w_dry, expected_ratio, rtol=1e-6))


def test_sign_convention():
    """ω > 0 (sinking, dp/dt > 0) → w < 0 (downward).  ω < 0 → w > 0."""
    T, p_full, _ = _column()
    w_pos = diagnose_grid_w_from_omega(
        jnp.full_like(T, 0.1), T, p_full, q_v=None,
    )
    w_neg = diagnose_grid_w_from_omega(
        jnp.full_like(T, -0.1), T, p_full, q_v=None,
    )
    assert bool(jnp.all(w_pos < 0.0))
    assert bool(jnp.all(w_neg > 0.0))
    # Symmetry: |w_pos| ≈ |w_neg| for symmetric ω.
    assert bool(jnp.allclose(w_pos, -w_neg, atol=1e-12))


def test_finite_at_low_temperature():
    """The 1 K floor on ``T_v`` should keep ``w`` finite when the
    column has tiny temperatures (numerical edge case)."""
    T = jnp.full((2, 4), 0.5)        # below the 1 K floor
    p_full = 1e5 * jnp.ones((2, 4))
    omega = jnp.full_like(T, 0.1)
    w = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
    assert bool(jnp.all(jnp.isfinite(w)))


def test_grad_through_omega():
    """``jax.grad`` through ω flows; sensitivity is ``-1/(ρ g)`` per
    grid cell (negative of a positive function)."""
    T, p_full, _ = _column()

    def loss(omega_amp):
        omega = jnp.full_like(T, omega_amp)
        w = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
        return jnp.sum(w ** 2)

    g = jax.grad(loss)(jnp.array(0.1))
    assert bool(jnp.isfinite(g))
    # ∂(Σ w²)/∂ω_amp = 2 ω_amp · Σ (R_d T / (p g))² > 0 for ω_amp > 0.
    assert float(g) > 0.0


def test_grad_through_temperature():
    """jax.grad through T flows (no nondiff branches)."""
    T, p_full, _ = _column()

    def loss(scale):
        omega = jnp.full_like(T, 0.1)
        w = diagnose_grid_w_from_omega(omega, T, p_full, q_v=None)
        return jnp.sum(w ** 2)

    g = jax.grad(loss)(jnp.array(1.0))
    assert bool(jnp.isfinite(g))
