"""Unit tests for the shared Q1 diagnostic heat-flux reduction.

``diagnostic_heat_flux_full`` is the ONE reduction every K-closure exposes as
``TurbulenceOutput.wtheta_flux`` (LES-suite Q1 diagnostic score). These check the
sign convention (down-gradient vs counter-gradient) and the half→full mapping in
isolation — analytic profiles, no scheme machinery.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    diagnostic_heat_flux_full,
)


def test_stable_downgradient_flux_is_negative():
    # TOA-first (index 0 = top): θ warmer aloft ⇒ ∂θ/∂z > 0 (stable). A local
    # closure (γ=0) transports heat DOWN-gradient ⇒ negative (downward) flux.
    theta = jnp.array([[303.0, 302.0, 301.0, 300.0]])
    dz_half = jnp.full((1, 3), 100.0)
    Kh_half = jnp.full((1, 3), 5.0)
    flux = diagnostic_heat_flux_full(theta, dz_half, Kh_half)
    assert flux.shape == theta.shape           # (ncol, nlev)
    assert bool(jnp.all(flux < 0.0))           # down-gradient in a stable column


def test_countergradient_flux_is_upward_at_zero_gradient():
    # Well-mixed layer (∂θ/∂z = 0) with a positive counter-gradient γ: a local
    # closure gives exactly zero flux, but the nonlocal γ drives an UPWARD flux
    # F = -Kh·(0 - γ) = +Kh·γ > 0 — the CBL mixed-layer signature no K≥0 can make.
    theta = jnp.full((1, 4), 300.0)
    dz_half = jnp.full((1, 3), 100.0)
    Kh_half = jnp.full((1, 3), 2.0)
    gamma = jnp.full((1, 3), 0.01)             # [K/m]
    flux_local = diagnostic_heat_flux_full(theta, dz_half, Kh_half)
    flux_nonlocal = diagnostic_heat_flux_full(theta, dz_half, Kh_half, gamma)
    assert bool(jnp.allclose(flux_local, 0.0))     # γ=0 ⇒ no flux at zero gradient
    assert bool(jnp.all(flux_nonlocal > 0.0))      # γ>0 ⇒ upward counter-gradient
    # F = Kh·γ = 2·0.01 = 0.02 K m/s on the interior (edges copy the interface).
    assert bool(jnp.allclose(flux_nonlocal, 0.02))


def test_gamma_reduces_downgradient_flux():
    # A positive γ opposes the down-gradient part: F = -Kh·(∂θ/∂z - γ) is strictly
    # LESS negative (more upward) than the γ=0 flux for the same Kh and gradient.
    theta = jnp.array([[303.0, 302.0, 301.0, 300.0]])   # ∂θ/∂z = +0.01
    dz_half = jnp.full((1, 3), 100.0)
    Kh_half = jnp.full((1, 3), 5.0)
    gamma = jnp.full((1, 3), 0.005)
    flux0 = diagnostic_heat_flux_full(theta, dz_half, Kh_half)
    flux_g = diagnostic_heat_flux_full(theta, dz_half, Kh_half, gamma)
    assert bool(jnp.all(flux_g > flux0))     # counter-gradient lifts the flux upward


if __name__ == "__main__":
    test_stable_downgradient_flux_is_negative()
    test_countergradient_flux_is_upward_at_zero_gradient()
    test_gamma_reduces_downgradient_flux()
    print("ok")
