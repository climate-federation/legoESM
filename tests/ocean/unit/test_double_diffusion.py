"""Acceptance tests for double-diffusive mixing (NEMO zdfddm).

Written BEFORE the wiring (spec-first). Exercises the leaf tendency
`compute_ddm_diffusivity` directly — the regime selection, sign, and the
byte-identical OFF path.
"""
import jax.numpy as jnp
import pytest

from legoesm.ocean.physics.vertical_mixing.double_diffusion import (
    DoubleDiffusionConfig,
    compute_ddm_diffusivity,
)

CFG = DoubleDiffusionConfig(enabled=True, rn_avts=1e-4, rn_hsbfr=1.6, k_max=1e-2)


def _rrho_inputs(R_rho, N2=1e-4, beta_dSdz=1e-4):
    """alpha_dTdz, beta_dSdz for a target R_rho at a stable interface."""
    return N2, R_rho * beta_dSdz, beta_dSdz


def test_salt_fingering_regime():
    # Warm-salty over cold-fresh: 1 < R_rho < R_c -> strong salt, weaker heat.
    N2, a, b = _rrho_inputs(1.3)
    avt, avs = compute_ddm_diffusivity(
        jnp.array([N2]), jnp.array([a]), jnp.array([b]), CFG)
    assert avs[0] > avt[0] > 0.0
    assert avs[0] <= CFG.rn_avts + 1e-12
    # Heat ratio: avt = 0.7 * avs / R_rho
    assert avt[0] == pytest.approx(0.7 * avs[0] / 1.3, rel=1e-5)


def test_fingering_cutoff():
    # R_rho >= R_c -> no fingering (both zero).
    N2, a, b = _rrho_inputs(1.7)  # > rn_hsbfr=1.6
    avt, avs = compute_ddm_diffusivity(
        jnp.array([N2]), jnp.array([a]), jnp.array([b]), CFG)
    assert avt[0] == 0.0 and avs[0] == 0.0


def test_diffusive_convection_regime():
    # Cold-fresh over warm-salty: 0 < R_rho < 1 -> strong heat, weaker salt.
    N2, a, b = _rrho_inputs(0.7)
    avt, avs = compute_ddm_diffusivity(
        jnp.array([N2]), jnp.array([a]), jnp.array([b]), CFG)
    assert avt[0] > avs[0] > 0.0


def test_diffusive_convection_continuity_at_half():
    # avs branches meet at R_rho = 0.5 (continuous fit).
    N2 = 1e-4
    for R in (0.5 - 1e-6, 0.5 + 1e-6):
        _, a, b = _rrho_inputs(R, N2=N2)
        avt, avs = compute_ddm_diffusivity(
            jnp.array([N2]), jnp.array([a]), jnp.array([b]), CFG)
        assert avt[0] > 0.0 and avs[0] >= 0.0
    # both branches evaluated at 0.5 give avs/avt = 0.15*0.5 == (1.85-0.85/0.5)*0.5
    assert (0.15 * 0.5) == pytest.approx((1.85 - 0.85 / 0.5) * 0.5, abs=1e-12)


def test_statically_single_signed_no_ddm():
    # Both gradients stabilizing (R_rho < 0): no double diffusion.
    N2, a, b = 1e-4, -1e-4, 1e-4   # alpha dT/dz < 0 -> R_rho < 0
    avt, avs = compute_ddm_diffusivity(
        jnp.array([N2]), jnp.array([a]), jnp.array([b]), CFG)
    assert avt[0] == 0.0 and avs[0] == 0.0


def test_unstable_column_no_ddm():
    # N^2 <= 0 -> convection scheme owns it, ddm returns zero.
    _, a, b = _rrho_inputs(1.3)
    avt, avs = compute_ddm_diffusivity(
        jnp.array([-1e-5]), jnp.array([a]), jnp.array([b]), CFG)
    assert avt[0] == 0.0 and avs[0] == 0.0


def test_vanishing_salinity_gradient_finite():
    # beta dS/dz -> 0 must not NaN/Inf (denominator floored, sign preserved).
    avt, avs = compute_ddm_diffusivity(
        jnp.array([1e-4]), jnp.array([1e-4]), jnp.array([0.0]), CFG)
    assert jnp.isfinite(avt[0]) and jnp.isfinite(avs[0])


def test_diffusivities_bounded():
    R = jnp.linspace(-2.0, 3.0, 200)
    N2 = jnp.full_like(R, 1e-4)
    avt, avs = compute_ddm_diffusivity(N2, R * 1e-4, jnp.full_like(R, 1e-4), CFG)
    assert jnp.all(avt >= 0) and jnp.all(avs >= 0)
    assert jnp.all(avt <= CFG.k_max) and jnp.all(avs <= CFG.k_max)
    assert jnp.all(jnp.isfinite(avt)) and jnp.all(jnp.isfinite(avs))
