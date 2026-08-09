"""Wing 2018 RCEMIP1 initial-condition profile tests.

Pins the analytical profiles against published constants + the
hydrostatic-balance + Poisson identities, then validates the
``theta_ref_fn`` closure integrates cleanly into the stretched
HeightCoordinate factory.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    WING_GAMMA, WING_P_SFC, WING_Q_SFC_DEFAULT, WING_Q_T,
    WING_T_V0, WING_Z_T, WING_Z_Q1, WING_Z_Q2, wing2018_T_v0,
    _VIRTUAL_T_FACTOR,
    make_wing2018_pressure_ref_fn, make_wing2018_qv_ref_fn,
    make_wing2018_temperature_ref_fn, make_wing2018_theta_ref_fn,
    wing2018_pressure_profile, wing2018_qv_profile,
    wing2018_temperature_profile, wing2018_theta_profile,
    wing2018_virtual_temperature_profile,
)
from legoesm.grids.vertical import create_stretched_height_coordinate

jax.config.update("jax_enable_x64", True)


def test_virtual_T_at_surface_equals_T_v0():
    """T_v(z=0) = T_v0 = T0·(1 + 0.608·q0) (Wing 2018 Eq. 3, T0 = case SST)."""
    T_v0 = float(wing2018_virtual_temperature_profile(jnp.asarray(0.0)))
    np.testing.assert_allclose(T_v0, WING_T_V0, rtol=1.0e-12)
    np.testing.assert_allclose(T_v0, wing2018_T_v0(300.0, WING_Q_SFC_DEFAULT),
                               rtol=1.0e-12)


def test_ic_is_near_80_percent_RH_and_not_supersaturated():
    """Wing 2018: "q0 ... adjusted so that the relative humidity is near 80 %
    in the lower atmosphere".

    This is the check that catches a mis-specified T_v0: pinning it at a
    fixed 295 K (instead of T0·(1+0.608·q0) = 303.4 K at SST 300) cools the
    initial column ~8 K, which pushes near-surface RH to 139 % — the model
    then starts supersaturated and condenses the excess in the first steps.
    """
    from legoesm.thermo import relative_humidity
    z = jnp.linspace(0.0, 2_000.0, 21)
    T = wing2018_temperature_profile(z)
    p = wing2018_pressure_profile(z)
    q_v = wing2018_qv_profile(z)
    rh = np.asarray(relative_humidity(T, p, q_v))
    assert rh.max() < 1.0, f"IC is supersaturated: max RH = {rh.max():.3f}"
    assert 0.75 < rh.min() and rh.max() < 0.90, (
        f"IC lower-atmosphere RH not near 80 %: {rh.min():.3f}-{rh.max():.3f}"
    )


def test_actual_T_at_surface_is_devirtualized_T_v0():
    """Actual (dry-bulb) T(z=0) = T_v0 / (1 + ε⁻¹·q_sfc) = T0 = 300 K for
    RCE300 — the analytic IC's surface air temperature equals the SST."""
    T0 = float(wing2018_temperature_profile(jnp.asarray(0.0)))
    expected = WING_T_V0 / (1.0 + _VIRTUAL_T_FACTOR * WING_Q_SFC_DEFAULT)
    np.testing.assert_allclose(T0, expected, rtol=1.0e-12)


def test_T_tropopause_cap_above_z_t():
    """T_v(z > z_t) = T_v(z_t) = WING_T_V0 - Γ·z_t (isothermal virtual cap ≈
    194.5 K). Actual T above the tropopause ≈ T_v (q_t = 10⁻¹¹)."""
    T_v_top = float(wing2018_virtual_temperature_profile(
        jnp.asarray(20_000.0),
    ))
    T_v_cap_exact = WING_T_V0 - WING_GAMMA * WING_Z_T
    np.testing.assert_allclose(T_v_top, T_v_cap_exact, rtol=1.0e-12)
    # Actual T at the same height differs by < 1e-10 K from T_v (q_t
    # is essentially zero).
    T_top = float(wing2018_temperature_profile(jnp.asarray(20_000.0)))
    np.testing.assert_allclose(T_top, T_v_top, rtol=1.0e-10)


def test_virtual_T_factor_matches_constants_epsilon():
    """Codex iter-1 fix: the virtual-T factor MUST be derived from
    constants.epsilon (R_d/R_v) so the value stays in sync with the
    rest of the codebase."""
    expected = 1.0 / constants.epsilon - 1.0
    np.testing.assert_allclose(_VIRTUAL_T_FACTOR, expected, rtol=1.0e-12)
    # Should be approximately 0.608 (the canonical value).
    assert 0.6 < _VIRTUAL_T_FACTOR < 0.62


def test_qv_at_surface_equals_q_sfc():
    """q_v(0) = q_sfc."""
    q0 = float(wing2018_qv_profile(jnp.asarray(0.0)))
    np.testing.assert_allclose(q0, WING_Q_SFC_DEFAULT, rtol=1.0e-12)


def test_qv_stratosphere_is_q_t():
    """q_v(z > z_t) = q_t (10⁻¹¹)."""
    q_top = float(wing2018_qv_profile(jnp.asarray(20_000.0)))
    np.testing.assert_allclose(q_top, WING_Q_T, rtol=1.0e-12)


def test_qv_decay_through_troposphere():
    """q_v should be monotonically decreasing from surface to tropopause
    (the product of two decaying exponentials is monotone)."""
    z = jnp.linspace(0.0, WING_Z_T - 100.0, 50)
    q = np.asarray(wing2018_qv_profile(z))
    assert np.all(np.diff(q) < 0.0)


def test_pressure_at_surface_equals_p_sfc():
    p0 = float(wing2018_pressure_profile(jnp.asarray(0.0)))
    np.testing.assert_allclose(p0, WING_P_SFC, rtol=1.0e-12)


def test_pressure_decreases_with_height():
    """p(z) strictly decreasing for any z > 0."""
    z = jnp.linspace(0.0, 30_000.0, 50)
    p = np.asarray(wing2018_pressure_profile(z))
    assert np.all(np.diff(p) < 0.0)


def test_pressure_continuous_at_tropopause():
    """p must be continuous across z_t (the troposphere + stratosphere
    branches use the same constant at z_t)."""
    z_below = jnp.asarray(WING_Z_T - 1.0e-6)
    z_above = jnp.asarray(WING_Z_T + 1.0e-6)
    p_below = float(wing2018_pressure_profile(z_below))
    p_above = float(wing2018_pressure_profile(z_above))
    np.testing.assert_allclose(p_below, p_above, rtol=1.0e-8)


def test_theta_at_surface_is_poisson_of_actual_T():
    """At z=0: θ = T_actual · (p_ref/p_sfc)^κ. T_actual at the
    surface is T_sfc (Codex iter-1: NOT T_v0). The Wing default
    p_sfc = 101480 Pa is slightly above p_ref = 100000 Pa so
    θ < T_sfc by the Exner correction (~0.4%)."""
    theta0 = float(wing2018_theta_profile(jnp.asarray(0.0)))
    T0 = WING_T_V0 / (1.0 + _VIRTUAL_T_FACTOR * WING_Q_SFC_DEFAULT)
    expected = T0 * (constants.p_ref / WING_P_SFC) ** constants.kappa
    np.testing.assert_allclose(theta0, expected, rtol=1.0e-12)


def test_theta_increases_with_height_in_troposphere():
    """Stable stratification (θ rising with height) is the canonical
    RCE tropospheric profile."""
    z = jnp.linspace(0.0, WING_Z_T - 100.0, 50)
    theta = np.asarray(wing2018_theta_profile(z))
    assert np.all(np.diff(theta) > 0.0), (
        "Wing 2018 troposphere is stably stratified; θ must rise with z"
    )


def test_wing_theta_ref_fn_integrates_with_stretched_grid():
    """The closure factory plugs straight into the stretched
    HeightCoordinate factory — pin that the resulting reference
    state is finite + physical."""
    theta_fn = make_wing2018_theta_ref_fn()
    hc = create_stretched_height_coordinate(
        n_levels=40, H=33_000.0, dz_sfc=50.0,
        theta_ref_fn=theta_fn,
    )
    assert bool(jnp.all(jnp.isfinite(hc.theta_ref)))
    assert bool(jnp.all(jnp.isfinite(hc.rho_ref)))
    assert bool(jnp.all(jnp.isfinite(hc.exner_ref)))
    # rho_ref should still decrease with height (storage top→surface →
    # rho INCREASES with index).
    rho = np.asarray(hc.rho_ref)
    assert rho[-1] > rho[0]
    # θ at surface ≈ T_actual(0)·(p_ref/p_sfc)^κ. Wing Eq. (3)
    # T_v0 = T0·(1+0.608·q0) = 303.4 K ⇒ T_actual(0) = T0 = 300 K,
    # p_ref/p_sfc ≈ 0.985 → θ ≈ 299 K at the lowest level.
    theta_sfc = float(hc.theta_ref[-1])
    assert 296.0 < theta_sfc < 301.0, (
        f"Wing IC surface θ unrealistic: {theta_sfc:.2f}"
    )


def test_virtual_T_lapse_rate_matches_Gamma():
    """∂T_v/∂z = -Γ in the troposphere — verify by finite difference."""
    z1, z2 = 100.0, 2_100.0
    T_v1 = float(wing2018_virtual_temperature_profile(jnp.asarray(z1)))
    T_v2 = float(wing2018_virtual_temperature_profile(jnp.asarray(z2)))
    dT_dz = (T_v2 - T_v1) / (z2 - z1)
    np.testing.assert_allclose(dT_dz, -WING_GAMMA, rtol=1.0e-12)


def test_qv_closure_factory_matches_direct():
    """``make_wing2018_qv_ref_fn`` produces the same q_v(z) as the
    direct function call (closure factory regression)."""
    z = jnp.linspace(0.0, 30_000.0, 20)
    q_v_direct = wing2018_qv_profile(z)
    q_v_closure = make_wing2018_qv_ref_fn()(z)
    np.testing.assert_allclose(q_v_closure, q_v_direct, rtol=0.0, atol=0.0)


def test_pressure_closure_factory_matches_direct():
    z = jnp.linspace(0.0, 30_000.0, 20)
    p_direct = wing2018_pressure_profile(z)
    p_closure = make_wing2018_pressure_ref_fn()(z)
    np.testing.assert_allclose(p_closure, p_direct, rtol=0.0, atol=0.0)


def test_temperature_closure_factory_matches_direct():
    z = jnp.linspace(0.0, 30_000.0, 20)
    T_direct = wing2018_temperature_profile(z)
    T_closure = make_wing2018_temperature_ref_fn()(z)
    np.testing.assert_allclose(T_closure, T_direct, rtol=0.0, atol=0.0)


def test_wing_profiles_are_jax_differentiable():
    """jax.grad through every profile returns a finite gradient."""
    z = 5_000.0
    for fn in (wing2018_temperature_profile, wing2018_qv_profile,
               wing2018_pressure_profile, wing2018_theta_profile):
        g = float(jax.grad(lambda z: jnp.sum(fn(jnp.asarray(z))))(z))
        assert np.isfinite(g)
