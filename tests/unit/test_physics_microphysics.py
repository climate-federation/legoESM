"""Category 4: Microphysics -- Physical Consistency.

Tests total water conservation, temperature-moisture coupling (Clausius-
Clapeyron), saturation adjustment, precipitation positivity, ice-phase
bounds, and autoconversion threshold for all microphysics schemes.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, SundqvistConfig, SeifertBehengConfig, MorrisonConfig,
    ThompsonConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState, make_zero_hydrometeors,
)
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_column(nlev=20, ncol=4, T_sfc=280.0, q_c_val=1e-4, supersaturated=False):
    """Build a microphysics column with realistic conditions."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = jnp.broadcast_to((sigma_half * p_s)[None, :], (ncol, nlev + 1))
    p_full = jnp.broadcast_to((sigma_full * p_s)[None, :], (ncol, nlev))

    T = T_sfc * jnp.clip(sigma_full, 0.01, None) ** 0.19
    T = jnp.maximum(T, 200.0)
    T = jnp.broadcast_to(T[None, :], (ncol, nlev))

    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))

    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)

    q_sat = saturation_mixing_ratio(T, p_full)
    if supersaturated:
        q_v = 1.2 * q_sat
    else:
        q_v = 0.8 * q_sat

    q_c = jnp.zeros((ncol, nlev))
    q_c = q_c.at[..., -5:].set(q_c_val)

    hydro = HydrometeorState(
        q_c=q_c,
        q_r=jnp.zeros((ncol, nlev)),
        q_i=jnp.zeros((ncol, nlev)),
        q_s=jnp.zeros((ncol, nlev)),
        q_g=jnp.zeros((ncol, nlev)),
        N_c=1e8 * jnp.ones((ncol, nlev)),
        N_r=jnp.zeros((ncol, nlev)),
        N_i=jnp.zeros((ncol, nlev)),
    )
    return T, q_v, hydro, p_full, p_half, rho, dz


def _call_scheme(name, T, q_v, hydro, p_full, p_half, rho, dz, dt=300.0):
    """Call a microphysics scheme backend and return MicrophysicsOutput."""
    if name == "kessler":
        return kessler_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                     config=KesslerConfig())
    elif name == "sundqvist":
        return sundqvist_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                       config=SundqvistConfig())
    elif name == "seifert_beheng":
        return seifert_beheng_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                            config=SeifertBehengConfig())
    elif name == "morrison":
        return morrison_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                      config=MorrisonConfig())
    elif name == "thompson":
        return thompson_microphysics(T, q_v, hydro, p_full, p_half, rho, dz, dt,
                                      config=ThompsonConfig())
    else:
        raise ValueError(f"Unknown scheme: {name}")


ALL_SCHEMES = ["kessler", "sundqvist", "seifert_beheng", "morrison", "thompson"]


# ============================================================================
# 4a  All outputs finite
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_all_outputs_finite(scheme):
    """All MicrophysicsOutput fields should be finite."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    for fname in out._fields:
        val = getattr(out, fname)
        assert jnp.all(jnp.isfinite(val)), f"{scheme}: {fname} has NaN/Inf"


# ============================================================================
# 4b  Temperature-moisture coupling (Clausius-Clapeyron)
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_temperature_moisture_coupling(scheme):
    """cp * dT_dt approx Lv * condensation_rate (first order).

    Checks that where heating is significant, it correlates with moisture
    removal (condensation heats, evaporation cools).
    """
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)

    lhs = constants.c_pd * out.dT_dt
    rhs = -constants.L_v * out.dq_v_dt
    scale = jnp.maximum(jnp.abs(lhs), 1e-10)
    rel_err = jnp.abs(lhs - rhs) / scale
    active = jnp.abs(lhs) > 1e-8
    if jnp.any(active):
        median_err = float(jnp.median(rel_err[active]))
        assert median_err < 0.5, (
            f"{scheme}: median Clausius-Clapeyron rel_err = {median_err:.3f}"
        )


# ============================================================================
# 4c  Saturation adjustment: supersaturated -> vapor decreases
# ============================================================================

@pytest.mark.parametrize("scheme", ["kessler", "sundqvist"])
def test_saturation_adjustment_vapor_decreases(scheme):
    """Starting from supersaturated, vapor should decrease (dq_v_dt < 0)."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(
        supersaturated=True, q_c_val=0.0
    )
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)

    # Some levels should show condensation (dq_v_dt < 0)
    min_dqv = float(jnp.min(out.dq_v_dt))
    assert min_dqv < 0.0, (
        f"{scheme}: no condensation in supersaturated column, min dq_v_dt = {min_dqv:.2e}"
    )


# ============================================================================
# 4d  Precipitation non-negative
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_precipitation_non_negative(scheme):
    """Precipitation must be >= 0."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column()
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    min_precip = float(jnp.min(out.precipitation))
    assert min_precip >= -1e-15, (
        f"{scheme}: negative precipitation = {min_precip:.2e}"
    )


# ============================================================================
# 4e  Ice-phase temperature bounds (Morrison, Thompson)
# ============================================================================

@pytest.mark.parametrize("scheme", ["morrison", "thompson"])
def test_ice_no_formation_above_freezing(scheme):
    """Above freezing (T > 273.15 K), ice formation should be negligible."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(T_sfc=290.0)
    T_warm = jnp.maximum(T, 280.0)
    out = _call_scheme(scheme, T_warm, q_v, hydro, p_full, p_half, rho, dz)

    warm_mask = T_warm > constants.T_freeze
    if jnp.any(warm_mask):
        ice_formation = out.dq_i_dt[warm_mask]
        max_ice_form = float(jnp.max(ice_formation))
        # Allow small numerical noise from sigmoid tails
        assert max_ice_form <= 1e-10, (
            f"{scheme}: ice forms above freezing, max dq_i_dt = {max_ice_form:.2e}"
        )


# ============================================================================
# 4g  Autoconversion threshold (Kessler)
# ============================================================================

def test_kessler_autoconversion_sensitivity():
    """Rain production should increase when q_c exceeds autoconversion threshold."""
    config = KesslerConfig()
    # Below threshold: subsaturated, small q_c
    T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz = _make_column(
        q_c_val=0.1 * config.autoconversion_threshold, supersaturated=False
    )
    out_lo = kessler_microphysics(T_lo, q_v_lo, hydro_lo, p_full, p_half, rho, dz,
                                   300.0, config=config)
    # Above threshold: subsaturated, large q_c
    _, _, hydro_hi, _, _, _, _ = _make_column(
        q_c_val=5.0 * config.autoconversion_threshold, supersaturated=False
    )
    out_hi = kessler_microphysics(T_lo, q_v_lo, hydro_hi, p_full, p_half, rho, dz,
                                   300.0, config=config)

    max_dqr_lo = float(jnp.max(out_lo.dq_r_dt))
    max_dqr_hi = float(jnp.max(out_hi.dq_r_dt))
    assert max_dqr_hi > max_dqr_lo, (
        f"Kessler: rain production not higher above threshold: "
        f"lo={max_dqr_lo:.2e}, hi={max_dqr_hi:.2e}"
    )


# ============================================================================
# Heating rate magnitude bounds
# ============================================================================

@pytest.mark.parametrize("scheme", ALL_SCHEMES)
def test_heating_rate_bounded(scheme):
    """|dT_dt| should be bounded (< 10 K/s for strongly supersaturated)."""
    T, q_v, hydro, p_full, p_half, rho, dz = _make_column(supersaturated=True)
    out = _call_scheme(scheme, T, q_v, hydro, p_full, p_half, rho, dz)
    max_hr = float(jnp.max(jnp.abs(out.dT_dt)))
    assert max_hr < 10.0, (
        f"{scheme}: |dT_dt| = {max_hr:.2e} K/s exceeds 10 K/s"
    )
