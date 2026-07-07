"""Smoke tests for warm-rain microphysics helpers (`_warm_rain.py`).

Run with:

    JAX_ENABLE_X64=1 .venv/bin/python3.14 -m pytest \
        tests/unit/test_warm_rain.py -v
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
)


@pytest.fixture
def column_state():
    """Build a small (ncol, nlev) column with realistic warm-cloud values."""
    ncol, nlev = 4, 6
    T = jnp.full((ncol, nlev), 285.0)            # mid-troposphere temperature [K]
    p = jnp.full((ncol, nlev), 8.0e4)            # ~800 hPa
    rho = p / (constants.R_d * T)
    q_v = jnp.full((ncol, nlev), 1.5e-2)         # high vapor (likely supersat)
    q_c = jnp.full((ncol, nlev), 5.0e-4)         # cloud water
    q_r = jnp.full((ncol, nlev), 2.0e-4)         # rain
    N_c = 1.0e8 * jnp.ones((ncol, nlev))         # cloud droplet number [1/kg]
    N_r = 1.0e3 * jnp.ones((ncol, nlev))         # rain number [1/kg]
    return T, p, rho, q_v, q_c, q_r, N_c, N_r


def test_saturation_adjustment(column_state):
    T, p, _, q_v, _, _, _, _ = column_state
    cond, q_sat = saturation_adjustment(T, q_v, p, dt=300.0)
    assert cond.shape == T.shape
    assert q_sat.shape == T.shape
    assert jnp.all(jnp.isfinite(cond))
    assert jnp.all(jnp.isfinite(q_sat))
    assert jnp.all(q_sat > 0.0)
    # Supersaturated air → positive condensation
    assert float(jnp.mean(cond)) > 0.0


def test_effective_Nc(column_state):
    *_, N_c, _ = column_state
    Nc_eff = effective_Nc(N_c, Nc_0=5.0e7)
    assert Nc_eff.shape == N_c.shape
    assert jnp.all(jnp.isfinite(Nc_eff))
    assert jnp.all(Nc_eff > 1.0)
    # Where N_c is set (>1) the effective value passes through.
    assert jnp.allclose(Nc_eff, N_c)
    # Where N_c is zero, fallback kicks in.
    fallback = effective_Nc(jnp.zeros_like(N_c), Nc_0=5.0e7)
    assert jnp.allclose(fallback, 5.0e7)


def test_autoconversion_sb(column_state):
    _, _, rho, _, q_c, _, N_c, _ = column_state
    dq_au, dN_au, x_c = autoconversion_sb(
        q_c, N_c, rho, k_au=9.44e9, x_star=2.6e-10,
    )
    assert dq_au.shape == q_c.shape
    assert dN_au.shape == q_c.shape
    assert x_c.shape == q_c.shape
    assert jnp.all(jnp.isfinite(dq_au))
    assert jnp.all(jnp.isfinite(dN_au))
    assert jnp.all(jnp.isfinite(x_c))
    assert jnp.all(dq_au >= 0.0)
    assert jnp.all(x_c > 0.0)


def test_autoconversion_sb_number_closure_newborn_mass_is_x_star(column_state):
    """SB2001/2006 number closure: dN_r_au = (mass rate)/x_star, i.e. each
    newborn rain drop carries the separation mass x_* (2.6e-10 kg ~ 79 um).

    Pre-fix the divisor was x_star*20 (a transplanted mass-rate coefficient
    constant), making newborn drops 20x too massive and 20x too few for a
    given autoconversion mass flux.
    """
    _, _, rho, _, q_c, _, N_c, _ = column_state
    x_star = 2.6e-10
    dq_au, dN_au, _ = autoconversion_sb(
        q_c, N_c, rho, k_au=9.44e9, x_star=x_star,
    )
    active = dN_au > 0.0
    assert bool(jnp.any(active)), "fixture produced no autoconversion"
    newborn_mass = jnp.where(active, dq_au * rho / jnp.where(active, dN_au, 1.0), x_star)
    assert bool(jnp.allclose(newborn_mass, x_star, rtol=1e-10)), (
        f"implied newborn rain-drop mass {float(jnp.max(newborn_mass)):.3e} kg "
        f"!= x_star={x_star:.3e} kg (SB2001 number closure au/x_*)."
    )


def test_accretion(column_state):
    _, _, rho, _, q_c, q_r, _, _ = column_state
    rate = accretion(q_c, q_r, rho, k_ac=5.25)
    assert rate.shape == q_c.shape
    assert jnp.all(jnp.isfinite(rate))
    assert jnp.all(rate >= 0.0)


def test_self_collection_breakup(column_state):
    _, _, rho, _, _, q_r, _, N_r = column_state
    dN_sc, dN_br = self_collection_breakup(
        N_r, q_r, rho, k_sc=7.12, breakup_sharpness=10.0, D_eq=9.0e-4,
    )
    assert dN_sc.shape == N_r.shape
    assert dN_br.shape == N_r.shape
    assert jnp.all(jnp.isfinite(dN_sc))
    assert jnp.all(jnp.isfinite(dN_br))
    # Self-collection always reduces number; breakup either zero or positive.
    assert jnp.all(dN_sc <= 0.0)
    assert jnp.all(dN_br >= 0.0)


def test_rain_evaporation(column_state):
    T, p, _, _, _, q_r, _, _ = column_state
    # Subsaturated vapor profile to drive evaporation.
    q_v_low = 1.0e-3 * jnp.ones_like(q_r)
    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)
    evap = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3)
    assert evap.shape == q_r.shape
    assert jnp.all(jnp.isfinite(evap))
    assert jnp.all(evap >= 0.0)
    # With substantial subsaturation, average evaporation is strictly positive.
    assert float(jnp.mean(evap)) > 0.0


def test_rain_evaporation_rh_deficit_floor(column_state):
    """``rh_deficit_floor`` suppresses only sub-resolution near-saturation evap.

    The floor is applied to the RELATIVE sub-saturation deficit
    ``(q_sat−q_v)/q_sat`` via a soft threshold ``clip(deficit−floor, 0)``.
    ``floor=0`` is exactly the legacy behaviour; a small floor (~1e-4) removes
    the spurious in-cloud evaporation that rectifies float32 saturation
    round-off (RH > 99.99 %) while leaving RESOLVED deficits — sub-cloud
    downdrafts and mixed-phase WBF (deficit ≫ floor) — essentially untouched.
    """
    T, p, _, _, _, q_r, _, _ = column_state
    q_v_low = 1.0e-3 * jnp.ones_like(q_r)          # strongly subsaturated
    _, q_sat = saturation_adjustment(T, q_v_low, p, dt=300.0)

    evap_legacy = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3)
    # floor=0 reproduces the legacy ungated result exactly.
    evap_floor0 = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3,
                                   rh_deficit_floor=0.0)
    assert jnp.array_equal(evap_floor0, evap_legacy)

    # Large RESOLVED deficit: q_v_low is far subsaturated (deficit ~1, ≫ floor),
    # so a 1e-4 floor changes the rate by < 0.1 % — WBF/sub-cloud preserved.
    evap_floored = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3,
                                    rh_deficit_floor=1.0e-4)
    assert jnp.all(evap_floored <= evap_legacy)
    rel = jnp.max(jnp.abs(evap_floored - evap_legacy)
                  / jnp.clip(evap_legacy, 1e-30))
    assert float(rel) < 1.0e-3

    # Near-saturation (deficit below the floor) is fully suppressed.
    q_v_near = q_sat * (1.0 - 5.0e-5)              # RH 99.995 %, deficit < floor
    evap_near = rain_evaporation(q_v_near, q_r, q_sat, evap_coeff=1.0e-3,
                                 rh_deficit_floor=1.0e-4)
    assert jnp.all(evap_near == 0.0)
    # …but the same near-saturation column evaporates without the floor.
    evap_near_legacy = rain_evaporation(q_v_near, q_r, q_sat, evap_coeff=1.0e-3)
    assert float(jnp.max(evap_near_legacy)) > 0.0

    # The floor is genuinely WIRED into the computation graph (not a dead
    # kwarg): differentiate the rate w.r.t. ``rh_deficit_floor`` itself. On a
    # resolved-subsaturation column the soft threshold ``clip(deficit−floor,0)``
    # is active, so d(evap)/d(floor) = −evap_coeff·q_r^0.525 < 0 — strictly
    # negative and finite. A revert of the ``deficit − floor`` behaviour (the
    # e72e4aea fix) would zero this gradient and fail the assertion.
    devap_dfloor = jax.grad(lambda fl: jnp.sum(
        rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3,
                         rh_deficit_floor=fl)))(1.0e-4)
    assert jnp.isfinite(devap_dfloor)
    assert float(devap_dfloor) < 0.0

    # Negative floor is clamped to 0 (no spurious evaporation boost / NaN).
    evap_neg = rain_evaporation(q_v_low, q_r, q_sat, evap_coeff=1.0e-3,
                                rh_deficit_floor=-1.0)
    assert jnp.array_equal(evap_neg, evap_legacy)

    # Boundary kink: deficit == floor exactly → soft threshold gives 0, and the
    # rate (and its q_v gradient) stay finite at the clip kink.
    floor = 1.0e-4
    q_v_at = q_sat * (1.0 - floor)                 # deficit == floor exactly
    evap_at = rain_evaporation(q_v_at, q_r, q_sat, evap_coeff=1.0e-3,
                               rh_deficit_floor=floor)
    assert jnp.allclose(evap_at, 0.0)
    g_at = jax.grad(lambda qv: jnp.sum(
        rain_evaporation(qv, q_r, q_sat, evap_coeff=1.0e-3,
                         rh_deficit_floor=floor)))(q_v_at)
    assert jnp.all(jnp.isfinite(g_at))

    # Very small q_sat (cold cirrus / tropopause): the 1e-10 denominator guard
    # keeps the rate and gradient finite (no Inf/NaN from deficit/q_sat).
    q_sat_tiny = jnp.full_like(q_r, 1.0e-12)
    q_v_tiny = jnp.zeros_like(q_r)
    evap_tiny = rain_evaporation(q_v_tiny, q_r, q_sat_tiny, evap_coeff=1.0e-3,
                                 rh_deficit_floor=floor)
    assert jnp.all(jnp.isfinite(evap_tiny)) and jnp.all(evap_tiny >= 0.0)
    # Differentiate w.r.t. the tiny q_sat itself so the clip(q_sat, 1e-10)
    # denominator floor is exercised under reverse-mode AD (zero subgradient at
    # the floor must not produce NaN/Inf).
    g_tiny = jax.grad(lambda qs: jnp.sum(
        rain_evaporation(q_v_tiny, q_r, qs, evap_coeff=1.0e-3,
                         rh_deficit_floor=floor)))(q_sat_tiny)
    assert jnp.all(jnp.isfinite(g_tiny))

    # Gradient w.r.t. q_v stays finite (clip-based floor is AD-safe).
    def _loss(qv):
        return jnp.sum(rain_evaporation(qv, q_r, q_sat, evap_coeff=1.0e-3,
                                        rh_deficit_floor=1.0e-4))
    g = jax.grad(_loss)(q_v_low)
    assert jnp.all(jnp.isfinite(g))


def test_jit_compiles(column_state):
    T, p, rho, q_v, q_c, q_r, N_c, N_r = column_state

    @jax.jit
    def chain(T_, q_v_, q_c_, q_r_, N_c_, N_r_, rho_, p_):
        cond, q_sat = saturation_adjustment(T_, q_v_, p_, dt=300.0)
        Nc_eff = effective_Nc(N_c_, Nc_0=5.0e7)
        dq_au, dN_au, _ = autoconversion_sb(
            q_c_, Nc_eff, rho_, k_au=9.44e9, x_star=2.6e-10,
        )
        acc = accretion(q_c_, q_r_, rho_, k_ac=5.25)
        evap = rain_evaporation(q_v_, q_r_, q_sat, evap_coeff=1.0e-3)
        return cond + dq_au + dN_au + acc + evap

    out = chain(T, q_v, q_c, q_r, N_c, N_r, rho, p)
    assert out.shape == T.shape
    assert jnp.all(jnp.isfinite(out))
