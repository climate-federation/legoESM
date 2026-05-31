"""SAM M2005 heat-balance snow melting PSMLT tests (iter-29).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:2030) melts snow to rain at the
ventilation-limited heat-conduction rate::

    PSMLT = 2π·N0S·KAP·(T−T₀)₊/L_f · [F1S/LAMS² + F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(−CONS35)]

with KAP = 1.414e3·μ (air thermal conductivity). legoESM previously used a
crude bulk ``melt_rate·q_s`` (constant timescale), so snow persisted too far
below the 0 °C level. Requires double-moment snow (N_s).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    snow_melting_psmlt,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _ps(T, q_s=1.0e-3, N_s=1.0e4, p=8.0e4, dt=None):
    rho = p / (constants.R_d * T)
    return float(snow_melting_psmlt(
        jnp.asarray(q_s), jnp.asarray(N_s), jnp.asarray(T),
        jnp.asarray(p), jnp.asarray(rho), _CFG, dt=dt))


def _hand(T, q_s=1.0e-3, N_s=1.0e4, p=8.0e4):
    rho = p / (constants.R_d * T)
    lams = (_CFG.rho_snow * math.pi * N_s / q_s) ** (1.0 / 3.0)
    lams = min(max(lams, _CFG.lams_min), _CFG.lams_max)
    n0s = N_s * lams
    dv = 8.794e-5 * T ** 1.81 / p
    mu = 1.496e-6 * T ** 1.5 / (T + 120.0)
    kap = 1.414e3 * mu
    sc = mu / (rho * dv)
    asn = _CFG.fall_a_s * (_CFG.rho_su / rho) ** 0.54
    cons35 = 2.5 + _CFG.fall_b_s / 2.0
    cons10 = math.gamma(cons35)
    vent = (_CFG.snow_vent_f1 / lams ** 2
            + _CFG.snow_vent_f2 * cons10 * (asn * rho / mu) ** 0.5
            * sc ** (1.0 / 3.0) * lams ** (-cons35))
    return (2 * math.pi * n0s * kap * max(T - constants.T_freeze, 0.0)
            / constants.L_f * vent)


def test_psmlt_matches_sam_formula():
    assert _ps(276.0) == pytest.approx(_hand(276.0), rel=1e-6)


def test_melting_faster_when_warmer():
    """Heat balance: more heat above 0 °C ⇒ faster melting."""
    assert _ps(278.0) > _ps(274.0) > 0.0


def test_no_melting_below_freezing():
    assert _ps(270.0) == 0.0
    assert _ps(constants.T_freeze) == 0.0


def test_psmlt_donor_clamped():
    assert _ps(285.0, q_s=1.0e-3, dt=20.0) <= 1.0e-3 / 20.0 + 1e-18


def _morrison(do_melt, T=276.0):
    # liquid-SATURATED so the melted rain doesn't evaporate (isolate melting).
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=jnp.full((1, 1), 1.0e-3), q_g=z,
        N_c=z, N_r=z, N_i=z, N_s=jnp.full((1, 1), 1.0e4),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsl), hm,
        jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0,
        MorrisonConfig(do_snow_melting=do_melt))
    return (float(out.dq_s_dt[0, 0]), float(out.dq_r_dt[0, 0]),
            float(out.dT_dt[0, 0]))


def test_melting_snow_to_rain_and_cools():
    """Saturated air: snow → rain (Δdq_s = −Δdq_r) and melting ABSORBS L_f
    (cooling)."""
    dqs_on, dqr_on, dT_on = _morrison(True)
    dqs_off, dqr_off, dT_off = _morrison(False)
    assert (dqs_on - dqs_off) == pytest.approx(-(dqr_on - dqr_off), abs=1e-9)
    assert (dqs_on - dqs_off) < 0.0       # snow melted
    assert dT_on < dT_off                 # cooling (L_f absorbed)


def test_psmlt_ad_safe():
    def loss(q_s0):
        rho = 8.0e4 / (constants.R_d * 278.0)
        return jnp.sum(snow_melting_psmlt(
            q_s0.reshape(1, 1), jnp.full((1, 1), 1.0e4), jnp.asarray(278.0),
            jnp.asarray(8.0e4), jnp.asarray(rho), _CFG, dt=20.0))

    for q0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
