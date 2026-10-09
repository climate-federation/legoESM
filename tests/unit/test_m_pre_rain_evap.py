"""SAM M2005 rain evaporation PRE tests (iter-21).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:1994-2010) evaporates rain by
diffusion + ventilation of the rain PSD::

    EPSR = 2π·N0R·DV·[F1R/LAMR² + F2R·CONS9·(ARN·ρ/μ)^½·SC^⅓·LAMR^(−CONS34)]
    PRE  = EPSR·(q_v−q_sat)/AB        (subsaturated ⇒ PRE<0)

This replaces legoESM's bulk ``evap_coeff·subsat·q_r^0.525`` (number-blind,
and with the default coeff ~3000× too fast). PRE uses the prognostic N_r
(double-moment rain) and the ventilation term that accelerates evaporation of
falling drops — the cold-pool driver for organised deep convection.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    rain_evaporation_m2005,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_specific_humidity


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _pre(q_r=1.0e-3, N_r=1.0e4, RH=0.8, T=290.0, p=9.0e4, dt=None):
    qsat = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    return float(rain_evaporation_m2005(
        jnp.asarray(RH * qsat), jnp.asarray(q_r), jnp.asarray(N_r),
        jnp.asarray(qsat), jnp.asarray(T), jnp.asarray(p),
        jnp.asarray(rho), _CFG, dt=dt))


def _pre_hand(q_r=1.0e-3, N_r=1.0e4, RH=0.8, T=290.0, p=9.0e4):
    qsat = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    lamr = (math.pi * constants.rho_water * N_r / (rho * q_r)) ** (1.0 / 3.0)
    lamr = min(max(lamr, _CFG.lamr_min), _CFG.lamr_max)
    n0r = N_r * lamr
    dv = 8.794e-5 * T ** 1.81 / p
    mu = 1.496e-6 * T ** 1.5 / (T + 120.0)
    sc = mu / (rho * dv)
    arn = _CFG.fall_a_r * (_CFG.rho_su / rho) ** 0.54
    cons34 = 2.5 + _CFG.fall_b_r / 2.0
    cons9 = math.gamma(cons34)
    dqsdt = constants.L_v * qsat / (constants.R_v * T ** 2)
    ab = 1.0 + dqsdt * constants.L_v / constants.c_pd
    epsr = (2 * math.pi * n0r * dv
            * (_CFG.rain_vent_f1 / lamr ** 2
               + _CFG.rain_vent_f2 * cons9 * (arn * rho / mu) ** 0.5
               * sc ** (1.0 / 3.0) * lamr ** (-cons34)))
    return epsr * (1.0 - RH) * qsat / ab


def test_pre_matches_sam_formula():
    assert _pre() == pytest.approx(_pre_hand(), rel=1e-6)


def test_pre_realistic_timescale():
    """The faithful PRE gives a minutes-scale evaporation time, not the
    sub-second runaway of the bulk evap_coeff=1 form."""
    e = _pre(q_r=1.0e-3, RH=0.8)
    tau = 1.0e-3 / e
    assert 100.0 < tau < 3000.0           # ~10 min, physical


def test_pre_zero_when_saturated():
    assert _pre(RH=1.0) == 0.0
    assert _pre(RH=1.1) == 0.0            # supersaturated ⇒ no rain evap


def test_pre_zero_without_rain():
    assert _pre(q_r=0.0, N_r=0.0) == 0.0


def test_pre_increases_with_subsaturation():
    dry = _pre(RH=0.5)
    moist = _pre(RH=0.9)
    assert dry > moist > 0.0


def test_pre_couples_to_drop_number():
    """At fixed rain mass, more drops (smaller, more total surface) evaporate
    faster — the number dependence the bulk q_r^0.525 form misses."""
    few = _pre(N_r=1.0e3)
    many = _pre(N_r=1.0e6)
    assert many > few > 0.0


def test_pre_donor_clamped():
    """Evaporation cannot exceed q_r/dt."""
    e = _pre(q_r=1.0e-5, RH=0.3, dt=20.0)   # tiny rain, very dry ⇒ would over-evap
    assert e <= 1.0e-5 / 20.0 + 1e-18


def test_pre_ad_safe():
    """grad finite at q_r=0 and N_r=0 (safe_pow on the clamped slope)."""
    qsat = float(saturation_specific_humidity(jnp.asarray(290.0), jnp.asarray(9.0e4)))

    def loss(args):
        qr, nr = args
        return jnp.sum(rain_evaporation_m2005(
            jnp.asarray(0.8 * qsat), qr.reshape(1, 1), nr.reshape(1, 1),
            jnp.asarray(qsat), jnp.asarray(290.0), jnp.asarray(9.0e4),
            jnp.asarray(1.08), _CFG, dt=20.0))

    for qr0, nr0 in [(0.0, 0.0), (1.0e-3, 1.0e4), (0.0, 1.0e4)]:
        g = jax.grad(loss)((jnp.asarray(qr0), jnp.asarray(nr0)))
        assert bool(jnp.all(jnp.isfinite(jnp.asarray([g[0], g[1]]))))


def test_nsubr_removes_rain_number_during_evaporation():
    """SAM NSUBR (codex iter-21 MED): rain evaporation removes rain NUMBER at
    the same mass-fraction rate (−evap·N_r/q_r), so N_r doesn't go stale as
    rain evaporates. Isolated via the subsaturated−saturated DIFFERENCE in
    dN_r: the state (q_r, N_r) is identical, so number sedimentation + self-
    collection cancel and only the evaporation number sink remains."""
    qsat = float(saturation_specific_humidity(jnp.asarray(290.0), jnp.asarray(9.0e4)))
    q_r, N_r = 1.0e-3, 1.0e4
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), q_r), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), N_r), N_i=z,
    )
    cfg = MorrisonConfig()

    def dN_r_at(RH):
        out = morrison_microphysics(
            jnp.full((1, 1), 290.0), jnp.full((1, 1), RH * qsat), hm,
            jnp.full((1, 1), 9.0e4), jnp.full((1, 2), 9.0e4),
            jnp.full((1, 1), 1.08), jnp.full((1, 1), 300.0), 20.0, cfg)
        return float(out.dN_r_dt[0, 0])

    # subsaturated (evaporating) minus saturated (no evap) ⇒ only the
    # evaporation number sink differs.
    delta = dN_r_at(0.8) - dN_r_at(1.0)
    evap = float(rain_evaporation_m2005(
        jnp.asarray(0.8 * qsat), jnp.asarray(q_r), jnp.asarray(N_r),
        jnp.asarray(qsat), jnp.asarray(290.0), jnp.asarray(9.0e4),
        jnp.asarray(1.08), cfg, dt=20.0))
    assert delta < 0.0
    assert delta == pytest.approx(-evap * N_r / q_r, rel=1e-6)


def test_bulk_rain_evap_scheme_runs():
    """The legacy ``rain_evap_scheme="bulk"`` branch (number-blind
    ``evap_coeff·subsat·q_r^0.525``) must still evaluate to a finite,
    evaporative (dq_v >= 0 in subsaturated air) tendency — it is the
    non-default dispatch branch and was previously unexercised."""
    qsat = float(saturation_specific_humidity(jnp.asarray(290.0), jnp.asarray(9.0e4)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    out = morrison_microphysics(
        jnp.full((1, 1), 290.0), jnp.full((1, 1), 0.8 * qsat), hm,
        jnp.full((1, 1), 9.0e4), jnp.full((1, 2), 9.0e4),
        jnp.full((1, 1), 1.08), jnp.full((1, 1), 300.0), 20.0,
        MorrisonConfig(rain_evap_scheme="bulk"),
    )
    assert jnp.all(jnp.isfinite(out.dq_v_dt))
    assert jnp.all(jnp.isfinite(out.dq_r_dt))
    # Subsaturated air over rain ⇒ net vapour source (rain evaporates).
    assert float(out.dq_v_dt[0, 0]) >= 0.0


def test_unknown_rain_evap_scheme_raises():
    """The morrison dispatcher rejects an unknown scheme."""
    qsat = float(saturation_specific_humidity(jnp.asarray(290.0), jnp.asarray(9.0e4)))
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z, q_g=z,
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    with pytest.raises(ValueError, match="Unknown rain_evap_scheme"):
        morrison_microphysics(
            jnp.full((1, 1), 290.0), jnp.full((1, 1), 0.8 * qsat), hm,
            jnp.full((1, 1), 9.0e4), jnp.full((1, 2), 9.0e4),
            jnp.full((1, 1), 1.08), jnp.full((1, 1), 300.0), 20.0,
            MorrisonConfig(rain_evap_scheme="bulk_typo"),
        )
