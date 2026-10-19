"""Scheme-level oracle-faithfulness tests for Kessler (1969) warm-rain microphysics.

Pins the classic Kessler process rates against an independent NumPy reimplementation
and locks the differentiable-surrogate departures as canaries.

FAITHFUL (classic Kessler forms + coefficients), each ISOLATED via a crafted
single-cell column so it is the only nonzero contributor to the pinned tendency
(saturated ⇒ no adjustment; q_r=0 ⇒ no accretion/evap; q_c=0 ⇒ no autoconv/
accretion), then pinned EXACTLY vs an independent NumPy reimpl using LITERAL
constants:
  * autoconversion A = k1·max(q_c − a, 0), k1 = a = 1e-3 (Kessler/Klemp-Wilhelmson);
  * accretion C = k2·q_c·q_r^0.875, k2 = 2.2, exponent 0.875;
  * latent heating dT/dt = L_v·(cond − evap)/c_pd;
  * rain evaporation = Klemp & Wilhelmson (1978) as coded in WRF
    module_mp_kessler.F (cgs rain content, p in Pa), pinned against a literal-
    constant NumPy reimpl, with its sub-saturation cap.
A separate constants canary pins the production private exponents == the literals.

DEPARTURE / SURROGATE (locked + labeled):
  * constant rain fall speed (V_t independent of q_r) ⇒ sedimentation flux LINEAR in
    q_r, NOT the Kessler q_r^1.1364 law; a single-layer precip magnitude also pins
    V_t = rain_fall_speed at the surface (ρ=ρ_sfc);
  * donor-clamped rain evaporation (evap·dt ≤ q_r) — a saturated-deficit canary;
  * AD-safe at the no-precip cold start: jax.grad wrt q_v AND q_r finite where the
    fractional powers q_r^0.875 / q_r^0.525 would otherwise trap;
  * unknown microphysics scheme raises ValueError (dispatch hardening).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.physics.microphysics._warm_rain import _RAIN_EVAP_VENT_EXP
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig,
    MicrophysicsConfig,
)
from legoesm.atmosphere.physics.microphysics.integration import get_microphysics_fn
from legoesm.atmosphere.physics.microphysics.kessler import (
    _KESSLER_ACCR_EXP,
    kessler_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_specific_humidity

from legoesm import constants


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _hydro(q_c, q_r):
    z = jnp.zeros_like(q_c)
    return HydrometeorState(q_c=q_c, q_r=q_r, q_i=z, q_s=z, q_g=z, N_c=z, N_r=z, N_i=z)


def _run(T, q_v, q_c, q_r, cfg, p=8.0e4, rho=1.0, dz=1000.0, dt=1.0):
    """Run kessler_microphysics on a single (1,1) cell."""
    def a(v):
        return jnp.array([[v]], dtype=jnp.float64)
    p_half = jnp.array([[p * 1.05, p * 0.95]], dtype=jnp.float64)
    return kessler_microphysics(
        a(T), a(q_v), _hydro(a(q_c), a(q_r)), a(p), p_half, a(rho), a(dz), dt, cfg,
    )


def _q_sat(T, p):
    return float(np.asarray(saturation_specific_humidity(jnp.array(T), jnp.array(p))).ravel()[0])


def _s(x):
    return float(np.asarray(x).ravel()[0])


def _autoconv_numpy(q_c, cfg):
    return cfg.autoconversion_rate * max(q_c - cfg.autoconversion_threshold, 0.0)


def _accretion_numpy(q_c, q_r, cfg):
    # Independent reimpl: LITERAL classic Kessler accretion exponent 0.875 (the
    # production private _KESSLER_ACCR_EXP is pinned == 0.875 by the constants test).
    return cfg.accretion_coeff * q_c * (max(q_r, 0.0) ** 0.875)


def _rain_evap_numpy(q_v, q_r, q_sat, p, rho, cfg):
    # Independent reimpl of WRF module_mp_kessler.F's ern / dt with LITERAL
    # constants: rcgs = 0.001*rho, (1.6+124.9*(rcgs*qr)**.2046)*(rcgs*qr)**.525
    # / (2.55e8/(p*qvs)+5.4e5) * dim(qvs,qv)/(rcgs*qvs)   (p in Pa)
    rcgs = 0.001 * rho
    c = rcgs * max(q_r, 0.0)
    return (cfg.evaporation_coeff * (1.6 + 124.9 * c ** 0.2046) * c ** 0.525
            / (2.55e8 / (p * q_sat) + 5.4e5) * max(q_sat - q_v, 0.0) / (rcgs * q_sat))


# ===========================================================================
# FAITHFUL forms
# ===========================================================================
def test_faithful_kessler_config_constants():
    """Canary: the Kessler process coefficients/exponents are the classic values."""
    c = KesslerConfig()
    assert c.autoconversion_rate == 1.0e-3        # k1 [1/s]
    assert c.autoconversion_threshold == 1.0e-3   # a [kg/kg]
    assert c.accretion_coeff == 2.2               # k2
    assert _KESSLER_ACCR_EXP == 0.875             # accretion q_r exponent
    assert _RAIN_EVAP_VENT_EXP == 0.525           # Marshall-Palmer ventilation


def test_faithful_kessler_autoconversion_pinned():
    """Isolated autoconversion: saturated (no adjustment), q_r=0 ⇒ dq_c=-k1·(q_c−a).

    q_v = q_sat makes the saturation adjustment inactive, q_r=0 kills accretion and
    rain evaporation, so the ONLY q_c sink is autoconversion and the ONLY q_r source
    is the same autoconversion (sedimentation acts on q_r=0 ⇒ 0).
    """
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    q_c = cfg.autoconversion_threshold + 5.0e-4  # small excess ⇒ unclamped
    o = _run(T, qs, q_c, 0.0, cfg, p=p)
    auto = _autoconv_numpy(q_c, cfg)
    assert _s(o.dq_c_dt) == pytest.approx(-auto, rel=1e-12, abs=1e-18)
    assert _s(o.dq_r_dt) == pytest.approx(+auto, rel=1e-12, abs=1e-18)
    # saturation adjustment truly inactive (no spurious condensation)
    assert _s(o.dq_v_dt) == pytest.approx(0.0, abs=1e-15)


def test_faithful_kessler_accretion_pinned():
    """Isolated accretion: saturated, q_c BELOW threshold (no autoconv) ⇒ dq_c=-k2·q_c·q_r^0.875."""
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    q_c = cfg.autoconversion_threshold * 0.3  # below threshold ⇒ autoconv = 0
    q_r = 1.0e-3
    o = _run(T, qs, q_c, q_r, cfg, p=p)
    accr = _accretion_numpy(q_c, q_r, cfg)
    assert _autoconv_numpy(q_c, cfg) == 0.0  # confirm autoconv gate is off
    assert _s(o.dq_c_dt) == pytest.approx(-accr, rel=1e-12, abs=1e-18)


@pytest.mark.parametrize("rho", [0.6, 1.0, 1.2])
def test_faithful_kessler_rain_evaporation_klemp_wilhelmson_pinned(rho):
    """Rain evaporation = Klemp & Wilhelmson (1978): subsaturated, q_c=0 ⇒ dq_v=+evap.

    Pinned against the literal-constant WRF form. q_c=0 kills autoconversion,
    accretion and cloud evaporation; the rate is far below both caps (q_r/dt and
    the sub-saturation), so the formula itself is exercised. The previous
    surrogate (evap_coeff·deficit·q_r^0.525) is ~10³× larger here and fails.
    """
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    q_v = qs * 0.8                  # RH = 80 %
    q_r = 1.0e-3
    o = _run(T, q_v, 0.0, q_r, cfg, p=p, rho=rho, dt=1.0)
    evap = _rain_evap_numpy(q_v, q_r, qs, p, rho, cfg)
    assert 1e-7 < evap < 1e-2 * q_r   # magnitude of the published rate; unclamped
    surrogate = 0.2 * q_r ** 0.525
    assert surrogate > 100.0 * evap
    assert _s(o.dq_v_dt) == pytest.approx(evap, rel=1e-12, abs=1e-18)
    assert _s(o.dq_c_dt) == pytest.approx(0.0, abs=1e-15)
    dT = -constants.L_v * evap / constants.c_pd
    assert _s(o.dT_dt) == pytest.approx(dT, rel=1e-12, abs=1e-18)


def test_faithful_kessler_autoconversion_threshold_gate():
    """The k1·max(q_c − a, 0) gate: no rain production below the threshold a."""
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    # q_c strictly below the threshold, q_r=0 ⇒ no rain produced at all
    below = _run(T, qs, cfg.autoconversion_threshold * 0.5, 0.0, cfg, p=p)
    assert _s(below.dq_r_dt) == pytest.approx(0.0, abs=1e-18)
    # just above ⇒ rain produced at exactly k1·(q_c − a)
    q_c = cfg.autoconversion_threshold + 2.0e-4
    above = _run(T, qs, q_c, 0.0, cfg, p=p)
    assert _s(above.dq_r_dt) == pytest.approx(_autoconv_numpy(q_c, cfg), rel=1e-12)


# ===========================================================================
# DEPARTURES (differentiable surrogates / numerics)
# ===========================================================================
def test_departure_kessler_constant_terminal_velocity():
    """Constant V_t ⇒ sedimentation precip LINEAR in q_r (not Kessler q_r^1.1364).

    The classic Kessler mass-weighted terminal velocity V_t = 36.34·(ρ q_r)^0.1364·
    (ρ_0/ρ)^0.5 is q_r-dependent, so precip ∝ q_r^1.1364.  This surrogate uses a
    FIXED rain_fall_speed, so precip ∝ q_r exactly (ratio 2.0 at 2× q_r).
    """
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    # saturated + q_c=0 ⇒ the only active process is q_r sedimentation
    q_r, rho, dz = 1.0e-4, 1.0, 5.0e3
    p_a = _s(_run(T, qs, 0.0, q_r, cfg, p=p, rho=rho, dz=dz, dt=1.0).precipitation)
    p_b = _s(_run(T, qs, 0.0, 2.0 * q_r, cfg, p=p, rho=rho, dz=dz, dt=1.0).precipitation)
    assert p_a > 0.0
    assert p_b / p_a == pytest.approx(2.0, rel=1e-9)          # constant V_t (linear)
    assert p_b / p_a != pytest.approx(2.0 ** 1.1364, rel=1e-3)  # NOT the Kessler law
    # Magnitude pin (single layer ⇒ ρ = ρ_sfc ⇒ density factor = 1): the surface
    # flux ρ·q_r·V_t with V_t = rain_fall_speed, confirming the flux form
    # (unclamped: V_t·dt/dz = 1e-3 ≪ 1).
    assert p_a == pytest.approx(rho * q_r * cfg.rain_fall_speed, rel=1e-9)
    # Pin the numeric DEFAULT explicitly (the flux pin above is form-only vs cfg).
    assert cfg.rain_fall_speed == 5.0    # default surrogate fall speed [m/s]


def test_departure_kessler_rain_evap_donor_clamped():
    """Rain evaporation is donor-clamped: evap·dt ≤ q_r (positivity surrogate).

    With a large deficit, a trace of rain and a long step, the unclamped KW78 rate
    exceeds q_r/dt, so production returns exactly q_r/dt — NOT the unclamped formula.
    """
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4
    qs = _q_sat(T, p)
    q_v = qs * 0.8          # RH 80 %, large deficit
    q_r = 1.0e-9
    dt = 1000.0
    o = _run(T, q_v, 0.0, q_r, cfg, p=p, rho=1.0, dt=dt)
    unclamped = _rain_evap_numpy(q_v, q_r, qs, p, 1.0, cfg)
    assert unclamped * dt > q_r                              # clamp genuinely fires
    assert _s(o.dq_v_dt) == pytest.approx(q_r / dt, rel=1e-12)  # clamped to q_r/dt


def test_departure_kessler_ad_safe_cold_start():
    """AD-safe at a no-precip cold start: jax.grad wrt BOTH q_v and q_r is finite.

    A precip-free column (q_c=q_r=0) is exactly where the fractional powers
    (accretion q_r^0.875, rain-evap q_r^0.525) and the saturation switch would
    otherwise leak inf/NaN cotangents.  The scheme is NOT globally smooth (it retains
    max/min/clip kinks); this pins only that the specific fractional-power / switch
    cold-start traps are guarded (safe_pow + the smooth adjustment).
    """
    cfg = KesslerConfig()
    T, p = 290.0, 8.0e4

    def out_qv(q_v_scalar):
        o = _run(T, q_v_scalar, 0.0, 0.0, cfg, p=p)
        return jnp.sum(o.dT_dt) + jnp.sum(o.precipitation) + jnp.sum(o.dq_r_dt)

    def out_qr(q_r_scalar):
        # q_r as the differentiated input, evaluated AT q_r=0 (the safe_pow trap)
        o = _run(T, _q_sat(T, p), 0.0, q_r_scalar, cfg, p=p)
        return jnp.sum(o.dq_r_dt) + jnp.sum(o.precipitation)

    g_qv = jax.grad(out_qv)(_q_sat(T, p) * 0.5)
    g_qr = jax.grad(out_qr)(0.0)
    assert np.isfinite(float(g_qv))
    assert np.isfinite(float(g_qr))     # fractional-power derivative guarded at q_r=0


def test_dispatch_unknown_scheme_raises():
    """Dispatch hardening: an unknown microphysics scheme raises ValueError."""
    with pytest.raises(ValueError, match="[Uu]nknown microphysics scheme"):
        get_microphysics_fn(MicrophysicsConfig(scheme="not_a_scheme"))


def test_kw78_rain_cannot_evaporate_past_the_cloud_left_subsaturation():
    """WRF ``max(-prod - qc, 0)`` cap: rain evaporates only the sub-saturation that
    remains after all cloud water has evaporated (``demand`` = the signed
    saturation-adjustment rate, negative when sub-saturated)."""
    from legoesm.atmosphere.physics.microphysics.kessler import kw78_rain_evaporation

    def a(v):
        return jnp.array([[v]], dtype=jnp.float64)
    T, p, dt = 290.0, 8.0e4, 1.0
    qs = _q_sat(T, p)
    q_v, q_r = qs * 0.8, 1.0e-3
    free = _s(kw78_rain_evaporation(a(q_v), a(q_r), a(0.0), a(qs), a(p), a(1.0),
                                    a(-1.0), dt))
    assert free == pytest.approx(_rain_evap_numpy(q_v, q_r, qs, p, 1.0, KesslerConfig()),
                                 rel=1e-12)
    # cloud water already able to absorb the whole deficit -> no rain evaporation
    assert _s(kw78_rain_evaporation(a(q_v), a(q_r), a(2e-3), a(qs), a(p), a(1.0),
                                    a(-1e-3), dt)) == 0.0
    # partly absorbed -> capped at the remainder
    capped = _s(kw78_rain_evaporation(a(q_v), a(q_r), a(1e-3), a(qs), a(p), a(1.0),
                                      a(-1e-3 - 0.5 * free), dt))
    assert capped == pytest.approx(0.5 * free, rel=1e-9)


def test_kw78_cap_is_the_unsmoothed_subsaturation_of_the_scheme():
    """In the scheme, the WRF ``max(-prod - qc, 0)`` cap uses the HARD adjustment
    amount ``prod = (q_v - q_s)/(1 + L_v dq_s/dT / c_pd)``, not the sigmoid-smoothed
    condensation rate (which is < half of it in sub-saturated air)."""
    cfg = KesslerConfig()
    T, p, dt = 290.0, 8.0e4, 1000.0
    qs = _q_sat(T, p)
    q_v, q_r = 0.5 * qs, 1.0e-2
    psy = 1.0 + constants.L_v ** 2 * qs / (constants.R_v * T ** 2 * constants.c_pd)
    cap = (qs - q_v) / (psy * dt)
    assert _rain_evap_numpy(q_v, q_r, qs, p, 1.0, cfg) > 2.0 * cap   # cap binds ...
    assert q_r / dt > 2.0 * cap                                      # ... alone
    o = _run(T, q_v, 0.0, q_r, cfg, p=p, rho=1.0, dt=dt)
    assert _s(o.dq_v_dt) == pytest.approx(cap, rel=1e-9)
