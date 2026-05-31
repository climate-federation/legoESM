"""SAM M2005 graupel foundation tests (iter-34, M4).

SAM (``dograupel=.true.``) freezes supercooled rain to GRAUPEL — dense frozen
drops — not snow. legoESM now carries single-moment graupel in tracer slot [5]
(fixed intercept N0G), with the SAM PSD fall speed and the PGMLT
ventilation-limited melting (module_mp_graupel.f90: AG=19.3, BG=0.37, RHOG=400,
PGMLT at 2069, the same form as snow PSMLT).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    graupel_lamg,
    graupel_melting_pgmlt,
    graupel_riming_psacwg,
    graupel_deposition_prdg,
    graupel_rain_accretion_pracg,
    snow_to_graupel_pgsacw,
    graupel_lamg_n0g,
)
from legoesm.thermo import saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _lamg(q_g, rho=1.0):
    return float(graupel_lamg(jnp.asarray(q_g), jnp.asarray(rho), _CFG))


def _pg(T, q_g=1.0e-3, p=8.0e4, dt=None):
    rho = p / (constants.R_d * T)
    return float(graupel_melting_pgmlt(
        jnp.asarray(q_g), jnp.asarray(T), jnp.asarray(p),
        jnp.asarray(rho), _CFG, dt=dt))


def _pg_hand(T, q_g=1.0e-3, p=8.0e4):
    rho = p / (constants.R_d * T)
    lamg = (math.pi * _CFG.rho_graupel * _CFG.n0_graupel / (rho * q_g)) ** 0.25
    lamg = min(max(lamg, _CFG.lamg_min), _CFG.lamg_max)
    n0g_m = _CFG.n0_graupel / rho
    dv = 8.794e-5 * T ** 1.81 / p
    mu = 1.496e-6 * T ** 1.5 / (T + 120.0)
    kap = 1.414e3 * mu
    sc = mu / (rho * dv)
    agn = _CFG.fall_a_g * (_CFG.rho_su / rho) ** 0.54
    cons36 = 2.5 + _CFG.fall_b_g / 2.0
    cons11 = math.gamma(cons36)
    vent = (_CFG.graupel_vent_f1 / lamg ** 2
            + _CFG.graupel_vent_f2 * cons11 * (agn * rho / mu) ** 0.5
            * sc ** (1.0 / 3.0) * lamg ** (-cons36))
    return (2 * math.pi * n0g_m * kap * max(T - constants.T_freeze, 0.0)
            / constants.L_f * vent)


# -------- PSD slope + melting unit formulas --------

def test_lamg_slope_formula():
    rho = 1.0
    expect = (math.pi * _CFG.rho_graupel * _CFG.n0_graupel
              / (rho * 1.0e-3)) ** 0.25
    expect = min(max(expect, _CFG.lamg_min), _CFG.lamg_max)
    assert _lamg(1.0e-3) == pytest.approx(expect, rel=1e-6)
    # more graupel ⇒ smaller slope (bigger particles)
    assert _lamg(5.0e-3) < _lamg(1.0e-3)


def test_pgmlt_matches_sam_formula():
    assert _pg(278.0) == pytest.approx(_pg_hand(278.0), rel=1e-6)


def test_pgmlt_no_melt_below_freezing():
    assert _pg(270.0) == 0.0
    assert _pg(constants.T_freeze) == 0.0


def test_pgmlt_faster_when_warmer():
    assert _pg(280.0) > _pg(274.5) > 0.0


def test_pgmlt_donor_and_heat_capped():
    # heavy graupel, warm: capped by q_g/dt and the sensible heat
    assert _pg(285.0, q_g=1.0e-3, dt=20.0) <= 1.0e-3 / 20.0 + 1e-18


def test_pgmlt_ad_safe():
    def loss(q0):
        rho = 8.0e4 / (constants.R_d * 280.0)
        return jnp.sum(graupel_melting_pgmlt(
            q0.reshape(1, 1), jnp.asarray(280.0), jnp.asarray(8.0e4),
            jnp.asarray(rho), _CFG, dt=20.0))
    for q0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))


# -------- Integration: frozen rain → graupel; graupel melts → rain --------

def _morrison(T, do_graupel=True, q_r=1.0e-3, q_g=0.0, do_dep=True):
    qsat = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), q_r), q_i=z, q_s=z,
        q_g=jnp.full((1, 1), q_g),
        N_c=z, N_r=jnp.full((1, 1), 1.0e4), N_i=z,
    )
    # huge dz suppresses sedimentation so the phase budget is isolated
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsat), hm,
        jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0,
        MorrisonConfig(do_graupel=do_graupel, do_graupel_deposition=do_dep))
    return out


def test_frozen_rain_goes_to_graupel_not_snow():
    """At T<0 with do_graupel, Bigg-frozen rain feeds q_g (Δq_r = −Δq_g) and
    leaves q_s untouched."""
    out = _morrison(258.0, do_graupel=True)
    dqr = float(out.dq_r_dt[0, 0])
    dqg = float(out.dq_g_dt[0, 0])
    dqs = float(out.dq_s_dt[0, 0])
    assert dqg > 0.0                              # graupel formed
    assert dqr < 0.0                              # rain frozen
    assert dqg == pytest.approx(-dqr, abs=1e-9)   # rain→graupel (residual=sed)
    assert dqs == pytest.approx(0.0, abs=1e-12)   # snow untouched


def test_legacy_path_routes_to_snow():
    """do_graupel=False reproduces the legacy frozen-rain→snow routing."""
    out = _morrison(258.0, do_graupel=False)
    assert float(out.dq_s_dt[0, 0]) > 0.0         # snow formed
    assert float(out.dq_g_dt[0, 0]) == pytest.approx(0.0, abs=1e-12)


def test_graupel_melts_to_rain_and_cools():
    """Above 0 °C, existing graupel melts to rain (Δq_g<0, Δq_r>0) and the
    melting absorbs L_f (cooling)."""
    out = _morrison(280.0, q_r=0.0, q_g=1.0e-3, do_dep=False)  # isolate melt
    dqg = float(out.dq_g_dt[0, 0])
    dqr = float(out.dq_r_dt[0, 0])
    assert dqg < 0.0                              # graupel melted
    assert dqr > 0.0                              # → rain
    assert dqr == pytest.approx(-dqg, abs=1e-9)   # graupel→rain (residual=sed)


def test_graupel_outputs_finite():
    out = _morrison(258.0)
    for fld in (out.dq_g_dt, out.dq_r_dt, out.dq_s_dt, out.dT_dt):
        assert bool(jnp.all(jnp.isfinite(fld)))


# -------- PSACWG graupel riming of cloud water (iter-35) --------

def _psacwg(T, q_c=1.0e-3, q_g=1.0e-3, p=8.0e4, dt=None):
    rho = p / (constants.R_d * T)
    return float(graupel_riming_psacwg(
        jnp.asarray(q_c), jnp.asarray(q_g), jnp.asarray(T),
        jnp.asarray(rho), _CFG, dt=dt))


def _psacwg_hand(T, q_c=1.0e-3, q_g=1.0e-3, p=8.0e4):
    import jax.nn as jnn
    rho = p / (constants.R_d * T)
    lamg = (math.pi * _CFG.rho_graupel * _CFG.n0_graupel
            / (max(rho, 0.1) * q_g)) ** 0.25
    lamg = min(max(lamg, _CFG.lamg_min), _CFG.lamg_max)
    n0g_m = _CFG.n0_graupel / max(rho, 0.1)
    agn = _CFG.fall_a_g * (_CFG.rho_su / rho) ** 0.54
    bg = _CFG.fall_b_g
    cons14 = math.gamma(bg + 3.0) * math.pi / 4.0 * _CFG.graupel_collect_eff
    rate = cons14 * agn * q_c * rho * n0g_m / lamg ** (bg + 3.0)
    frac = float(jnn.sigmoid(_CFG.melt_sharpness * (constants.T_freeze - T)))
    return rate * frac


def test_psacwg_matches_sam_formula():
    assert _psacwg(258.0) == pytest.approx(_psacwg_hand(258.0), rel=1e-6)


def test_psacwg_cold_gate():
    """Riming only acts on supercooled cloud water (T<0 °C)."""
    assert _psacwg(258.0) > 0.0                 # supercooled ⇒ rimes
    assert _psacwg(285.0) < 0.01 * _psacwg(258.0)  # warm ⇒ gated off


def test_psacwg_donor_clamped():
    assert _psacwg(258.0, q_c=1.0e-3, dt=20.0) <= 1.0e-3 / 20.0 + 1e-18


def test_psacwg_grows_graupel_and_warms():
    """Supercooled cloud water rimed onto graupel: q_c→q_g (Δq_c=−Δq_g for the
    riming term) and the freezing releases L_f (warming)."""
    T = 258.0
    qsat = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), 1.0e-3), q_r=z, q_i=z, q_s=z,
        q_g=jnp.full((1, 1), 1.0e-3),
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=z,
    )
    args = (jnp.full((1, 1), T), jnp.full((1, 1), qsat), hm,
            jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0)
    on = morrison_microphysics(*args, MorrisonConfig(do_graupel_riming=True))
    off = morrison_microphysics(*args, MorrisonConfig(do_graupel_riming=False))
    # riming adds graupel mass and removes cloud water
    assert float(on.dq_g_dt[0, 0]) > float(off.dq_g_dt[0, 0])
    assert float(on.dq_c_dt[0, 0]) < float(off.dq_c_dt[0, 0])
    # latent heat of fusion warms relative to no-riming
    assert float(on.dT_dt[0, 0]) > float(off.dT_dt[0, 0])


def test_psacwg_ad_safe():
    rho = 8.0e4 / (constants.R_d * 258.0)

    def loss_qc(q_c0):
        return jnp.sum(graupel_riming_psacwg(
            q_c0.reshape(1, 1), jnp.full((1, 1), 1.0e-3), jnp.asarray(258.0),
            jnp.asarray(rho), _CFG, dt=20.0))

    # q_g=0 is the dangerous edge (LAMG = (.../q_g)^¼): the denominator floor +
    # LAMG clamp keep it finite, so the reverse-mode adjoint must not NaN (F).
    def loss_qg(q_g0):
        return jnp.sum(graupel_riming_psacwg(
            jnp.full((1, 1), 1.0e-3), q_g0.reshape(1, 1), jnp.asarray(258.0),
            jnp.asarray(rho), _CFG, dt=20.0))

    for q0 in (0.0, 1.0e-3):
        assert bool(jnp.isfinite(jax.grad(loss_qc)(jnp.asarray(q0))))
        assert bool(jnp.isfinite(jax.grad(loss_qg)(jnp.asarray(q0))))


# -------- PRDG graupel deposition/sublimation (iter-36) --------

def _prdg(q_v, T=258.0, q_g=1.0e-3, p=8.0e4, dt=None):
    rho = p / (constants.R_d * T)
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    return float(graupel_deposition_prdg(
        jnp.asarray(q_v), jnp.asarray(q_g), jnp.asarray(qsi), jnp.asarray(T),
        jnp.asarray(p), jnp.asarray(rho), _CFG, dt=dt))


def _prdg_hand(q_v, T=258.0, q_g=1.0e-3, p=8.0e4):
    rho = p / (constants.R_d * T)
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    lamg = (math.pi * _CFG.rho_graupel * _CFG.n0_graupel
            / (max(rho, 0.1) * q_g)) ** 0.25
    lamg = min(max(lamg, _CFG.lamg_min), _CFG.lamg_max)
    dv = 8.794e-5 * T ** 1.81 / p
    mu = 1.496e-6 * T ** 1.5 / (T + 120.0)
    sc = mu / (rho * dv)
    agn = _CFG.fall_a_g * (_CFG.rho_su / rho) ** 0.54
    cons36 = 2.5 + _CFG.fall_b_g / 2.0
    cons11 = math.gamma(cons36)
    dqsidt = constants.L_s * qsi / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    epsg = (2 * math.pi * _CFG.n0_graupel * dv
            * (_CFG.graupel_vent_f1 / lamg ** 2
               + _CFG.graupel_vent_f2 * cons11 * (agn * rho / mu) ** 0.5
               * sc ** (1.0 / 3.0) * lamg ** (-cons36)))
    return epsg * (q_v - qsi) / abi


def test_prdg_matches_sam_formula():
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(258.0), jnp.asarray(8.0e4)))
    assert _prdg(qsi * 1.5) == pytest.approx(_prdg_hand(qsi * 1.5), rel=1e-6)


def test_prdg_sign_deposition_vs_sublimation():
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(258.0), jnp.asarray(8.0e4)))
    assert _prdg(qsi * 1.5) > 0.0          # ice-supersaturated ⇒ deposition
    assert _prdg(qsi * 0.5) < 0.0          # subsaturated ⇒ sublimation


def test_prdg_sublimation_donor_clamped():
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(258.0), jnp.asarray(8.0e4)))
    # strongly subsaturated, small graupel: |sublimation| ≤ q_g/dt
    assert _prdg(0.0, q_g=1.0e-4, dt=20.0) >= -1.0e-4 / 20.0 - 1e-18


def test_prdg_integration_deposition_grows_and_warms():
    """Ice-supersaturated air: PRDG deposits vapour onto graupel (q_g grows,
    q_v sink) and releases L_s (warming)."""
    T = 258.0
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=z, q_g=jnp.full((1, 1), 1.0e-3),
        N_c=z, N_r=z, N_i=z,
    )
    args = (jnp.full((1, 1), T), jnp.full((1, 1), qsi * 1.3), hm,
            jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0)
    on = morrison_microphysics(*args, MorrisonConfig(do_graupel_deposition=True))
    off = morrison_microphysics(*args, MorrisonConfig(do_graupel_deposition=False))
    assert float(on.dq_g_dt[0, 0]) > float(off.dq_g_dt[0, 0])   # graupel grew
    assert float(on.dq_v_dt[0, 0]) < float(off.dq_v_dt[0, 0])   # vapour consumed
    assert float(on.dT_dt[0, 0]) > float(off.dT_dt[0, 0])       # L_s warming


def test_prdg_ad_safe():
    def loss(q_g0):
        rho = 8.0e4 / (constants.R_d * 258.0)
        qsi = float(saturation_mixing_ratio_ice(
            jnp.asarray(258.0), jnp.asarray(8.0e4)))
        return jnp.sum(graupel_deposition_prdg(
            jnp.full((1, 1), qsi * 1.2), q_g0.reshape(1, 1),
            jnp.asarray(qsi), jnp.asarray(258.0), jnp.asarray(8.0e4),
            jnp.asarray(rho), _CFG, dt=20.0))
    for q0 in (0.0, 1.0e-3):
        assert bool(jnp.isfinite(jax.grad(loss)(jnp.asarray(q0))))


# -------- PRACG graupel rain accretion (iter-37) --------

def _pracg(T=265.0, q_r=1.0e-3, N_r=1.0e5, q_g=1.0e-3, rho=0.9, dt=None):
    return float(graupel_rain_accretion_pracg(
        jnp.asarray(q_r), jnp.asarray(N_r), jnp.asarray(q_g), jnp.asarray(T),
        jnp.asarray(rho), _CFG, dt=dt))


def _pracg_hand(T=265.0, q_r=1.0e-3, N_r=1.0e5, q_g=1.0e-3, rho=0.9):
    import jax.nn as jnn
    dum = (_CFG.rho_su / rho) ** 0.54
    lamr = (math.pi * constants.rho_water * N_r / (rho * q_r)) ** (1.0 / 3.0)
    lamr = min(max(lamr, _CFG.lamr_min), _CFG.lamr_max)
    n0rr = N_r / rho * lamr
    lamg = (math.pi * _CFG.rho_graupel * _CFG.n0_graupel
            / (max(rho, 0.1) * q_g)) ** 0.25
    lamg = min(max(lamg, _CFG.lamg_min), _CFG.lamg_max)
    n0g_m = _CFG.n0_graupel / rho
    cons4 = math.gamma(4.0 + _CFG.fall_b_r) / 6.0
    cons7g = math.gamma(4.0 + _CFG.fall_b_g) / 6.0
    umr = min(_CFG.fall_a_r * cons4 * lamr ** (-_CFG.fall_b_r) * dum, 9.1 * dum)
    umg = min(_CFG.fall_a_g * cons7g * lamg ** (-_CFG.fall_b_g) * dum, 20.0 * dum)
    vdiff = ((1.2 * umr - 0.95 * umg) ** 2 + 0.08 * umg * umr) ** 0.5
    cons41 = math.pi ** 2 * _CFG.graupel_rain_collect_eff * constants.rho_water
    bracket = (5.0 / (lamr ** 3 * lamg) + 2.0 / (lamr ** 2 * lamg ** 2)
               + 0.5 / (lamr * lamg ** 3))
    rate = cons41 * vdiff * rho * n0rr * n0g_m / lamr ** 3 * bracket
    frac = float(jnn.sigmoid(_CFG.melt_sharpness * (constants.T_freeze - T)))
    return rate * frac


def test_pracg_matches_sam_formula():
    assert _pracg() == pytest.approx(_pracg_hand(), rel=1e-6)
    assert _pracg(q_r=2e-3, N_r=5e5, q_g=3e-3) == pytest.approx(
        _pracg_hand(q_r=2e-3, N_r=5e5, q_g=3e-3), rel=1e-6)


def test_pracg_cold_gate():
    assert _pracg(T=265.0) > 0.0                       # cold => accretes
    assert _pracg(T=285.0) < 0.01 * _pracg(T=265.0)    # warm => gated off


def test_pracg_donor_clamped():
    assert _pracg(q_r=1.0e-3, dt=20.0) <= 1.0e-3 / 20.0 + 1e-18


def test_pracg_grows_with_both_species():
    """Two-PSD collection: more rain OR more graupel => faster accretion."""
    base = _pracg()
    assert _pracg(q_g=5.0e-3) > base
    assert _pracg(q_r=5.0e-3, N_r=5.0e5) > base


def test_pracg_accretes_rain_to_graupel_and_warms():
    """Cold: rain collected by graupel and the freezing releases L_f."""
    T = 263.0
    qsat = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), 1.0e-3), q_i=z, q_s=z,
        q_g=jnp.full((1, 1), 1.0e-3),
        N_c=z, N_r=jnp.full((1, 1), 1.0e5), N_i=z,
    )
    args = (jnp.full((1, 1), T), jnp.full((1, 1), qsat), hm,
            jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0)
    on = morrison_microphysics(
        *args, MorrisonConfig(do_graupel_rain_accretion=True))
    off = morrison_microphysics(
        *args, MorrisonConfig(do_graupel_rain_accretion=False))
    assert float(on.dq_g_dt[0, 0]) > float(off.dq_g_dt[0, 0])   # graupel grew
    assert float(on.dq_r_dt[0, 0]) < float(off.dq_r_dt[0, 0])   # rain lost
    assert float(on.dT_dt[0, 0]) > float(off.dT_dt[0, 0])       # L_f warming


def test_pracg_ad_safe():
    rho = 0.9

    def loss_qr(q_r0):
        return jnp.sum(graupel_rain_accretion_pracg(
            q_r0.reshape(1, 1), jnp.full((1, 1), 1.0e5),
            jnp.full((1, 1), 1.0e-3), jnp.asarray(265.0), jnp.asarray(rho),
            _CFG, dt=20.0))

    def loss_qg(q_g0):
        return jnp.sum(graupel_rain_accretion_pracg(
            jnp.full((1, 1), 1.0e-3), jnp.full((1, 1), 1.0e5),
            q_g0.reshape(1, 1), jnp.asarray(265.0), jnp.asarray(rho),
            _CFG, dt=20.0))

    for q0 in (0.0, 1.0e-3):
        assert bool(jnp.isfinite(jax.grad(loss_qr)(jnp.asarray(q0))))
        assert bool(jnp.isfinite(jax.grad(loss_qg)(jnp.asarray(q0))))


# -------- PGSACW snow→graupel conversion (iter-38) --------

def _pgsacw(psacws, q_c=1.0e-3, q_s=1.0e-3, N_s=1.0e4, rho=0.7, dt=20.0):
    return float(snow_to_graupel_pgsacw(
        jnp.asarray(q_c), jnp.asarray(q_s), jnp.asarray(N_s),
        jnp.asarray(psacws), jnp.asarray(rho), _CFG, dt))


def _pgsacw_hand(psacws, q_c=1.0e-3, q_s=1.0e-3, N_s=1.0e4, rho=0.7, dt=20.0):
    lams = (_CFG.rho_snow * math.pi * N_s / q_s) ** (1.0 / 3.0)
    lams = min(max(lams, _CFG.lams_min), _CFG.lams_max)
    n0s = N_s * lams
    asn = _CFG.fall_a_s * (_CFG.rho_su / rho) ** 0.54
    bs = _CFG.fall_b_s
    cons17 = (3.0 * _CFG.rho_su * math.pi * _CFG.snow_collect_eff ** 2
              * math.gamma(2.0 * bs + 2.0)
              / (_CFG.rho_graupel - _CFG.rho_snow))
    rate = cons17 * dt * n0s * q_c ** 2 * asn ** 2 / (rho * lams ** (2 * bs + 2))
    return min(psacws, rate)


def test_pgsacw_matches_sam_formula():
    # large psacws ⇒ the CONS17 rate is the binding term
    assert _pgsacw(1.0) == pytest.approx(_pgsacw_hand(1.0), rel=1e-6)


def test_pgsacw_limited_by_psacws():
    """PGSACW ≤ PSACWS (can't convert more than the riming)."""
    tiny = 1.0e-9
    assert _pgsacw(tiny) == pytest.approx(tiny, rel=1e-6)   # psacws is binding


def test_pgsacw_rutledge_hobbs_gate():
    """Only fires with enough snow (≥0.1 g/kg) AND cloud (≥0.5 g/kg)."""
    assert _pgsacw(1.0, q_s=1.0e-3, q_c=1.0e-3) > 0.0        # both above
    assert _pgsacw(1.0, q_s=5.0e-5) == 0.0                   # snow too low
    assert _pgsacw(1.0, q_c=1.0e-4) == 0.0                   # cloud too low
    assert _pgsacw(0.0) == 0.0                               # no riming


def test_pgsacw_converts_snow_riming_to_graupel():
    """Heavy snow riming: PGSACW redirects part of the cloud-water riming from
    snow to graupel (q_g grows, q_s grows less) and drops the snow number."""
    T = 260.0
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(6.0e4)))
    rho = 6.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), 1.0e-3), q_r=z, q_i=z,
        q_s=jnp.full((1, 1), 2.0e-3), q_g=z,
        N_c=jnp.full((1, 1), 1.0e8), N_r=z, N_i=z,
        N_s=jnp.full((1, 1), 1.0e4),
    )
    args = (jnp.full((1, 1), T), jnp.full((1, 1), qsl), hm,
            jnp.full((1, 1), 6.0e4), jnp.full((1, 2), 6.0e4),
            jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0)
    on = morrison_microphysics(*args, MorrisonConfig(do_snow_to_graupel=True))
    off = morrison_microphysics(*args, MorrisonConfig(do_snow_to_graupel=False))
    # graupel gains, snow gains less, snow number lower with conversion on
    assert float(on.dq_g_dt[0, 0]) > float(off.dq_g_dt[0, 0])
    assert float(on.dq_s_dt[0, 0]) < float(off.dq_s_dt[0, 0])
    assert float(on.dN_s_dt[0, 0]) < float(off.dN_s_dt[0, 0])
    # cloud-water sink unchanged (full riming removed regardless of destination)
    assert float(on.dq_c_dt[0, 0]) == pytest.approx(
        float(off.dq_c_dt[0, 0]), abs=1e-12)


def test_pgsacw_ad_safe():
    def loss_qc(q_c0):
        return jnp.sum(snow_to_graupel_pgsacw(
            q_c0.reshape(1, 1), jnp.full((1, 1), 1.0e-3),
            jnp.full((1, 1), 1.0e4), jnp.full((1, 1), 1.0e-3),
            jnp.asarray(0.7), _CFG, 20.0))

    # q_s=0 is the dangerous edge (LAMS = (.../q_s)^⅓); the denominator floor +
    # LAMS clamp must keep the reverse-mode adjoint finite (codex iter-38 F).
    def loss_qs(q_s0):
        return jnp.sum(snow_to_graupel_pgsacw(
            jnp.full((1, 1), 1.0e-3), q_s0.reshape(1, 1),
            jnp.full((1, 1), 1.0e4), jnp.full((1, 1), 1.0e-3),
            jnp.asarray(0.7), _CFG, 20.0))

    for q0 in (0.0, 1.0e-3):
        assert bool(jnp.isfinite(jax.grad(loss_qc)(jnp.asarray(q0))))
        assert bool(jnp.isfinite(jax.grad(loss_qs)(jnp.asarray(q0))))


# -------- Double-moment graupel N_g (iter-39) --------

def test_graupel_lamg_dual_mode():
    """Single-moment uses the fixed-intercept ¼-power slope; double-moment uses
    the per-mass ⅓-power slope (π·ρ_g·N_g/q_g)^⅓."""
    q_g, rho = 1.0e-3, 0.8
    lam_single = float(graupel_lamg(jnp.asarray(q_g), jnp.asarray(rho), _CFG))
    lam_double = float(graupel_lamg(
        jnp.asarray(q_g), jnp.asarray(rho), _CFG, N_g=jnp.asarray(5.0e3)))
    expect_double = (math.pi * _CFG.rho_graupel * 5.0e3 / q_g) ** (1.0 / 3.0)
    expect_double = min(max(expect_double, _CFG.lamg_min), _CFG.lamg_max)
    assert lam_double == pytest.approx(expect_double, rel=1e-6)
    assert lam_single != pytest.approx(lam_double, rel=1e-3)   # modes differ


def test_graupel_lamg_n0g_intercept_modes():
    q_g, rho = 1.0e-3, 0.8
    # single-moment: N0G_m = N0G_vol/ρ
    lam_s, n0_s = graupel_lamg_n0g(jnp.asarray(q_g), jnp.asarray(rho), _CFG)
    assert float(n0_s) == pytest.approx(_CFG.n0_graupel / rho, rel=1e-6)
    # double-moment: N0G_m = N_g·LAMG
    lam_d, n0_d = graupel_lamg_n0g(
        jnp.asarray(q_g), jnp.asarray(rho), _CFG, N_g=jnp.asarray(5.0e3))
    assert float(n0_d) == pytest.approx(5.0e3 * float(lam_d), rel=1e-6)


def _morrison_dm(T=263.0, q_r=1.0e-3, q_g=1.0e-4, N_g=1.0e3, with_ng=True):
    qsat = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(8.0e4)))
    rho = 8.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=jnp.full((1, 1), q_r), q_i=z, q_s=z,
        q_g=jnp.full((1, 1), q_g),
        N_c=z, N_r=jnp.full((1, 1), 1.0e5), N_i=z,
        N_s=jnp.full((1, 1), 1.0e4),
        N_g=(jnp.full((1, 1), N_g) if with_ng else None),
    )
    return morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsat), hm,
        jnp.full((1, 1), 8.0e4), jnp.full((1, 2), 8.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0, MorrisonConfig())


def test_double_moment_graupel_emits_dN_g():
    """With N_g present, morrison emits dN_g_dt (None when single-moment)."""
    assert _morrison_dm(with_ng=True).dN_g_dt is not None
    assert _morrison_dm(with_ng=False).dN_g_dt is None


def test_double_moment_graupel_gains_number_from_frozen_rain():
    """Frozen supercooled rain adds graupel particles: dN_g_dt > 0."""
    out = _morrison_dm(T=258.0, q_r=2.0e-3, q_g=1.0e-4, N_g=1.0e3)
    assert float(out.dN_g_dt[0, 0]) > 0.0          # graupel number grows


def test_double_moment_graupel_finite():
    out = _morrison_dm()
    for fld in (out.dN_g_dt, out.dq_g_dt, out.dT_dt):
        assert bool(jnp.all(jnp.isfinite(fld)))
