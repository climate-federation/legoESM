"""SAM M2005 snow vapor deposition/sublimation PRDS tests (iter-27).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:2039/3476) grows/sublimates SNOW by
bulk diffusion of the snow PSD::

    EPSS = 2π·N0S·ρ·DV·[F1S/LAMS² + F2S·CONS10·(ASN·ρ/μ)^½·SC^⅓·LAMS^(−CONS35)]
    PRDS = EPSS·(q_v−q_sat_i)/ABI       (>0 deposition, <0 sublimation)

legoESM previously deposited vapour only onto cloud ice (PRD), never snow —
so the anvil/stratiform snow could not grow by deposition and dry-downdraft
snow sublimation cooling was absent. Requires double-moment snow (N_s).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    snow_deposition_m2005,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio_ice


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _prds(ssat, q_s=1.0e-3, N_s=1.0e4, T=240.0, p=3.0e4, dt=None):
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    return float(snow_deposition_m2005(
        jnp.asarray((1.0 + ssat) * qsi), jnp.asarray(q_s), jnp.asarray(N_s),
        jnp.asarray(qsi), jnp.asarray(T), jnp.asarray(p),
        jnp.asarray(rho), _CFG, dt=dt))


def _hand(ssat, q_s=1.0e-3, N_s=1.0e4, T=240.0, p=3.0e4):
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    lams = (_CFG.rho_snow * math.pi * N_s / q_s) ** (1.0 / 3.0)
    lams = min(max(lams, _CFG.lams_min), _CFG.lams_max)
    n0s = N_s * lams
    dv = 8.794e-5 * T ** 1.81 / p
    mu = 1.496e-6 * T ** 1.5 / (T + 120.0)
    sc = mu / (rho * dv)
    asn = _CFG.fall_a_s * (_CFG.rho_su / rho) ** 0.54
    cons35 = 2.5 + _CFG.fall_b_s / 2.0
    cons10 = math.gamma(cons35)
    dqsidt = constants.L_s * qsi / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    epss = (2 * math.pi * n0s * rho * dv
            * (_CFG.snow_vent_f1 / lams ** 2
               + _CFG.snow_vent_f2 * cons10 * (asn * rho / mu) ** 0.5
               * sc ** (1.0 / 3.0) * lams ** (-cons35)))
    return epss * (ssat * qsi) / abi


def test_prds_matches_sam_formula():
    assert _prds(0.1) == pytest.approx(_hand(0.1), rel=1e-6)


def test_deposition_grows_snow_when_supersaturated():
    assert _prds(0.1) > 0.0
    assert _prds(0.2) > _prds(0.1)        # more supersat ⇒ faster


def test_sublimation_when_subsaturated():
    assert _prds(-0.2) < 0.0              # snow sublimates


def test_sublimation_donor_clamped():
    """Sublimation cannot exceed q_s/dt."""
    r = _prds(-0.9, q_s=1.0e-5, dt=20.0)
    assert r >= -1.0e-5 / 20.0 - 1e-18 and r < 0.0


def test_no_snow_deposition_without_snow():
    assert _prds(0.1, q_s=0.0, N_s=0.0) == 0.0


def _morrison_diff(do_prds, ssat=0.1, T=240.0):
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(T), jnp.asarray(3.0e4)))
    rho = 3.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=jnp.full((1, 1), 1.0e-3), q_g=z,
        N_c=z, N_r=z, N_i=z, N_s=jnp.full((1, 1), 1.0e4),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), (1.0 + ssat) * qsi), hm,
        jnp.full((1, 1), 3.0e4), jnp.full((1, 2), 3.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9),   # huge dz ⇒ no sed
        20.0, MorrisonConfig(N_i0=0.0, do_snow_deposition=do_prds))
    return (float(out.dq_s_dt[0, 0]), float(out.dq_v_dt[0, 0]),
            float(out.dT_dt[0, 0]))


def test_deposition_conserves_water_and_releases_latent_heat():
    """on−off difference isolates PRDS: q_v → q_s (Δdq_s = −Δdq_v) with the
    deposition latent heat (L_s) warming."""
    dqs_on, dqv_on, dT_on = _morrison_diff(True)
    dqs_off, dqv_off, dT_off = _morrison_diff(False)
    assert (dqs_on - dqs_off) == pytest.approx(-(dqv_on - dqv_off), abs=1e-12)
    assert (dqs_on - dqs_off) > 0.0       # snow grew
    assert dT_on > dT_off                 # L_s warming


def test_sublimation_cools():
    """Sublimating snow (subsaturated) ABSORBS L_s ⇒ cooling (the downdraft
    cooling missing before)."""
    _, _, dT_on = _morrison_diff(True, ssat=-0.2)
    _, _, dT_off = _morrison_diff(False, ssat=-0.2)
    assert dT_on < dT_off


def test_single_moment_has_no_snow_deposition():
    """N_s=None ⇒ no PRDS (snow deposition needs the PSD)."""
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(240.0), jnp.asarray(3.0e4)))
    rho = 3.0e4 / (constants.R_d * 240.0)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=z, q_s=jnp.full((1, 1), 1.0e-3), q_g=z,
        N_c=z, N_r=z, N_i=z,                       # N_s=None default
    )
    out = morrison_microphysics(
        jnp.full((1, 1), 240.0), jnp.full((1, 1), 1.1 * qsi), hm,
        jnp.full((1, 1), 3.0e4), jnp.full((1, 2), 3.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0,
        MorrisonConfig(N_i0=0.0))
    # snow tendency comes only from sedimentation (~0 with huge dz), no growth.
    assert abs(float(out.dq_s_dt[0, 0])) < 1.0e-10


def test_prds_ad_safe():
    qsi = float(saturation_mixing_ratio_ice(jnp.asarray(240.0), jnp.asarray(3.0e4)))

    def loss(q_s0):
        rho = 3.0e4 / (constants.R_d * 240.0)
        return jnp.sum(snow_deposition_m2005(
            jnp.asarray(1.1 * qsi), q_s0.reshape(1, 1),
            jnp.full((1, 1), 1.0e4), jnp.asarray(qsi), jnp.asarray(240.0),
            jnp.asarray(3.0e4), jnp.asarray(rho), _CFG, dt=20.0))

    for q0 in (0.0, 1.0e-3):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
