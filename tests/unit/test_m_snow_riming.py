"""SAM M2005 snow riming of cloud water PSACWS tests (iter-28).

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:2909): falling snow collects cloud
droplets, which freeze onto it (riming)::

    PSACWS = Γ(BS+3)·π/4·ECI · ASN · q_c·ρ · N0S / LAMS^(BS+3)      (ECI=0.7)

legoESM previously used a crude ``rime_coeff·q_s·q_c·f_ice``. PSACWS is the
PSD-integrated collection — the dominant mixed-phase snow growth + glaciation.
Requires double-moment snow (N_s).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    snow_riming_psacws,
)
from legoesm.atmosphere.physics.microphysics.morrison import (
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


jax.config.update("jax_enable_x64", True)
_CFG = MorrisonConfig()


def _ps(q_c=5.0e-4, q_s=1.0e-3, N_s=1.0e4, T=260.0, p=5.0e4, dt=None):
    rho = p / (constants.R_d * T)
    return float(snow_riming_psacws(
        jnp.asarray(q_c), jnp.asarray(q_s), jnp.asarray(N_s),
        jnp.asarray(T), jnp.asarray(rho), _CFG, dt=dt))


def _hand(q_c=5.0e-4, q_s=1.0e-3, N_s=1.0e4, T=260.0, p=5.0e4):
    rho = p / (constants.R_d * T)
    lams = (_CFG.rho_snow * math.pi * N_s / q_s) ** (1.0 / 3.0)
    lams = min(max(lams, _CFG.lams_min), _CFG.lams_max)
    n0s = N_s * lams
    asn = _CFG.fall_a_s * (_CFG.rho_su / rho) ** 0.54
    bs = _CFG.fall_b_s
    cons13 = math.gamma(bs + 3.0) * math.pi / 4.0 * _CFG.snow_collect_eff
    rime_frac = 1.0 / (1.0 + math.exp(-_CFG.melt_sharpness
                                      * (constants.T_freeze - T)))
    return cons13 * asn * q_c * rho * n0s / lams ** (bs + 3.0) * rime_frac


def test_psacws_matches_sam_formula():
    assert _ps() == pytest.approx(_hand(), rel=1e-6)


def test_more_snow_rimes_faster():
    """More snow ⇒ more collecting surface ⇒ faster riming."""
    assert _ps(q_s=5.0e-3) > _ps(q_s=1.0e-3)


def test_no_riming_above_freezing():
    """Riming is supercooled droplets freezing on snow ⇒ ~0 above 0 °C."""
    assert _ps(T=280.0) < 1.0e-3 * _ps(T=260.0)


def test_no_riming_without_snow_or_cloud():
    assert _ps(q_s=0.0, N_s=0.0) == 0.0
    assert _ps(q_c=0.0) == 0.0


def test_psacws_donor_clamped():
    assert _ps(q_c=1.0e-5, q_s=5.0e-3, dt=20.0) <= 1.0e-5 / 20.0 + 1e-18


def _morrison_diff(do_rime, T=260.0):
    qsl = float(saturation_mixing_ratio(jnp.asarray(T), jnp.asarray(5.0e4)))
    rho = 5.0e4 / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=jnp.full((1, 1), 5.0e-4), q_r=z, q_i=z,
        q_s=jnp.full((1, 1), 1.0e-3), q_g=z, N_c=z, N_r=z, N_i=z,
        N_s=jnp.full((1, 1), 1.0e4),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), qsl), hm,
        jnp.full((1, 1), 5.0e4), jnp.full((1, 2), 5.0e4),
        jnp.full((1, 1), rho), jnp.full((1, 1), 1.0e9), 20.0,
        # Disable PGSACW so the riming goes wholly to snow (this test isolates
        # the snow-riming→snow conservation; iter-38 added snow→graupel which
        # would otherwise split the riming between q_s and q_g).
        MorrisonConfig(do_snow_riming=do_rime, do_snow_to_graupel=False))
    return (float(out.dq_c_dt[0, 0]), float(out.dq_s_dt[0, 0]),
            float(out.dT_dt[0, 0]))


def test_riming_conserves_water_and_releases_latent_heat():
    """on−off difference: cloud water → snow (Δdq_c = −Δdq_s), L_f warming."""
    dqc_on, dqs_on, dT_on = _morrison_diff(True)
    dqc_off, dqs_off, dT_off = _morrison_diff(False)
    assert (dqc_on - dqc_off) == pytest.approx(-(dqs_on - dqs_off), abs=1e-12)
    assert (dqc_on - dqc_off) < 0.0       # cloud water collected
    assert dT_on > dT_off                 # L_f warming


def test_psacws_ad_safe():
    def loss(q_c0):
        rho = 5.0e4 / (constants.R_d * 260.0)
        return jnp.sum(snow_riming_psacws(
            q_c0.reshape(1, 1), jnp.full((1, 1), 1.0e-3),
            jnp.full((1, 1), 1.0e4), jnp.asarray(260.0),
            jnp.asarray(rho), _CFG, dt=20.0))

    for q0 in (0.0, 5.0e-4):
        g = jax.grad(loss)(jnp.asarray(q0))
        assert bool(jnp.isfinite(g))
