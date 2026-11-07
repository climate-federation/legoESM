"""SAM M2005 ice depositional-growth PRD — algebraic M2005-form pin.

gSAM ``MICRO_M2005`` (module_mp_graupel.f90:3427-3514) grows/sublimates the
cloud-ice PSD by bulk vapour diffusion::

    LAMI = (CONS12·N_i/q_i)^(1/3),  CONS12 = GAMMA(1+DI)·CI = rho_ci·pi   (:578, :2549)
    N0I  = N_i·LAMI                                                        (:2551)
    EPSI = 2*pi*N0I*rho*DV/LAMI^2 = (2*pi/CONS12^(1/3))*rho*DV*N_i^(2/3)*q_i^(1/3)  (:3427)
    DV   = 8.794e-5 * T^1.81 / p                                           (:1573)
    ABI  = 1 + DQSIDT*L_s/c_p,  DQSIDT = L_s*q_sat_i/(R_v*T^2)             (:1589)
    PRD  = EPSI * (q_v - q_sat_i) / ABI                                    (:3470/3480)

legoesm's ``morrison.py`` (ice_deposition_scheme="m2005", :405-417) implements
this SAME EPSI/DV/ABI/CONS12 ALGEBRA (verified against the on-disk gSAM oracle),
so these tests pin that the JAX code evaluates the gSAM M2005 PRD FORM correctly.

Scope of the pin (what it is NOT): it is an ALGEBRAIC-form pin under legoESM's own
thermodynamics, not a bit-for-bit gSAM numeric reference. The oracle here reuses
legoESM's ``saturation_specific_humidity_ice`` (constant-L_s Clausius-Clapeyron), which
differs from gSAM's Flatau ``POLYSVP`` mixing-ratio curve by ~0.3% at 250 K / 40 kPa
(and legoESM's ``R_v`` differs from gSAM's in the last significant digit). Departures
beyond that (all interior to this fixture, so the pin stays exact for the CODE): the
``ice_deposition_efficiency`` multiplier (gSAM has none), the ``q_i_min_growth`` floor
+ non-negative clips, the optional homogeneous-nucleation EPSI boost, and gSAM's
pre-EPSI LAMI bounds / N0I-NI3D update (not applied in legoESM's PRD block).

The sibling suites pin KK2000 auto/accretion, PRE (rain evap) and PRCI (ice->snow)
against their SAM closed forms; ``test_m2005_ice_deposition.py`` covers PRD's SIGN
and SCALING but NOT its absolute value. This file adds the missing ABSOLUTE pin
(the ``_prd_formula`` companion), so the M2005-form diffusional-growth claim at
morrison.py:394 has a closed-form test behind it.

Isolation: with ``q_c=q_r=q_s=q_g=0``, ``N_i0=0`` (Cooper off), ``do_graupel``
off and ``q_v`` ice-supersaturated but liquid-SUBsaturated, the vapour budget
``dq_v_dt = -cond + evap - dq_i_dep - dq_i_nuc - prds - prdg`` collapses to
``dq_v_dt = -dq_i_dep = -PRD`` (PRCI ice->snow is a mass transfer, not a vapour
sink). So ``-dq_v_dt`` observes PRD directly.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.morrison import (
    _DV_PREFACTOR,
    _DV_T_EXPONENT,
    morrison_microphysics,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import (
    saturation_specific_humidity,
    saturation_specific_humidity_ice,
)

jax.config.update("jax_enable_x64", True)

_RHO_CI = 500.0  # cloud-ice bulk density [kg/m^3] (MorrisonConfig.rho_cloud_ice)


def _neg_dqv(T, p, q_v, q_i, N_i, eff, dt=20.0):
    """Return -dq_v_dt for a single ice-only cell (= PRD when isolated)."""
    rho = p / (constants.R_d * T)
    z = jnp.zeros((1, 1))
    hm = HydrometeorState(
        q_c=z, q_r=z, q_i=jnp.full((1, 1), q_i), q_s=z, q_g=z,
        N_c=z, N_r=z, N_i=jnp.full((1, 1), N_i),
    )
    out = morrison_microphysics(
        jnp.full((1, 1), T), jnp.full((1, 1), q_v), hm,
        jnp.full((1, 1), p), jnp.full((1, 2), p),
        jnp.full((1, 1), rho), jnp.full((1, 1), 300.0), dt,
        MorrisonConfig(
            ice_deposition_scheme="m2005", ice_deposition_efficiency=eff,
            N_i0=0.0, do_graupel=False, morrison_flavor="sam",
            # Bulletproof the -dq_v_dt == PRD isolation, incl. the joint ice
            # limiter qi_scale (morrison.py:696-719) that rescales the
            # sublimation branch of dq_i_dep against its qi_sink_total (:710):
            # melt_rate=0 kills the melt sink and PRCI is 0 for negative supersat
            # (deposition-only formula), so |PRD| is the ONLY sink entering
            # qi_scale here => qi_scale==1. (Ice SEDIMENTATION, :1060, is a
            # separate q_i sink that does NOT enter qi_scale and does NOT touch
            # the vapour budget, so it is irrelevant to -dq_v_dt.) homogeneous
            # off keeps the default EPSI (no boost/cap).
            melt_rate=0.0, ice_to_snow_scheme="m2005_autoconv",
            homogeneous_ice_nucleation=False,
        ),
    )
    return -float(out.dq_v_dt[0, 0])


def _prd_formula(T, p, q_v, q_i, N_i, eff):
    """Independent NumPy PRD (gSAM MICRO_M2005), literal constants."""
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    rho = p / (constants.R_d * T)
    dv = 8.794e-5 * T ** 1.81 / p
    dqsidt = constants.L_s * qsi / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    cons12_cbrt = (_RHO_CI * math.pi) ** (1.0 / 3.0)
    epsi = (2.0 * math.pi / cons12_cbrt) * rho * dv * N_i ** (2.0 / 3.0) * q_i ** (1.0 / 3.0)
    return eff * epsi * (q_v - qsi) / abi


# ---------------------------------------------------------------------------
# Absolute closed-form pins
# ---------------------------------------------------------------------------

def test_prd_deposition_matches_sam_formula():
    """Ice-supersaturated (liquid-subsaturated) cell: -dq_v_dt == PRD > 0."""
    T, p, q_i, N_i = 250.0, 4.0e4, 1.0e-4, 1.0e5
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    qsl = float(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))
    q_v = 1.1 * qsi
    assert qsi < q_v < qsl, "q_v must be ice-supersat but liquid-subsat (no condensation)"

    got = _neg_dqv(T, p, q_v, q_i, N_i, eff=1.0)
    exp = _prd_formula(T, p, q_v, q_i, N_i, eff=1.0)
    assert exp > 0.0 and got > 0.0                    # non-vacuous: real deposition
    assert math.isclose(got, exp, rel_tol=1e-6), f"got {got:.6e} vs PRD {exp:.6e}"


def test_prd_sublimation_matches_sam_formula():
    """Mildly ice-subsaturated cell with ample ice (donor/joint clamps inactive):
    -dq_v_dt == PRD < 0 (sublimation, vapour source)."""
    T, p, q_i, N_i, dt = 250.0, 4.0e4, 1.0e-3, 1.0e5, 20.0
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    q_v = 0.98 * qsi                                   # 2% ice-subsaturated
    exp = _prd_formula(T, p, q_v, q_i, N_i, eff=1.0)
    # Donor clamp (subl_neg=max(min(dep_raw,0),-q_i/dt)) AND the joint qi_scale
    # limiter are inactive when |PRD|*dt << q_i: with melt_rate=0 and PRCI=0
    # (negative supersat), |PRD| is the only sink entering qi_scale, so this
    # bound proves qi_scale==1. (Sedimentation does not enter either and does
    # not affect dq_v_dt.)
    assert abs(exp) * dt < 0.1 * q_i, "sublimation clamp would engage; raise q_i"
    got = _neg_dqv(T, p, q_v, q_i, N_i, eff=1.0, dt=dt)
    assert exp < 0.0 and got < 0.0                     # non-vacuous: real sublimation
    assert math.isclose(got, exp, rel_tol=1e-6), f"got {got:.6e} vs PRD {exp:.6e}"


# ---------------------------------------------------------------------------
# Departure canaries
# ---------------------------------------------------------------------------

def test_prd_efficiency_is_a_linear_departure_from_gsam():
    """``ice_deposition_efficiency`` is an ADDITIONAL explicit linear multiplier
    on the gSAM PRD (gSAM has no such factor, i.e. eff=1) -- one of several
    departures (see the module docstring: floors, clips, thermodynamics, LAMI
    bounds), not the only one. Halving eff exactly halves the deposition rate,
    which pins that it enters as a linear prefactor."""
    T, p, q_i, N_i = 250.0, 4.0e4, 1.0e-4, 1.0e5
    qsi = float(saturation_specific_humidity_ice(jnp.asarray(T), jnp.asarray(p)))
    q_v = 1.1 * qsi
    full = _neg_dqv(T, p, q_v, q_i, N_i, eff=1.0)
    half = _neg_dqv(T, p, q_v, q_i, N_i, eff=0.5)
    assert full > 0.0
    assert math.isclose(half, 0.5 * full, rel_tol=1e-9)


def test_dv_diffusivity_constants_match_sam():
    """The Hall-Pruppacher vapour-diffusivity constants used by PRD are the
    gSAM values ``DV = 8.794e-5 * T^1.81 / p`` (module_mp_graupel.f90:1573).
    Closes the loop between the literal oracle above and the module constants."""
    assert _DV_PREFACTOR == 8.794e-5
    assert _DV_T_EXPONENT == 1.81
