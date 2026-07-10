"""Kessler warm-rain microphysics column backend.

A simple one-moment warm-rain scheme tracking cloud water and rain.
Processes: saturation adjustment, autoconversion, accretion, evaporation,
rain sedimentation, and latent heating.

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water
  Substance in Atmospheric Circulations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    rain_evaporation,
    safe_pow,
    donor_clamp_scale,
    saturation_adjustment,
)
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


# Kessler accretion exponent + density floor (fixed).
_KESSLER_ACCR_EXP = 0.875
_RHO_FLOOR = 0.1

__physics_contract__ = {
    "summary": (
        "Kessler (1969) single-moment warm-rain microphysics: saturation "
        "adjustment (cloud condensation/evaporation), cloud->rain "
        "autoconversion, accretion, rain evaporation and rain sedimentation to "
        "surface precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt is consistent with the "
        "vapour<->liquid phase-change rate via L_v (condensation warms, "
        "evaporation cools). Water is redistributed vapour<->cloud<->rain; "
        "SURFACE PRECIPITATION (>= 0) removes water from the column, so column "
        "moisture is NOT conserved and the falling condensate carries enthalpy "
        "out -- no contract-level conservation is claimed. Mixing ratios >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Kessler (1969), Meteorol. Monogr. 10(32), Amer. Meteorol. Soc.",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — supersaturated air condenses "
        "with L_v heating; cloud above the autoconversion threshold makes rain "
        "+ surface precip; subsaturated rain evaporates; mixing ratios stay "
        ">= 0; differentiable wrt T, q_v."
    ),
}


def kessler_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: KesslerConfig = KesslerConfig(),
) -> MicrophysicsOutput:
    """Compute Kessler microphysics tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    hydrometeors : HydrometeorState
        Hydrometeor state (only q_c, q_r used).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : KesslerConfig

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    sharpness = config.saturation_sharpness

    # 1. Saturation adjustment — the SHARED ``_warm_rain.saturation_adjustment``
    # helper computes ``q_sat`` and the supersaturation residual (in fp64 when
    # x64 is available — the issue #618 robustness path the fp32 LES/CRM
    # production config relies on; an inline fp32 copy here re-exposed the
    # ~65 % LWP deficit #618 fixed), applies the psychrometric correction, and
    # donor-clamps the evaporation branch against ``q_c``.  Kessler EXTENDS it
    # with the previously MISSING positive-branch bound: condensation is
    # donor-clamped by the available ``q_v``.  Without that clamp the coupled
    # driver floors ``q_v`` independently while keeping the full ``q_c``
    # increment, so saturation adjustment created cloud water from vapour that
    # was floored away → ``q_c`` accumulated to physically impossible
    # ~1 kg/kg (opaque clouds, planetary-albedo runaway, the coupled cold
    # drift / OLR collapse).
    _dt_safe = jnp.maximum(dt, 1e-10)
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, _dt_safe, sharpness, q_c=q_c,
    )
    q_v_avail = jnp.clip(q_v, 0.0, None)
    condensation = jnp.minimum(condensation, q_v_avail / _dt_safe)

    dq_v_sat = -condensation
    dq_c_sat = condensation

    # 2. Autoconversion: cloud -> rain (threshold excess).  Uses raw
    # ``q_c`` so the joint donor clamp below scales it consistently
    # with the other ``q_c`` sinks (an earlier form used
    # ``q_c_updated = q_c + cond·dt`` here, which double-counted the
    # condensation budget when joint sinks were active and made the
    # joint clamp inconsistent).
    autoconv = config.autoconversion_rate * jnp.maximum(
        q_c - config.autoconversion_threshold, 0.0
    )

    # 3. Accretion: cloud collected by rain.  Fractional powers of q_r
    # have unbounded derivative at q_r=0 — safe_pow handles the AD guard.
    accretion = config.accretion_coeff * q_c * safe_pow(q_r, _KESSLER_ACCR_EXP)

    # 4. Evaporation of rain (q_r^0.525) via the shared donor-limited
    # helper — bounds ``evap·dt ≤ q_r`` so one explicit step can't
    # evaporate more rain than exists.  Centralises the formula so the
    # four schemes (kessler/morrison/thompson/seifert_beheng) share
    # one implementation.
    evaporation = rain_evaporation(
        q_v, q_r, q_sat, config.evaporation_coeff, dt=dt,
    )

    # === Joint donor clamp on q_c sinks ===
    # Saturation-evaporation, autoconversion, and accretion can each
    # individually be donor-clamped, but the *combined* sink rate can
    # still exceed ``q_c / dt`` and drive ``q_c`` negative on an
    # explicit Euler step.  E.g. with ``q_c = 1e-4``, ``q_r = 5e-3``,
    # 80 % RH at 290 K, accretion + saturation-evap together over a
    # 1200-s step demanded ~26× the available q_c (Codex audit cycle
    # 2 / cycle 3 follow-up).  Mirror the Morrison/Thompson joint
    # donor clamp so total water is conserved when sinks compete.
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    qc_sink_total = cond_evap_sink + autoconv + accretion
    qc_avail = jnp.clip(q_c, 0.0)
    # Shared AD-safe donor clamp helper.  Replaces the previous
    # boolean ``sink_active`` double-where with an explicit
    # ``divisor_floor=1e-15`` that bounds the worst-case VJP under
    # fp32 to ``-q / 1e-30 ≈ -1e30`` (safely within fp32 dynamic
    # range).
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    autoconv = autoconv * qc_scale
    accretion = accretion * qc_scale
    # Scale the evaporation branch only; positive condensation
    # (saturation adjustment from supersaturation) is unaffected.
    condensation = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )
    dq_v_sat = -condensation
    dq_c_sat = condensation

    # 5. Rain sedimentation
    rho_sfc = rho[:, -1:]
    V_t = config.rain_fall_speed * jnp.sqrt(
        rho_sfc / jnp.clip(rho, _RHO_FLOOR)
    )
    # Joint q_r donor cap: pass evaporation as ``extra_sink`` so the
    # sedimentation flux limiter accounts for the rain evaporation that
    # also removes q_r in the same step.  Codex iter-29 #1.
    sed_tend, precipitation = sedimentation_tendency(
        q_r, rho, V_t, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )

    # 6. Latent heating
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd

    # Combine tracer tendencies — sources match the scaled sinks so
    # total water (q_v + q_c + q_r) is conserved per layer (modulo
    # rain evaporation, which exchanges with q_v, and sedimentation,
    # which redistributes q_r vertically).
    dq_v_dt = dq_v_sat + evaporation
    dq_c_dt = dq_c_sat - autoconv - accretion
    dq_r_dt = autoconv + accretion - evaporation + sed_tend

    # Precipitation comes from the dt-limited surface flux returned by
    # ``sedimentation_tendency`` so column water conservation holds
    # exactly when the CFL limiter fires.  An earlier formulation
    # diagnosed precipitation as ``q_r_bot · ρ · V_t``, which exceeds
    # the actual amount removed from the column whenever
    # ``V_t·dt/dz_bot > 1``.

    # Pin dtype to the input precision so we never silently promote
    # the unused-tendency placeholders to f64 under x64 mode.
    _dtype = T.dtype
    z = jnp.zeros((ncol, nlev), dtype=_dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
        # Positive saturation-adjustment condensation (the q_v sink that becomes
        # cloud water) for the coupled pipeline's joint vapour donor clamp.
        dq_v_to_qc_dt=jnp.maximum(dq_c_sat, 0.0),
    )
