"""Seifert-Beheng two-moment warm-rain microphysics.

A two-moment scheme tracking mass and number concentration of cloud
droplets and rain drops. Processes: saturation adjustment, autoconversion
(mass-dependent), accretion, self-collection, breakup, rain evaporation,
and sedimentation.

All operations use smooth (differentiable) approximations.

Faithfulness
------------
This module's warm-rain rates are SIMPLIFIED smooth proxies, NOT the published
SB2001 universal functions, and the class of departure of each process is:

- Autoconversion (``autoconversion_sb``): ``k_au·q_c^2·sigmoid(x_c/x_*−1)·rho``
  — DEPARTURE. The published form is ``PRC ∝ q_c^4·N_c^-2·(1+phi_au/(1−tau)^2)``
  with the universal function ``phi_au(tau)=600·tau^0.68·(1−tau^0.68)^3``
  (tau = rain-water fraction). The proxy has neither ``phi_au`` nor the q_c^4
  scaling — it is a mean-mass-gated ``q_c^2`` law.
- Accretion (``accretion``): ``k_ac·rho·q_c·q_r`` — DEPARTURE. The published
  form multiplies by ``phi_ac(tau)=(tau/(tau+5e-4))^4`` and uses the fixed
  5.78 kernel; the proxy is bilinear with no tau suppression.
- Self-collection/breakup (``self_collection_breakup``): sigmoid-breakup proxy
  — DEPARTURE from the SB2001 NRAGG ``2−exp(2300·(1/lambda−300µm))`` form.
- Rain-number closure ``dN_r_au = au/x_*`` — FAITHFUL (SB2001 newborn-drop
  separation mass).

The PUBLISHED SB2001 universal functions ARE available in the model:
``autoconversion_sb2001`` / ``accretion_sb2001`` in ``_warm_rain.py`` transcribe
the gSAM IRAIN=1 MASS rates (PRC, PRA) plus the SB2001 rain-NUMBER closure
(au/x_*), oracle-pinned by ``tests/unit/test_seifert_beheng_faithful.py`` and
reachable through Morrison via
``MorrisonConfig(warm_rain_scheme="seifert_beheng_sb2001")``;
``self_collection_breakup_sb2001`` implements the SB2001 NRAGG functional form
(before its explicit-Euler stability safeguards — a lambda clamp and an
equilibrium non-overshoot step limiter, not the raw Fortran rate). Two
faithfulness gaps remain, both documented and both moot at the shipped default
``predict_Nc=False``: (1) the SB2001-SPECIFIC cloud-NUMBER autoconversion sink
``2·PRC·rho/x_*`` is not represented (under ``predict_Nc=True`` Morrison applies
a GENERIC ``-PRC·rho/x_c`` instead; no NPRA accretion sink for any path);
(2) ``nu`` is a fixed shape, not gSAM's spatially-diagnosed pgam. Wiring a selectable faithful form into THIS
standalone module (and validating it through the RCE-realism / AMIP gates before
making it the default) is a documented follow-up — the default here stays the
simplified proxy so shipped climate behaviour is unchanged.

References
----------
- Seifert, A., & Beheng, K. D. (2001). A double-moment parameterization for
  simulating autoconversion, accretion and selfcollection. Atmos. Res.,
  59-60, 265-281.  [the warm-rain universal functions phi_au / phi_ac]
- Seifert, A., & Beheng, K. D. (2006). A two-moment cloud microphysics
  parameterization for mixed-phase clouds. Part 1: Model description.
  Meteorol. Atmos. Phys., 92, 45-66.  [the two-moment framework]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
    safe_pow,
    donor_clamp_scale,
)


# Fixed SB fall-speed constants.
_RHO_FLOOR = 0.1
_VT_CLIP_RAIN = 20.0

__physics_contract__ = {
    "summary": (
        "SB-inspired two-moment warm-rain microphysics: prognostic cloud + rain "
        "mass AND number, with saturation adjustment, autoconversion, accretion, "
        "self-collection, breakup, rain evaporation and size-sorted sedimentation "
        "to surface precipitation. The rate CLOSURES are SIMPLIFIED smooth proxies "
        "(NOT the published SB2001 universal functions phi_au/phi_ac) — see the "
        "module 'Faithfulness' docstring; the faithful forms are selectable via "
        "Morrison's warm_rain_scheme='seifert_beheng_sb2001'."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.N_c": "1/m^3",
        "hydrometeors.N_r": "1/m^3", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt is consistent with the "
        "vapour<->liquid phase change via L_v. Mass and number are redistributed "
        "cloud<->rain; SURFACE PRECIPITATION (>= 0) removes water, so column "
        "moisture is NOT conserved -- no contract-level conservation is claimed. "
        "Mixing ratios and number concentrations stay >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Seifert & Beheng (2001), Atmos. Res. 59-60, 265-281",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py — a cloudy two-moment column "
        "autoconverts to rain and precipitates; number concentrations respond "
        "to self-collection/breakup; all masses and numbers stay >= 0; "
        "differentiable wrt T, q_v, N_c."
    ),
}


def seifert_beheng_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: SeifertBehengConfig = SeifertBehengConfig(),
) -> MicrophysicsOutput:
    """Compute Seifert-Beheng two-moment warm-rain tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
    # donor-clamped — see _warm_rain.saturation_adjustment.
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, dt, sharpness, q_c=q_c,
    )

    # 1. Autoconversion (mass-dependent).  Use the autoconversion-
    # specific sharpness so the normalised-argument sigmoid (iter-97)
    # is not driven 100× too steep by ``saturation_sharpness`` (which
    # is calibrated for kg/kg-scale ``excess``, not for the
    # dimensionless ``x_c/x_star − 1``).
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star,
        config.autoconversion_sharpness,
    )

    # 2. Accretion
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)

    # 3-4. Self-collection and breakup
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )

    # 5. Rain evaporation
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff, dt=dt)

    # === Joint donor clamp on q_c sinks ===
    # ``saturation_adjustment`` already donor-clamps the evaporation
    # branch in isolation (negative ``condensation`` ≥ ``-q_c/dt``),
    # but the COMBINED rate ``-condensation_evap + dq_c_au + dq_c_ac``
    # can still exceed ``q_c/dt`` and drive ``q_c`` negative AND
    # break total-water conservation.  Probe with q_c=1e-4, q_r=5e-3,
    # 80 % RH at 290 K: SB drove q_c → -4e-3 over 1200 s and lost
    # 5.5e-3 kg/kg of total water.  Mirror Morrison/Thompson's joint
    # clamp (Codex audit cycle 2 + follow-up).
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    qc_sink_total = cond_evap_sink + dq_c_au + dq_c_ac
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    condensation = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )
    # The autoconverted droplet number ``dN_r_au`` must scale
    # consistently to preserve mass-per-droplet ``x_star``.
    dN_r_au = dN_r_au * qc_scale

    # 6. Sedimentation — Marshall-Palmer fall speed (q_r * rho/rho_sfc)^b_v_r
    # has fractional exponent (b_v_r=0.5); guard the AD path with safe_pow.
    rho_sfc = rho[:, -1:]
    V_t_r = config.a_v_r * safe_pow(
        jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, _RHO_FLOOR), config.b_v_r,
    )
    V_t_r = jnp.clip(V_t_r, 0.0, _VT_CLIP_RAIN)
    # Joint q_r donor cap: pass evaporation as ``extra_sink`` so sed +
    # evap together cannot remove more rain than is locally available
    # (codex iter-29 #1).
    sed_r, precipitation = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )

    # 7. Latent heating
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd

    # Combine tendencies — joint-scaled sinks/sources conserve total
    # water (q_v + q_c + q_r) per layer (modulo rain-evap exchange
    # with q_v and sedimentation).
    dq_v_dt = -condensation + evaporation
    dq_c_dt = condensation - dq_c_au - dq_c_ac
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + sed_r
    # AD-safe N_c tendency (issue #249).  ``eps=1e-15`` sits well below
    # the physical droplet-mass scale (``x_c ≈ 1e-15 kg`` at
    # ``q_c=1e-7 kg/kg``) and 5 decades above the prior ``clip(x_c,
    # 1e-20)`` floor; the masked-out residue is unphysical.  See
    # ``morrison.py`` for the derivation.
    dN_c_dt = safe_divide(-dq_c_au * rho, x_c, eps=1e-15)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br

    # === Non-negativity floor on the NUMBER tendencies ===
    # ``dN_c_dt`` (autoconversion droplet sink) and the rain-number
    # self-collection sink ``dN_r_sc`` are explicit and proportional to
    # the current number. Unbounded, a single Euler step can overshoot
    # the available number and drive ``N_c``/``N_r`` negative; the
    # self-collection rate then flips sign and the number runs away
    # exponentially (observed in an RCE plane-CRM restart: ``N_r`` →
    # −4e7 within ~400 steps, growing ~3.4x/step). A negative number is
    # unphysical and corrupts every PSD-derived rate (mean drop mass
    # ``x = q·ρ/N``, slope ``λ``, fall speed). Cap the NET sink so the
    # post-step number cannot fall below zero — sources (positive
    # tendencies) pass through unchanged. This mirrors Morrison's
    # ``n_s_new``/``n_g_new`` consistency limiter for snow/graupel.
    dN_c_dt = jnp.maximum(dN_c_dt, -jnp.clip(N_c, 0.0) / dt)
    dN_r_dt = jnp.maximum(dN_r_dt, -jnp.clip(N_r, 0.0) / dt)

    # Precipitation now comes from the dt-limited bottom flux returned
    # by ``sedimentation_tendency`` so column water conservation holds
    # exactly when the CFL limiter fires.

    # Pin dtype to the input precision so we never silently promote
    # the unused-species placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=z,
        precipitation=precipitation,
    )
