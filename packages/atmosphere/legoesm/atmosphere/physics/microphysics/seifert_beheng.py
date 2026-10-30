"""Seifert-Beheng two-moment warm-rain microphysics, as gSAM M2005 runs it.

At the hard limit of its three ``gsam_satadj_*`` fields (defaults smooth, RULE 5)
the scheme IS gSAM 1.8.8's liquid-only Morrison (2005) microphysics with the
Seifert & Beheng (2001) warm rain: ``micro_proc`` (SRC/MICRO_M2005/
microphysics.f90) with ``doicemicro=.false.`` (ILIQ=1), ``dosb_warm_rain=
.true.`` (IRAIN=1), ``dopredictNc=.false.`` (INUM=1, droplet number specified
as ``SeifertBehengConfig.Nc_0``) and ``dofix_pgam=.false.`` -- the gSAM
defaults apart from the two switches that make it warm-rain-only and SB2001:

1. ``satadj_liquid``: the state is put on the liquid saturation curve
   (:func:`_warm_rain.saturation_adjustment_gsam`; smooth by default, exact at
   the hard limit of the three ``gsam_satadj_*`` fields, RULE 5);
2. ``M2005MICRO_GRAUPEL`` (module_mp_graupel.f90) on the adjusted state:
   small-amount removal, rain and droplet size distributions with the number
   reset at the slope bounds, SB2001 autoconversion / accretion, SB2001 rain
   self-collection + breakup, M2005 rain evaporation, the q_c / q_r
   conservation limiters, rain-number evaporation sink, and sedimentation of
   rain (mass and number) and cloud droplets (Stokes) from the post-source
   state with gSAM's CFL sub-steps (:func:`_warm_rain.m2005_liquid_step`).

Oracle-pinned at the hard limit of the three ``gsam_satadj_*`` fields (whole
call, fp64, max relative error <= 1e-10 on T, q_v, q_c,
q_r, N_r and surface precipitation against the compiled gSAM) by
``tests/atmosphere/microphysics/unit/test_gsam_sb_liquid_oracle.py``.

gSAM's data-dependent sedimentation sub-step count runs in Morrison's fixed
maximum loop (``MorrisonConfig.sed_cfl_substeps_max`` default) and the call
FAILS (``equinox.error_if`` + NaN outputs, the Morrison strict path) when a
column needs more sub-steps than that maximum.

``N_c`` is not evolved (``dN_c_dt = 0``; INUM=1).  Fields of
:class:`SeifertBehengConfig` that the gSAM scheme has no counterpart for (the
former proxy-rate coefficients, fall-speed law, smoothing sharpnesses and the
opt-in hard-adjustment guard) must stay at their defaults; setting one raises.

References
----------
- Seifert, A., & Beheng, K. D. (2001). A double-moment parameterization for
  simulating autoconversion, accretion and selfcollection. Atmos. Res.,
  59-60, 265-281.
- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new double-
  moment microphysics parameterization for application in cloud and climate
  models. Part I. J. Atmos. Sci., 62, 1665-1677.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.atmosphere.physics.microphysics.config import (
    MorrisonConfig,
    SeifertBehengConfig,
)
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    apply_sedimentation_poison,
    sedimentation_overflow_guard,
)
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    m2005_liquid_step,
    saturation_adjustment_gsam,
)

# gSAM M2005 rain constants (AR, BR, F1R, F2R, LAMMINR/LAMMAXR, RHOSU, NRAGG):
# Morrison's sam-flavour parameter set, which IS gSAM's (resolve_morrison_flavor
# returns the sam config unchanged).
_GSAM_M2005_PARAMS = MorrisonConfig(morrison_flavor="sam")
# Static bound of the fixed-shape sedimentation sub-step loop: Morrison's
# default (``MorrisonConfig.sed_cfl_substeps_max``), reused unchanged.
_SED_SUBSTEPS_MAX = MorrisonConfig._field_defaults["sed_cfl_substeps_max"]
# The only SeifertBehengConfig field gSAM reads (NDCNST, as a per-volume count).
_GSAM_FIELDS = ("Nc_0", "gsam_satadj_excess_sharpness", "gsam_satadj_qc_floor_sharpness",
                "gsam_satadj_p_min_width_pa", "sb2001_autoconv_ramp_top")

__physics_contract__ = {
    "summary": (
        "Seifert-Beheng two-moment warm rain as gSAM 1.8.8 M2005 runs it "
        "(liquid only, SB2001 warm rain, specified droplet number): liquid "
        "saturation adjustment (smooth by default, exact at the hard "
        "limit), SB2001 autoconversion/accretion/"
        "self-collection+breakup, M2005 rain evaporation, conservation "
        "limiters, CFL-sub-stepped rain and cloud-droplet sedimentation to "
        "surface precipitation. Oracle-pinned to the compiled gSAM at the hard limit."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.N_r": "1/m^3",
        "p_full": "Pa", "rho": "kg/m^3", "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt = (L_v/c_pd) x net "
        "vapour->liquid conversion. SURFACE PRECIPITATION (>= 0, rain plus "
        "cloud-droplet fallout) removes water: column q_v+q_c+q_r changes by "
        "-precipitation*dt/(rho dz) summed, so column moisture is NOT "
        "conserved and no contract-level conservation is claimed. Mixing "
        "ratios and number concentrations stay >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "gSAM 1.8.8 SRC/MICRO_M2005 (Morrison et al. 2005; Seifert & "
        "Beheng 2001)"
    ),
    "idealized_test": (
        "tests/atmosphere/microphysics/unit/test_gsam_sb_liquid_oracle.py -- "
        "warm and supercooled columns against the RUNNING gSAM oracle "
        "(fixture), plus column water closure and finite gradients."
    ),
}


def _reject_fields_without_gsam_counterpart(config: SeifertBehengConfig) -> None:
    """Raise if ``config`` sets a field gSAM's M2005 has no counterpart for."""
    bad = []
    for f in SeifertBehengConfig._fields:
        if f in _GSAM_FIELDS:
            continue
        try:    # a traced (trained) value is never "the default"
            differs = bool(getattr(config, f)
                           != SeifertBehengConfig._field_defaults[f])
        except jax.errors.ConcretizationTypeError:
            differs = True
        if differs:
            bad.append(f)
    if bad:
        raise ValueError(
            "seifert_beheng runs gSAM 1.8.8 M2005 (liquid only, SB2001 warm "
            f"rain), which has no counterpart for {bad}; leave them at their "
            "defaults (only Nc_0, the specified droplet number, the three "
            "gsam_satadj_* smooth-switch fields and sb2001_autoconv_ramp_top "
            "are read).")


def _gsam_call(T, q_v, hydrometeors, p_full, rho, dz, dt, config):
    """satadj_liquid then the M2005 liquid-only step (no sub-step guard)."""
    _reject_fields_without_gsam_counterpart(config)
    T1, q_v1, q_c1 = saturation_adjustment_gsam(
        T, q_v, hydrometeors.q_c, p_full,
        excess_sharpness=config.gsam_satadj_excess_sharpness,
        qc_floor_sharpness=config.gsam_satadj_qc_floor_sharpness,
        p_min_width_pa=config.gsam_satadj_p_min_width_pa)
    return m2005_liquid_step(
        T1, q_v1, q_c1, hydrometeors.q_r, hydrometeors.N_r, p_full, rho, dz,
        dt, config.Nc_0, _GSAM_M2005_PARAMS, _SED_SUBSTEPS_MAX,
        sb2001_autoconv_ramp_top=config.sb2001_autoconv_ramp_top)


def _guard(state):
    """The strict sub-step gate on ``state`` (last entry = required count)."""
    *fields, required = state
    (required, *fields), poison = sedimentation_overflow_guard(
        required, _SED_SUBSTEPS_MAX, (required, *fields),
        "seifert_beheng sedimentation: a column needs more CFL sub-steps "
        f"than the fixed maximum {_SED_SUBSTEPS_MAX} (gSAM's sub-step "
        "count is data-dependent); shorten the physics sub-step")
    return (*fields, required), poison


def seifert_beheng_step(T, q_v, hydrometeors, p_full, rho, dz, dt,
                        config: SeifertBehengConfig = SeifertBehengConfig()):
    """The post-step state of one gSAM call: ``(T, q_v, q_c, q_r, N_r,
    precipitation [kg/m^2/s], required sedimentation sub-steps)``.

    ``N_r`` per-volume [1/m^3].  Fails (``equinox.error_if``; NaN state) when a
    column needs more sedimentation sub-steps than the fixed maximum.
    :func:`seifert_beheng_microphysics` returns the same call as tendencies
    ``(state - input)/dt``.
    """
    state, poison = _guard(_gsam_call(T, q_v, hydrometeors, p_full, rho, dz,
                                      dt, config))
    bias = jnp.where(jnp.isnan(poison), jnp.nan, 0.0)
    *fields, required = state
    return (*(f * poison + bias for f in fields), required)


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
    """Seifert-Beheng warm-rain tendencies as gSAM M2005 computes them (exactly
    at the ``gsam_satadj_*`` hard limit; the defaults are smooth).

    Same interface as all microphysics backends; ``hydrometeors.N_c`` and the
    ice species are not read (droplet number is ``config.Nc_0``; liquid only).
    ``p_half`` is unused.  Tendencies are ``(state - input)/dt`` of
    :func:`seifert_beheng_step`.
    """
    del p_half
    (T1, q_v1, q_c1, q_r1, N_r1, precipitation, required), poison = _guard(
        _gsam_call(T, q_v, hydrometeors, p_full, rho, dz, dt, config))
    z = jnp.zeros_like(T)
    out = MicrophysicsOutput(
        dT_dt=(T1 - T) / dt,
        dq_v_dt=(q_v1 - q_v) / dt,
        dq_c_dt=(q_c1 - hydrometeors.q_c) / dt,
        dq_r_dt=(q_r1 - hydrometeors.q_r) / dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=(N_r1 - hydrometeors.N_r) / dt,
        dN_i_dt=z,
        precipitation=precipitation,
        precip_frozen=jnp.zeros_like(precipitation),   # liquid only
        sed_substeps_required=required,
    )
    return apply_sedimentation_poison(out, poison)
