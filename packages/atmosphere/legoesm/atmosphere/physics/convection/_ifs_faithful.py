"""Faithful IFS convection chain, off by default (BechtoldConfig.use_ifs_ascent).

Chains, in the source's order (see CONTRACT in `_ifs_closure.py`):
    trigger (ifs_departure_search_refined)
      -> first_guess_mass_flux  (cumastrn's ZMFUB first guess)
      -> ascent (ifs_updraught_ascent) launched with THAT base mass flux
      -> ifs_closure (only RESCALES the flux profiles by ZMFS, per cumastrn)
      -> ifs_convective_tendencies

Declared gaps (do not silently "fix" these):
  * No downdraught: M_d = 0, lddraf = False are passed to the tendency module.
  * No CMT: du_dt_conv / dv_dt_conv are returned as None.
  * No friction velocity: `ustar` is not available at this call site and is
    not on the config; the IFS default 0.1 m/s (what cumastrn itself passes
    to the trigger) is used instead, to be threaded from the surface scheme
    later.
  * Total-physics tendencies (PTENT/PTENQ) are NOT available from the caller;
    zeros are used as the "other physics" closure input (see TOTAL TENDENCIES).

INCOMPLETE.  This chain is not yet a faithful reduction of cumastrn and must
not be used to score a run.  The stage cumastrn runs between the closure and
the tendencies -- CUFLXN (cumastrn.F90:1104; cuflxn.F90:250-251 subtracts the
environmental transport, :297-328 builds the below-cloud-base heat and
moisture fluxes) -- is MISSING here, so the tendency module receives raw plume
fluxes together with a tapered below-base mass flux whose heat and moisture
fluxes were never set.  Measured symptom on a frozen deep column: a
+-11000 K/day dipole across the two lowest levels.
"""
from __future__ import annotations

import jax.numpy as jnp

from legoesm.thermo import saturation_specific_humidity
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.convection._ifs_test_ascent import (
    IFSTestAscentConfig,
    half_level_env,
)
from legoesm.atmosphere.physics.convection._ifs_ascent import IFSAscentConfig
from legoesm.atmosphere.physics.convection._ifs_tendencies import IFSTendencyConfig
from legoesm.constants import R_d

from ._ifs_test_ascent import ifs_departure_search_refined
from ._ifs_ascent import ifs_updraught_ascent
from ._ifs_closure import (
    first_guess_mass_flux, ifs_closure, IFSClosureConfig,
    subcloud_mse_supply,
    RG, RCPD, RLVTT, ZDQMIN_FRAC, ZDQMIN_ABS_FLOOR, ZDH_DQMIN_SCALE,
)
from ._ifs_tendencies import ifs_convective_tendencies

__physics_contract__ = {
    "summary": (
        "Faithful IFS convection chain, selected as a whole by "
        "BechtoldConfig.use_ifs_ascent (static python bool, off by default): "
        "cubasen trigger -> cumastrn first-guess cloud-base mass flux -> "
        "cuascn ascent launched at that flux -> cumastrn rescale -> cudtdqn "
        "tendencies. Surface-last arrays; IFS level JK maps to our j = JK-1."
    ),
    "inputs": {
        "T": "K", "q_v": "kg kg-1", "p_full": "Pa", "p_half": "Pa",
        "u": "m s-1", "v": "m s-1",
        "shf_w_m2": "W m-2 (IFS sign: negative = upward)",
        "lhf_w_m2": "W m-2 (IFS sign: negative = upward)",
        "land_frac": "1", "dT_dt_adv": "K s-1", "dq_dt_adv": "kg kg-1 s-1",
        "dT_dt_other": "K s-1 (total physics tendency; None = not carried)",
        "dq_dt_other": "kg kg-1 s-1 (total physics tendency; None = not carried)",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K s-1", "dq_v_dt": "kg kg-1 s-1",
        "dq_c_conv_dt": "kg kg-1 s-1", "cape": "Pa",
        "convective_mask": "1 (bool)", "M_u": "kg m-2 s-1",
    },
    "sign_convention": (
        "mass fluxes positive upward; surface heat fluxes follow the IFS "
        "PAHFS/PQHFL convention (negative = upward); pressure increases with "
        "the array index."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "OpenIFS cubasen/cuascn/cumastrn/cudtdqn (main 8f6f722); declared "
        "gaps: no downdraught (M_d = 0), no CMT, no environmental condensate "
        "for entrainment, friction velocity fixed at the cumastrn call value, "
        "and no total-physics tendency pair (so shallow columns are inert)"
    ),
    "idealized_test": "tests/unit/test_ifs_faithful.py",
}

# No new float tunables on this path; nested configs carry their own specs.
__param_spec__ = {
    "closure": {"type": "config", "class": "IFSClosureConfig"},
}


# cumastrn passes 0.1 m/s as the friction velocity to cubasen; this call site
# has no surface scheme yet, so the chain falls back to that same value.
# DECLARED GAP: thread the real ustar from the surface scheme.
_USTAR_FALLBACK = 0.1   # m/s, cumastrn.F90 CUBASEN call argument


def _hydrostatic_geopotential(T, q_v, p_half):
    """Surface-relative hydrostatic geopotential, surface-last.

    Same construction as the test harness: geo_half[:, nlev] = 0 and the
    layer contributions are accumulated downward-to-upward via a flipped
    cumsum (no python loop).  geo_full is the half-level midpoint.
    """
    nlev = T.shape[1]
    geo_half = jnp.zeros_like(p_half)
    Tv = virtual_temperature(T, q_v)
    dphi = R_d * Tv * jnp.log(p_half[:, 1:] / p_half[:, :-1])
    # dphi[:, j] is the increment across full level j; the surface is at
    # index nlev, so accumulate from the surface upward (reverse cumsum).
    geo_half = geo_half.at[:, :-1].set(jnp.cumsum(dphi[:, ::-1], axis=1)[:, ::-1])
    geo_full = 0.5 * (geo_half[:, :-1] + geo_half[:, 1:])
    return geo_full, geo_half


def ifs_faithful_convection(T, q_v, p_full, p_half, u, v, conv_prog_profile,
                            conv_stoch_state, prng_key, dt, config,
                            shf_w_m2, lhf_w_m2, land_frac,
                            dT_dt_adv, dq_dt_adv, dT_dt_other, dq_dt_other,
                            out_ctor, *, ustar=None):
    """Run the faithful IFS chain. Returns (ConvectionOutput, M_u_profile).

    `dT_dt_adv`/`dq_dt_adv` are the dynamics (advective) pair for the closure
    (PTENTA/PTENQA). `dT_dt_other`/`dq_dt_other` stand in for the total
    physics tendencies (PTENT/PTENQ); the caller currently has no such pair,
    so pass zeros -- do NOT fabricate a total here.

    `ustar` is not available at this call site; None falls back to the IFS
    default 0.1 m/s that cumastrn itself passes to the trigger (declared gap:
    to be threaded from the surface scheme later).
    """
    if dT_dt_other is None:
        # MISSING: total physics tendencies (PTENT/PTENQ) are not assembled
        # by the caller; zeros are the documented fallback.
        dT_dt_other = jnp.zeros_like(T)
        dq_dt_other = jnp.zeros_like(q_v)
    if ustar is None:
        ustar = _USTAR_FALLBACK

    # saturation on the shared utility -- this port is specific humidity
    clo_cfg = IFSClosureConfig()   # source defaults (sucumf.F90)
    qs = saturation_specific_humidity(T, p_full)

    # geopotential, hydrostatic and surface-relative (trigger + closure both
    # integrate geopotential differences; zeros are not acceptable)
    geo_full, geo_half = _hydrostatic_geopotential(T, q_v, p_half)

    # --- 1. trigger / departure-point search ---
    trig = ifs_departure_search_refined(
        T, q_v, p_full, p_half, geo_full=geo_full, geo_half=geo_half,
        shf_w_m2=shf_w_m2, lhf_w_m2=lhf_w_m2, ustar=ustar, land_frac=land_frac,
        dq_dt_adv=dq_dt_adv, cfg=IFSTestAscentConfig())

    # the trigger returns ldcum as a bool-VALUED FLOAT (its contract); the
    # closure/ascent/tendency modules take a real boolean mask
    ldcum = jnp.asarray(trig.ldcum) > 0.5
    ktype = trig.ktype
    k_dpl, k_cbot = trig.k_dpl, trig.k_cbot
    T_u0, q_u0, l_u0 = trig.T_u, trig.q_u, trig.l_u
    w_base, klab0 = trig.w_base, trig.klab

    # --- 2. cumastrn contract: ascent runs ONCE at the FIRST-GUESS M_b ---
    # zdhpbl / zdh exactly as ifs_closure builds them (see its body).
    nlev = T.shape[1]
    jj = jnp.arange(nlev)[None, :]
    k_start = jnp.maximum(
        jnp.min(jnp.where(p_full > clo_cfg.njkt2_pa,
                          jnp.broadcast_to(jj, p_full.shape),
                          jnp.full_like(p_full, p_full.shape[-1] - 1)), axis=-1),
        1)
    zdhpbl = subcloud_mse_supply(dT_dt_other, dq_dt_other, p_half, k_cbot, k_start)
    idx = jnp.arange(T.shape[0])
    kb = jnp.asarray(k_cbot).astype(jnp.int32)
    t_u_b = T_u0[idx, kb]
    q_u_b = q_u0[idx, kb]
    l_u_b = l_u0[idx, kb]
    t_b = T[idx, kb]      # environment at cloud base (no half-level T/q here)
    q_b = q_v[idx, kb]
    zqumqe = q_u_b + l_u_b - q_b
    zdqmin = jnp.maximum(ZDQMIN_FRAC * q_b, ZDQMIN_ABS_FLOOR)   # :567
    # cumastrn.F90:569:  ZDH=RG*MAX(ZDH,1.E5_JPRB*ZDQMIN)
    zdh_base = RG * jnp.maximum(
        RCPD * (t_u_b - t_b) + RLVTT * zqumqe, ZDH_DQMIN_SCALE * zdqmin)
    M_b, zmfmax, ldcum = first_guess_mass_flux(
        p_half, k_cbot, ldcum, ktype, zdhpbl, zdh_base, dt, clo_cfg)

    # Half-level environment by the source's own cuinin rule (NOT arithmetic
    # means); plitot is the environmental condensate the ascent entrains, which
    # this call site does not carry -> zeros, a declared gap.
    T_h, q_h, _s_h = half_level_env(T, q_v, p_full, p_half, geo_full, geo_half,
                                    IFSTestAscentConfig())
    plitot = jnp.zeros_like(T)
    asc = ifs_updraught_ascent(
        T, q_v, qs, p_full, p_half, geo_full, geo_half,
        T_h, q_h, plitot,
        ldcum, ktype, k_dpl, k_cbot, klab0,
        T_u0, q_u0, l_u0, w_base, M_b, dt, cfg=IFSAscentConfig())
    # ascent yields the flux profiles at first-guess M_b (attribute `M`);
    # the closure then only RESCALES them (no second ascent).

    # --- 3. closure: rescales the once-computed ascent profiles (no 2nd ascent)
    clo = ifs_closure(
        asc.M, asc.PMFUS, asc.PMFUQ, asc.PMFUL, asc.PLUDE,
        asc.PDMFUP, asc.PMFUDE_RATE, asc.PDMFEN,
        asc.T_u, asc.q_u, asc.l_u, asc.k_ctop, asc.pwmean, asc.zdpmean,
        ldcum, ktype, k_cbot, k_dpl,
        T, q_v, qs, p_full, p_half, geo_full, geo_half, T_h, q_h,
        dT_dt_other, dq_dt_other, dT_dt_adv, dq_dt_adv,
        land_frac, config.dx_m, dt, clo_cfg)

    # --- 4. tendencies.  DECLARED GAP: the downdraught is not part of this
    # chain yet, so M_d = 0, lddraf = False and the PMFD* fluxes are zero.
    zero = jnp.zeros_like(clo["M_u"])
    dT_dt, dq_dt, _penth = ifs_convective_tendencies(
        T, q_v, qs, p_full, p_half, geo_full, geo_half, T_h, q_h,
        clo["ldcum"], ktype, asc.k_ctop,
        jnp.zeros_like(asc.k_ctop),                 # k_dtop (no downdraught)
        jnp.zeros_like(clo["ldcum"], dtype=bool),   # lddraf
        clo["M_u"], zero,
        clo["PMFUS"], zero, clo["PMFUQ"], zero,
        clo["PMFUL"], clo["PLUDE"], clo["PDMFUP"],
        dt, IFSTendencyConfig())

    out = out_ctor(
        dT_dt=dT_dt,
        dq_v_dt=dq_dt,
        dq_c_conv_dt=jnp.zeros_like(dq_dt),  # detrained condensate (PLUDE) source
        cape=clo["zcape"],
        convective_mask=ldcum,
        du_dt_conv=None,                     # no CMT on the faithful path yet
        dv_dt_conv=None,
        dq_r_conv_dt=None,                   # tendency module returns no precip rate
    )
    return out, clo["M_u"]
