"""Morrison double-moment ice+liquid microphysics.

Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
(Cooper 1986), depositional growth, Bergeron process, riming, snow
aggregation, and melting. Tracks cloud water, rain, ice, and snow.

All operations use smooth (differentiable) approximations.

Double-moment NUMBER-budget faithfulness vs the SAM M2005 oracle
(module_mp_graupel.f90) — status of the gaps from the fall-speed-audit
follow-up (m2005-double-moment-numberbudget-followup), 2026-07-17:

ADDRESSED
- Cooper nucleation target counts the TOTAL frozen number: the deficit is
  ``kc2 − (NI3D + NS3D + NG3D)`` (SAM :3396-3399).  With prognostic
  N_s/N_g the target subtracts them too; single-moment snow/graupel
  contribute zero (prior behavior, byte-identical).
- NPRCI (ice→snow autoconversion number, SAM :3323-3326): verified WIRED
  as ``dN_i_autoconv`` (= PRCI/CONS22 capped at N_i/dt) — an N_i sink and,
  under double-moment snow, an N_s source.  The gap was stale comments
  claiming it was dropped, not the code.
- NSMLTR/NGMLTR (SAM :2189-2203, :2211): melted snow-flake / graupel-
  particle number now becomes RAIN number (·rho for the per-volume N_r)
  instead of vanishing; single owner with the NSMLTS/NGMLTG sinks.
- Melt/freeze number-routing comments made to agree with the double-moment
  branches (freeze_N_to_graupel = SAM NNUCCR; dN_g_melt = NGMLTG).

KNOWN, DELIBERATE departures (documented where they occur)
- Default (no prognostic N_s/N_g): bulk q-power snow fall speed and the
  fixed-N0G Marshall-Palmer graupel closure — outside the SAM oracle
  scope, which is inherently two-moment (departure #5 of the fall-speed
  audit).

References
----------
- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
  double-moment microphysics parameterization. Part I: Description.
  J. Atmos. Sci., 62, 1665-1677.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    autoconversion_sb2001,
    autoconversion_kk2000,
    accretion,
    accretion_sb2001,
    accretion_kk2000,
    self_collection_breakup,
    self_collection_breakup_sb2001,
    rain_freezing_bigg,
    snow_deposition_m2005,
    snow_riming_psacws,
    snow_melting_psmlt,
    snow_self_aggregation_nsagg,
    graupel_lamg,
    graupel_melting_pgmlt,
    graupel_riming_psacwg,
    graupel_deposition_prdg,
    graupel_rain_accretion_pracg,
    snow_to_graupel_pgsacw,
    rain_evaporation,
    rain_evaporation_m2005,
    safe_pow,
    donor_clamp_scale,
)
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


# --- fixed M2005 PSD / fall-speed / nucleation / transport constants ---
_RHO_FLOOR = 0.1                 # air-density floor in divisions [kg/m^3]
_FALL_RHO_EXP = 0.54             # (rho_su/rho)^0.54 fall-speed density correction (r/s/g)
_FALL_RHO_EXP_ICE = 0.35         # ice fall-speed density correction exponent
_VT_CAP_RAIN = 9.1               # rain fall-speed cap [m/s]
_VT_CAP_GRAUPEL = 20.0           # graupel fall-speed cap [m/s]
_VT_CLIP_RAIN = 20.0             # rain fall-speed clip ceiling [m/s]
_VT_CLIP_FROZEN = 5.0            # snow/ice fall-speed clip ceiling [m/s]
_VT_CAP_SNOW_ICE = 1.2           # snow/ice-cap fall-speed factor [m/s]
_DV_PREFACTOR = 8.794e-5         # Hall-Pruppacher vapour diffusivity prefactor
_DV_T_EXPONENT = 1.81
_COOPER_EXP_CAP = 80.0           # Cooper ice-nuclei exp argument cap
_NUC_T_THRESHOLD_K = 265.15      # heterogeneous ice-nucleation T threshold [K]
_NUC_RH_ICE_THRESHOLD = 1.08     # ice-nucleation RH_ice gate
_NUC_RH_LIQ_THRESHOLD = 0.999    # liquid-nucleation RH_liq gate
_FERRIER_ICE_NUMBER_DENOM = 1080.0  # Ferrier ice-number diagnostic denominator

def resolve_morrison_flavor(config: MorrisonConfig) -> MorrisonConfig:
    """Resolve ``morrison_flavor`` into the concrete parameter set.

    ``"sam"`` returns ``config`` unchanged (the SAM/gSAM M2005 defaults already
    in ``MorrisonConfig``). ``"mg"`` overrides the parameters where E3SM/CESM
    Morrison-Gettelman (micro_mg_utils.F90) differs from SAM:

    =====================  ============  =============
    field                  SAM           MG
    =====================  ============  =============
    ``lami_max``           1/1 µm        1/10 µm  (LAMMAXI; bounds ice number)
    ``snow_aggregation_eii`` 0.1         0.5      (eii; ice→snow & snow self-agg)
    ``rho_snow``           100 kg/m³     250      (rhosn)
    ``fall_b_i``           0.865         1.0      (bi)
    ``ice_to_snow_scheme`` m2005_autoconv mg_ferrier (180-s Ferrier prci)
    =====================  ============  =============

    Warm rain (``kk2000``) and ice deposition (``m2005``) are already MG-faithful
    and unchanged. Validated against the E3SM MG reference (codex audit).
    """
    if config.morrison_flavor == "mg":
        return config._replace(
            lami_max=1.0 / 10.0e-6,  # coeff-ok: max ice slope (1/10um)
            snow_aggregation_eii=0.5,
            rho_snow=250.0,  # coeff-ok: snow bulk density default [kg/m^3]
            fall_b_i=1.0,
            ice_to_snow_scheme="mg_ferrier",
        )
    elif config.morrison_flavor == "sam":
        return config
    raise ValueError(
        f"Unknown morrison_flavor: {config.morrison_flavor!r}; "
        f"choose 'mg' (global default) or 'sam' (CRM)."
    )


__physics_contract__ = {
    "summary": (
        "Morrison, Curry & Khvorostyanov (2005) double-moment ice+liquid "
        "microphysics: extends two-moment warm rain with ice nucleation, "
        "deposition/sublimation, riming, melting and snow, with mass+number "
        "prognostic and sedimentation to surface precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "hydrometeors.q_s": "kg/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s", "dq_s_dt": "kg/kg/s",
        "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)", "dN_i_dt": "1/(kg s)",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt uses L_v (vapour<->"
        "liquid), L_s (vapour<->ice deposition/sublimation) and L_f (freezing/"
        "melting), consistent with each phase-change rate. Water is "
        "redistributed among vapour/cloud/rain/ice/snow; SURFACE PRECIPITATION "
        "(rain+snow melt, >= 0) removes water, so column moisture is NOT "
        "conserved -- no contract-level conservation is claimed. Masses and "
        "numbers stay >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Morrison, Curry & Khvorostyanov (2005), J. Atmos. Sci. 62, 1665-1677",
    "idealized_test": (
        "tests/unit/test_physics_microphysics.py + tests/unit/"
        "test_rce_ice_microphysics.py — a cold supersaturated column nucleates "
        "ice with L_s heating; riming/melting move mass between phases; snow + "
        "rain reach the surface; all masses/numbers >= 0."
    ),
}


def morrison_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: MorrisonConfig = MorrisonConfig(),
) -> MicrophysicsOutput:
    """Compute Morrison double-moment microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    # Resolve the SAM/MG flavor into concrete parameters (static Python branch
    # on the config string — no traced control flow).
    config = resolve_morrison_flavor(config)
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    q_i = hydrometeors.q_i
    q_s = hydrometeors.q_s
    q_g = hydrometeors.q_g
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    N_i = hydrometeors.N_i
    # Optional prognostic snow number [1/kg] (per-mass). None ⇒ single-moment
    # snow (bulk fall speed, no snow-number budget); an array ⇒ double-moment
    # snow (PSD slope LAMS, PSD fall speed UMS/UNS, snow-number budget).
    N_s = hydrometeors.N_s
    snow_double_moment = N_s is not None
    # Optional prognostic graupel number [1/kg] (per-mass). None ⇒ single-moment
    # graupel (fixed intercept N0G); an array ⇒ double-moment graupel (PSD slope
    # LAMG from N_g, number budget). Requires do_graupel.
    N_g = hydrometeors.N_g
    graupel_double_moment = (N_g is not None) and config.do_graupel
    N_g_arg = N_g if graupel_double_moment else None
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(
        N_c, config.Nc_0,
        predict_Nc=getattr(config, "predict_Nc", False),
        nc_specified_field=getattr(config, "nc_from_aerosol", False),
    )

    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    # Pass ``q_c`` so the evaporation branch (negative ``condensation``)
    # is donor-clamped: evaporation cannot drive ``q_c`` below zero in
    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, dt, sharpness, q_c=q_c,
        hard_adjust=config.hard_saturation_adjustment,
        hard_threshold=config.hard_sat_adjust_threshold,
        hard_max_heating_K=config.hard_sat_max_heating_K,
    )
    # Sub-grid in-cloud closure (Morrison & Gettelman 2008): evaluate the
    # warm-rain rates on the IN-CLOUD water q_c/cf and scale back by cf, so the
    # non-linear KK2000/SB rates see the (higher) in-cloud concentration rather
    # than the grid-mean.  cf is the Sundqvist √-form from the local RH, floored
    # for AD/numeric safety.  When disabled, cf_eff=1 ⇒ identity (grid-mean).
    if getattr(config, "subgrid_autoconversion", False):
        RH = q_v / jnp.maximum(q_sat, 1.0e-10)
        arg = (1.0 - RH) / max(1.0 - config.subgrid_rh_crit, 1.0e-6)
        arg_safe = jnp.where(arg > 0.0, arg, 1.0)
        cf_sg = jnp.where(arg > 0.0, 1.0 - jnp.sqrt(arg_safe), 1.0)
        cf_eff = jnp.clip(cf_sg, config.subgrid_cf_min, 1.0)
    else:
        cf_eff = jnp.ones_like(q_c)
    q_c_ic = q_c / cf_eff
    q_r_ic = q_r / cf_eff
    # Warm-rain autoconversion + accretion. KK2000 (default) is the SAM
    # M2005 oracle scheme; Seifert-Beheng retained for back-compat.
    if config.warm_rain_scheme == "kk2000":
        dq_c_au, dN_r_au, x_c = autoconversion_kk2000(q_c_ic, N_c_eff, rho, dt)
        dq_c_au = dq_c_au * cf_eff
        dN_r_au = dN_r_au * cf_eff
        dq_c_ac = accretion_kk2000(q_c_ic, q_r_ic) * cf_eff
    elif config.warm_rain_scheme == "seifert_beheng":
        dq_c_au, dN_r_au, x_c = autoconversion_sb(
            q_c_ic, N_c_eff, rho, config.k_au, config.x_star,
            config.autoconversion_sharpness,
        )
        dq_c_au = dq_c_au * cf_eff
        dN_r_au = dN_r_au * cf_eff
        dq_c_ac = accretion(q_c_ic, q_r_ic, rho, config.k_ac) * cf_eff
    elif config.warm_rain_scheme == "seifert_beheng_sb2001":
        # PUBLISHED SB2001 universal functions (phi_au, phi_ac) — the faithful
        # gSAM IRAIN=1 MASS rates (module_mp_graupel.f90:1835-1844, :1960-1962),
        # as opposed to the simplified "seifert_beheng" proxy above. tau is a
        # scale-invariant ratio so the in-cloud (q/cf) rescaling leaves it
        # unchanged. NOTE the subgrid enhancement is NOT the same across the
        # laws: with N_c_eff held at the grid-mean (not rescaled by cf), the
        # q_c^4 autoconversion gets a cf^-3 enhancement vs cf^-1 for accretion —
        # STRONGER than the kk2000 path. This (and the un-rescaled in-cloud N_c)
        # is a known subgrid-closure limitation, moot at the default
        # ``subgrid_autoconversion=False`` (cf_eff=1). The SB2001-specific
        # cloud-NUMBER autoconv factor (2·PRC·rho/x_*) is not applied — the
        # generic ``-dq_c_au·rho/x_c`` sink below (predict_Nc=True) is used
        # instead; see ``autoconversion_sb2001``.
        dq_c_au, dN_r_au, x_c = autoconversion_sb2001(
            q_c_ic, q_r_ic, N_c_eff, rho,
        )
        dq_c_au = dq_c_au * cf_eff
        dN_r_au = dN_r_au * cf_eff
        dq_c_ac = accretion_sb2001(q_c_ic, q_r_ic, rho) * cf_eff
    else:
        raise ValueError(
            f"Unknown warm_rain_scheme: {config.warm_rain_scheme!r}; choose "
            f"'kk2000' (SAM M2005 default), 'seifert_beheng' (simplified "
            f"proxy) or 'seifert_beheng_sb2001' (published SB2001 universal "
            f"functions)."
        )
    # Rain self-collection + breakup. "sb2001" (SAM NRAGG) self-collects
    # ~5580× faster than the legacy k_sc=1e-3 sigmoid form, so rain coalesces
    # to the SB equilibrium drop size (feeds the PSD fall speed + evaporation).
    if config.rain_selfcoll_scheme == "sb2001":
        dN_r_selfcoll = self_collection_breakup_sb2001(
            N_r, q_r, rho, config, dt=dt)
    elif config.rain_selfcoll_scheme == "legacy":
        dN_r_sc, dN_r_br = self_collection_breakup(
            N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
        )
        dN_r_selfcoll = dN_r_sc + dN_r_br
    else:
        raise ValueError(
            f"Unknown rain_selfcoll_scheme: {config.rain_selfcoll_scheme!r}. "
            f"Expected 'sb2001' (SAM NRAGG, default) or 'legacy'."
        )
    if config.rain_evap_scheme == "m2005":
        evaporation = rain_evaporation_m2005(
            q_v, q_r, N_r, q_sat, T, p_full, rho, config, dt=dt)
    elif config.rain_evap_scheme == "bulk":
        evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff, dt=dt)
    else:
        raise ValueError(
            f"Unknown rain_evap_scheme: {config.rain_evap_scheme!r}. "
            f"Expected 'm2005' (SAM PRE, default) or 'bulk' (legacy)."
        )

    # === ICE PHASE ===
    T_freeze = constants.T_freeze
    # f_ice transitions around ``cooper_T_act`` (≈ 265 K by default), NOT
    # ``T_freeze`` (273.15 K).  Cooper (1986) IN observations start at
    # ~ −5 °C; primary nucleation is parameterised to switch on a few K
    # below freezing, not at the melting point.  At T ≪ cooper_T_act
    # ``f_ice → 1`` (cold nucleation fully active); at T ≫ cooper_T_act
    # ``f_ice → 0`` (warm columns: every ice source gated off).
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # 1. Ice nucleation — Cooper (1986) as in SAM M2005 INUC=0
    #    (module_mp_graupel.f90:3387-3398).
    # Target number kc2 = min(N_i0·exp(cooper_a·(T_f−T)), N_i_nuc_max)/ρ
    # (canonical Cooper 0.005 L⁻¹ = N_i0=5 m⁻³ base; cap 500 L⁻¹).
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    N_i_target = jnp.minimum(
        config.N_i0 * jnp.exp(
            jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), _COOPER_EXP_CAP)
        ),
        config.N_i_nuc_max,
    ) / jnp.clip(rho, _RHO_FLOOR)
    # SAM compares kc2 to the TOTAL frozen number NI3D+NS3D+NG3D
    # (module_mp_graupel.f90:3396-3399): nucleation only tops the column up
    # to the Cooper target COUNTING crystals already held as snow/graupel.
    # With prognostic N_s/N_g (double-moment) subtract them too — otherwise
    # a column whose snow/graupel number already satisfies the target keeps
    # nucleating cloud ice (and its MI0 mass source), a spurious ice source
    # (codex iter-9, resolved now that double-moment snow/graupel exist).
    # Single-moment snow/graupel carry no number and contribute zero — the
    # prior behavior, byte-identical.  N_i stays UNclipped (exact old AD
    # path); N_s/N_g are clipped at 0 so a transient negative number can
    # never INFLATE nucleation.  All three are per-mass [1/kg], matching
    # the /rho-converted target.
    # SUPERSATURATION GATE (the M6 fix): SAM fires deposition-nucleation
    # only where (RH_liq ≥ 0.999 AND T ≤ 265.15 K) OR RH_ice ≥ 1.08 — NOT
    # in any cold air. The previous code used only the f_ice temperature
    # gate, nucleating ice even in subsaturated cold layers (spurious ice).
    # Smooth sigmoid surrogates for the hard SAM thresholds keep AD clean.
    rh_liq = q_v / jnp.clip(q_sat, 1.0e-12)
    rh_ice = q_v / jnp.clip(q_sat_i, 1.0e-12)
    gate_liq = (
        jax.nn.sigmoid(config.nuc_rh_sharpness * (rh_liq - _NUC_RH_LIQ_THRESHOLD))
        * jax.nn.sigmoid(config.nuc_T_sharpness * (_NUC_T_THRESHOLD_K - T))
    )
    gate_ice = jax.nn.sigmoid(config.nuc_rh_sharpness * (rh_ice - _NUC_RH_ICE_THRESHOLD))
    nuc_gate = jnp.maximum(gate_liq, gate_ice)
    N_frozen = N_i
    if snow_double_moment:
        N_frozen = N_frozen + jnp.clip(N_s, 0.0)
    if graupel_double_moment:
        N_frozen = N_frozen + jnp.clip(N_g, 0.0)
    dN_i_nuc = (
        jnp.clip(N_i_target - N_frozen, 0.0) / jnp.clip(dt, 1.0) * nuc_gate
    )
    # Mass source: each new crystal starts at MI0 = 4/3·π·ρ_ci·r³ (SAM
    # MNUCCD = NNUCCD·MI0), so freshly-nucleated ice carries mass rather
    # than relying solely on subsequent deposition.
    mi0 = (
        4.0 / 3.0 * jnp.pi * config.rho_cloud_ice
        * config.ice_nuc_radius ** 3
    )
    dq_i_nuc = dN_i_nuc * mi0

    # 1b. Homogeneous ice nucleation (Koop 2000 / Kärcher-Lohmann 2002) — OPT-IN.
    # SAM's Cooper-only ice (≤500/L) leaves NOTHING to cap cirrus ice-super-
    # saturation, so violent convective outflow can pile q_v to RH_ice ≫ 100 %
    # faster than the N_i^⅔·q_i^⅓ deposition bootstraps. Real cirrus homogeneous
    # freezing of aqueous haze bursts a high crystal number once RH_ice exceeds
    # the homogeneous threshold S_hom(T); the fresh crystals (+ seed mass) feed
    # the EXISTING M2005 deposition below (N_i / q_i boosts), which then deposits
    # the excess vapour and pins RH_ice near S_hom. Folded into dN_i_nuc/dq_i_nuc
    # AFTER §2 so every downstream budget (vapour sink, ice mass/number, L_s
    # heat, donor clamps) accounts for it once, consistently.
    if config.homogeneous_ice_nucleation:
        s_hom = jnp.clip(
            config.koop_s_hom_a - config.koop_s_hom_b * T,
            config.koop_s_hom_min, config.koop_s_hom_max,
        )
        # Two smooth gates: (i) ice supersaturation past the homogeneous
        # threshold RH_ice ≥ S_hom; (ii) COLD temperature T ≤ hom_freeze_T_max
        # (homogeneous freezing of aqueous haze is a deep-cold-cirrus process,
        # T ≲ −38 °C — do NOT reuse f_ice, which stays ≈1 up to ~260 K and would
        # let it fire far too warm).
        hom_gate = (
            jax.nn.sigmoid(config.hom_ice_nuc_sharpness * (rh_ice - s_hom))
            * jax.nn.sigmoid(
                config.hom_freeze_T_sharpness * (config.hom_freeze_T_max - T))
        )
        n_hom_target = config.hom_ice_nuc_N / jnp.clip(rho, _RHO_FLOOR)  # per-mass
        # Top up to n_hom_target counting the COOPER crystals nucleated in this
        # SAME step (dN_i_nuc*dt), not just the pre-existing N_i.  Cooper's
        # ``gate_ice`` opens at RH_ice >= 1.08, which is ALWAYS satisfied when the
        # homogeneous gate (RH_ice >= S_hom ~ 1.47) is open, so both sources fire
        # together: subtracting only N_i summed the two targets and landed N_i at
        # ~1.5x hom_ice_nuc_N (measured 4.419e6 = 2.948e6 + 1.474e6 /kg at
        # 228 K/222 hPa).  Same "count what is already there" logic as Cooper's
        # own N_frozen subtraction at :372-376.
        dN_i_hom = (
            jnp.clip(
                n_hom_target - jnp.clip(N_i, 0.0) - dN_i_nuc * jnp.clip(dt, 1.0),
                0.0,
            )
            / jnp.clip(dt, 1.0) * hom_gate
        )
        dq_i_hom = dN_i_hom * mi0
    else:
        dN_i_hom = jnp.zeros_like(N_i)
        dq_i_hom = jnp.zeros_like(q_v)

    # 2. Depositional growth + sublimation (SAM M2005 PRD/EPRD,
    #    module_mp_graupel.f90:3427-3514). q_sat_i computed above (§1).
    q_i_eff = jnp.maximum(jnp.clip(q_i, 0.0), config.q_i_min_growth)
    # Crystals + seed mass freshly produced by homogeneous nucleation (§1b) are
    # available to deposit vapour THIS step (homogeneous freezing + diffusional
    # growth are near-instantaneous vs dt) — boost the EPSI number/mass so the
    # cap engages without a one-step lag. Python static-bool gate (config field,
    # NOT traced) so the default-OFF path is the ORIGINAL operation graph,
    # byte-identical to the SAM-faithful Cooper-only deposition.
    if config.homogeneous_ice_nucleation:
        N_i_dep = jnp.clip(N_i, 0.0) + dN_i_hom * dt
        q_i_dep_eff = jnp.maximum(q_i_eff + dq_i_hom * dt, config.q_i_min_growth)
    else:
        N_i_dep = jnp.clip(N_i, 0.0)
        q_i_dep_eff = q_i_eff
    # SAM PRCI ice→snow autoconversion (set inside the m2005 deposition block,
    # which provides DV/ABI/q_sat_i); 0 for the heuristic deposition path.
    ice_to_snow_m2005 = jnp.zeros_like(jnp.clip(q_i, 0.0))
    # M1b deposition size-split tail (cloud-ice depositional growth PAST DCS
    # that SAM routes to snow); 0 unless the m2005 block sets it.
    dep_to_snow_m1b = jnp.zeros_like(jnp.clip(q_i, 0.0))
    # Ice-PSD slope/intercept (LAMI/N0I) and the snow-autoconversion size
    # threshold DCS are PSD properties, independent of the deposition scheme,
    # but the ice->snow autoconversion (m2005_autoconv / mg_ferrier) needs
    # them.  Compute them unconditionally so aggregation works under ANY
    # ice_deposition_scheme — the "heuristic" deposition path leaves the m2005
    # block unexecuted, which previously left LAMI/N0I unbound and crashed the
    # mg_ferrier autoconversion (the morrison_flavor="mg" default pairs
    # mg_ferrier with — but does not force — m2005 deposition).
    cons12_cbrt = (config.rho_cloud_ice * jnp.pi) ** (1.0 / 3.0)
    dcs = config.ice_snow_d_auto
    lami_ac = cons12_cbrt * safe_pow(
        jnp.clip(N_i, 0.0) / jnp.maximum(q_i_eff, 1.0e-20), 1.0 / 3.0)
    n0i_ac = jnp.clip(N_i, 0.0) * lami_ac
    if config.ice_deposition_scheme == "m2005":
        # M2005-FORM bulk diffusional growth (gSAM EPSI/ABI/CONS12 structure;
        # legoESM applies its OWN q_sat_i, the ice_deposition_efficiency
        # multiplier, and the q_i floors/non-negative clips in place of gSAM's
        # pre-EPSI LAMI bounds / N0I-NI3D update — see test_m_prd_ice_deposition):
        #   PRD = EPSI·(q_v − q_sat_i)/ABI
        #   EPSI = 2π·N_i·ρ·DV/LAMI, LAMI = (CONS12·N_i/q_i)^(1/3)
        #        = (2π/CONS12^⅓)·ρ·DV·N_i^⅔·q_i^⅓   (the q_i^⅓·N_i^⅔ scaling)
        #   CONS12 = Γ(1+DI)·CI = RHOI·π   (DI=3, CI=RHOI·π/6; N_i [1/kg]).
        #   DV  = 8.794e-5·T^1.81/p           (vapour diffusivity [m²/s])
        #   ABI = 1 + (dq_sat_i/dT)·L_s/c_p   (psychrometric correction),
        #         dq_sat_i/dT = L_s·q_sat_i/(R_v·T²)   (Clausius–Clapeyron).
        # (q_v − q_sat_i) < 0 ⇒ SUBLIMATION (negative). Tuned by the
        # dimensionless ``ice_deposition_efficiency``.
        dv_vap = _DV_PREFACTOR * safe_pow(T, _DV_T_EXPONENT) / jnp.clip(p_full, 1.0)
        dqsidt = constants.L_s * q_sat_i / (constants.R_v * T ** 2)
        abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
        epsi = (
            2.0 * jnp.pi / cons12_cbrt
            * rho * dv_vap
            * safe_pow(N_i_dep, 2.0 / 3.0)
            * safe_pow(q_i_dep_eff, 1.0 / 3.0)
        )
        dep_raw = (
            config.ice_deposition_efficiency * epsi
            * (q_v - q_sat_i) / abi
        )
        # DEPOSITION (positive) + SUBLIMATION (negative) are BOTH governed
        # by ice supersaturation + existing ice, NOT Cooper activation
        # (SAM PRD/EPRD have no temperature gate). The growth term is NOT
        # multiplied by ``f_ice`` (iter-14, Codex iters 8/9/13): the
        # ``EPSI ∝ N_i^(2/3)`` factor already self-gates on ice presence
        # (N_i=0 ⇒ EPSI=0 ⇒ deposition=0), so f_ice was a REDUNDANT gate
        # that wrongly suppressed warm-mixed-phase (265–273 K) ice growth
        # and the emergent WBF there. Sublimation stays donor-clamped.
        dep_pos = jnp.maximum(dep_raw, 0.0)
        if config.homogeneous_ice_nucleation:
            # Stability backstop (ON-path only ⇒ OFF graph unchanged): the
            # homogeneous-boosted EPSI can make the explicit deposition stiff
            # (EPSI·dt/ABI → 1); cap the positive deposition at the available
            # ice supersaturation so q_v cannot OVERSHOOT below q_sat_i in one
            # step (which would flip to spurious sublimation / oscillation).
            # This is the analytic single-step relaxation limit, NOT a clip of a
            # physical quantity — deposition physically halts at saturation.
            dep_pos = jnp.minimum(
                dep_pos,
                jnp.maximum(q_v - q_sat_i, 0.0) / jnp.clip(dt, 1.0),
            )
        subl_neg = jnp.maximum(
            jnp.minimum(dep_raw, 0.0),
            -jnp.clip(q_i, 0.0) / jnp.clip(dt, 1.0),
        )
        if config.homogeneous_ice_nucleation:
            # SYMMETRIC partner of the dep_pos cap above (ON-path only => OFF
            # graph unchanged).  Sublimation physically halts at ice saturation
            # just as deposition does, so it cannot exceed the available vapour
            # DEFICIT (q_sat_i - q_v)/dt.  Without this the cap is one-sided and
            # the boosted-EPSI stiff regime becomes a 2-step limit cycle:
            # deposit the whole excess -> strongly ice-subsaturated -> sublimate
            # the fresh ice straight back (measured RH_ice 2.88 -> 0.27 -> 2.22
            # -> 0.61 ... at 228 K, dt=240 s), a spurious +/-1 K/step latent-
            # heating sawtooth.  The q_i donor clamp above bounds the OTHER side
            # (cannot sublimate more ice than exists); both bounds are needed.
            subl_neg = jnp.maximum(
                subl_neg,
                -jnp.maximum(q_sat_i - q_v, 0.0) / jnp.clip(dt, 1.0),
            )
        dq_i_dep = dep_pos + subl_neg
        # PRCI: ice→snow autoconversion — depositional growth of the cloud-ice
        # PSD across the snow-size threshold DCS (module_mp_graupel.f90:
        # 3322-3326). LAMI=(ρ_ci·π·N_i/q_i)^⅓, N0I=N_i·LAMI; with
        # CONS21=4/(DCS·ρ_ci), CONS22=π·ρ_ci·DCS³/6 (ρ_ci cancels):
        #   PRCI = CONS22·CONS21·(q_v−q_sat_i)₊·ρ·N0I·exp(−LAMI·DCS)·DV/ABI
        #        = (2π·DCS²/3)·ρ·N0I·exp(−LAMI·DCS)·DV·(q_v−q_sat_i)₊/ABI.
        # Only POSITIVE ice supersaturation grows ice across DCS into snow;
        # self-gates on N_i (N0I∝N_i ⇒ 0 when no ice). The NUMBER transfer
        # NPRCI = PRCI/CONS22 (capped at N_i/dt, SAM :3325-3326) IS wired:
        # it is ``dN_i_autoconv`` below — an N_i sink AND (double-moment
        # snow) an N_s source, exactly SAM's -NPRCI in NI3DTEN / +NPRCI in
        # NS3DTEN.  (An earlier comment here claimed it was dropped; that
        # predated double-moment snow and was stale, not the code.)
        ice_to_snow_m2005 = (
            (2.0 * jnp.pi / 3.0) * dcs ** 2 * rho * n0i_ac
            * jnp.exp(-lami_ac * dcs) * dv_vap
            * jnp.maximum(q_v - q_sat_i, 0.0) / abi
        )
        # SAM skips PRCI for QI < QSMALL (1e-14); the q_i_eff floor (1e-9) is
        # used only inside LAMI, so gate the autoconv on ACTUAL ice (codex
        # iter-20 D) — avoids converting trace/absent ice to snow.
        ice_to_snow_m2005 = jnp.where(
            jnp.clip(q_i, 0.0) > 1.0e-14, ice_to_snow_m2005, 0.0)
        # M1b deposition size-split (mp_graupel.f90:3466-3481): only the part
        # of the ice-PSD depositional growth BELOW DCS stays cloud ice; DUM =
        # 1−exp(−LAMI·DCS)(1+LAMI·DCS) is that below-DCS fraction. The TAIL
        # (1−DUM, ice grown PAST DCS) is added to SNOW when snow is present
        # (else stays cloud ice — no receiver). Only DEPOSITION (dep_pos) is
        # split; sublimation shrinks the in-place PSD. The tail is routed via
        # ``prds`` below, so the vapour-sink budget + latent heat + dq_s stay
        # consistent (prds≥0 whenever dep_pos>0 — same (q_v−q_sat_i) sign).
        dep_dum = 1.0 - jnp.exp(-lami_ac * dcs) * (1.0 + lami_ac * dcs)
        ice_dep_frac = jnp.where(jnp.clip(q_s, 0.0) > 1.0e-14, dep_dum, 1.0)
        dep_to_snow_m1b = dep_pos * (1.0 - ice_dep_frac)
        dq_i_dep = dep_pos * ice_dep_frac + subl_neg
    elif config.ice_deposition_scheme == "heuristic":
        # Legacy deposition-only form (no sublimation): dep_coeff has units.
        S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
        dq_i_dep = (
            config.dep_coeff
            * jnp.maximum(S_i, 0.0)
            * q_i_eff
            * safe_pow(N_i, 1.0 / 3.0)
            * f_ice
        )
    else:
        raise ValueError(
            f"Unknown ice_deposition_scheme: "
            f"{config.ice_deposition_scheme!r}; choose 'm2005' (SAM "
            f"M2005 diffusional growth) or 'heuristic' (legacy)."
        )

    # Fold homogeneous nucleation (§1b) into the Cooper nucleation source so the
    # vapour sink (−dq_i_nuc), ice mass (+dq_i_nuc), ice number (+dN_i_nuc), L_s
    # heat (+L_s·dq_i_nuc) and the vapour donor clamp all account for it ONCE.
    # The bulk of the supersaturation removal is the boosted deposition dq_i_dep
    # (via N_i_dep/q_i_dep_eff above); this seed term is the small MNUCCD mass.
    # Static-bool gate ⇒ default-OFF leaves dN_i_nuc/dq_i_nuc untouched.
    if config.homogeneous_ice_nucleation:
        dN_i_nuc = dN_i_nuc + dN_i_hom
        dq_i_nuc = dq_i_nuc + dq_i_hom

    # 3. Wegener-Bergeron-Findeisen: cloud water -> ice in the mixed phase.
    # SAM M2005 (faithful) has NO explicit Bergeron rate — WBF EMERGES from
    # the ice deposition above (PRD draws q_v toward ice saturation) plus the
    # saturation adjustment (evaporates cloud water as q_v drops below liquid
    # saturation). So the conversion is deposition-rate-limited, not forced.
    # The legacy "bergeron_heuristic" over-glaciates (~50× faster than the
    # deposition physics supports). M2, iter-13.
    if config.wbf_scheme == "emergent":
        bergeron = jnp.zeros_like(q_c)
    elif config.wbf_scheme == "bergeron_heuristic":
        # ``f_ice`` collapses the warm-column tail; the T window centres the
        # heuristic on the classic −15 °C WBF peak.
        berg_window = (
            jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
            * jax.nn.sigmoid(
                config.melt_sharpness * (T - (config.T_center - config.T_width))
            )
        )
        bergeron = (
            config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window * f_ice
        )
    else:
        raise ValueError(
            f"Unknown wbf_scheme: {config.wbf_scheme!r}; choose 'emergent' "
            f"(SAM M2005, deposition-limited) or 'bergeron_heuristic'."
        )

    # 4. Riming: ice/snow collect cloud water (supercooled droplets freeze on).
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    # Snow riming: faithful SAM PSACWS (PSD collection) when double-moment;
    # else the crude bulk rate. dt=None ⇒ NO internal donor clamp — the shared
    # q_c donor clamp below caps it ONCE alongside the other q_c sinks (codex
    # iter-28 E: an internal clamp would double-clamp and distort the relative
    # allocation when PSACWS is much faster than autoconv/accretion). Warm-
    # temperature collection is omitted (pure cold-cloud riming; legoESM has no
    # wet-snow/graupel path — codex B); above 0 °C the snow melts anyway.
    if snow_double_moment and config.do_snow_riming:
        riming_s = snow_riming_psacws(q_c, q_s, N_s, T, rho, config, dt=None)
    else:
        riming_s = (config.rime_coeff * jnp.clip(q_s, 0.0)
                    * jnp.clip(q_c, 0.0) * f_ice)

    # 4b. Graupel riming PSACWG: graupel collects supercooled cloud water and
    #     grows (q_c→q_g + L_f) — the primary graupel growth path. dt=None ⇒ the
    #     shared q_c donor clamp below caps it ONCE alongside the other q_c sinks
    #     (same no-double-clamp pattern as snow PSACWS, codex iter-28 E).
    if config.do_graupel and config.do_graupel_riming:
        riming_g = graupel_riming_psacwg(
            q_c, q_g, T, rho, config, dt=None, N_g=N_g_arg)
    else:
        riming_g = jnp.zeros_like(jnp.clip(q_c, 0.0))

    # 5. Ice → snow autoconversion. "m2005_autoconv" = SAM PRCI (deposition-
    #    driven across DCS, computed in the m2005 deposition block above);
    #    "heuristic" = legacy constant-rate. The downstream q_i donor clamp
    #    protects q_i regardless of the rate.
    if config.ice_to_snow_scheme == "m2005_autoconv":
        aggregation = ice_to_snow_m2005
    elif config.ice_to_snow_scheme == "mg_ferrier":
        # E3SM Morrison-Gettelman ice→snow autoconversion (micro_mg_utils.F90
        # ``ice_autoconversion``, Ferrier 1994): a FIXED 180-s timescale
        # conversion of the ice-PSD tail beyond D_cs — independent of
        # instantaneous supersaturation (unlike SAM PRCI). Reuses the LAMI/N0I
        # from the m2005 deposition block (MG flavor pairs mg_ferrier with
        # ice_deposition_scheme="m2005").
        #   d_rat = LAMI·DCS;  NPRCI = N0I/(LAMI·180)·exp(−d_rat);
        #   m_ip  = (ρ_ci·π/6)/LAMI³;
        #   PRCI  = m_ip·NPRCI·(((d_rat+3)·d_rat+6)·d_rat+6).
        # The product is algebraically PRCI = (ρ_ci π/6)·N0I/(180·LAMI⁴)·…;
        # with q_i = (ρ_ci π)·N0I/LAMI⁴ (Γ(4)=6 inverse-exponential PSD),
        # N0I/LAMI⁴ = q_i/(ρ_ci π) cancels the LAMI³/LAMI⁴ entirely:
        #   PRCI = (q_i/1080)·exp(−d_rat)·(((d_rat+3)d_rat+6)d_rat+6).
        # This avoids forming 1/LAMI³ (= 1/safe_pow(1e-30,3) = 1/1e-90, which
        # UNDERFLOWS float32 → Inf·0 = NaN when N_i→0; codex precision review),
        # is identical for N_i>0, and stays finite at LAMI=0 (→ q_i/180, the
        # bare 180-s timescale).
        d_rat = lami_ac * dcs
        ferrier = (jnp.clip(q_i, 0.0) / _FERRIER_ICE_NUMBER_DENOM) * jnp.exp(-d_rat) \
            * (((d_rat + 3.0) * d_rat + 6.0) * d_rat + 6.0)
        aggregation = jnp.where(
            (jnp.clip(q_i, 0.0) > 1.0e-14) & (T <= T_freeze), ferrier, 0.0)
    elif config.ice_to_snow_scheme == "heuristic":
        aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice
    else:
        raise ValueError(
            f"Unknown ice_to_snow_scheme: {config.ice_to_snow_scheme!r}. "
            f"Expected 'm2005_autoconv' (SAM PRCI, default), 'mg_ferrier' "
            f"(E3SM MG), or 'heuristic'."
        )

    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    melt_ice = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
    )
    # Snow melting: faithful SAM PSMLT (heat-balance-limited, PSD) when double-
    # moment; else the crude bulk rate. dt=dt applies PSMLT's sensible-heat cap
    # (can't over-cool below 0 °C, codex iter-29 D) + the mass donor clamp;
    # melt_snow is the SOLE q_s sink so the shared q_s clamp below is a no-op
    # (no double-clamp distortion). PSMLT melts faster well below the 0 °C
    # level and slower right at it (vs the bulk's constant rate).
    if snow_double_moment and config.do_snow_melting:
        melt_snow = snow_melting_psmlt(q_s, N_s, T, p_full, rho, config, dt=dt)
    else:
        melt_snow = jnp.minimum(
            config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
            jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
        )

    # 6b. Graupel melting PGMLT → rain (single-moment, heat-balance-limited) —
    #     the SAME ventilation form as snow PSMLT with the graupel PSD. dt
    #     applies the donor + sensible-heat caps; melt_graupel is the SOLE q_g
    #     sink so the sed extra_sink below accounts for it (no double-clamp).
    #     Tropical graupel mostly melts before reaching the surface.
    if config.do_graupel and config.do_graupel_melting:
        melt_graupel = graupel_melting_pgmlt(
            q_g, T, p_full, rho, config, dt=dt, N_g=N_g_arg)
    else:
        melt_graupel = jnp.zeros_like(jnp.clip(q_g, 0.0))

    # JOINT fusion heat limiter (codex iter-34 D): snow PSMLT + graupel PGMLT
    # each cap their OWN melt by the available above-freezing sensible heat, and
    # melt_ice has only a mass cap — but all three absorb L_f from the SAME
    # reservoir, so without a joint cap three species each melting near their
    # individual heat limit could cool the cell below 0 °C in one step. Scale
    # all three down proportionally so the SUM can't over-cool:
    #   (melt_ice+melt_snow+melt_graupel)·dt·L_f ≤ c_p·(T−T₀).
    total_melt = melt_ice + melt_snow + melt_graupel
    heat_avail = (constants.c_pd * jnp.maximum(T - constants.T_freeze, 0.0)
                  / (constants.L_f * jnp.maximum(dt, 1.0e-12)))
    melt_heat_scale = jnp.where(
        total_melt > heat_avail,
        heat_avail / jnp.maximum(total_melt, 1.0e-30), 1.0)
    melt_ice = melt_ice * melt_heat_scale
    melt_snow = melt_snow * melt_heat_scale
    melt_graupel = melt_graupel * melt_heat_scale

    # 7. Bigg immersion freezing of SUPERCOOLED rain (releases L_f). Critical
    #    for the phase + buoyancy of cold convective updrafts, which otherwise
    #    carry liquid rain far above the freezing level. SAM freezes the dense
    #    frozen drops to GRAUPEL (do_graupel, the default, iter-34); the legacy
    #    do_graupel=False path routes them to snow instead.
    if config.rain_freeze_scheme == "bigg":
        freeze_rain, freeze_N_r = rain_freezing_bigg(
            q_r, N_r, T, rho, config, dt=dt)
    elif config.rain_freeze_scheme == "none":
        freeze_rain = jnp.zeros_like(jnp.clip(q_r, 0.0))
        freeze_N_r = jnp.zeros_like(freeze_rain)
    else:
        raise ValueError(
            f"Unknown rain_freeze_scheme: {config.rain_freeze_scheme!r}. "
            f"Expected 'bigg' (SAM Bigg 1953, default) or 'none'."
        )

    # Graupel rain accretion PRACG (cold branch): falling graupel sweeps up rain
    # which FREEZES onto it (q_r→q_g + L_f). dt=None ⇒ the shared rain donor
    # clamp below caps it ONCE with the other q_r sinks (no double-clamp).
    if config.do_graupel and config.do_graupel_rain_accretion:
        pracg = graupel_rain_accretion_pracg(q_r, N_r, q_g, T, rho, config,
                                             dt=None, N_g=N_g_arg)
    else:
        pracg = jnp.zeros_like(jnp.clip(q_r, 0.0))

    # === JOINT DONOR CLAMP for the non-sedimentation rain MASS sinks ===
    # Evaporation, Bigg freezing and PRACG are each individually clamped to
    # q_r/dt, but their SUM could still exceed it (codex iter-24 D). Rescale all
    # by a common factor so the combined non-sed removal ≤ q_r/dt; sedimentation
    # then accounts for the scaled sum via its extra_sink. Mass is conserved
    # (the matching q_v / q_s / q_g sources scale too).
    rain_nonsed_sink = evaporation + freeze_rain + pracg
    rain_scale = donor_clamp_scale(jnp.clip(q_r, 0.0), rain_nonsed_sink, dt)
    evaporation = evaporation * rain_scale
    freeze_rain = freeze_rain * rain_scale
    freeze_N_r = freeze_N_r * rain_scale
    pracg = pracg * rain_scale

    # Route the Bigg-frozen rain by category: to GRAUPEL (SAM-faithful — dense
    # frozen drops) when do_graupel, else to SNOW (legacy). The L_f release is
    # identical either way (it stays in dT below); only the destination differs.
    # NUMBER routing: freeze_N_r always leaves the rain number; it arrives as
    # snow number on the legacy path (freeze_N_to_snow) and as GRAUPEL number
    # on the do_graupel path when a prognostic N_g is carried
    # (freeze_N_to_graupel in the graupel budget below, SAM NNUCCR).  Only
    # single-moment graupel — no N_g state — drops the frozen-drop number.
    if config.do_graupel:
        freeze_to_graupel = freeze_rain
        freeze_to_snow = jnp.zeros_like(freeze_rain)
        freeze_N_to_snow = jnp.zeros_like(freeze_N_r)
    else:
        freeze_to_graupel = jnp.zeros_like(freeze_rain)
        freeze_to_snow = freeze_rain
        freeze_N_to_snow = freeze_N_r

    # === DONOR CLAMP for q_i sinks ===
    # q_i sinks sharing this explicit-Euler step: aggregation, melt_ice
    # (both also sources elsewhere, via riming_i / dq_r), AND ice
    # SUBLIMATION (the negative branch of dq_i_dep).  Without the joint
    # clamp the combined sinks · dt can exceed available q_i — each is
    # individually bounded but their SUM is not — sending q_i negative;
    # sublimation in particular could also overlap sedimentation below
    # (sed reserves up to q_i/dt independently).  Backport of the Thompson
    # fix.  Mass is conserved because each sink rate appears once in dq_i_dt
    # as a sink and once elsewhere as a source — a uniform rescale of both
    # pieces preserves the budget.  dq_i_dep's negative (sublimation) branch
    # is FINAL here: the later qv clamp only rescales its positive
    # (deposition) branch, so max(-dq_i_dep, 0) is the true sublimation sink.
    ice_subl_sink = jnp.maximum(-dq_i_dep, 0.0)
    qi_sink_total = aggregation + melt_ice + ice_subl_sink
    qi_avail = jnp.clip(q_i, 0.0)
    qi_scale = donor_clamp_scale(qi_avail, qi_sink_total, dt)
    aggregation = aggregation * qi_scale
    melt_ice = melt_ice * qi_scale
    # Scale ONLY the sublimation (negative) branch of dq_i_dep; deposition
    # (positive, a q_i SOURCE) is unaffected.  The matching vapour source
    # (-dq_i_dep in dq_v_dt) and L_s heating read this same scaled value, so
    # mass + energy stay closed.
    dq_i_dep = jnp.where(dq_i_dep < 0.0, dq_i_dep * qi_scale, dq_i_dep)

    # Snow vapour deposition / sublimation PRDS must be known BEFORE the q_s
    # donor clamp so snow sublimation can be reserved against q_s jointly
    # with melt_snow (mirrors Thompson; the cloud-ice M1b tail is added so
    # the snow mass / latent-heat / vapour-sink budgets stay consistent).
    if snow_double_moment and config.do_snow_deposition:
        prds = snow_deposition_m2005(
            q_v, q_s, N_s, q_sat_i, T, p_full, rho, config, dt=dt)
    else:
        prds = jnp.zeros_like(jnp.clip(q_s, 0.0))
    prds = prds + dep_to_snow_m1b

    # === DONOR CLAMP for q_s sinks ===
    # q_s sinks sharing this explicit step: melt_snow AND snow SUBLIMATION
    # (the negative branch of prds).  Each is individually q_s-limited but
    # their SUM (plus sedimentation, reserved separately below) can exceed
    # q_s/dt.  Backport of the Thompson joint clamp.
    snow_subl_sink = jnp.maximum(-prds, 0.0)
    qs_sink_total = melt_snow + snow_subl_sink
    qs_avail = jnp.clip(q_s, 0.0)
    qs_scale = donor_clamp_scale(qs_avail, qs_sink_total, dt)
    melt_snow = melt_snow * qs_scale
    # Scale only the sublimation (negative) branch of prds; deposition
    # (positive, a q_s SOURCE) is vapour-limited later by the qv clamp.
    prds = jnp.where(prds < 0.0, prds * qs_scale, prds)

    # === DONOR CLAMP for q_c sinks ===
    # Scale q_c-consuming processes (autoconversion, accretion, Bergeron,
    # riming) by a common factor so the total loss per timestep does not
    # exceed available q_c.  Without this clamp, default rates at dt = 1200s
    # in a mixed-phase column drive q_c negative on a single explicit step
    # (bergeron alone gives bergeron_rate * q_c * dt = 1.2 * q_c).  Mass is
    # conserved because each process's matching source term in dq_r/dq_i/dq_s
    # gets the same scale factor (the rates appear once as sinks in dq_c and
    # once as sources elsewhere, so a uniform rescale preserves the budget).
    #
    # Include the *evaporation* branch of saturation adjustment (negative
    # ``condensation``) in the q_c sink budget — otherwise a subsaturated
    # clear-air column with q_c just above zero can lose more q_c to
    # evaporation + accretion combined than is available, going negative
    # (Codex audit cycle 2: "subsaturated clear air can create negative
    # cloud water").  ``saturation_adjustment`` has already donor-clamped
    # ``-condensation`` against ``q_c`` in isolation; including it here
    # makes the joint budget consistent when other q_c sinks are active.
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    # Homogeneous freezing of cloud water below ≈−40 °C (SAM module_mp_graupel
    # .f90:4661-4669): all supercooled cloud water freezes to cloud ice, the
    # droplet number becomes ice number, and L_f is released. Smooth sigmoid
    # threshold for AD (SAM uses a hard T≤233.15 switch). The rain analog is
    # the steep Bigg rate (iter-24). Gated by ``do_homogeneous_freezing``.
    if config.do_homogeneous_freezing:
        homo_frac = jax.nn.sigmoid(
            config.homogeneous_freeze_sharpness
            * (config.homogeneous_freeze_T - T))
        homo_freeze_c = (homo_frac * jnp.clip(q_c, 0.0)
                         / jnp.maximum(dt, 1.0e-10))
        # ICE-NUMBER source from homogeneous droplet freezing, as a RELAXATION
        # TOWARD THE DROPLET-NUMBER TARGET — NOT a per-step rate.
        #
        # The mass rate ``homo_freeze_c`` self-gates (∝ q_c → 0 in clear air).
        # The NUMBER, though, uses ``N_c_eff`` which under specified-Nc
        # (predict_Nc=False, or zero prognostic N_c) is the CONSTANT ``Nc_0``
        # (1e8 /m³). The original ``homo_frac·N_c_eff/ρ/dt`` form injected
        # ~Nc_0/ρ (≈1e8 /kg) ice crystals EVERY step into any cold cloudy
        # layer. Because specified N_c is never depleted, convective columns
        # that keep condensing trace q_c re-froze the full Nc_0 population each
        # step, so N_i accumulated without bound (~5e8 /kg per step) and
        # overflowed fp32 to NaN within ~5 steps of an RCEMIP restart — even
        # with a non-negativity floor and a LAMI cap (the cap's 1 µm ceiling
        # permits ~1e10 /kg, far above where the deposition/fall-speed terms
        # break down).
        #
        # Physically you can only freeze the droplets that EXIST: the ice
        # number contributed by homogeneous freezing approaches the droplet
        # concentration and then STOPS. Mirror the Cooper-nucleation
        # relaxation form (``clip(target − N_i, 0)/dt``) so the source
        # saturates at the per-mass droplet target instead of firing forever.
        # Gated on cloud presence (SAM's ``IF QC3D ≥ QSMALL``).
        # The droplet-number target is additionally capped at the ice-number
        # ceiling ``N_i_nuc_max`` (SAM 500 /L). The specified droplet number
        # (``Nc_0`` = 1e8 /m³, continental) is FAR above any realistic ice
        # crystal concentration; converting it wholesale gives N_i ~ 3e8 /kg,
        # which — though now bounded — still drives the deposition / fall-speed
        # terms past the fp32-stable range. Capping the FROZEN ice number at
        # the same physical ceiling Cooper nucleation already uses keeps N_i at
        # the proven-stable ~5e6 /kg scale.
        homo_target_N = jnp.minimum(
            jnp.clip(N_c_eff, 0.0), config.N_i_nuc_max
        ) / jnp.clip(rho, _RHO_FLOOR)
        # Count the crystals nucleated THIS step against the same ceiling.  In
        # cold cloudy air (T < homogeneous_freeze_T = 233 K with q_c > 0, so
        # RH_liq ~ 1 => RH_ice ~ 1.54 > S_hom(228 K) = 1.47) droplet freezing
        # and cirrus nucleation BOTH fire, and both relaxed toward their target
        # from the OLD N_i, stacking to ~1.5x the intended ice-number ceiling
        # (same double-count class as the Cooper/homogeneous one fixed at :414).
        # ``dN_i_nuc`` here already carries Cooper + homogeneous (folded at
        # :587).  ON-path only so the default graph is byte-identical; the same
        # correction is warranted for Cooper alone on the default path but that
        # is a compatibility-sensitive change for in-flight runs, deferred.
        if config.homogeneous_ice_nucleation:
            _homo_deficit = jnp.clip(
                homo_target_N - jnp.clip(N_i, 0.0)
                - dN_i_nuc * jnp.clip(dt, 1.0), 0.0)
        else:
            _homo_deficit = jnp.clip(homo_target_N - jnp.clip(N_i, 0.0), 0.0)
        homo_freeze_N = jnp.where(
            jnp.clip(q_c, 0.0) > 1.0e-14,
            homo_frac * _homo_deficit / jnp.maximum(dt, 1.0e-10),
            0.0,
        )
    else:
        homo_freeze_c = jnp.zeros_like(jnp.clip(q_c, 0.0))
        homo_freeze_N = jnp.zeros_like(homo_freeze_c)
    qc_sink_total = (dq_c_au + dq_c_ac + bergeron + riming_i + riming_s
                     + riming_g + cond_evap_sink + homo_freeze_c)
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    homo_freeze_c = homo_freeze_c * qc_scale
    homo_freeze_N = homo_freeze_N * qc_scale
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
    riming_g = riming_g * qc_scale
    # PGSACW: heavily-rimed snow converts to graupel. Uses the DONOR-CLAMPED
    # riming_s (so PGSACW ≤ the actual snow riming). It REDIRECTS a portion of
    # the riming to graupel — q_c still loses the full riming_s, snow gains
    # (riming_s − pgsacw), graupel gains pgsacw (mass conserved). The snow loses
    # number NSCNG = ρ_sn/(ρ_g−ρ_sn)·pgsacw/MG0·ρ (bounded by N_s/dt).
    if (snow_double_moment and config.do_graupel
            and config.do_snow_to_graupel):
        pgsacw = snow_to_graupel_pgsacw(
            q_c, q_s, N_s, riming_s, rho, config, dt)
        # NSCNG snow-number sink = (ρ_sn/(ρ_g−ρ_sn)·PGSACW)/MG0. PGSACW is a
        # MIXING-ratio rate [kg/kg/s] and MG0 the embryo mass [kg], so DUM/MG0 is
        # already a PER-MASS number rate [1/(kg·s)] — matching legoESM's per-mass
        # N_s (LAMS has no ρ_air). SAM's extra ·RHO would make it per-VOLUME;
        # dropped here (codex iter-38 D). Capped at N_s/dt.
        nscng = jnp.minimum(
            (config.rho_snow / (config.rho_graupel - config.rho_snow)
             * pgsacw / config.graupel_embryo_mass),
            jnp.clip(N_s, 0.0) / jnp.maximum(dt, 1.0e-12))
    else:
        pgsacw = jnp.zeros_like(jnp.clip(q_c, 0.0))
        nscng = jnp.zeros_like(pgsacw)
    # Scale the evaporation branch by the same factor: when condensation
    # is negative (evaporation), reduce its magnitude proportionally so
    # q_c doesn't go negative.  When condensation is positive
    # (saturation adjustment from supersaturation), the scaling does
    # nothing because ``cond_evap_sink = 0``.
    condensation = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )
    # Number tendency for autoconverted droplets must scale identically.
    dN_r_au = dN_r_au * qc_scale

    # === DONOR CLAMP for q_v sinks ===
    # The vapor budget in this scheme is ``dq_v_dt = -condensation +
    # evaporation - dq_i_dep``.  Positive ``condensation`` and positive
    # ``dq_i_dep`` together remove vapor; if their combined rate · dt
    # exceeds available ``q_v``, the explicit step drives q_v < 0.
    # ``saturation_adjustment`` clamps ``condensation`` against q_v in
    # isolation, but in a supersaturated icy layer the ice deposition
    # ``dq_i_dep`` can still over-draw q_v on its own or jointly with
    # condensation.  Mirror the q_c / q_i clamps: rescale all positive
    # vapor sinks (and their matching sources / latent heat) so the
    # combined removal cannot exceed the locally-available vapor mass.
    # Codex iter-25 finding #3.
    # Vapor donor clamp via the shared AD-safe helper
    # (donor_clamp_scale).  The helper's ``divisor_floor`` (default
    # 1e-15) keeps the VJP bounded under fp32 even for tiny
    # supersaturated icy layers where the sink is small but positive.
    # Snow vapour deposition/sublimation PRDS (incl. the M1b cloud-ice tail)
    # is now computed ABOVE, before the q_s donor clamp, so snow sublimation
    # is reserved against q_s jointly with melt_snow.  Its positive
    # (deposition) branch is vapour-limited just below; its negative
    # (sublimation) branch was already snow-limited by the q_s clamp.
    # Graupel vapor deposition/sublimation PRDG (the graupel analog of PRDS):
    # deposition (positive) joins the vapour-sink budget below; sublimation
    # (negative) is already graupel-limited inside the helper.
    if config.do_graupel and config.do_graupel_deposition:
        prdg = graupel_deposition_prdg(
            q_v, q_g, q_sat_i, T, p_full, rho, config, dt=dt, N_g=N_g_arg)
    else:
        prdg = jnp.zeros_like(jnp.clip(q_g, 0.0))
    if config.homogeneous_ice_nucleation:
        # === JOINT ICE-PHASE DEPOSITION BOUND === (ON-path only => the OFF
        # operation graph is unchanged.)
        # Cloud-ice deposition, snow PRDS and graupel PRDG all deposit vapour
        # onto ICE and all relax toward the SAME equilibrium q_sat_i, so their
        # SUM cannot exceed the available ice supersaturation in one explicit
        # step.  The per-branch cap at :502 bounds only cloud ice; with
        # pre-existing snow/graupel the parallel sinks push the total past it
        # (measured 228 K/222 hPa, dt=240 s, q_g = 1 g/kg: one step took
        # RH_ice 2.88 -> 0.52, a 48 % undershoot below ice saturation that the
        # next step partly sublimated back).  Liquid ``condensation`` is
        # deliberately EXCLUDED: it targets the LIQUID curve q_sat_l > q_sat_i,
        # so it is a physically distinct equilibrium, not part of this budget.
        # This is the ice-phase analogue of the q_c / q_i / q_v donor clamps and
        # uses the same AD-safe helper.
        # SCOPE (do not overstate): this is a FIXED-TEMPERATURE limiter on those
        # THREE deposition terms, NOT a guarantee that the step ends at
        # RH_ice >= 1.  The MI0 nucleation seed mass is a separate sink, liquid
        # condensation relaxes to a DIFFERENT curve, and the L_s release moves
        # q_sat_i itself (+1.24 K / +15.7 % q_sat_i in the q_g = 1 g/kg case).
        _ice_dep_total = (jnp.maximum(dq_i_dep, 0.0) + jnp.maximum(prds, 0.0)
                          + jnp.maximum(prdg, 0.0))
        _ice_dep_scale = donor_clamp_scale(
            jnp.maximum(q_v - q_sat_i, 0.0), _ice_dep_total, dt)
        # Scale ONLY the positive (deposition) branches; the negative
        # (sublimation) branches are condensate-limited, not vapour-limited,
        # and cloud-ice sublimation already carries its own symmetric bound.
        dq_i_dep = jnp.where(dq_i_dep > 0.0, dq_i_dep * _ice_dep_scale, dq_i_dep)
        prds = jnp.where(prds > 0.0, prds * _ice_dep_scale, prds)
        prdg = jnp.where(prdg > 0.0, prdg * _ice_dep_scale, prdg)
    cond_pos = jnp.maximum(condensation, 0.0)
    # Nucleation mass source dq_i_nuc also consumes vapor (deposition onto
    # new crystals), so include it among the positive vapor sinks.
    qv_sink_total = (cond_pos + jnp.maximum(dq_i_dep, 0.0) + dq_i_nuc
                     + jnp.maximum(prds, 0.0) + jnp.maximum(prdg, 0.0))
    qv_avail = jnp.clip(q_v, 0.0)
    qv_scale = donor_clamp_scale(qv_avail, qv_sink_total, dt)
    # Scale only the positive (vapor-consuming) branch of condensation;
    # negative condensation (evaporation) is unaffected.
    condensation = jnp.where(condensation > 0.0, condensation * qv_scale, condensation)
    # Scale only the positive (vapor-consuming) DEPOSITION; sublimation
    # (negative dq_i_dep) is ice-limited not vapor-limited (Codex iter-8).
    dq_i_dep = jnp.where(dq_i_dep > 0.0, dq_i_dep * qv_scale, dq_i_dep)
    # Same for the SNOW deposition PRDS (positive = vapour-limited; negative
    # sublimation is already snow-limited).
    prds = jnp.where(prds > 0.0, prds * qv_scale, prds)
    # ... and GRAUPEL deposition PRDG (positive vapour-limited; sublimation
    # graupel-limited). This scaling happens HERE, before every downstream use
    # of prdg — the sed extra_sink, the L_s heating, dq_v_dt and dq_g_dt all read
    # this SAME post-scale prdg, so vapour⇌graupel mass stays conserved when the
    # qv clamp fires (codex iter-36 D). Only the positive (deposition) branch is
    # vapour-limited; the negative (sublimation) branch is already q_g-limited.
    prdg = jnp.where(prdg > 0.0, prdg * qv_scale, prdg)
    # Nucleation is vapour-limited (deposition of MI0 seed mass onto new
    # crystals): scale BOTH the mass AND the number by the same qv_scale so
    # the SAM relation MNUCCD = NNUCCD·MI0 is preserved under the donor
    # clamp (Codex iter-9: don't create number without its seed mass).
    dq_i_nuc = dq_i_nuc * qv_scale
    dN_i_nuc = dN_i_nuc * qv_scale

    # === SEDIMENTATION ===
    # Faithfulness: the m2005_psd mass-/number-weighted fall speeds are pinned to
    # the gSAM MICRO_M2005 Fortran oracle to round-off (rel 1e-12) in
    # tests/unit/test_m2005_fall_speed_faithful.py — rain/ice mass (UMR/UMI) via
    # the sedimentation surface flux precip = V_t*q*rho, and every species' mass
    # AND number speed (UNR/UNI/UMS/UNS/UMG/UNG, snow+graupel in double-moment
    # mode) via a sedimentation_tendency intercept.  NOTE `config` is already
    # flavor-resolved here (line ~178): `config.fall_b_i` is 0.865 (gSAM MK tune)
    # for morrison_flavor="sam" but 1.0 (M2005-original) for the DEFAULT "mg" —
    # the runtime default ice fall exponent is NOT the raw-default 0.865.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, _RHO_FLOOR)
    if config.fall_speed_scheme == "m2005_psd":
        # Faithful SAM M2005 mass-weighted PSD fall speeds for the DOUBLE-
        # MOMENT species (rain via N_r, cloud ice via N_i; slopes
        # module_mp_graupel.f90:1640 / 2549, UM 1854-1855 / 4230-4238):
        #   LAMR = (π·ρ_w·N_r/q_r)^⅓,  LAMI = (ρ_ci·π·N_i/q_i)^⅓   (per-mass
        #     N[/kg], q[kg/kg]; NO ρ factor — matches the EFFI/deposition slope)
        #   UM   = a·Γ(4+b)/6 · LAM^−b · (ρ_su/ρ)^0.54
        # clamped to SAM's slope + "realistic fallspeed" limits. Snow/graupel use
        # this SAM PSD fall speed only when a prognostic N_s/N_g is supplied (the
        # snow_double_moment / graupel_double_moment branches below); the DEFAULT
        # single-moment path keeps the legacy bulk q-power V_t (snow) / fixed-N0G
        # closure (graupel).
        dum = safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXP)
        # Cloud ICE uses the Ikawa-Saito 1991 density exponent 0.35 (gSAM
        # AIN=(RHOSU/RHO)^0.35·AI, module_mp_graupel.f90:1542/4230), NOT the
        # 0.54 Heymsfield-Bensemer that rain/snow use (ARN/ASN=DUM·a) — so ice
        # falls slightly SLOWER in thin upper-trop air than the 0.54 form (the
        # 0.54 over-sped the anvil ice ~20-30%; iter-209 fix, pairs with #6).
        dum_i = safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXP_ICE)
        cons4 = math.gamma(4.0 + config.fall_b_r) / 6.0
        cons28 = math.gamma(4.0 + config.fall_b_i) / 6.0
        qr_pos = jnp.clip(q_r, 0.0)
        qi_pos = jnp.clip(q_i, 0.0)
        # Floor q in the slope denominator (empty cell ⇒ large LAM ⇒ small
        # drops ⇒ clamp to LAM_max ⇒ V_t masked to 0 below). The 1e-20 floor
        # is a pure AD/zero-division safety floor (jnp.maximum gates the grad).
        # UNIT NOTE: SAM's LAMR/LAMI use PER-MASS number [#/kg]. In legoESM
        # N_i IS per-mass (Cooper nucleation divides by ρ), but N_r is stored
        # PER-VOLUME [#/m³] (the warm-rain autoconversion source dN_r∝ρ/x* and
        # the self-collection mean-drop mass q_r·ρ/N_r both treat it per-vol).
        # So LAMR converts N_r→per-mass via N_r/ρ (extra ρ in the denominator),
        # while LAMI uses N_i/q_i directly. QSMALL=1e-14 matches SAM.
        lamr = safe_pow(
            jnp.pi * constants.rho_water * jnp.clip(N_r, 0.0)
            / jnp.maximum(rho * qr_pos, 1.0e-20), 1.0 / 3.0)
        lamr = jnp.clip(lamr, config.lamr_min, config.lamr_max)
        V_t_r = config.fall_a_r * cons4 * safe_pow(lamr, -config.fall_b_r) * dum
        V_t_r = jnp.minimum(V_t_r, _VT_CAP_RAIN * dum)
        V_t_r = jnp.where(qr_pos > 1.0e-14, V_t_r, 0.0)
        lami = safe_pow(
            config.rho_cloud_ice * jnp.pi * jnp.clip(N_i, 0.0)
            / jnp.maximum(qi_pos, 1.0e-20), 1.0 / 3.0)
        lami = jnp.clip(lami, config.lami_min, config.lami_max)
        V_t_i = config.fall_a_i * cons28 * safe_pow(lami, -config.fall_b_i) * dum_i
        dum_i_cap = _VT_CAP_SNOW_ICE * dum_i
        V_t_i = jnp.minimum(V_t_i, dum_i_cap)
        V_t_i = jnp.where(qi_pos > 1.0e-14, V_t_i, 0.0)
        if snow_double_moment:
            # SAM PSD snow fall speeds (module_mp_graupel.f90:1719/1854-1856):
            #   LAMS = (π·ρ_sn·N_s/q_s)^⅓ (per-mass N_s), clamped;
            #   UMS = AS·Γ(4+BS)/6·LAMS^−BS·(ρ_su/ρ)^0.54  (mass), cap 1.2·dum;
            #   UNS = AS·Γ(1+BS)·LAMS^−BS·(ρ_su/ρ)^0.54     (number).
            qs_pos = jnp.clip(q_s, 0.0)
            lams = safe_pow(
                config.rho_snow * jnp.pi * jnp.clip(N_s, 0.0)
                / jnp.maximum(qs_pos, 1.0e-20), 1.0 / 3.0)
            lams = jnp.clip(lams, config.lams_min, config.lams_max)
            cons3_s = math.gamma(4.0 + config.fall_b_s) / 6.0
            cons5_s = math.gamma(1.0 + config.fall_b_s)
            V_t_s = config.fall_a_s * cons3_s * safe_pow(
                lams, -config.fall_b_s) * dum
            V_t_s = jnp.minimum(V_t_s, _VT_CAP_SNOW_ICE * dum)
            V_t_s = jnp.where(qs_pos > 1.0e-14, V_t_s, 0.0)
            V_n_s = config.fall_a_s * cons5_s * safe_pow(
                lams, -config.fall_b_s) * dum
            V_n_s = jnp.minimum(V_n_s, _VT_CAP_SNOW_ICE * dum)
            V_n_s = jnp.where(qs_pos > 1.0e-14, V_n_s, 0.0)
        else:
            V_t_s = config.a_v_s * safe_pow(
                jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
            V_t_s = jnp.clip(V_t_s, 0.0, _VT_CLIP_FROZEN)
            V_n_s = V_t_s
        # NUMBER-weighted fall speeds UNR/UNI (SAM module_mp_graupel.f90:
        # 1857/4237 rain, 4230 ice; caps 4268-4271): UN = a·Γ(1+b)/6·… no —
        # UNR=ARN·CONS6/LAMR^BR with CONS6=Γ(1+BR), UNI=AIN·CONS27/LAMI^BI with
        # CONS27=Γ(1+BI). Number falls SLOWER than mass (UNR/UMR=Γ(1+b)/(Γ(4+b)/6)
        # ≈0.31), so number sediments with size-sorting (big particles, which
        # carry more mass, fall faster). Same realistic-fallspeed caps as mass.
        cons6 = math.gamma(1.0 + config.fall_b_r)
        cons27 = math.gamma(1.0 + config.fall_b_i)
        V_n_r = config.fall_a_r * cons6 * safe_pow(lamr, -config.fall_b_r) * dum
        V_n_r = jnp.minimum(V_n_r, _VT_CAP_RAIN * dum)
        V_n_r = jnp.where(qr_pos > 1.0e-14, V_n_r, 0.0)
        V_n_i = config.fall_a_i * cons27 * safe_pow(lami, -config.fall_b_i) * dum_i
        V_n_i = jnp.minimum(V_n_i, dum_i_cap)
        V_n_i = jnp.where(qi_pos > 1.0e-14, V_n_i, 0.0)
    elif config.fall_speed_scheme == "bulk_qpower":
        # Legacy Marshall-Palmer bulk: V_t = a_v·(q·ρ/ρ_sfc)^b_v with
        # fractional exponents; safe_pow guards cold-start (q=0) AD.
        V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
        V_t_r = jnp.clip(V_t_r, 0.0, _VT_CLIP_RAIN)
        V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
        V_t_i = jnp.clip(V_t_i, 0.0, _VT_CLIP_FROZEN)
        V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
        V_t_s = jnp.clip(V_t_s, 0.0, _VT_CLIP_FROZEN)
        # Bulk scheme: number sediments at the mass fall speed (no PSD size-
        # sorting available without the PSD slope).
        V_n_r, V_n_i, V_n_s = V_t_r, V_t_i, V_t_s
    else:
        raise ValueError(
            f"Unknown fall_speed_scheme: {config.fall_speed_scheme!r}. "
            f"Expected 'm2005_psd' (SAM PSD mass-weighted, default) or "
            f"'bulk_qpower' (legacy)."
        )

    # Graupel fall speed. Default is the single-moment fixed-N0G intercept
    # closure (graupel_lamg with N_g=None); a prognostic N_g selects the SAM
    # double-moment PSD slope (LAMG=(π·ρ_g·N_g/q_g)^⅓). UMG = AG·Γ(4+BG)/6·
    # LAMG^(−BG)·(ρ_su/ρ)^0.54, capped at a realistic graupel terminal
    # velocity (graupel falls FAST — denser than snow).
    if config.do_graupel:
        dum_g = safe_pow(config.rho_su / jnp.clip(rho, _RHO_FLOOR), _FALL_RHO_EXP)
        lamg = graupel_lamg(q_g, rho, config, N_g=N_g_arg)
        cons7_g = math.gamma(4.0 + config.fall_b_g) / 6.0
        V_t_g = config.fall_a_g * cons7_g * safe_pow(
            lamg, -config.fall_b_g) * dum_g
        V_t_g = jnp.minimum(V_t_g, _VT_CAP_GRAUPEL * dum_g)
        V_t_g = jnp.where(jnp.clip(q_g, 0.0) > 1.0e-14, V_t_g, 0.0)
        # NUMBER-weighted graupel fall speed UNG = AG·Γ(1+BG)·LAMG^(−BG)·dum
        # (slower than mass ⇒ size-sorting), only needed for double-moment N_g.
        cons8_g = math.gamma(1.0 + config.fall_b_g)
        V_n_g = config.fall_a_g * cons8_g * safe_pow(
            lamg, -config.fall_b_g) * dum_g
        V_n_g = jnp.minimum(V_n_g, _VT_CAP_GRAUPEL * dum_g)
        V_n_g = jnp.where(jnp.clip(q_g, 0.0) > 1.0e-14, V_n_g, 0.0)
    else:
        V_t_g = jnp.zeros_like(jnp.clip(q_g, 0.0))
        V_n_g = V_t_g

    # Joint donor caps: each `extra_sink` is the in-column sink that
    # shares the same explicit-Euler step as sedimentation.
    # Without these, the post-donor-clamp in-column sinks (aggregation +
    # melt_ice for q_i, melt_snow for q_s, evaporation for q_r) ALREADY
    # consume up to q/dt, AND sed independently can drain another q/dt
    # — driving the pool negative.  Mirrors the iter-29 q_r/evap fix.
    sed_r, precip_r = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation + freeze_rain + pracg,
    )
    sed_i, precip_i = sedimentation_tendency(
        q_i, rho, V_t_i, dz, dt=dt, return_surface_flux=True,
        extra_sink=aggregation + melt_ice + jnp.maximum(-dq_i_dep, 0.0),
    )
    sed_s, precip_s = sedimentation_tendency(
        q_s, rho, V_t_s, dz, dt=dt, return_surface_flux=True,
        extra_sink=melt_snow + jnp.maximum(-prds, 0.0),
    )
    # Graupel sedimentation. In-column q_g sinks sharing the step are melting
    # AND sublimation (the negative PRDG branch); both reserve mass so sed +
    # melt + sublimation ≤ q_g/dt (deposition is a SOURCE, not in the clamp).
    sed_g, precip_g = sedimentation_tendency(
        q_g, rho, V_t_g, dz, dt=dt, return_surface_flux=True,
        extra_sink=melt_graupel + jnp.maximum(-prdg, 0.0),
    )
    # NUMBER sedimentation (so the rain/ice number falls WITH the mass and the
    # PSD stays consistent in a column). N_i is per-mass ⇒ the same flux form
    # as q (sedimentation_tendency directly). N_r is per-VOLUME ⇒ sediment
    # N_r/ρ (per-mass) and multiply the tendency back by ρ; the SAME ρ_eff is
    # used for the divide AND the routine so the flux is exactly V·N_r in all
    # layers (codex iter-23 A — a bare clip in only the divide would corrupt
    # the flux where ρ<0.1). ρ_eff is constant over the step ⇒ ρ·d(N_r/ρ)/dt =
    # dN_r/dt and the per-volume flux divergence is exact.
    rho_eff = jnp.maximum(rho, _RHO_FLOOR)
    sed_N_r = rho_eff * sedimentation_tendency(
        jnp.clip(N_r, 0.0) / rho_eff, rho_eff, V_n_r, dz, dt=dt)
    # Ice-NUMBER sinks sharing this step with number sedimentation — melt,
    # sublimation (both mass-proportional; melt_ice / dq_i_dep are FINAL
    # donor-clamped values here) and the ice->snow autoconversion number
    # transfer.  Computed BEFORE the sedimentation call so their sum is
    # RESERVED via extra_sink: without the joint reservation, sedimentation
    # could remove all N_i on top of these sinks, the non-negativity floor
    # would clamp, and the post-step LAMI lower bound would then RECREATE
    # number to match surviving q_i — an artificial ice-number source with
    # no phase-transfer counterpart (codex 2026-07-28 round 2).
    _qi_floor = jnp.clip(q_i, 1.0e-15)
    _ni_pos = jnp.clip(N_i, 0.0)
    dN_i_melt = melt_ice * _ni_pos / _qi_floor
    dN_i_subl = jnp.maximum(-dq_i_dep, 0.0) * _ni_pos / _qi_floor
    if config.ice_to_snow_scheme in ("m2005_autoconv", "mg_ferrier"):
        _cons22 = (
            jnp.pi * config.rho_cloud_ice
            * config.ice_snow_d_auto ** 3 / 6.0
        )
        dN_i_autoconv = jnp.minimum(
            aggregation / _cons22,
            _ni_pos / jnp.clip(dt, 1.0),
        )
    else:
        dN_i_autoconv = aggregation * _ni_pos / _qi_floor
    sed_N_i = sedimentation_tendency(
        jnp.clip(N_i, 0.0), rho, V_n_i, dz, dt=dt,
        extra_sink=dN_i_melt + dN_i_subl + dN_i_autoconv)
    if snow_double_moment:
        # N_s is per-mass ⇒ same flux form as q (UNS number-weighted speed).
        sed_N_s = sedimentation_tendency(
            jnp.clip(N_s, 0.0), rho, V_n_s, dz, dt=dt)
    if graupel_double_moment:
        # N_g per-mass ⇒ same flux form as q (UNG number-weighted speed).
        sed_N_g = sedimentation_tendency(
            jnp.clip(N_g, 0.0), rho, V_n_g, dz, dt=dt)

    # === LATENT HEATING ===
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd

    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        + L_s * (dq_i_dep + dq_i_nuc + prds + prdg) / c_pd  # +snow/graupel dep
        # Cloud water → ice/snow freezing releases latent heat of fusion
        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
        # riming is supercooled-droplet capture by ice/snow.  Both are
        # phase changes that release L_f; the moist-enthalpy invariant
        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
        # column conservation.  Magnitude estimate: ~2 K/day at default
        # rates in mixed-phase clouds.
        + L_f * (bergeron + riming_i + riming_s + riming_g) / c_pd
        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
        + L_f * freeze_rain / c_pd          # Bigg rain freezing releases L_f
        + L_f * pracg / c_pd                # rain frozen onto graupel (PRACG)
        + L_f * homo_freeze_c / c_pd        # cloud-water homogeneous freezing
    )

    # === COMBINE TENDENCIES ===
    dq_v_dt = -condensation + evaporation - dq_i_dep - dq_i_nuc - prds - prdg
    dq_c_dt = (
        condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
        - riming_g - homo_freeze_c
    )
    dq_r_dt = (
        dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel
        - freeze_rain - pracg + sed_r
    )
    dq_i_dt = (
        dq_i_dep + dq_i_nuc + bergeron + riming_i - aggregation
        - melt_ice + homo_freeze_c + sed_i
    )
    # Snow riming grows snow EXCEPT the portion PGSACW that converts to graupel.
    dq_s_dt = (aggregation + (riming_s - pgsacw) - melt_snow + freeze_to_snow
               + prds + sed_s)
    # Graupel (single-moment): gains the Bigg-frozen rain + cloud-water riming
    # (PSACWG) + rain accretion (PRACG) + snow→graupel conversion (PGSACW) +
    # vapor deposition (PRDG, can be negative=sublimation), loses melt → rain,
    # sediments (fast).
    dq_g_dt = (freeze_to_graupel + riming_g + pracg + pgsacw + prdg
               - melt_graupel + sed_g)

    # AD-safe number-concentration tendencies (issue #249).
    #
    # ``dN_c_dt = -dq_c_au · ρ / x_c``: ``x_c = q_c·ρ/N_c`` has no
    # upstream clip (``effective_Nc`` clips ``N_c`` only), so ``x_c``
    # can land anywhere in ``(0, ∞)`` including the AD-unsafe range
    # ``(1e-20, 1e-15)`` where the legacy ``clip(x_c, 1e-20)`` floor
    # was inactive but ``-dq_c_au·ρ / x_c²`` cotangents reached
    # ``∼1e30`` — the dominant NaN-gradient source in the issue's
    # repro.  ``safe_divide(eps=1e-15)`` masks the unphysical residue
    # below the cloud-water scale (``q_c=1e-7 kg/kg`` → ``x_c ≈ 1e-15
    # kg``); cells that legitimately contain cloud stay in the divide
    # branch.  ``dq_c_au ∝ q_c² ∝ x_c²`` also vanishes there, so
    # ``fill=0.0`` matches the physical limit.
    #
    # ``dN_i_dt`` keeps the legacy ``clip(q_i, 1e-15) + divide`` form:
    # the floor is large enough relative to ``aggregation ∝ q_i`` that
    # the divide's cotangent stays bounded, and ``clip``'s zero VJP in
    # the floor-active branch already breaks the AD propagation.
    # ``safe_divide`` here would lose the trace-positive
    # ``q_i ∈ (0, 1e-15)`` scaling of the legacy expression
    # (``aggregation · N_i / 1e-15 ∝ q_i``); the per-mass-rate rewrite
    # is also non-equivalent in that regime (codex round 4).  Issue
    # #249 listed this as a vanishing-numerator site, but the
    # ``q_i``-floor clip already saves the AD path.
    # Cloud-droplet NUMBER sink from RIMING (SAM NPSACWS/NPSACWG, codex iter-35
    # B): ice/snow/graupel each sweep up cloud droplets at the SAME collection
    # kernel for number as for mass, so the number removed per unit time is the
    # rimed-mass fraction × N_c (NPSACW/PSACW = N_c/q_c). The riming masses are
    # already donor-clamped (qc_scale ⇒ Σ sinks ≤ q_c/dt), so this is bounded by
    # N_c/dt. Without it, riming would shrink the mean droplet size (lower q_c,
    # same N_c) and spuriously slow autoconversion in mixed-phase cloud.
    dN_c_riming = ((riming_i + riming_s + riming_g)
                   * jnp.clip(N_c, 0.0) / jnp.clip(q_c, 1e-15))
    if getattr(config, "predict_Nc", False):
        dN_c_dt = safe_divide(-dq_c_au * rho, x_c, eps=1e-15) - dN_c_riming
    else:
        # SAM dopredictNc=.false.: droplet number is the specified constant Nc_0
        # (used via N_c_eff); the prognostic field is NOT evolved, so no sink can
        # drive it negative.
        dN_c_dt = jnp.zeros_like(q_c)
    # Rain-number loss during evaporation (SAM NSUBR, module_mp_graupel.f90:
    # 2177-2181): NSUBR = (PRE·dt/q_r)·N_r/dt = −evap·N_r/q_r — the same
    # fraction of number removed as mass, so the mean drop size is preserved
    # and N_r does not go stale as rain evaporates. evap is already donor-
    # clamped to q_r/dt, so the removed fraction ≤ 1 (SAM's MAX(-1,DUM)).
    dN_r_evap = evaporation * jnp.clip(N_r, 0.0) / jnp.clip(q_r, 1e-15)
    # NSMLTR / NGMLTR (SAM module_mp_graupel.f90:2189-2203, applied at :2211
    # NR3DTEN += (NSUBR - NSMLTR - NGMLTR)): every melted snow flake / graupel
    # particle becomes a RAIN drop — number is conserved across the phase
    # change, at the same fractional rate as the melted mass.  These are also
    # the NSMLTS/NGMLTG sinks of the snow/graupel number budgets below (single
    # owner: defined once here, reused there).  melt_snow / melt_graupel are
    # FINAL at this point (joint fusion-heat scale + donor clamps applied
    # above).  N_s/N_g are per-mass [1/kg] while N_r is per-volume [1/m^3],
    # hence the *rho on the rain-side source.  Without this the melted number
    # simply vanished: rain in melting layers under-counted drops -> too-large
    # mean size -> too-fast fallout / too-little evaporation.
    melt_N_to_rain = jnp.zeros_like(q_r)
    # NMLTR from cloud ice: melted crystals become rain drops (the Morrison
    # reference transfers this number; snow/graupel below already did — the
    # cloud-ice channel was missing, codex 2026-07-28).  dN_i_melt is
    # per-mass; *rho converts to the per-volume rain number.
    melt_N_to_rain = melt_N_to_rain + dN_i_melt * rho
    if snow_double_moment:
        dN_s_melt = melt_snow * jnp.clip(N_s, 0.0) / jnp.clip(q_s, 1e-15)
        melt_N_to_rain = melt_N_to_rain + dN_s_melt * rho
    if graupel_double_moment:
        dN_g_melt = melt_graupel * jnp.clip(N_g, 0.0) / jnp.clip(q_g, 1e-15)
        melt_N_to_rain = melt_N_to_rain + dN_g_melt * rho
    # Bigg freezing removes the frozen rain drops from the rain number.
    dN_r_dt = (dN_r_au + dN_r_selfcoll - dN_r_evap - freeze_N_r + sed_N_r
               + melt_N_to_rain)
    # Ice-number sink from ice→snow autoconversion. For SAM PRCI the removed
    # crystals are DCS-SIZED, so NPRCI = PRCI / m_DCS (m_DCS = CONS22 =
    # π·ρ_ci·DCS³/6), clamped to N_i/dt (codex iter-20 B) — NOT the mean-mass
    # rate, which would over-remove number. The heuristic scheme keeps the
    # legacy mean-mass form (clip(q_i,1e-15) floor preserves the AD path).
    # +homo_freeze_N: cloud droplets that homogeneously freeze become ice
    # crystals (SAM NI3D += NC3D).
    #
    # Number leaves WITH the mass (SAM: melting/sublimation deplete NI3D
    # alongside QI3D).  Without these sinks, sedimented crystals melt or
    # sublimate their MASS away while their NUMBER accumulates forever —
    # measured as century5's residual Ni^max growth engine (argmax at
    # 277 K, surface level: +4 C "ice number" with no melting sink,
    # 2026-07-28; the same surface pile ended century3 at N_i=1e193).
    dN_i_dt = (dN_i_nuc - dN_i_autoconv + homo_freeze_N + sed_N_i
               - dN_i_melt - dN_i_subl)

    # === Non-negativity floor on the cloud/rain/ice NUMBER tendencies ===
    # Each individual number sink above is bounded (evap/riming ∝ N/dt,
    # ice autoconv clamped to N_i/dt), but their SUM — plus an inward
    # sedimentation divergence — can still overshoot the locally
    # available number in one explicit step and drive N negative. A
    # negative number concentration is unphysical and poisons every
    # PSD-derived quantity (mean mass q·ρ/N, slope λ, fall speed),
    # cascading to NaN. Cap the NET sink so the post-step number cannot
    # fall below zero; positive sources pass through. N_s/N_g already
    # get the equivalent treatment via their LAMS/LAMG consistency
    # limiters (``n_s_new``/``n_g_new``) below.
    dN_c_dt = jnp.maximum(dN_c_dt, -jnp.clip(N_c, 0.0) / jnp.clip(dt, 1.0))
    dN_r_dt = jnp.maximum(dN_r_dt, -jnp.clip(N_r, 0.0) / jnp.clip(dt, 1.0))
    dN_i_dt = jnp.maximum(dN_i_dt, -jnp.clip(N_i, 0.0) / jnp.clip(dt, 1.0))

    # SAM N_i consistency limiter (mirrors the N_s pattern below): bound the
    # POST-STEP number so LAMI stays in [lami_min, lami_max] w.r.t. the
    # POST-STEP mass, and CLEAR the number entirely below QSMALL — orphan
    # number (q_i ~ 0, N_i > 0) is what accumulated into the century3/5
    # surface pile (N_i = 1e193 by day 803).  Applied AFTER every physical
    # number tendency (in-place SAM reset semantics: adjusting the OLD state
    # alternated 0 <-> lami_min across steps, and a q_i floor in the lower
    # bound created number in CLEAR AIR — both codex 2026-07-28 round 1).
    q_i_new = jnp.maximum(jnp.clip(q_i, 0.0) + dq_i_dt * dt, 0.0)
    _ci_psd = jnp.pi * config.rho_cloud_ice
    n_i_hi = config.lami_max ** 3 * q_i_new / _ci_psd
    n_i_lo = config.lami_min ** 3 * q_i_new / _ci_psd
    n_i_new = jnp.clip(jnp.clip(N_i, 0.0) + dN_i_dt * dt, n_i_lo, n_i_hi)
    n_i_new = jnp.where(q_i_new > 1.0e-14, n_i_new, 0.0)
    dN_i_dt = (n_i_new - jnp.clip(N_i, 0.0)) / jnp.maximum(dt, 1.0e-10)

    # Snow NUMBER budget (double-moment snow). Number is CONSERVED across the
    # phase changes: the ice→snow autoconversion that removes dN_i_autoconv
    # from ice ADDS it to snow (NPRCI), and the frozen rain drops removed from
    # N_r (freeze_N_r, per-volume ⇒ /ρ for per-mass N_s) become snow particles
    # (NNUCCR). Snow loses number to melting (∝ the melted mass fraction), to
    # self-aggregation (NSAGG: flakes merge, mass conserved), and sediments at
    # the number-weighted UNS. The PRDS/PSACWS number changes (flakes just grow)
    # are intentionally zero.
    if snow_double_moment:
        # dN_s_melt (NSMLTS) defined ONCE above with the NSMLTR rain-number
        # source, so the snow sink and the rain source can never diverge.
        # NSUBS: snow-number sink during SUBLIMATION (SAM), removing number at
        # the same fractional rate as mass (mean-size preserving) — codex
        # iter-27 C. Deposition (PRDS>0) grows existing flakes ⇒ no number
        # change. The N_s limiter below is the backstop.
        dN_s_subl = (jnp.minimum(prds, 0.0)
                     * jnp.clip(N_s, 0.0) / jnp.clip(q_s, 1e-15))
        # NSAGG snow self-aggregation: a physical number sink (merging flakes),
        # now explicit rather than left to the consistency limiter (iter-30).
        nsagg = (snow_self_aggregation_nsagg(q_s, N_s, rho, config, dt=dt)
                 if config.do_snow_aggregation
                 else jnp.zeros_like(q_s))
        dN_s_dt = (dN_i_autoconv + freeze_N_to_snow / jnp.clip(rho, _RHO_FLOOR)
                   - dN_s_melt + dN_s_subl - nsagg - nscng + sed_N_s)
        # SAM N_s consistency limiter (module_mp_graupel.f90:1727-1737): clamp
        # N_s so the snow PSD slope LAMS stays in [lams_min, lams_max]. Uses
        # the snow mass AFTER the step (q_s + dq_s·dt) as the reference. This
        # BOUNDS N_s — substituting for the deferred NSAGG self-aggregation
        # (codex iter-26 E: N_s would otherwise only grow) — and removes stale
        # snow number when the snow melts/sediments away (N_s_hi ∝ q_s_new → 0
        # ⇒ N_s → 0, codex C). N_s_{lo,hi} = LAMS_{min,max}³·q_s/(π·ρ_sn).
        q_s_new = jnp.maximum(jnp.clip(q_s, 0.0) + dq_s_dt * dt, 0.0)
        _cs = jnp.pi * config.rho_snow
        n_s_hi = config.lams_max ** 3 * q_s_new / _cs
        n_s_lo = config.lams_min ** 3 * q_s_new / _cs
        n_s_new = jnp.clip(jnp.clip(N_s, 0.0) + dN_s_dt * dt, n_s_lo, n_s_hi)
        dN_s_dt = (n_s_new - jnp.clip(N_s, 0.0)) / jnp.maximum(dt, 1.0e-10)
    else:
        dN_s_dt = None

    # Graupel NUMBER budget (double-moment graupel). New graupel PARTICLES come
    # from frozen rain (NNUCCR: each frozen drop ⇒ a graupel particle) and the
    # snow→graupel embryos (NSCNG). Riming/rain-accretion/deposition GROW
    # existing graupel (no number change). Graupel loses number to melting + the
    # sublimation branch of PRDG (∝ melted/sublimated mass fraction) and
    # sediments at the number-weighted UNG. A SAM-style consistency limiter
    # keeps LAMG ∈ [lamg_min, lamg_max].
    if graupel_double_moment:
        # dN_g_melt (NGMLTG) defined ONCE above with the NGMLTR rain-number
        # source, so the graupel sink and the rain source can never diverge.
        dN_g_subl = (jnp.minimum(prdg, 0.0)
                     * jnp.clip(N_g, 0.0) / jnp.clip(q_g, 1e-15))
        # Frozen rain drops → graupel particles (per-volume freeze_N_r ⇒ /ρ for
        # per-mass N_g); the snow→graupel embryos NSCNG (already per-mass).
        freeze_N_to_graupel = (freeze_N_r / jnp.clip(rho, _RHO_FLOOR)
                               if config.do_graupel
                               else jnp.zeros_like(q_g))
        dN_g_dt = (freeze_N_to_graupel + nscng
                   - dN_g_melt + dN_g_subl + sed_N_g)
        q_g_new = jnp.maximum(jnp.clip(q_g, 0.0) + dq_g_dt * dt, 0.0)
        _cg = jnp.pi * config.rho_graupel
        n_g_hi = config.lamg_max ** 3 * q_g_new / _cg
        n_g_lo = config.lamg_min ** 3 * q_g_new / _cg
        n_g_new = jnp.clip(jnp.clip(N_g, 0.0) + dN_g_dt * dt, n_g_lo, n_g_hi)
        dN_g_dt = (n_g_new - jnp.clip(N_g, 0.0)) / jnp.maximum(dt, 1.0e-10)
    else:
        dN_g_dt = None

    # Precipitation (rain + ice + snow at surface) uses the dt-limited
    # surface flux from ``sedimentation_tendency`` so column water
    # conservation holds exactly when the CFL limiter fires.
    precipitation = precip_r + precip_i + precip_s + precip_g

    # (No placeholder outputs here — every MicrophysicsOutput field below is
    # a computed tendency, so no dtype pin is needed; a former bare
    # ``jnp.zeros(...)`` expression at this point was dead code.)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=dq_g_dt,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=dN_i_dt,
        precipitation=precipitation,
        dN_s_dt=dN_s_dt,
        dN_g_dt=dN_g_dt,
    )
