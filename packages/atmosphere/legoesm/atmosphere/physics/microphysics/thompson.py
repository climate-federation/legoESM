"""Thompson hybrid-moment microphysics.

Extends Morrison with graupel formation from intense riming and
gamma distribution shape corrections for autoconversion/accretion.

All operations use smooth (differentiable) approximations.

Faithfulness to Thompson et al. (2008) / WRF ``module_mp_thompson.F`` (oracle)
----------------------------------------------------------------------------
FAITHFUL (forms matched to the Thompson-2008 / M2005-lineage algorithm):
  * **Capacitance ice vapour-diffusion growth** (PRD), ``ice_growth_scheme=
    "capacitance"`` (default): ``EPSI = (2π/CONS12^⅓)·ρ·DV·N_i^⅔·q_i_eff^⅓`` and
    ``PRD = η·EPSI·(q_v−q_sat_i)/ABI`` with the published vapour-diffusivity fit
    ``DV = 8.794e-5·T^1.81/p`` and psychrometric correction
    ``ABI = 1 + (dq_sat_i/dT)·L_s/c_p`` (Thompson 2008, following Reisner 1998).
    ``q_i_eff = max(clip(q_i, 0), q_i_min_growth)`` is the growth-floored ice.
  * **PRCI depositional ice→snow autoconversion**: ``PRCI = (2π·D_cs²/3)·ρ·N0I·
    exp(−LAMI·D_cs)·DV·(q_v−q_sat_i)₊/ABI``, ``LAMI=(CONS12·N_i/max(q_i_eff,1e-20))^⅓``,
    ``N0I=N_i·LAMI``; number removal ``NPRCI = PRCI/m(D_cs)``, ``m(D_cs)=π·ρ_ci·
    D_cs³/6`` (removes D_cs-sized crystals, not the mean mass).
  * **Thompson-2008 snow** (``snow_scheme="thompson2008"``, in ``_thompson_snow.py``):
    Field-2005 bimodal-PSD moments → mass-weighted, density-corrected fall speed
    and ventilated vapour deposition (PRDS) with the oracle's T-ramped
    capacitance ``C_sqrd=0.3 → C_cube=0.5`` (the former fixed 0.15 departure
    closed 2026-07-17). Pinned in ``test_thompson_snow.py``.
  * **Cooper (1986) ice nucleation**: target number ``N_i0·exp(a·max(T_freeze−T,
    0))`` (exp-argument capped for fp overflow), min'd to ``N_i_nuc_max`` (SAM
    500 /L) and divided by ρ to per-mass; ``N_i`` relaxes toward it, ``f_ice``-gated.
  * **Gamma-distribution shape ratio** ``Γ(μ+4)/Γ(μ+1) = (μ+3)(μ+2)(μ+1)`` applied
    to the autoconversion/accretion rates.
DEPARTURES / SURROGATES (NOT Thompson closed forms; documented + labeled):
  * **Warm rain is Seifert-Beheng (2001)** (``autoconversion_sb``/``accretion``/
    ``self_collection_breakup``/``rain_evaporation`` from ``_warm_rain``, gamma-
    corrected) — NOT Thompson's warm-rain (Berry-Reinhardt/KK2000) closure.
  * **Riming, Bergeron, and melting are bulk first-order relaxations**
    (``rate·q·window``), not the Thompson collection/melting integrals.
  * **Graupel is a simplified extension**: ``graupel_frac = sigmoid(s·(riming−
    threshold))`` times a fixed conversion fraction. Thompson-2008 **Part II has
    NO graupel category** (graupel enters the later full WRF scheme); this is a
    legoESM deep-convection extension.
  * **Bulk power-law fall speeds** ``a·q^b`` (capped) for rain/ice/graupel; only
    SNOW uses the faithful Thompson mass-weighted fall speed.
  * The ``"heuristic"`` ice-growth (``dep_coeff·max(S_i,0)·q_i_eff·N_i^⅓·f_ice``)
    and ``"bulk_qpower"`` snow are legacy surrogate fallbacks.
  * All phase switches are sigmoids (``f_ice``/``melt_frac``/Bergeron window/
    graupel), and every sink is donor-clamped / cap-limited (single-step
    relaxation limits, ``V_t`` clips, Cooper-exp cap, ρ floor) for AD-safe
    positivity — smooth departures from the hard Fortran branches.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_thompson_faithful.py``
(scheme level) + ``tests/unit/test_thompson_snow.py`` (faithful snow helper).

References
----------
- Thompson, G., Field, P. R., Rasmussen, R. M., & Hall, W. D. (2008).
  Explicit forecasts of winter precipitation using an improved bulk
  microphysics scheme. Part II: Implementation of a new snow
  parameterization. Mon. Wea. Rev., 136, 5095-5115.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
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
from legoesm.atmosphere.physics._shared import safe_divide
from legoesm.atmosphere.physics.microphysics._thompson_snow import (
    snow_fall_speed as _thompson_snow_fall_speed,
    snow_deposition as _thompson_snow_deposition,
)
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


# Diffusivity + Cooper nucleation + fall-speed caps (fixed).
_RHO_FLOOR = 0.1
_DV_PREFACTOR = 8.794e-5
_DV_T_EXP = 1.81
_COOPER_EXP_CAP = 80.0
_VT_CLIP_FROZEN = 5.0
_VT_CLIP_GRAUPEL = 30.0
_VT_CLIP_RAIN = 20.0

def _gamma_ratio(mu):
    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)


__physics_contract__ = {
    "summary": (
        "Thompson et al. (2008) hybrid-moment microphysics: extends the "
        "double-moment ice+liquid scheme with graupel formation from intense "
        "riming and a faithful snow parameterization; sedimentation of rain/"
        "snow/graupel to surface precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "hydrometeors.q_s": "kg/kg", "hydrometeors.q_g": "kg/kg",
        "hydrometeors.N_c": "1/m^3", "hydrometeors.N_r": "1/m^3",
        "hydrometeors.N_i": "1/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s", "dq_s_dt": "kg/kg/s",
        "dq_g_dt": "kg/kg/s", "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)",
        "dN_i_dt": "1/(kg s)", "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt uses L_v/L_s/L_f, "
        "consistent with each phase-change rate. Water is redistributed among "
        "vapour/cloud/rain/ice/snow/graupel; SURFACE PRECIPITATION (>= 0) "
        "removes water, so column moisture is NOT conserved -- no contract-level "
        "conservation is claimed. Masses and numbers stay >= 0. An unknown "
        "snow_scheme raises ValueError (dispatch hardening)."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Thompson, Field, Rasmussen & Hall (2008), Mon. Wea. Rev. 136, 5095-5115",
    "idealized_test": (
        "tests/unit/test_rce_ice_microphysics.py — a deep convective column "
        "forms graupel via riming; snow/graupel/rain reach the surface; latent "
        "heating tracks the phase changes; masses/numbers >= 0."
    ),
}


def thompson_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: ThompsonConfig = ThompsonConfig(),
) -> MicrophysicsOutput:
    """Compute Thompson hybrid-moment microphysics tendencies.

    Parameters
    ----------
    T, q_v, hydrometeors, p_full, p_half, rho, dz, dt, config
        Same interface as all microphysics backends.

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    # Validate the snow scheme on the static config value at fn entry so a
    # typo cannot silently select the bulk power-law fall speed (site below)
    # while leaving Thompson-2008 snow deposition disabled — a physics change.
    if config.snow_scheme not in ("thompson2008", "bulk_qpower"):
        raise ValueError(
            f"Unknown snow_scheme {config.snow_scheme!r}; "
            "expected one of: 'thompson2008', 'bulk_qpower'."
        )
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    q_i = hydrometeors.q_i
    q_s = hydrometeors.q_s
    q_g = hydrometeors.q_g
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    N_i = hydrometeors.N_i
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    # === WARM RAIN ===
    # Saturation adjustment — convert increment [kg/kg] to tendency [kg/kg/s].
    # Pass ``q_c`` so the evaporation branch (negative ``condensation``) is
    # donor-clamped: evaporation cannot drive ``q_c`` below zero in
    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, dt, sharpness, q_c=q_c,
    )

    # Gamma distribution corrections
    gamma_c = _gamma_ratio(config.mu_c)
    gamma_r = _gamma_ratio(config.mu_r)
    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=6)
    gamma_r_norm = gamma_r / _gamma_ratio(0.0)

    # Autoconversion (gamma-corrected)
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star,
        config.autoconversion_sharpness, gamma_norm=gamma_c_norm,
    )

    # Accretion (gamma-corrected)
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)

    # Self-collection / breakup
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )

    # Rain evaporation.  ``rain_evap_rh_floor`` suppresses evaporation where the
    # liquid sub-saturation is below the float32 saturation resolution
    # (RH ≳ 99.995 %): the ungated ``clip(q_sat−q_v,0)`` term rectifies float32
    # round-off into spurious in-cloud evaporation that recycles
    # rain→vapour→cloud and inflated the float32 LWP (18-24% DYCOMS fp32-vs-fp64
    # spread). Applied to the deficit, so resolved deficits (WBF, sub-cloud
    # downdrafts) evaporate normally; no-op at float64 for a saturated cloud.
    # See _warm_rain.
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff, dt=dt,
                                   rh_deficit_floor=config.rain_evap_rh_floor)

    # === ICE PHASE (Morrison processes) ===
    T_freeze = constants.T_freeze
    # f_ice transitions around ``cooper_T_act`` (≈ 265 K by default),
    # NOT ``T_freeze``.  See morrison.py for the rationale.
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # Ice nucleation (Cooper 1986).  The ``max(T_freeze − T, 0)``
    # floor inside the exponential leaves the bare ``N_i0/rho``
    # target active above freezing; gating with ``f_ice`` shuts
    # nucleation off in warm columns (mirrors Morrison fix +
    # matches the gating already applied to all other ice sources).
    # Cap at ``N_i_nuc_max`` (SAM 500 L⁻¹) BEFORE the ρ-divide. The bare
    # Cooper exponential ``exp(cooper_a·(T_freeze−T))`` diverges at the
    # very cold tropopause / sponge temperatures reached in an RCEMIP
    # column and overflows fp32 (→ N_i = inf → NaN in tracer slot 8).
    # ``jnp.minimum`` clamps even an already-inf exponential back to the
    # finite cap. Mirrors ``morrison.py`` (which has always capped here).
    N_i_target = jnp.minimum(
        config.N_i0 * jnp.exp(
            jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), _COOPER_EXP_CAP)
        ),
        config.N_i_nuc_max,
    ) / jnp.clip(rho, _RHO_FLOOR)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0) * f_ice

    # === Ice depositional growth / sublimation ===
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    q_i_eff = jnp.maximum(jnp.clip(q_i, 0.0), config.q_i_min_growth)
    # Shared diffusional-growth thermodynamics (Thompson 2008 / Reisner 1998):
    #   DV  = 8.794e-5·T^1.81/p          vapour diffusivity [m²/s]
    #   ABI = 1 + (dq_sat_i/dT)·L_s/c_p  psychrometric correction
    #   CONS12 = ρ_ci·π  (mass–size for spherical ice, m = (ρ_ci·π/6)·D³)
    cons12_cbrt = (config.rho_cloud_ice * jnp.pi) ** (1.0 / 3.0)
    dv_vap = _DV_PREFACTOR * safe_pow(T, _DV_T_EXP) / jnp.clip(p_full, 1.0)
    dqsidt = constants.L_s * q_sat_i / (constants.R_v * T ** 2)
    abi = 1.0 + dqsidt * constants.L_s / constants.c_pd
    if config.ice_growth_scheme == "capacitance":
        # FAITHFUL capacitance-based vapour-diffusion growth (replaces the
        # legacy ``dep_coeff·S_i·q_i·N_i^⅓`` heuristic):
        #   PRD = EPSI·(q_v−q_sat_i)/ABI,
        #   EPSI = 2π·N_i·ρ·DV/LAMI = (2π/CONS12^⅓)·ρ·DV·N_i^⅔·q_i_eff^⅓
        #     (q_i_eff = max(clip(q_i,0), q_i_min_growth), the growth-floored ice).
        # ``EPSI ∝ N_i^⅔`` self-gates on ice presence (no f_ice gate needed,
        # so warm mixed-phase ice growth / WBF is not spuriously suppressed —
        # see morrison.py). Sublimation (q_v<q_sat_i) is donor-clamped to q_i.
        epsi = (
            2.0 * jnp.pi / cons12_cbrt * rho * dv_vap
            * safe_pow(jnp.clip(N_i, 0.0), 2.0 / 3.0)
            * safe_pow(q_i_eff, 1.0 / 3.0)
        )
        dep_raw = config.ice_deposition_efficiency * epsi * (q_v - q_sat_i) / abi
        # Cap positive deposition at the available ice supersaturation per
        # step: deposition physically HALTS at saturation, so it cannot draw
        # q_v below q_sat_i in one explicit step. Without this the stiff
        # capacitance growth (EPSI·dt/ABI → 1) can overshoot, dumping a latent-
        # heating spike into convective cores that drives the updraft (and, in
        # RCEMIP/RRTMGP, a w-runaway). Analytic single-step relaxation limit,
        # NOT a clip of a physical quantity. Mirrors morrison.py.
        dep_pos = jnp.minimum(
            jnp.maximum(dep_raw, 0.0),
            jnp.maximum(q_v - q_sat_i, 0.0) / jnp.clip(dt, 1.0),
        )
        subl_neg = jnp.maximum(
            jnp.minimum(dep_raw, 0.0),
            -jnp.clip(q_i, 0.0) / jnp.clip(dt, 1.0),
        )
        dq_i_dep = dep_pos + subl_neg
    elif config.ice_growth_scheme == "heuristic":
        dq_i_dep = (
            config.dep_coeff * jnp.maximum(S_i, 0.0)
            * q_i_eff * safe_pow(N_i, 1.0 / 3.0) * f_ice
        )
    else:
        raise ValueError(
            f"Unknown ice_growth_scheme: {config.ice_growth_scheme!r}; "
            f"choose 'capacitance' or 'heuristic'."
        )

    # Bergeron — sharper warm cutoff via ``f_ice`` so the loose
    # ``melt_sharpness=2`` sigmoid tail does not leak ~1e-6 at
    # T = 280 K (mirrors Morrison fix).
    berg_window = (
        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
    )
    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window * f_ice

    # Riming
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    total_riming = riming_i + riming_s

    # === Ice → snow autoconversion (PRCI) ===
    # FAITHFUL Thompson-2008 ice→snow conversion: cloud ice whose
    # depositional growth carries the PSD ACROSS the snow-size threshold
    # D_cs becomes snow (Thompson 2008 §; M2005 PRCI lineage):
    #   PRCI = (2π·D_cs²/3)·ρ·N0I·exp(−LAMI·D_cs)·DV·(q_v−q_sat_i)₊/ABI,
    #   LAMI = (CONS12·N_i/max(q_i_eff, 1e-20))^⅓,  N0I = N_i·LAMI.
    # This is FAR stronger than the legacy ``agg_coeff·q_i`` relaxation in
    # supersaturated convective cores, so cloud ice drains into fast-falling
    # snow instead of accumulating and driving a latent-heating w-runaway
    # (the RCEMIP/RRTMGP blow-up). Self-gates on ice supersaturation + N_i.
    if config.ice_growth_scheme == "capacitance":
        lami_ac = cons12_cbrt * safe_pow(
            jnp.clip(N_i, 0.0) / jnp.maximum(q_i_eff, 1.0e-20), 1.0 / 3.0)
        n0i_ac = jnp.clip(N_i, 0.0) * lami_ac
        aggregation = (
            (2.0 * jnp.pi / 3.0) * config.ice_snow_d_auto ** 2 * rho * n0i_ac
            * jnp.exp(-lami_ac * config.ice_snow_d_auto) * dv_vap
            * jnp.maximum(q_v - q_sat_i, 0.0) / abi
        )
        aggregation = jnp.where(jnp.clip(q_i, 0.0) > 1.0e-14, aggregation, 0.0)
    else:
        aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice

    # === Snow vapour deposition / sublimation (faithful Thompson-2008) ===
    # The distinctive Thompson-2008 snow grows by vapour deposition on the
    # bimodal snow PSD — a major upper-tropospheric vapour sink that converts
    # the supersaturation feeding deep convection into precipitating snow.
    # PRDS>0 deposits vapour onto snow (q_v→q_s, +L_s); PRDS<0 sublimates.
    # ``ice_growth_scheme="capacitance"`` enables it; "heuristic" keeps it off.
    if config.ice_growth_scheme == "capacitance" and config.snow_scheme == "thompson2008":
        prds = _thompson_snow_deposition(q_v, q_s, q_sat_i, T, p_full, rho, dt)
    else:
        prds = jnp.zeros_like(jnp.clip(q_s, 0.0))

    # Melting (clamp to available mass so an explicit Euler step cannot
    # drive q_i / q_s / q_g negative — same pattern Morrison already uses).
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    dt_safe = jnp.maximum(dt, 1e-10)
    melt_ice = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / dt_safe,
    )
    melt_snow = jnp.minimum(
        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
        jnp.clip(q_s, 0.0) / dt_safe,
    )

    # === Melt graupel (independent of riming-graupel pathway) ===
    melt_graupel = jnp.minimum(
        config.melt_rate * jnp.clip(q_g, 0.0) * melt_frac,
        jnp.clip(q_g, 0.0) / dt_safe,
    )

    # === DONOR CLAMP for q_c sinks (see morrison.py for rationale) ===
    # Include the evaporation branch of saturation_adjustment (negative
    # condensation) in the q_c sink budget so subsaturated clear-air
    # columns cannot drive q_c negative (Codex audit cycle 2).
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    qc_sink_total = (
        dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
    )
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
    total_riming = riming_i + riming_s
    dN_r_au = dN_r_au * qc_scale
    # Scale negative-condensation (evaporation) branch by the same
    # factor; positive condensation is unaffected (cond_evap_sink = 0).
    condensation = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )

    # === GRAUPEL (Thompson extension) — computed AFTER the donor clamp ===
    # The threshold check ``total_riming > threshold`` must be against
    # the ACTUAL post-clamp riming rate, not the pre-clamp demand.  An
    # earlier form computed ``graupel_frac`` from the pre-clamp
    # ``total_riming`` and then scaled the resulting
    # ``rime_to_graupel`` by ``qc_scale`` — but a heavily-clamped
    # column where post-clamp ``total_riming`` is *below* threshold
    # would still see ``graupel_frac ≈ 1`` (the pre-clamp demand was
    # well above threshold), driving a spurious 50 % conversion of the
    # actually-tiny riming flux into graupel (Codex audit cycle 2:
    # "Thompson ``graupel_frac`` uses pre-clamp ``total_riming``").
    #
    # The donor split (rime_to_graupel_from_i / rime_to_graupel_from_s)
    # remains in place so a column with ``q_i = 0`` and ``riming_s > 0``
    # never drives q_i negative through the rime → graupel pathway.
    graupel_frac = jax.nn.sigmoid(
        config.graupel_sharpness * (total_riming - config.rime_to_graupel_threshold)
    )
    rime_to_graupel_from_i = config.rime_to_graupel_rate * riming_i * graupel_frac
    rime_to_graupel_from_s = config.rime_to_graupel_rate * riming_s * graupel_frac
    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

    # === DONOR CLAMP for q_i sinks ===
    # q_i sinks: ice→snow autoconversion (aggregation), melt_ice,
    # rime_to_graupel_from_i, AND ice SUBLIMATION (the negative branch of the
    # capacitance deposition — absent from the old deposition-only heuristic).
    # Riming_i is a q_i source (not a sink), so excluded. Without the joint
    # clamp, combined sinks > q_i/dt drive q_i negative (each is individually
    # bounded but their SUM is not).
    ice_subl_sink = jnp.maximum(-dq_i_dep, 0.0)
    qi_sink_total = (aggregation + melt_ice + rime_to_graupel_from_i
                     + ice_subl_sink)
    qi_avail = jnp.clip(q_i, 0.0)
    qi_scale = donor_clamp_scale(qi_avail, qi_sink_total, dt)
    aggregation = aggregation * qi_scale
    melt_ice = melt_ice * qi_scale
    rime_to_graupel_from_i = rime_to_graupel_from_i * qi_scale
    # Scale only the SUBLIMATION (negative) branch of dq_i_dep; deposition
    # (positive, a q_i source) is unaffected.
    dq_i_dep = jnp.where(dq_i_dep < 0.0, dq_i_dep * qi_scale, dq_i_dep)

    # === DONOR CLAMP for q_s sinks ===
    # q_s sinks: melt_snow, rime_to_graupel_from_s, AND snow SUBLIMATION (the
    # negative branch of the faithful snow deposition prds). Each is bounded
    # individually but their SUM can exceed q_s/dt.
    snow_subl_sink = jnp.maximum(-prds, 0.0)
    qs_sink_total = melt_snow + rime_to_graupel_from_s + snow_subl_sink
    qs_avail = jnp.clip(q_s, 0.0)
    qs_scale = donor_clamp_scale(qs_avail, qs_sink_total, dt)
    melt_snow = melt_snow * qs_scale
    rime_to_graupel_from_s = rime_to_graupel_from_s * qs_scale
    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s
    # Scale only the sublimation (negative) branch of prds; deposition
    # (positive, a q_s source) is unaffected.
    prds = jnp.where(prds < 0.0, prds * qs_scale, prds)

    # === DONOR CLAMP for q_v sinks ===
    # Vapor budget: dq_v_dt = -condensation + evaporation - dq_i_dep.
    # Positive condensation AND positive dq_i_dep together remove
    # vapor; without a joint clamp, supersaturated icy layers can
    # over-draw q_v.  Mirror Morrison's iter-25 q_v clamp.  Codex
    # iter-29 #2.
    #
    # Vapor donor clamp via the shared AD-safe helper
    # (donor_clamp_scale).  See morrison.py for the rationale.
    # ``prds`` (snow deposition) is the THIRD positive vapour sink in icy,
    # supersaturated layers — include it in the joint clamp so condensation +
    # ice deposition + snow deposition together cannot over-draw q_v.
    cond_pos = jnp.maximum(condensation, 0.0)
    qv_sink_total = cond_pos + jnp.maximum(dq_i_dep, 0.0) + jnp.maximum(prds, 0.0)
    qv_avail = jnp.clip(q_v, 0.0)
    qv_scale = donor_clamp_scale(qv_avail, qv_sink_total, dt)
    condensation = jnp.where(condensation > 0.0, condensation * qv_scale, condensation)
    # Scale ONLY the depositional (positive, vapour-limited) branch of dq_i_dep;
    # the sublimation (negative) branch is a vapour SOURCE, not a sink — it is
    # already donor-clamped to q_i in the q_i clamp above and must NOT be scaled
    # by qv_scale. In exactly-dry air (q_v=0) qv_scale=0, so the previous
    # unconditional ``dq_i_dep *= qv_scale`` zeroed legitimate ice sublimation
    # (no vapour source, no sublimation cooling) — codex round 1 finding 2.
    # This now matches the positive-branch ``where`` already used for
    # condensation and prds.
    dq_i_dep = jnp.where(dq_i_dep > 0.0, dq_i_dep * qv_scale, dq_i_dep)
    # Scale only the depositional (positive, vapour-limited) branch of prds;
    # sublimation (negative) is already snow-limited inside the snow module.
    prds = jnp.where(prds > 0.0, prds * qv_scale, prds)

    # === SEDIMENTATION ===
    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
    # [0.25, 0.5]); guard the AD path with safe_pow.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, _RHO_FLOOR)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, _VT_CLIP_RAIN)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, _VT_CLIP_FROZEN)
    if config.snow_scheme == "thompson2008":
        # FAITHFUL Thompson-2008 mass-weighted snow fall speed from the bimodal
        # PSD (Field-2005 moments + density correction) — replaces the capped
        # bulk power law so snow precipitates at a physical, ρ-corrected speed
        # instead of being trapped aloft.
        V_t_s = _thompson_snow_fall_speed(q_s, rho, T)
    else:
        V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
        V_t_s = jnp.clip(V_t_s, 0.0, _VT_CLIP_FROZEN)
    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
    V_t_g = jnp.clip(V_t_g, 0.0, _VT_CLIP_GRAUPEL)

    # Joint donor caps: each `extra_sink` is the in-column sink that
    # shares the same explicit-Euler step as sedimentation.  Without
    # these, post-donor-clamp in-column sinks ALREADY consume up to
    # q/dt, AND sed independently can drain another q/dt — driving the
    # pool negative.  Mirrors the iter-29 q_r/evap fix; extended to
    # ALL sed paths in iter-73.
    sed_r, precip_r = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )
    # The post-clamp ice/snow SUBLIMATION (negative deposition branch) is also
    # an in-column q_i / q_s sink sharing this explicit step — it must be in the
    # sed ``extra_sink`` budget too.  Omitting it let sublimation + sedimentation
    # each draw up to q/dt and drove q_i / q_s negative on a subsaturated
    # sedimenting column even at the standard dt=300 s (physics-validator probe).
    ice_subl_post = jnp.maximum(-dq_i_dep, 0.0)
    snow_subl_post = jnp.maximum(-prds, 0.0)
    sed_i, precip_i = sedimentation_tendency(
        q_i, rho, V_t_i, dz, dt=dt, return_surface_flux=True,
        extra_sink=aggregation + melt_ice + rime_to_graupel_from_i
        + ice_subl_post,
    )
    sed_s, precip_s = sedimentation_tendency(
        q_s, rho, V_t_s, dz, dt=dt, return_surface_flux=True,
        extra_sink=melt_snow + rime_to_graupel_from_s + snow_subl_post,
    )
    sed_g, precip_g = sedimentation_tendency(
        q_g, rho, V_t_g, dz, dt=dt, return_surface_flux=True,
        extra_sink=melt_graupel,
    )

    # === LATENT HEATING ===
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd
    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        + L_s * dq_i_dep / c_pd
        # Snow vapour deposition releases L_s (sublimation absorbs it).
        + L_s * prds / c_pd
        # Cloud water → ice/snow freezing releases L_f (Bergeron, riming).
        # See morrison.py for the moist-enthalpy rationale; Thompson
        # mirrors Morrison's ice-phase latent heating.
        + L_f * (bergeron + riming_i + riming_s) / c_pd
        - L_f * (melt_ice + melt_snow + melt_graupel) / c_pd
    )

    # === COMBINE TENDENCIES ===
    # Conservation: each rime-to-graupel donor leaves its parent
    # species and arrives in q_g.  The total mass moved is
    # ``rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s``.
    # An earlier form used 1.0 / 0.5 / 1.5 splits on ``rime_to_graupel``
    # which (a) drove ``q_i`` negative when only snow was being rimed
    # (``q_i = 0`` but ``riming_s > 0``), and (b) created mass
    # apparently from nothing in the same regime.  See the
    # ``=== GRAUPEL ===`` block above for the donor-split rationale.
    dq_v_dt = -condensation + evaporation - dq_i_dep - prds
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
    dq_i_dt = (
        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
        - rime_to_graupel_from_i + sed_i
    )
    dq_s_dt = (aggregation + riming_s + prds - melt_snow
               - rime_to_graupel_from_s + sed_s)
    dq_g_dt = rime_to_graupel - melt_graupel + sed_g

    # AD-safe number-concentration tendencies (issue #249).  ``dN_c_dt``
    # uses ``safe_divide(eps=1e-15)`` (5 decades above the prior
    # ``clip(x_c, 1e-20)`` floor; cells masked out fall well below the
    # physical droplet-mass scale).  ``dN_i_dt`` keeps the legacy
    # ``clip(q_i, 1e-15) + divide`` form: the floor is high enough that
    # ``aggregation ∝ q_i`` divided by it stays bounded, and the
    # clip's zero VJP at the floor branch already protects AD.  See
    # ``morrison.py`` for the full justification.
    dN_c_dt = safe_divide(-dq_c_au * rho, x_c, eps=1e-15)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    if config.ice_growth_scheme == "capacitance":
        # PRCI removes D_cs-SIZED crystals: NPRCI = PRCI / m(D_cs), with
        # m(D_cs) = CONS22 = π·ρ_ci·D_cs³/6 (clamped to N_i/dt). Using the
        # mean-mass rate (∝ N_i/q_i) would over-remove ice number.
        cons22 = jnp.pi * config.rho_cloud_ice * config.ice_snow_d_auto ** 3 / 6.0
        dN_i_autoconv = jnp.minimum(
            aggregation / cons22, jnp.clip(N_i, 0.0) / jnp.clip(dt, 1.0))
        dN_i_dt = dN_i_nuc - dN_i_autoconv
    else:
        dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

    # Non-negativity floors on the prognostic number tendencies. The rain
    # self-collection sink ``dN_r_sc`` and the autoconversion droplet sink
    # ``dN_c_dt`` are explicit and ∝ the current number; unbounded, one Euler
    # step overshoots the available number, drives N negative, and the
    # self-collection rate then flips sign and runs away (RCE restart: a
    # number slot → −1.6e9 within ~600 steps). Cap the NET sink so the
    # post-step number cannot fall below zero; positive sources pass through.
    dt_floor = jnp.clip(dt, 1.0)
    dN_c_dt = jnp.maximum(dN_c_dt, -jnp.clip(N_c, 0.0) / dt_floor)
    dN_r_dt = jnp.maximum(dN_r_dt, -jnp.clip(N_r, 0.0) / dt_floor)
    dN_i_dt = jnp.maximum(dN_i_dt, -jnp.clip(N_i, 0.0) / dt_floor)

    # Precipitation uses the dt-limited surface flux from
    # ``sedimentation_tendency`` so column water conservation holds
    # exactly when the CFL limiter fires.
    precipitation = precip_r + precip_i + precip_s + precip_g

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
    )
