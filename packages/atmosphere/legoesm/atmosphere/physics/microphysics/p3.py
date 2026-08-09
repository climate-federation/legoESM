"""P3 (Predicted Particle Properties) microphysics.

Single ice category with predicted rime mass and volume, following
Morrison & Milbrandt (2015). Ice particle properties (fall speed,
density) are diagnosed from the predicted state rather than assumed
from a fixed habit, collapsing the snow/graupel distinction into a
continuous spectrum.

Liquid phase reuses Seifert-Beheng warm-rain helpers from _warm_rain.py.

Slot reuse
----------
P3 uses the HydrometeorState slots normally occupied by snow and graupel
to carry its two predicted rime variables:

    HydrometeorState.q_s  → q_rim  [kg/kg]       rime mass mixing ratio
    HydrometeorState.q_g  → B_rim  [m³/kg_air]   rime volume per kg air

The rime density rho_rim = q_rim / B_rim [kg/m³] determines particle
habit and fall speed, spanning unrimed aggregates (~50 kg/m³) through
dense graupel (~900 kg/m³) as a smooth, differentiable function.

Corresponding output slots:
    MicrophysicsOutput.dq_s_dt → dq_rim_dt  [kg/kg/s]
    MicrophysicsOutput.dq_g_dt → dB_rim_dt  [m³/kg_air/s]  (unit differs
        from other schemes; integration code applies it correctly as a
        generic tendency regardless of physical units)

Faithfulness
------------
This is a P3-STRUCTURED scheme (single free ice category with predicted rime
mass + volume), but most of its process RATES are simplified surrogates, NOT the
gSAM P3 oracle (``MICRO_P3/module_mp_p3.f90``):

- Cooper (1986) ice NUCLEATION — ORACLE-FAITHFUL gSAM P3 scheme-1 semantics
  (module_mp_p3.f90:3084-3095), smoothed for differentiability. Faithful
  pieces (all pinned by ``tests/.../test_p3_cooper_faithful.py``): base curve
  ``N_i0·exp(cooper_a·(T_freeze−T))/rho`` (N_i0=5.0=0.005·1000,
  cooper_a=0.304, per-kg rho-divide), scheme-1 CAP 100/L·SCF at SCF=1 (:3090;
  this column scheme has no SCPF cloud fraction, and gSAM's scpf_ON=.false.
  default also runs SCF=1 — scheme 2 would use 150/L at :3136), nucleation
  GATE ``T<−15 °C AND supi≥0.05`` (:3084), the (target−N_i)/dt relaxation,
  and the SEED mass ``mi0=4/3π·900·(1e-6)³`` (:233 — gSAM hardcodes
  900 kg/m³, not ``constants.rho_ice``). The former departures (500/L cap,
  −8 °C gate with no supersaturation requirement, 917 seed density) were
  closed 2026-07-17. Remaining structural deltas (documented, deliberate):
  the hard Fortran gates become SIGMOIDS (``ice_sigmoid_sharpness`` on T,
  ``cooper_supi_sharpness`` on supi) so the scheme stays differentiable, plus
  a rho floor, a ``clip(dt,1)`` floor, and an exponent cap absent from the
  raw Fortran.
- Ice deposition, cloud/rain RIMING, aggregation, MELTING, and ice/rain FALL
  SPEED (:264-384) — SURROGATES, NOT amenable to a closed-form coefficient-level
  pin. gSAM P3 computes each of these by interpolating a LOOKUP TABLE
  (``f1pr02``..``f1pr14`` / γ-distribution integrals, module_mp_p3.f90:2440-2666);
  legoESM uses invented algebraic forms (``dep_coeff·q_i·N_i^⅓``,
  ``rime_coeff·q_i·q_c``, ``agg_coeff·N_i``, ``melt_rate·q_i·sigmoid``,
  ``a_v_i·(q_i·rho)^b``) with no coefficient-level counterpart (they could only be
  pinned against compiled gSAM output or frozen lookup-table fixtures).
- WARM RAIN reuses the SIMPLIFIED ``_warm_rain`` helpers (``autoconversion_sb``
  q_c²·sigmoid proxy, bilinear ``accretion``, legacy ``self_collection_breakup``,
  simplified ``rain_evaporation``) — NOT the published SB2001 universal functions
  (``autoconversion_sb2001``/``accretion_sb2001``, reachable via Morrison's
  ``warm_rain_scheme="seifert_beheng_sb2001"``).

References
----------
- Morrison, H. & Milbrandt, J. A. (2015). Parameterization of cloud
  microphysics based on the prediction of bulk ice particle properties.
  Part I: Scheme description and idealized tests.
  J. Atmos. Sci., 72, 287-311.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio_ice as _saturation_mixing_ratio_ice
from legoesm.thermo import homogeneous_freezing_rh_factor as _homogeneous_freezing_rh_factor
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
from legoesm.atmosphere.physics.microphysics.config import P3Config
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


# Fixed P3 nucleation / fall-speed constants.
_RHO_FLOOR = 0.1
_COOPER_EXP_CAP = 80.0
_VT_CLIP_RAIN = 20.0

# --- Initial ice crystal mass at nucleation (gSAM P3 module_mp_p3.f90:233) ---
# P3 seeds each freshly-nucleated crystal with the ORACLE seed mass
# ``mi0 = 4/3·π·900·(IceNucleiRadius)³`` ≈ 3.77e-15 kg: gSAM hardcodes a
# 900 kg/m³ nucleus density (NOT constants.rho_ice = 917 — the former ~1.9 %
# heavier legoESM seed was a documented departure, closed 2026-07-17) at the
# default 1-µm radius (micro_params.f90:214).  Removed from vapour so N_i and
# q_i stay consistent right after nucleation instead of collapsing the
# diagnosed mean crystal mass q_i/N_i to zero.
_ICE_NUC_RADIUS_M = 1.0e-6                                     # [m]
_RHO_ICE_NUC = 900.0   # [kg/m³] gSAM P3 hardcoded seed density (mi0, :233)
_M_I0 = 4.0 / 3.0 * math.pi * _RHO_ICE_NUC * _ICE_NUC_RADIUS_M ** 3  # [kg]

__physics_contract__ = {
    "summary": (
        "P3-STRUCTURED (Predicted Particle Properties; Morrison & Milbrandt 2015) "
        "ice microphysics: a single free ice category with prognostic rime mass "
        "and rime volume (evolving density/fall speed) plus simplified "
        "SB-style warm rain. The ice process RATES (deposition, riming, "
        "aggregation, melting, fall speed) are algebraic SURROGATES for P3's "
        "lookup-table physics, not the P3 oracle; Cooper ice nucleation is "
        "ORACLE-FAITHFUL gSAM scheme-1 semantics (base curve, 100/L cap, "
        "T<-15C AND supi>=0.05 gate, 900 kg/m^3 seed), sigmoid-smoothed for "
        "differentiability — see the module 'Faithfulness' docstring. "
        "Sedimentation to surface precipitation."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "hydrometeors.q_c": "kg/kg",
        "hydrometeors.q_r": "kg/kg", "hydrometeors.q_i": "kg/kg",
        "hydrometeors.q_s": "kg/kg (P3 rime mass q_rim)",
        "hydrometeors.q_g": "m^3/kg_air (P3 rime volume B_rim)",
        "hydrometeors.N_c": "1/m^3", "hydrometeors.N_r": "1/m^3",
        "hydrometeors.N_i": "1/kg", "p_full": "Pa", "rho": "kg/m^3",
        "dz": "m", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_dt": "kg/kg/s",
        "dq_r_dt": "kg/kg/s", "dq_i_dt": "kg/kg/s",
        "dq_s_dt": "kg/kg/s (rime mass q_rim)",
        "dq_g_dt": "m^3/kg_air/s (rime volume B_rim)",
        "dN_c_dt": "1/(m^3 s)", "dN_r_dt": "1/(m^3 s)", "dN_i_dt": "1/(kg s)",
        "precipitation": "kg/m^2/s",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. Latent heating dT_dt uses L_v/L_s/L_f, "
        "consistent with each phase-change rate. The free ice category's rime "
        "mass/volume evolve its density and fall speed; SURFACE PRECIPITATION "
        "(>= 0) removes water, so column moisture is NOT conserved -- no "
        "contract-level conservation is claimed. Masses and numbers stay >= 0."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": "Morrison & Milbrandt (2015), J. Atmos. Sci. 72, 287-311",
    "idealized_test": (
        "tests/unit/test_rce_ice_microphysics.py — riming raises the ice "
        "category's rime fraction (density/fall speed increase); ice/rain reach "
        "the surface; latent heating tracks phase changes; masses/numbers >= 0."
    ),
}


def p3_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: P3Config = P3Config(),
) -> MicrophysicsOutput:
    """Compute P3 microphysics tendencies.

    Liquid phase: Seifert-Beheng two-moment warm rain.
    Ice phase: single predicted-property category (Morrison & Milbrandt 2015).

    Parameters
    ----------
    T : (ncol, nlev) Temperature [K]
    q_v : (ncol, nlev) Water vapour [kg/kg]
    hydrometeors : HydrometeorState
        q_s slot carries q_rim [kg/kg]; q_g slot carries B_rim [m³/kg_air].
    p_full, p_half : (ncol, nlev), (ncol, nlev+1) Pressure [Pa]
    rho : (ncol, nlev) Air density [kg/m³]
    dz : (ncol, nlev) Layer thickness [m]
    dt : float  Physics time step [s]
    config : P3Config

    Returns
    -------
    MicrophysicsOutput
        dq_s_dt carries dq_rim_dt [kg/kg/s].
        dq_g_dt carries dB_rim_dt [m³/kg_air/s].
    """
    ncol, nlev = T.shape

    # Unpack — q_s/q_g slots reinterpreted as P3 rime variables.
    q_c   = hydrometeors.q_c
    q_r   = hydrometeors.q_r
    q_i   = hydrometeors.q_i
    q_rim = hydrometeors.q_s   # rime mass [kg/kg]
    B_rim = hydrometeors.q_g   # rime volume [m³/kg_air]
    N_c   = hydrometeors.N_c
    N_r   = hydrometeors.N_r
    N_i   = hydrometeors.N_i

    # =========================================================================
    # WARM RAIN (Seifert-Beheng liquid phase — identical to Morrison)
    # =========================================================================
    N_c_eff = effective_Nc(N_c, config.Nc_0)

    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, dt, config.saturation_sharpness, q_c=q_c,
        hard_adjust=config.hard_saturation_adjustment,
        hard_threshold=config.hard_sat_adjust_threshold,
        hard_max_heating_K=config.hard_sat_max_heating_K,
    )
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star,
        config.autoconversion_sharpness,
    )
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff, dt=dt)

    # =========================================================================
    # DIAGNOSE ICE PARTICLE PROPERTIES
    # =========================================================================
    # Rime fraction: fraction of total ice mass that is rime [0, 1].
    q_i_safe = jnp.maximum(q_i, 1e-15)
    f_rim = jnp.clip(
        safe_divide(jnp.clip(q_rim, 0.0), q_i_safe, eps=1e-15), 0.0, 1.0,
    )

    # Rime density: rho_rim = q_rim [kg/kg] / B_rim [m³/kg_air] = [kg/m³].
    # safe_divide guards the AD path when B_rim ≈ 0 (no rime yet).
    rho_rim = jnp.clip(
        safe_divide(jnp.clip(q_rim, 0.0), jnp.maximum(B_rim, 1e-30), eps=1e-30),
        config.rho_rim_min,
        config.rho_rim_max,
    )

    # Fall speed density enhancement: denser (more rimed) particles fall faster.
    # safe_pow guards fractional power when rho_rim → rho_rim_min.
    density_factor = safe_pow(rho_rim / config.rho_ice_ref, config.c_rim_fallspeed)

    # =========================================================================
    # ICE PHASE
    # =========================================================================
    T_freeze = constants.T_freeze

    # Smooth mask: ice processes active below cooper_T_act.
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # Ice supersaturation (used by the nucleation gate and deposition below).
    # PLAIN ice saturation — the nucleation gate must NOT see the
    # homogeneous-freezing allowance: the oracle gate (supi >= 0.05 on plain
    # q_sat_i) is what decides whether crystals appear at all, and raising its
    # denominator by rh_homo suppressed nucleation outright (the gSAM Cooper-cap
    # test collapsed from 6667 to 9e-24 /kg/s).
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0

    # DEPOSITION target only: IFS/SAM homogeneous-freezing allowance lets
    # pristine air below 235 K hold ice supersaturation up to rh_homo (gSAM
    # cloud.f90); withdrawn where cloud ice is already present at scheme entry (the cloud.f90 qci gate; see thermo.homogeneous_freezing_rh_factor).
    q_sat_i_dep = q_sat_i * _homogeneous_freezing_rh_factor(
        T, q_i, enabled=config.homogeneous_ice_supersaturation,
    )

    # 1. Ice nucleation (Cooper 1986, gSAM P3 scheme-1 semantics, smoothed).
    #
    # ORACLE GATE (module_mp_p3.f90:3084): nucleate only where
    # ``T < 258.15 K (−15 °C) AND supi >= 0.05`` — smoothed to a product of
    # sigmoids so the scheme stays differentiable (sharpnesses are numerics
    # params, not tunables).  The supersaturation factor also kills the bare
    # ``N_i0/rho`` target left active above freezing by the
    # ``max(T_freeze − T, 0)`` floor inside the exponential (the old
    # f_ice-only gate at cooper_T_act = −8 °C nucleated warmer AND in
    # ice-SUBSATURATED air — the documented REALISM gap, closed 2026-07-17).
    # f_ice (a warmer, general mixed-phase gate) still gates deposition,
    # riming, rain-riming, aggregation.
    f_nuc = (
        jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_nuc - T))
        * jax.nn.sigmoid(
            config.cooper_supi_sharpness * (S_i - config.cooper_supi_min))
    )
    # Cap at ``N_i_nuc_max`` — ORACLE scheme-1 value 100 L⁻¹·SCF with SCF = 1
    # (module_mp_p3.f90:3090; this column scheme has no SCPF cloud fraction, and
    # gSAM's own scpf_ON=.false. default also runs SCF = 1; scheme 2 uses
    # 150 L⁻¹·SCF at :3136).  Applied BEFORE the ρ-divide: the bare Cooper
    # exponential overflows fp32 at the very cold tropopause / sponge
    # temperatures of an RCEMIP column (→ N_i = inf → NaN in tracer slot 8).
    # ``jnp.minimum`` clamps even an inf exponential to the finite cap.
    N_i_target = jnp.minimum(
        config.N_i0
        * jnp.exp(jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), _COOPER_EXP_CAP)),
        config.N_i_nuc_max,
    ) / jnp.clip(rho, _RHO_FLOOR)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0) * f_nuc
    # Nucleation MASS source: each new crystal carries the seed mass m_i0
    # (vapour → ice, +L_s), mirroring morrison.py's MNUCCD = NNUCCD·MI0.
    # Number-only nucleation left q_i/N_i → 0 and skewed the N_i^(1/3)
    # deposition closure below.  [1/(kg·s)]·[kg] = [kg/kg/s].
    dq_i_nuc = dN_i_nuc * _M_I0

    # 2. Vapour deposition on ice (subsaturated wrt ice: sublimation handled
    # by the jnp.maximum(S_i, 0) gate — only deposition grows q_i here;
    # sublimation is a separate pathway not yet included, consistent with
    # Morrison which also omits explicit sublimation).
    q_i_eff = jnp.maximum(jnp.clip(q_i, 0.0), config.q_i_min_growth)
    # Deposition is driven by the excess over the ALLOWED target: in pristine
    # air below 235 K that is rh_homo*q_sat_i, so vapour accumulates to the
    # homogeneous-freezing threshold instead of depositing immediately.
    S_i_dep = q_v / jnp.clip(q_sat_i_dep, 1e-10) - 1.0
    dq_i_dep = (
        config.dep_coeff
        * jnp.maximum(S_i_dep, 0.0)
        * q_i_eff
        * safe_pow(N_i, 1.0 / 3.0)
        * f_ice
    )
    # Deposition does not change rime variables (deposited vapour is
    # unrimed crystal mass), so dq_rim_dep = 0, dB_rim_dep = 0.

    # 3. Cloud riming: ice collects supercooled cloud droplets.
    # q_rim gains mass; B_rim gains volume at the accreted rime density.
    riming = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    dq_rim_rime   = riming                                   # [kg/kg/s]
    dB_rim_rime   = riming / config.rho_rim_accrete          # [m³/kg_air/s]

    # 4. Rain riming: ice collects rain drops (freezes them into rime).
    # q_r is a source of rime mass; rain_rime drains q_r and grows q_i + q_rim.
    rain_rime     = config.rain_rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_r, 0.0) * f_ice
    dq_rim_rr     = rain_rime                                # [kg/kg/s]
    dB_rim_rr     = rain_rime / config.rho_rim_accrete       # [m³/kg_air/s]

    # 5. Self-collection (aggregation): N_i decreases, q_i is conserved.
    # Larger aggregates have same mass but fewer particles.
    aggregation_N = config.agg_coeff * jnp.clip(N_i, 0.0) * f_ice  # [1/kg/s]
    # Donor-clamp: prevent aggregation from driving N_i negative at long dt.
    aggregation_N = aggregation_N * donor_clamp_scale(
        jnp.clip(N_i, 0.0), aggregation_N, dt,
    )

    # 6. Melting: all ice (including rime) melts to rain when T > T_freeze.
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    melt_ice  = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
    )
    # Rime melts in proportion to rime fraction.
    melt_rim  = melt_ice * f_rim                             # [kg/kg/s]
    # Volume lost = rime mass lost / rime density.
    # safe_divide guards when rho_rim → rho_rim_min.
    # eps=1e-30 (not rho_rim_min) because rho_rim is already clipped ≥ rho_rim_min;
    # using rho_rim_min as eps would zero the result exactly at the floor.
    melt_B_rim = safe_divide(melt_rim, rho_rim, eps=1e-30)  # [m³/kg_air/s]

    # =========================================================================
    # DONOR CLAMPS
    # =========================================================================

    # --- q_i sinks: melt_ice (riming and deposition are sources) ---
    qi_sink_total = melt_ice
    qi_avail      = jnp.clip(q_i, 0.0)
    qi_scale      = donor_clamp_scale(qi_avail, qi_sink_total, dt)
    melt_ice      = melt_ice  * qi_scale
    melt_rim      = melt_rim  * qi_scale
    melt_B_rim    = melt_B_rim * qi_scale

    # --- q_rim sinks: melt_rim ---
    qrim_sink_total = melt_rim
    qrim_avail      = jnp.clip(q_rim, 0.0)
    qrim_scale      = donor_clamp_scale(qrim_avail, qrim_sink_total, dt)
    melt_rim        = melt_rim   * qrim_scale
    melt_B_rim      = melt_B_rim * qrim_scale

    # --- q_r sinks: evaporation + rain_rime + sedimentation ---
    # rain_rime freezes q_r into ice; guard jointly with evaporation.
    qr_sink_total = evaporation + rain_rime
    qr_avail      = jnp.clip(q_r, 0.0)
    qr_scale      = donor_clamp_scale(qr_avail, qr_sink_total, dt)
    evaporation   = evaporation * qr_scale
    rain_rime     = rain_rime   * qr_scale
    dq_rim_rr     = dq_rim_rr   * qr_scale
    dB_rim_rr     = dB_rim_rr   * qr_scale

    # --- q_c sinks: autoconversion + accretion + riming ---
    cond_evap_sink = jnp.maximum(-condensation, 0.0)
    qc_sink_total  = dq_c_au + dq_c_ac + riming + cond_evap_sink
    qc_avail       = jnp.clip(q_c, 0.0)
    qc_scale       = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    dq_c_au        = dq_c_au   * qc_scale
    dq_c_ac        = dq_c_ac   * qc_scale
    riming         = riming    * qc_scale
    dq_rim_rime    = dq_rim_rime * qc_scale
    dB_rim_rime    = dB_rim_rime * qc_scale
    condensation   = jnp.where(
        condensation < 0.0, condensation * qc_scale, condensation,
    )
    dN_r_au        = dN_r_au   * qc_scale

    # --- q_v sinks: condensation + deposition + nucleation seed mass ---
    cond_pos       = jnp.maximum(condensation, 0.0)
    qv_sink_total  = cond_pos + jnp.maximum(dq_i_dep, 0.0) + dq_i_nuc
    qv_avail       = jnp.clip(q_v, 0.0)
    qv_scale       = donor_clamp_scale(qv_avail, qv_sink_total, dt)
    condensation   = jnp.where(condensation > 0.0, condensation * qv_scale, condensation)
    dq_i_dep       = dq_i_dep  * qv_scale
    # Scale mass AND number by the same qv_scale so the per-crystal seed
    # mass m_i0 stays consistent when vapour limits nucleation (mirrors
    # morrison.py).
    dq_i_nuc       = dq_i_nuc  * qv_scale
    dN_i_nuc       = dN_i_nuc  * qv_scale

    # =========================================================================
    # SEDIMENTATION
    # =========================================================================
    rho_sfc    = rho[:, -1:]
    rho_ratio  = rho / jnp.clip(rho_sfc, _RHO_FLOOR)

    # Rain fall speed (Marshall-Palmer).
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, _VT_CLIP_RAIN)

    # Ice fall speed (P3: base power law × rime-density enhancement).
    V_t_i = (
        config.a_v_i
        * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
        * density_factor
    )
    V_t_i = jnp.clip(V_t_i, 0.0, 10.0)

    # Joint q_r sink for the sed cap: evaporation AND rain_rime both
    # drain q_r in this step.  Including only ``evaporation`` lets
    # sedimentation remove the remaining ``q_r - evap·dt`` while
    # rain_rime still removes its share on top — for heavy q_i + q_r
    # at long dt the combined drain exceeds q_r and the explicit
    # Euler step drives q_r negative.  Empirical probe (T=260 K,
    # q_r=5e-3, q_i=1e-3, q_rim=5e-4, dz=500 m, dt=1200 s): without
    # rain_rime in the sed cap q_r ended at -5.4e-4 kg/kg.  With it,
    # q_r stays >= 0.
    sed_r, precip_r = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation + rain_rime,
    )
    sed_i, precip_i = sedimentation_tendency(
        q_i, rho, V_t_i, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=melt_ice,
    )
    # B_rim sediments at the same speed as q_i (same particles).
    # sedimentation_tendency handles [m³/kg_air] correctly: the flux
    # V_t * B_rim * rho has units [m³/m²/s] and the tendency
    # (flux_in − flux_out)/(rho·dz) recovers [m³/kg_air/s].
    sed_B_rim, _ = sedimentation_tendency(
        B_rim, rho, V_t_i, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=melt_B_rim,
    )
    # q_rim also sediments with q_i (rime is part of the ice particle).
    sed_rim, _ = sedimentation_tendency(
        q_rim, rho, V_t_i, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=melt_rim,
    )

    # =========================================================================
    # LATENT HEATING
    # =========================================================================
    L_v  = constants.L_v
    L_s  = constants.L_s
    L_f  = constants.L_f
    c_pd = constants.c_pd

    dT_dt = (
        L_v * condensation / c_pd
        - L_v * evaporation / c_pd
        # Vapour → ice (deposition + nucleation seed mass): releases L_s.
        + L_s * (dq_i_dep + dq_i_nuc) / c_pd
        # Cloud/rain riming: supercooled liquid → ice, releases L_f.
        + L_f * (riming + rain_rime) / c_pd
        # Melting: ice → liquid, absorbs L_f.
        - L_f * melt_ice / c_pd
    )

    # =========================================================================
    # COMBINE TENDENCIES
    # =========================================================================
    dq_v_dt   = -condensation + evaporation - dq_i_dep - dq_i_nuc
    dq_c_dt   = condensation - dq_c_au - dq_c_ac - riming
    dq_r_dt   = dq_c_au + dq_c_ac - evaporation + melt_ice - rain_rime + sed_r
    dq_i_dt   = dq_i_dep + dq_i_nuc + riming + rain_rime - melt_ice + sed_i
    dq_rim_dt = dq_rim_rime + dq_rim_rr - melt_rim + sed_rim
    dB_rim_dt = dB_rim_rime + dB_rim_rr - melt_B_rim + sed_B_rim

    # Number tendencies.
    # Cloud-droplet number sinks: autoconversion (mean-mass x_c per drop) PLUS
    # riming and accretion, which sweep up WHOLE droplets — number removed in
    # proportion to the cloud mass consumed (SAM NPSACWS/NPRA; mirrors
    # morrison.py's dN_c_riming).  Without these, riming/accretion shrink the
    # mean droplet size (lower q_c, same N_c) and spuriously slow
    # autoconversion in mixed-phase cloud.  riming/dq_c_ac are already
    # qc_scale-donor-clamped, so the sink is bounded by N_c/dt (plus the
    # explicit floor below).  [kg/kg/s]·[1/m³]/[kg/kg] = [1/(m³·s)].
    dN_c_collect = ((riming + dq_c_ac)
                    * jnp.clip(N_c, 0.0) / jnp.clip(q_c, 1e-15))
    dN_c_dt = safe_divide(-dq_c_au * rho, x_c, eps=1e-15) - dN_c_collect

    # PHYSICAL step for all number caps/floors: use the ACTUAL dt (only floored
    # by a tiny eps to stay finite at dt=0), NOT clip(dt, 1.0). The over-step
    # number limit is N/dt; a clip-at-1 would cap subsecond-dt LES steps at
    # N/1 < N/dt, transferring fewer crystals than the mass melted/rimed and
    # leaving stale number (codex mass-number-consistency fix).
    dt_step = jnp.maximum(dt, 1.0e-6)
    # Rain riming (rain mass collected onto ice: the ``- rain_rime`` term in
    # dq_r_dt): whole drops leave the rain category, so N_r must drop in
    # proportion to the rimed rain-mass fraction (mirrors dN_c_collect).
    # Without this sink, rimed rain keeps its number and the mean rain mass
    # collapses. [kg/kg/s]·[1/m³]/[kg/kg] = [1/(m³·s)].
    dN_r_rime = rain_rime * jnp.clip(N_r, 0.0) / jnp.clip(q_r, 1e-15)
    # Melting transfers NUMBER with the mass (Morrison & Milbrandt 2015): the
    # melted crystals leave N_i and reappear as rain drops (N_r source,
    # per-volume ⇒ ×ρ). melt_ice is already donor-clamped to q_i/dt (line ~300),
    # so dN_i_melt <= N_i/dt; the explicit N_i/dt_step cap is a redundant guard
    # keyed to the PHYSICAL step so exactly the SAME count is removed from ice
    # and added to rain. Otherwise the net-N_i floor below would shrink only the
    # ice sink (not the rain source) when melting and aggregation overlap,
    # minting spurious rain drops.
    dN_i_melt = jnp.minimum(
        melt_ice * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15),
        jnp.clip(N_i, 0.0) / dt_step,
    )
    # Aggregation (ice self-collection, internal to N_i) is bounded to the ice
    # number remaining AFTER the melt transfer, so aggregation + melting cannot
    # jointly drive N_i < 0 — keeping the melt↔rain number equality exact
    # without relying on the blanket floor.
    agg_eff = jnp.minimum(
        aggregation_N,
        jnp.clip(jnp.clip(N_i, 0.0) - dN_i_melt * dt_step, 0.0) / dt_step,
    )
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br + dN_i_melt * rho - dN_r_rime
    dN_i_dt = dN_i_nuc - agg_eff - dN_i_melt

    # Non-negativity floors on the prognostic number tendencies, keyed to the
    # PHYSICAL step dt_step (NOT clip(dt, 1.0)): the floor -N/dt_step removes at
    # most all of N over the step. Using clip(dt, 1.0) would clip a subsecond
    # (dt < 1) melt/rime sink up to -N/1s, shrinking the ice/rain number removal
    # while the paired source kept the full transfer — breaking the melt↔rain
    # and rain-riming number equalities (codex subsecond-dt fix). Ice melt +
    # aggregation are already bounded to N_i/dt_step above, so the N_i floor is
    # inactive; the cloud/rain self-collection and droplet-autoconversion SINKS
    # are ∝ the current number and still need capping so one Euler step cannot
    # overshoot the available number (RCE restart drove a number slot → −4.4e8
    # in ~600 steps). Positive sources pass through; the floor bounds only the
    # net sink, so the positive melt source (dN_i_melt·ρ) survives intact.
    dN_c_dt = jnp.maximum(dN_c_dt, -jnp.clip(N_c, 0.0) / dt_step)
    dN_r_dt = jnp.maximum(dN_r_dt, -jnp.clip(N_r, 0.0) / dt_step)
    dN_i_dt = jnp.maximum(dN_i_dt, -jnp.clip(N_i, 0.0) / dt_step)

    precipitation = precip_r + precip_i

    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_rim_dt,    # q_s slot carries q_rim tendency
        dq_g_dt=dB_rim_dt,    # q_g slot carries B_rim tendency [m³/kg_air/s]
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=dN_i_dt,
        precipitation=precipitation,
    )
