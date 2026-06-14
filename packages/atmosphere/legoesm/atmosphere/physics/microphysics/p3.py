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

References
----------
- Morrison, H. & Milbrandt, J. A. (2015). Parameterization of cloud
  microphysics based on the prediction of bulk ice particle properties.
  Part I: Scheme description and idealized tests.
  J. Atmos. Sci., 72, 287-311.
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

    # 1. Ice nucleation (Cooper 1986, smoothed).
    #
    # The ``max(T_freeze − T, 0)`` floor inside the exponential leaves
    # the bare ``N_i0/rho`` target active above freezing; without an
    # ``f_ice`` gate ``dN_i_nuc`` nucleated ~28 crystals / kg / s at
    # T = 290 K (probe).  Multiplying by ``f_ice`` (≈ 0 above
    # cooper_T_act) shuts nucleation off in warm columns and matches
    # the gating already applied to deposition, riming, rain-riming,
    # and aggregation — mirrors the same fix landed in
    # morrison.py / thompson.py.
    # Cap at ``N_i_nuc_max`` (SAM 500 L⁻¹) BEFORE the ρ-divide: the bare
    # Cooper exponential overflows fp32 at the very cold tropopause /
    # sponge temperatures of an RCEMIP column (→ N_i = inf → NaN in
    # tracer slot 8). ``jnp.minimum`` clamps even an inf exponential to
    # the finite cap. Mirrors morrison.py / thompson.py.
    N_i_target = jnp.minimum(
        config.N_i0
        * jnp.exp(jnp.minimum(config.cooper_a * jnp.maximum(T_freeze - T, 0.0), _COOPER_EXP_CAP)),
        config.N_i_nuc_max,
    ) / jnp.clip(rho, _RHO_FLOOR)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0) * f_ice

    # 2. Vapour deposition on ice (subsaturated wrt ice: sublimation handled
    # by the jnp.maximum(S_i, 0) gate — only deposition grows q_i here;
    # sublimation is a separate pathway not yet included, consistent with
    # Morrison which also omits explicit sublimation).
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    q_i_eff = jnp.maximum(jnp.clip(q_i, 0.0), config.q_i_min_growth)
    dq_i_dep = (
        config.dep_coeff
        * jnp.maximum(S_i, 0.0)
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

    # --- q_v sinks: condensation + deposition ---
    cond_pos       = jnp.maximum(condensation, 0.0)
    qv_sink_total  = cond_pos + jnp.maximum(dq_i_dep, 0.0)
    qv_avail       = jnp.clip(q_v, 0.0)
    qv_scale       = donor_clamp_scale(qv_avail, qv_sink_total, dt)
    condensation   = jnp.where(condensation > 0.0, condensation * qv_scale, condensation)
    dq_i_dep       = dq_i_dep  * qv_scale

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
        + L_s * dq_i_dep / c_pd
        # Cloud/rain riming: supercooled liquid → ice, releases L_f.
        + L_f * (riming + rain_rime) / c_pd
        # Melting: ice → liquid, absorbs L_f.
        - L_f * melt_ice / c_pd
    )

    # =========================================================================
    # COMBINE TENDENCIES
    # =========================================================================
    dq_v_dt   = -condensation + evaporation - dq_i_dep
    dq_c_dt   = condensation - dq_c_au - dq_c_ac - riming
    dq_r_dt   = dq_c_au + dq_c_ac - evaporation + melt_ice - rain_rime + sed_r
    dq_i_dt   = dq_i_dep + riming + rain_rime - melt_ice + sed_i
    dq_rim_dt = dq_rim_rime + dq_rim_rr - melt_rim + sed_rim
    dB_rim_dt = dB_rim_rime + dB_rim_rr - melt_B_rim + sed_B_rim

    # Number tendencies.
    dN_c_dt = safe_divide(-dq_c_au * rho, x_c, eps=1e-15)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    # Aggregation reduces N_i (self-collection of ice particles).
    dN_i_dt = dN_i_nuc - aggregation_N

    # Non-negativity floors on the prognostic number tendencies. The explicit
    # rain self-collection / droplet-autoconversion / ice-aggregation SINKS are
    # ∝ the current number; unbounded, one Euler step overshoots the available
    # number, drives N negative, and self-collection then runs away (RCE
    # restart: a number slot → −4.4e8 within ~600 steps). Cap each NET sink so
    # the post-step number stays ≥ 0; positive sources pass through unchanged.
    dt_floor = jnp.clip(dt, 1.0)
    dN_c_dt = jnp.maximum(dN_c_dt, -jnp.clip(N_c, 0.0) / dt_floor)
    dN_r_dt = jnp.maximum(dN_r_dt, -jnp.clip(N_r, 0.0) / dt_floor)
    dN_i_dt = jnp.maximum(dN_i_dt, -jnp.clip(N_i, 0.0) / dt_floor)

    precipitation = precip_r + precip_i

    # Pin dtype so unused placeholders don't silently promote under x64.
    jnp.zeros((ncol, nlev), dtype=T.dtype)
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
