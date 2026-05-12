"""Thompson hybrid-moment microphysics.

Extends Morrison with graupel formation from intense riming and
gamma distribution shape corrections for autoconversion/accretion.

All operations use smooth (differentiable) approximations.

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
)
from legoesm.atmosphere.physics.microphysics.config import ThompsonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


def _gamma_ratio(mu):
    """Gamma(mu+4)/Gamma(mu+1) = (mu+3)(mu+2)(mu+1) for integer-like mu."""
    return (mu + 3.0) * (mu + 2.0) * (mu + 1.0)


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
    gamma_c_norm = gamma_c / _gamma_ratio(0.0)  # normalize to mu=0 baseline (=24)
    gamma_r_norm = gamma_r / _gamma_ratio(0.0)

    # Autoconversion (gamma-corrected)
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness, gamma_norm=gamma_c_norm,
    )

    # Accretion (gamma-corrected)
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac, gamma_norm=gamma_r_norm)

    # Self-collection / breakup
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )

    # Rain evaporation
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff, dt=dt)

    # === ICE PHASE (Morrison processes) ===
    T_freeze = constants.T_freeze
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # Ice nucleation
    N_i_target = config.N_i0 * jnp.exp(
        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    ) / jnp.clip(rho, 0.1)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)

    # Depositional growth.  Same heuristic form as Morrison —
    # ``q_i_min_growth`` floor only, so fresh nucleation can grow.
    # See morrison.py for the rationale.
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

    # Bergeron
    berg_window = (
        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
    )
    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window

    # Riming
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    total_riming = riming_i + riming_s

    # Aggregation
    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice

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
    qc_scale = jnp.minimum(
        1.0,
        qc_avail / jnp.maximum(qc_sink_total * dt_safe, 1e-30),
    )
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
    # Mirror of the q_c clamp.  q_i sinks: aggregation, melt_ice,
    # rime_to_graupel_from_i.  Riming_i is a q_i source (not a sink),
    # so it is NOT included.  Without this clamp, an explicit Euler
    # step with combined sinks > q_i / dt drives q_i negative.
    qi_sink_total = aggregation + melt_ice + rime_to_graupel_from_i
    qi_avail = jnp.clip(q_i, 0.0)
    qi_scale = jnp.minimum(
        1.0,
        qi_avail / jnp.maximum(qi_sink_total * dt_safe, 1e-30),
    )
    aggregation = aggregation * qi_scale
    melt_ice = melt_ice * qi_scale
    rime_to_graupel_from_i = rime_to_graupel_from_i * qi_scale

    # === DONOR CLAMP for q_s sinks ===
    # q_s sinks: melt_snow, rime_to_graupel_from_s.
    qs_sink_total = melt_snow + rime_to_graupel_from_s
    qs_avail = jnp.clip(q_s, 0.0)
    qs_scale = jnp.minimum(
        1.0,
        qs_avail / jnp.maximum(qs_sink_total * dt_safe, 1e-30),
    )
    melt_snow = melt_snow * qs_scale
    rime_to_graupel_from_s = rime_to_graupel_from_s * qs_scale
    rime_to_graupel = rime_to_graupel_from_i + rime_to_graupel_from_s

    # === SEDIMENTATION ===
    # Marshall-Palmer fall speeds use fractional exponents (b_v_x in
    # [0.25, 0.5]); guard the AD path with safe_pow.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)
    V_t_g = config.a_v_g * safe_pow(jnp.clip(q_g, 0.0) * rho_ratio, config.b_v_g)
    V_t_g = jnp.clip(V_t_g, 0.0, 30.0)

    sed_r, precip_r = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt, return_surface_flux=True,
    )
    sed_i, precip_i = sedimentation_tendency(
        q_i, rho, V_t_i, dz, dt=dt, return_surface_flux=True,
    )
    sed_s, precip_s = sedimentation_tendency(
        q_s, rho, V_t_s, dz, dt=dt, return_surface_flux=True,
    )
    sed_g, precip_g = sedimentation_tendency(
        q_g, rho, V_t_g, dz, dt=dt, return_surface_flux=True,
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
    dq_v_dt = -condensation + evaporation - dq_i_dep
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + melt_graupel + sed_r
    dq_i_dt = (
        dq_i_dep + bergeron + riming_i - aggregation - melt_ice
        - rime_to_graupel_from_i + sed_i
    )
    dq_s_dt = aggregation + riming_s - melt_snow - rime_to_graupel_from_s + sed_s
    dq_g_dt = rime_to_graupel - melt_graupel + sed_g

    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

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
