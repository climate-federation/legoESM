"""Morrison double-moment ice+liquid microphysics.

Extends Seifert-Beheng warm-rain with ice-phase processes: nucleation
(Cooper 1986), depositional growth, Bergeron process, riming, snow
aggregation, and melting. Tracks cloud water, rain, ice, and snow.

All operations use smooth (differentiable) approximations.

References
----------
- Morrison, H., Curry, J. A., & Khvorostyanov, V. I. (2005). A new
  double-moment microphysics parameterization. Part I: Description.
  J. Atmos. Sci., 62, 1665-1677.
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
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


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
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    q_i = hydrometeors.q_i
    q_s = hydrometeors.q_s
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    N_i = hydrometeors.N_i
    sharpness = config.saturation_sharpness

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    # === WARM RAIN (shared Seifert-Beheng helpers) ===
    # Pass ``q_c`` so the evaporation branch (negative ``condensation``)
    # is donor-clamped: evaporation cannot drive ``q_c`` below zero in
    # subsaturated clear air.  See _warm_rain.saturation_adjustment.
    condensation, q_sat = saturation_adjustment(
        T, q_v, p_full, dt, sharpness, q_c=q_c,
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

    # === ICE PHASE ===
    T_freeze = constants.T_freeze
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # 1. Ice nucleation (Cooper 1986, smoothed)
    N_i_target = config.N_i0 * jnp.exp(
        config.cooper_a * jnp.maximum(T_freeze - T, 0.0)
    ) / jnp.clip(rho, 0.1)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)

    # 2. Depositional growth.
    # Heuristic Morrison form: dq_i/dt ∝ S_i · q_i · N_i^(1/3) · f_ice.
    # The full diffusional-growth scaling is q_i^(1/3) · N_i^(2/3),
    # but switching to that scaling without re-tuning ``dep_coeff``
    # changes the deposition magnitude by O(10^3) at typical mid-cloud
    # values — a calibration change beyond the scope of this audit.
    # We keep the heuristic form here and only add a ``q_i_min_growth``
    # floor so freshly-nucleated ice (N_i > 0, q_i ≈ 0) can begin to
    # grow at all instead of being pinned to zero deposition.
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

    # 3. Bergeron process: cloud water -> ice in mixed-phase zone
    berg_window = (
        jax.nn.sigmoid(config.melt_sharpness * (T_freeze - T))
        * jax.nn.sigmoid(config.melt_sharpness * (T - (config.T_center - config.T_width)))
    )
    bergeron = config.bergeron_rate * jnp.clip(q_c, 0.0) * berg_window

    # 4. Riming: ice/snow collect cloud water
    riming_i = config.rime_coeff * jnp.clip(q_i, 0.0) * jnp.clip(q_c, 0.0) * f_ice
    riming_s = config.rime_coeff * jnp.clip(q_s, 0.0) * jnp.clip(q_c, 0.0) * f_ice

    # 5. Snow aggregation: ice -> snow
    aggregation = config.agg_coeff * jnp.clip(q_i, 0.0) * f_ice

    # 6. Melting near T_freeze: ice/snow -> rain (clipped to available mass)
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    melt_ice = jnp.minimum(
        config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac,
        jnp.clip(q_i, 0.0) / jnp.maximum(dt, 1e-10),
    )
    melt_snow = jnp.minimum(
        config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac,
        jnp.clip(q_s, 0.0) / jnp.maximum(dt, 1e-10),
    )

    # === DONOR CLAMP for q_i sinks ===
    # Mirror of the q_c donor clamp: aggregation and melt_ice are q_i
    # sinks that share the same explicit-Euler step with riming_i (a
    # q_i source from q_c).  Without this clamp, the combined
    # (aggregation + melt_ice) · dt can exceed available q_i, sending
    # q_i negative.  Mass is conserved because each sink rate appears
    # once in dq_i_dt as a sink and once elsewhere as a source — a
    # uniform rescale of both pieces preserves the budget.
    qi_sink_total = aggregation + melt_ice
    qi_avail = jnp.clip(q_i, 0.0)
    qi_scale = donor_clamp_scale(qi_avail, qi_sink_total, dt)
    aggregation = aggregation * qi_scale
    melt_ice = melt_ice * qi_scale

    # === DONOR CLAMP for q_s sinks ===
    # melt_snow is a q_s sink.  Same logic as the q_i clamp.
    qs_sink_total = melt_snow
    qs_avail = jnp.clip(q_s, 0.0)
    qs_scale = donor_clamp_scale(qs_avail, qs_sink_total, dt)
    melt_snow = melt_snow * qs_scale

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
    qc_sink_total = dq_c_au + dq_c_ac + bergeron + riming_i + riming_s + cond_evap_sink
    qc_avail = jnp.clip(q_c, 0.0)
    qc_scale = donor_clamp_scale(qc_avail, qc_sink_total, dt)
    dq_c_au = dq_c_au * qc_scale
    dq_c_ac = dq_c_ac * qc_scale
    bergeron = bergeron * qc_scale
    riming_i = riming_i * qc_scale
    riming_s = riming_s * qc_scale
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
    cond_pos = jnp.maximum(condensation, 0.0)
    qv_sink_total = cond_pos + jnp.maximum(dq_i_dep, 0.0)
    qv_avail = jnp.clip(q_v, 0.0)
    qv_scale = donor_clamp_scale(qv_avail, qv_sink_total, dt)
    # Scale only the positive (vapor-consuming) branch of condensation;
    # negative condensation (evaporation) is unaffected.
    condensation = jnp.where(condensation > 0.0, condensation * qv_scale, condensation)
    dq_i_dep = dq_i_dep * qv_scale

    # === SEDIMENTATION ===
    # Marshall-Palmer fall speeds V_t = a_v * (q * rho / rho_sfc)^b_v use
    # fractional exponents (b_v_r=0.5, b_v_i=0.25, b_v_s=0.3); guard the
    # AD path with safe_pow so cold-start columns (q=0) don't NaN gradients.
    rho_sfc = rho[:, -1:]
    rho_ratio = rho / jnp.clip(rho_sfc, 0.1)
    V_t_r = config.a_v_r * safe_pow(jnp.clip(q_r, 0.0) * rho_ratio, config.b_v_r)
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * safe_pow(jnp.clip(q_i, 0.0) * rho_ratio, config.b_v_i)
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * safe_pow(jnp.clip(q_s, 0.0) * rho_ratio, config.b_v_s)
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)

    # Joint donor caps: each `extra_sink` is the in-column sink that
    # shares the same explicit-Euler step as sedimentation.
    # Without these, the post-donor-clamp in-column sinks (aggregation +
    # melt_ice for q_i, melt_snow for q_s, evaporation for q_r) ALREADY
    # consume up to q/dt, AND sed independently can drain another q/dt
    # — driving the pool negative.  Mirrors the iter-29 q_r/evap fix.
    sed_r, precip_r = sedimentation_tendency(
        q_r, rho, V_t_r, dz, dt=dt,
        return_surface_flux=True,
        extra_sink=evaporation,
    )
    sed_i, precip_i = sedimentation_tendency(
        q_i, rho, V_t_i, dz, dt=dt, return_surface_flux=True,
        extra_sink=aggregation + melt_ice,
    )
    sed_s, precip_s = sedimentation_tendency(
        q_s, rho, V_t_s, dz, dt=dt, return_surface_flux=True,
        extra_sink=melt_snow,
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
        # Cloud water → ice/snow freezing releases latent heat of fusion
        # (~333 kJ/kg).  Bergeron is liquid → ice via the WBF mechanism,
        # riming is supercooled-droplet capture by ice/snow.  Both are
        # phase changes that release L_f; the moist-enthalpy invariant
        # ``h = c_pd T + L_v q_v - L_f q_ice`` requires this term for
        # column conservation.  Magnitude estimate: ~2 K/day at default
        # rates in mixed-phase clouds.
        + L_f * (bergeron + riming_i + riming_s) / c_pd
        - L_f * (melt_ice + melt_snow) / c_pd
    )

    # === COMBINE TENDENCIES ===
    dq_v_dt = -condensation + evaporation - dq_i_dep
    dq_c_dt = condensation - dq_c_au - dq_c_ac - bergeron - riming_i - riming_s
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + melt_ice + melt_snow + sed_r
    dq_i_dt = dq_i_dep + bergeron + riming_i - aggregation - melt_ice + sed_i
    dq_s_dt = aggregation + riming_s - melt_snow + sed_s

    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br
    dN_i_dt = dN_i_nuc - aggregation * jnp.clip(N_i, 0.0) / jnp.clip(q_i, 1e-15)

    # Precipitation (rain + ice + snow at surface) uses the dt-limited
    # surface flux from ``sedimentation_tendency`` so column water
    # conservation holds exactly when the CFL limiter fires.
    precipitation = precip_r + precip_i + precip_s

    # Pin dtype to the input precision so we never silently promote
    # the unused-species placeholders to f64 under x64 mode.
    z = jnp.zeros((ncol, nlev), dtype=T.dtype)
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=dq_i_dt,
        dq_s_dt=dq_s_dt,
        dq_g_dt=z,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=dN_i_dt,
        precipitation=precipitation,
    )
