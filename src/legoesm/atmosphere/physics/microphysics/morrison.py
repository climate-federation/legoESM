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
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.config import MorrisonConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


def _saturation_mixing_ratio_ice(T, p):
    """Ice saturation mixing ratio (Clausius-Clapeyron over ice)."""
    e_sat_i = 611.2 * jnp.exp(
        constants.L_s / constants.R_v * (1.0 / constants.T_freeze - 1.0 / T)
    )
    return constants.epsilon * e_sat_i / jnp.clip(p - e_sat_i, 1.0)


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

    N_c_eff = jnp.where(N_c > 1.0, N_c, config.Nc_0 * jnp.ones_like(N_c))

    # === WARM RAIN (same as Seifert-Beheng) ===
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess

    # Autoconversion
    x_c = jnp.clip(q_c, 0.0) * rho / jnp.clip(N_c_eff, 1.0)
    onset = jax.nn.sigmoid(sharpness * (x_c - config.x_star))
    dq_c_au = config.k_au * jnp.clip(q_c, 0.0) ** 2 * onset / rho
    dN_r_au = dq_c_au * rho / (config.x_star * 20.0)

    # Accretion
    dq_c_ac = config.k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho

    # Self-collection / breakup
    dN_r_sc = -config.k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho
    D_r = jnp.clip(
        (jnp.clip(q_r, 0.0) * rho / jnp.clip(N_r, 1.0) / (jnp.pi / 6.0 * constants.rho_water)),
        0.0,
    ) ** (1.0 / 3.0)
    breakup_frac = jax.nn.sigmoid(config.breakup_sharpness * (D_r - config.D_eq))
    dN_r_br = -dN_r_sc * breakup_frac

    # Rain evaporation
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    evaporation = config.evap_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525

    # === ICE PHASE ===
    T_freeze = constants.T_freeze
    f_ice = jax.nn.sigmoid(config.ice_sigmoid_sharpness * (config.cooper_T_act - T))

    # 1. Ice nucleation (Cooper 1986, smoothed)
    N_i_target = config.N_i0 * jnp.exp(
        config.cooper_a * jax.nn.softplus(T_freeze - T)
    ) / jnp.clip(rho, 0.1)
    dN_i_nuc = jnp.clip(N_i_target - N_i, 0.0) / jnp.clip(dt, 1.0)

    # 2. Depositional growth
    q_sat_i = _saturation_mixing_ratio_ice(T, p_full)
    S_i = q_v / jnp.clip(q_sat_i, 1e-10) - 1.0
    dq_i_dep = (
        config.dep_coeff
        * jax.nn.softplus(S_i)
        * jnp.clip(q_i, 0.0)
        * jnp.clip(N_i, 0.0) ** (1.0 / 3.0)
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

    # 6. Melting near T_freeze: ice/snow -> rain
    melt_frac = jax.nn.sigmoid(config.melt_sharpness * (T - T_freeze))
    melt_ice = config.melt_rate * jnp.clip(q_i, 0.0) * melt_frac
    melt_snow = config.melt_rate * jnp.clip(q_s, 0.0) * melt_frac

    # === SEDIMENTATION ===
    rho_sfc = rho[:, -1:]
    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    V_t_i = config.a_v_i * (jnp.clip(q_i, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_i
    V_t_i = jnp.clip(V_t_i, 0.0, 5.0)
    V_t_s = config.a_v_s * (jnp.clip(q_s, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_s
    V_t_s = jnp.clip(V_t_s, 0.0, 5.0)

    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)
    sed_i = sedimentation_tendency(q_i, rho, V_t_i, dz)
    sed_s = sedimentation_tendency(q_s, rho, V_t_s, dz)

    # === LATENT HEATING ===
    L_v = constants.L_v
    L_s = constants.L_s
    L_f = constants.L_f
    c_pd = constants.c_pd

    dT_dt = (
        L_v * condensation / c_pd
        + L_s * dq_i_dep / c_pd
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

    # Precipitation (rain + ice + snow at surface)
    precip_r = jnp.clip(q_r[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)
    precip_i = jnp.clip(q_i[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_i[:, -1], 0.0)
    precip_s = jnp.clip(q_s[:, -1], 0.0) * rho[:, -1] * jnp.clip(V_t_s[:, -1], 0.0)
    precipitation = precip_r + precip_i + precip_s

    z = jnp.zeros((ncol, nlev))
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
