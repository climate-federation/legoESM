"""Kessler warm-rain microphysics column backend.

A simple one-moment warm-rain scheme tracking cloud water and rain.
Processes: saturation adjustment, autoconversion, accretion, evaporation,
rain sedimentation, and latent heating.

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Kessler (1969): On the Distribution and Continuity of Water
  Substance in Atmospheric Circulations.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.config import KesslerConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


def kessler_microphysics(
    T: jax.Array,
    q_v: jax.Array,
    hydrometeors: HydrometeorState,
    p_full: jax.Array,
    p_half: jax.Array,
    rho: jax.Array,
    dz: jax.Array,
    dt: float,
    config: KesslerConfig = KesslerConfig(),
) -> MicrophysicsOutput:
    """Compute Kessler microphysics tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    hydrometeors : HydrometeorState
        Hydrometeor state (only q_c, q_r used).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    rho : jax.Array
        Air density [kg/m^3], shape (ncol, nlev).
    dz : jax.Array
        Layer thickness [m], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : KesslerConfig

    Returns
    -------
    MicrophysicsOutput
    """
    ncol, nlev = T.shape
    q_c = hydrometeors.q_c
    q_r = hydrometeors.q_r
    sharpness = config.saturation_sharpness

    # Saturation mixing ratio
    q_sat = saturation_mixing_ratio(T, p_full)

    # 1. Saturation adjustment — convert from increment [kg/kg] to tendency [kg/kg/s]
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess / dt  # [kg/kg/s]

    dq_v_sat = -condensation
    dq_c_sat = condensation

    # 2. Autoconversion: cloud -> rain (smooth softplus)
    # dq_c_sat is a tendency [kg/kg/s]; multiply by dt to get increment [kg/kg]
    q_c_updated = q_c + dq_c_sat * dt
    autoconv = config.autoconversion_rate * jax.nn.softplus(
        q_c_updated - config.autoconversion_threshold
    )

    # 3. Accretion: cloud collected by rain
    accretion = config.accretion_coeff * q_c * jnp.clip(q_r, 0.0) ** 0.875

    # 4. Evaporation of rain
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    evaporation = config.evaporation_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525

    # 5. Rain sedimentation
    rho_sfc = rho[:, -1:]
    V_t = config.rain_fall_speed * jnp.sqrt(
        rho_sfc / jnp.clip(rho, 0.1)
    )
    sed_tend = sedimentation_tendency(q_r, rho, V_t, dz)

    # 6. Latent heating
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd

    # Combine tracer tendencies
    dq_v_dt = dq_v_sat + evaporation
    dq_c_dt = dq_c_sat - autoconv - accretion
    dq_r_dt = autoconv + accretion - evaporation + sed_tend

    # Precipitation: surface flux
    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
    V_t_bot = V_t[:, -1]
    precipitation = q_r_bot * rho[:, -1] * V_t_bot

    z = jnp.zeros((ncol, nlev))
    z1 = jnp.zeros((ncol,))
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=z,
        dN_r_dt=z,
        dN_i_dt=z,
        precipitation=precipitation,
    )
