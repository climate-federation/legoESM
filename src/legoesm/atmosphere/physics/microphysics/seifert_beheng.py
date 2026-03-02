"""Seifert-Beheng two-moment warm-rain microphysics.

A two-moment scheme tracking mass and number concentration of cloud
droplets and rain drops. Processes: saturation adjustment, autoconversion
(mass-dependent), accretion, self-collection, breakup, rain evaporation,
and sedimentation.

All operations use smooth (differentiable) approximations.

References
----------
- Seifert, A., & Beheng, K. D. (2001). A two-moment cloud microphysics
  parameterization for mixed-phase clouds. Meteorol. Atmos. Phys., 77, 127-151.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import saturation_mixing_ratio
from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)


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
    """Compute Seifert-Beheng two-moment warm-rain tendencies.

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
    N_c = hydrometeors.N_c
    N_r = hydrometeors.N_r
    sharpness = config.saturation_sharpness

    # Effective N_c: use config default if zero
    N_c_eff = jnp.where(N_c > 1.0, N_c, config.Nc_0 * jnp.ones_like(N_c))

    # Saturation adjustment
    q_sat = saturation_mixing_ratio(T, p_full)
    excess = q_v - q_sat
    cond_frac = jax.nn.sigmoid(sharpness * excess)
    condensation = cond_frac * excess

    # 1. Autoconversion (mass-dependent)
    x_c = jnp.clip(q_c, 0.0) * rho / jnp.clip(N_c_eff, 1.0)
    onset = jax.nn.sigmoid(sharpness * (x_c - config.x_star))
    dq_c_au = config.k_au * jnp.clip(q_c, 0.0) ** 2 * onset / rho
    dN_r_au = dq_c_au * rho / (config.x_star * 20.0)

    # 2. Accretion
    dq_c_ac = config.k_ac * jnp.clip(q_c, 0.0) * jnp.clip(q_r, 0.0) * rho

    # 3. Self-collection
    dN_r_sc = -config.k_sc * jnp.clip(N_r, 0.0) * jnp.clip(q_r, 0.0) * rho

    # 4. Breakup (opposes self-collection for large drops)
    # Mean rain drop diameter estimate: D_r ~ (q_r * rho / (N_r * pi/6 * rho_w))^(1/3)
    rho_w = constants.rho_water
    D_r = jnp.clip(
        (jnp.clip(q_r, 0.0) * rho / jnp.clip(N_r, 1.0) / (jnp.pi / 6.0 * rho_w)),
        0.0,
    ) ** (1.0 / 3.0)
    breakup_frac = jax.nn.sigmoid(config.breakup_sharpness * (D_r - config.D_eq))
    dN_r_br = -dN_r_sc * breakup_frac

    # 5. Rain evaporation
    subsaturation = jnp.clip(q_sat - q_v, 0.0) / jnp.clip(q_sat, 1e-10)
    evaporation = config.evap_coeff * subsaturation * jnp.clip(q_r, 0.0) ** 0.525

    # 6. Sedimentation
    rho_sfc = rho[:, -1:]
    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)

    # 7. Latent heating
    dT_dt = constants.L_v * condensation / constants.c_pd

    # Combine tendencies
    dq_v_dt = -condensation + evaporation
    dq_c_dt = condensation - dq_c_au - dq_c_ac
    dq_r_dt = dq_c_au + dq_c_ac - evaporation + sed_r
    dN_c_dt = -dq_c_au * rho / jnp.clip(x_c, 1e-20)
    dN_r_dt = dN_r_au + dN_r_sc + dN_r_br

    # Precipitation
    q_r_bot = jnp.clip(q_r[:, -1], 0.0)
    precipitation = q_r_bot * rho[:, -1] * jnp.clip(V_t_r[:, -1], 0.0)

    z = jnp.zeros((ncol, nlev))
    return MicrophysicsOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_dt=dq_c_dt,
        dq_r_dt=dq_r_dt,
        dq_i_dt=z,
        dq_s_dt=z,
        dq_g_dt=z,
        dN_c_dt=dN_c_dt,
        dN_r_dt=dN_r_dt,
        dN_i_dt=z,
        precipitation=precipitation,
    )
