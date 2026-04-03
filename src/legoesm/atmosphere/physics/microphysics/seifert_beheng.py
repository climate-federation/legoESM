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
from legoesm.atmosphere.physics.microphysics.config import SeifertBehengConfig
from legoesm.atmosphere.physics.microphysics.output import (
    HydrometeorState,
    MicrophysicsOutput,
    sedimentation_tendency,
)
from legoesm.atmosphere.physics.microphysics._warm_rain import (
    saturation_adjustment,
    effective_Nc,
    autoconversion_sb,
    accretion,
    self_collection_breakup,
    rain_evaporation,
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

    N_c_eff = effective_Nc(N_c, config.Nc_0)

    condensation, q_sat = saturation_adjustment(T, q_v, p_full, dt, sharpness)

    # 1. Autoconversion (mass-dependent)
    dq_c_au, dN_r_au, x_c = autoconversion_sb(
        q_c, N_c_eff, rho, config.k_au, config.x_star, sharpness,
    )

    # 2. Accretion
    dq_c_ac = accretion(q_c, q_r, rho, config.k_ac)

    # 3-4. Self-collection and breakup
    dN_r_sc, dN_r_br = self_collection_breakup(
        N_r, q_r, rho, config.k_sc, config.breakup_sharpness, config.D_eq,
    )

    # 5. Rain evaporation
    evaporation = rain_evaporation(q_v, q_r, q_sat, config.evap_coeff)

    # 6. Sedimentation
    rho_sfc = rho[:, -1:]
    V_t_r = config.a_v_r * (jnp.clip(q_r, 0.0) * rho / jnp.clip(rho_sfc, 0.1)) ** config.b_v_r
    V_t_r = jnp.clip(V_t_r, 0.0, 20.0)
    sed_r = sedimentation_tendency(q_r, rho, V_t_r, dz)

    # 7. Latent heating
    dT_dt = constants.L_v * (condensation - evaporation) / constants.c_pd

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
