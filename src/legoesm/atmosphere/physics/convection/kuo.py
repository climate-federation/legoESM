"""Kuo moisture convergence convection scheme.

A moisture-convergence-based convection parameterization. Convective
heating and moistening are proportional to the column-integrated moisture
excess above saturation, partitioned by alpha_heat.

Algorithm (Kuo 1965/1974 — simplified column formulation):
1. Compute column moisture excess above saturation
2. Smooth sigmoid trigger based on moisture convergence
3. Compute moist adiabat reference profile
4. Relax temperature and moisture toward reference profiles
5. Diagnose precipitation from moisture convergence

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Kuo, H. L. (1965). On formation and intensification of tropical
  cyclones through latent heat release by cumulus convection.
  J. Atmos. Sci., 22, 40-63.
- Kuo, H. L. (1974). Further studies of the parameterization of the
  influence of cumulus convection on large-scale flow.
  J. Atmos. Sci., 31, 1232-1240.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import (
    saturation_mixing_ratio,
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import KuoConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def kuo_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: KuoConfig = KuoConfig(),
) -> ConvectionOutput:
    """Compute Kuo moisture convergence convection tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : KuoConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    # 1. Saturation mixing ratio
    q_sat = saturation_mixing_ratio(T, p_full)  # (ncol, nlev)

    # 2. Column moisture excess: softplus for smooth positive part
    excess = jax.nn.softplus(q_v - q_sat)  # (ncol, nlev)
    MC = jnp.sum(excess * dp, axis=1) / constants.g  # (ncol,) [kg/m^2]

    # 3. Smooth trigger based on moisture convergence
    trigger = jax.nn.sigmoid(
        config.smooth_trigger_sharpness * (MC - config.mc_threshold)
    )  # (ncol,)

    # 4. Moist adiabatic reference profile from surface temperature
    T_base = T[:, -1]  # (ncol,)
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)

    # 5. Heating tendency: relax toward moist adiabat
    dT_dt = (
        trigger[:, None]
        * config.alpha_heat
        * (T_moist - T)
        / config.tau_relax
    )  # (ncol, nlev)

    # 6. Moistening tendency: relax toward saturation
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    dq_v_dt = (
        trigger[:, None]
        * (1.0 - config.alpha_heat)
        * (q_sat_moist - q_v)
        / config.tau_relax
    )  # (ncol, nlev)

    # 7. Precipitation from implied condensation (latent heat budget closure)
    # P = ∫ (dT/dt * c_pd / L_v) dp/g — ensures energy-moisture consistency
    implied_condensation = dT_dt * constants.c_pd / constants.L_v  # (ncol, nlev)
    precipitation = jnp.clip(
        jnp.sum(implied_condensation * dp, axis=1) / constants.g,
        0.0,
        None,
    )  # (ncol,)

    # 8. CAPE diagnostic
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=trigger,
    )
