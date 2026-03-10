"""Simplified EDMF convection — NOT a full Siebesma et al. (2007) EDMF.

A simplified single-updraft mass-flux convection scheme inspired by the
EDMF framework. Includes a prognostic updraft area fraction, entraining
plume equations, and mass-flux tendencies.

This is a reduced-complexity surrogate, not a production-grade EDMF
implementation. Key simplifications vs. a full EDMF:
- Single updraft plume (no multi-plume or downdraft components)
- Simplified entrainment/detrainment (exponential dilution)
- No stochastic or turbulence-based triggering
- No interaction with a sub-grid PDF closure

Suitable for idealized experiments and as a development baseline.

Algorithm:
1. Compute CAPE and moist adiabat for activation
2. Diagnose equilibrium updraft area fraction from CAPE
3. Prognostically relax a_u toward equilibrium
4. Compute entraining updraft T, q (exponential dilution)
5. Diagnose buoyancy and updraft velocity
6. Compute mass flux M_u = rho * a_u * w_u
7. Compute MF tendencies (detrainment form)
8. Diagnose precipitation from condensate detrainment

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity
  mass-flux approach for the convective boundary layer.
  J. Atmos. Sci., 64, 1230-1248.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics.thermodynamics import (
    saturation_mixing_ratio,
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import EDMFConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def edmf_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    a_u: jax.Array,
    dt: float,
    config: EDMFConfig = EDMFConfig(),
) -> Tuple[ConvectionOutput, jax.Array]:
    """Compute simplified EDMF convection tendencies.

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
    a_u : jax.Array
        Updraft area fraction, shape (ncol,).
    dt : float
        Model time step [s].
    config : EDMFConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    a_u_new : jax.Array
        Updated updraft area fraction, shape (ncol,).
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    # 1. Heights and density from hydrostatic balance
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)  # (ncol, nlev)
    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))  # (ncol, nlev)

    # Cumulative height from surface
    dz_rev = dz[:, ::-1]
    z = jnp.cumsum(dz_rev, axis=1)[:, ::-1]  # (ncol, nlev)

    # 2. CAPE and moist adiabat
    T_base = T[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    q_sat = saturation_mixing_ratio(T, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    # 3. Diagnosed equilibrium updraft area fraction
    a_u_eq = (
        jax.nn.sigmoid((cape - config.cape_threshold) / config.cape_activation_scale)
        * config.a_u_init
    )  # (ncol,)

    # 4. Prognostic update: relax a_u toward equilibrium, clip to [0, 0.5]
    a_u_new = a_u + dt * (a_u_eq - a_u) / config.tau_a
    a_u_new = jnp.clip(a_u_new, 0.0, 0.5)

    # 5. Entraining updraft properties (exponential dilution)
    dilution = jnp.exp(-config.epsilon_0 * z)  # (ncol, nlev)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_u = dilution * q_sat_moist + (1.0 - dilution) * q_v

    # 6. Buoyancy: B = g * (T_v_u - T_v_env) / T_v_env
    T_v_env = T * (1.0 + 0.61 * q_v)
    T_v_u = T_u * (1.0 + 0.61 * q_u)
    B = constants.g * (T_v_u - T_v_env) / jnp.clip(T_v_env, 1.0, None)  # (ncol, nlev)

    # 7. Updraft velocity: w_u = sqrt(2 * cumsum(clip(B*dz, 0)) + w_u_min^2)
    # Integrate buoyancy from surface upward
    B_dz = jnp.clip(B * dz, 0.0, None)
    B_dz_rev = B_dz[:, ::-1]
    B_integral = jnp.cumsum(B_dz_rev, axis=1)[:, ::-1]  # (ncol, nlev)
    w_u = jnp.sqrt(2.0 * B_integral + config.w_u_min ** 2)  # (ncol, nlev)

    # 8. Mass flux: M_u = rho * a_u * w_u
    M_u = rho * a_u_new[:, None] * w_u  # (ncol, nlev)

    # 9. MF tendencies (detrainment form)
    dT_dt = -M_u * config.delta_0 * (T_u - T) / jnp.clip(rho, 0.01, None)
    dq_v_dt = -M_u * config.delta_0 * (q_u - q_v) / jnp.clip(rho, 0.01, None)

    # 10. Precipitation from condensate detrainment
    condensate = jnp.clip(q_u - saturation_mixing_ratio(T_u, p_full), 0.0, None)
    precipitation = jnp.clip(
        jnp.sum(M_u * config.delta_0 * condensate * dp / constants.g, axis=1),
        0.0,
        None,
    )  # (ncol,)

    # Convective mask from CAPE activation
    convective_mask = jax.nn.sigmoid(
        (cape - config.cape_threshold) / config.cape_activation_scale
    )

    conv_out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=convective_mask,
    )

    return conv_out, a_u_new
