"""Prognostic Mass-Flux convection scheme (Arakawa-Wu type).

A mass-flux-based convection parameterization with a prognostic base
mass flux variable M_c. The scheme uses quasi-equilibrium closure with
CAPE-based activation and produces compensating subsidence tendencies.

Algorithm:
1. Compute CAPE and moist adiabat for equilibrium closure
2. Diagnose equilibrium mass flux from CAPE (sigmoid activation)
3. Prognostically relax M_c toward equilibrium
4. Compute entraining updraft properties (exponential dilution)
5. Compute compensating subsidence heating/drying tendencies
6. Diagnose precipitation from detrainment of condensate

All operations use smooth (differentiable) approximations for
compatibility with jax.grad.

References
----------
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
"""

from __future__ import annotations

from typing import Tuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    compute_moist_adiabat,
    compute_cape,
)
from legoesm.atmosphere.physics.convection.config import MassFluxConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


def mass_flux_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_c: jax.Array,
    dt: float,
    config: MassFluxConfig = MassFluxConfig(),
) -> Tuple[ConvectionOutput, jax.Array]:
    """Compute Prognostic Mass-Flux convection tendencies.

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
    M_c : jax.Array
        Base mass flux [kg/m^2/s], shape (ncol,).
    dt : float
        Model time step [s].
    config : MassFluxConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    M_c_new : jax.Array
        Updated base mass flux, shape (ncol,).
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    # 1. Approximate heights and density
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)  # (ncol, nlev)
    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))  # (ncol, nlev)

    # Cumulative height from surface (bottom is index -1)
    dz_rev = dz[:, ::-1]  # surface first
    z_cumsum = jnp.cumsum(dz_rev, axis=1)[:, ::-1]  # (ncol, nlev)
    z = z_cumsum  # height above surface at full levels

    # 2. CAPE and moist adiabat
    T_base = T[:, -1]  # (ncol,)
    T_moist = compute_moist_adiabat(T_base, p_full)  # (ncol, nlev)
    cape = compute_cape(T, T_moist, p_full, p_half)  # (ncol,)

    # 3. Quasi-equilibrium closure: sigmoid activation on CAPE
    M_eq = (
        jax.nn.sigmoid((cape - config.cape_threshold) / config.cape_activation_scale)
        * config.M_scale
    )  # (ncol,)

    # 4. Prognostic update: relax M_c toward M_eq, softplus floor
    M_c_new = M_c + dt * (M_eq - M_c) / config.tau_adj
    M_c_new = jax.nn.softplus(M_c_new)  # ensure non-negative

    # 5. Mass flux profile: sinusoidal in pressure
    p_base = p_full[:, -1:]  # (ncol, 1)
    p_top = p_full[:, :1]    # (ncol, 1)
    p_range = jnp.clip(p_base - p_top, 1.0, None)
    m_profile = jnp.sin(
        jnp.pi * (p_base - p_full) / p_range
    )  # (ncol, nlev) sinusoidal, 0 at base/top

    # 6. Entraining updraft temperature (exponential dilution)
    dilution = jnp.exp(-config.epsilon_0 * z)  # (ncol, nlev)
    T_u = dilution * T_moist + (1.0 - dilution) * T

    # Updraft moisture
    q_sat_u = saturation_mixing_ratio(T_u, p_full)
    q_u = dilution * q_sat_u + (1.0 - dilution) * q_v

    # 7. Environmental tendencies from mass flux
    # Two contributions (standard mass-flux decomposition):
    #   (a) Compensating subsidence: (M/rho) * (dT/dz + g/cp) for T,
    #                                (M/rho) * dq/dz for moisture
    #   (b) Detrainment mixing:      +delta * M * (T_u - T_env) / rho

    # Environmental gradient (centered differences, zero at boundaries)
    dT_dz = jnp.zeros_like(T)
    dT_dz = dT_dz.at[:, 1:-1].set(
        (T[:, :-2] - T[:, 2:]) / jnp.clip(z[:, :-2] - z[:, 2:], 1.0, None)
    )

    dq_dz = jnp.zeros_like(q_v)
    dq_dz = dq_dz.at[:, 1:-1].set(
        (q_v[:, :-2] - q_v[:, 2:]) / jnp.clip(z[:, :-2] - z[:, 2:], 1.0, None)
    )

    rho_safe = jnp.clip(rho, 0.01, None)
    M_profile = M_c_new[:, None] * m_profile  # (ncol, nlev)

    # (a) Compensating subsidence (Tiedtke 1989; Siebesma et al. 2007):
    #     For T: (M/rho)*(dT/dz + g/cp) — includes adiabatic compression
    #     For q: (M/rho)*dq/dz — conserved variable, no adiabatic correction
    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
    dq_subsidence = (M_profile / rho_safe) * dq_dz

    # (b) Detrainment: updraft air mixes into environment
    dT_detrain = config.delta_0 * M_profile * (T_u - T) / rho_safe
    dq_detrain = config.delta_0 * M_profile * (q_u - q_v) / rho_safe

    dT_dt = dT_subsidence + dT_detrain  # (ncol, nlev)
    dq_v_dt = dq_subsidence + dq_detrain  # (ncol, nlev)

    # 8. Precipitation from detrainment of condensate
    condensate = jnp.clip(q_u - saturation_mixing_ratio(T_u, p_full), 0.0, None)
    # Precipitation: delta_0 [1/m] * M_profile [kg/m^2/s] * condensate [kg/kg] * dz [m]
    # gives [kg/m^2/s]. Using dp/g would introduce an extra density factor.
    precipitation = jnp.clip(
        jnp.sum(
            config.delta_0 * M_profile * condensate * dz,
            axis=1,
        ),
        0.0,
        None,
    )  # (ncol,) [kg/m^2/s]

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

    return conv_out, M_c_new
