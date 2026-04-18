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

from typing import NamedTuple, Tuple

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


class MassFluxClosureDiagnostics(NamedTuple):
    """Intermediate closure state reused by physical and ML mass-flux paths."""

    dz: jax.Array
    rho: jax.Array
    z: jax.Array
    T_moist: jax.Array
    cape: jax.Array
    M_eq: jax.Array
    M_c_new: jax.Array
    convective_mask: jax.Array


def diagnose_mass_flux_closure(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_c: jax.Array,
    dt: float,
    config: MassFluxConfig = MassFluxConfig(),
) -> MassFluxClosureDiagnostics:
    """Diagnose closure terms before computing mass-flux tendencies."""
    del q_v  # retained for interface symmetry with full convection call
    dp = p_half[:, 1:] - p_half[:, :-1]  # (ncol, nlev)

    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)
    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))

    dz_rev = dz[:, ::-1]
    z = jnp.cumsum(dz_rev, axis=1)[:, ::-1]

    T_base = T[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)

    M_eq = (
        jax.nn.sigmoid((cape - config.cape_threshold) / config.cape_activation_scale)
        * config.M_scale
    )
    M_c_new = jnp.maximum(M_c + dt * (M_eq - M_c) / config.tau_adj, 0.0)
    convective_mask = jax.nn.sigmoid(
        (cape - config.cape_threshold) / config.cape_activation_scale
    )

    return MassFluxClosureDiagnostics(
        dz=dz,
        rho=rho,
        z=z,
        T_moist=T_moist,
        cape=cape,
        M_eq=M_eq,
        M_c_new=M_c_new,
        convective_mask=convective_mask,
    )


def mass_flux_convection_from_closure(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    closure: MassFluxClosureDiagnostics,
    config: MassFluxConfig = MassFluxConfig(),
) -> ConvectionOutput:
    """Compute mass-flux tendencies from a supplied closure state."""
    del p_half
    dz = closure.dz
    rho = closure.rho
    z = closure.z
    T_moist = closure.T_moist
    cape = closure.cape
    M_c_new = closure.M_c_new
    convective_mask = closure.convective_mask
    p_base = p_full[:, -1:]  # (ncol, 1)
    p_top = p_full[:, :1]  # (ncol, 1)
    p_range = jnp.clip(p_base - p_top, 1.0, None)
    m_profile = jnp.sin(
        jnp.pi * (p_base - p_full) / p_range
    )  # (ncol, nlev) sinusoidal, 0 at base/top

    dilution = jnp.exp(-config.epsilon_0 * z)  # (ncol, nlev)
    T_u = dilution * T_moist + (1.0 - dilution) * T

    # Updraft moisture starts from a saturated cloud-base parcel and is then
    # diluted toward the environmental humidity profile by entrainment.
    q_sat_u = saturation_mixing_ratio(T_u, p_full)
    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
    q_u = dilution * q_sat_base + (1.0 - dilution) * q_v

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

    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
    dq_subsidence = (M_profile / rho_safe) * dq_dz

    dT_detrain = config.delta_0 * M_profile * (T_u - T) / rho_safe
    dq_detrain = config.delta_0 * M_profile * (q_u - q_v) / rho_safe

    dT_dt = dT_subsidence + dT_detrain  # (ncol, nlev)
    dq_v_dt = dq_subsidence + dq_detrain  # (ncol, nlev)

    condensate = jnp.clip(q_u - saturation_mixing_ratio(T_u, p_full), 0.0, None)
    precipitation = jnp.clip(
        jnp.sum(
            config.delta_0 * M_profile * condensate * dz,
            axis=1,
        ),
        0.0,
        None,
    )  # (ncol,) [kg/m^2/s]

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        precipitation=precipitation,
        cape=cape,
        convective_mask=convective_mask,
    )


def mass_flux_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_c: jax.Array,
    dt: float,
    config: MassFluxConfig = MassFluxConfig(),
) -> Tuple[ConvectionOutput, jax.Array]:
    """Compute Prognostic Mass-Flux convection tendencies."""
    closure = diagnose_mass_flux_closure(
        T=T,
        q_v=q_v,
        p_full=p_full,
        p_half=p_half,
        M_c=M_c,
        dt=dt,
        config=config,
    )
    conv_out = mass_flux_convection_from_closure(
        T=T,
        q_v=q_v,
        p_full=p_full,
        p_half=p_half,
        closure=closure,
        config=config,
    )
    return conv_out, closure.M_c_new
