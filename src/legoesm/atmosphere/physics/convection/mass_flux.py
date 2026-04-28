"""Mass-flux convection schemes (Arakawa-Wu and simplified EDMF).

This module provides two prognostic mass-flux convection
parameterizations that share most of their numerics:

1. **Prognostic Mass-Flux** (``mass_flux_convection``): Arakawa-Wu type
   scheme with a single column-mean prognostic variable ``M_c``. The
   vertical profile of the mass flux is a fixed sinusoidal shape;
   ``M_c`` relaxes toward an equilibrium value diagnosed from CAPE.

2. **Simplified EDMF** (``edmf_convection``): single-updraft mass-flux
   scheme inspired by Siebesma et al. (2007). The prognostic variable
   is the updraft area fraction ``a_u``; the mass-flux profile
   ``M_u(z) = ρ · a_u · w_u(z)`` is built from a buoyancy-derived
   updraft velocity. This is a reduced-complexity surrogate, not a
   full EDMF (single plume, no downdrafts, no stochastic triggering,
   no PDF closure).

Both schemes share the same downstream kernel: given a vertical
mass-flux profile ``M(z)``, an entraining/diluting updraft ``(T_u,
q_u)``, and grid geometry, they apply the same compensating
subsidence + detrainment tendencies and diagnose surface
precipitation from the column integral of detrained condensate.
The kernel is factored into ``_apply_mass_flux_kernel`` so the two
schemes differ only in (a) the prognostic update rule and (b) the
M(z) profile.

All operations use smooth (differentiable) approximations for
compatibility with ``jax.grad``.

References
----------
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity
  mass-flux approach for the convective boundary layer.
  J. Atmos. Sci., 64, 1230-1248.
- Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models. Mon. Wea. Rev., 117,
  1779-1800.
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
from legoesm.atmosphere.physics.convection.config import (
    EDMFConfig,
    MassFluxConfig,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput


# =============================================================================
# Shared helpers — used by both the Arakawa-Wu and simplified EDMF paths.
# =============================================================================


def _compute_column_geometry(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute layer thickness, density, and surface-relative height.

    Returns ``(dz, rho, z)``, all of shape ``(ncol, nlev)``. ``dz`` is
    the layer thickness from hydrostatic balance using the mid-layer
    pressure; ``rho`` is the dry-air density at full levels; ``z`` is
    the cumulative height above the surface (note: levels are ordered
    top-down, so ``z[:, -1]`` is the surface).
    """
    dp = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = constants.R_d * T * dp / (constants.g * jnp.clip(p_mid, 1.0, None))
    dz = jnp.abs(dz)
    rho = p_full / (constants.R_d * jnp.clip(T, 1.0, None))
    # Cumulative height from the surface (level nlev-1) upward.
    z = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    return dz, rho, z


def _compute_cape_diagnostics(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    cape_threshold: float,
    cape_activation_scale: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute moist-adiabat profile, CAPE, and a smooth convective mask.

    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` is shape
    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are shape
    ``(ncol,)``. The mask is a sigmoid of ``(cape -
    cape_threshold) / cape_activation_scale`` and is reused as the
    smooth activation factor for both schemes.
    """
    T_base = T[:, -1]
    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)
    convective_mask = jax.nn.sigmoid(
        (cape - cape_threshold) / cape_activation_scale
    )
    return T_moist, cape, convective_mask


def _compute_centered_gradients(
    T: jax.Array,
    q_v: jax.Array,
    z: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Centered vertical gradients with zero edges via ``jnp.pad``.

    Single Pad HLO op vs. allocate-zeros + scatter. Returns
    ``(dT_dz, dq_dz)``, both shape ``(ncol, nlev)``.
    """
    dz_centered = jnp.clip(z[:, :-2] - z[:, 2:], 1.0, None)
    dT_dz = jnp.pad((T[:, :-2] - T[:, 2:]) / dz_centered, ((0, 0), (1, 1)))
    dq_dz = jnp.pad((q_v[:, :-2] - q_v[:, 2:]) / dz_centered, ((0, 0), (1, 1)))
    return dT_dz, dq_dz


def _apply_mass_flux_kernel(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    T_u: jax.Array,
    q_u: jax.Array,
    M_profile: jax.Array,
    z: jax.Array,
    rho: jax.Array,
    delta_0: float,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Mass-flux core kernel: compensating subsidence + detrainment.

    Given the vertical mass-flux profile ``M_profile`` and the
    entraining updraft properties ``(T_u, q_u)``, returns
    ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape ``(ncol, nlev)``.

    The decomposition follows Tiedtke (1989) / Siebesma et al. (2007):
      (a) Compensating subsidence: ``(M/rho) * (dT/dz + g/c_p)`` for T
          and ``(M/rho) * dq/dz`` for moisture (q is conserved so no
          adiabatic correction).
      (b) Detrainment mixing: ``+delta_0 * M * (X_u - X) / rho``.

    The convective source for cloud water is the per-level detrained
    condensate rate ``dq_c_conv_dt = delta_0 * M * condensate / rho``
    [kg/kg/s], non-negative by construction. Microphysics processes
    this through its full chain (autoconversion, sedimentation,
    evaporation) and produces the resulting surface precipitation;
    convection no longer assumes the condensate falls instantly.
    Unit check: (1/m) * (kg/m²/s) * (kg/kg) / (kg/m³) = 1/s × kg/kg.
    """
    dT_dz, dq_dz = _compute_centered_gradients(T, q_v, z)
    rho_safe = jnp.clip(rho, 0.01, None)

    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
    dq_subsidence = (M_profile / rho_safe) * dq_dz

    dT_detrain = delta_0 * M_profile * (T_u - T) / rho_safe
    dq_detrain = delta_0 * M_profile * (q_u - q_v) / rho_safe

    dT_dt = dT_subsidence + dT_detrain
    dq_v_dt = dq_subsidence + dq_detrain

    condensate = jnp.clip(q_u - saturation_mixing_ratio(T_u, p_full), 0.0, None)
    dq_c_conv_dt = delta_0 * M_profile * condensate / rho_safe
    return dT_dt, dq_v_dt, dq_c_conv_dt


# =============================================================================
# Arakawa-Wu prognostic mass-flux (M_c × fixed sinusoidal profile)
# =============================================================================


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

    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
    )

    M_eq = convective_mask * config.M_scale
    M_c_new = jnp.maximum(M_c + dt * (M_eq - M_c) / config.tau_adj, 0.0)

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

    p_base = p_full[:, -1:]
    p_top = p_full[:, :1]
    p_range = jnp.clip(p_base - p_top, 1.0, None)
    # Sinusoidal vertical profile: 0 at base/top, peak at mid-troposphere.
    m_profile = jnp.sin(jnp.pi * (p_base - p_full) / p_range)
    M_profile = M_c_new[:, None] * m_profile

    # Entraining updraft: thermal dilutes from moist-adiabat parcel toward
    # environment; moisture starts saturated at the cloud base parcel and
    # dilutes toward the environmental humidity.
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_sat_base = saturation_mixing_ratio(T[:, -1:], p_full[:, -1:])
    q_u = dilution * q_sat_base + (1.0 - dilution) * q_v

    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_u=q_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
    )

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
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


# =============================================================================
# Simplified EDMF (M_u(z) = ρ · a_u · w_u(z) from a buoyancy integral)
# =============================================================================


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
        Temperature at full levels [K], shape ``(ncol, nlev)``.
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape ``(ncol, nlev)``.
    p_full : jax.Array
        Pressure at full levels [Pa], shape ``(ncol, nlev)``.
    p_half : jax.Array
        Pressure at half levels [Pa], shape ``(ncol, nlev+1)``.
    a_u : jax.Array
        Updraft area fraction, shape ``(ncol,)``.
    dt : float
        Model time step [s].
    config : EDMFConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    a_u_new : jax.Array
        Updated updraft area fraction, shape ``(ncol,)``.
    """
    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
    )

    # Diagnosed equilibrium updraft area fraction; prognostic relaxation
    # then clipping to a physically plausible range.
    a_u_eq = convective_mask * config.a_u_init
    a_u_new = a_u + dt * (a_u_eq - a_u) / config.tau_a
    a_u_new = jnp.clip(a_u_new, 0.0, 0.5)

    # Entraining updraft: thermal dilutes from the moist adiabat toward
    # environment; moisture follows the moist-adiabat saturation profile
    # (this differs from mass_flux, which dilutes from a single base
    # parcel — a small but deliberate scientific distinction between
    # the two schemes).
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_u = dilution * q_sat_moist + (1.0 - dilution) * q_v

    # Buoyancy: B = g * (T_v_u - T_v_env) / T_v_env (only positive part
    # contributes to updraft kinetic energy).
    T_v_env = T * (1.0 + 0.61 * q_v)
    T_v_u = T_u * (1.0 + 0.61 * q_u)
    B = constants.g * (T_v_u - T_v_env) / jnp.clip(T_v_env, 1.0, None)

    # Updraft velocity from buoyancy integral (surface upward), with a
    # small floor for numerical stability.
    B_dz_rev = jnp.clip(B * dz, 0.0, None)[:, ::-1]
    B_integral = jnp.cumsum(B_dz_rev, axis=1)[:, ::-1]
    w_u = jnp.sqrt(2.0 * B_integral + config.w_u_min ** 2)

    # Mass flux profile: M_u(z) = rho * a_u * w_u(z).
    M_profile = rho * a_u_new[:, None] * w_u

    del dz  # (kept for interface symmetry — kernel no longer needs it)
    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_u=q_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
    )

    conv_out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
    )
    return conv_out, a_u_new
