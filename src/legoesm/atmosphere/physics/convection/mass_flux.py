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
from legoesm.atmosphere.physics._shared import (
    compute_layer_dz,
    compute_rho,
    virtual_temperature,
)
from legoesm.atmosphere.physics.thermodynamics import (
    parcel_profile_and_cape,
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
    q_v: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute layer thickness, density, and surface-relative height.

    Returns ``(dz, rho, z)``, all of shape ``(ncol, nlev)``.  Delegates
    to the shared atmosphere column helpers (`compute_layer_dz`,
    `compute_rho`) so a single hypsometric/EOS convention is used by
    every parameterization.  When ``q_v`` is supplied the geometry
    uses virtual temperature — moist tropical columns are ~1 % thicker
    and ~1 % less dense than the dry calculation, which biases the
    mass-flux closure when omitted.  Levels are ordered top-down, so
    ``z[:, -1]`` is the surface.
    """
    dz = compute_layer_dz(T, p_half, q_v=q_v)
    rho = compute_rho(T, p_full, q_v=q_v)
    # Full-level (cell-centre) height above the surface.  Cumulative
    # ``cumsum(dz[::-1])[::-1]`` gives the height of the *top* of each
    # layer (interface above the level); subtracting half the local
    # thickness places the height at the layer mid-point, which is
    # where ``T``/``q`` live.  An earlier formulation used the layer-top
    # value, biasing parcel ascent diagnostics by ~½ layer per level
    # (~10–250 m depending on resolution).  Codex iter-3 finding #8.
    z_top = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z = z_top - 0.5 * dz
    return dz, rho, z


def _compute_cape_diagnostics(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    cape_threshold: float,
    cape_activation_scale: float,
    q_v: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute parcel profile, CAPE, and a smooth convective mask.

    When ``q_v`` is provided the parcel is lifted as **dry adiabat below
    the LCL, moist adiabat above** (using the surface-layer water-vapor
    mixing ratio as the launch humidity) and CAPE is computed with
    **virtual temperature**.  This is the physically correct trigger
    for unsaturated boundary layers; the legacy ``q_v=None`` path keeps
    the saturated-from-base assumption for callers that have not been
    migrated.

    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` has shape
    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are ``(ncol,)``.
    The mask is a sigmoid of ``(cape - cape_threshold) /
    cape_activation_scale``.
    """
    # Shared parcel -> CAPE recipe (dry->LCL->moist lift + virtual-T CAPE when
    # q_v is threaded; saturated-from-base legacy when q_v is None).
    T_moist, cape = parcel_profile_and_cape(T, p_full, p_half, q_v=q_v)
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


def stratosphere_mass_flux_gate(
    p_full: jax.Array,
    p_min_convection: float = 10_000.0,
    p_gate_sharpness: float = 1_500.0,
) -> jax.Array:
    """Smooth sigmoid factor in [0, 1] that vanishes above the
    tropopause (low ``p``) and equals one in the troposphere.

    Multiplying any mass-flux profile by the returned factor prevents
    convective tendencies from accumulating in the model top layer,
    where the small mass per unit area (Δp/g) would amplify modest
    heating into unphysical spikes (>400 K observed in 1-year RCE).

    Defaults: cutoff at 100 hPa (canonical tropical tropopause) with
    a 15-hPa transition width.  This gives factor ≈ 0.013 at the
    model top (35 hPa), 0.034 at 50 hPa, 0.5 at 100 hPa, 0.91 at
    130 hPa, and ≈ 1.0 below 200 hPa — i.e. the gate is *actually
    closed* (not merely attenuated) in the deep stratosphere while
    leaving the upper troposphere unaffected.  ``p_gate_sharpness``
    must be << ``p_min_convection`` for the sigmoid to saturate
    within the integration range; sharpness ≥ p_min only attenuates.

    Differentiable everywhere; ``p_gate_sharpness`` sets the width of
    the transition (Pa).
    """
    return jax.nn.sigmoid(
        (p_full - p_min_convection) / jnp.maximum(p_gate_sharpness, 1.0)
    )


def _apply_mass_flux_kernel(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    T_u: jax.Array,
    q_v_u: jax.Array,
    q_c_u: jax.Array,
    M_profile: jax.Array,
    z: jax.Array,
    rho: jax.Array,
    delta_0: float,
    M_u_max: float,
    p_min_convection: float = 10_000.0,
    p_gate_sharpness: float = 1_500.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Mass-flux core kernel: compensating subsidence + detrainment.

    Given the vertical mass-flux profile ``M_profile`` and the
    entraining updraft thermodynamics ``(T_u, q_v_u, q_c_u)``,
    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
    ``(ncol, nlev)``.

    The decomposition follows Tiedtke (1989) / Siebesma et al. (2007):
      (a) Compensating subsidence: ``(M/rho) * (dT/dz + g/c_p)`` for T
          and ``(M/rho) * dq/dz`` for moisture (q is conserved so no
          adiabatic correction).
      (b) Detrainment mixing: ``+delta_0 * M * (X_u - X) / rho``.

    The mass flux ``M_profile`` is gated by a smooth sigmoid in
    pressure so that levels above ``p_min_convection`` (default 100
    hPa, the canonical tropical tropopause) receive no convective
    tendency.  See ``stratosphere_mass_flux_gate`` for details.

    The convective source for cloud water is the per-level detrainment
    of the plume's cloud water:
    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
    by construction.  Splitting the plume into separate vapor (``q_v_u``)
    and cloud (``q_c_u``) pieces — instead of a single ``q_u`` that
    conflates total water with vapor — is what makes the column MSE
    budget close.  The earlier formulation passed ``q_u = q_v_u + q_c_u``
    as if it were vapor and computed condensate as ``max(q_u - q_sat,
    0)``, which is essentially zero for an entraining-diluted plume —
    the leaf then leaked latent energy.  Microphysics processes
    ``dq_c_conv_dt`` through its full chain (autoconversion,
    sedimentation, evaporation) and produces the resulting surface
    precipitation; convection no longer assumes the condensate falls
    instantly.
    Unit check: (1/m) * (kg/m²/s) * (kg/kg) / (kg/m³) = 1/s × kg/kg.
    """
    dT_dz, dq_dz = _compute_centered_gradients(T, q_v, z)
    rho_safe = jnp.clip(rho, 0.01, None)

    # Per-level mass-flux cap.  The plume integrator can yield ``M_u``
    # that grows with height when ``epsilon > delta`` (entraining
    # plumes) or that responds non-linearly to a high-CAPE column.
    # Per-layer convective heating ``≈ delta_0 · M_u · (T_u−T)/ρ``
    # scales linearly with ``M_u``, so an uncapped ``M_u`` produces
    # column heating well in excess of what surface fluxes can supply
    # and destabilises the integration.  Callers thread the cap from
    # their config NamedTuple (typically ``config.M_b_max ≈ 0.05
    # kg/m²/s``, the literature peak tropical updraft mass flux); the
    # clip bounds per-layer tendencies without distorting the moist
    # adiabat or the q_v / q_c split.
    M_profile = jnp.clip(M_profile, 0.0, M_u_max)

    # Stratospheric pressure gate — see ``stratosphere_mass_flux_gate``.
    M_profile = M_profile * stratosphere_mass_flux_gate(
        p_full, p_min_convection, p_gate_sharpness,
    )

    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
    dq_subsidence = (M_profile / rho_safe) * dq_dz

    dT_detrain = delta_0 * M_profile * (T_u - T) / rho_safe
    dq_detrain = delta_0 * M_profile * (q_v_u - q_v) / rho_safe

    dT_dt = dT_subsidence + dT_detrain
    dq_v_dt = dq_subsidence + dq_detrain

    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
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
    dz, rho, z = _compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
        q_v=q_v,
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

    # Entraining updraft.  The undiluted plume rises with a fixed
    # total-water reservoir equal to the actual launched-parcel vapor
    # ``q_v_sfc`` (NOT ``q_sat_sfc`` — using the saturation value here
    # would let a 5%-RH desert column produce convective cloud water
    # because the plume would "remember" being saturated at base when
    # it never was).  At each level the undiluted plume vapor saturates
    # at ``min(q_sat_moist, q_v_sfc)`` and condenses the excess; below
    # the actual LCL (``q_sat_moist > q_v_sfc``) condensation is zero.
    # Entrainment dilutes both T and q_v with environmental values; the
    # entrained env air carries no q_c, so q_c_u just scales by
    # ``dilution``.  The fix keeps the saturated-surface case
    # (``q_v_sfc = q_sat_sfc``) identical to the old formula.
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_v_sfc = q_v[:, -1:]
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_v_u_undiluted = jnp.minimum(q_sat_moist, q_v_sfc)
    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
    q_v_u = dilution * q_v_u_undiluted + (1.0 - dilution) * q_v
    q_c_u = dilution * q_c_u_undiluted

    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_v_u=q_v_u,
        q_c_u=q_c_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
        M_u_max=config.M_b_max,
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
    dz, rho, z = _compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
        q_v=q_v,
    )

    # Diagnosed equilibrium updraft area fraction; prognostic relaxation
    # then clipping to a physically plausible range.
    a_u_eq = convective_mask * config.a_u_init
    a_u_new = a_u + dt * (a_u_eq - a_u) / config.tau_a
    a_u_new = jnp.clip(a_u_new, 0.0, 0.5)

    # Entraining updraft.  Same correction as in ``mass_flux_convection``:
    # the undiluted plume's water reservoir is the *actual* launched-parcel
    # vapor ``q_v_sfc``, not ``q_sat_sfc``.  Without this the plume would
    # condense in dry columns even when the surface parcel never reached
    # saturation (e.g. 5%-RH desert column gives ~2 mm/day of spurious
    # convective precipitation — see ``test_no_cloud_water_in_dry_column``).
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_v_sfc = q_v[:, -1:]
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_v_u_undiluted = jnp.minimum(q_sat_moist, q_v_sfc)
    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
    q_v_u = dilution * q_v_u_undiluted + (1.0 - dilution) * q_v
    q_c_u = dilution * q_c_u_undiluted

    # Buoyancy: B = g * (T_v_u - T_v_env) / T_v_env.  Vapor contributes
    # ``+(R_v/R_d - 1) q_v ≈ +0.608 q_v`` (water vapour is lighter than
    # dry air) and cloud water contributes ``-q_c`` (the loaded
    # condensate is mass drag, not buoyancy).  An earlier formulation
    # used ``T_v = T*(1 + 0.61*(q_v + q_c))`` which treated q_c with the
    # *wrong sign* — making cloudy plumes spuriously buoyant — so we
    # use the standard form here.  ``virtual_temperature`` returns
    # ``T*(1 + 0.608·q_v)`` from ``constants.epsilon``; the cloud-water
    # loading term ``-T_u·q_c_u`` is added explicitly.
    T_v_env = virtual_temperature(T, q_v)
    T_v_u = virtual_temperature(T_u, q_v_u) - T_u * q_c_u
    B = constants.g * (T_v_u - T_v_env) / jnp.clip(T_v_env, 1.0, None)

    # Updraft velocity from buoyancy integral (surface upward), with a
    # small floor for numerical stability.
    B_dz_rev = jnp.clip(B * dz, 0.0, None)[:, ::-1]
    B_integral = jnp.cumsum(B_dz_rev, axis=1)[:, ::-1]
    # Floor the sqrt argument at a tiny positive so the updraft velocity stays
    # AD-safe even if a caller sets ``w_u_min = 0``: at a no-convection column
    # ``B_integral = 0`` and ``2B + w_u_min^2 = 0`` would give sqrt'(0) = inf
    # (NaN reverse-mode grad).  Forward is unchanged for any w_u_min > 0 or
    # buoyant column (the 1e-12 m^2/s^2 floor is far below w_u_min^2 ~ 0.01).
    w_u = jnp.sqrt(jnp.maximum(2.0 * B_integral + config.w_u_min ** 2, 1e-12))

    # Mass flux profile: M_u(z) = rho * a_u * w_u(z).
    M_profile = rho * a_u_new[:, None] * w_u

    del dz  # (kept for interface symmetry — kernel no longer needs it)
    dT_dt, dq_v_dt, dq_c_conv_dt = _apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_v_u=q_v_u,
        q_c_u=q_c_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
        M_u_max=config.M_b_max,
    )

    conv_out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
    )
    return conv_out, a_u_new
