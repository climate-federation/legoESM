"""Boussinesq Hydrostatic Primitive Equations on the cubed-sphere.

Vector-invariant form (consistent with shallow_water.py and primitive_eq.py):

    du/dt =  (zeta+f)*v - dB/dx - (1/rho_0)*dp'/dx + A_h*lap(u) + d/dz(A_v*du/dz)
    dv/dt = -(zeta+f)*u - dB/dy - (1/rho_0)*dp'/dy + A_h*lap(v) + d/dz(A_v*dv/dz)
    dT/dt = -u*dT/dx - v*dT/dy - w*dT/dz + K_h*lap(T) + d/dz(K_v*dT/dz)
    dS/dt = -u*dS/dx - v*dS/dy - w*dS/dz + K_h*lap(S) + d/dz(K_v*dS/dz)

Vertical velocity w is diagnosed from continuity.
Density from Wright (1997) EOS.
Pressure from hydrostatic balance.

Time stepping: split-explicit barotropic/baroclinic (see barotropic.py).
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    divergence_3d,
    hyperdiffusion_3d,
)
from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    upwind_vertical_gradient,
)
from legoesm.ocean.state import OceanState, OceanTendencies, OceanConfig
from legoesm.ocean.physics.mixing import laplacian_viscosity_3d, vertical_diffusion


def ocean_baroclinic_tendencies(
    state: OceanState,
    grid: CubedSphereGrid,
    z_coord: OceanZStarCoordinate,
    config: OceanConfig = OceanConfig(),
) -> OceanTendencies:
    """Compute 3D baroclinic tendencies (slow mode).

    Called once per RK3 stage. This is the ocean analog of
    compressible_euler_slow_tendencies() and hydrostatic_tendencies().

    Parameters
    ----------
    state : OceanState
        Current ocean state.
    grid : CubedSphereGrid
        Horizontal grid.
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    config : OceanConfig
        Model configuration.

    Returns
    -------
    OceanTendencies : Time derivatives for all prognostic variables.
    """
    u = state.u.data       # (6, n, n, nlev)
    v = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data    # (6, n, n)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data  # (6, n, n)
    mask_3d = mask[..., jnp.newaxis]  # (6, n, n, 1)

    g = config.g
    rho_0 = config.rho_0

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(eta, H_bathy, z_coord)      # (6, n, n)
    h_k = compute_layer_thickness(eta, H_bathy, z_coord)   # (6, n, n, nlev)

    # --- 2. Density from EOS ---
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta, z_coord.dz_ref, J, rho_0, g,
    )
    rho = wright_eos(T, S, p_hydro)
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient ---
    # p'(z) = integral_{z}^{0} rho' * g * dz'  (top-down cumsum)
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]  # (6, n, n, nlev)
    dp_layer = rho_prime * g * dz_actual
    # Pressure at top of each layer from cumulative sum
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    # Pressure at cell center
    p_prime = p_prime + 0.5 * dp_layer

    # Gradient on unmasked field to avoid coastline discontinuity;
    # masking applied to the result (land tendencies zeroed at step 12).
    dp_dx = gradient_x_3d(p_prime, grid)
    dp_dy = gradient_y_3d(p_prime, grid)

    # --- 4. Diagnose vertical velocity ---
    div_v = divergence_3d(u * mask_3d, v * mask_3d, grid)  # (6, n, n, nlev)
    w = _diagnose_w(div_v, h_k)  # (6, n, n, nlev+1)

    # --- 5. Vorticity (relative only; planetary f handled by barotropic) ---
    # The planetary Coriolis f must be applied at the barotropic substep
    # level (small rotations per substep, f*dt_s ~ 0.01 rad). Applying it
    # here at the baroclinic level (f*dt ~ 0.5 rad) creates large velocity
    # changes in one step that generate spurious gravity waves.
    zeta = vorticity_3d(u * mask_3d, v * mask_3d, grid)

    # --- 6. Bernoulli function (kinetic energy only for ocean) ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = gradient_x_3d(K, grid)
    dK_dy = gradient_y_3d(K, grid)

    # --- 7. Vector-invariant momentum (ζ only, not ζ+f) ---
    du_dt = zeta * v - dK_dx - dp_dx / rho_0
    dv_dt = -zeta * u - dK_dy - dp_dy / rho_0

    # --- 8. Vertical advection of u, v ---
    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

    # --- 9. Tracer advection (T, S) ---
    dT_dx = gradient_x_3d(T, grid)
    dT_dy = gradient_y_3d(T, grid)
    dT_dt = -(u * dT_dx + v * dT_dy)
    dT_dt = dT_dt + _vertical_advection_ocean(T, w, z_coord, J)

    dS_dx = gradient_x_3d(S, grid)
    dS_dy = gradient_y_3d(S, grid)
    dS_dt = -(u * dS_dx + v * dS_dy)
    dS_dt = dS_dt + _vertical_advection_ocean(S, w, z_coord, J)

    # --- 10. Mixing ---
    # Velocity: masked before Laplacian (no-slip BC, u=0 on land)
    # Tracers: UNmasked to avoid coastline discontinuity (T, S are smooth
    # across land/ocean boundaries; masking before Laplacian creates a
    # spurious jump that cascades into pressure gradient errors)
    if config.A_h > 0:
        du_dt = du_dt + laplacian_viscosity_3d(u * mask_3d, grid, config.A_h)
        dv_dt = dv_dt + laplacian_viscosity_3d(v * mask_3d, grid, config.A_h)
    if config.K_h > 0:
        dT_dt = dT_dt + laplacian_viscosity_3d(T, grid, config.K_h)
        dS_dt = dS_dt + laplacian_viscosity_3d(S, grid, config.K_h)
    if config.A_v > 0:
        du_dt = du_dt + vertical_diffusion(u, z_coord, J, config.A_v)
        dv_dt = dv_dt + vertical_diffusion(v, z_coord, J, config.A_v)
    if config.K_v > 0:
        dT_dt = dT_dt + vertical_diffusion(T, z_coord, J, config.K_v)
        dS_dt = dS_dt + vertical_diffusion(S, z_coord, J, config.K_v)

    # --- 11. Hyperdiffusion ---
    if config.hyperdiff_coeff > 0:
        du_dt = du_dt + hyperdiffusion_3d(u * mask_3d, grid, config.hyperdiff_coeff)
        dv_dt = dv_dt + hyperdiffusion_3d(v * mask_3d, grid, config.hyperdiff_coeff)
        dT_dt = dT_dt + hyperdiffusion_3d(T, grid, config.hyperdiff_coeff)
        dS_dt = dS_dt + hyperdiffusion_3d(S, grid, config.hyperdiff_coeff)

    # --- 12. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 13. Free-surface tendency ---
    # deta/dt = -sum_k div(h_k * v_k)
    flux_div = divergence_3d(u * h_k * mask_3d, v * h_k * mask_3d, grid)
    deta_dt = -jnp.sum(flux_div, axis=-1) * mask

    # --- Build tendency pytree ---
    dims_3d = ("face", "x", "y", "level")
    dims_2d = ("face", "x", "y")

    return OceanTendencies(
        du_dt=Field(data=du_dt, name="du_dt", dims=dims_3d, units="m/s^2"),
        dv_dt=Field(data=dv_dt, name="dv_dt", dims=dims_3d, units="m/s^2"),
        dT_dt=Field(data=dT_dt, name="dT_dt", dims=dims_3d, units="degC/s"),
        dS_dt=Field(data=dS_dt, name="dS_dt", dims=dims_3d, units="PSU/s"),
        deta_dt=Field(data=deta_dt, name="deta_dt", dims=dims_2d, units="m/s"),
        dH_bathy_dt=Field(
            data=jnp.zeros_like(H_bathy), name="dH_bathy_dt",
            dims=dims_2d, units="m/s",
        ),
        dland_mask_dt=Field(
            data=jnp.zeros_like(mask), name="dland_mask_dt",
            dims=dims_2d, units="1/s",
        ),
    )


def _diagnose_w(
    div_v: jnp.ndarray,
    h_k: jnp.ndarray,
) -> jnp.ndarray:
    """Diagnose vertical velocity from continuity.

    w(z) = -integral_{-H}^{z} div(v) dz'

    Integrated bottom-up. w=0 at ocean floor.

    Parameters
    ----------
    div_v : array
        Horizontal divergence, shape (..., nlev).
    h_k : array
        Layer thickness [m], shape (..., nlev).

    Returns
    -------
    array : Vertical velocity at interfaces [m/s], shape (..., nlev+1).
        w[..., 0] is at the surface, w[..., -1] = 0 at the bottom.
    """
    # Mass flux per layer: div(v) * h_k
    div_h = div_v * h_k  # (..., nlev)

    # Cumulative sum from bottom up: w at each interface
    # w[k-1/2] = -sum_{k'=k}^{nlev-1} div_h[k']
    div_h_rev = div_h[..., ::-1]
    cumsum_rev = jnp.cumsum(div_h_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]  # (..., nlev)

    # w at interfaces: w[0]=surface, w[nlev]=0 (bottom)
    zeros_bottom = jnp.zeros((*div_v.shape[:-1], 1))
    w = jnp.concatenate([w_inner, zeros_bottom], axis=-1)
    return w


def _vertical_advection_ocean(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Compute vertical advection -w * d(field)/dz with upwind scheme.

    Parameters
    ----------
    field : array
        Field at full levels, shape (..., nlev).
    w_half : array
        Vertical velocity at interfaces [m/s], shape (..., nlev+1).
    z_coord : OceanZStarCoordinate
        Vertical coordinate.
    jacobian : array
        Dynamic Jacobian, shape (...).

    Returns
    -------
    array : Vertical advection tendency, shape (..., nlev).
    """
    # w at full levels (average of interfaces)
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])

    # Convert w to z* coordinates: w* = w / J
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    w_star = w_full / jac_safe

    # Actual spacing between full levels
    dz_half = z_coord.dz_half_ref * jac_safe  # (..., nlev-1)
    grad = upwind_vertical_gradient(field, dz_half, w_star)
    return -w_star * grad
