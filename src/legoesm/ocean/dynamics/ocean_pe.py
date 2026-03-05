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

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.core.operators_3d import (
    vorticity_3d,
    gradient_x_3d,
    gradient_y_3d,
    divergence_3d,
    hyperdiffusion_3d,
    fv_flux_divergence_3d,
    fv_scalar_advection_3d,
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
    physics_fn=None,
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
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )      # (6, n, n)
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )   # (6, n, n, nlev)

    # --- 2. Density from EOS ---
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta_safe, z_coord.dz_ref, J, rho_0, g,
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

    # --- 4. Diagnose vertical velocity from flux divergence ---
    # The layer continuity equation is  dh_k/dt + div(v_k * h_k) + w_{k-1/2} - w_{k+1/2} = 0.
    # Vertical velocity must be diagnosed from div(v*h), NOT from div(v)*h,
    # because h_k varies horizontally and div(v*h) ≠ h*div(v).
    # This also makes w consistent with the free-surface tendency (step 13).
    # FV flux divergence returns -div(h*v), negate to get div(h*v).
    flux_div_k = -fv_flux_divergence_3d(
        h_k * mask_3d, u * mask_3d, v * mask_3d, grid, g=0.0,
    )  # (6, n, n, nlev)
    w = _diagnose_w_from_flux_div(flux_div_k)  # (6, n, n, nlev+1)

    # Velocity divergence (needed for skew-symmetric momentum, step 7).
    div_v = divergence_3d(u * mask_3d, v * mask_3d, grid)  # (6, n, n, nlev)

    # --- 5. Vorticity ---
    zeta = vorticity_3d(u * mask_3d, v * mask_3d, grid)

    # --- 6. Bernoulli function (kinetic energy only for ocean) ---
    K = 0.5 * (u**2 + v**2)
    dK_dx = gradient_x_3d(K, grid)
    dK_dy = gradient_y_3d(K, grid)

    # --- 7. Vector-invariant momentum (skew-symmetric / energy-preserving) ---
    # Planetary Coriolis is split: barotropic mode (depth-mean) is integrated
    # in substeps, while baroclinic shear (deviation from depth-mean) is
    # handled here to preserve full (zeta + f) dynamics without double counting.
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask
    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask
    u_prime = (u - U_bar[..., jnp.newaxis]) * mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * mask_3d
    f_3d = grid.f[..., jnp.newaxis]

    # Standard vector-invariant form is the advective form of momentum:
    #   du/dt = (ζ+f)v − ∂K/∂x
    # This conserves energy only when div(u)=0.  For divergent
    # (free-surface) flow, the skew-symmetric correction −½ u div(u)
    # restores the discrete energy identity  dK/dt = −div(K u):
    du_dt = (zeta * v + f_3d * v_prime - dK_dx
             - 0.5 * u * div_v - dp_dx / rho_0)
    dv_dt = (-zeta * u - f_3d * u_prime - dK_dy
             - 0.5 * v * div_v - dp_dy / rho_0)

    # --- 8. Vertical advection of u, v ---
    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

    # --- 9. Tracer tendencies (vectorized over T,S) ---
    # Avoid duplicated operator launches by treating tracers as a batch axis.
    tracers = jnp.stack([T, S], axis=0)  # (2, 6, n, n, nlev)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        # Upwind FV horizontal advection: conservative and monotonic
        # via MUSCL reconstruction with slope limiting.
        dtr_dt = fv_scalar_advection_3d(tr, u * mask_3d, v * mask_3d, grid)
        dtr_dt = dtr_dt + _vertical_advection_ocean(tr, w, z_coord, J)

        if physics_fn is None:
            if config.K_h > 0:
                dtr_dt = dtr_dt + laplacian_viscosity_3d(tr, grid, config.K_h)
            if config.K_v > 0:
                dtr_dt = dtr_dt + vertical_diffusion(tr, z_coord, J, config.K_v)
            if config.hyperdiff_coeff > 0:
                dtr_dt = dtr_dt + hyperdiffusion_3d(tr, grid, config.hyperdiff_coeff)

        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 10. Mixing ---
    if physics_fn is None:
        # Legacy hardcoded mixing
        # Velocity: masked before Laplacian (no-slip BC, u=0 on land)
        # Tracers: UNmasked to avoid coastline discontinuity (T, S are smooth
        # across land/ocean boundaries; masking before Laplacian creates a
        # spurious jump that cascades into pressure gradient errors)
        if config.A_h > 0:
            vel_masked = jnp.stack([u * mask_3d, v * mask_3d], axis=0)
            vel_lap = jax.vmap(
                lambda q: laplacian_viscosity_3d(q, grid, config.A_h),
                in_axes=0, out_axes=0,
            )(vel_masked)
            du_dt = du_dt + vel_lap[0]
            dv_dt = dv_dt + vel_lap[1]
        if config.A_v > 0:
            vel = jnp.stack([u, v], axis=0)
            vel_vdiff = jax.vmap(
                lambda q: vertical_diffusion(q, z_coord, J, config.A_v),
                in_axes=0, out_axes=0,
            )(vel)
            du_dt = du_dt + vel_vdiff[0]
            dv_dt = dv_dt + vel_vdiff[1]

        # --- 11. Hyperdiffusion ---
        if config.hyperdiff_coeff > 0:
            du_dt = du_dt + hyperdiffusion_3d(u * mask_3d, grid, config.hyperdiff_coeff)
            dv_dt = dv_dt + hyperdiffusion_3d(v * mask_3d, grid, config.hyperdiff_coeff)
    else:
        # New physics module system
        phys = physics_fn(state, grid, z_coord)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 12. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 13. Free-surface tendency ---
    # deta/dt = -sum_k div(h_k * v_k)
    # Reuse flux_div_k from step 4 to guarantee exact consistency
    # between the vertical velocity diagnosis and the surface tendency.
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

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


def _diagnose_w_from_flux_div(
    flux_div_k: jnp.ndarray,
) -> jnp.ndarray:
    """Diagnose vertical velocity from the flux divergence div(v*h_k).

    From the layer continuity equation:
        dh_k/dt + div(v_k * h_k) + w_{k-1/2} - w_{k+1/2} = 0

    Assuming w = 0 at the ocean floor and integrating bottom-up:
        w_{k-1/2} = -sum_{k'=k}^{nlev-1} div(v_k' * h_k')

    Using div(v*h) directly (rather than div(v)*h) ensures exact
    consistency with the free-surface tendency  dη/dt = -Σ div(v*h_k)
    and avoids the spurious source from the product rule on variable h.

    Parameters
    ----------
    flux_div_k : array
        Horizontal flux divergence div(v_k * h_k), shape (..., nlev).

    Returns
    -------
    array : Vertical velocity at interfaces [m/s], shape (..., nlev+1).
        w[..., 0] is at the surface, w[..., -1] = 0 at the bottom.
    """
    # Cumulative sum from bottom up: w at each interface
    fd_rev = flux_div_k[..., ::-1]
    cumsum_rev = jnp.cumsum(fd_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]  # (..., nlev)

    # w at interfaces: w[0]=surface, w[nlev]=0 (bottom)
    zeros_bottom = jnp.zeros(
        (*flux_div_k.shape[:-1], 1), dtype=flux_div_k.dtype,
    )
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

    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)

    # Actual spacing between full levels (physical z spacing)
    dz_half = z_coord.dz_half_ref * jac_safe  # (..., nlev-1)
    grad = upwind_vertical_gradient(field, dz_half, w_full)
    return -w_full * grad
