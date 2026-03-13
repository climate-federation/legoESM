"""Boussinesq Hydrostatic Primitive Equations on the lat-lon grid (FV).

Finite-volume formulation using PPM transport for tracers and
conservative flux divergence for the free-surface equation.

Vector-invariant momentum form with skew-symmetric correction for
energy conservation under divergent (free-surface) flow.

Boundary conditions:
- Longitude: periodic
- Latitude: solid wall at poles (v=0, no normal flux)

References
----------
- Griffies (2004): Fundamentals of Ocean Climate Models (MOM framework)
- Lin (2004): A "Vertically Lagrangian" FV Dynamical Core
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.field import Field
from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.eos import wright_eos, compute_hydrostatic_pressure
from legoesm.ocean.vertical import (
    OceanZStarCoordinate,
    compute_layer_thickness,
    compute_ocean_jacobian,
    upwind_vertical_gradient,
)
from legoesm.ocean.state import (
    LatLonOceanState,
    LatLonOceanTendencies,
    LatLonOceanConfig,
)
from legoesm.ocean.dynamics.latlon_operators import (
    fv_divergence_latlon_3d,
    fv_scalar_advection_latlon_3d,
    gradient_x_latlon,
    gradient_y_latlon,
    vorticity_latlon,
    laplacian_latlon,
)


def _diagnose_w_from_flux_div(
    flux_div_k: jnp.ndarray,
    z_coord: OceanZStarCoordinate | None = None,
) -> jnp.ndarray:
    """Diagnose z-star transport velocity from flux divergence (bottom-up cumsum).

    Parameters
    ----------
    flux_div_k : array, shape (..., nlev)
    z_coord : OceanZStarCoordinate or None
        When provided, applies the z-star correction so that ẇ[0] = 0
        at the surface and ẇ[nlev] = 0 at the bottom.

    Returns
    -------
    w : array, shape (..., nlev+1)
        w[..., 0] at surface, w[..., -1] = 0 at bottom.
    """
    fd_rev = flux_div_k[..., ::-1]
    cumsum_rev = jnp.cumsum(fd_rev, axis=-1)
    w_inner = -cumsum_rev[..., ::-1]
    zeros_bottom = jnp.zeros((*flux_div_k.shape[:-1], 1), dtype=flux_div_k.dtype)
    w_euler = jnp.concatenate([w_inner, zeros_bottom], axis=-1)

    if z_coord is None:
        return w_euler

    sigma = (z_coord.z_half_ref + z_coord.H_max) / z_coord.H_max
    deta_dt = w_euler[..., 0:1]
    return w_euler - sigma * deta_dt


def _vertical_advection_ocean(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Vertical advection -w * d(field)/dz with upwind scheme (momentum).

    Parameters
    ----------
    field : array, shape (..., nlev)
    w_half : array, shape (..., nlev+1)
    z_coord : OceanZStarCoordinate
    jacobian : array, shape (...)

    Returns
    -------
    tendency : array, shape (..., nlev)
    """
    w_full = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    dz_half = z_coord.dz_half_ref * jac_safe
    grad = upwind_vertical_gradient(field, dz_half, w_full)
    return -w_full * grad


def _flux_form_vertical_advection_tracer(
    field: jnp.ndarray,
    w_half: jnp.ndarray,
    z_coord: OceanZStarCoordinate,
    jacobian: jnp.ndarray,
) -> jnp.ndarray:
    """Conservative flux-form vertical advection for tracers (lat-lon).

    See ocean_pe._flux_form_vertical_advection_tracer for full docstring.
    """
    T_above = field[..., :-1]
    T_below = field[..., 1:]
    w_interior = w_half[..., 1:-1]

    T_at_interface = jnp.where(w_interior > 0, T_below, T_above)
    F_interior = w_interior * T_at_interface

    zeros = jnp.zeros((*field.shape[:-1], 1), dtype=field.dtype)
    flux = jnp.concatenate([zeros, F_interior, zeros], axis=-1)

    jac_safe = jnp.maximum(jacobian[..., jnp.newaxis], 1.0e-10)
    h_k = z_coord.dz_ref * jac_safe

    return (flux[..., 1:] - flux[..., :-1]) / h_k


def latlon_ocean_baroclinic_tendencies(
    state: LatLonOceanState,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonOceanConfig = LatLonOceanConfig(),
    physics_fn=None,
) -> LatLonOceanTendencies:
    """Compute 3D baroclinic tendencies on a lat-lon grid.

    Parameters
    ----------
    state : LatLonOceanState
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonOceanConfig
    physics_fn : callable, optional

    Returns
    -------
    LatLonOceanTendencies
    """
    u = state.u.data       # (n_lat, n_lon, nlev)
    v = state.v.data
    T = state.T.data
    S = state.S.data
    eta = state.eta.data    # (n_lat, n_lon)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    mask_3d = mask[..., jnp.newaxis]

    g = config.g
    rho_0 = config.rho_0
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta.dtype)
    eta_floor = min_water_col - H_bathy
    eta_safe = jnp.maximum(eta, eta_floor) * mask

    # --- 1. Layer thickness and Jacobian ---
    J = compute_ocean_jacobian(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    h_k = compute_layer_thickness(
        eta_safe, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )

    # --- 2. Density from EOS ---
    p_hydro = compute_hydrostatic_pressure(
        jnp.full_like(T, rho_0), eta_safe, z_coord.dz_ref, J, rho_0, g,
    )
    rho = wright_eos(T, S, p_hydro)
    rho_prime = rho - rho_0

    # --- 3. Baroclinic pressure gradient ---
    dz_actual = z_coord.dz_ref * J[..., jnp.newaxis]
    dp_layer = rho_prime * g * dz_actual
    p_prime = jnp.cumsum(dp_layer, axis=-1) - dp_layer
    p_prime = p_prime + 0.5 * dp_layer

    dp_dx = gradient_x_latlon(p_prime, grid)
    dp_dy = gradient_y_latlon(p_prime, grid)

    # --- 4. Vertical velocity from FV flux divergence ---
    flux_div_k = fv_divergence_latlon_3d(
        h_k * u * mask_3d, h_k * v * mask_3d, grid,
    )
    w = _diagnose_w_from_flux_div(flux_div_k, z_coord)

    # Velocity divergence for skew-symmetric momentum
    div_v = fv_divergence_latlon_3d(u * mask_3d, v * mask_3d, grid)

    # --- 5. Vorticity ---
    zeta = vorticity_latlon(u * mask_3d, v * mask_3d, grid)

    # --- 6. Bernoulli function ---
    K = 0.5 * (u ** 2 + v ** 2)
    dK_dx = gradient_x_latlon(K, grid)
    dK_dy = gradient_y_latlon(K, grid)

    # --- 7. Vector-invariant momentum (skew-symmetric) ---
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_water_col)
    U_bar = jnp.sum(u * h_k, axis=-1) / H_total * mask
    V_bar = jnp.sum(v * h_k, axis=-1) / H_total * mask
    u_prime = (u - U_bar[..., jnp.newaxis]) * mask_3d
    v_prime = (v - V_bar[..., jnp.newaxis]) * mask_3d
    f_3d = grid.f[..., jnp.newaxis]

    du_dt = (zeta * v + f_3d * v_prime - dK_dx
             - 0.5 * u * div_v - dp_dx / rho_0)
    dv_dt = (-zeta * u - f_3d * u_prime - dK_dy
             - 0.5 * v * div_v - dp_dy / rho_0)

    # --- 8. Vertical advection of u, v ---
    du_dt = du_dt + _vertical_advection_ocean(u, w, z_coord, J)
    dv_dt = dv_dt + _vertical_advection_ocean(v, w, z_coord, J)

    # --- 9. Tracer tendencies (PPM horizontal + vertical) ---
    tracers = jnp.stack([T, S], axis=0)

    def tracer_tendency(tr: jnp.ndarray) -> jnp.ndarray:
        dtr_dt = fv_scalar_advection_latlon_3d(tr, u * mask_3d, v * mask_3d, grid)
        dtr_dt = dtr_dt + _vertical_advection_ocean(tr, w, z_coord, J)

        if physics_fn is None:
            if config.K_h > 0:
                dtr_dt = dtr_dt + config.K_h * laplacian_latlon(tr, grid)
            if config.K_v > 0:
                # J is horizontal-only (n_lat, n_lon) — broadcasts with
                # vertical arrays via the trailing newaxis.
                jac_v = jnp.maximum(J[..., jnp.newaxis], 1e-10)
                dz_actual_loc = z_coord.dz_ref * jac_v       # (..., nlev)
                # Vertical diffusion: d/dz(K_v * d(tr)/dz)
                dtr_dz_half = jnp.diff(tr, axis=-1) / (
                    z_coord.dz_half_ref * jac_v               # broadcasts to (..., nlev-1)
                )
                flux = config.K_v * dtr_dz_half
                zeros_face = jnp.zeros(
                    (*tr.shape[:-1], 1), dtype=tr.dtype,
                )
                flux_full = jnp.concatenate([zeros_face, flux, zeros_face], axis=-1)
                dtr_dt = dtr_dt + (
                    flux_full[..., :-1] - flux_full[..., 1:]
                ) / dz_actual_loc

        return dtr_dt

    tracer_tend = jax.vmap(tracer_tendency, in_axes=0, out_axes=0)(tracers)
    dT_dt = tracer_tend[0]
    dS_dt = tracer_tend[1]

    # --- 10. Mixing ---
    if physics_fn is None:
        if config.A_h > 0:
            du_dt = du_dt + config.A_h * laplacian_latlon(u * mask_3d, grid)
            dv_dt = dv_dt + config.A_h * laplacian_latlon(v * mask_3d, grid)
        if config.A_v > 0:
            jac_v = jnp.maximum(J[..., jnp.newaxis], 1e-10)
            for vel, dvel_dt_ref in [(u, "u"), (v, "v")]:
                dv_dz_half = jnp.diff(vel, axis=-1) / (
                    z_coord.dz_half_ref * jac_v
                )
                flux = config.A_v * dv_dz_half
                zeros_face = jnp.zeros((*vel.shape[:-1], 1), dtype=vel.dtype)
                flux_full = jnp.concatenate([zeros_face, flux, zeros_face], axis=-1)
                vdiff = (flux_full[..., :-1] - flux_full[..., 1:]) / (
                    z_coord.dz_ref * jac_v
                )
                if dvel_dt_ref == "u":
                    du_dt = du_dt + vdiff
                else:
                    dv_dt = dv_dt + vdiff

        if config.hyperdiff_coeff > 0:
            du_dt = du_dt - config.hyperdiff_coeff * laplacian_latlon(
                laplacian_latlon(u * mask_3d, grid), grid,
            )
            dv_dt = dv_dt - config.hyperdiff_coeff * laplacian_latlon(
                laplacian_latlon(v * mask_3d, grid), grid,
            )
    else:
        phys = physics_fn(state, grid, z_coord)
        du_dt = du_dt + phys.du_dt.data
        dv_dt = dv_dt + phys.dv_dt.data
        dT_dt = dT_dt + phys.dT_dt.data
        dS_dt = dS_dt + phys.dS_dt.data

    # --- 11. Land masking ---
    du_dt = du_dt * mask_3d
    dv_dt = dv_dt * mask_3d
    dT_dt = dT_dt * mask_3d
    dS_dt = dS_dt * mask_3d

    # --- 12. Free-surface tendency ---
    deta_dt = -jnp.sum(flux_div_k, axis=-1) * mask

    dims_3d = ("lat", "lon", "level")
    dims_2d = ("lat", "lon")

    return LatLonOceanTendencies(
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
