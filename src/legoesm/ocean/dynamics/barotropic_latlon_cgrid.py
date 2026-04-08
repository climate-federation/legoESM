"""Barotropic solver for the lat-lon C-grid FV ocean model.

Forward-backward substeps for 2D free-surface gravity waves on the
Arakawa C-grid:

    d(eta)/dt = -div(H_total * U_bar, H_total * V_bar)   [cell centers]
    d(U_bar)/dt = f * V_bar_at_u - g * d(eta)/dx          [u-points]
    d(V_bar)/dt = -f * U_bar_at_v - g * d(eta)/dy         [v-points]

The C-grid layout uses compact (single-cell) gradient and divergence
stencils, eliminating the 2*dx checkerboard null space of the A-grid.

Parallels barotropic_latlon.py (A-grid) but with staggered variables.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonCGridOceanState, LatLonCGridOceanConfig
from legoesm.ocean.dynamics.latlon_cgrid_operators import (
    divergence_cgrid,
    gradient_x_cgrid,
    gradient_y_cgrid,
    coriolis_cgrid,
    laplacian_cgrid,
)


def _depth_average_to_faces(
    u_3d: jnp.ndarray,
    v_3d: jnp.ndarray,
    h_k: jnp.ndarray,
    min_water_col: jnp.ndarray,
    mask: jnp.ndarray,
    u_mask: jnp.ndarray,
    v_mask: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute depth-averaged velocities at C-grid face points.

    Parameters
    ----------
    u_3d : (n_lat, n_lon+1, nlev)
    v_3d : (n_lat+1, n_lon, nlev)
    h_k : (n_lat, n_lon, nlev) layer thickness at cell centers.
    min_water_col : scalar
    mask : (n_lat, n_lon)
    u_mask : (n_lat, n_lon+1)
    v_mask : (n_lat+1, n_lon)

    Returns
    -------
    U_bar : (n_lat, n_lon+1)
    V_bar : (n_lat+1, n_lon)
    """
    # h at u-faces: average of adjacent cells flanking face j
    # Face j is between cell (j-1) mod n_lon and cell j
    h_east = h_k
    h_west = jnp.roll(h_k, 1, axis=1)
    h_u = 0.5 * (h_west + h_east)  # (n_lat, n_lon, nlev)
    h_u = jnp.concatenate([h_u, h_u[:, 0:1, :]], axis=1)  # (n_lat, n_lon+1, nlev)

    H_u = jnp.maximum(jnp.sum(h_u, axis=-1), min_water_col)
    U_bar = jnp.sum(u_3d * h_u, axis=-1) / H_u * u_mask

    # h at v-faces: average of adjacent cell h
    h_south = h_k[:-1]
    h_north = h_k[1:]
    h_v_interior = 0.5 * (h_south + h_north)  # (n_lat-1, n_lon, nlev)
    n_lon = h_k.shape[1]
    nlev = h_k.shape[2]
    zero_row = jnp.zeros((1, n_lon, nlev), dtype=h_k.dtype)
    h_v = jnp.concatenate([zero_row, h_v_interior, zero_row], axis=0)

    H_v = jnp.maximum(jnp.sum(h_v, axis=-1), min_water_col)
    V_bar = jnp.sum(v_3d * h_v, axis=-1) / H_v * v_mask

    return U_bar, V_bar


def barotropic_substeps_latlon_cgrid(
    state: LatLonCGridOceanState,
    dt_s: float,
    n_substeps: int,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonCGridOceanConfig,
) -> LatLonCGridOceanState:
    """Run barotropic substeps on a C-grid lat-lon grid.

    Parameters
    ----------
    state : LatLonCGridOceanState
        State after slow tendency application.
    dt_s : float
        Substep size [seconds].
    n_substeps : int
    grid : LatLonGrid
    z_coord : OceanZStarCoordinate
    config : LatLonCGridOceanConfig

    Returns
    -------
    LatLonCGridOceanState with updated eta and velocity.
    """
    g = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u_mask = state.u_mask.data
    v_mask = state.v_mask.data
    u = state.u.data
    v = state.v.data
    eta_raw = state.eta.data
    min_water_col = jnp.asarray(config.min_water_column_m, dtype=eta_raw.dtype)
    dt_s = jnp.asarray(dt_s, dtype=eta_raw.dtype)
    g = g.astype(eta_raw.dtype)
    eta_floor = min_water_col - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    # Depth-averaged velocity
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    U_bar, V_bar = _depth_average_to_faces(
        u, v, h_k, min_water_col, mask, u_mask, v_mask,
    )

    # Semi-implicit Coriolis parameter at face points
    f_cell = grid.f.astype(eta.dtype)
    # f at u-points: face j is between cell (j-1) mod n_lon and cell j
    f_u = 0.5 * (jnp.roll(f_cell, 1, axis=1) + f_cell)
    f_u = jnp.concatenate([f_u, f_u[:, 0:1]], axis=1)
    # f at v-points
    f_v_interior = 0.5 * (f_cell[:-1] + f_cell[1:])
    f_v = jnp.concatenate([f_cell[0:1], f_v_interior, f_cell[-1:]], axis=0)

    alpha_u = (0.5 * f_u * dt_s).astype(eta.dtype)
    alpha_v = (0.5 * f_v * dt_s).astype(eta.dtype)

    # Barotropic diffusion
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))

    def substep_body(i, carry):
        eta_c, U_bar_c, V_bar_c = carry

        H_total_c = jnp.maximum(eta_c + H_bathy, min_water_col) * mask

        # Forward: update eta from continuity (C-grid divergence)
        # Need U_bar * H_total at u-points. H_total is cell-centered,
        # interpolate to faces.
        H_u = 0.5 * (jnp.roll(H_total_c, 1, axis=1) + H_total_c)
        H_u = jnp.concatenate([H_u, H_u[:, 0:1]], axis=1)
        H_v_interior = 0.5 * (H_total_c[:-1] + H_total_c[1:])
        n_lon_loc = H_total_c.shape[1]
        zero_row = jnp.zeros((1, n_lon_loc), dtype=eta.dtype)
        H_v = jnp.concatenate([zero_row, H_v_interior, zero_row], axis=0)

        flux_u = H_u * U_bar_c * u_mask
        flux_v = H_v * V_bar_c * v_mask

        div_flux = divergence_cgrid(
            flux_u, flux_v, grid, u_mask=u_mask, v_mask=v_mask,
        ).astype(eta.dtype)
        eta_new = jnp.maximum(eta_c - dt_s * div_flux, eta_floor) * mask

        # Backward: update velocity with UPDATED eta (compact gradient)
        deta_dx = gradient_x_cgrid(eta_new, grid).astype(eta.dtype)
        deta_dy = gradient_y_cgrid(eta_new, grid).astype(eta.dtype)

        # Average V to u-points for Coriolis
        # Face j is between cell (j-1) and cell j; use v-points at j-1 and j
        V_west = jnp.roll(V_bar_c, 1, axis=1)
        V_at_u = 0.25 * (V_bar_c[:-1] + V_bar_c[1:] + V_west[:-1] + V_west[1:])
        V_at_u = jnp.concatenate([V_at_u, V_at_u[:, 0:1]], axis=1)

        # Average U to v-points for Coriolis
        U_at_v_interior = 0.25 * (
            U_bar_c[:-1, :-1] + U_bar_c[:-1, 1:]
            + U_bar_c[1:, :-1] + U_bar_c[1:, 1:]
        )
        zero_row_u = jnp.zeros((1, n_lon_loc), dtype=eta.dtype)
        U_at_v = jnp.concatenate([zero_row_u, U_at_v_interior, zero_row_u], axis=0)

        # Semi-implicit Coriolis at u-points
        denom_u = 1.0 + alpha_u ** 2
        rhs_u = U_bar_c + alpha_u * V_at_u - dt_s * g * deta_dx
        rhs_v_corr = V_at_u - alpha_u * U_bar_c
        U_bar_new = (rhs_u + alpha_u * rhs_v_corr) / denom_u * u_mask

        # Semi-implicit Coriolis at v-points
        denom_v = 1.0 + alpha_v ** 2
        rhs_v = V_bar_c - alpha_v * U_at_v - dt_s * g * deta_dy
        rhs_u_corr = U_at_v + alpha_v * V_bar_c
        V_bar_new = (rhs_v - alpha_v * (-rhs_u_corr)) / denom_v * v_mask
        # Simplified: V_bar_new = (rhs_v + alpha_v * rhs_u_corr) / denom_v
        # but with correct sign convention for southern hemisphere
        V_bar_new = (V_bar_c - alpha_v * U_at_v - dt_s * g * deta_dy
                     + alpha_v * (U_at_v + alpha_v * V_bar_c)) / denom_v * v_mask
        # Expand:  (V_bar_c*(1+alpha_v^2) - alpha_v*U_at_v + alpha_v*U_at_v
        #            - dt_s*g*deta_dy) / denom_v
        # Hmm, that cancels. Let me redo the semi-implicit properly.

        # Proper semi-implicit Coriolis (at each face independently):
        # At u-point:
        #   U* = U + dt*(f*V_at_u - g*deta/dx)
        #   U_new = (U* + alpha_u * V_at_u_*) / (1 + alpha_u^2)
        # But V_at_u_* is not known yet. Standard approach: predict-correct.
        # Simpler: use the forward-backward pattern where U updates first,
        # then V uses the new U.

        # Forward-backward with Coriolis:
        # Step 1: U_new = U + dt * (f_u * V_at_u - g * deta/dx)
        U_bar_new = (U_bar_c + dt_s * (f_u * V_at_u - g * deta_dx)) * u_mask

        # Step 2: V_new = V + dt * (-f_v * U_at_v_new - g * deta/dy)
        # Use updated U for averaging
        U_new_at_v_interior = 0.25 * (
            U_bar_new[:-1, :-1] + U_bar_new[:-1, 1:]
            + U_bar_new[1:, :-1] + U_bar_new[1:, 1:]
        )
        U_new_at_v = jnp.concatenate(
            [zero_row_u, U_new_at_v_interior, zero_row_u], axis=0,
        )
        V_bar_new = (V_bar_c + dt_s * (-f_v * U_new_at_v - g * deta_dy)) * v_mask

        # Optional Laplacian damping on eta
        if config.barotropic_diffusion_alpha > 0.0:
            nu_dt = baro_alpha * grid.area
            eta_new = (
                eta_new + nu_dt * laplacian_cgrid(eta_new, grid, mask=mask).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask

        return (eta_new, U_bar_new, V_bar_new)

    if config.differentiable_barotropic:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f), _ = jax.lax.scan(
            scan_body, (eta, U_bar, V_bar), xs=None, length=n_substeps,
        )
    else:
        eta_f, U_bar_f, V_bar_f = jax.lax.fori_loop(
            0, n_substeps, substep_body, (eta, U_bar, V_bar),
        )

    # Correct 3D velocities: preserve baroclinic structure
    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + U_bar_f[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_f[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
