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

    # Barotropic diffusion — flux-form with face-centered coefficient.
    # Using div(nu_face * grad(eta)) instead of nu_cell * div(grad(eta))
    # ensures exact volume conservation (divergence theorem: sum of
    # div(F)*area = 0 for any flux F with no-flux BCs).
    # The cell-center form nu_cell * laplacian(eta) is non-conservative
    # when nu_cell varies spatially (area varies as cos(lat) on latlon).
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))

    # Precompute face-centered diffusion coefficients (grid geometry only,
    # constant across substeps).
    if config.barotropic_diffusion_alpha > 0.0:
        area = grid.area  # (n_lat, n_lon)
        # u-face coefficient: average of adjacent cell areas
        nu_face_u = baro_alpha * 0.5 * (jnp.roll(area, 1, axis=1) + area)
        nu_face_u = jnp.concatenate([nu_face_u, nu_face_u[:, 0:1]], axis=1)
        # v-face coefficient: average of adjacent cell areas
        nu_face_v_interior = baro_alpha * 0.5 * (area[:-1] + area[1:])
        zero_row_nu = jnp.zeros((1, area.shape[1]), dtype=eta.dtype)
        nu_face_v = jnp.concatenate(
            [zero_row_nu, nu_face_v_interior, zero_row_nu], axis=0,
        )
        # Face masks for land boundaries (zero flux at coastlines)
        diff_u_mask = mask * jnp.roll(mask, 1, axis=1)
        diff_u_mask = jnp.concatenate(
            [diff_u_mask, diff_u_mask[:, 0:1]], axis=1,
        )
        diff_v_mask_interior = mask[:-1] * mask[1:]
        zero_row_m = jnp.zeros((1, mask.shape[1]), dtype=mask.dtype)
        diff_v_mask = jnp.concatenate(
            [zero_row_m, diff_v_mask_interior, zero_row_m], axis=0,
        )

    # Accumulators for time-averaged barotropic transport (Phase 2a, issue #102).
    # These accumulate the mass fluxes H*U_bar at each substep so the tracer
    # equation can use transport consistent with the barotropic continuity.
    n_lat = eta.shape[0]
    n_lon = eta.shape[1]
    Hu_sum = jnp.zeros((n_lat, n_lon + 1), dtype=eta.dtype)
    Hv_sum = jnp.zeros((n_lat + 1, n_lon), dtype=eta.dtype)
    eta_sum = jnp.zeros((n_lat, n_lon), dtype=eta.dtype)
    U_sum = jnp.zeros((n_lat, n_lon + 1), dtype=eta.dtype)
    V_sum = jnp.zeros((n_lat + 1, n_lon), dtype=eta.dtype)

    def substep_body(i, carry):
        (eta_c, U_bar_c, V_bar_c,
         Hu_sum_c, Hv_sum_c, eta_sum_c, U_sum_c, V_sum_c) = carry

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

        # Accumulate transport for barotropic-averaged tracer advection
        Hu_sum_new = Hu_sum_c + flux_u.astype(eta.dtype)
        Hv_sum_new = Hv_sum_c + flux_v.astype(eta.dtype)

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

        # Forward-backward Coriolis (Matsuno stepping):
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

        # Optional Laplacian damping on eta (flux-form: conservative)
        if config.barotropic_diffusion_alpha > 0.0:
            grad_x = gradient_x_cgrid(eta_new * mask, grid)
            grad_y = gradient_y_cgrid(eta_new * mask, grid)
            flux_x = nu_face_u * grad_x * diff_u_mask
            flux_y = nu_face_v * grad_y * diff_v_mask
            eta_new = (
                eta_new + divergence_cgrid(flux_x, flux_y, grid).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask

        # Accumulate eta, U_bar, V_bar AFTER diffusion for time-averaging
        eta_sum_new = eta_sum_c + eta_new
        U_sum_new = U_sum_c + U_bar_new
        V_sum_new = V_sum_c + V_bar_new

        return (eta_new, U_bar_new, V_bar_new,
                Hu_sum_new, Hv_sum_new, eta_sum_new, U_sum_new, V_sum_new)

    init_carry = (eta, U_bar, V_bar, Hu_sum, Hv_sum, eta_sum, U_sum, V_sum)

    if config.differentiable_barotropic:
        def scan_body(carry, _):
            new_carry = substep_body(0, carry)
            return new_carry, None

        (eta_f, U_bar_f, V_bar_f,
         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f), _ = jax.lax.scan(
            scan_body, init_carry, xs=None, length=n_substeps,
        )
    else:
        (eta_f, U_bar_f, V_bar_f,
         Hu_sum_f, Hv_sum_f, eta_sum_f, U_sum_f, V_sum_f) = jax.lax.fori_loop(
            0, n_substeps, substep_body, init_carry,
        )

    # Time-averaged barotropic transport
    Hu_avg = Hu_sum_f / n_substeps
    Hv_avg = Hv_sum_f / n_substeps

    # Time-averaged eta and barotropic velocity for baroclinic coupling
    eta_avg = eta_sum_f / n_substeps
    U_bar_avg = U_sum_f / n_substeps
    V_bar_avg = V_sum_f / n_substeps

    # Correct 3D velocities: preserve baroclinic structure.
    # Use time-averaged barotropic velocity for the 3D correction to ensure
    # consistency with eta_avg (the time-averaged eta used for layer thicknesses).
    u_baro_old = U_bar[..., jnp.newaxis]
    v_baro_old = V_bar[..., jnp.newaxis]
    u_prime = u - u_baro_old
    v_prime = v - v_baro_old
    u_new = (u_prime + U_bar_avg[..., jnp.newaxis]) * u_mask[..., jnp.newaxis]
    v_new = (v_prime + V_bar_avg[..., jnp.newaxis]) * v_mask[..., jnp.newaxis]

    state_new = state._replace(
        eta=state.eta.replace(data=eta_avg),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
    return state_new, (Hu_avg, Hv_avg)
