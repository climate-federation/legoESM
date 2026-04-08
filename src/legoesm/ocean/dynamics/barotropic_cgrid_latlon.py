"""C-grid barotropic solver for the lat-lon FV ocean model.

Uses Arakawa C-grid staggering internally: u at east faces, v at north
faces, eta at cell centers.  Compact 1-cell gradient and divergence
operators eliminate the 2-delta-x computational mode that plagues the
A-grid solver (see issues #58, #87).

Forward-backward substeps for 2D free-surface gravity waves:

    d(eta)/dt  = -div(H * U_east, H * V_north)       [compact FV]
    d(U_east)/dt = +f * V_avg - g * d(eta)/dx_east   [compact grad]
    d(V_north)/dt = -f * U_avg - g * d(eta)/dy_north  [compact grad]

The solver takes cell-center depth-averaged velocity on entry, projects
to face locations, runs substeps on the staggered grid, and projects
back to cell centers on exit.  The baroclinic velocity structure is
preserved via the standard reconciliation: u_new = (u - U_bar_old) + U_bar_new.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.ocean.vertical import OceanZStarCoordinate, compute_layer_thickness
from legoesm.ocean.state import LatLonOceanState, LatLonOceanConfig
from legoesm.ocean.dynamics.latlon_operators import (
    laplacian_latlon,
    _neumann_fill_latlon,
)


# -----------------------------------------------------------------------
# C-grid operators (compact 1-cell stencils)
# -----------------------------------------------------------------------

def _cgrid_divergence(u_east, v_north, grid, mask):
    """Compact C-grid divergence at cell centers.

    u_east:  (n_lat, n_lon) — zonal velocity at east faces.
    v_north: (n_lat+1, n_lon) — meridional velocity at north faces.
    Returns: (n_lat, n_lon) — divergence at cell centers.
    """
    R = grid.radius
    dlat = grid.dlat
    dy_face = R * dlat  # length of east face (constant)

    # Zonal: (u_east[j] - u_west[j]) * dy_face / area
    # u_west[j] = u_east[j-1] (periodic)
    net_zonal = (u_east - jnp.roll(u_east, 1, axis=1)) * dy_face

    # Meridional: (v_north[i+1]*dx_v[i+1] - v_north[i]*dx_v[i]) / area
    # dx_v = R * dlon * cos(lat_v)
    dx_v = grid.dx_v  # (n_lat+1,)
    v_dx = v_north * dx_v[:, None]  # (n_lat+1, n_lon)
    net_merid = v_dx[1:] - v_dx[:-1]  # (n_lat, n_lon)

    return (net_zonal + net_merid) / grid.area * mask


def _cgrid_gradient_x(f, grid):
    """Compact pressure gradient at east faces (u-points).

    f: (n_lat, n_lon) — scalar at cell centers.
    Returns: (n_lat, n_lon) — df/dx at east faces.
    """
    # (f[j+1] - f[j]) / (R * dlon * cos_lat)
    # dx_cell = R * dlon * cos_lat, shape (n_lat,)
    df = jnp.roll(f, -1, axis=1) - f
    return df / grid.dx_cell[:, None]


def _cgrid_gradient_y(f, grid):
    """Compact pressure gradient at north faces (v-points).

    f: (n_lat, n_lon) — scalar at cell centers.
    Returns: (n_lat+1, n_lon) — df/dy at north faces.

    Zero at the pole faces (solid wall: no normal pressure gradient).
    """
    dy = grid.dy_cell  # R * dlat
    df_inner = (f[1:] - f[:-1]) / dy  # (n_lat-1, n_lon)
    n_lon = f.shape[1]
    zeros = jnp.zeros((1, n_lon), dtype=f.dtype)
    return jnp.concatenate([zeros, df_inner, zeros], axis=0)


# -----------------------------------------------------------------------
# Public entry point
# -----------------------------------------------------------------------

def barotropic_cgrid_substeps_latlon(
    state: LatLonOceanState,
    dt_s: float,
    n_substeps: int,
    grid: LatLonGrid,
    z_coord: OceanZStarCoordinate,
    config: LatLonOceanConfig,
) -> LatLonOceanState:
    """Run barotropic substeps with C-grid staggered velocity.

    Parameters
    ----------
    state : LatLonOceanState
        State after slow tendency application (A-grid velocity at cell centers).
    dt_s : float
        Substep size [seconds].
    n_substeps : int
        Number of barotropic substeps.
    grid : LatLonGrid
        Must have C-grid metrics populated (lat_v, cos_lat_v, etc.).
    z_coord : OceanZStarCoordinate
    config : LatLonOceanConfig

    Returns
    -------
    LatLonOceanState with updated eta and velocity (back at cell centers).
    """
    g_val = jnp.asarray(config.g)
    H_bathy = state.H_bathy.data
    mask = state.land_mask.data
    u = state.u.data          # (n_lat, n_lon, nlev) at cell centers
    v = state.v.data          # (n_lat, n_lon, nlev) at cell centers
    eta_raw = state.eta.data  # (n_lat, n_lon)
    min_wc = jnp.asarray(config.min_water_column_m, dtype=eta_raw.dtype)
    dt_s = jnp.asarray(dt_s, dtype=eta_raw.dtype)
    g_val = g_val.astype(eta_raw.dtype)
    eta_floor = min_wc - H_bathy
    eta = jnp.maximum(eta_raw, eta_floor) * mask

    n_lat = grid.n_lat
    n_lon = grid.n_lon

    # --- Face masks ---
    # u-face active if both flanking cells are ocean
    u_mask = mask * jnp.roll(mask, -1, axis=1)
    # v-face active if both flanking cells are ocean, zero at poles
    zeros_row = jnp.zeros((1, n_lon), dtype=mask.dtype)
    v_mask = jnp.concatenate([zeros_row, mask[:-1] * mask[1:], zeros_row], axis=0)

    # --- Depth-averaged velocity at cell centers ---
    h_k = compute_layer_thickness(
        eta, H_bathy, z_coord, min_water_column_m=config.min_water_column_m,
    )
    H_total = jnp.maximum(jnp.sum(h_k, axis=-1), min_wc)
    U_bar_cc = jnp.sum(u * h_k, axis=-1) / H_total * mask  # cell center
    V_bar_cc = jnp.sum(v * h_k, axis=-1) / H_total * mask  # cell center

    # --- Project to faces ---
    U_east = 0.5 * (U_bar_cc + jnp.roll(U_bar_cc, -1, axis=1)) * u_mask
    V_north_inner = 0.5 * (V_bar_cc[:-1] + V_bar_cc[1:]) * (mask[:-1] * mask[1:])
    V_north = jnp.concatenate([zeros_row, V_north_inner, zeros_row], axis=0)

    # --- Coriolis parameter at faces ---
    f_u = grid.f.astype(eta.dtype)  # (n_lat, n_lon), same lat as u-face
    f_v = grid.f_v.astype(eta.dtype)  # (n_lat+1, n_lon)

    # --- Barotropic diffusion on eta ---
    baro_alpha = jnp.asarray(
        config.barotropic_diffusion_alpha, dtype=eta.dtype,
    ) * (dt_s / jnp.asarray(config.barotropic_diffusion_dt_ref, dtype=eta.dtype))
    nu_dt = (baro_alpha * grid.area).astype(eta.dtype)

    def substep_body(i, carry):
        eta_c, U_e, V_n = carry

        # --- Total water column at faces ---
        H_c = jnp.maximum(eta_c + H_bathy, min_wc) * mask
        H_u = 0.5 * (H_c + jnp.roll(H_c, -1, axis=1)) * u_mask
        H_v_inner = 0.5 * (H_c[:-1] + H_c[1:]) * (mask[:-1] * mask[1:])
        H_v = jnp.concatenate([zeros_row, H_v_inner, zeros_row], axis=0)

        # --- Forward: update eta from compact C-grid divergence ---
        div_flux = _cgrid_divergence(H_u * U_e, H_v * V_n, grid, mask)
        eta_new = jnp.maximum(eta_c - dt_s * div_flux, eta_floor) * mask

        # --- Backward: compact pressure gradient at faces ---
        deta_dx = _cgrid_gradient_x(eta_new, grid).astype(eta.dtype)
        deta_dy = _cgrid_gradient_y(eta_new, grid).astype(eta.dtype)

        # --- Coriolis: cross-average velocity to opposite face ---
        # V at u-points: average 4 surrounding v-values
        V_at_u = 0.25 * (
            V_n[:-1, :] + V_n[1:, :]
            + jnp.roll(V_n[:-1, :], -1, axis=1)
            + jnp.roll(V_n[1:, :], -1, axis=1)
        )
        # U at v-points: average 4 surrounding u-values
        U_at_v_inner = 0.25 * (
            U_e[:-1, :] + U_e[1:, :]
            + jnp.roll(U_e[:-1, :], 1, axis=1)
            + jnp.roll(U_e[1:, :], 1, axis=1)
        )
        U_at_v = jnp.concatenate([zeros_row, U_at_v_inner, zeros_row], axis=0)

        # --- Semi-implicit Coriolis + backward PGF ---
        alpha_u = (0.5 * f_u * dt_s).astype(eta.dtype)
        denom_u = 1.0 + alpha_u ** 2
        rhs_u = U_e + alpha_u * V_at_u - dt_s * g_val * deta_dx
        rhs_vu = V_at_u - alpha_u * U_e
        U_e_new = (rhs_u + alpha_u * rhs_vu) / denom_u * u_mask

        alpha_v = (0.5 * f_v * dt_s).astype(eta.dtype)
        denom_v = 1.0 + alpha_v ** 2
        rhs_v = V_n - alpha_v * U_at_v - dt_s * g_val * deta_dy
        rhs_uv = U_at_v + alpha_v * V_n
        V_n_new = (rhs_v + alpha_v * rhs_uv) / denom_v * v_mask

        # --- Optional Laplacian damping on eta ---
        if config.barotropic_diffusion_alpha > 0.0:
            eta_new = (
                eta_new + nu_dt * laplacian_latlon(
                    eta_new, grid, mask=mask,
                ).astype(eta.dtype)
            ) * mask
            eta_new = jnp.maximum(eta_new, eta_floor) * mask

        return (eta_new, U_e_new, V_n_new)

    if config.differentiable_barotropic:
        def scan_body(carry, _):
            return substep_body(0, carry), None
        (eta_f, U_east_f, V_north_f), _ = jax.lax.scan(
            scan_body, (eta, U_east, V_north), xs=None, length=n_substeps,
        )
    else:
        eta_f, U_east_f, V_north_f = jax.lax.fori_loop(
            0, n_substeps, substep_body, (eta, U_east, V_north),
        )

    # --- Project face velocities back to cell centers ---
    U_bar_f = 0.5 * (U_east_f + jnp.roll(U_east_f, 1, axis=1)) * mask
    V_bar_f = 0.5 * (V_north_f[:-1] + V_north_f[1:]) * mask

    # --- Correct 3D velocities: preserve baroclinic structure ---
    u_baro_prime = u - U_bar_cc[..., jnp.newaxis]
    v_baro_prime = v - V_bar_cc[..., jnp.newaxis]
    u_new = (u_baro_prime + U_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]
    v_new = (v_baro_prime + V_bar_f[..., jnp.newaxis]) * mask[..., jnp.newaxis]

    return state._replace(
        eta=state.eta.replace(data=eta_f),
        u=state.u.replace(data=u_new),
        v=state.v.replace(data=v_new),
    )
