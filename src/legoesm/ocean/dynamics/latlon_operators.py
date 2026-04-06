"""Finite-volume operators on the latitude-longitude grid.

Provides PPM (Piecewise Parabolic Method) reconstruction and conservative
flux divergence for the lat-lon ocean dynamical core.

Grid conventions (LatLonGrid):
- Shape: (n_lat, n_lon) for 2D, (n_lat, n_lon, nlev) for 3D
- Latitude (axis 0): South-to-North, bounded (wall BCs at poles)
- Longitude (axis 1): periodic

References
----------
- Colella & Woodward (1984): The PPM method for gas dynamics
- Lin & Rood (1996): Multidimensional flux-form semi-Lagrangian transport
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid


# =============================================================================
# PPM reconstruction
# =============================================================================

def _ppm_edge_values_periodic(q: jnp.ndarray) -> jnp.ndarray:
    """Compute PPM edge values along the last axis (periodic BC).

    Fourth-order interpolation to cell interfaces with monotonicity clip.

    Parameters
    ----------
    q : array, shape (..., N)
        Cell-averaged values.

    Returns
    -------
    q_edge : array, shape (..., N)
        Edge values at left interface of each cell (periodic: N edges for N cells).
    """
    # 4th-order Colella-Woodward edge interpolation
    qm2 = jnp.roll(q, 2, axis=-1)
    qm1 = jnp.roll(q, 1, axis=-1)
    qp1 = jnp.roll(q, -1, axis=-1)
    # Edge at left face of cell i: between cell i-1 and cell i
    q_edge = (7.0 / 12.0) * (qm1 + q) - (1.0 / 12.0) * (qm2 + qp1)
    # Monotonicity clip
    q_min = jnp.minimum(qm1, q)
    q_max = jnp.maximum(qm1, q)
    q_edge = jnp.clip(q_edge, q_min, q_max)
    return q_edge


def _ppm_edge_values_bounded(q: jnp.ndarray) -> jnp.ndarray:
    """Compute PPM edge values along the first axis (bounded/wall BC).

    Returns N+1 edge values for N cells along axis 0.

    Parameters
    ----------
    q : array, shape (N, ...)
        Cell-averaged values along latitude.

    Returns
    -------
    q_edge : array, shape (N+1, ...)
        Edge values at cell interfaces, including boundary faces.
    """
    N = q.shape[0]
    # Interior edges (i=1..N-1): 4th-order where possible, 2nd-order near boundaries
    # Edge between cell i-1 and cell i
    edges = []
    # Bottom boundary (south pole wall): extrapolate
    edges.append(q[0:1])  # shape (1, ...)

    # i=1: only 2nd order (not enough stencil for 4th)
    e1 = 0.5 * (q[0:1] + q[1:2])
    edges.append(e1)

    if N > 3:
        # Interior: 4th-order for i=2..N-2
        qm1 = q[0:N - 3]   # i-2
        q0 = q[1:N - 2]     # i-1
        q1 = q[2:N - 1]     # i
        q2 = q[3:N]         # i+1
        e_int = (7.0 / 12.0) * (q0 + q1) - (1.0 / 12.0) * (qm1 + q2)
        # Monotonicity clip
        q_min = jnp.minimum(q0, q1)
        q_max = jnp.maximum(q0, q1)
        e_int = jnp.clip(e_int, q_min, q_max)
        edges.append(e_int)
    elif N > 2:
        # Only one interior edge at i=2, use 2nd order
        e2 = 0.5 * (q[1:2] + q[2:3])
        edges.append(e2)

    # i=N-1: 2nd order (only when not already covered by interior/elif)
    if N > 3:
        eN = 0.5 * (q[N - 2:N - 1] + q[N - 1:N])
        edges.append(eN)

    # Top boundary (north pole wall): extrapolate
    edges.append(q[N - 1:N])  # shape (1, ...)

    q_edge = jnp.concatenate(edges, axis=0)  # (N+1, ...)
    return q_edge


def _ppm_limit(q_bar: jnp.ndarray, q_L: jnp.ndarray, q_R: jnp.ndarray):
    """Colella-Woodward monotonicity limiter.

    Prevents new extrema in the piecewise parabolic reconstruction.

    Parameters
    ----------
    q_bar : cell average
    q_L : left edge value
    q_R : right edge value

    Returns
    -------
    q_L, q_R : limited edge values
    """
    # Detect local extrema: flatten parabola
    is_extremum = (q_R - q_bar) * (q_bar - q_L) <= 0.0
    q_L = jnp.where(is_extremum, q_bar, q_L)
    q_R = jnp.where(is_extremum, q_bar, q_R)

    # Check overshoot conditions
    dm = q_R - q_L
    d6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    # Left overshoot
    left_os = dm * d6 > dm ** 2
    q_L = jnp.where(left_os, 3.0 * q_bar - 2.0 * q_R, q_L)

    # Right overshoot
    right_os = dm * d6 < -(dm ** 2)
    q_R = jnp.where(right_os, 3.0 * q_bar - 2.0 * q_L, q_R)

    return q_L, q_R


# =============================================================================
# FV flux divergence on lat-lon
# =============================================================================

def fv_divergence_latlon(
    Fu: jnp.ndarray,
    Fv: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Conservative FV flux divergence on a lat-lon grid.

    Computes div(F) = (1/A) * [flux_east - flux_west + flux_north - flux_south]

    using centered interface averaging:
    - Zonal (longitude): periodic BC
    - Meridional (latitude): wall BC at poles (v=0)

    Parameters
    ----------
    Fu : array, shape (n_lat, n_lon)
        Zonal flux component (e.g., h*u for mass flux).
    Fv : array, shape (n_lat, n_lon)
        Meridional flux component (e.g., h*v for mass flux).
    grid : LatLonGrid
        Horizontal grid.
    mask : array, shape (n_lat, n_lon), optional
        Ocean mask (1 = ocean, 0 = land).  When provided, interface
        fluxes between an ocean cell and a land cell are forced to zero
        (wall boundary condition), preventing spurious mass flux through
        coastlines.

    Returns
    -------
    div_F : array, shape (n_lat, n_lon)
        Flux divergence.
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat
    cos_lat = grid.cos_lat  # (n_lat,)

    # --- Zonal fluxes (periodic in longitude) ---
    # Flux at east face of cell (i, j): between cell j and j+1
    # Interface velocity: average of adjacent cells
    Fu_east = 0.5 * (Fu + jnp.roll(Fu, -1, axis=1))
    if mask is not None:
        # Zero interface flux between ocean and land cells
        ocean_both_zonal = mask * jnp.roll(mask, -1, axis=1)
        Fu_east = Fu_east * ocean_both_zonal
    # Face length in meridional direction: R * dlat
    zonal_flux = Fu_east * R * dlat  # (n_lat, n_lon)
    # Net zonal flux: east - west = F_east(j) - F_east(j-1)
    net_zonal = zonal_flux - jnp.roll(zonal_flux, 1, axis=1)

    # --- Meridional fluxes (wall BC at poles) ---
    # cos(lat) at cell interfaces (between lat[i] and lat[i+1])
    lat = grid.lat  # (n_lat,)
    lat_half = 0.5 * (lat[:-1] + lat[1:])  # (n_lat-1,)
    cos_lat_half = jnp.cos(lat_half)  # (n_lat-1,)

    # Interface flux: average adjacent cells
    Fv_north_inner = 0.5 * (Fv[:-1] + Fv[1:])  # (n_lat-1, n_lon)
    if mask is not None:
        # Zero interface flux at land-ocean boundaries
        ocean_both_merid = mask[:-1] * mask[1:]  # (n_lat-1, n_lon)
        Fv_north_inner = Fv_north_inner * ocean_both_merid
    # Face length in zonal direction: R * cos(lat_face) * dlon
    merid_flux_inner = Fv_north_inner * R * cos_lat_half[:, None] * dlon

    # Wall BCs: zero flux at south and north poles
    zeros_row = jnp.zeros((1, grid.n_lon), dtype=Fv.dtype)
    merid_flux = jnp.concatenate(
        [zeros_row, merid_flux_inner, zeros_row], axis=0,
    )  # (n_lat+1, n_lon)
    # Net meridional flux: north - south
    net_merid = merid_flux[1:] - merid_flux[:-1]  # (n_lat, n_lon)

    # Divergence = (net_zonal + net_merid) / area
    div_F = (net_zonal + net_merid) / grid.area
    return div_F


def fv_divergence_latlon_3d(
    Fu: jnp.ndarray,
    Fv: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """3D FV flux divergence: vmap over vertical levels.

    Parameters
    ----------
    Fu, Fv : array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    div_F : array, shape (n_lat, n_lon, nlev)
    """
    # Move level axis to front for vmap
    Fu_t = jnp.moveaxis(Fu, -1, 0)  # (nlev, n_lat, n_lon)
    Fv_t = jnp.moveaxis(Fv, -1, 0)

    div_t = jax.vmap(
        lambda fu, fv: fv_divergence_latlon(fu, fv, grid),
    )(Fu_t, Fv_t)  # (nlev, n_lat, n_lon)

    return jnp.moveaxis(div_t, 0, -1)  # (n_lat, n_lon, nlev)


def fv_scalar_advection_latlon(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """PPM scalar advection on a lat-lon grid (advective form).

    Computes -div(q * v) using PPM reconstruction at cell interfaces
    with upwind selection.

    Parameters
    ----------
    q : array, shape (n_lat, n_lon)
        Scalar field.
    u, v : array, shape (n_lat, n_lon)
        Velocity components.
    grid : LatLonGrid

    Returns
    -------
    dq_dt : array, shape (n_lat, n_lon)
        Advective tendency.
    """
    R = grid.radius
    dlon = grid.dlon
    dlat = grid.dlat

    # --- Zonal advection (periodic, PPM along axis 1) ---
    # PPM edge values along longitude
    q_edge_lon = _ppm_edge_values_periodic(q)  # (n_lat, n_lon) — left face
    q_L_lon = q_edge_lon  # left edge of cell j
    q_R_lon = jnp.roll(q_edge_lon, -1, axis=-1)  # right edge of cell j
    q_L_lon, q_R_lon = _ppm_limit(q, q_L_lon, q_R_lon)

    # Interface velocity at east face: between cell j and j+1
    u_east = 0.5 * (u + jnp.roll(u, -1, axis=1))
    # Upwind selection: east face uses q_R of cell j if u>0, q_L of cell j+1 if u<0
    q_east = jnp.where(u_east > 0, q_R_lon, jnp.roll(q_L_lon, -1, axis=1))
    # Zonal flux: u * q * face_length = u * q * R * dlat
    zonal_flux = u_east * q_east * R * dlat
    net_zonal = zonal_flux - jnp.roll(zonal_flux, 1, axis=1)

    # --- Meridional advection (bounded, PPM along axis 0) ---
    # PPM edge values along latitude
    q_edge_lat = _ppm_edge_values_bounded(q)  # (n_lat+1, n_lon)
    q_L_lat = q_edge_lat[:-1]  # south edge of cell i (= north edge of i-1)
    q_R_lat = q_edge_lat[1:]   # north edge of cell i
    q_L_lat, q_R_lat = _ppm_limit(q, q_L_lat, q_R_lat)

    # Interface velocity at north face: between cell i and i+1
    v_north_inner = 0.5 * (v[:-1] + v[1:])  # (n_lat-1, n_lon)
    # Upwind selection
    q_north_inner = jnp.where(
        v_north_inner > 0, q_R_lat[:-1], q_L_lat[1:],
    )

    # Face length at lat interface: R * cos(lat_half) * dlon
    lat_half = 0.5 * (grid.lat[:-1] + grid.lat[1:])
    cos_lat_half = jnp.cos(lat_half)[:, None]
    merid_flux_inner = v_north_inner * q_north_inner * R * cos_lat_half * dlon

    # Wall BC: zero flux at poles
    zeros_row = jnp.zeros((1, grid.n_lon), dtype=q.dtype)
    merid_flux = jnp.concatenate(
        [zeros_row, merid_flux_inner, zeros_row], axis=0,
    )
    net_merid = merid_flux[1:] - merid_flux[:-1]

    dq_dt = -(net_zonal + net_merid) / grid.area
    return dq_dt


def fv_scalar_advection_latlon_3d(
    q: jnp.ndarray,
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """3D PPM scalar advection: vmap over vertical levels.

    Parameters
    ----------
    q, u, v : array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    dq_dt : array, shape (n_lat, n_lon, nlev)
    """
    q_t = jnp.moveaxis(q, -1, 0)
    u_t = jnp.moveaxis(u, -1, 0)
    v_t = jnp.moveaxis(v, -1, 0)

    dq_t = jax.vmap(
        lambda qi, ui, vi: fv_scalar_advection_latlon(qi, ui, vi, grid),
    )(q_t, u_t, v_t)

    return jnp.moveaxis(dq_t, 0, -1)


# =============================================================================
# Gradient operators
# =============================================================================

def gradient_x_latlon(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Zonal gradient df/dx on a lat-lon grid (centered, periodic).

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    df_dx : array, same shape as f
    """
    # 2-cell centered difference: (f[j+1] - f[j-1]) / dx
    # dx spans 2 cells = R * 2 * dlon * cos(lat)
    df = jnp.roll(f, -1, axis=1) - jnp.roll(f, 1, axis=1)
    if f.ndim == 2:
        return df / grid.dx
    else:
        return df / grid.dx[..., jnp.newaxis]


def gradient_y_latlon(
    f: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Meridional gradient df/dy on a lat-lon grid (centered, bounded).

    Zero-gradient boundary condition at the poles.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    df_dy : array, same shape as f
    """
    # Interior: 2-cell centered difference
    # Boundary: one-sided (or zero-gradient)
    dy = grid.dy  # scalar, spans 2 cells

    # Pad with boundary values (zero-gradient)
    f_pad = jnp.concatenate([f[0:1], f, f[-1:]], axis=0)

    df = f_pad[2:] - f_pad[:-2]  # same shape as f
    return df / dy


def divergence_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Velocity divergence on a lat-lon grid.

    div(v) = (1/(R cos φ)) * [du/dλ + d(v cos φ)/dφ]

    Parameters
    ----------
    u, v : array, shape (n_lat, n_lon)
    grid : LatLonGrid

    Returns
    -------
    div : array, shape (n_lat, n_lon)
    """
    du_dlon = gradient_x_latlon(u, grid)
    # For d(v cos φ)/dy, compute as gradient of (v * cos_lat)
    v_cos = v * grid.cos_lat[:, None]
    dv_cos_dy = gradient_y_latlon(v_cos, grid)
    return du_dlon + dv_cos_dy / grid.cos_lat[:, None]


def vorticity_latlon(
    u: jnp.ndarray,
    v: jnp.ndarray,
    grid: LatLonGrid,
) -> jnp.ndarray:
    """Relative vorticity on a lat-lon grid.

    zeta = (1/(R cos φ)) * [dv/dλ - d(u cos φ)/dφ]

    Parameters
    ----------
    u, v : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    zeta : array, same shape as u
    """
    dv_dx = gradient_x_latlon(v, grid)
    u_cos = u * grid.cos_lat[:, None] if u.ndim == 2 else u * grid.cos_lat[:, None, None]
    du_cos_dy = gradient_y_latlon(u_cos, grid)
    if u.ndim == 2:
        return dv_dx - du_cos_dy / grid.cos_lat[:, None]
    else:
        return dv_dx - du_cos_dy / grid.cos_lat[:, None, None]


def _neumann_fill_latlon(
    f: jnp.ndarray,
    mask: jnp.ndarray,
) -> jnp.ndarray:
    """Fill land cells with nearest ocean-neighbor value (4-connected).

    Enforces zero-gradient (Neumann) boundary conditions at land-ocean
    interfaces so that gradient and Laplacian operators do not see a
    sharp jump between ocean values and masked zeros.

    Checks all 4 neighbors (N, S, E, W with periodic E/W wrapping) so
    that both meridional and zonal coastlines are handled correctly.
    Without zonal neighbors, E/W coastlines still see the ocean-to-zero
    discontinuity in the zonal gradient.

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    mask : array, shape (n_lat, n_lon)
        Ocean mask (1 = ocean, 0 = land).

    Returns
    -------
    f_filled : array, same shape as f
    """
    m = mask
    filled = f
    # Multiple passes to propagate through consecutive land cells
    # (e.g. 2 land cells between the last ocean cell and the pole).
    for _ in range(3):
        # South neighbor (one cell toward south pole)
        f_s = jnp.concatenate([filled[0:1], filled[:-1]], axis=0)
        m_s = jnp.concatenate([m[0:1], m[:-1]], axis=0)
        # North neighbor (one cell toward north pole)
        f_n = jnp.concatenate([filled[1:], filled[-1:]], axis=0)
        m_n = jnp.concatenate([m[1:], m[-1:]], axis=0)
        # West neighbor (periodic)
        f_w = jnp.roll(filled, 1, axis=1)
        m_w = jnp.roll(m, 1, axis=1)
        # East neighbor (periodic)
        f_e = jnp.roll(filled, -1, axis=1)
        m_e = jnp.roll(m, -1, axis=1)

        is_land = m < 0.5

        # Weighted average of all ocean neighbors
        if f.ndim > 2:
            m_s_e = m_s[..., jnp.newaxis]
            m_n_e = m_n[..., jnp.newaxis]
            m_w_e = m_w[..., jnp.newaxis]
            m_e_e = m_e[..., jnp.newaxis]
            is_land_e = is_land[..., jnp.newaxis]
        else:
            m_s_e = m_s
            m_n_e = m_n
            m_w_e = m_w
            m_e_e = m_e
            is_land_e = is_land

        nbr_sum = f_s * m_s_e + f_n * m_n_e + f_w * m_w_e + f_e * m_e_e
        nbr_count = m_s_e + m_n_e + m_w_e + m_e_e
        nbr_avg = nbr_sum / jnp.maximum(nbr_count, 1.0)

        has_any_nbr = (m_s + m_n + m_w + m_e) > 0.0
        if f.ndim > 2:
            has_any_nbr_e = has_any_nbr[..., jnp.newaxis]
        else:
            has_any_nbr_e = has_any_nbr

        filled = jnp.where(is_land_e & has_any_nbr_e, nbr_avg, filled)
        # Expand the effective mask so the next pass can propagate further.
        m = jnp.where(is_land & has_any_nbr, 1.0, m)

    return filled


def laplacian_latlon(
    f: jnp.ndarray,
    grid: LatLonGrid,
    *,
    mask: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """Scalar Laplacian on a lat-lon grid.

    lap(f) = (1/(R² cos² φ)) d²f/dλ² + (1/(R² cos φ)) d/dφ(cos φ df/dφ)

    Parameters
    ----------
    f : array, shape (n_lat, n_lon) or (n_lat, n_lon, nlev)
    grid : LatLonGrid
    mask : array, shape (n_lat, n_lon), optional
        Ocean mask (1 = ocean, 0 = land).  When provided, land cells are
        filled with their nearest meridional ocean-neighbor value before
        computing meridional gradients.  This enforces a zero-flux
        (Neumann) boundary condition at land-ocean interfaces and
        prevents the spurious horizontal stripes that otherwise arise
        from diffusing against a sharp masked discontinuity.

    Returns
    -------
    lap_f : array, same shape as f
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon

    # d²f/dλ² (periodic)
    d2f_dlon2 = (
        jnp.roll(f, -1, axis=1) - 2.0 * f + jnp.roll(f, 1, axis=1)
    ) / dlon ** 2

    # d/dφ(cos φ df/dφ): finite difference with cos(lat) metric
    # When a mask is provided, fill land cells so meridional gradients
    # do not see the ocean-to-zero discontinuity.
    f_merid = _neumann_fill_latlon(f, mask) if mask is not None else f

    # Pad in lat with boundary values
    f_pad = jnp.concatenate([f_merid[0:1], f_merid, f_merid[-1:]], axis=0)

    # df/dφ at half-levels
    df_north = (f_pad[2:] - f_pad[1:-1]) / dlat  # north face of cell i
    df_south = (f_pad[1:-1] - f_pad[:-2]) / dlat  # south face of cell i

    # cos(lat) at half-levels
    lat = grid.lat
    cos_half_north = jnp.cos(
        jnp.concatenate([0.5 * (lat[:-1] + lat[1:]), jnp.array([jnp.pi / 2])])
    )
    cos_half_south = jnp.cos(
        jnp.concatenate([jnp.array([-jnp.pi / 2]), 0.5 * (lat[:-1] + lat[1:])])
    )

    if f.ndim == 2:
        d_cos_df = (
            cos_half_north[:, None] * df_north - cos_half_south[:, None] * df_south
        ) / dlat
        cos_lat_2d = grid.cos_lat[:, None]
    else:
        d_cos_df = (
            cos_half_north[:, None, None] * df_north
            - cos_half_south[:, None, None] * df_south
        ) / dlat
        cos_lat_2d = grid.cos_lat[:, None, None]

    lap_f = (1.0 / (R ** 2 * cos_lat_2d ** 2)) * d2f_dlon2 + (
        1.0 / (R ** 2 * cos_lat_2d)
    ) * d_cos_df

    return lap_f
