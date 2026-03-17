"""Finite-volume PPM transport operators on the latitude-longitude grid.

Adapts the cubed-sphere PPM operators (operators_fv.py) for the lat-lon grid.

Key simplifications vs cubed-sphere:
- No face connectivity or angle rotation
- dy is constant, dx varies only with latitude
- Periodic longitude wrapping via halo exchange
- Zero-gradient polar boundaries

Key design (same as cubed-sphere):
- PPM reconstruction is purely 1D along grid lines
- Both lon and lat fluxes are computed on the SAME unmodified field
  (no directional splitting) to preserve geostrophic balance

References
----------
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core (FV3)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.operators_fv import _ppm_edge_values, _ppm_limit
from legoesm.grids.halo_latlon import pad_halo_latlon, pad_halo_vector_latlon


# ==============================================================================
# PPM reconstruction for lat-lon grid
# ==============================================================================

def _ppm_reconstruct_lon(q_pad_h2, limiter=True):
    """PPM reconstruction in the longitude direction.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (n_lat+4, n_lon+4)
        Scalar padded with halo=2.
    limiter : bool
        Apply Colella-Woodward limiter.

    Returns
    -------
    q_left, q_right : each shape (n_lat, n_lon+1)
        Left and right states at longitude interfaces.
    """
    # Strip latitude halo, keep longitude halo
    q = q_pad_h2[2:-2, :]  # (n_lat, n_lon+4)

    # Transpose so axis=-2 is the longitude direction: (n_lon+4, n_lat)
    q_t = q.T  # (n_lon+4, n_lat)

    # Edge values: (n_lon+3, n_lat)
    q_hat_t = _ppm_edge_values(q_t)

    # Parabola for interior cells
    a_L_t = q_hat_t[:-1, :]   # (n_lon+2, n_lat)
    a_R_t = q_hat_t[1:, :]
    q_c_t = q_t[1:-1, :]      # (n_lon+2, n_lat)

    if limiter:
        a_L_t, a_R_t = _ppm_limit(q_c_t, a_L_t, a_R_t)

    # Extract interface states: n_lon+1 interfaces
    q_left_t = a_R_t[:-1, :]   # (n_lon+1, n_lat)
    q_right_t = a_L_t[1:, :]

    return q_left_t.T, q_right_t.T  # each (n_lat, n_lon+1)


def _ppm_reconstruct_lat(q_pad_h2, limiter=True):
    """PPM reconstruction in the latitude direction.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (n_lat+4, n_lon+4)
        Scalar padded with halo=2.
    limiter : bool
        Apply Colella-Woodward limiter.

    Returns
    -------
    q_left, q_right : each shape (n_lat+1, n_lon)
        Left and right states at latitude interfaces.
    """
    # Strip longitude halo, keep latitude halo
    q = q_pad_h2[:, 2:-2]  # (n_lat+4, n_lon)

    # axis=-2 is already latitude: (n_lat+4, n_lon)
    q_hat = _ppm_edge_values(q)  # (n_lat+3, n_lon)

    a_L = q_hat[:-1, :]   # (n_lat+2, n_lon)
    a_R = q_hat[1:, :]
    q_c = q[1:-1, :]      # (n_lat+2, n_lon)

    if limiter:
        a_L, a_R = _ppm_limit(q_c, a_L, a_R)

    q_left = a_R[:-1, :]   # (n_lat+1, n_lon)
    q_right = a_L[1:, :]

    return q_left, q_right


# ==============================================================================
# Unsplit 2D flux-form transport
# ==============================================================================

def fv_flux_divergence_latlon(q, u, v, grid, limiter=True):
    """Conservative flux-form 2D transport using PPM on the lat-lon grid.

    Both lon and lat fluxes are computed on the SAME unmodified field q
    (no directional splitting) to preserve geostrophic balance.

    Parameters
    ----------
    q : jax.Array, shape (n_lat, n_lon)
        Scalar field to transport (e.g. fluid depth h).
    u, v : jax.Array, shape (n_lat, n_lon)
        Velocity components (zonal, meridional).
    grid : LatLonGrid
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.

    Returns
    -------
    jax.Array, shape (n_lat, n_lon)
        Flux divergence tendency: dq/dt = -div(q * v).
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon

    # Pad with halo=2
    q_pad = pad_halo_latlon(q, halo=2)
    u_pad, v_pad = pad_halo_vector_latlon(u, v, halo=2)

    # --- Longitude flux ---
    q_L_lon, q_R_lon = _ppm_reconstruct_lon(q_pad, limiter)  # (n_lat, n_lon+1)

    # u at longitude interfaces (average neighboring cells)
    u_strip = u_pad[2:-2, :]  # (n_lat, n_lon+4)
    u_iface = 0.5 * (u_strip[:, 1:-2] + u_strip[:, 2:-1])  # (n_lat, n_lon+1)

    q_face_lon = jnp.where(u_iface > 0, q_L_lon, q_R_lon)

    # Edge length perpendicular to longitude (hy = R * dlat, constant)
    hy = R * dlat
    Phi_lon = u_iface * hy * q_face_lon  # (n_lat, n_lon+1)

    # --- Latitude flux ---
    q_L_lat, q_R_lat = _ppm_reconstruct_lat(q_pad, limiter)  # (n_lat+1, n_lon)

    # v at latitude interfaces
    v_strip = v_pad[:, 2:-2]  # (n_lat+4, n_lon)
    v_iface = 0.5 * (v_strip[1:-2, :] + v_strip[2:-1, :])  # (n_lat+1, n_lon)

    q_face_lat = jnp.where(v_iface > 0, q_L_lat, q_R_lat)

    # Edge length perpendicular to latitude at interfaces: hx = R * dlon * cos(lat_iface)
    n_lat = grid.n_lat
    lat_iface = jnp.linspace(-jnp.pi / 2 + dlat / 2 - dlat / 2,
                              jnp.pi / 2 - dlat / 2 + dlat / 2,
                              n_lat + 1)
    # Simpler: interfaces are at cell boundaries
    lat_iface = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat + 1)
    hx_iface = R * dlon * jnp.cos(lat_iface)[:, None]  # (n_lat+1, 1) -> broadcast

    Phi_lat = v_iface * hx_iface * q_face_lat  # (n_lat+1, n_lon)

    # --- Net flux divergence ---
    net_lon = Phi_lon[:, 1:] - Phi_lon[:, :-1]   # (n_lat, n_lon)
    net_lat = Phi_lat[1:, :] - Phi_lat[:-1, :]   # (n_lat, n_lon)

    return -(net_lon + net_lat) / grid.area


def fv_scalar_advection_latlon(q, u, v, grid, limiter=True):
    """PPM advection of scalar q by (u,v) on the lat-lon grid (advective form).

    For tracers/temperature where we want -v·grad(q), not -div(q*v):

        -v·∇q = -div(q v) + q div(v)

    Uses the same FV transport operator for both terms, preserving
    constant-field invariance under divergent flow.

    Parameters
    ----------
    q : jax.Array, shape (n_lat, n_lon)
    u, v : jax.Array, shape (n_lat, n_lon)
    grid : LatLonGrid
    limiter : bool

    Returns
    -------
    jax.Array, shape (n_lat, n_lon)
    """
    flux_form = fv_flux_divergence_latlon(q, u, v, grid, limiter)
    div_v = -fv_flux_divergence_latlon(jnp.ones_like(q), u, v, grid, limiter=False)
    return flux_form + q * div_v


# ==============================================================================
# PPM-compatible gradients
# ==============================================================================

def fv_gradient_lon(q, grid):
    """PPM-compatible longitude gradient using 4th-order edge values.

    Computes dq/dx at cell centers by differencing PPM edge values
    at the left and right cell boundaries.

    Parameters
    ----------
    q : jax.Array, shape (n_lat, n_lon)
    grid : LatLonGrid

    Returns
    -------
    jax.Array, shape (n_lat, n_lon)
    """
    q_pad = pad_halo_latlon(q, halo=2)

    # Strip latitude halo
    q_strip = q_pad[2:-2, :]  # (n_lat, n_lon+4)

    # Transpose for PPM along longitude
    q_t = q_strip.T  # (n_lon+4, n_lat)
    q_hat_t = _ppm_edge_values(q_t)  # (n_lon+3, n_lat)

    # Interior edges: n_lon+1
    q_edges_t = q_hat_t[1:-1, :]  # (n_lon+1, n_lat)

    # Gradient: (right - left) / cell_width
    dq_t = q_edges_t[1:, :] - q_edges_t[:-1, :]  # (n_lon, n_lat)

    # grid.dx spans 2 cells, single-cell = dx/2
    return dq_t.T / (grid.dx / 2.0)


def fv_gradient_lat(q, grid):
    """PPM-compatible latitude gradient using 4th-order edge values.

    Parameters
    ----------
    q : jax.Array, shape (n_lat, n_lon)
    grid : LatLonGrid

    Returns
    -------
    jax.Array, shape (n_lat, n_lon)
    """
    q_pad = pad_halo_latlon(q, halo=2)

    # Strip longitude halo
    q_strip = q_pad[:, 2:-2]  # (n_lat+4, n_lon)

    # axis=-2 is already latitude
    q_hat = _ppm_edge_values(q_strip)  # (n_lat+3, n_lon)

    # Interior edges
    q_edges = q_hat[1:-1, :]  # (n_lat+1, n_lon)

    # Gradient: (right - left) / cell_width
    dq = q_edges[1:, :] - q_edges[:-1, :]  # (n_lat, n_lon)

    # grid.dy spans 2 cells, single-cell = dy/2
    return dq / (grid.dy / 2.0)
