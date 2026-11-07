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

from legoesm.core.operators_fv import ppm_edge_values, ppm_limit
from legoesm.grids.halo_latlon import (
    pad_halo_latlon,
    pad_halo_vector_latlon,
    pad_halo_latlon_3d,
)


def lat_v_interfaces(grid):
    """v-face latitudes for the PPM meridional flux metric: ``grid.lat_v``.

    Returns the grid's OWNED v-face latitudes (interior faces at
    cell-center midpoints, end faces by half-cell extrapolation —
    ``grids.latlon.compute_v_face_coords``), which land on ±π/2 only
    when the grid actually reaches the poles.  Consistent with
    ``latlon_cgrid_operators.divergence_cgrid`` (which reads
    ``grid.cos_lat_v``) and with non-uniform / face-defined latitude
    grids (Mercator, Veros-style), whose true faces are not
    center-midpoints.

    NEVER fabricate ±π/2 ends here: under the SPMD/MPI decompositions
    ``grid`` is a lat-band or 2-D tile slice (``slice_latlon_grid_to_band``
    / ``slice_latlon_grid_to_block_2d`` carry ``lat_v[s:e+1]``), so a
    local end face is an INTERIOR partition cut.  A hard-coded pole
    turned every cut face into a near-zero-length wall
    (``cos(±π/2) -> 1e-10`` clamp) in the flux metric
    ``hx = R*dlon*cos(lat_v)`` — spuriously blocking PPM meridional
    transport through the cut (codex M3a findings 2/6; the same
    regression class ``compute_v_face_coords`` documents from Codex
    review Stage 3-E round 3).

    Parameters
    ----------
    grid : LatLonGrid

    Returns
    -------
    lat_v : jax.Array, shape (n_lat+1,)
    """
    return grid.lat_v


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
    q_hat_t = ppm_edge_values(q_t)

    # Parabola for interior cells
    a_L_t = q_hat_t[:-1, :]   # (n_lon+2, n_lat)
    a_R_t = q_hat_t[1:, :]
    q_c_t = q_t[1:-1, :]      # (n_lon+2, n_lat)

    if limiter:
        a_L_t, a_R_t = ppm_limit(q_c_t, a_L_t, a_R_t)

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
    q_hat = ppm_edge_values(q)  # (n_lat+3, n_lon)

    a_L = q_hat[:-1, :]   # (n_lat+2, n_lon)
    a_R = q_hat[1:, :]
    q_c = q[1:-1, :]      # (n_lat+2, n_lon)

    if limiter:
        a_L, a_R = ppm_limit(q_c, a_L, a_R)

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
    dlon = grid.dlon

    # Pad with halo=2
    q_pad = pad_halo_latlon(q, halo=2)
    u_pad, v_pad = pad_halo_vector_latlon(u, v, halo=2)

    # --- Longitude flux ---
    q_L_lon, q_R_lon = _ppm_reconstruct_lon(q_pad, limiter)  # (n_lat, n_lon+1)

    # u at longitude interfaces (average neighboring cells)
    u_strip = u_pad[2:-2, :]  # (n_lat, n_lon+4)
    u_iface = 0.5 * (u_strip[:, 1:-2] + u_strip[:, 2:-1])  # (n_lat, n_lon+1)

    q_face_lon = jnp.where(u_iface >= 0, q_L_lon, q_R_lon)

    # Edge length perpendicular to longitude: cell-row height
    # (1D over latitude → Mercator-safe).
    hy = (grid.dy * 0.5)[:, None]                            # (n_lat, 1)
    Phi_lon = u_iface * hy * q_face_lon  # (n_lat, n_lon+1)

    # --- Latitude flux ---
    q_L_lat, q_R_lat = _ppm_reconstruct_lat(q_pad, limiter)  # (n_lat+1, n_lon)

    # v at latitude interfaces
    v_strip = v_pad[:, 2:-2]  # (n_lat+4, n_lon)
    v_iface = 0.5 * (v_strip[1:-2, :] + v_strip[2:-1, :])  # (n_lat+1, n_lon)

    q_face_lat = jnp.where(v_iface >= 0, q_L_lat, q_R_lat)

    # Edge length perpendicular to latitude at interfaces:
    # hx = R * dlon * cos(lat_v).  ``grid.cos_lat_v`` is the band-correct
    # v-face metric (pre-sliced by slice_latlon_grid_to_band under MPI/SPMD;
    # clamped to 1e-10 only at the true global poles).  Hard-coding ±π/2
    # endpoints here would collapse interior band-cut faces to ~zero area.
    hx_iface = R * dlon * grid.cos_lat_v[:, None]

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

def fv_gradient_lon_3d(q_3d, grid, padded=None):
    """3D-native PPM-compatible longitude gradient.

    PPM-compatible longitude gradient: one halo pad + one PPM
    reconstruction shared across all vertical levels.

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    padded : jax.Array, optional
        Pre-padded field with halo=2, shape ``(n_lat+4, n_lon+4, nlev)``.
        When supplied, the internal ``pad_halo_latlon_3d`` call is
        skipped — useful for paired (∂/∂lon, ∂/∂lat) calls on the
        same input where the halo pad can be shared.

    Returns
    -------
    jax.Array, shape (n_lat, n_lon, nlev)
    """
    if padded is None:
        # Pad once for all levels.
        padded = pad_halo_latlon_3d(q_3d, halo=2)        # (n_lat+4, n_lon+4, nlev)
    # Strip latitude halo; longitude axis is now axis -2 (the axis
    # ``ppm_edge_values`` operates on).
    q_strip = padded[2:-2, :, :]                         # (n_lat, n_lon+4, nlev)
    q_hat = ppm_edge_values(q_strip)                    # (n_lat, n_lon+3, nlev)
    q_edges = q_hat[:, 1:-1, :]                          # (n_lat, n_lon+1, nlev)
    dq = q_edges[:, 1:, :] - q_edges[:, :-1, :]          # (n_lat, n_lon, nlev)
    return dq / (grid.dx[..., None] / 2.0)


def fv_gradient_lat_3d(q_3d, grid, padded=None):
    """3D-native PPM-compatible latitude gradient.

    PPM-compatible latitude gradient: one halo pad + one PPM
    reconstruction shared across all vertical levels.  Optional
    ``padded=`` skips the internal halo pad — see
    :func:`fv_gradient_lon_3d` for usage.

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    padded : jax.Array, optional
        Pre-padded field, shape ``(n_lat+4, n_lon+4, nlev)``.

    Returns
    -------
    jax.Array, shape (n_lat, n_lon, nlev)
    """
    if padded is None:
        padded = pad_halo_latlon_3d(q_3d, halo=2)        # (n_lat+4, n_lon+4, nlev)
    # Strip longitude halo.  ``ppm_edge_values`` operates on axis -2,
    # so swap the lat/lon axes so latitude lives there.
    q_strip = padded[:, 2:-2, :]                         # (n_lat+4, n_lon, nlev)
    q_t = jnp.swapaxes(q_strip, 0, 1)                    # (n_lon, n_lat+4, nlev)
    q_hat_t = ppm_edge_values(q_t)                      # (n_lon, n_lat+3, nlev)
    q_edges_t = q_hat_t[:, 1:-1, :]                      # (n_lon, n_lat+1, nlev)
    dq_t = q_edges_t[:, 1:, :] - q_edges_t[:, :-1, :]    # (n_lon, n_lat, nlev)
    return jnp.swapaxes(dq_t, 0, 1) / (grid.dy[:, None, None] / 2.0)


# ==============================================================================
# C-grid PPM transport (face-centered velocities)
# ==============================================================================

def cgrid_fv_flux_divergence_latlon(q, u_face, v_face, grid, limiter=True):
    """Conservative PPM flux divergence using C-grid face velocities.

    Like ``fv_flux_divergence_latlon`` but takes velocities already at faces
    (Arakawa C-grid staggering), eliminating the velocity interpolation step.
    Shared by atmosphere and ocean lat-lon C-grid dycores.

    Parameters
    ----------
    q : jax.Array, shape (n_lat, n_lon)
        Scalar field at cell centers (e.g. fluid depth h, or tracer).
    u_face : jax.Array, shape (n_lat, n_lon+1)
        Zonal velocity at longitude interfaces (C-grid u-points).
    v_face : jax.Array, shape (n_lat+1, n_lon)
        Meridional velocity at latitude interfaces (C-grid v-points).
    grid : LatLonGrid
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.

    Returns
    -------
    jax.Array, shape (n_lat, n_lon)
        Flux divergence tendency: dq/dt = -div(q * v).
    """
    R = grid.radius
    dlon = grid.dlon

    # Pad scalar with halo=2 for PPM reconstruction
    q_pad = pad_halo_latlon(q, halo=2)

    # --- Longitude flux ---
    q_L_lon, q_R_lon = _ppm_reconstruct_lon(q_pad, limiter)  # (n_lat, n_lon+1)
    q_face_lon = jnp.where(u_face >= 0, q_L_lon, q_R_lon)

    # Face length perpendicular to longitude: cell-row height (1D over
    # latitude → Mercator-safe).
    hy = (grid.dy * 0.5)[:, None]                            # (n_lat, 1)
    Phi_lon = u_face * hy * q_face_lon  # (n_lat, n_lon+1)

    # --- Latitude flux ---
    q_L_lat, q_R_lat = _ppm_reconstruct_lat(q_pad, limiter)  # (n_lat+1, n_lon)
    q_face_lat = jnp.where(v_face >= 0, q_L_lat, q_R_lat)

    # Face length at latitude interfaces: R * dlon * cos(lat_v).  Uses the
    # band-correct pre-sliced ``grid.cos_lat_v`` (see fv_flux_divergence_latlon).
    hx_iface = R * dlon * grid.cos_lat_v[:, None]
    Phi_lat = v_face * hx_iface * q_face_lat  # (n_lat+1, n_lon)

    # --- Net flux divergence ---
    net_lon = Phi_lon[:, 1:] - Phi_lon[:, :-1]   # (n_lat, n_lon)
    net_lat = Phi_lat[1:, :] - Phi_lat[:-1, :]   # (n_lat, n_lon)

    return -(net_lon + net_lat) / grid.area


