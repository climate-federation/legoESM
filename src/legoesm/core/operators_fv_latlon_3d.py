"""Native 3D FV transport operators on the lat-lon grid.

Pads halos once for all levels simultaneously, replacing the per-level
vmap approach that produced O(nlev) separate halo exchanges under
multi-GPU sharding.  PPM reconstruction is still 1D along grid lines,
but the stencil operations are applied to all levels in one pass.

All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.core.operators_fv import _ppm_edge_values, _ppm_limit
from legoesm.grids.halo_latlon import (
    pad_halo_latlon_3d,
    pad_halo_vector_latlon_3d,
)
from legoesm.grids.latlon import LatLonGrid


# ==============================================================================
# PPM reconstruction — native 3D
# ==============================================================================

def _ppm_reconstruct_lon_3d(q_pad_h2, limiter=True):
    """PPM reconstruction in longitude for all levels at once.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (n_lat+4, n_lon+4, nlev)
    limiter : bool

    Returns
    -------
    q_left, q_right : each shape (n_lat, n_lon+1, nlev)
    """
    # Strip latitude halo, keep longitude halo
    q = q_pad_h2[2:-2, :, :]  # (n_lat, n_lon+4, nlev)

    # axis -2 is longitude (n_lon+4) — _ppm_edge_values operates here
    q_hat = _ppm_edge_values(q)  # (n_lat, n_lon+3, nlev)

    a_L = q_hat[:, :-1, :]   # (n_lat, n_lon+2, nlev)
    a_R = q_hat[:, 1:, :]
    q_c = q[:, 1:-1, :]      # (n_lat, n_lon+2, nlev)

    if limiter:
        a_L, a_R = _ppm_limit(q_c, a_L, a_R)

    q_left = a_R[:, :-1, :]   # (n_lat, n_lon+1, nlev)
    q_right = a_L[:, 1:, :]

    return q_left, q_right


def _ppm_reconstruct_lat_3d(q_pad_h2, limiter=True):
    """PPM reconstruction in latitude for all levels at once.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (n_lat+4, n_lon+4, nlev)
    limiter : bool

    Returns
    -------
    q_left, q_right : each shape (n_lat+1, n_lon, nlev)
    """
    # Strip longitude halo, keep latitude halo
    q = q_pad_h2[:, 2:-2, :]  # (n_lat+4, n_lon, nlev)

    # Move lat axis (0) to axis -2 (1) so _ppm_edge_values operates on it
    q_moved = jnp.moveaxis(q, 0, 1)  # (n_lon, n_lat+4, nlev)
    q_hat_moved = _ppm_edge_values(q_moved)  # (n_lon, n_lat+3, nlev)
    q_hat = jnp.moveaxis(q_hat_moved, 1, 0)  # (n_lat+3, n_lon, nlev)

    a_L = q_hat[:-1, :, :]   # (n_lat+2, n_lon, nlev)
    a_R = q_hat[1:, :, :]
    q_c = q[1:-1, :, :]      # (n_lat+2, n_lon, nlev)

    if limiter:
        a_L, a_R = _ppm_limit(q_c, a_L, a_R)

    q_left = a_R[:-1, :, :]   # (n_lat+1, n_lon, nlev)
    q_right = a_L[1:, :, :]

    return q_left, q_right


# ==============================================================================
# Unsplit 2D flux-form transport — native 3D
# ==============================================================================

def fv_flux_divergence_latlon_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: LatLonGrid, limiter: bool = True,
) -> jax.Array:
    """Conservative FV flux divergence at all levels (single halo pad).

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (n_lat, n_lon, nlev)
    """
    R = grid.radius
    dlat = grid.dlat
    dlon = grid.dlon

    # Pad all fields with halo=2, once for all levels
    q_pad = pad_halo_latlon_3d(q_3d, halo=2)
    u_pad, v_pad = pad_halo_vector_latlon_3d(u_3d, v_3d, halo=2)

    # --- Longitude flux ---
    q_L_lon, q_R_lon = _ppm_reconstruct_lon_3d(q_pad, limiter)  # (n_lat, n_lon+1, nlev)

    u_strip = u_pad[2:-2, :, :]  # (n_lat, n_lon+4, nlev)
    u_iface = 0.5 * (u_strip[:, 1:-2, :] + u_strip[:, 2:-1, :])  # (n_lat, n_lon+1, nlev)

    q_face_lon = jnp.where(u_iface > 0, q_L_lon, q_R_lon)

    hy = R * dlat
    Phi_lon = u_iface * hy * q_face_lon  # (n_lat, n_lon+1, nlev)

    # --- Latitude flux ---
    q_L_lat, q_R_lat = _ppm_reconstruct_lat_3d(q_pad, limiter)  # (n_lat+1, n_lon, nlev)

    v_strip = v_pad[:, 2:-2, :]  # (n_lat+4, n_lon, nlev)
    v_iface = 0.5 * (v_strip[1:-2, :, :] + v_strip[2:-1, :, :])  # (n_lat+1, n_lon, nlev)

    q_face_lat = jnp.where(v_iface > 0, q_L_lat, q_R_lat)

    n_lat = grid.n_lat
    lat_iface = jnp.linspace(-jnp.pi / 2, jnp.pi / 2, n_lat + 1)
    hx_iface = R * dlon * jnp.cos(lat_iface)[:, None, None]  # (n_lat+1, 1, 1)

    Phi_lat = v_iface * hx_iface * q_face_lat  # (n_lat+1, n_lon, nlev)

    # --- Net flux divergence ---
    net_lon = Phi_lon[:, 1:, :] - Phi_lon[:, :-1, :]   # (n_lat, n_lon, nlev)
    net_lat = Phi_lat[1:, :, :] - Phi_lat[:-1, :, :]   # (n_lat, n_lon, nlev)

    return -(net_lon + net_lat) / grid.area[:, :, None]


def fv_scalar_advection_latlon_3d(
    q_3d: jax.Array, u_3d: jax.Array, v_3d: jax.Array,
    grid: LatLonGrid, limiter: bool = True,
) -> jax.Array:
    """PPM advection of scalar at all levels (single halo pad).

    Parameters
    ----------
    q_3d : jax.Array, shape (n_lat, n_lon, nlev)
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    limiter : bool

    Returns
    -------
    jax.Array : shape (n_lat, n_lon, nlev)
    """
    flux_form = fv_flux_divergence_latlon_3d(q_3d, u_3d, v_3d, grid, limiter)
    div_v = -fv_flux_divergence_latlon_3d(
        jnp.ones_like(q_3d), u_3d, v_3d, grid, limiter=False)
    return flux_form + q_3d * div_v
