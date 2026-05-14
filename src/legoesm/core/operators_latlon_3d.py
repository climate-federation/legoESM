"""Native 3D operators for lat-lon grids.

All operators pad halos once for all levels simultaneously, avoiding
the per-level vmap approach that produced O(nlev) separate halo
exchanges under multi-GPU sharding.

All functions operate on raw jax.Array data with shape
(n_lat, n_lon, nlev), where the level axis is last.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm.grids.latlon import LatLonGrid
from legoesm.grids.halo_latlon import (
    pad_halo_latlon_3d,
    pad_halo_latlon_vector_3d,
    pad_halo_vector_latlon_3d,
)


# ==============================================================================
# Gradient operators
# ==============================================================================

def gradient_x_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute x-gradient at all levels (single 3D halo pad).

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dx, shape (n_lat, n_lon, nlev).
    """
    padded = pad_halo_latlon_3d(field_3d)
    # Centered difference: (f[j, i+1, :] - f[j, i-1, :]) / dx
    df_dx = (padded[1:-1, 2:] - padded[1:-1, :-2]) / grid.dx[:, :, None]
    return df_dx


def gradient_y_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute y-gradient at all levels (single 3D halo pad).

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : d(field)/dy, shape (n_lat, n_lon, nlev).
    """
    padded = pad_halo_latlon_3d(field_3d)
    # grid.dy is (n_lat,); broadcast over (n_lon, nlev). The denominator
    # is twice the cell-row's single-cell height, which equals the
    # cell-centre-to-cell-centre 2-cell distance on uniform-dlat grids
    # (the only setting this A-grid path is used in; Mercator is
    # C-grid-only). Leading-order approximation on non-uniform grids.
    df_dy = (padded[2:, 1:-1] - padded[:-2, 1:-1]) / grid.dy[:, None, None]
    return df_dy


# ==============================================================================
# Divergence and curl
# ==============================================================================

def divergence_3d(u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute divergence at all levels (single 3D halo pad).

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Divergence, shape (n_lat, n_lon, nlev).
    """
    # Form fluxes with metric factors. grid.dy is (n_lat,) → broadcast.
    flux_x = u_3d * (grid.dy * 0.5)[:, None, None]
    flux_y = v_3d * (grid.dx * 0.5)[:, :, None]

    flux_x_pad, flux_y_pad = pad_halo_vector_latlon_3d(flux_x, flux_y)

    d_flux_x = flux_x_pad[1:-1, 2:] - flux_x_pad[1:-1, :-2]
    d_flux_y = flux_y_pad[2:, 1:-1] - flux_y_pad[:-2, 1:-1]

    return (d_flux_x + d_flux_y) / (2.0 * grid.area[:, :, None])


def vorticity_3d(u_3d: jax.Array, v_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute vorticity at all levels (single 3D halo pad).

    Parameters
    ----------
    u_3d, v_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Vorticity, shape (n_lat, n_lon, nlev).
    """
    v_metric = v_3d * (grid.dy * 0.5)[:, None, None]
    u_metric = u_3d * (grid.dx * 0.5)[:, :, None]

    v_pad = pad_halo_latlon_vector_3d(v_metric)
    u_pad = pad_halo_latlon_vector_3d(u_metric)

    dv_dx = v_pad[1:-1, 2:] - v_pad[1:-1, :-2]
    du_dy = u_pad[2:, 1:-1] - u_pad[:-2, 1:-1]

    return (dv_dx - du_dy) / (2.0 * grid.area[:, :, None])


# ==============================================================================
# Laplacian and hyperdiffusion
# ==============================================================================

def laplacian_3d(field_3d: jax.Array, grid: LatLonGrid) -> jax.Array:
    """Compute Laplacian at all levels (single 3D halo pad).

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid

    Returns
    -------
    jax.Array : Laplacian, shape (n_lat, n_lon, nlev).
    """
    padded = pad_halo_latlon_3d(field_3d)

    d2f_dx2 = (
        padded[1:-1, 2:] - 2.0 * field_3d + padded[1:-1, :-2]
    ) / (grid.dx**2 / 4.0)[:, :, None]

    # Uniform-dlat second-derivative; this A-grid 3D Laplacian only
    # runs on uniform global lat-lon grids (Mercator is C-grid-only).
    d2f_dy2 = (
        padded[2:, 1:-1] - 2.0 * field_3d + padded[:-2, 1:-1]
    ) / ((grid.dy**2 / 4.0)[:, None, None])

    return d2f_dx2 + d2f_dy2


def hyperdiffusion_3d(field_3d: jax.Array, grid: LatLonGrid, coeff: float) -> jax.Array:
    """Compute hyperdiffusion at all levels: -coeff * nabla^4(field).

    Parameters
    ----------
    field_3d : jax.Array, shape (n_lat, n_lon, nlev)
    grid : LatLonGrid
    coeff : float

    Returns
    -------
    jax.Array : Hyperdiffusion tendency, shape (n_lat, n_lon, nlev).
    """
    lap1 = laplacian_3d(field_3d, grid)
    lap2 = laplacian_3d(lap1, grid)
    return -coeff * lap2
