"""Halo-aware versions of plane stencil operators for MPI dycore.

Operators in :mod:`plane_operators` use ``jnp.roll`` for periodic
boundaries. Under MPI domain decomposition, ``jnp.roll`` wraps to
the wrong side (local boundary instead of neighbor rank's
opposite-edge data). This module provides equivalent operators
that operate on a HALO-PADDED array (shape
``(ny_local + 2h, nx_local + 2h, ...)``), using slicing into the
halo cells instead of wrapping.

Caller workflow:

1. Local interior state has shape ``(ny_local, nx_local, ...)``.
2. Call :func:`legoesm.parallel.plane_mpi.exchange_halo_plane_yxz`
   to fill halos → padded shape ``(ny_local + 2h, nx_local + 2h, ...)``.
3. Apply these halo-aware operators on the padded array → they
   return INTERIOR-shape results ``(ny_local, nx_local, ...)``
   that can be combined into tendencies.

Single-rank fallback: with ``layout.n_ranks == 1``,
``exchange_halo_plane_yxz`` uses ``jnp.pad(mode='wrap')`` so the
padded array contains the correct periodic data — the halo-aware
operators then produce bit-identical results to the unpadded
``jnp.roll`` versions.

All operators in this module assume ``halo == 1`` (matches the
existing plane CRM stencil radius). All operators use the
last-two-axes convention ``(..., ny, nx)`` to match
``plane_operators.py``.

Equivalence: for any function ``f`` in :mod:`plane_operators` with
stencil radius 1, the relation

    f(arr) == _f_halo(jnp.pad(arr, ((0,0),)*(arr.ndim-2)+((1,1),(1,1)),
                              mode='wrap'))

holds bit-for-bit (no extra arithmetic, just slice indexing). The
multi-rank case differs only in that the padded halos come from
neighbor ranks rather than ``jnp.pad(mode='wrap')``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp


def grad_x_3d_halo(phi_pad: jax.Array, grid, halo: int = 1) -> jax.Array:
    """Halo-aware ``grad_x_3d``: (phi[j,i] - phi[j,i-1])/dx.

    ``phi_pad`` shape ``(..., ny_local + 2h, nx_local + 2h)``.
    Returns interior ``(..., ny_local, nx_local)``.
    """
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    return (interior - shifted_minus_x) / grid.dx


def grad_y_3d_halo(phi_pad: jax.Array, grid, halo: int = 1) -> jax.Array:
    """Halo-aware ``grad_y_3d``: (phi[j,i] - phi[j-1,i])/dy."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    return (interior - shifted_minus_y) / grid.dy


def divergence_3d_halo(
    u_pad: jax.Array, v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``divergence_3d``: (u[i+1]-u[i])/dx + (v[j+1]-v[j])/dy."""
    h = halo
    u_int = u_pad[..., h:-h, h:-h]
    u_shifted_plus_x = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    v_int = v_pad[..., h:-h, h:-h]
    v_shifted_plus_y = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    du_dx = (u_shifted_plus_x - u_int) / grid.dx
    dv_dy = (v_shifted_plus_y - v_int) / grid.dy
    return du_dx + dv_dy


def laplacian_3d_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 5-point Laplacian."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    plus_x = phi_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    plus_y = phi_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    lap_x = (plus_x - 2.0 * interior + minus_x) / (grid.dx ** 2)
    lap_y = (plus_y - 2.0 * interior + minus_y) / (grid.dy ** 2)
    return lap_x + lap_y


def interp_cell_to_xface_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (phi[i] + phi[i-1])`` at x-face."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_x = phi_pad[..., h:-h, h - 1 : -h - 1]
    return 0.5 * (interior + shifted_minus_x)


def interp_cell_to_yface_halo(
    phi_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (phi[j] + phi[j-1])`` at y-face."""
    h = halo
    interior = phi_pad[..., h:-h, h:-h]
    shifted_minus_y = phi_pad[..., h - 1 : -h - 1, h:-h]
    return 0.5 * (interior + shifted_minus_y)


def interp_xface_to_cell_halo(
    u_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (u[i] + u[i+1])`` at cell centre."""
    h = halo
    interior = u_pad[..., h:-h, h:-h]
    shifted_plus_x = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    return 0.5 * (interior + shifted_plus_x)


def interp_yface_to_cell_halo(
    v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware ``0.5 * (v[j] + v[j+1])`` at cell centre."""
    h = halo
    interior = v_pad[..., h:-h, h:-h]
    shifted_plus_y = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    return 0.5 * (interior + shifted_plus_y)


def interp_yface_to_xface_halo(
    v_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 4-point corner average: v at y-face → x-face."""
    h = halo
    v_int = v_pad[..., h:-h, h:-h]
    v_im1 = v_pad[..., h:-h, h - 1 : -h - 1]
    v_jp1 = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h:-h]
    v_im1_jp1 = v_pad[..., h + 1 : -h + 1 if h > 1 else None, h - 1 : -h - 1]
    return 0.25 * (v_int + v_im1 + v_jp1 + v_im1_jp1)


def interp_xface_to_yface_halo(
    u_pad: jax.Array, grid, halo: int = 1,
) -> jax.Array:
    """Halo-aware 4-point corner average: u at x-face → y-face."""
    h = halo
    u_int = u_pad[..., h:-h, h:-h]
    u_jm1 = u_pad[..., h - 1 : -h - 1, h:-h]
    u_ip1 = u_pad[..., h:-h, h + 1 : -h + 1 if h > 1 else None]
    u_jm1_ip1 = u_pad[..., h - 1 : -h - 1, h + 1 : -h + 1 if h > 1 else None]
    return 0.25 * (u_int + u_jm1 + u_ip1 + u_jm1_ip1)
