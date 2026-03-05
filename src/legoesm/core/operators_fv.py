"""FV3-style finite-volume transport operators on the cubed-sphere grid.

Implements PPM (Piecewise Parabolic Method) reconstruction with
Colella-Woodward monotonicity limiting and Lin-Rood directional
operator splitting for conservative 2D transport.

Key design:
- PPM reconstruction is purely 1D along grid lines
- At cube edges, pad_halo(halo=2) provides neighbor data
- The 1D stencil never encounters coordinate discontinuities
- Directional splitting processes x-sweeps and y-sweeps independently

Key functions
-------------
fv_flux_divergence : Conservative flux-form transport using PPM + Lin-Rood
fv_scalar_advection : Advective (non-conservative) transport using PPM

References
----------
- Colella & Woodward (1984): The Piecewise Parabolic Method (PPM)
- Lin & Rood (1996): Multidimensional Flux-Form Semi-Lagrangian Transport
- Lin (2004): A "Vertically Lagrangian" Finite-Volume Dynamical Core (FV3)
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.grids.cubed_sphere import CubedSphereGrid
from legoesm.grids.halo import pad_halo, pad_halo_vector


# ==============================================================================
# PPM edge reconstruction
# ==============================================================================

def _ppm_edge_values(q_1d):
    """4th-order edge values from cell averages along last-but-one axis.

    Parameters
    ----------
    q_1d : jax.Array, shape (..., M, K) where M = n+4
        Cell averages with halo=2, transverse halo stripped.

    Returns
    -------
    q_hat : jax.Array, shape (..., M-1, K)
        Edge values at interfaces between cells.
    """
    # 4th-order interior edges: (n+1 values from M=n+4 data)
    q_hat_inner = ((7.0 / 12.0) * (q_1d[..., 1:-2, :] + q_1d[..., 2:-1, :])
                   - (1.0 / 12.0) * (q_1d[..., :-3, :] + q_1d[..., 3:, :]))

    # 2nd-order boundary edges (outermost)
    q_hat_lo = 0.5 * (q_1d[..., 0:1, :] + q_1d[..., 1:2, :])
    q_hat_hi = 0.5 * (q_1d[..., -2:-1, :] + q_1d[..., -1:, :])

    # All M-1 = n+3 edge values
    q_hat = jnp.concatenate([q_hat_lo, q_hat_inner, q_hat_hi], axis=-2)

    # Monotonicity: clamp each edge between its two flanking cell values
    q_lo = jnp.minimum(q_1d[..., :-1, :], q_1d[..., 1:, :])
    q_hi = jnp.maximum(q_1d[..., :-1, :], q_1d[..., 1:, :])
    q_hat = jnp.clip(q_hat, q_lo, q_hi)

    return q_hat


def _ppm_limit(q_bar, q_L, q_R):
    """Colella-Woodward monotonicity limiter for PPM.

    Limits left/right parabola edge values to prevent new extrema.

    Parameters
    ----------
    q_bar : jax.Array, shape (...)
        Cell averages.
    q_L, q_R : jax.Array, shape (...)
        Left and right edge values per cell.

    Returns
    -------
    q_L_lim, q_R_lim : jax.Array
        Limited edge values.
    """
    # Detect local extrema: parabola should be flat
    is_extremum = (q_R - q_bar) * (q_bar - q_L) <= 0

    dm = q_R - q_L
    d6 = 6.0 * (q_bar - 0.5 * (q_L + q_R))

    # Overshoot on the left: parabola peak/trough outside [q_L, q_R]
    over_L = dm * d6 > dm ** 2
    q_L_lim = jnp.where(over_L, 3.0 * q_bar - 2.0 * q_R, q_L)

    # Overshoot on the right
    over_R = dm * d6 < -(dm ** 2)
    q_R_lim = jnp.where(over_R, 3.0 * q_bar - 2.0 * q_L, q_R)

    # At local extrema, flatten to cell average
    q_L_lim = jnp.where(is_extremum, q_bar, q_L_lim)
    q_R_lim = jnp.where(is_extremum, q_bar, q_R_lim)

    return q_L_lim, q_R_lim


def _ppm_reconstruct_x(q_pad_h2, limiter=True):
    """PPM reconstruction in x-direction.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (6, n+4, n+4)
        Scalar padded with halo=2.
    limiter : bool
        Apply Colella-Woodward limiter.

    Returns
    -------
    q_left, q_right : each shape (6, n+1, n)
        Left and right states at x-direction interfaces.
        q_left[i] = right-edge of cell i (cell to left of interface i)
        q_right[i] = left-edge of cell i+1 (cell to right of interface i)
    """
    q = q_pad_h2[:, :, 2:-2]  # (6, n+4, n) — strip transverse halo

    # Edge values: (6, n+3, n) at all M-1 interfaces
    q_hat = _ppm_edge_values(q)

    # Parabola for cells 1..n+2 (interior cells in padded array)
    a_L = q_hat[..., :-1, :]   # left edge of each cell, (6, n+2, n)
    a_R = q_hat[..., 1:, :]    # right edge
    q_c = q[..., 1:-1, :]       # cell centers, (6, n+2, n)

    if limiter:
        a_L, a_R = _ppm_limit(q_c, a_L, a_R)

    # Extract n+1 interior interface states
    q_left = a_R[..., :-1, :]   # right-edge of cell to left of interface
    q_right = a_L[..., 1:, :]   # left-edge of cell to right of interface

    return q_left, q_right


def _ppm_reconstruct_y(q_pad_h2, limiter=True):
    """PPM reconstruction in y-direction.

    Parameters
    ----------
    q_pad_h2 : jax.Array, shape (6, n+4, n+4)

    Returns
    -------
    q_left, q_right : each shape (6, n, n+1)
    """
    q = q_pad_h2[:, 2:-2, :]  # (6, n, n+4) — strip transverse halo
    # Transpose to reuse x-direction logic
    q_t = jnp.swapaxes(q, -2, -1)  # (6, n+4, n)
    q_hat = _ppm_edge_values(q_t)

    a_L = q_hat[..., :-1, :]
    a_R = q_hat[..., 1:, :]
    q_c = q_t[..., 1:-1, :]

    if limiter:
        a_L, a_R = _ppm_limit(q_c, a_L, a_R)

    q_left_t = a_R[..., :-1, :]
    q_right_t = a_L[..., 1:, :]

    return jnp.swapaxes(q_left_t, -2, -1), jnp.swapaxes(q_right_t, -2, -1)


# ==============================================================================
# Single-direction flux computation
# ==============================================================================

def _fv_flux_x(q, u, v, grid, dt, limiter=True):
    """Compute x-direction flux divergence using PPM.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field (e.g. fluid depth, density).
    u, v : jax.Array, shape (6, n, n)
        Cell-center velocities.
    grid : CubedSphereGrid
    dt : float
        Time step (for CFL-based limiting, currently unused).
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Flux divergence contribution from x-direction.
    """
    # Pad scalar with halo=2 for PPM reconstruction
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)

    # Pad velocity with halo=2 for interface velocity
    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    # PPM reconstruction: q_L (right-edge of left cell), q_R (left-edge of right cell)
    q_L, q_R = _ppm_reconstruct_x(q_pad, limiter)  # each (6, n+1, n)

    # Interface velocity: average of adjacent cells in the padded array
    # u_pad[:, :, 2:-2] strips y-halo → (6, n+4, n)
    u_strip = u_pad[:, :, 2:-2]  # (6, n+4, n)
    # Interior interface velocity: average at n+1 interfaces (padded indices 1..n+1)
    u_iface = 0.5 * (u_strip[:, 1:-2, :] + u_strip[:, 2:-1, :])  # (6, n+1, n)

    # Upwind selection
    q_face = jnp.where(u_iface > 0, q_L, q_R)

    # Edge metric: dy at interface
    hy = grid.hy_ext_h2[:, :, 2:-2]  # (6, n+4, n) - strip y halo
    hy_iface = 0.5 * (hy[:, 1:-2, :] + hy[:, 2:-1, :])  # (6, n+1, n)

    # Flux
    F = u_iface * hy_iface * q_face  # (6, n+1, n)

    # Divergence: -(F[i+1/2] - F[i-1/2]) / area
    return -(F[:, 1:, :] - F[:, :-1, :]) / grid.area


def _fv_flux_y(q, u, v, grid, dt, limiter=True):
    """Compute y-direction flux divergence using PPM.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field.
    u, v : jax.Array, shape (6, n, n)
        Cell-center velocities.
    grid : CubedSphereGrid
    dt : float
    limiter : bool

    Returns
    -------
    jax.Array, shape (6, n, n)
        Flux divergence contribution from y-direction.
    """
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)

    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    q_L, q_R = _ppm_reconstruct_y(q_pad, limiter)  # each (6, n, n+1)

    # Interface velocity in y
    v_strip = v_pad[:, 2:-2, :]  # (6, n, n+4)
    v_iface = 0.5 * (v_strip[:, :, 1:-2] + v_strip[:, :, 2:-1])  # (6, n, n+1)

    q_face = jnp.where(v_iface > 0, q_L, q_R)

    # Edge metric: dx at interface
    hx = grid.hx_ext_h2[:, 2:-2, :]  # (6, n, n+4) - strip x halo
    hx_iface = 0.5 * (hx[:, :, 1:-2] + hx[:, :, 2:-1])  # (6, n, n+1)

    G = v_iface * hx_iface * q_face  # (6, n, n+1)

    return -(G[:, :, 1:] - G[:, :, :-1]) / grid.area


# ==============================================================================
# Advective (non-conservative) single-direction sweeps
# ==============================================================================

def _fv_advect_x(q, u, v, grid, dt, limiter=True):
    """Compute x-direction advective tendency -u * dq/dx using PPM.

    Uses PPM reconstruction to compute upwind interface values, then
    computes the advective flux as u * q_face (without multiplying by q
    in the flux — this gives -v·∇q rather than -∇·(qv)).

    Parameters
    ----------
    q, u, v : jax.Array, shape (6, n, n)
    grid : CubedSphereGrid
    dt : float
    limiter : bool

    Returns
    -------
    jax.Array, shape (6, n, n)
    """
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)

    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    q_L, q_R = _ppm_reconstruct_x(q_pad, limiter)  # each (6, n+1, n)

    u_strip = u_pad[:, :, 2:-2]
    u_iface = 0.5 * (u_strip[:, 1:-2, :] + u_strip[:, 2:-1, :])  # (6, n+1, n)

    q_face = jnp.where(u_iface > 0, q_L, q_R)

    # Advective flux: just u * q_face (no density)
    hy = grid.hy_ext_h2[:, :, 2:-2]
    hy_iface = 0.5 * (hy[:, 1:-2, :] + hy[:, 2:-1, :])

    F = u_iface * hy_iface * q_face

    return -(F[:, 1:, :] - F[:, :-1, :]) / grid.area


def _fv_advect_y(q, u, v, grid, dt, limiter=True):
    """Compute y-direction advective tendency -v * dq/dy using PPM."""
    q_pad = pad_halo(q, halo=2, interp_offsets=grid.halo_interp_offsets_h2)

    u_pad, v_pad = pad_halo_vector(
        u, v,
        grid.cos_angle, grid.sin_angle,
        grid.cos_angle_padded_h2, grid.sin_angle_padded_h2,
        interp_offsets=grid.halo_interp_offsets_h2, halo=2,
    )

    q_L, q_R = _ppm_reconstruct_y(q_pad, limiter)

    v_strip = v_pad[:, 2:-2, :]
    v_iface = 0.5 * (v_strip[:, :, 1:-2] + v_strip[:, :, 2:-1])

    q_face = jnp.where(v_iface > 0, q_L, q_R)

    hx = grid.hx_ext_h2[:, 2:-2, :]
    hx_iface = 0.5 * (hx[:, :, 1:-2] + hx[:, :, 2:-1])

    G = v_iface * hx_iface * q_face

    return -(G[:, :, 1:] - G[:, :, :-1]) / grid.area


# ==============================================================================
# Lin-Rood 2D operator-split transport
# ==============================================================================

def fv_flux_divergence(q, u, v, grid, dt, limiter=True, x_first=True):
    """Conservative flux-form 2D transport using PPM + directional splitting.

    Implements the Lin-Rood (1996) algorithm: split into 1D x-sweep and
    y-sweep using PPM reconstruction. Each directional flux is a telescoping
    sum (F[i+1/2] - F[i-1/2]), so the global sum is zero by construction.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field to transport (e.g. fluid depth h, density rho).
    u, v : jax.Array, shape (6, n, n)
        Velocity components (grid-aligned).
    grid : CubedSphereGrid
    dt : float
        Time step [s].
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.
    x_first : bool
        If True, x-sweep first then y-sweep; if False, reversed.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Flux divergence tendency: dq/dt.
    """
    if x_first:
        dq_1 = _fv_flux_x(q, u, v, grid, dt, limiter)
        q_star = q + dt * dq_1
        dq_2 = _fv_flux_y(q_star, u, v, grid, dt, limiter)
    else:
        dq_1 = _fv_flux_y(q, u, v, grid, dt, limiter)
        q_star = q + dt * dq_1
        dq_2 = _fv_flux_x(q_star, u, v, grid, dt, limiter)

    return dq_1 + dq_2


def fv_scalar_advection(q, u, v, grid, dt, limiter=True, x_first=True):
    """PPM advection of scalar q by (u,v) — advective (non-conservative) form.

    For tracers/temperature where we want -v·grad(q), not -div(q*v).
    Uses the same PPM reconstruction and directional splitting as the
    flux-form operator, but without flux form.

    Parameters
    ----------
    q : jax.Array, shape (6, n, n)
        Scalar field to advect.
    u, v : jax.Array, shape (6, n, n)
        Velocity components.
    grid : CubedSphereGrid
    dt : float
        Time step [s].
    limiter : bool
        Apply Colella-Woodward monotonicity limiter.
    x_first : bool
        Sweep direction ordering.

    Returns
    -------
    jax.Array, shape (6, n, n)
        Advective tendency: -v·grad(q).
    """
    if x_first:
        dq_1 = _fv_advect_x(q, u, v, grid, dt, limiter)
        q_star = q + dt * dq_1
        dq_2 = _fv_advect_y(q_star, u, v, grid, dt, limiter)
    else:
        dq_1 = _fv_advect_y(q, u, v, grid, dt, limiter)
        q_star = q + dt * dq_1
        dq_2 = _fv_advect_x(q_star, u, v, grid, dt, limiter)

    return dq_1 + dq_2
